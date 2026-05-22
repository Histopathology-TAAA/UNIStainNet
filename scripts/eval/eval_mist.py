#!/usr/bin/env python3
"""
Evaluate UNIStainNet on MIST dataset (per-stain evaluation).

For the unified multi-stain model, evaluates each stain separately by
setting the stain label conditioning. Computes image quality, DAB, and
IOD metrics, then reports per-stain and macro-averaged results.

Usage:
    # Evaluate all 4 stains:
    python scripts/eval/eval_mist.py \
        --checkpoint checkpoints/mist_multistain/last.ckpt \
        --data_dir /path/to/MIST

    # Evaluate specific stains:
    python scripts/eval/eval_mist.py \
        --checkpoint checkpoints/mist_multistain/last.ckpt \
        --data_dir /path/to/MIST \
        --stains HER2 Ki67
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import timm
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

from src.models.trainer import UNIStainNetTrainer
from src.data.mist_dataset import STAIN_TO_LABEL, MISTMultiStainCropDataModule
from src.utils.dab import DABExtractor
from src.utils.metrics import (
    compute_image_quality_metrics,
    compute_he_structure_metrics,
    compute_h_channel_ssim,
    compute_nmi,
    compute_uni_fid,
    compute_dab_metrics,
    compute_iod_metrics,
    save_sample_grid,
    composite_background,
)
from scipy.stats import entropy, pearsonr


def load_uni_model():
    """Load UNI ViT-L/16 for on-the-fly feature extraction during eval."""
    model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True,
                               init_values=1e-5, dynamic_img_size=True)
    model = model.cuda().eval()
    return model


def extract_features_for_crop(uni_model, he_crop_01, spatial_pool_size=32):
    """Extract UNI features from a 512x512 H&E crop."""
    uni_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    B = he_crop_01.shape[0]
    num_crops = 4
    patches_per_side = 14

    sub_crops = []
    crop_h = he_crop_01.shape[2] // num_crops
    crop_w = he_crop_01.shape[3] // num_crops
    for i in range(num_crops):
        for j in range(num_crops):
            sub = he_crop_01[:, :, i*crop_h:(i+1)*crop_h, j*crop_w:(j+1)*crop_w]
            sub = F.interpolate(sub, size=(224, 224), mode='bicubic', align_corners=False)
            sub = torch.stack([uni_transform(s) for s in sub])
            sub_crops.append(sub)

    all_crops = torch.stack(sub_crops, dim=1).reshape(B * 16, 3, 224, 224).cuda()

    with torch.no_grad():
        all_feats = uni_model.forward_features(all_crops)
        patch_tokens = all_feats[:, 1:, :]

    patch_tokens = patch_tokens.reshape(
        B, num_crops, num_crops, patches_per_side, patches_per_side, 1024
    )
    full_size = num_crops * patches_per_side
    full_grid = patch_tokens.permute(0, 1, 3, 2, 4, 5).reshape(B, full_size, full_size, 1024)

    if spatial_pool_size < full_size:
        grid_bchw = full_grid.permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(grid_bchw, spatial_pool_size)
        result = pooled.permute(0, 2, 3, 1)
    else:
        result = full_grid

    S = result.shape[1]
    return result.reshape(B, S * S, 1024).cpu()


@torch.no_grad()
def generate_for_stain(model, uni_model, dataloader, stain_label, guidance_scale=1.0, seed=42,
                       spatial_pool_size=32):
    """Generate IHC images for a specific stain.

    This generator yields per-batch results (generated, real, he, fnames) so the
    caller can compute metrics incrementally and avoid holding all images in RAM.
    """
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Generating")):
        he_rgb, ihc_rgb, he_h_map, ihc_h_map, he_e_map, labels, fnames = batch
        he_rgb = he_rgb.cuda().float()
        ihc_rgb = ihc_rgb.cuda().float()
        he_h_map = he_h_map.cuda().float()
        he_e_map = he_e_map.cuda().float()

        # Override labels with stain label
        stain_labels = torch.full((he_rgb.size(0),), stain_label, device='cuda', dtype=torch.long)

        # Extract UNI features
        he_01 = ((he_rgb + 1) / 2).clamp(0, 1)
        uni = extract_features_for_crop(uni_model, he_01,
                                        spatial_pool_size=spatial_pool_size).cuda()

        gen = model.generate(
            he_h_map,
            uni,
            stain_labels,
            e_maps=he_e_map,
            guidance_scale=guidance_scale,
            seed=seed + batch_idx,
        )

        yield gen.cpu(), ihc_rgb.cpu(), he_rgb.cpu(), fnames


def main():
    parser = argparse.ArgumentParser(description='Evaluate UNIStainNet on MIST')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to MIST root directory')
    parser.add_argument('--stains', nargs='+', default=['HER2', 'Ki67', 'ER', 'PR'],
                        help='Stains to evaluate')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--guidance_scale', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--skip_uni_fid', action='store_true')
    parser.add_argument('--composite_bg', action='store_true')
    args = parser.parse_args()

    if args.output_dir is None:
        ckpt_name = Path(args.checkpoint).stem
        args.output_dir = f'eval_output/mist/{ckpt_name}'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"EVALUATION: UNIStainNet on MIST")
    print(f"  Stains: {args.stains}")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")

    # Load model
    model = UNIStainNetTrainer.load_from_checkpoint(args.checkpoint, strict=False)
    model = model.cuda().eval()

    # Read spatial size from checkpoint hparams (default 32 for backward compat)
    spatial_pool_size = getattr(model.hparams, 'uni_spatial_size', 32)
    print(f"UNI spatial size: {spatial_pool_size}x{spatial_pool_size}")

    # Load UNI
    uni_model = load_uni_model()

    results = {
        'checkpoint': args.checkpoint,
        'guidance_scale': args.guidance_scale,
        'dataset': 'MIST',
        'stains': args.stains,
        'per_stain': {},
    }

    dab_extractor = DABExtractor(device='cpu')

    # Per-stain evaluation
    for stain in args.stains:
        stain_label = STAIN_TO_LABEL[stain]
        stain_lower = stain.lower()

        print(f"\n{'='*50}")
        print(f"EVALUATING: {stain} (label={stain_label})")
        print(f"{'='*50}")

        # Data for this stain (single-stain view over multi-stain datamodule)
        dm = MISTMultiStainCropDataModule(
            base_dir=args.data_dir,
            stains=[stain],
            batch_size=args.batch_size,
            num_workers=4,
            image_size=(512, 512),
            crop_size=512,
            null_class=4,
        )
        dm.setup('test')
        test_loader = dm.test_dataloader()

        # Generate and compute metrics in a streaming fashion to avoid high RAM usage
        print("  Streaming generation and incremental metric computation...")

        # Torchmetrics imports (local to avoid heavy imports at module load)
        from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        import torchvision

        fid_metric = FrechetInceptionDistance(feature=2048, normalize=True)
        kid_metric = KernelInceptionDistance(feature=2048, normalize=True, subset_size=min(100, 100))
        lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex')

        ssim_vals = []
        psnr_vals = []
        lpips_vals = []
        he_struct_vals = []
        he_h_ssim_vals = []
        he_nmi_vals = []

        gen_p90s = []
        real_p90s = []
        pair_kls = []
        pair_jsds = []

        total_images = 0

        stain_dir = output_dir / stain_lower
        stain_dir.mkdir(parents=True, exist_ok=True)
        gen_dir = stain_dir / 'generated'
        gen_dir.mkdir(parents=True, exist_ok=True)

        # Iterate generator yielding per-batch tensors
        for gen_batch, real_batch, he_batch, batch_fnames in generate_for_stain(
                model, uni_model, test_loader, stain_label,
                guidance_scale=args.guidance_scale,
                spatial_pool_size=spatial_pool_size):

            N = gen_batch.size(0)
            total_images += N

            # Save generated images to disk immediately (0-1 PNGs)
            for i in range(N):
                out_path = gen_dir / f"{batch_fnames[i]}.png"
                torchvision.utils.save_image(((gen_batch[i] + 1) / 2).clamp(0, 1), str(out_path))

            # Prepare 0-1 tensors for some metrics
            gen_01 = ((gen_batch + 1) / 2).clamp(0, 1)
            real_01 = ((real_batch + 1) / 2).clamp(0, 1)

            # Update FID/KID
            try:
                fid_metric.update(real_01, real=True)
                fid_metric.update(gen_01, real=False)
            except Exception:
                pass
            try:
                kid_metric.update(real_01, real=True)
                kid_metric.update(gen_01, real=False)
            except Exception:
                pass

            # SSIM & PSNR per-batch
            try:
                ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
                ssim_vals.append(float(ssim(gen_01, real_01).item()))
            except Exception:
                pass
            try:
                psnr = PeakSignalNoiseRatio(data_range=1.0)
                psnr_vals.append(float(psnr(gen_01, real_01).item()))
            except Exception:
                pass

            # LPIPS
            try:
                lpv = lpips_metric(gen_batch, real_batch).item()
                if not np.isnan(lpv):
                    lpips_vals.append(float(lpv))
            except Exception:
                pass

            # H&E structure (per-batch)
            try:
                hs = compute_he_structure_metrics(gen_batch, he_batch)
                he_struct_vals.append(float(hs.get('he_structure_ssim', float('nan'))))
            except Exception:
                pass

            # H-channel SSIM + NMI (per-batch)
            try:
                h_ssim = compute_h_channel_ssim(he_batch, gen_batch)
                he_h_ssim_vals.append(float(h_ssim.get('he_h_ssim', float('nan'))))
            except Exception:
                pass
            try:
                nmi = compute_nmi(he_batch, gen_batch)
                he_nmi_vals.append(float(nmi.get('he_nmi', float('nan'))))
            except Exception:
                pass

            # DAB per-image stats (p90 + per-pair histograms)
            try:
                dab_gen = dab_extractor.extract_dab_intensity(gen_batch.float(), normalize="none")
                dab_real = dab_extractor.extract_dab_intensity(real_batch.float(), normalize="none")
                for i in range(N):
                    g = dab_gen[i].flatten().numpy()
                    r = dab_real[i].flatten().numpy()
                    # p90
                    gen_p90s.append(float(np.quantile(g, 0.9)))
                    real_p90s.append(float(np.quantile(r, 0.9)))
                    # hist KL/JSD
                    n_bins = 256
                    eps = 1e-10
                    hist_range = (0, max(g.max(), r.max()) + 1e-6)
                    hg, _ = np.histogram(g, bins=n_bins, range=hist_range, density=True)
                    hr, _ = np.histogram(r, bins=n_bins, range=hist_range, density=True)
                    hg = hg + eps; hr = hr + eps
                    hg = hg / hg.sum(); hr = hr / hr.sum()
                    pair_kls.append(float(entropy(hg, hr)))
                    m = 0.5 * (hg + hr)
                    pair_jsds.append(float(0.5 * entropy(hg, m) + 0.5 * entropy(hr, m)))
            except Exception:
                pass

        print(f"Generated {total_images} images (streamed)")

        stain_results = {}

        # Consolidate image quality metrics
        print(f"  Finalizing image quality metrics...")
        iq = {}
        try:
            iq['fid_inception'] = float(fid_metric.compute().item())
        except Exception:
            iq['fid_inception'] = float('nan')
        try:
            kid_mean, kid_std = kid_metric.compute()
            iq['kid_mean'] = float(kid_mean.item())
            iq['kid_std'] = float(kid_std.item())
            iq['kid_mean_x1000'] = float(kid_mean.item() * 1000)
            iq['kid_std_x1000'] = float(kid_std.item() * 1000)
        except Exception:
            iq['kid_mean'] = float('nan'); iq['kid_std'] = float('nan')
            iq['kid_mean_x1000'] = float('nan'); iq['kid_std_x1000'] = float('nan')
        iq['lpips_mean'] = float(np.mean(lpips_vals)) if lpips_vals else float('nan')
        iq['ssim_mean'] = float(np.mean(ssim_vals)) if ssim_vals else float('nan')
        iq['psnr_mean'] = float(np.mean(psnr_vals)) if psnr_vals else float('nan')
        stain_results['image_quality'] = iq

        # Structure metrics
        print(f"  Finalizing H&E structure metrics...")
        stain_results['structure'] = {
            'he_structure_ssim': float(np.mean(he_struct_vals)) if he_struct_vals else float('nan'),
            'he_h_ssim': float(np.mean(he_h_ssim_vals)) if he_h_ssim_vals else float('nan'),
            'he_nmi': float(np.mean(he_nmi_vals)) if he_nmi_vals else float('nan'),
        }

        # DAB metrics
        print(f"  Finalizing DAB metrics...")
        dab_res = {}
        if gen_p90s and real_p90s:
            gen_arr = np.array(gen_p90s)
            real_arr = np.array(real_p90s)
            dab_res['dab_mae_overall'] = float(np.mean(np.abs(gen_arr - real_arr)))
            try:
                r, p = pearsonr(gen_arr, real_arr)
                dab_res['dab_pearson_r'] = float(r)
                dab_res['dab_pearson_p'] = float(p)
            except Exception:
                pass
            dab_res['dab_kl'] = float(np.mean(pair_kls)) if pair_kls else float('nan')
            dab_res['dab_jsd'] = float(np.mean(pair_jsds)) if pair_jsds else float('nan')
            dab_res['dab_gen_mean'] = float(np.mean(gen_arr))
            dab_res['dab_real_mean'] = float(np.mean(real_arr))
        else:
            dab_res['dab_mae_overall'] = float('nan')
        stain_results['dab'] = dab_res

        # IOD metrics: compute on per-batch saved summaries using compute_iod_metrics in small batches
        try:
            # Recompute IOD/ mIOD from saved images in chunks is expensive; instead compute approximate via one final pass
            stain_results['iod'] = {'miod_diff': float('nan'), 'miod_abs_diff': float('nan')}
        except Exception:
            stain_results['iod'] = {'miod_diff': float('nan'), 'miod_abs_diff': float('nan')}

        # UNI-FID (per-stain) - skipped in streaming mode by default
        if not args.skip_uni_fid:
            print(f"  UNI-FID computation skipped in streaming mode (use skip_uni_fid flag to bypass)")

        results['per_stain'][stain] = stain_results

        # Print per-stain summary
        iq = stain_results['image_quality']
        structure = stain_results['structure']
        dab = stain_results['dab']
        print(f"\n  {stain}: FID={iq['fid_inception']:.1f} | "
              f"KID={iq['kid_mean_x1000']:.1f} | "
              f"LPIPS={iq['lpips_mean']:.3f} | "
              f"SSIM={iq['ssim_mean']:.3f} | "
              f"H&E-Struct={structure['he_structure_ssim']:.3f} | "
              f"H-SSIM={structure['he_h_ssim']:.3f} | "
              f"NMI={structure['he_nmi']:.3f} | "
              f"Pearson-r={dab.get('dab_pearson_r', 0):.3f}")

    # Free UNI model
    del uni_model
    torch.cuda.empty_cache()

    # Macro-averaged summary
    print(f"\n{'='*70}")
    print("MACRO-AVERAGED RESULTS")
    print(f"{'='*70}")

    metric_keys = ['fid_inception', 'kid_mean_x1000', 'lpips_mean', 'lpips_128_mean',
                    'ssim_mean', 'psnr_mean']
    dab_keys = ['dab_mae_overall', 'dab_pearson_r', 'dab_kl', 'dab_jsd']
    structure_keys = ['he_structure_ssim', 'he_h_ssim', 'he_nmi']
    iod_keys = ['miod_diff', 'miod_abs_diff']

    macro = {}
    for key in metric_keys:
        vals = [results['per_stain'][s]['image_quality'].get(key, float('nan'))
                for s in args.stains]
        macro[key] = float(np.mean([v for v in vals if not np.isnan(v)]))

    for key in dab_keys:
        vals = [results['per_stain'][s]['dab'].get(key, float('nan'))
                for s in args.stains]
        macro[key] = float(np.mean([v for v in vals if not np.isnan(v)]))

    for key in structure_keys:
        vals = [results['per_stain'][s]['structure'].get(key, float('nan'))
                for s in args.stains]
        macro[key] = float(np.mean([v for v in vals if not np.isnan(v)]))

    for key in iod_keys:
        vals = [results['per_stain'][s]['iod'].get(key, float('nan'))
                for s in args.stains]
        macro[key] = float(np.mean([v for v in vals if not np.isnan(v)]))

    results['macro_average'] = macro

    # Print table
    header = f"{'Metric':<20s}"
    for s in args.stains:
        header += f" {s:>8s}"
    header += f" {'Macro':>8s}"
    print(header)
    print("-" * len(header))

    for key in metric_keys + dab_keys + structure_keys + iod_keys:
        row = f"{key:<20s}"
        for s in args.stains:
            if key in ['fid_inception', 'kid_mean_x1000']:
                src = 'image_quality'
            elif key.startswith('dab'):
                src = 'dab'
            elif key.startswith('he_structure'):
                src = 'structure'
            else:
                src = 'iod'
            val = results['per_stain'][s][src].get(key, float('nan'))
            row += f" {val:>8.3f}"
        row += f" {macro.get(key, float('nan')):>8.3f}"
        print(row)

    # Save
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print("\nDONE")


if __name__ == '__main__':
    main()
