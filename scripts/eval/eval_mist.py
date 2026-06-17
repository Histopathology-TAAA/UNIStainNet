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

# pyrefly: ignore [missing-import]
import torch
import torch.nn.functional as F
import timm
import torchvision.transforms as transforms
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm

from src.models.trainer import UNIStainNetTrainer
from src.data.mist_dataset import MISTMultiStainCropDataModule, STAIN_TO_LABEL
from src.models.deepliif_stainer import DeepLIIFStainer
from src.utils.dab import DABExtractor
from src.utils.metrics import (
    compute_image_quality_metrics,
    compute_uni_fid,
    compute_dab_metrics,
    compute_iod_metrics,
    compute_he_h_ssim,
    compute_he_nmi,
    save_sample_grid,
    composite_background,
)
from src.utils.ki67_evaluator import (
    Ki67ClinicalEvaluator,
    compute_ki67_summary,
    print_ki67_summary,
)


def load_uni_model():
    """Load UNI ViT-L/16 for on-the-fly feature extraction during eval."""
    model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True,
                               init_values=1e-5, dynamic_img_size=True)
    model = model.cuda().eval()
    return model


def extract_features_from_sub_crops(uni_model, uni_sub_crops, spatial_pool_size=32):
    """Extract UNI features directly from dataloader sub-crops."""
    B = uni_sub_crops.shape[0]
    num_crops = 4
    patches_per_side = 14

    # uni_sub_crops is [B, 16, 3, 224, 224]
    all_crops = uni_sub_crops.reshape(B * 16, 3, 224, 224).cuda()

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
                       spatial_pool_size=32, no_downcasting=False, aligned=False,
                       deepliif_stainer=None, hema_channels=1):
    """Generate IHC images for a specific stain."""
    all_gen, all_real, all_he, all_fnames = [], [], [], []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Generating")):
        he, her2, he_h, ihc_h, uni_sub_crops, labels, fnames = batch
        he, her2 = he.cuda().float(), her2.cuda().float()
        he_h = he_h.cuda().float()
        ihc_h = ihc_h.cuda().float()

        # Override labels with stain label
        stain_labels = torch.full((he.size(0),), stain_label, device='cuda', dtype=torch.long)
        
        edge_input = ihc_h if aligned else he_h
        
        # h_channel: determine based on Hybrid DeepLIIF mode or legacy mode
        if deepliif_stainer is not None:
            if not aligned:
                # Case B: Standard inference uses Analytical H&E
                edge_input = he_h
                h_channel = he_h
                align_source = he_h
            else:
                # Case A: Uses DeepLIIF on IHC + Normalization
                raw_ihc_hema = deepliif_stainer.extract_hematoxylin(her2)
                
                # Statistical Normalization
                mean_he = he_h.mean(dim=[2, 3], keepdim=True)
                std_he = he_h.std(dim=[2, 3], keepdim=True) + 1e-8
                mean_ihc = raw_ihc_hema.mean(dim=[2, 3], keepdim=True)
                std_ihc = raw_ihc_hema.std(dim=[2, 3], keepdim=True) + 1e-8
                
                ihc_h_norm = (raw_ihc_hema - mean_ihc) / std_ihc * std_he + mean_he
                ihc_h_norm = ihc_h_norm.clamp(-1, 1)
                
                edge_input = ihc_h_norm
                h_channel = ihc_h_norm
                align_source = he_h
        else:
            h_channel = edge_input
            align_source = he_h

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # Extract UNI features
            uni = extract_features_from_sub_crops(uni_model, uni_sub_crops,
                                                  spatial_pool_size=spatial_pool_size).cuda()

            gen_out = model.generate(he, uni, stain_labels,
                                 guidance_scale=guidance_scale,
                                 seed=seed + batch_idx,
                                 edge_input=edge_input,
                                 he_h=align_source,
                                 h_channel=h_channel)
            
            # Backward compatibility: older models return tensor, newer alignment models return tuple
            if isinstance(gen_out, tuple):
                gen = gen_out[0]
            else:
                gen = gen_out

            gen = gen.float()

        if no_downcasting:
            all_gen.append(gen.cpu())
            all_real.append(her2.cpu())
            all_he.append(he.cpu())
        else:
            all_gen.append(gen.cpu().to(torch.float16))
            all_real.append(her2.cpu().to(torch.float16))
            all_he.append(he.cpu().to(torch.float16))
        all_fnames.extend(fnames)

    return torch.cat(all_gen), torch.cat(all_real), torch.cat(all_he), all_fnames


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
    parser.add_argument('--no_downcasting', action='store_true', help='Disable float16 downcasting for metric accuracy')
    parser.add_argument('--enable_stn_alignment', action='store_true', help='Enable STN alignment during evaluation (disabled by default)')
    parser.add_argument('--aligned', action='store_true', help='Evaluate on the aligned case using IHC ground-truth structure.')
    parser.add_argument('--random_seed', action='store_true', help='Use a random seed instead of fixed seed 42')
    parser.add_argument('--dab_threshold', type=float, default=0.15,
                        help='DAB optical-density threshold for Ki67 positive cell calling. '
                             'Lower = more sensitive.  Adjust per-dataset for white-balance '
                             'and contrast differences.  Default: 0.15')
    parser.add_argument('--ki67_eval_method', type=str, choices=['deepliif', 'stardist'], default='deepliif',
                        help='Method to evaluate Ki67 Labeling Index.')
    parser.add_argument('--seg_thresh', type=int, default=130,
                        help='DeepLIIF segmentation intensity threshold.')
    parser.add_argument('--marker_thresh', type=str, default='default',
                        help="DeepLIIF marker intensity threshold (int or 'default').")
    args = parser.parse_args()

    if not args.random_seed:
        pl.seed_everything(42, workers=True)

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

    # Control the Alignment Network (STN) during evaluation.
    # By default, bypassing the STN entirely guarantees 100% structural fidelity 
    # to the input H&E and prevents interpolation blur from degrading the SSIM/NMI metrics.
    # However, it can be enabled via the --enable_stn_alignment flag.
    if args.enable_stn_alignment:
        if not hasattr(model.generator, 'alignment_net'):
            print("WARNING: --enable_stn_alignment was passed, but this checkpoint was NOT trained with STN alignment.")
            print("STN evaluation will be skipped for this run.")
        else:
            if hasattr(model.generator, 'use_alignment'):
                model.generator.use_alignment = True
            if hasattr(model, 'generator_ema') and hasattr(model.generator_ema, 'use_alignment'):
                model.generator_ema.use_alignment = True

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

    # Initialize DeepLIIFStainer early so it can be passed to the evaluator
    deepliif_stainer = None
    if getattr(model.hparams, 'deepliif_weights_path', None) or args.aligned:
        deepliif_weights_path = getattr(model.hparams, 'deepliif_weights_path', 'deepliif-weights/DeepLIIF_Latest_Model')
        deepliif_stainer = DeepLIIFStainer(weights_path=deepliif_weights_path)
        
    # Ki67 Clinical Evaluator (lazy — StarDist loads on first call)
    ki67_evaluator = None
    if 'Ki67' in args.stains:
        ki67_evaluator = Ki67ClinicalEvaluator(
            dab_threshold=args.dab_threshold, 
            deepliif_stainer=deepliif_stainer,
            eval_method=args.ki67_eval_method,
            seg_thresh=args.seg_thresh,
            marker_thresh=args.marker_thresh
        )
        print(f"[INFO] Ki67 clinical evaluator enabled (Method={args.ki67_eval_method}, DAB thresh={args.dab_threshold})")

    # Per-stain evaluation
    for stain in args.stains:
        stain_label = STAIN_TO_LABEL[stain]
        stain_lower = stain.lower()

        print(f"\n{'='*50}")
        print(f"EVALUATING: {stain} (label={stain_label})")
        print(f"{'='*50}")

        # Data for this stain
        dm = MISTMultiStainCropDataModule(
            base_dir=args.data_dir,
            stains=[stain],
            batch_size=args.batch_size,
            num_workers=4,
            image_size=(512, 512),
            crop_size=512,
            null_class=stain_label,
        )
        dm.setup('test')
        test_loader = dm.val_dataloader()

        # Initialize DeepLIIF if checkpoint used it
        hema_channels = getattr(model.hparams, 'hema_channels', 1)
        # DeepLIIF logic is now handled above globally for this script
        if args.aligned and hasattr(model.generator, 'use_alignment') and model.generator.use_alignment:
            pass # deepliif_stainer is already initialized above if needed
            print(f"[INFO] DeepLIIF is enabled. Using {hema_channels}-channel setup.")

        # Generate
        gen, real, he, fnames = generate_for_stain(
            model, uni_model, test_loader, stain_label,
            guidance_scale=args.guidance_scale,
            seed=None if args.random_seed else 42,
            spatial_pool_size=spatial_pool_size,
            no_downcasting=args.no_downcasting,
            aligned=args.aligned,
            deepliif_stainer=deepliif_stainer,
            hema_channels=hema_channels,
        )
        print(f"Generated {len(gen)} images")

        if args.composite_bg:
            gen = composite_background(gen, he)

        # Save sample grid
        stain_dir = output_dir / stain_lower
        stain_dir.mkdir(parents=True, exist_ok=True)
        save_sample_grid(he, real, gen, stain_dir / 'sample_grid.png', n=16)

        stain_results = {}

        # Image quality
        print(f"  Computing image quality metrics...")
        stain_results['image_quality'] = compute_image_quality_metrics(gen, real)

        # DAB metrics (no class labels for MIST)
        print(f"  Computing DAB metrics...")
        stain_results['dab'] = compute_dab_metrics(gen, real, labels=None, dab_extractor=dab_extractor)

        # IOD metrics
        print(f"  Computing IOD metrics...")
        stain_results['iod'] = compute_iod_metrics(gen, real, labels=None)

        # HE-H SSIM and HE-NMI metrics
        print(f"  Computing structure metrics...")
        struct_target = real if args.aligned else he
        is_target_he = not args.aligned
        
        h_ssim = compute_he_h_ssim(gen, struct_target, is_target_he=is_target_he, dab_extractor=dab_extractor)
        nmi = compute_he_nmi(gen, struct_target, is_target_he=is_target_he, dab_extractor=dab_extractor)
        
        prefix = 'ihc' if args.aligned else 'he'
        stain_results['he_structure'] = {
            f'{prefix}_h_ssim_mean': h_ssim['ssim_mean'],
            f'{prefix}_h_ssim_std': h_ssim['ssim_std'],
            f'{prefix}_nmi_mean': nmi['nmi_mean'],
            f'{prefix}_nmi_std': nmi['nmi_std'],
        }

        # UNI-FID (per-stain)
        if not args.skip_uni_fid:
            print(f"  Computing UNI-FID...")
            try:
                stain_results['image_quality']['fid_uni'] = compute_uni_fid(gen, real)
            except Exception as e:
                print(f"    UNI-FID skipped: {e}")

        # Ki67 Clinical Evaluation (cell-level metrics)
        if stain == 'Ki67' and ki67_evaluator is not None:
            print(f"  Computing Ki67 clinical metrics (StarDist)...")
            real_li_scores = []
            fake_li_scores = []

            N = gen.shape[0]
            for i in tqdm(range(N), desc="  Ki67 scoring"):
                real_cells, real_pos, real_li = ki67_evaluator.compute_labeling_index(real[i])
                fake_cells, fake_pos, fake_li = ki67_evaluator.compute_labeling_index(gen[i])

                real_li_scores.append(real_li)
                fake_li_scores.append(fake_li)

            # Compute global summary
            ki67_summary = compute_ki67_summary(real_li_scores, fake_li_scores)
            print_ki67_summary(ki67_summary)
            stain_results['ki67_clinical'] = ki67_summary

        results['per_stain'][stain] = stain_results

        # Print per-stain summary
        iq = stain_results['image_quality']
        dab = stain_results['dab']
        he_struct = stain_results['he_structure']
        print(f"\n  {stain}: FID={iq['fid_inception']:.1f} | "
              f"KID={iq['kid_mean_x1000']:.1f} | "
              f"LPIPS={iq['lpips_mean']:.3f} | "
              f"SSIM={iq['ssim_mean']:.3f} | "
              f"Pearson-r={dab.get('dab_pearson_r', 0):.3f} | "
              f"{prefix.upper()}-H-SSIM={he_struct[f'{prefix}_h_ssim_mean']:.3f} | "
              f"{prefix.upper()}-NMI={he_struct[f'{prefix}_nmi_mean']:.3f}")
        if 'ki67_clinical' in stain_results:
            ki = stain_results['ki67_clinical']
            print(f"         Ki67: MAE={ki['ki67_li_mae']:.2f}% | "
                  f"r={ki['ki67_li_pearson_r']:.3f} | "
                  f"Concordance={ki['ki67_tier_concordance']*100:.1f}% | "
                  f"Kappa={ki['ki67_tier_kappa']:.3f}")

        # Explicitly free memory before next stain
        del gen, real, he, fnames
        del dm, test_loader
        import gc
        gc.collect()

    # Free UNI model
    del uni_model
    torch.cuda.empty_cache()

    # Macro-averaged summary
    print(f"\n{'='*70}")
    print("MACRO-AVERAGED RESULTS")
    print(f"{'='*70}")

    metric_keys = ['fid_inception', 'fid_uni', 'kid_mean_x1000', 'lpips_mean', 'lpips_128_mean',
                    'ssim_mean', 'psnr_mean']
    dab_keys = ['dab_mae_overall', 'dab_pearson_r', 'dab_kl', 'dab_jsd']
    iod_keys = ['miod_diff', 'miod_abs_diff']
    prefix = 'ihc' if args.aligned else 'he'
    he_struct_keys = [f'{prefix}_h_ssim_mean', f'{prefix}_nmi_mean']
    ki67_keys = ['ki67_li_mae', 'ki67_li_pearson_r', 'ki67_tier_concordance', 'ki67_tier_kappa']

    macro = {}
    for key in metric_keys:
        vals = [results['per_stain'][s]['image_quality'].get(key, float('nan'))
                for s in args.stains]
        macro[key] = float(np.mean([v for v in vals if not np.isnan(v)]))

    for key in dab_keys:
        vals = [results['per_stain'][s]['dab'].get(key, float('nan'))
                for s in args.stains]
        macro[key] = float(np.mean([v for v in vals if not np.isnan(v)]))

    for key in iod_keys:
        vals = [results['per_stain'][s]['iod'].get(key, float('nan'))
                for s in args.stains]
        macro[key] = float(np.mean([v for v in vals if not np.isnan(v)]))

    for key in he_struct_keys:
        vals = [results['per_stain'][s]['he_structure'].get(key, float('nan'))
                for s in args.stains]
        macro[key] = float(np.mean([v for v in vals if not np.isnan(v)]))

    # Ki67 clinical metrics (only present for Ki67 stain)
    for key in ki67_keys:
        vals = [results['per_stain'][s].get('ki67_clinical', {}).get(key, float('nan'))
                for s in args.stains]
        valid = [v for v in vals if not np.isnan(v)]
        macro[key] = float(np.mean(valid)) if valid else float('nan')

    results['macro_average'] = macro

    # Print table
    header = f"{'Metric':<20s}"
    for s in args.stains:
        header += f" {s:>8s}"
    header += f" {'Macro':>8s}"
    print(header)
    print("-" * len(header))

    all_table_keys = metric_keys + dab_keys + iod_keys + he_struct_keys
    # Append Ki67 rows only when Ki67 was evaluated
    has_ki67 = any('ki67_clinical' in results['per_stain'].get(s, {}) for s in args.stains)
    if has_ki67:
        all_table_keys += ki67_keys

    for key in all_table_keys:
        row = f"{key:<20s}"
        for s in args.stains:
            if key in ['fid_inception', 'fid_uni', 'kid_mean_x1000', 'lpips_mean', 'lpips_128_mean', 'ssim_mean', 'psnr_mean']:
                src = 'image_quality'
            elif key.startswith('dab'):
                src = 'dab'
            elif key.startswith('miod'):
                src = 'iod'
            elif key.startswith('ki67'):
                src = 'ki67_clinical'
            else:
                src = 'he_structure'
            val = results['per_stain'][s].get(src, {}).get(key, float('nan'))
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
