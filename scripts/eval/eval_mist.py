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
import numpy as np
from tqdm import tqdm

from src.models.trainer import UNIStainNetTrainer
from src.data.mist_dataset import MISTMultiStainCropDataModule, STAIN_TO_LABEL
from src.utils.dab import DABExtractor
from src.utils.metrics import (
    compute_image_quality_metrics,
    compute_uni_fid,
    compute_dab_metrics,
    compute_iod_metrics,
    save_sample_grid,
    composite_background,
)


@torch.no_grad()
def generate_for_stain(model, dataloader, stain_label, guidance_scale=1.0, seed=42):
    """Generate IHC images for a specific stain using the model's internal UNI pipeline."""
    all_gen, all_real, all_he = [], [], []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating")):
        he_rgb, ihc_rgb, he_h_map, _ihc_h_map, _labels, _fnames = batch
        he_rgb = he_rgb.cuda().float()
        ihc_rgb = ihc_rgb.cuda().float()
        he_h_map = he_h_map.cuda().float()

        # Override labels with the requested stain label for conditioning
        stain_labels = torch.full((he_rgb.size(0),), stain_label, device='cuda', dtype=torch.long)

        # Use model's internal UNI feature extraction pipeline
        uni_sub_crops = model._prepare_uni_sub_crops_from_tensor(he_rgb)
        uni = model._extract_uni_from_sub_crops(uni_sub_crops)

        # Generate from H-map input (not H&E RGB)
        gen = model.generate(he_h_map, uni, stain_labels,
                             guidance_scale=guidance_scale,
                             seed=seed + batch_idx)

        all_gen.append(gen.cpu())
        all_real.append(ihc_rgb.cpu())
        all_he.append(he_rgb.cpu())

    return torch.cat(all_gen), torch.cat(all_real), torch.cat(all_he)


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

    # Read image size and UNI spatial size from checkpoint hparams
    image_size = getattr(model.hparams, 'image_size', 512)
    uni_spatial_size = getattr(model.hparams, 'uni_spatial_size', 32)
    print(f"Image size: {image_size}x{image_size}")
    print(f"UNI spatial size: {uni_spatial_size}x{uni_spatial_size}")

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

        # Use MISTMultiStainCropDataModule with a single stain
        dm = MISTMultiStainCropDataModule(
            base_dir=args.data_dir,
            stains=[stain],
            batch_size=args.batch_size,
            num_workers=4,
            image_size=(image_size, image_size),
            crop_size=image_size,
            null_class=4,
        )
        dm.setup('test')
        test_loader = dm.test_dataloader()

        # Generate
        gen, real, he = generate_for_stain(
            model, test_loader, stain_label,
            guidance_scale=args.guidance_scale)
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

        # UNI-FID (per-stain)
        if not args.skip_uni_fid:
            print(f"  Computing UNI-FID...")
            try:
                stain_results['image_quality']['fid_uni'] = compute_uni_fid(gen, real)
            except Exception as e:
                print(f"    UNI-FID skipped: {e}")

        results['per_stain'][stain] = stain_results

        # Print per-stain summary
        iq = stain_results['image_quality']
        dab = stain_results['dab']
        print(f"\n  {stain}: FID={iq['fid_inception']:.1f} | "
              f"KID={iq['kid_mean_x1000']:.1f} | "
              f"LPIPS={iq['lpips_mean']:.3f} | "
              f"SSIM={iq['ssim_mean']:.3f} | "
              f"Pearson-r={dab.get('dab_pearson_r', 0):.3f}")

    # Free UNI model (lazily loaded inside the trainer)
    if model._uni_model is not None:
        del model._uni_model
    torch.cuda.empty_cache()

    # Macro-averaged summary
    print(f"\n{'='*70}")
    print("MACRO-AVERAGED RESULTS")
    print(f"{'='*70}")

    metric_keys = ['fid_inception', 'kid_mean_x1000', 'lpips_mean', 'lpips_128_mean',
                    'ssim_mean', 'psnr_mean']
    dab_keys = ['dab_mae_overall', 'dab_pearson_r', 'dab_kl', 'dab_jsd']
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

    for key in metric_keys + dab_keys + iod_keys:
        row = f"{key:<20s}"
        for s in args.stains:
            if key in ['fid_inception', 'kid_mean_x1000']:
                src = 'image_quality'
            elif key.startswith('dab'):
                src = 'dab'
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
