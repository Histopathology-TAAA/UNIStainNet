#!/usr/bin/env python3
"""
Generate 10-image H&E | Real IHC | Generated IHC grids per validation patient.

For each patient, collects up to 10 samples distributed across all 4 stains
and saves a single composite grid. No metrics — just visual outputs.

Usage:
    python scripts/eval/eval_patient_plots.py \
        --checkpoint checkpoints/acrobat_512/acrobat_epoch=040_step=059845.ckpt \
        --h5_path /scratch/exp1/a.ayman/diff/data/acrobat-breast-patches-v3/patches_v3.h5
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import timm
import torchvision.transforms as transforms
import torchvision
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from src.models.trainer import UNIStainNetTrainer
from src.data.acrobat_dataset import ACROBATDataModule, STAIN_TO_LABEL


# Validation patients from train_acrobat.sh
VAL_PATIENTS = [
    12, 13, 20, 105, 108, 109, 115, 122, 123, 134, 135, 141, 144,
    151, 152, 155, 160, 161, 165, 168, 174, 175, 176, 182, 192, 197,
    202, 211, 217, 222, 225, 229, 237, 239, 258, 259, 264, 266
]

STAIN_ORDER = ['ER']
LABEL_TO_STAIN = {v: k for k, v in STAIN_TO_LABEL.items()}
N_SAMPLES = 10


def load_uni_model():
    model = timm.create_model(
        "hf-hub:MahmoodLab/uni", pretrained=True,
        init_values=1e-5, dynamic_img_size=True
    )
    model = model.cuda().eval()
    return model


def extract_features_for_crop(uni_model, he_crop_01, spatial_pool_size=32):
    """Extract UNI patch-token features from a 512x512 H&E crop."""
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


def composite_background(generated, he_images, threshold=0.85):
    """Replace background regions in generated images with white."""
    he_01 = ((he_images + 1) / 2).clamp(0, 1)
    brightness = he_01.mean(dim=1, keepdim=True)
    tissue = (brightness < threshold).float()
    tissue = F.avg_pool2d(tissue, kernel_size=7, stride=1, padding=3)
    tissue = (tissue > 0.3).float()
    tissue = F.avg_pool2d(tissue, kernel_size=11, stride=1, padding=5)
    return generated * tissue + 1.0 * (1.0 - tissue)


def save_labeled_grid(he, real, generated, fnames, output_path, n=10):
    """Save a grid with H&E | Real | Generated columns and stain/filename labels."""
    n = min(n, len(he))
    grid_images = []
    for i in range(n):
        grid_images.extend([
            ((he[i] + 1) / 2).clamp(0, 1),
            ((real[i] + 1) / 2).clamp(0, 1),
            ((generated[i] + 1) / 2).clamp(0, 1),
        ])
    grid = torchvision.utils.make_grid(grid_images, nrow=3, padding=2)
    grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(grid_np)

    draw = ImageDraw.Draw(pil_img)
    h = pil_img.height // n
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for i in range(n):
        y = i * h + 5
        label = fnames[i] if i < len(fnames) else ""
        draw.text((5, y), label, fill=(255, 255, 0), font=font)

    pil_img.save(output_path)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate per-patient 10-image plots (no metrics)'
    )
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--h5_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='eval_output/patient_plots')
    parser.add_argument('--guidance_scale', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--composite_bg', action='store_true',
                        help='Replace background with white')
    parser.add_argument('--val_patients', nargs='+', type=int, default=None,
                        help='Patient IDs (default: validation split from train_acrobat.sh)')
    parser.add_argument('--stains', nargs='+', default=STAIN_ORDER,
                        help=f'Stains to generate (default: {STAIN_ORDER})')
    parser.add_argument('--n_samples', type=int, default=N_SAMPLES,
                        help=f'Samples per patient grid (default: {N_SAMPLES})')
    args = parser.parse_args()

    val_patients = args.val_patients if args.val_patients is not None else VAL_PATIENTS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"PER-PATIENT PLOTS: {len(val_patients)} patients")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output dir: {output_dir}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print("=" * 60)

    print("Loading model...")
    model = UNIStainNetTrainer.load_from_checkpoint(args.checkpoint, strict=False)
    model = model.cuda().eval()
    spatial_pool_size = getattr(model.hparams, 'uni_spatial_size', 32)

    print("Loading UNI...")
    uni_model = load_uni_model()

    for patient_id in tqdm(sorted(val_patients), desc="Patients"):
        he_parts, real_parts, gen_parts, fname_parts = [], [], [], []
        n_per_stain = max(1, args.n_samples // len(args.stains))

        for stain in args.stains:
            stain_label = STAIN_TO_LABEL[stain]

            dm = ACROBATDataModule(
                h5_path=args.h5_path,
                train_patients=[0],
                val_patients=[patient_id],
                stains=[stain],
                batch_size=min(args.batch_size, n_per_stain),
                num_workers=4,
                image_size=(512, 512),
                crop_size=512,
                null_class=stain_label,
            )
            dm.setup('test')
            test_loader = dm.test_dataloader()

            if len(test_loader.dataset) == 0:
                continue

            for batch in test_loader:
                he, ihc, _uni_sub, labels, batch_fnames = batch
                he, ihc = he.cuda().float(), ihc.cuda().float()

                if he.size(0) > n_per_stain:
                    he = he[:n_per_stain]
                    ihc = ihc[:n_per_stain]
                    batch_fnames = batch_fnames[:n_per_stain]

                stain_labels = torch.full(
                    (he.size(0),), stain_label, device='cuda', dtype=torch.long
                )

                he_01 = ((he + 1) / 2).clamp(0, 1)
                uni = extract_features_for_crop(
                    uni_model, he_01,
                    spatial_pool_size=spatial_pool_size
                ).cuda()

                gen = model.generate(
                    he, uni, stain_labels,
                    guidance_scale=args.guidance_scale,
                    seed=42,
                )

                he_parts.append(he.cpu())
                real_parts.append(ihc.cpu())
                gen_parts.append(gen.cpu())
                fname_parts.extend(batch_fnames)
                break  # one batch per stain

        if not he_parts:
            print(f"  Patient {patient_id}: no data, skipping")
            continue

        he_all = torch.cat(he_parts)[:args.n_samples]
        real_all = torch.cat(real_parts)[:args.n_samples]
        gen_all = torch.cat(gen_parts)[:args.n_samples]
        fnames_all = fname_parts[:args.n_samples]

        if args.composite_bg:
            gen_all = composite_background(gen_all, he_all)

        out_path = output_dir / f"patient_{patient_id}_samples.png"
        save_labeled_grid(he_all, real_all, gen_all, fnames_all, out_path, n=args.n_samples)

    del uni_model
    torch.cuda.empty_cache()
    print(f"\nDone. {len(val_patients)} patient plots saved to {output_dir}")


if __name__ == '__main__':
    main()
