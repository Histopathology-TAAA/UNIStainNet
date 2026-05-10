#!/usr/bin/env python3
"""
Train a unified multi-stain UNIStainNet on ACROBAT HDF5 patch dataset.

Stain embedding convention (compatible with the MIST trainer):
    0 = HER2, 1 = KI67, 2 = ER, 3 = PGR, 4 = null (CFG dropout)

The dataset is split by patient ID so that patches from the same patient
never leak across train/validation.

Usage:
    python scripts/train/train_acrobat.py --h5_path /path/to/pipeline_test.h5
    python scripts/train/train_acrobat.py --h5_path /path/to/pipeline_test.h5 --stains HER2 KI67
    python scripts/train/train_acrobat.py --h5_path /path/to/pipeline_test.h5 --batch_size 8 --max_epochs 200
"""

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from src.models.trainer import UNIStainNetTrainer
from src.data.acrobat_dataset import ACROBATDataModule


# Default patient split (11 patients in pipeline_test.h5).
# Validation patients chosen to include most stains.
DEFAULT_TRAIN_PATIENTS = [1, 10, 100, 102, 104, 105, 106, 107]   # 8 patients
DEFAULT_VAL_PATIENTS = [0, 101, 103]                               # 3 patients (all have full stain coverage)


def main():
    parser = argparse.ArgumentParser(description='Train UNIStainNet on ACROBAT')
    parser.add_argument('--h5_path', type=str, required=True,
                        help='Path to pipeline_test.h5 (or equivalent HDF5 file)')
    parser.add_argument('--stains', nargs='+',
                        default=['HER2', 'KI67', 'ER', 'PGR'],
                        help='Stains to train on (default: all 4)')
    parser.add_argument('--train_patients', nargs='+', type=int,
                        default=DEFAULT_TRAIN_PATIENTS,
                        help='Patient IDs for training set')
    parser.add_argument('--val_patients', nargs='+', type=int,
                        default=DEFAULT_VAL_PATIENTS,
                        help='Patient IDs for validation set')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/acrobat_512',
                        help='Checkpoint save directory')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (16 for 80GB A100, 8 for 40GB)')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Max epochs')
    parser.add_argument('--wandb_name', type=str, default='acrobat_multistain',
                        help='Wandb run name')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Training resolution (512 or 1024)')
    args = parser.parse_args()

    image_sz = (args.image_size, args.image_size)
    crop_sz = args.image_size

    print("=" * 70)
    print(f"TRAINING: Unified Multi-Stain UNIStainNet on ACROBAT")
    print(f"  HDF5:       {args.h5_path}")
    print(f"  Stains:     {args.stains}")
    print(f"  Train pts:  {args.train_patients}")
    print(f"  Val pts:    {args.val_patients}")
    print(f"  Resolution: {args.image_size}×{args.image_size}")
    print("=" * 70)

    # Paper hyperparameters
    model = UNIStainNetTrainer(
        num_classes=5,          # 4 stains + null
        null_class=4,
        class_dim=64,
        uni_dim=1024,
        ndf=64,
        input_skip=True,
        edge_encoder='v2',
        edge_base_ch=32,
        uni_spatial_size=32,
        label_names=['HER2', 'KI67', 'ER', 'PGR'],
        # Optimizer
        gen_lr=1e-4,
        disc_lr=4e-4,
        warmup_steps=1000,
        # Loss weights
        lpips_weight=1.0,
        lpips_256_weight=0.5,
        lpips_512_weight=0.0,
        he_edge_weight=0.5,
        l1_lowres_weight=1.0,
        adversarial_weight=0.0,
        uncond_disc_weight=1.0,
        dab_intensity_weight=0.2,
        dab_contrast_weight=0.0,
        dab_sharpness_weight=0.0,
        gram_style_weight=0.0,
        edge_weight=0.0,
        crop_disc_weight=0.0,
        feat_match_weight=10.0,
        patchnce_weight=0.0,
        bg_white_weight=0.0,
        # GAN training
        r1_weight=10.0,
        r1_every=16,
        adversarial_start_step=2000,
        # CFG dropout
        cfg_drop_class_prob=0.10,
        cfg_drop_uni_prob=0.10,
        cfg_drop_both_prob=0.05,
        # EMA
        ema_decay=0.999,
        # On-the-fly UNI extraction
        extract_uni_on_the_fly=True,
        uni_spatial_pool_size=32,
    )

    dm = ACROBATDataModule(
        h5_path=args.h5_path,
        train_patients=args.train_patients,
        val_patients=args.val_patients,
        stains=args.stains,
        batch_size=args.batch_size,
        num_workers=4,
        image_size=image_sz,
        crop_size=crop_sz,
        null_class=4,
    )

    ckpt_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename='acrobat_{epoch:03d}_{step:06d}',
        save_top_k=3,
        monitor='val/lpips',
        mode='min',
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    wandb_logger = WandbLogger(
        project='unistainnet',
        name=args.wandb_name,
        save_dir='wandb',
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=1,
        precision='bf16',
        callbacks=[ckpt_callback, lr_monitor],
        logger=wandb_logger,
        log_every_n_steps=10,
        val_check_interval=1.0,
    )

    trainer.fit(model, dm, ckpt_path=args.resume_from)
    print("Training complete!")


if __name__ == "__main__":
    main()
