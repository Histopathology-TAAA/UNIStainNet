#!/usr/bin/env python3
"""
Train a unified multi-stain UNIStainNet on MIST IHC stains (HER2, Ki67, ER, PR).

A single model conditioned on a stain-type embedding. The class embedding
(nn.Embedding(5, 256)) is repurposed as a stain embedding:
    0 = HER2, 1 = Ki67, 2 = ER, 3 = PR, 4 = null (CFG dropout)

Usage:
    python scripts/train/train_mist.py --data_dir /path/to/MIST
    python scripts/train/train_mist.py --data_dir /path/to/MIST --stains HER2 Ki67
    python scripts/train/train_mist.py --data_dir /path/to/MIST --batch_size 8
"""

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from src.models.trainer import UNIStainNetTrainer
from src.data.mist_dataset import MISTMultiStainCropDataModule


def main():
    parser = argparse.ArgumentParser(description='Train UNIStainNet on MIST')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to MIST root directory (contains HER2/, Ki67/, etc.)')
    parser.add_argument('--stains', nargs='+', default=['HER2', 'Ki67', 'ER', 'PR'],
                        help='Stains to include (default: all 4)')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/mist_multistain',
                        help='Checkpoint save directory')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (16 for 80GB A100, 8 for 40GB)')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Max epochs')
    parser.add_argument('--wandb_name', type=str, default='mist_multistain',
                        help='Wandb run name')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--case_a_prob', type=float, default=0.0,
                        help='Probability of choosing Case A (IHC H-channel as edge input) '
                             'per training step. 0.0=always Case B, 1.0=always Case A. '
                             'Default: 0.0 (pure Case B, original behaviour)')
    parser.add_argument('--use_alignment', action='store_true',
                        help='Enable Spatial Alignment Network (STN) for fixing misalignment in Case A.')
    parser.add_argument('--edge_encoder', type=str, default='v2', choices=['v1', 'v2', 'none'],
                        help="Edge encoder type: 'v1', 'v2', or 'none'. Default: 'v2'")
    parser.add_argument('--uni_perceptual_weight', type=float, default=0.0,
                        help='Weight for the UNI perceptual feature loss (only active when Case A is running). Default: 0.0')
    args = parser.parse_args()

    print("=" * 70)
    print(f"TRAINING: Unified Multi-Stain UNIStainNet")
    print(f"  Stains: {args.stains}")
    print("=" * 70)

    # Convert 'none' to Python None/False
    edge_enc = None if args.edge_encoder == 'none' else args.edge_encoder

    # Paper hyperparameters
    model = UNIStainNetTrainer(
        # Architecture (identical to BCI except no class conditioning)
        num_classes=5,          # 4 stains + null
        null_class=4,
        class_dim=64,
        uni_dim=1024,
        ndf=64,
        input_skip=True,
        edge_encoder=edge_enc,
        edge_base_ch=32,
        uni_spatial_size=32,    # 32x32 patch tokens from UNI
        label_names=['HER2', 'Ki67', 'ER', 'PR'],
        # Optimizer
        gen_lr=1e-4,
        disc_lr=4e-4,
        warmup_steps=1000,
        # Loss weights (paper configuration)
        lpips_weight=1.0,
        lpips_256_weight=0.5,
        lpips_512_weight=0.0,
        he_edge_weight=0.5,
        l1_lowres_weight=1.0,
        adversarial_weight=0.0,
        uncond_disc_weight=1.0,
        dab_intensity_weight=0.2,
        dab_contrast_weight=0.0,    # No class ordering across stains
        dab_sharpness_weight=0.0,
        gram_style_weight=0.0,
        edge_weight=0.0,
        crop_disc_weight=0.0,
        feat_match_weight=10.0,
        patchnce_weight=0.0,
        bg_white_weight=0.0,
        uni_perceptual_weight=args.uni_perceptual_weight,
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
        # Case A/B H-channel training switch
        case_a_prob=args.case_a_prob,
        # Spatial Alignment Network (STN)
        use_alignment=args.use_alignment,
    )

    dm = MISTMultiStainCropDataModule(
        base_dir=args.data_dir,
        stains=args.stains,
        batch_size=args.batch_size,
        num_workers=4,
        image_size=(512, 512),
        crop_size=512,
        null_class=4,
    )

    ckpt_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename='mist_{epoch:03d}_{step:06d}',
        save_top_k=3,
        monitor='val/lpips',
        mode='min',
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    wandb_logger = WandbLogger(
        project='Destaining-unistainnet-v3',
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
