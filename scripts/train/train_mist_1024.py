#!/usr/bin/env python3
"""
Train unified multi-stain UNIStainNet on MIST at native 1024x1024 resolution.

Same architecture and losses as train_mist.py but at 1024:
- image_size=1024 adds enc0 (1024->512) and dec1 (512->1024) levels
- No cropping — full 1024x1024 images used directly
- Discriminator/edge encoder operate at 512 internally

Usage:
    python scripts/train/train_mist_1024.py --data_dir /path/to/MIST
    python scripts/train/train_mist_1024.py --data_dir /path/to/MIST --batch_size 4
"""

import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from src.models.trainer import UNIStainNetTrainer
from src.data.mist_dataset import MISTMultiStainCropDataModule


def main():
    parser = argparse.ArgumentParser(description='Train UNIStainNet on MIST 1024')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to MIST root directory (contains HER2/, Ki67/, etc.)')
    parser.add_argument('--stains', nargs='+', default=['HER2', 'Ki67', 'ER', 'PR'],
                        help='Stains to include (default: all 4)')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/mist_1024')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (8 for 80GB A100, 4 if OOM)')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--wandb_name', type=str, default='mist_1024')
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--pretrained_512_ckpt', type=str, default=None,
                        help='Path to a 512-resolution checkpoint to initialize weights from')
    args = parser.parse_args()

    print("=" * 70)
    print("TRAINING: Unified Multi-Stain UNIStainNet — NATIVE 1024x1024")
    print(f"  Stains: {args.stains}")
    print("=" * 70)

    model = UNIStainNetTrainer(
        # Architecture
        num_classes=5,
        null_class=4,
        class_dim=64,
        uni_dim=1024,
        ndf=64,
        input_skip=True,
        edge_encoder='v2',
        edge_base_ch=32,
        uni_spatial_size=32,
        image_size=1024,
        label_names=['HER2', 'Ki67', 'ER', 'PR'],
        # Optimizer
        gen_lr=1e-4,
        disc_lr=4e-4,
        warmup_steps=1000,
        # Loss weights (same as 512 MIST)
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

    dm = MISTMultiStainCropDataModule(
        base_dir=args.data_dir,
        stains=args.stains,
        batch_size=args.batch_size,
        num_workers=4,
        image_size=(1024, 1024),
        crop_size=1024,
        null_class=4,
    )

    ckpt_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename='mist_1024_{epoch:03d}_{step:06d}',
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

    if args.pretrained_512_ckpt:
        print(f"Loading pretrained 512 weights from {args.pretrained_512_ckpt}...")
        ckpt = torch.load(args.pretrained_512_ckpt, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        # Filter out UNI model keys if they exist
        keys_to_remove = [k for k in state_dict if k.startswith('_uni_model.')]
        
        # Filter out keys with shape mismatches (e.g. generator.output layers which change size from 512 to 1024)
        model_state = model.state_dict()
        for k, v in state_dict.items():
            if k in model_state and v.shape != model_state[k].shape:
                print(f"Skipping key {k} due to shape mismatch: {v.shape} vs {model_state[k].shape}")
                keys_to_remove.append(k)
                
        for k in set(keys_to_remove):
            if k in state_dict:
                del state_dict[k]
                
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    trainer.fit(model, dm, ckpt_path=args.resume_from)
    print("Training complete!")


if __name__ == "__main__":
    main()
