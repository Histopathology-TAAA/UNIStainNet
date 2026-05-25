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

Option A (resume): resume from step-48k checkpoint with tightened weights.
Option B (new run): add --eosin_multi_scale for 32x32 Eosin injection at D5.
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
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='Gradient accumulation steps (e.g., set to 2 for effective batch size 8 if physical is 4)')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Max epochs')
    parser.add_argument('--wandb_name', type=str, default='mist_multistain',
                        help='Wandb run name')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--case_b_prob', type=float, default=0.25,
                        help='Fraction of steps using H&E H-map (Case B / misaligned). '
                             '0.25 = 75%% aligned, 0.5 = classic 50/50. '
                             'Do not go below 0.20 (inference OOD risk).')
    parser.add_argument('--use_eosin_encoder', action='store_true', default=True,
                        help='Enable Eosin bottleneck injection (requires trainA-E / valA-E dirs). '
                             'Disable with --no_use_eosin_encoder if E-map dirs are not available.')
    parser.add_argument('--no_use_eosin_encoder', dest='use_eosin_encoder', action='store_false')
    # -------------------------------------------------------------------------
    # Option B: multi-scale Eosin injection (default OFF)
    #
    # Adds a second Eosin injection at decoder D5 (32×32) in addition to the
    # bottleneck (16×16). At 32×32 the 30px misalignment is <2px — still safe.
    # Gives the decoder a finer membrane signal, intended to help the HER2
    # membrane localization problem.
    #
    # HOW TO ENABLE for a new run:
    #   Add --eosin_multi_scale to your training command.
    #   Do NOT use with --resume_from — incompatible with checkpoints trained
    #   without it (different EosinEncoder state_dict keys: stage1/stage2/...
    #   vs the old sequential encoder.0/encoder.1/...).
    # -------------------------------------------------------------------------
    parser.add_argument('--eosin_multi_scale', action='store_true', default=False,
                        help='[Option B — new runs only] Second Eosin injection at D5 (32x32). '
                             'Incompatible with existing checkpoints. Only use for fresh runs.')
    parser.add_argument('--no_eosin_multi_scale', dest='eosin_multi_scale', action='store_false')
    parser.add_argument('--use_attention_for_spade', action='store_true', default=False,
                        help='Use decoder cross-attention output as SPADE spatial conditioning map '
                            '(instead of raw UNI spatial maps).')
    parser.add_argument('--no_use_attention_for_spade', dest='use_attention_for_spade', action='store_false')
    parser.add_argument('--enable_attention_residual', action='store_true', default=True,
                        help='Enable independent attention residual path in decoder (ablation toggle).')
    parser.add_argument('--disable_attention_residual', dest='enable_attention_residual', action='store_false')
    parser.add_argument('--spade_use_uni', action='store_true', default=True,
                        help='Condition SPADE blocks on UNI spatial maps. Disable for label-only SPADE ablation.')
    parser.add_argument('--no_spade_use_uni', dest='spade_use_uni', action='store_false')
    parser.add_argument('--edge_encoder', type=str, default='v2', choices=['none', 'v1', 'v2'],
                        help='Edge encoder mode: none disables it, v1 enables Sobel single-scale, '
                             'v2 enables multi-scale Sobel.')
    parser.add_argument('--use_h_adapter', action='store_true', default=False,
                        help='Enable H-channel domain adapter (IHC→H&E transformation). '
                             'Applied only during training on case A (IHC H-maps). '
                             'Default: disabled.')
    parser.add_argument('--log_val_lpips', action='store_true', default=True,
                        help='Log LPIPS during validation (default: enabled).')
    parser.add_argument('--no_log_val_lpips', dest='log_val_lpips', action='store_false')
    parser.add_argument('--log_val_ssim', action='store_true', default=True,
                        help='Log SSIM during validation (default: enabled).')
    parser.add_argument('--no_log_val_ssim', dest='log_val_ssim', action='store_false')
    parser.add_argument('--log_val_dab_mae', action='store_true', default=True,
                        help='Log DAB MAE during validation (default: enabled).')
    parser.add_argument('--no_log_val_dab_mae', dest='log_val_dab_mae', action='store_false')
    parser.add_argument('--log_val_fid', action='store_true', default=False,
                        help='Log FID during validation (default: disabled).')
    parser.add_argument('--no_log_val_fid', dest='log_val_fid', action='store_false')
    parser.add_argument('--log_val_kid', action='store_true', default=False,
                        help='Log KID during validation (default: disabled).')
    parser.add_argument('--no_log_val_kid', dest='log_val_kid', action='store_false')
    parser.add_argument('--log_val_unifid', action='store_true', default=False,
                        help='Log UNI-FID during validation (default: disabled).')
    parser.add_argument('--no_log_val_unifid', dest='log_val_unifid', action='store_false')
    args = parser.parse_args()

    print("=" * 70)
    print(f"TRAINING: Unified Multi-Stain UNIStainNet")
    print(f"  Stains: {args.stains}")
    if args.eosin_multi_scale:
        print("  [Option B] Multi-scale Eosin injection ENABLED (32x32 + 16x16)")
    print(f"  use_attention_for_spade: {args.use_attention_for_spade}")
    print(f"  enable_attention_residual: {args.enable_attention_residual}")
    print(f"  spade_use_uni: {args.spade_use_uni}")
    print(f"  edge_encoder: {args.edge_encoder}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Loss config (single edit point for ablations)
    #
    # Paper/original implementation baseline reference (as commonly used defaults):
    #   lpips_weight=1.0
    #   adversarial_weight=1.0
    #   dab_intensity_weight=0.1
    #   dab_contrast_weight=0.05
    #   other auxiliary losses typically disabled (0.0)
    #
    # Current V4 tuned defaults below (edit this dict to add/remove/change losses).
    # -------------------------------------------------------------------------
    LOSS_WEIGHTS = {
        'lpips_weight': 1.0,
        'lpips_256_weight': 0.5,
        'lpips_512_weight': 0.0,
        'l1_fullres_weight': 1.0,
        'lpips_fullres_weight': 1.0,
        'he_edge_weight': 0.5,
        'l1_lowres_weight': 1.0,
        'adversarial_weight': 0.0,
        'uncond_disc_weight': 1.0,
        'dab_intensity_weight': 0.2,
        'dab_contrast_weight': 0.0,
        'dab_sharpness_weight': 0.0,
        'gram_style_weight': 0.0,
        'edge_weight': 0.0,
        'crop_disc_weight': 0.0,
        'feat_match_weight': 10.0,
        'patchnce_weight': 0.0,
        'bg_white_weight': 0.0,
        # IHC / DAB losses
        'ihc_edge_weight': 0.1,
        'dab_histo_weight': 0.3,
        'dab_block_weight': 0.5,
        'dab_block_size': 32,
        'dab_sparsity_weight': 0.3,
        'dab_sparsity_margin': 0.05,
        # Stain-conditioned discriminator
        'proj_disc_weight': 2.0,
    }

    model = UNIStainNetTrainer(
        # Architecture
        num_classes=5,          # 4 stains + null
        null_class=4,
        class_dim=64,
        uni_dim=1024,
        ndf=64,
        input_skip=True,
        edge_encoder=False if args.edge_encoder == 'none' else args.edge_encoder,
        edge_base_ch=32,
        uni_spatial_size=32,    # 32x32 patch tokens from UNI
        label_names=['HER2', 'Ki67', 'ER', 'PR'],
        use_h_adapter=args.use_h_adapter,
        # Optimizer
        gen_lr=1e-4,
        disc_lr=4e-4,
        warmup_steps=1000,
        # Loss weights (centralized)
        **LOSS_WEIGHTS,
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
        # Decoder conditioning ablations
        use_attention_for_spade=args.use_attention_for_spade,
        enable_attention_residual=args.enable_attention_residual,
        spade_use_uni=args.spade_use_uni,
        # Domain routing
        case_b_prob=args.case_b_prob,
        # Eosin injection
        use_eosin_encoder=args.use_eosin_encoder,
        eosin_out_ch=64,
        eosin_multi_scale=args.eosin_multi_scale,   # Option B — default False
        # Validation metrics
        log_val_lpips=args.log_val_lpips,
        log_val_ssim=args.log_val_ssim,
        log_val_dab_mae=args.log_val_dab_mae,
        log_val_fid=args.log_val_fid,
        log_val_kid=args.log_val_kid,
        log_val_unifid=args.log_val_unifid,
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
        project='Destaining-UNIStainNet-V2-ER-PR-Only',
        # entity='histo-TAAAA',
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
        # accumulate_grad_batches=args.accum_steps,
    )

    trainer.fit(model, dm, ckpt_path=args.resume_from)
    print("Training complete!")


if __name__ == "__main__":
    main()
