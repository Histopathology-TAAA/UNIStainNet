#!/usr/bin/env python3
"""
Stage-2 spatial finetuning for UNIStainNet on ACROBAT.

Loads a pretrained Stage-1 UNIStainNet checkpoint, freezes the encoder
and bottleneck, adds a SpatialGNN module, and finetunes only the GNN +
decoder SPADE blocks with spatial grid batches.

Per-stain GNN hop counts:
    ER   = 3 hops  (~1.5 mm context at 10X 512px)
    PR   = 3 hops
    HER2 = 5 hops  (~2.5 mm)
    KI67 = 7 hops  (~3.5 mm)

Usage:
    python scripts/train/train_spatial_acrobat.py \\
        --checkpoint checkpoints/unistainnet_acrobat_v1.ckpt \\
        --h5_path data/processed/acrobat/patches.h5 \\
        --train_patients 0,1,2,3,4,5,6,7 \\
        --val_patients 8,9
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Ensure submodule root is on path
_UNISTAINNET_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_UNISTAINNET_ROOT))

from src.models.trainer import UNIStainNetTrainer
from src.models.generator import SPADEUNetGenerator
from src.models.spatial_gnn import SpatialGNN
from src.data.acrobat_dataset import ACROBATMultiStainDataset, STAIN_TO_LABEL
from src.data.spatial_sampler import spatial_collate_fn


# Per-stain GNN hop count
STAIN_HOPS = {'HER2': 5, 'KI67': 7, 'ER': 3, 'PGR': 3}


class SpatialTrainer(UNIStainNetTrainer):
    """UNIStainNet training module extended with SpatialGNN.

    Freezes encoder + bottleneck, adds GAT-based spatial message passing
    that conditions each patch's decoder on its spatial neighbors.
    """

    def __init__(self, gnn_hops=5, gnn_hidden=256, spatial_dim=64,
                 grid_size=3, **kwargs):
        # Force spatial flags in generator
        kwargs['use_spatial'] = True
        kwargs['spatial_dim'] = spatial_dim
        super().__init__(**kwargs)

        self.gnn_hops = gnn_hops
        self.grid_size = grid_size
        self.gnn = SpatialGNN(
            feature_dim=512,
            hidden_dim=gnn_hidden,
            output_dim=spatial_dim,
            num_layers=gnn_hops,
        )

    def freeze_encoder(self):
        """Freeze encoder + bottleneck. Only decoder SPADE + GNN trained."""
        frozen_prefixes = (
            'enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'enc5',
            'bottleneck', 'uni_processor', 'class_embed',
        )
        for name, param in self.generator.named_parameters():
            if any(name.startswith(p) for p in frozen_prefixes):
                param.requires_grad = False

        n_total = sum(p.numel() for p in self.generator.parameters())
        n_trainable = sum(p.numel() for p in self.generator.parameters()
                          if p.requires_grad)
        n_gnn = sum(p.numel() for p in self.gnn.parameters())
        print(f"Encoder frozen. Generator: {n_trainable:,}/{n_total:,} trainable")
        print(f"SpatialGNN: {n_gnn:,} params")

    def configure_optimizers(self):
        """Optimizer for GNN + decoder params only."""
        gen_params = [p for p in self.generator.parameters() if p.requires_grad]
        gnn_params = list(self.gnn.parameters())
        if self.patchnce_loss is not None:
            gen_params += list(self.patchnce_loss.parameters())

        opt_g = torch.optim.Adam(
            gen_params + gnn_params,
            lr=self.hparams.gen_lr,
            betas=(0.0, 0.999),
        )
        disc_params = list(self.discriminator.parameters())
        if self.crop_discriminator is not None:
            disc_params += list(self.crop_discriminator.parameters())
        if self.uncond_discriminator is not None:
            disc_params += list(self.uncond_discriminator.parameters())
        opt_d = torch.optim.Adam(
            disc_params,
            lr=self.hparams.disc_lr,
            betas=(0.0, 0.999),
        )
        return [opt_g, opt_d]

    def _get_encoder_features(self, he):
        """Extract bottleneck features for GNN input (frozen, no grad)."""
        with torch.no_grad():
            if self.generator.enc0 is not None:
                e0 = self.generator.enc0(he)
                enc_input = e0
            else:
                enc_input = he
            e1 = self.generator.enc1(enc_input)
            e2 = self.generator.enc2(e1)
            e3 = self.generator.enc3(e2)
            e4 = self.generator.enc4(e3)
            e5 = self.generator.enc5(e4)
        return e5.mean(dim=[2, 3])  # [B, 512] global average pooled

    def training_step(self, batch, batch_idx):
        he, ihc, uni_or_crops, labels, fnames, positions = batch
        opt_g, opt_d = self.optimizers()

        # On-the-fly UNI extraction
        if self._uni_extract_on_the_fly:
            uni = self._extract_uni_from_sub_crops(uni_or_crops)
        else:
            uni = uni_or_crops

        # CFG dropout
        labels_dropped, uni_dropped = self._apply_cfg_dropout(labels, uni)

        if self.hparams.disable_uni:
            uni_dropped = torch.zeros_like(uni_dropped)
        if self.hparams.disable_class:
            labels_dropped = torch.full_like(labels_dropped, self.null_class)

        # Spatial GNN: get encoder features, run message passing
        encoder_feats = self._get_encoder_features(he)
        spatial_emb = self.gnn(encoder_feats, positions)

        # Generator forward with spatial context
        generated = self.generator(he, uni_dropped, labels_dropped,
                                   spatial_emb=spatial_emb)

        # --- Loss computations (same as UNIStainNetTrainer) ---

        # LPIPS at reduced resolution
        lpips_size = self.hparams.image_size // 4
        gen_lpips = F.interpolate(generated, size=lpips_size,
                                  mode='bilinear', align_corners=False)
        ihc_lpips = F.interpolate(ihc, size=lpips_size,
                                  mode='bilinear', align_corners=False)
        loss_lpips = self.lpips_fn(gen_lpips, ihc_lpips).mean()
        loss_g = self.hparams.lpips_weight * loss_lpips
        self.log('train/lpips', loss_lpips, prog_bar=True)

        # LPIPS fine
        if self.hparams.lpips_256_weight > 0:
            fine_size = self.hparams.image_size // 2
            gen_fine = F.interpolate(generated, size=fine_size,
                                     mode='bilinear', align_corners=False)
            ihc_fine = F.interpolate(ihc, size=fine_size,
                                     mode='bilinear', align_corners=False)
            loss_lpips_fine = self.lpips_fn(gen_fine, ihc_fine).mean()
            loss_g = loss_g + self.hparams.lpips_256_weight * loss_lpips_fine
            self.log('train/lpips_fine', loss_lpips_fine)

        # L1 at low resolution
        if self.hparams.l1_lowres_weight > 0:
            gen_64 = F.interpolate(generated, size=64,
                                   mode='bilinear', align_corners=False)
            ihc_64 = F.interpolate(ihc, size=64,
                                   mode='bilinear', align_corners=False)
            loss_l1 = F.l1_loss(gen_64, ihc_64)
            loss_g = loss_g + self.hparams.l1_lowres_weight * loss_l1
            self.log('train/l1_lowres', loss_l1)

        # DAB intensity
        if self.hparams.dab_intensity_weight > 0:
            loss_dab = self.compute_dab_intensity_loss(generated, ihc)
            loss_g = loss_g + self.hparams.dab_intensity_weight * loss_dab

        # DAB contrast
        if self.hparams.dab_contrast_weight > 0:
            loss_dab_contrast = self.compute_dab_contrast_loss(generated, labels)
            loss_g = loss_g + self.hparams.dab_contrast_weight * loss_dab_contrast

        # H&E edge preservation
        if self.hparams.he_edge_weight > 0:
            loss_he_edge = self.compute_he_edge_loss(generated, he)
            loss_g = loss_g + self.hparams.he_edge_weight * loss_he_edge

        # Adversarial (after warmup)
        loss_adv = torch.tensor(0.0, device=self.device)
        if self.global_step >= self.hparams.adversarial_start_step:
            # Generator adversarial
            if self.hparams.adversarial_weight > 0:
                fake_cond = torch.cat([he, generated], dim=1)
                fake_pred = self.discriminator(fake_cond)
                loss_adv = self.hparams.adversarial_weight * sum(
                    -p.mean() for p in fake_pred
                )
                loss_g = loss_g + loss_adv

            # Feature matching
            if self.hparams.feat_match_weight > 0 and self.uncond_discriminator is not None:
                real_feats = self.uncond_discriminator(ihc, return_features=True)
                fake_feats = self.uncond_discriminator(generated, return_features=True)
                loss_fm = sum(
                    F.l1_loss(f, r.detach()) for f, r in zip(fake_feats, real_feats)
                ) / len(real_feats)
                loss_g = loss_g + self.hparams.feat_match_weight * loss_fm

            # Unconditional adversarial
            if self.hparams.uncond_disc_weight > 0 and self.uncond_discriminator is not None:
                fake_uncond_pred = self.uncond_discriminator(generated)
                loss_uncond = self.hparams.uncond_disc_weight * sum(
                    -p.mean() for p in fake_uncond_pred
                )
                loss_g = loss_g + loss_uncond

        # Generator step
        lr_scale = self._get_lr_scale()
        for pg in opt_g.param_groups:
            pg['lr'] = pg['lr'] * lr_scale
        opt_g.zero_grad()
        self.manual_backward(loss_g)
        opt_g.step()

        # Discriminator step
        if self.global_step >= self.hparams.adversarial_start_step:
            opt_d.zero_grad()
            loss_d_total = torch.tensor(0.0, device=self.device)

            # Conditional discriminator
            if self.hparams.adversarial_weight > 0:
                fake_cond = torch.cat([he, generated.detach()], dim=1)
                real_cond = torch.cat([he, ihc], dim=1)
                fake_pred = self.discriminator(fake_cond)
                real_pred = self.discriminator(real_cond)
                loss_d = sum(
                    F.relu(1.0 + p).mean() + F.relu(1.0 - r).mean()
                    for p, r in zip(fake_pred, real_pred)
                ) / len(fake_pred)
                loss_d_total = loss_d_total + loss_d

            # Unconditional discriminator
            if self.hparams.uncond_disc_weight > 0 and self.uncond_discriminator is not None:
                fake_pred = self.uncond_discriminator(generated.detach())
                real_pred = self.uncond_discriminator(ihc)
                loss_ud = (
                    F.relu(1.0 + fake_pred).mean() +
                    F.relu(1.0 - real_pred).mean()
                )
                loss_d_total = loss_d_total + self.hparams.uncond_disc_weight * loss_ud

            if loss_d_total.item() > 0:
                self.manual_backward(loss_d_total)
                opt_d.step()

            # R1 gradient penalty
            if (self.hparams.r1_weight > 0
                    and self.global_step % self.hparams.r1_every == 0):
                ihc_r1 = ihc.detach().requires_grad_(True)
                real_cond_r1 = torch.cat([he, ihc_r1], dim=1)
                real_pred_r1 = self.discriminator(real_cond_r1)
                r1_loss = sum(
                    (p ** 2).sum() * 0.5 for p in
                    torch.autograd.grad(
                        outputs=[r.mean() for r in real_pred_r1],
                        inputs=ihc_r1,
                        create_graph=True,
                        retain_graph=True,
                    )[0].view(ihc_r1.shape[0], -1)
                ) / ihc_r1.shape[0]
                (self.hparams.r1_weight * r1_loss).backward()
                opt_d.step()

        self._update_ema()

        self.log('train/loss_g', loss_g, prog_bar=True)
        return loss_g

    def validation_step(self, batch, batch_idx):
        he, ihc, uni_or_crops, labels, fnames, positions = batch

        if self._uni_extract_on_the_fly:
            uni = self._extract_uni_from_sub_crops(uni_or_crops)
        else:
            uni = uni_or_crops

        encoder_feats = self._get_encoder_features(he)
        spatial_emb = self.gnn(encoder_feats, positions)

        generated = self.generator_ema(he, uni, labels, spatial_emb=spatial_emb)

        lpips_size = self.hparams.image_size // 4
        gen_lpips = F.interpolate(generated, size=lpips_size,
                                  mode='bilinear', align_corners=False)
        ihc_lpips = F.interpolate(ihc, size=lpips_size,
                                  mode='bilinear', align_corners=False)
        val_lpips = self.lpips_fn(gen_lpips, ihc_lpips).mean()
        self.log('val/lpips', val_lpips, prog_bar=True)

        return val_lpips


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Stage-2 Spatial GNN finetuning for UNIStainNet')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to Stage-1 UNIStainNet checkpoint')
    parser.add_argument('--h5_path', type=str, required=True)
    parser.add_argument('--train_patients', type=str, required=True,
                        help='Comma-separated patient IDs for training')
    parser.add_argument('--val_patients', type=str, required=True)
    parser.add_argument('--gnn_hops', type=int, default=5,
                        help='GNN hop count (3=ER/PR, 5=HER2, 7=Ki67)')
    parser.add_argument('--gnn_hidden', type=int, default=256)
    parser.add_argument('--spatial_dim', type=int, default=64)
    parser.add_argument('--grid_size', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=9)
    parser.add_argument('--gen_lr', type=float, default=5e-5)
    parser.add_argument('--disc_lr', type=float, default=2e-4)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--stains', nargs='+',
                        default=['HER2', 'KI67', 'ER', 'PGR'])
    args = parser.parse_args()

    train_patients = [int(x) for x in args.train_patients.split(',')]
    val_patients = [int(x) for x in args.val_patients.split(',')]

    # Load Stage-1 checkpoint
    print(f"Loading Stage-1 checkpoint: {args.checkpoint}")
    model = SpatialTrainer.load_from_checkpoint(
        args.checkpoint,
        gnn_hops=args.gnn_hops,
        gnn_hidden=args.gnn_hidden,
        spatial_dim=args.spatial_dim,
        grid_size=args.grid_size,
        gen_lr=args.gen_lr,
        disc_lr=args.disc_lr,
        strict=False,  # GNN params don't exist in Stage-1 ckpt
    )

    # Freeze encoder
    model.freeze_encoder()

    # Data modules
    from torch.utils.data import DataLoader

    train_ds = ACROBATMultiStainDataset(
        h5_path=args.h5_path,
        stains=args.stains,
        patient_ids=train_patients,
        image_size=(512, 512),
        crop_size=512,
        augment=True,
    )
    val_ds = ACROBATMultiStainDataset(
        h5_path=args.h5_path,
        stains=args.stains,
        patient_ids=val_patients,
        image_size=(512, 512),
        crop_size=512,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.grid_size ** 2,
        shuffle=True, num_workers=4, pin_memory=True,
        collate_fn=lambda b: spatial_collate_fn(b, grid_size=args.grid_size),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.grid_size ** 2,
        shuffle=False, num_workers=4, pin_memory=True,
        collate_fn=lambda b: spatial_collate_fn(b, grid_size=args.grid_size),
    )

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath='checkpoints/spatial',
        filename=f'spatial-gnn{args.gnn_hops}hop-{{epoch:02d}}-{{val/lpips:.3f}}',
        save_top_k=3,
        monitor='val/lpips',
        mode='min',
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_cb, lr_monitor],
        accelerator='gpu',
        devices=1,
        precision='16-mixed',
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
