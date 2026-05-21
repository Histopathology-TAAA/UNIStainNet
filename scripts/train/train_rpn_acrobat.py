#!/usr/bin/env python3
"""
Stage-2 Region Context finetuning for UNIStainNet on ACROBAT.

Uses a tiny RegionEncoder CNN (~50K params) that processes a low-res
H&E thumbnail (stored during patching) to produce a tissue compartment
feature map. Each patch bilinearly samples its region embedding from
this map, giving the generator awareness of its tissue context.

Per-stain grid resolution (higher = finer boundary detection):
    ER   = 8×8  (tumor vs stroma at ~1.9mm resolution)
    PR   = 8×8
    HER2 = 16×16 (~0.9mm — finer for membrane boundary detection)
    KI67 = 8×8  (cell density at ~1.9mm resolution)

VRAM: ~2 GB (same as UNIStainNet baseline)
Overhead: ~0.95× baseline throughput

Usage:
    python scripts/train/train_rpn_acrobat.py \\
        --checkpoint checkpoints/unistainnet_acrobat_v1.ckpt \\
        --h5_path data/processed/acrobat/patches.h5
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

_UNISTAINNET_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_UNISTAINNET_ROOT))

from src.models.trainer import UNIStainNetTrainer
from src.models.region_encoder import RegionEncoder
from src.data.acrobat_dataset import ACROBATMultiStainDataset, LABEL_TO_STAIN


class RPNTrainer(UNIStainNetTrainer):
    """UNIStainNet trainer extended with RegionEncoder for tissue context.

    Freezes encoder + bottleneck. RegionEncoder processes the WSI thumbnail
    (pre-stored in HDF5) to produce a compartment feature map. Each patch
    samples its region embedding via bilinear interpolation at its WSI
    position. Generator decoder + RegionEncoder are jointly finetuned.
    """

    def __init__(self, region_grid=8, region_dim=64, **kwargs):
        kwargs['use_spatial'] = True
        kwargs['spatial_dim'] = region_dim
        super().__init__(**kwargs)

        self.region_grid = region_grid
        self.region_encoder = RegionEncoder(
            output_dim=region_dim, grid_size=region_grid,
        )

        # Populated by training script before fit()
        self._h5_path = None
        self._patient_ids = None
        self._coord_x = None       # tensor [N_he] — global HE index → x
        self._coord_y = None       # tensor [N_he] — global HE index → y
        self._wsi_widths = {}      # pid → width in pixels
        self._wsi_heights = {}     # pid → height in pixels
        # Thumbnail cache: pid → [1, 64, G, G] feature map (GPU)
        self._thumb_cache = {}

    def setup_patient_data(self, h5_path, train_patients):
        """Load coordinate arrays + WSI dimensions from HDF5.

        Called once before training starts.
        """
        import h5py
        self._h5_path = h5_path
        self._patient_ids = set(train_patients)

        with h5py.File(h5_path, 'r') as f:
            self._coord_x = torch.as_tensor(f['index/coord_x'][:])
            self._coord_y = torch.as_tensor(f['index/coord_y'][:])

            for pid in train_patients:
                grp_name = f'patient_{pid}'
                if grp_name in f:
                    self._wsi_widths[pid] = f[grp_name].attrs['wsi_width']
                    self._wsi_heights[pid] = f[grp_name].attrs['wsi_height']

        print(f"Region data loaded: {len(self._wsi_widths)} patients with thumbnails")

    def _get_thumbnail(self, pid):
        """Load and encode a patient's thumbnail, caching the feature map."""
        if pid in self._thumb_cache:
            return self._thumb_cache[pid]

        import h5py
        with h5py.File(self._h5_path, 'r') as f:
            thumb = torch.as_tensor(
                f[f'patient_{pid}/thumbnail'][:]
            ).permute(2, 0, 1).unsqueeze(0)  # [1, 3, 512, 512]

        thumb = thumb.to(self.device)
        with torch.no_grad() if not self.training else torch.enable_grad():
            feat_map = self.region_encoder(thumb)
        self._thumb_cache[pid] = feat_map
        return feat_map

    def _get_region_emb(self, fnames):
        """Compute region embeddings for a batch.

        Per sample: load thumbnail (cached) → get feature map →
        RoIAlign around patch position → pool → project → flat vector.
        """
        device = self.device
        nxs, nys = [], []
        feat_maps = []

        for fname in fnames:
            # Parse: p{pid}_{stain}_{he_idx}_{ihc_idx}
            parts = fname.split('_')
            pid = int(parts[0][1:])
            he_idx = int(parts[2])

            feat_maps.append(self._get_thumbnail(pid))

            w = self._wsi_widths.get(pid, 1)
            h = self._wsi_heights.get(pid, 1)
            nxs.append((self._coord_x[he_idx].float() / w).to(device))
            nys.append((self._coord_y[he_idx].float() / h).to(device))

        # Batch samples from same WSI share feature maps.
        # For samples from different WSIs, process per unique feature map.
        # Simple approach: one forward per sample (tiny CNN, negligible overhead).
        embs = []
        for i, (nx, ny) in enumerate(zip(nxs, nys)):
            emb = self.region_encoder.get_region_emb(
                feat_maps[i],
                nx.unsqueeze(0),   # [1]
                ny.unsqueeze(0),   # [1]
            )
            embs.append(emb)

        return torch.cat(embs, dim=0)  # [B, spatial_dim]

    def freeze_encoder(self):
        """Freeze encoder + bottleneck + UNI processor."""
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
        n_region = sum(p.numel() for p in self.region_encoder.parameters())
        print(f"Encoder frozen: {n_trainable:,}/{n_total:,} generator params trainable")
        print(f"RegionEncoder: {n_region:,} params (all trainable)")

    def configure_optimizers(self):
        gen_params = [p for p in self.generator.parameters() if p.requires_grad]
        region_params = list(self.region_encoder.parameters())
        if self.patchnce_loss is not None:
            gen_params += list(self.patchnce_loss.parameters())

        opt_g = torch.optim.Adam(
            gen_params + region_params,
            lr=self.hparams.gen_lr,
            betas=(0.0, 0.999),
        )
        disc_params = list(self.discriminator.parameters())
        if self.crop_discriminator is not None:
            disc_params += list(self.crop_discriminator.parameters())
        if self.uncond_discriminator is not None:
            disc_params += list(self.uncond_discriminator.parameters())
        opt_d = torch.optim.Adam(
            disc_params, lr=self.hparams.disc_lr, betas=(0.0, 0.999),
        )
        return [opt_g, opt_d]

    def on_train_epoch_start(self):
        """Clear thumbnail cache each epoch to prevent memory creep."""
        self._thumb_cache.clear()

    def on_validation_epoch_start(self):
        self._thumb_cache.clear()

    def training_step(self, batch, batch_idx):
        he, ihc, uni_or_crops, labels, fnames = batch
        opt_g, opt_d = self.optimizers()

        # Region embeddings from thumbnail + WSI position
        region_emb = self._get_region_emb(fnames)

        if self._uni_extract_on_the_fly:
            uni = self._extract_uni_from_sub_crops(uni_or_crops)
        else:
            uni = uni_or_crops

        labels_dropped, uni_dropped = self._apply_cfg_dropout(labels, uni)
        if self.hparams.disable_uni:
            uni_dropped = torch.zeros_like(uni_dropped)
        if self.hparams.disable_class:
            labels_dropped = torch.full_like(labels_dropped, self.null_class)

        generated = self.generator(he, uni_dropped, labels_dropped,
                                   spatial_emb=region_emb)

        # --- Losses (same as UNIStainNetTrainer) ---
        lpips_size = self.hparams.image_size // 4
        gen_lpips = F.interpolate(generated, size=lpips_size,
                                  mode='bilinear', align_corners=False)
        ihc_lpips = F.interpolate(ihc, size=lpips_size,
                                  mode='bilinear', align_corners=False)
        loss_lpips = self.lpips_fn(gen_lpips, ihc_lpips).mean()
        loss_g = self.hparams.lpips_weight * loss_lpips
        self.log('train/lpips', loss_lpips, prog_bar=True)

        if self.hparams.lpips_256_weight > 0:
            fine_size = self.hparams.image_size // 2
            gen_fine = F.interpolate(generated, size=fine_size,
                                     mode='bilinear', align_corners=False)
            ihc_fine = F.interpolate(ihc, size=fine_size,
                                     mode='bilinear', align_corners=False)
            loss_lpips_fine = self.lpips_fn(gen_fine, ihc_fine).mean()
            loss_g = loss_g + self.hparams.lpips_256_weight * loss_lpips_fine

        if self.hparams.l1_lowres_weight > 0:
            gen_64 = F.interpolate(generated, size=64,
                                   mode='bilinear', align_corners=False)
            ihc_64 = F.interpolate(ihc, size=64,
                                   mode='bilinear', align_corners=False)
            loss_l1 = F.l1_loss(gen_64, ihc_64)
            loss_g = loss_g + self.hparams.l1_lowres_weight * loss_l1

        if self.hparams.dab_intensity_weight > 0:
            loss_dab = self.compute_dab_intensity_loss(generated, ihc)
            loss_g = loss_g + self.hparams.dab_intensity_weight * loss_dab

        if self.hparams.dab_contrast_weight > 0:
            loss_dab_contrast = self.compute_dab_contrast_loss(generated, labels)
            loss_g = loss_g + self.hparams.dab_contrast_weight * loss_dab_contrast

        if self.hparams.he_edge_weight > 0:
            loss_he_edge = self.compute_he_edge_loss(generated, he)
            loss_g = loss_g + self.hparams.he_edge_weight * loss_he_edge

        # Adversarial
        loss_adv = torch.tensor(0.0, device=self.device)
        if self.global_step >= self.hparams.adversarial_start_step:
            if self.hparams.adversarial_weight > 0:
                fake_cond = torch.cat([he, generated], dim=1)
                fake_pred = self.discriminator(fake_cond)
                loss_adv = self.hparams.adversarial_weight * sum(
                    -p.mean() for p in fake_pred)
                loss_g = loss_g + loss_adv

            if self.hparams.feat_match_weight > 0 and self.uncond_discriminator is not None:
                real_feats = self.uncond_discriminator(ihc, return_features=True)
                fake_feats = self.uncond_discriminator(generated, return_features=True)
                loss_fm = sum(
                    F.l1_loss(f, r.detach()) for f, r in zip(fake_feats, real_feats)
                ) / len(real_feats)
                loss_g = loss_g + self.hparams.feat_match_weight * loss_fm

            if self.hparams.uncond_disc_weight > 0 and self.uncond_discriminator is not None:
                fake_uncond = self.uncond_discriminator(generated)
                loss_uncond = self.hparams.uncond_disc_weight * sum(
                    -p.mean() for p in fake_uncond)
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

            if self.hparams.adversarial_weight > 0:
                fake_cond = torch.cat([he, generated.detach()], dim=1)
                real_cond = torch.cat([he, ihc], dim=1)
                fake_pred_d = self.discriminator(fake_cond)
                real_pred_d = self.discriminator(real_cond)
                loss_d = sum(
                    F.relu(1.0 + p).mean() + F.relu(1.0 - r).mean()
                    for p, r in zip(fake_pred_d, real_pred_d)
                ) / len(fake_pred_d)
                loss_d_total = loss_d_total + loss_d

            if self.hparams.uncond_disc_weight > 0 and self.uncond_discriminator is not None:
                fake_up = self.uncond_discriminator(generated.detach())
                real_up = self.uncond_discriminator(ihc)
                loss_ud = (F.relu(1.0 + fake_up).mean()
                           + F.relu(1.0 - real_up).mean())
                loss_d_total = loss_d_total + self.hparams.uncond_disc_weight * loss_ud

            if loss_d_total.item() > 0:
                self.manual_backward(loss_d_total)
                opt_d.step()

            if (self.hparams.r1_weight > 0
                    and self.global_step % self.hparams.r1_every == 0):
                ihc_r1 = ihc.detach().requires_grad_(True)
                real_cond_r1 = torch.cat([he, ihc_r1], dim=1)
                real_pred_r1 = self.discriminator(real_cond_r1)
                r1_loss = sum(
                    (p ** 2).sum() * 0.5 for p in
                    torch.autograd.grad(
                        outputs=[r.mean() for r in real_pred_r1],
                        inputs=ihc_r1, create_graph=True, retain_graph=True,
                    )[0].view(ihc_r1.shape[0], -1)
                ) / ihc_r1.shape[0]
                (self.hparams.r1_weight * r1_loss).backward()
                opt_d.step()

        self._update_ema()
        self.log('train/loss_g', loss_g, prog_bar=True)
        return loss_g

    def validation_step(self, batch, batch_idx):
        he, ihc, uni_or_crops, labels, fnames = batch

        region_emb = self._get_region_emb(fnames)

        if self._uni_extract_on_the_fly:
            uni = self._extract_uni_from_sub_crops(uni_or_crops)
        else:
            uni = uni_or_crops

        with torch.no_grad():
            generated = self.generator_ema(he, uni, labels,
                                           spatial_emb=region_emb)

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
        description='Stage-2 Region Context finetuning (RPN) for UNIStainNet')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to Stage-1 UNIStainNet checkpoint')
    parser.add_argument('--h5_path', type=str, required=True)
    parser.add_argument('--train_patients', type=str, required=True,
                        help='Comma-separated patient IDs')
    parser.add_argument('--val_patients', type=str, required=True)
    parser.add_argument('--region_grid', type=int, default=8,
                        help='RegionEncoder grid resolution (8=1.9mm, 16=0.9mm)')
    parser.add_argument('--region_dim', type=int, default=64)
    parser.add_argument('--gen_lr', type=float, default=5e-5,
                        help='Finetuning LR for decoder + RegionEncoder')
    parser.add_argument('--disc_lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--stains', nargs='+',
                        default=['HER2', 'KI67', 'ER', 'PGR'])
    args = parser.parse_args()

    train_patients = [int(x) for x in args.train_patients.split(',')]
    val_patients = [int(x) for x in args.val_patients.split(',')]

    # Load Stage-1 checkpoint
    print(f"Loading Stage-1 checkpoint: {args.checkpoint}")
    model = RPNTrainer.load_from_checkpoint(
        args.checkpoint,
        region_grid=args.region_grid,
        region_dim=args.region_dim,
        gen_lr=args.gen_lr,
        disc_lr=args.disc_lr,
        strict=False,
    )

    # Load patient coordinate data from HDF5
    model.setup_patient_data(args.h5_path, train_patients + val_patients)

    # Freeze encoder
    model.freeze_encoder()

    # Data
    from torch.utils.data import DataLoader

    train_ds = ACROBATMultiStainDataset(
        h5_path=args.h5_path, stains=args.stains,
        patient_ids=train_patients,
        image_size=(512, 512), crop_size=512, augment=True,
    )
    val_ds = ACROBATMultiStainDataset(
        h5_path=args.h5_path, stains=args.stains,
        patient_ids=val_patients,
        image_size=(512, 512), crop_size=512, augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath='checkpoints/rpn',
        filename=f'rpn-g{args.region_grid}-{{epoch:02d}}-{{val/lpips:.3f}}',
        save_top_k=3, monitor='val/lpips', mode='min',
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_cb, LearningRateMonitor()],
        accelerator='gpu', devices=1, precision='16-mixed',
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
