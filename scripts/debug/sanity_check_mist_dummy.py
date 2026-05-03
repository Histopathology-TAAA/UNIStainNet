"""Sanity check: run one training_step and one validation_step with dummy tensors."""

import torch
import pytorch_lightning as pl

from src.models.trainer import UNIStainNetTrainer


def make_dummy_batch(batch_size=2, image_size=512, num_classes=5):
    he_rgb = torch.rand(batch_size, 3, image_size, image_size) * 2 - 1
    ihc_rgb = torch.rand(batch_size, 3, image_size, image_size) * 2 - 1
    he_h_map = torch.rand(batch_size, 1, image_size, image_size) * 2 - 1
    ihc_h_map = torch.rand(batch_size, 1, image_size, image_size) * 2 - 1
    labels = torch.randint(0, num_classes - 1, (batch_size,), dtype=torch.long)
    fnames = [f"sample_{i}.png" for i in range(batch_size)]
    return he_rgb, ihc_rgb, he_h_map, ihc_h_map, labels, fnames


class DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return make_dummy_batch(batch_size=1)


def collate_dummy(batch):
    he_rgb, ihc_rgb, he_h_map, ihc_h_map, labels, fnames = zip(*batch)
    he_rgb = torch.cat(he_rgb, dim=0)
    ihc_rgb = torch.cat(ihc_rgb, dim=0)
    he_h_map = torch.cat(he_h_map, dim=0)
    ihc_h_map = torch.cat(ihc_h_map, dim=0)
    labels = torch.cat(labels, dim=0)
    fnames = [f[0] if isinstance(f, list) else f for f in fnames]
    return he_rgb, ihc_rgb, he_h_map, ihc_h_map, labels, fnames


class DummyDataModule(pl.LightningDataModule):
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            DummyDataset(), batch_size=2, shuffle=False, collate_fn=collate_dummy
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            DummyDataset(), batch_size=2, shuffle=False, collate_fn=collate_dummy
        )


def main():
    torch.manual_seed(0)

    model = UNIStainNetTrainer(
        num_classes=5,
        null_class=4,
        class_dim=64,
        uni_dim=1024,
        ndf=64,
        input_skip=True,
        edge_encoder=False,
        uni_spatial_size=32,
        image_size=512,
        extract_uni_on_the_fly=True,
        uni_spatial_pool_size=32,
        # Disable extra losses/discriminators for a fast shape check
        adversarial_weight=0.0,
        uncond_disc_weight=0.0,
        crop_disc_weight=0.0,
        feat_match_weight=0.0,
        dab_intensity_weight=0.0,
        dab_contrast_weight=0.0,
        dab_sharpness_weight=0.0,
        gram_style_weight=0.0,
        edge_weight=0.0,
        he_edge_weight=0.0,
        bg_white_weight=0.0,
        patchnce_weight=0.0,
        l1_lowres_weight=1.0,
        l1_fullres_weight=1.0,
        lpips_weight=1.0,
        lpips_256_weight=0.5,
        lpips_fullres_weight=1.0,
    )

    # Avoid loading UNI model by returning random tokens
    def _fake_extract(_self, uni_sub_crops):
        b = uni_sub_crops.shape[0]
        device = uni_sub_crops.device
        return torch.randn(b, 32 * 32, 1024, device=device)

    model._extract_uni_from_sub_crops = _fake_extract.__get__(model, UNIStainNetTrainer)

    dm = DummyDataModule()

    # Inspect one batch shapes
    batch = next(iter(dm.train_dataloader()))
    he_rgb, ihc_rgb, he_h_map, ihc_h_map, labels, fnames = batch
    print("Batch shapes:")
    print("  he_rgb:", tuple(he_rgb.shape))
    print("  ihc_rgb:", tuple(ihc_rgb.shape))
    print("  he_h_map:", tuple(he_h_map.shape))
    print("  ihc_h_map:", tuple(ihc_h_map.shape))
    print("  labels:", tuple(labels.shape), labels.dtype)
    print("  fnames:", fnames)

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
        log_every_n_steps=1,
    )

    trainer.fit(model, dm)
    print("Sanity check complete.")


if __name__ == "__main__":
    main()
