"""
Multi-stain dataset for training a single model on all MIST IHC stains.

Combines HER2, Ki67, ER, PR into one dataset, returning a stain label (0-3)
instead of a class label. Reuses the same crop + UNI sub-crop pipeline
from CropPairedDataset.

Stain label mapping:
    0 = HER2, 1 = Ki67, 2 = ER, 3 = PR, 4 = null (CFG dropout)
"""

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pytorch_lightning as pl
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from src.data.bci_dataset import CropPairedDataset


STAIN_TO_LABEL = {'HER2': 0, 'Ki67': 1, 'ER': 2, 'PR': 3}
LABEL_TO_STAIN = {v: k for k, v in STAIN_TO_LABEL.items()}


class MISTMultiStainCropDataset(CropPairedDataset):
    """Multi-stain MIST dataset with paired RGB + H-map inputs.

    Expects per-stain folders with the following structure:
        trainA/, trainA-H/, trainB/, trainB-H/, valA/, valA-H/, valB/, valB-H/
    Returns: (he_rgb, ihc_rgb, he_h_map, ihc_h_map, stain_label, filename)
    """

    def __init__(
        self,
        base_dir: str,
        stains: List[str],
        split: str = 'train',
        image_size: Tuple[int, int] = (512, 512),
        crop_size: int = 512,
        augment: bool = False,
        null_class: int = 4,
    ):
        super().__init__(
            he_dir='.',  # placeholder, we override __getitem__
            ihc_dir='.',
            image_size=image_size,
            crop_size=crop_size,
            augment=augment,
            null_class=null_class,
        )

        self.base_dir = Path(base_dir)
        self.samples = []  # (he_path, ihc_path, he_h_path, ihc_h_path, stain_label)

        split_he = 'trainA' if split == 'train' else 'valA'
        split_ihc = 'trainB' if split == 'train' else 'valB'
        valid_exts = ('.jpg', '.jpeg', '.png')

        for stain in stains:
            if stain not in STAIN_TO_LABEL:
                raise ValueError(f"Unknown stain: {stain}. Must be one of {list(STAIN_TO_LABEL.keys())}")

            stain_label = STAIN_TO_LABEL[stain]
            he_dir = self.base_dir / stain / split_he
            ihc_dir = self.base_dir / stain / split_ihc
            he_h_dir = self.base_dir / stain / f"{split_he}-H"
            ihc_h_dir = self.base_dir / stain / f"{split_ihc}-H"

            if not he_dir.exists():
                raise FileNotFoundError(f"H&E directory not found: {he_dir}")
            if not ihc_dir.exists():
                raise FileNotFoundError(f"IHC directory not found: {ihc_dir}")
            if not he_h_dir.exists():
                raise FileNotFoundError(f"H&E H-map directory not found: {he_h_dir}")
            if not ihc_h_dir.exists():
                raise FileNotFoundError(f"IHC H-map directory not found: {ihc_h_dir}")

            he_files = sorted([f for f in os.listdir(he_dir)
                               if f.lower().endswith(valid_exts)])
            ihc_files = sorted([f for f in os.listdir(ihc_dir)
                                if f.lower().endswith(valid_exts)])

            # Match by stem (RGB and -H may have different extensions)
            he_stems = {Path(f).stem: f for f in he_files}
            ihc_stems = {Path(f).stem: f for f in ihc_files}

            he_h_files = sorted([f for f in os.listdir(he_h_dir)
                                 if f.lower().endswith(valid_exts)])
            ihc_h_files = sorted([f for f in os.listdir(ihc_h_dir)
                                  if f.lower().endswith(valid_exts)])
            he_h_stems = {Path(f).stem: f for f in he_h_files}
            ihc_h_stems = {Path(f).stem: f for f in ihc_h_files}

            common = sorted(
                set(he_stems.keys())
                & set(ihc_stems.keys())
                & set(he_h_stems.keys())
                & set(ihc_h_stems.keys())
            )

            for stem in common:
                self.samples.append((
                    he_dir / he_stems[stem],
                    ihc_dir / ihc_stems[stem],
                    he_h_dir / he_h_stems[stem],
                    ihc_h_dir / ihc_h_stems[stem],
                    stain_label,
                ))

            print(f"  {stain} ({split}): {len(common)} pairs")

        # Per-stain counts for logging
        from collections import Counter
        dist = Counter(s[4] for s in self.samples)
        stain_counts = {LABEL_TO_STAIN[k]: v for k, v in sorted(dist.items())}
        print(f"Multi-Stain Crop Dataset ({split}): {len(self.samples)} total | {stain_counts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        he_path, ihc_path, he_h_path, ihc_h_path, stain_label = self.samples[idx]
        he_img = Image.open(he_path).convert('RGB')
        ihc_img = Image.open(ihc_path).convert('RGB')
        he_h_img = Image.open(he_h_path).convert('L')
        ihc_h_img = Image.open(ihc_h_path).convert('L')
        return self._process_quad(
            he_img,
            ihc_img,
            he_h_img,
            ihc_h_img,
            stain_label,
            he_path.name,
        )

    def _random_crop_quad(self, he_img, ihc_img, he_h_img, ihc_h_img):
        """Take the same random crop from all four images."""
        w, h = he_img.size
        if w < self.crop_size or h < self.crop_size:
            raise ValueError(
                f"Image size {w}x{h} smaller than crop size {self.crop_size}"
            )
        if w == self.crop_size and h == self.crop_size:
            return he_img, ihc_img, he_h_img, ihc_h_img

        left = random.randint(0, w - self.crop_size)
        top = random.randint(0, h - self.crop_size)
        box = (left, top, left + self.crop_size, top + self.crop_size)
        return (
            he_img.crop(box),
            ihc_img.crop(box),
            he_h_img.crop(box),
            ihc_h_img.crop(box),
        )

    def _apply_paired_augmentations_quad(self, he_img, ihc_img, he_h_img, ihc_h_img):
        """Apply identical spatial transforms to all four images."""
        if random.random() > 0.5:
            he_img = TF.hflip(he_img)
            ihc_img = TF.hflip(ihc_img)
            he_h_img = TF.hflip(he_h_img)
            ihc_h_img = TF.hflip(ihc_h_img)
        if random.random() > 0.5:
            he_img = TF.vflip(he_img)
            ihc_img = TF.vflip(ihc_img)
            he_h_img = TF.vflip(he_h_img)
            ihc_h_img = TF.vflip(ihc_h_img)
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            angle = k * 90
            he_img = TF.rotate(he_img, angle)
            ihc_img = TF.rotate(ihc_img, angle)
            he_h_img = TF.rotate(he_h_img, angle)
            ihc_h_img = TF.rotate(ihc_h_img, angle)
        if random.random() > 0.7:
            angle = random.uniform(-15, 15)
            translate = [random.uniform(-0.05, 0.05) * self.image_size[1],
                         random.uniform(-0.05, 0.05) * self.image_size[0]]
            scale = random.uniform(0.9, 1.1)
            he_img = TF.affine(he_img, angle, translate, scale, shear=0,
                               interpolation=T.InterpolationMode.BILINEAR)
            ihc_img = TF.affine(ihc_img, angle, translate, scale, shear=0,
                                interpolation=T.InterpolationMode.BILINEAR)
            he_h_img = TF.affine(he_h_img, angle, translate, scale, shear=0,
                                 interpolation=T.InterpolationMode.BILINEAR)
            ihc_h_img = TF.affine(ihc_h_img, angle, translate, scale, shear=0,
                                  interpolation=T.InterpolationMode.BILINEAR)
        return he_img, ihc_img, he_h_img, ihc_h_img

    def _process_quad(self, he_img, ihc_img, he_h_img, ihc_h_img, label, filename):
        """Common processing for RGB + H-map pairs."""
        he_crop, ihc_crop, he_h_crop, ihc_h_crop = self._random_crop_quad(
            he_img, ihc_img, he_h_img, ihc_h_img
        )

        if self.augment:
            he_crop, ihc_crop, he_h_crop, ihc_h_crop = self._apply_paired_augmentations_quad(
                he_crop, ihc_crop, he_h_crop, ihc_h_crop
            )
            he_aug = self._apply_he_color_augmentation(he_crop)
        else:
            he_aug = he_crop

        he_tensor = TF.normalize(TF.to_tensor(he_aug), [0.5] * 3, [0.5] * 3)
        ihc_tensor = TF.normalize(TF.to_tensor(ihc_crop), [0.5] * 3, [0.5] * 3)
        he_h_tensor = TF.normalize(TF.to_tensor(he_h_crop), [0.5], [0.5])
        ihc_h_tensor = TF.normalize(TF.to_tensor(ihc_h_crop), [0.5], [0.5])

        return he_tensor, ihc_tensor, he_h_tensor, ihc_h_tensor, label, filename


class MISTMultiStainCropDataModule(pl.LightningDataModule):
    """Lightning DataModule for multi-stain MIST training."""

    def __init__(
        self,
        base_dir: str,
        stains: Optional[List[str]] = None,
        batch_size: int = 4,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (512, 512),
        crop_size: int = 512,
        null_class: int = 4,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.stains = stains or ['HER2', 'Ki67', 'ER', 'PR']
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.crop_size = crop_size
        self.null_class = null_class

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = MISTMultiStainCropDataset(
                base_dir=self.base_dir,
                stains=self.stains,
                split='train',
                image_size=self.image_size,
                crop_size=self.crop_size,
                augment=True,
                null_class=self.null_class,
            )
        if stage in ('fit', 'validate', 'test') or stage is None:
            self.val_dataset = MISTMultiStainCropDataset(
                base_dir=self.base_dir,
                stains=self.stains,
                split='val',
                image_size=self.image_size,
                crop_size=self.crop_size,
                augment=False,
                null_class=self.null_class,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return self.val_dataloader()
