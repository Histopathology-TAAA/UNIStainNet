"""
Multi-stain dataset for training a single model on all MIST IHC stains.

Combines HER2, Ki67, ER, PR into one dataset, returning a stain label (0-3)
instead of a class label. Reuses the same crop + UNI sub-crop pipeline
from CropPairedDataset.

Stain label mapping:
    0 = HER2, 1 = Ki67, 2 = ER, 3 = PR, 4 = null (CFG dropout)

Batch tuple (Case A/B support):
    (he_tensor, ihc_tensor, he_h_tensor, ihc_h_tensor, uni_sub_crops, label, filename)
    - he_h_tensor:  [1, H, W] in [-1, 1]  — H-channel of H&E  (Case B edge source)
    - ihc_h_tensor: [1, H, W] in [-1, 1]  — H-channel of IHC  (Case A edge source)
    H-channel directories expected alongside the RGB dirs, e.g.:
        trainA/  trainA-H/  trainB/  trainB-H/
        valA/    valA-H/    valB/    valB-H/
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pytorch_lightning as pl

from src.data.bci_dataset import CropPairedDataset


STAIN_TO_LABEL = {'HER2': 0, 'Ki67': 1, 'ER': 2, 'PR': 3}
LABEL_TO_STAIN = {v: k for k, v in STAIN_TO_LABEL.items()}


class MISTMultiStainCropDataset(CropPairedDataset):
    """Multi-stain MIST dataset with random 512 crops from native 1024x1024.

    Loads all 4 MIST stains into a single dataset. Each sample returns a
    stain label (0-3) as the conditioning signal, reusing the class embedding
    slot in the generator.
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
        split_he_h = split_he + '-H'
        split_ihc_h = split_ihc + '-H'
        valid_exts = ('.jpg', '.jpeg', '.png')

        for stain in stains:
            if stain not in STAIN_TO_LABEL:
                raise ValueError(f"Unknown stain: {stain}. Must be one of {list(STAIN_TO_LABEL.keys())}")

            stain_label = STAIN_TO_LABEL[stain]
            he_dir = self.base_dir / stain / split_he
            ihc_dir = self.base_dir / stain / split_ihc
            he_h_dir = self.base_dir / stain / split_he_h
            ihc_h_dir = self.base_dir / stain / split_ihc_h

            if not he_dir.exists():
                raise FileNotFoundError(f"H&E directory not found: {he_dir}")
            if not ihc_dir.exists():
                raise FileNotFoundError(f"IHC directory not found: {ihc_dir}")
            if not he_h_dir.exists():
                raise FileNotFoundError(f"H&E H-channel directory not found: {he_h_dir}")
            if not ihc_h_dir.exists():
                raise FileNotFoundError(f"IHC H-channel directory not found: {ihc_h_dir}")

            he_files = sorted([f for f in os.listdir(he_dir)
                               if f.lower().endswith(valid_exts)])
            ihc_files = sorted([f for f in os.listdir(ihc_dir)
                                if f.lower().endswith(valid_exts)])
            he_h_files = sorted([f for f in os.listdir(he_h_dir)
                                 if f.lower().endswith(valid_exts)])
            ihc_h_files = sorted([f for f in os.listdir(ihc_h_dir)
                                  if f.lower().endswith(valid_exts)])

            # Match by stem (H&E may be .jpg, IHC may be .png)
            he_stems = {Path(f).stem: f for f in he_files}
            ihc_stems = {Path(f).stem: f for f in ihc_files}
            he_h_stems = {Path(f).stem: f for f in he_h_files}
            ihc_h_stems = {Path(f).stem: f for f in ihc_h_files}
            common = sorted(
                set(he_stems.keys()) & set(ihc_stems.keys())
                & set(he_h_stems.keys()) & set(ihc_h_stems.keys())
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
        # H-channel images are single-channel grayscale PNGs
        he_h_img = Image.open(he_h_path).convert('L')
        ihc_h_img = Image.open(ihc_h_path).convert('L')
        return self._process_pair_with_h(
            he_img, ihc_img, he_h_img, ihc_h_img, stain_label, he_path.name
        )

    def _process_pair_with_h(self, he_img, ihc_img, he_h_img, ihc_h_img, label, filename):
        """Extends CropPairedDataset._process_pair to also return H-channel tensors.

        Returns:
            he_tensor:   [3, H, W] in [-1, 1]
            ihc_tensor:  [3, H, W] in [-1, 1]
            he_h_tensor: [1, H, W] in [-1, 1]  — H-channel of H&E
            ihc_h_tensor:[1, H, W] in [-1, 1]  — H-channel of IHC
            uni_sub_crops: [16, 3, 224, 224]
            label: int
            filename: str
        """
        from src.data.bci_dataset import CropPairedDataset
        import random
        import torchvision.transforms as T
        import torchvision.transforms.functional as TF

        # --- Same random crop position for all four images ---
        w, h = he_img.size
        if w > self.crop_size and h > self.crop_size:
            left = random.randint(0, w - self.crop_size)
            top = random.randint(0, h - self.crop_size)
            he_img = he_img.crop((left, top, left + self.crop_size, top + self.crop_size))
            ihc_img = ihc_img.crop((left, top, left + self.crop_size, top + self.crop_size))
            he_h_img = he_h_img.crop((left, top, left + self.crop_size, top + self.crop_size))
            ihc_h_img = ihc_h_img.crop((left, top, left + self.crop_size, top + self.crop_size))

        # --- Paired spatial augmentations (same for all) ---
        if self.augment:
            if random.random() > 0.5:
                he_img = TF.hflip(he_img); ihc_img = TF.hflip(ihc_img)
                he_h_img = TF.hflip(he_h_img); ihc_h_img = TF.hflip(ihc_h_img)
            if random.random() > 0.5:
                he_img = TF.vflip(he_img); ihc_img = TF.vflip(ihc_img)
                he_h_img = TF.vflip(he_h_img); ihc_h_img = TF.vflip(ihc_h_img)
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])
                he_img = TF.rotate(he_img, k * 90); ihc_img = TF.rotate(ihc_img, k * 90)
                he_h_img = TF.rotate(he_h_img, k * 90); ihc_h_img = TF.rotate(ihc_h_img, k * 90)
            # H&E color jitter (H-channel not color-jittered — structural channel)
            if random.random() > 0.5:
                he_img = TF.adjust_brightness(he_img, random.uniform(0.9, 1.1))
            if random.random() > 0.5:
                he_img = TF.adjust_contrast(he_img, random.uniform(0.9, 1.1))
            if random.random() > 0.5:
                he_img = TF.adjust_saturation(he_img, random.uniform(0.9, 1.1))

        # --- UNI sub-crops from augmented H&E (3-ch) ---
        uni_sub_crops = self._prepare_uni_sub_crops(he_img)

        # --- Tensorize RGB images [-1, 1] ---
        he_tensor = TF.normalize(TF.to_tensor(he_img), [0.5]*3, [0.5]*3)
        ihc_tensor = TF.normalize(TF.to_tensor(ihc_img), [0.5]*3, [0.5]*3)

        # --- Tensorize H-channel images [-1, 1] (single channel) ---
        # TF.to_tensor on 'L' mode gives [1, H, W] in [0, 1]
        he_h_tensor = TF.normalize(TF.to_tensor(he_h_img), [0.5], [0.5])
        ihc_h_tensor = TF.normalize(TF.to_tensor(ihc_h_img), [0.5], [0.5])

        return he_tensor, ihc_tensor, he_h_tensor, ihc_h_tensor, uni_sub_crops, label, filename


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
