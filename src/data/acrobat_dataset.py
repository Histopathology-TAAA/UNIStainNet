"""
ACROBAT dataset: H&E–IHC patch pairs read from a pre-registered HDF5 file.

All IHC slides are warped into H&E space by VALIS, so patches at the same
(x, y) coordinates are perfectly aligned. Pair arrays in the HDF5 index group
tell us which HE patch goes with which IHC patch — pairs are sparse, so do not
assume a 1:1 layout.

The per-worker HDF5 handle pattern avoids pickling file handles across
DataLoader workers.
"""

from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pytorch_lightning as pl

from src.data.bci_dataset import CropPairedDataset

# Stain label mapping compatible with UNIStainNetTrainer.
# null_class=4 is reserved for CFG dropout (same convention as MIST).
STAIN_TO_LABEL = {'HER2': 0, 'KI67': 1, 'ER': 2, 'PGR': 3}
LABEL_TO_STAIN = {v: k for k, v in STAIN_TO_LABEL.items()}

# ---------------------------------------------------------------------------
# Per-worker HDF5 handle (each DataLoader worker is a separate process)
# ---------------------------------------------------------------------------

_worker_h5: Optional[h5py.File] = None


class _H5WorkerInitializer:
    """Picklable callable that opens a per-worker HDF5 handle.

    Using a class (instead of a closure) ensures compatibility with the
    ``spawn`` multiprocessing start method on Windows.
    """
    def __init__(self, h5_path: str):
        self.h5_path = h5_path

    def __call__(self, worker_id: int):
        global _worker_h5
        _worker_h5 = h5py.File(self.h5_path, 'r')


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ACROBATMultiStainDataset(CropPairedDataset):
    """Multi-stain ACROBAT dataset reading pre-extracted 1024×1024 patches from HDF5.

    Inherits crop, paired-augmentation, and UNI-sub-crop logic from
    CropPairedDataset.  Only the image-loading path changes: numpy arrays
    from HDF5 → PIL → the same `_process_pair` pipeline.
    """

    def __init__(
        self,
        h5_path: str,
        stains: List[str],
        patient_ids: List[int],
        image_size: Tuple[int, int] = (512, 512),
        crop_size: int = 512,
        augment: bool = False,
        null_class: int = 4,
    ):
        # CropPairedDataset expects directory args — we pass placeholders
        # because file I/O is handled entirely through HDF5.
        super().__init__(
            he_dir='.',
            ihc_dir='.',
            image_size=image_size,
            crop_size=crop_size,
            augment=augment,
            null_class=null_class,
        )

        self.h5_path = h5_path
        self.stains = stains
        self.patient_ids = set(patient_ids)

        # Pre-load lightweight index arrays; close the file afterwards.
        with h5py.File(h5_path, 'r') as f:
            all_pids = f['index/patient_id'][:]                     # (N_he,) int32

            patient_mask = np.isin(all_pids, list(self.patient_ids))

            self.pairs: dict[str, torch.Tensor] = {}
            for stain in stains:
                pair_key = f'pair_{stain.lower()}'
                pairs = f[f'index/{pair_key}'][:]                  # (N_stain, 2)  [he_idx, ihc_idx]
                keep = patient_mask[pairs[:, 0]]
                self.pairs[stain] = torch.from_numpy(pairs[keep])

        # Flat index: every (stain, pair_idx) is a sample
        self.samples: List[Tuple[str, int]] = []
        for stain in stains:
            n = len(self.pairs[stain])
            for i in range(n):
                self.samples.append((stain, i))

        # Logging
        from collections import Counter
        sc = Counter(s[0] for s in self.samples)
        print(f"ACROBAT Dataset (patients {sorted(self.patient_ids)}): "
              f"{len(self.samples)} total pairs | {dict(sc)}")

    # ------------------------------------------------------------------
    # HDF5 handle helpers
    # ------------------------------------------------------------------

    def _get_h5(self) -> h5py.File:
        """Return a process-local HDF5 handle, opening one if needed."""
        global _worker_h5
        if _worker_h5 is None:
            _worker_h5 = h5py.File(self.h5_path, 'r')
        return _worker_h5

    @staticmethod
    def worker_init_fn(h5_path: str):
        """Convenience wrapper so callers don't import the module-level helper."""
        return _H5WorkerInitializer(h5_path)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        stain, pair_idx = self.samples[idx]
        he_idx, ihc_idx = self.pairs[stain][pair_idx]
        he_idx, ihc_idx = int(he_idx), int(ihc_idx)

        h5 = self._get_h5()

        # Read uint8 arrays from HDF5 and wrap as PIL images.
        he_arr = h5['he'][he_idx]                           # (1024, 1024, 3)
        ihc_arr = h5[stain.lower()][ihc_idx]                # (1024, 1024, 3)

        he_img = Image.fromarray(he_arr, 'RGB')
        ihc_img = Image.fromarray(ihc_arr, 'RGB')

        stain_label = STAIN_TO_LABEL[stain]
        filename = f"p{h5['index/patient_id'][he_idx]}_{stain}_{he_idx}_{ihc_idx}"

        return self._process_pair(he_img, ihc_img, stain_label, filename)


# ---------------------------------------------------------------------------
# Lightning DataModule
# ---------------------------------------------------------------------------

class ACROBATDataModule(pl.LightningDataModule):
    """Lightning DataModule that splits ACROBAT data by patient ID.

    Train and validation sets are defined by disjoint patient ID lists so
    that patches from the same patient never leak across splits.
    """

    def __init__(
        self,
        h5_path: str,
        train_patients: List[int],
        val_patients: List[int],
        stains: Optional[List[str]] = None,
        batch_size: int = 4,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (512, 512),
        crop_size: int = 512,
        null_class: int = 4,
    ):
        super().__init__()
        self.h5_path = h5_path
        self.train_patients = train_patients
        self.val_patients = val_patients
        self.stains = stains or ['HER2', 'KI67', 'ER', 'PGR']
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.crop_size = crop_size
        self.null_class = null_class

        # Pre-build worker init so it can be passed to every DataLoader.
        self._worker_init_fn = _H5WorkerInitializer(h5_path)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ACROBATMultiStainDataset(
                h5_path=self.h5_path,
                stains=self.stains,
                patient_ids=self.train_patients,
                image_size=self.image_size,
                crop_size=self.crop_size,
                augment=True,
                null_class=self.null_class,
            )
        if stage in ('fit', 'validate', 'test') or stage is None:
            self.val_dataset = ACROBATMultiStainDataset(
                h5_path=self.h5_path,
                stains=self.stains,
                patient_ids=self.val_patients,
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
            worker_init_fn=self._worker_init_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=self._worker_init_fn,
        )

    def test_dataloader(self):
        return self.val_dataloader()
