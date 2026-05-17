"""
Spatial grid batch sampler for GNN-based patch generation.

Groups patches from the same WSI (same patient) into KxK grids for
GNN message passing. When spatial coordinates are unavailable in the
HDF5, uses index-based pseudo-positions (nearby indices in a raster
scan are likely spatially adjacent).

Intended for use with ACROBATMultiStainDataset during Stage-2 training
where each batch must form a coherent spatial graph.
"""

from typing import Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import Sampler
import h5py


def _load_patient_patches(h5_path: str, stains: List[str],
                          patient_ids: List[int]) -> Dict:
    """Load per-stain, per-patient patch groupings from HDF5 index.

    Returns:
        dict: stain -> {pid: [(he_idx, ihc_idx), ...]}
    """
    result: Dict[str, Dict[int, list]] = {}
    with h5py.File(h5_path, 'r') as f:
        all_pids = f['index/patient_id'][:]
        for stain in stains:
            pair_key = f'index/pair_{stain.lower()}'
            pairs = f[pair_key][:]  # (N, 2)
            result[stain] = {}
            for pid in patient_ids:
                mask = all_pids[pairs[:, 0]] == pid
                indices = np.where(mask)[0]
                if len(indices) > 0:
                    he_idx = pairs[indices, 0]
                    ihc_idx = pairs[indices, 1]
                    result[stain][pid] = list(zip(
                        indices.astype(int).tolist(),
                        he_idx.astype(int).tolist(),
                        ihc_idx.astype(int).tolist(),
                    ))
    return result


def _try_load_coords(h5_path: str) -> Optional[np.ndarray]:
    """Try to load explicit patch coordinates from HDF5.

    Returns (N, 2) array of (row, col) or None if not available.
    """
    with h5py.File(h5_path, 'r') as f:
        for key in ['index/he_coords', 'index/coords', 'index/positions']:
            if key in f:
                return f[key][:]
        # Try he group
        for key in ['he/coords', 'he/positions']:
            if key in f:
                return f[key][:]
    return None


class SpatialGridSampler(Sampler):
    """Samples batches as KxK spatial grids from the same WSI/patient.

    Each batch forms a graph where patches from the same patient can
    exchange information via GNN message passing.

    Args:
        h5_path: path to ACROBAT HDF5 file
        stains: list of stain keys (e.g. ['her2', 'ki67', 'er', 'pgr'])
        patient_ids: patient IDs to sample from
        grid_size: K for KxK grid (default 3). Total batch = grid_size**2
        samples_per_epoch: number of grid samples per epoch
    """

    def __init__(self, h5_path: str, stains: List[str],
                 patient_ids: List[int], grid_size: int = 3,
                 samples_per_epoch: int = 2000):
        self.h5_path = h5_path
        self.stains = [s.lower() for s in stains]
        self.grid_size = grid_size
        self.samples_per_epoch = samples_per_epoch
        self.batch_size = grid_size ** 2

        # Build per-stain, per-patient patch index
        self.patient_patches = _load_patient_patches(
            h5_path, self.stains, patient_ids
        )

        # Try to load explicit coordinates
        self.coords = _try_load_coords(h5_path)

        # Collect all valid (stain, patient) combinations
        self.combinations = []
        for stain in self.stains:
            if stain in self.patient_patches:
                for pid, patches in self.patient_patches[stain].items():
                    if len(patches) >= self.batch_size:
                        self.combinations.append((stain, pid, len(patches)))

        if not self.combinations:
            raise ValueError(
                f"No patients have >= {self.batch_size} patches. "
                f"Reduce grid_size from {grid_size}."
            )

        print(f"SpatialGridSampler: {len(self.combinations)} (stain, patient) "
              f"combos with >= {self.batch_size} patches")

    def _sample_grid_indices(self) -> List[int]:
        """Sample grid_size**2 patch indices from a random patient."""
        stain, pid, n_patches = self.combinations[
            np.random.randint(len(self.combinations))
        ]
        patches = self.patient_patches[stain][pid]

        if len(patches) == self.batch_size:
            selected = patches
        else:
            indices = np.random.choice(len(patches), self.batch_size,
                                       replace=False)
            selected = [patches[i] for i in indices]

        # Return the pair indices (first element of each tuple)
        return [s[0] for s in selected]

    def get_positions(self, he_indices: List[int]) -> torch.Tensor:
        """Get grid positions for a set of HE patch indices.

        Uses explicit coordinates if available, otherwise falls back to
        index-based pseudo-positions (raster-scan order assumption).
        """
        if self.coords is not None:
            coords = self.coords[he_indices]
            # Normalize to grid
            coords = coords - coords.min(axis=0)
            scale = max(coords.max(axis=0))
            if scale > 0:
                coords = coords / scale * (self.grid_size - 1)
            return torch.from_numpy(coords).float()
        else:
            # Pseudo-positions: assign to grid in raster order
            positions = torch.tensor([
                [i // self.grid_size, i % self.grid_size]
                for i in range(len(he_indices))
            ]).float()
            return positions

    def __iter__(self):
        for _ in range(self.samples_per_epoch):
            indices = self._sample_grid_indices()
            yield indices

    def __len__(self):
        return self.samples_per_epoch


def spatial_collate_fn(batch, grid_size=3):
    """Collate dataset items into a spatial grid batch.

    Args:
        batch: list of (he, ihc, uni_or_crops, label, fname) tuples
        grid_size: K for KxK grid

    Returns:
        he: [K*K, 3, H, W]
        ihc: [K*K, 3, H, W]
        uni_or_crops: [K*K, ...]
        labels: [K*K]
        fnames: list of str
        positions: [K*K, 2] grid positions for GNN
    """
    he, ihc, uni, labels, fnames = zip(*batch)

    he = torch.stack(he)
    ihc = torch.stack(ihc)
    if isinstance(uni[0], torch.Tensor):
        uni = torch.stack(uni)
    labels = torch.tensor(labels, dtype=torch.long)

    # Generate grid positions
    side = grid_size
    positions = torch.tensor([
        [i // side, i % side] for i in range(len(he))
    ]).float()

    return he, ihc, uni, labels, list(fnames), positions
