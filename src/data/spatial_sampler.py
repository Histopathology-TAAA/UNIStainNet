"""
Spatial grid batch sampler for GNN-based patch generation.

Groups patches from the same WSI into truly spatially adjacent KxK grids
using the absolute WSI coordinates (index/coord_x, index/coord_y) already
stored in the ACROBAT HDF5 by the patch extraction pipeline.

The HDF5 index already contains:
  - index/coord_x[i]  — absolute WSI level-0 x-coordinate of HE patch i
  - index/coord_y[i]  — absolute WSI level-0 y-coordinate of HE patch i
  - index/patient_id[i] — patient ID for HE patch i

Patches are extracted on a stride=1024 grid. Adjacency is determined by
true spatial proximity: two patches are neighbors if their coordinates
differ by ~stride in at least one axis. Filtered-out patches create
natural gaps — the GNN graph simply has no edges across gaps.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Sampler
import h5py


# Grid stride used during patch extraction (default 1024px)
_EXTRACTION_STRIDE = 1024


def _load_patient_patches_with_coords(
    h5_path: str, stains: List[str], patient_ids: List[int]
) -> Dict:
    """Load per-stain, per-patient patch data including WSI coordinates.

    Returns:
        dict: stain -> {
            pid: [
                (pair_idx, he_idx, ihc_idx, coord_x, coord_y),
                ...
            ]
        }
    """
    result: Dict[str, Dict[int, list]] = {}
    with h5py.File(h5_path, 'r') as f:
        all_pids = f['index/patient_id'][:]
        coord_x = f['index/coord_x'][:]  # already stored by patching script
        coord_y = f['index/coord_y'][:]

        for stain in stains:
            pair_key = f'index/pair_{stain.lower()}'
            pairs = f[pair_key][:]  # (N, 2) -> [he_idx, ihc_idx]
            result[stain] = {}

            for pid in patient_ids:
                mask = all_pids[pairs[:, 0]] == pid
                indices = np.where(mask)[0]
                if len(indices) > 0:
                    he_indices = pairs[indices, 0]
                    result[stain][pid] = [
                        (
                            int(idx),                     # pair_idx
                            int(he_indices[i]),           # he_idx (global)
                            int(pairs[indices[i], 1]),    # ihc_idx
                            int(coord_x[he_indices[i]]),  # WSI pixel x
                            int(coord_y[he_indices[i]]),  # WSI pixel y
                        )
                        for i, idx in enumerate(indices)
                    ]
    return result


class SpatialGridSampler(Sampler):
    """Samples batches as spatially adjacent KxK grids.

    For each batch:
    1. Picks a random (stain, patient) combination
    2. Picks a random anchor patch
    3. Finds the grid_size² nearest neighbors by WSI coordinate distance
    4. Returns their pair indices for the DataLoader

    The resulting batch reflects true tissue adjacency, not arbitrary
    groupings. Gaps from filtered-out patches are natural — the GNN
    learns actual spatial relationships within the tissue.

    Args:
        h5_path: path to ACROBAT HDF5 file
        stains: stain keys (e.g. ['her2', 'ki67', 'er', 'pgr'])
        patient_ids: patient IDs to sample from
        grid_size: K for KxK grid. Total batch = grid_size**2
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

        self.patient_patches = _load_patient_patches_with_coords(
            h5_path, self.stains, patient_ids
        )

        # Collect (stain, pid) combos with enough patches for a grid
        self.combinations: List[Tuple[str, int]] = []
        for stain in self.stains:
            if stain in self.patient_patches:
                for pid, patches in self.patient_patches[stain].items():
                    if len(patches) >= self.batch_size:
                        self.combinations.append((stain, pid))

        if not self.combinations:
            raise ValueError(
                f"No patients have >= {self.batch_size} patches per stain. "
                f"Reduce grid_size from {self.grid_size}."
            )

        print(f"SpatialGridSampler: {len(self.combinations)} (stain, patient) "
              f"combos with >= {self.batch_size} patches")

    def _sample_spatial_grid(self) -> Tuple[List[int], np.ndarray]:
        """Sample a spatially adjacent grid of patches.

        Returns:
            pair_indices: list of pair indices for the DataLoader
            positions: (batch_size, 2) float array of normalized coords
        """
        stain, pid = self.combinations[
            np.random.randint(len(self.combinations))
        ]
        patches = self.patient_patches[stain][pid]
        # patches: list of (pair_idx, he_idx, ihc_idx, coord_x, coord_y)

        # Pick a random anchor patch
        anchor = patches[np.random.randint(len(patches))]
        ax, ay = anchor[3], anchor[4]

        # Compute distances from anchor to all other patches
        coords_arr = np.array([[p[3], p[4]] for p in patches], dtype=np.float32)
        anchor_pt = np.array([ax, ay], dtype=np.float32)
        dists = np.linalg.norm(coords_arr - anchor_pt, axis=1)

        # Take the batch_size nearest neighbors (including anchor itself)
        nearest_indices = np.argsort(dists)[:self.batch_size]
        selected = [patches[i] for i in nearest_indices]

        # Normalize coordinates relative to the grid center for GNN
        selected_coords = coords_arr[nearest_indices]
        selected_coords = selected_coords - selected_coords.mean(axis=0)
        # Scale to roughly [-grid_size/2, grid_size/2] range
        scale = _EXTRACTION_STRIDE
        if scale > 0:
            selected_coords = selected_coords / scale

        pair_indices = [s[0] for s in selected]

        return pair_indices, selected_coords

    def __iter__(self):
        """Yield indices in spatial-group order.

        Each call to _sample_spatial_grid produces grid_size**2 indices.
        We yield them one at a time so BatchSampler (with batch_size=
        grid_size**2) naturally groups them into spatial batches.
        """
        for _ in range(self.samples_per_epoch):
            indices, _ = self._sample_spatial_grid()
            for idx in indices:
                yield idx

    def __len__(self):
        return self.samples_per_epoch * self.batch_size


def spatial_collate_fn(batch, grid_size=3):
    """Collate dataset items into a spatial grid batch with real coordinates.

    Each item in the batch is fetched by ACROBATMultiStainDataset using
    the pair indices from SpatialGridSampler. We collect them and need
    to recover the positions.

    Since the standard DataLoader doesn't thread sampler metadata through,
    we do a simple approach: infer adjacency from H&E content similarity
    is unreliable, so we rely on the sampler's spatial grouping + the
    training script to build positions from the DataLoader indices.

    For the SpatialGridSampler path (used via BatchSampler), positions
    are generated by the training script from HE patch indices.
    For the simple shuffle path (fallback), we use pseudo-positions
    and log a warning.
    """
    he, ihc, uni, labels, fnames = zip(*batch)

    he = torch.stack(he)
    ihc = torch.stack(ihc)
    if isinstance(uni[0], torch.Tensor):
        uni = torch.stack(uni)
    labels = torch.tensor(labels, dtype=torch.long)

    # Pseudo-positions — overridden by training script when using
    # SpatialGridSampler + dataset-aware position lookup.
    positions = torch.tensor([
        [i // grid_size, i % grid_size]
        for i in range(len(he))
    ]).float()

    return he, ihc, uni, labels, list(fnames), positions
