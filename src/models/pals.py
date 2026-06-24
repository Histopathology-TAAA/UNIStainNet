"""
Multi-Level Protein Awareness (MLPA) Loss — PGVMS (TMI 2026).

Reference: Chen et al., "Pathological Semantics-Preserving Learning for
H&E-to-IHC Virtual Staining" (MICCAI 2024) and its journal extension
PGVMS (TMI 2026).

The MLPA loss replaces naive pixel-level DAB losses with a biologically-
motivated three-level protein expression comparison:

    Level 1 — Global:       Per-image mean Focal Optical Density (FOD)
    Level 2 — Spatial:      4×4 block sums of FOD (where is the protein?)
    Level 3 — Distribution: 20-bin histogram of FOD (what shape is the
                             expression distribution?)

Adapted from PSPStain/PGVMS for use with UNIStainNet's PyTorch Lightning
pipeline.  All hard-coded .cuda() calls have been replaced with device-
agnostic tensor creation so the module follows whatever device the input
tensors live on.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPA_LOSS(nn.Module):
    """Multi-Level Protein Awareness loss.

    Computes Focal Optical Density (FOD) maps from generated and real IHC
    images via Beer-Lambert colour deconvolution, then compares them at
    three spatial scales with adaptive weighting.

    Parameters
    ----------
    alpha : float
        Focal exponent.  Higher values amplify strongly-stained regions.
        PGVMS default: 1.8.
    thresh_fod : float
        FOD values below this threshold are zeroed (background suppression).
    thresh_mask : float
        FOD values above this threshold are considered "protein-expressing"
        and contribute to the pseudo-mask used by CTPC/PCLS.
    num_blocks : int
        Number of spatial grid cells (√num_blocks × √num_blocks) for the
        regional block loss.
    num_bins : int
        Number of histogram bins for the distribution loss.
    tolerance : float
        If the global FOD difference between fake and real is within
        ±tolerance of the real FOD, only the histogram loss is applied.
        Otherwise both global and histogram losses are active.
    """

    def __init__(
        self,
        alpha: float = 1.8,
        thresh_fod: float = 0.15,
        thresh_mask: float = 0.68,
        num_blocks: int = 16,
        num_bins: int = 20,
        tolerance: float = 0.4,
    ):
        super(MLPA_LOSS, self).__init__()

        # ── H-DAB stain matrix (Ruifrok & Johnston) ────────────────
        # Rows: DAB (brown), Hematoxylin (blue).
        # Stored as buffers so they move to the correct device automatically.
        rgb_from_hed = torch.tensor([
            [0.65, 0.70, 0.29],
            [0.07, 0.99, 0.11],
            [0.27, 0.57, 0.78],
        ])
        hed_from_rgb = torch.linalg.inv(rgb_from_hed)
        self.register_buffer('rgb_from_hed', rgb_from_hed.float())
        self.register_buffer('hed_from_rgb', hed_from_rgb.float())

        # Luminance coefficients for RGB → grayscale
        coeffs = torch.tensor([0.2125, 0.7154, 0.0721]).view(3, 1)
        self.register_buffer('coeffs', coeffs.float())

        # ── Focal OD parameters ────────────────────────────────────
        self.alpha = alpha
        self.thresh_fod = thresh_fod
        self.thresh_mask = thresh_mask
        self.num_blocks = num_blocks
        self.num_bins = num_bins
        self.tolerance = tolerance

        # Calibration constant to avoid log(0)
        self.register_buffer(
            'adjust_calibration',
            torch.tensor(10.0 ** (-(math.e) ** (1.0 / self.alpha))),
        )

        # log(1e-6) pre-computed for the Beer-Lambert step
        self.register_buffer(
            'log_adjust',
            torch.log(torch.tensor(1e-6)),
        )

        # ── Losses ─────────────────────────────────────────────────
        self.mse_loss = nn.MSELoss()
        self.mse_loss_2 = nn.MSELoss(reduction='none')

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, inputs, targets):
        """Compute MLPA loss between generated (inputs) and real (targets) IHC.

        Args:
            inputs:  [B, 3, H, W] generated IHC in [-1, 1]
            targets: [B, 3, H, W] real IHC in [-1, 1]

        Returns:
            loss_mlpa:   scalar loss
            input_mask:  [B, H, W] pseudo-mask for generated image
            target_mask: [B, H, W] pseudo-mask for real image
        """
        # Channel-last for colour deconvolution
        inputs_reshape = inputs.permute(0, 2, 3, 1)
        targets_reshape = targets.permute(0, 2, 3, 1)

        inputs_OD, input_block, input_histo, input_mask = self.compute_OD(inputs_reshape)
        targets_OD, target_block, target_histo, target_mask = self.compute_OD(targets_reshape)

        B = inputs.shape[0]
        H = inputs.shape[2]
        W = inputs.shape[3]
        area = float(H * W)

        # Level 1 — Global average
        mlpa_avg = self.mse_loss_2(inputs_OD, targets_OD) / (area ** 2)

        # Level 2 — Histogram (20 bins)
        mlpa_histo = (
            ((input_histo / area - target_histo / area) ** 2).sum(1)
        ) / B

        # Level 3 — Block (4×4 grid)
        mlpa_block = self.mse_loss(
            input_block / (area / self.num_blocks),
            target_block / (area / self.num_blocks),
        )

        # Adaptive weighting:
        #   close_enough → histogram only (distribution shape)
        #   far apart    → global + histogram (magnitude + shape)
        close_enough = (inputs_OD - targets_OD >= targets_OD * -self.tolerance) & \
                       (inputs_OD - targets_OD <= targets_OD * self.tolerance)
        loss_mlpa = torch.sum(torch.where(
            close_enough,
            mlpa_histo,
            mlpa_avg + mlpa_histo,
        ))
        loss_mlpa = loss_mlpa + mlpa_block

        return loss_mlpa, input_mask, target_mask

    # ------------------------------------------------------------------
    # FOD Computation
    # ------------------------------------------------------------------

    def compute_OD(self, image):
        """Compute Focal Optical Density statistics from an IHC image.

        Args:
            image: [B, H, W, 3] RGB in [-1, 1] or [0, 1]

        Returns:
            avg:   [B]          per-image FOD sum
            block: [B, 16]      per-block FOD sums (4×4 grid)
            histo: [B, 20]      20-bin histogram of FOD values
            mask:  [B, H, W]    binary pseudo-mask for CTPC
        """
        assert image.shape[-1] == 3

        device = image.device

        # ── 1. Colour deconvolution → DAB optical density ──────────
        ihc_hed = self._separate_stains(image, self.hed_from_rgb)
        null = torch.zeros_like(ihc_hed[:, :, :, 0])

        # DAB-only RGB reconstruction
        ihc_d = self._combine_stains(
            torch.stack((null, null, ihc_hed[:, :, :, 2]), axis=-1),
            self.rgb_from_hed,
        )

        # Grayscale DAB
        grey_d = self._rgb2gray(ihc_d)
        grey_d = grey_d.clamp(0.0, 1.0)

        # ── 2. Focal Optical Density ───────────────────────────────
        calib = self.adjust_calibration.to(device)
        FOD = torch.log10(1.0 / (grey_d + calib))
        FOD = FOD.clamp(min=0.0)
        FOD = FOD ** self.alpha

        # ── 3. Thresholding ────────────────────────────────────────
        FOD_relu = torch.where(
            FOD < self.thresh_fod,
            torch.tensor(0.0, device=device),
            FOD,
        )

        # Pseudo-mask for CTPC (binary: 0 or 1)
        mask_OD = torch.where(
            FOD < self.thresh_mask,
            torch.tensor(0.0, device=device),
            FOD,
        )
        mask_OD = mask_OD.squeeze(-1).detach()
        mask_OD = torch.where(mask_OD > 0, torch.tensor(1.0, device=device), mask_OD)

        # ── 4. Global average ──────────────────────────────────────
        avg = torch.sum(FOD_relu, dim=(1, 2, 3))

        # ── 5. Block sums (4×4 grid) ───────────────────────────────
        n_blocks_per_side = int(math.sqrt(self.num_blocks))
        block_h = image.shape[1] // n_blocks_per_side
        block_w = image.shape[2] // n_blocks_per_side

        tensor_blocks = FOD_relu.squeeze(-1).unfold(
            1, block_h, block_h
        ).unfold(
            2, block_w, block_w
        )
        block = tensor_blocks.sum(dim=(3, 4))

        # ── 6. Histogram (20 bins) ─────────────────────────────────
        flattened = FOD.squeeze(-1).flatten(1, 2)
        histo = self._calculate_histo_sums(
            flattened,
            self.num_bins,
            0.0,
            math.e,
        )

        return avg, block, histo, mask_OD

    # ------------------------------------------------------------------
    # Beer-Lambert colour deconvolution helpers
    # ------------------------------------------------------------------

    def _separate_stains(self, rgb, conv_matrix):
        """RGB → optical-density stain concentrations."""
        rgb = torch.maximum(rgb, torch.tensor(1e-6, device=rgb.device))
        log_adjust = self.log_adjust.to(rgb.device)
        stains = torch.matmul(torch.log(rgb) / log_adjust, conv_matrix)
        return torch.maximum(stains, torch.tensor(0.0, device=rgb.device))

    def _combine_stains(self, stains, conv_matrix):
        """Stain concentrations → RGB (inverse of _separate_stains)."""
        log_adjust = self.log_adjust.to(stains.device)
        rgb = torch.exp(-torch.matmul(stains * -log_adjust, conv_matrix))
        return torch.clamp(rgb, min=0.0, max=1.0)

    def _rgb2gray(self, rgb):
        """RGB → grayscale using luminance weights."""
        return torch.matmul(rgb, self.coeffs.to(rgb.device))

    # ------------------------------------------------------------------
    # Histogram helper
    # ------------------------------------------------------------------

    def _calculate_histo_sums(self, features, num_bins, min_val, max_val):
        """Sum feature values falling into each of `num_bins` equal-width bins.

        Args:
            features: [B, N] flattened FOD values
            num_bins: number of histogram bins
            min_val, max_val: value range

        Returns:
            batch_sums: [B, num_bins] sum of values in each bin
        """
        device = features.device
        bucket_width = (max_val - min_val) / num_bins
        normalized = (features - min_val) / bucket_width
        histo_indices = normalized.clamp(0, num_bins - 1).long()

        batch_sums = torch.zeros(features.shape[0], num_bins, device=device)
        for i in range(features.shape[0]):
            for j in range(num_bins):
                indices = (histo_indices[i] == j).nonzero(as_tuple=True)[0]
                if indices.numel() > 0:
                    batch_sums[i, j] = torch.sum(features[i, indices])

        return batch_sums
