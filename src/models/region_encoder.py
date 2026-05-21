"""
Region Encoder for patch-level tissue compartment context.

A small CNN (4 strided conv layers, ~400K params) processes a 512×512
low-res H&E thumbnail of the full WSI, producing a spatial feature map
that encodes tissue compartments. Per-patch region embeddings are
obtained by RoIAlign pooling around the patch's WSI position.

The thumbnail is stored during patching and loaded at training time.

Trained from scratch jointly with the generator SPADE decoder.
"""

import torch
import torch.nn as nn
from torchvision.ops import roi_align


class RegionEncoder(nn.Module):
    """CNN that encodes a WSI thumbnail into a compartment feature map.

    Architecture: 4× Conv2d(stride=2) → 32× downsampled feature map
    Input:  [B, 3, 512, 512]  low-res H&E thumbnail (uint8)
    Output: [B, C, 32, 32]    feature map (~0.5mm per cell at 15mm WSI)

    Args:
        output_dim: channels in the feature map (default 64)
        roi_size: output size of RoIAlign (default 8 → 8×8 per region)
        roi_cells: how many feature map cells to pool around each patch
                   (default 8 → ~4mm context window)
    """

    def __init__(self, output_dim=64, roi_size=8, roi_cells=8, spatial_dim=64):
        super().__init__()
        self.output_dim = output_dim
        self.roi_size = roi_size
        self.roi_cells = roi_cells
        self.spatial_dim = spatial_dim

        # 4 strided conv layers: 512→256→128→64→32
        # Channels: 3→32→64→64→output_dim
        self.encoder = nn.Sequential(
            # 512→256
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            # 256→128
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True),
            # 128→64
            nn.Conv2d(64, 64, 5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            # 64→32
            nn.Conv2d(64, output_dim, 5, stride=2, padding=2),
        )
        # Feature map: [B, output_dim, 32, 32]

        # Project RoI features to a flat spatial embedding.
        # SpatialAdaptive pooling preserves SOME layout while being compact:
        # [B, C, R, R] → AdaptiveAvgPool2d(2) → [B, C, 2, 2] → flatten → [B, C*4]
        # → fc → [B, spatial_dim]
        # This is better than global average pool because it retains
        # coarse spatial structure (e.g., "left half is tumor, right half stroma")
        # while still outputting a flat vector for the current SPADE interface.
        self.roi_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.roi_projector = nn.Sequential(
            nn.Linear(output_dim * 4, spatial_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(spatial_dim * 2, spatial_dim),
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"RegionEncoder: {n_params:,} params, "
              f"feature map {output_dim}×32×32, "
              f"RoIAlign {roi_size}×{roi_size} over {roi_cells}×{roi_cells} cells, "
              f"project to {spatial_dim}d")

    def forward(self, thumbnail):
        """Encode WSI thumbnail.

        Args:
            thumbnail: [B, 3, 512, 512] uint8 [0, 255]

        Returns:
            feature_map: [B, output_dim, 32, 32]
        """
        x = thumbnail.float() / 127.5 - 1.0   # [0,255] → [-1,1]
        return self.encoder(x)

    def sample_region(self, feature_map, norm_x, norm_y):
        """RoIAlign around patch position on the feature map.

        Pools the local tissue architecture around each patch — gland
        boundaries, stromal texture, cell density patterns — into a
        fixed-size region feature.

        Args:
            feature_map: [B, C, 32, 32]
            norm_x: [B] patch X position normalized to [0, 1]
            norm_y: [B] patch Y position normalized to [0, 1]

        Returns:
            region_feat: [B, C, roi_size, roi_size]
        """
        B = feature_map.shape[0]
        device = feature_map.device

        if not isinstance(norm_x, torch.Tensor):
            norm_x = torch.as_tensor(norm_x, device=device).float()
        if not isinstance(norm_y, torch.Tensor):
            norm_y = torch.as_tensor(norm_y, device=device).float()

        # Patch center in feature map coordinates [0, 32]
        cx = norm_x * 32.0
        cy = norm_y * 32.0
        half = self.roi_cells / 2.0

        # RoI format: [batch_idx, x1, y1, x2, y2] in feature map coords
        rois = torch.stack([
            torch.arange(B, device=device).float(),
            cx - half,
            cy - half,
            cx + half,
            cy + half,
        ], dim=1)  # [B, 5]

        return roi_align(
            feature_map, rois,
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=1.0,
            aligned=True,
        )  # [B, C, roi_size, roi_size]

    def get_region_emb(self, feature_map, norm_x, norm_y):
        """RoIAlign → spatial pool → project → flat embedding.

        The AdaptiveAvgPool2d(2,2) preserves coarse layout (e.g. "tumor
        on left, stroma on right") while producing a compact flat vector
        that the current SPADEBlock interface accepts.

        Args:
            feature_map: [B, C, 32, 32]
            norm_x, norm_y: [B] normalized patch positions [0, 1]

        Returns:
            region_emb: [B, spatial_dim] flat spatial conditioning vector
        """
        roi = self.sample_region(feature_map, norm_x, norm_y)   # [B, C, R, R]
        pooled = self.roi_pool(roi)                              # [B, C, 2, 2]
        flat = pooled.flatten(1)                                 # [B, C*4]
        return self.roi_projector(flat)                          # [B, spatial_dim]
