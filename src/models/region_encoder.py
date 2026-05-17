"""
Region Encoder for patch-level tissue compartment context.

A tiny CNN (5 conv layers, ~50K params) processes a 512×512 low-res
H&E thumbnail of the full WSI, producing a spatial feature map that
encodes tissue compartments (tumor, stroma, background, boundaries,
cell density gradients). Per-patch region embeddings are obtained by
bilinearly sampling the feature map at the patch's normalized WSI
position.

The thumbnail is stored during patching (patch_extraction.py) and
loaded at training time — no registered WSI access needed.

Trained from scratch jointly with the generator SPADE decoder at low
learning rate (5e-5). No ImageNet or pathology pretraining dependency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionEncoder(nn.Module):
    """Tiny CNN that encodes a WSI thumbnail into a compartment feature map.

    Architecture: 5× Conv2d(stride=2) → 64×16×16 → Pool → 64×8×8

    Input:  [1, 3, 512, 512]  low-res H&E thumbnail
    Output: [1, 64, 8, 8]     compartment feature map

    At 2.5X equivalent on a 15mm tissue section, each 8×8 cell covers
    ~1.9mm — the right scale for tumor-stroma boundaries and gland
    architecture.

    Args:
        output_dim: dimension of region embedding vectors (default 64)
        grid_size: spatial resolution of output feature map (default 8)
    """

    def __init__(self, output_dim=64, grid_size=8):
        super().__init__()
        self.output_dim = output_dim
        self.grid_size = grid_size

        # 5 strided conv layers: 512→256→128→64→32→16
        # Channels: 3→32→64→128→64→64
        self.encoder = nn.Sequential(
            # layer 1: 512→256
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            # layer 2: 256→128
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True),
            # layer 3: 128→64
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            # layer 4: 64→32
            nn.Conv2d(128, 64, 5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            # layer 5: 32→16
            nn.Conv2d(64, output_dim, 5, stride=2, padding=2),
        )

        # Pool to target grid size
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))

        # For annotation: param count
        n_params = sum(p.numel() for p in self.parameters())
        print(f"RegionEncoder: {n_params:,} params, "
              f"output {output_dim}×{grid_size}×{grid_size}")

    def forward(self, thumbnail):
        """Encode a WSI thumbnail into a compartment feature map.

        Args:
            thumbnail: [B, 3, 512, 512] in [0, 255] uint8 range
                       (normalized internally)

        Returns:
            feature_map: [B, output_dim, grid_size, grid_size]
        """
        # Normalize from uint8 [0,255] to [-1, 1]
        x = thumbnail.float() / 127.5 - 1.0
        x = self.encoder(x)
        x = self.pool(x)
        return x

    @staticmethod
    def sample(feature_map, norm_x, norm_y):
        """Bilinearly sample region embedding at a normalized WSI position.

        Args:
            feature_map: [B, output_dim, grid_size, grid_size]
            norm_x: [B] or scalar — normalized x position [0, 1]
            norm_y: [B] or scalar — normalized y position [0, 1]

        Returns:
            region_emb: [B, output_dim] — per-patch region embedding
        """
        B = feature_map.shape[0]
        device = feature_map.device

        if not isinstance(norm_x, torch.Tensor):
            norm_x = torch.tensor(norm_x, device=device)
        if not isinstance(norm_y, torch.Tensor):
            norm_y = torch.tensor(norm_y, device=device)

        # grid_sample expects [B, H_out, W_out, 2] in [-1, 1]
        # (-1,-1) = top-left, (1,1) = bottom-right
        grid_x = norm_x.float() * 2.0 - 1.0  # [0,1] → [-1,1]
        grid_y = norm_y.float() * 2.0 - 1.0

        grid = torch.stack([grid_x, grid_y], dim=1)  # [B, 2]
        grid = grid.view(B, 1, 1, 2)                 # [B, 1, 1, 2]

        sampled = F.grid_sample(
            feature_map, grid,
            mode='bilinear', padding_mode='border', align_corners=True,
        )  # [B, output_dim, 1, 1]

        return sampled.squeeze(-1).squeeze(-1)  # [B, output_dim]
