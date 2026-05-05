"""
Edge encoders for UNIStainNet: parallel structure pathway from H-map edges.

Both encoders accept [B, 1, 512, 512] Hematoxylin channel (H-map) in [-1, 1].
- EdgeEncoder (v1): Sobel → [gx, gy] (2ch) → sequential multi-scale CNN
- MultiScaleEdgeEncoder (v2): Per-scale [h, gx, gy] (3ch) → independent CNNs at each resolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeEncoder(nn.Module):
    """Lightweight encoder that extracts multi-scale edge features from H&E input.

    Extracts Sobel edges from grayscale H&E, then encodes them through a small
    CNN to produce multi-scale feature maps. These are concatenated with the
    main encoder's skip connections in the decoder, giving the generator an
    explicit structural signal.

    Key insight: H&E input and generated output share the exact same spatial
    frame (no misalignment). So edge features from H&E are pixel-aligned with
    the decoder's output — unlike real HER2 ground truth.
    """

    def __init__(self, base_ch=32):
        super().__init__()
        # Sobel kernels (fixed, not learned)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # Edge feature encoder: 2ch (grad_x, grad_y) → multi-scale features
        # Mirrors the main encoder's spatial hierarchy
        self.enc1 = nn.Sequential(  # 512→256, out: base_ch
            nn.Conv2d(2, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(  # 256→128, out: base_ch*2
            nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc3 = nn.Sequential(  # 128→64, out: base_ch*4
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc4 = nn.Sequential(  # 64→32, out: base_ch*4
            nn.Conv2d(base_ch * 4, base_ch * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, h_map):
        """
        Args:
            h_map: [B, 1, 512, 512] Hematoxylin channel in [-1, 1]

        Returns:
            dict of edge features at each decoder resolution:
                256: [B, base_ch, 256, 256]
                128: [B, base_ch*2, 128, 128]
                64:  [B, base_ch*4, 64, 64]
                32:  [B, base_ch*4, 32, 32]
        """
        # Rescale [-1, 1] → [0, 1] for consistent Sobel magnitudes
        gray = (h_map + 1) / 2  # [B, 1, 512, 512]

        # Sobel edge detection
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        edges = torch.cat([gx, gy], dim=1)  # [B, 2, 512, 512]

        # Multi-scale encoding
        e1 = self.enc1(edges)   # [B, base_ch, 256, 256]
        e2 = self.enc2(e1)     # [B, base_ch*2, 128, 128]
        e3 = self.enc3(e2)     # [B, base_ch*4, 64, 64]
        e4 = self.enc4(e3)     # [B, base_ch*4, 32, 32]

        return {256: e1, 128: e2, 64: e3, 32: e4}


class MultiScaleEdgeEncoder(nn.Module):
    """Multi-scale edge encoder with independent per-scale edge extraction.

    Improvements over EdgeEncoder (v1):
    1. Structure-aware 3ch input per scale: [h_map, gx, gy] — the raw H-map value
       is preserved alongside its gradients so the network can distinguish intensity
       from edge contrast (e.g., bright nuclei boundary vs dark-on-dark membrane).
    2. Multi-scale Sobel: Edges extracted independently at each resolution before
       encoding. Fine 2-5px membrane edges don't get lost through sequential pooling.
    3. Edge features at 512: Provides features at output resolution for fine
       structure preservation (cell walls, membrane patterns).
    """

    def __init__(self, base_ch=32):
        super().__init__()
        # Fixed Sobel kernels for structural prior
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # Per-scale feature extractors
        # Input: 1ch H-map + 2ch Sobel (gx, gy) = 3ch at each scale
        in_ch = 3

        # 512→512 (edge features at output resolution)
        self.scale_512 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 256×256
        self.scale_256 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 128×128
        self.scale_128 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch * 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 64×64
        self.scale_64 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch * 4, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 32×32
        self.scale_32 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch * 4, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def _extract_edges_at_scale(self, h_map_01, size):
        """Downsample H-map, extract Sobel edges, return [h_map, gx, gy]."""
        if size < h_map_01.shape[-1]:
            h = F.interpolate(h_map_01, size=size, mode='bilinear', align_corners=False)
        else:
            h = h_map_01  # [B, 1, size, size]
        gx = F.conv2d(h, self.sobel_x, padding=1)
        gy = F.conv2d(h, self.sobel_y, padding=1)
        return torch.cat([h, gx, gy], dim=1)  # [B, 3, size, size]

    def forward(self, h_map):
        """
        Args:
            h_map: [B, 1, 512, 512] Hematoxylin channel in [-1, 1]

        Returns:
            dict of edge features at each decoder resolution:
                512: [B, base_ch, 512, 512]
                256: [B, base_ch, 256, 256]
                128: [B, base_ch*2, 128, 128]
                64:  [B, base_ch*4, 64, 64]
                32:  [B, base_ch*4, 32, 32]
        """
        h_map_01 = (h_map + 1) / 2  # [-1, 1] → [0, 1] for consistent Sobel magnitudes

        return {
            512: self.scale_512(self._extract_edges_at_scale(h_map_01, 512)),
            256: self.scale_256(self._extract_edges_at_scale(h_map_01, 256)),
            128: self.scale_128(self._extract_edges_at_scale(h_map_01, 128)),
            64: self.scale_64(self._extract_edges_at_scale(h_map_01, 64)),
            32: self.scale_32(self._extract_edges_at_scale(h_map_01, 32)),
        }
