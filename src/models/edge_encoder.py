"""
Edge encoders for UNIStainNet: parallel structure pathway from H-channel edges.

- EdgeEncoder (v1): Sobel on 1-ch H-image → multi-scale CNN
- MultiScaleEdgeEncoder (v2): Independent per-scale edge extraction from 1-ch H-image

Both encoders now accept [B, 1, H, W] H-channel images (hematoxylin channel)
instead of full RGB. This gives the encoder a cleaner structural signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeEncoder(nn.Module):
    """Lightweight encoder that extracts multi-scale edge features from a H-channel image.

    Accepts a single-channel H-image [B, 1, 512, 512], applies Sobel edge detection,
    then encodes through a small CNN to produce multi-scale feature maps.
    No architecture weight change vs the original: enc1 still takes 2-ch Sobel output.
    """

    def __init__(self, base_ch=32, learnable_sobel=False):
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2)
        
        if learnable_sobel:
            self.sobel_x = nn.Parameter(sobel_x)
            self.sobel_y = nn.Parameter(sobel_y)
        else:
            self.register_buffer('sobel_x', sobel_x)
            self.register_buffer('sobel_y', sobel_y)

        # Edge feature encoder: 2ch (grad_x, grad_y) → multi-scale features
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

    def forward(self, h_image):
        """
        Args:
            h_image: [B, 1, 512, 512] in [-1, 1] — H-channel (single channel)

        Returns:
            dict of edge features at each decoder resolution:
                256: [B, base_ch, 256, 256]
                128: [B, base_ch*2, 128, 128]
                64:  [B, base_ch*4, 64, 64]
                32:  [B, base_ch*4, 32, 32]
        """
        # Map to [0, 1]
        gray = (h_image + 1) / 2  # [B, 1, 512, 512]

        # Sobel edge detection
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        edges = torch.cat([gx, gy], dim=1)  # [B, 2, 512, 512]

        # Multi-scale encoding
        e1 = self.enc1(edges)   # [B, base_ch, 256, 256]
        e2 = self.enc2(e1)      # [B, base_ch*2, 128, 128]
        e3 = self.enc3(e2)      # [B, base_ch*4, 64, 64]
        e4 = self.enc4(e3)      # [B, base_ch*4, 32, 32]

        return {256: e1, 128: e2, 64: e3, 32: e4}


class MultiScaleEdgeEncoder(nn.Module):
    """Multi-scale edge encoder from a 1-channel H-image.

    Accepts a single-channel H-image [B, 1, 512, 512].
    At each scale: H-channel + 2-ch Sobel = 3-ch input (was 5-ch with RGB).
    Independent per-scale edge extraction so fine edges are not lost.
    Provides features at 512 for fine structure at output resolution.
    """

    def __init__(self, base_ch=32, learnable_sobel=False):
        super().__init__()
        # Sobel kernels for structural prior
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2)
        
        if learnable_sobel:
            self.sobel_x = nn.Parameter(sobel_x)
            self.sobel_y = nn.Parameter(sobel_y)
        else:
            self.register_buffer('sobel_x', sobel_x)
            self.register_buffer('sobel_y', sobel_y)

        # Per-scale feature extractors
        # Input: 1-ch H-channel + 2-ch Sobel = 3-ch at each scale
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

    def _extract_edges_at_scale(self, h_01, size):
        """Downsample H-channel, extract Sobel edges, return H+edges [B, 3, size, size]."""
        if size < h_01.shape[-1]:
            h = F.interpolate(h_01, size=size, mode='bilinear', align_corners=False)
        else:
            h = h_01
        # h: [B, 1, size, size]
        gx = F.conv2d(h, self.sobel_x, padding=1)
        gy = F.conv2d(h, self.sobel_y, padding=1)
        return torch.cat([h, gx, gy], dim=1)  # [B, 3, size, size]

    def forward(self, h_image):
        """
        Args:
            h_image: [B, 1, 512, 512] in [-1, 1] — H-channel (single channel)

        Returns:
            dict of edge features at each decoder resolution:
                512: [B, base_ch, 512, 512]
                256: [B, base_ch, 256, 256]
                128: [B, base_ch*2, 128, 128]
                64:  [B, base_ch*4, 64, 64]
                32:  [B, base_ch*4, 32, 32]
        """
        h_01 = (h_image + 1) / 2  # [0, 1] for consistent edge magnitudes

        return {
            512: self.scale_512(self._extract_edges_at_scale(h_01, 512)),
            256: self.scale_256(self._extract_edges_at_scale(h_01, 256)),
            128: self.scale_128(self._extract_edges_at_scale(h_01, 128)),
            64:  self.scale_64(self._extract_edges_at_scale(h_01, 64)),
            32:  self.scale_32(self._extract_edges_at_scale(h_01, 32)),
        }
