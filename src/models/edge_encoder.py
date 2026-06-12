"""
Edge encoders for UNIStainNet: parallel structure pathway.

- EdgeEncoder (v1): Multi-scale CNN from structural input
- MultiScaleEdgeEncoder (v2): Independent per-scale extraction

Both encoders support two input modes controlled by ``input_channels``:
  - ``input_channels=1`` (legacy): 1-ch H-channel → Sobel edges → CNN
  - ``input_channels=3`` (DeepLIIF): 3-ch virtual Hematoxylin → CNN directly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeEncoder(nn.Module):
    """Lightweight encoder that extracts multi-scale edge features.

    Accepts either a 1-channel H-image (legacy) or 3-channel DeepLIIF
    Hematoxylin, then encodes through a small CNN to produce multi-scale
    feature maps for skip connections into the decoder.

    Args:
        base_ch: Base channel count for feature maps.
        learnable_sobel: If True, Sobel kernels are learnable (only used
            when ``input_channels=1``).
        input_channels: Number of input channels. 1 = legacy H-channel
            (uses Sobel), 3 = DeepLIIF Hematoxylin (passed directly).
    """

    def __init__(self, base_ch=32, learnable_sobel=False, input_channels=1):
        super().__init__()
        self.input_channels = input_channels

        # Sobel kernels (used only when input_channels=1)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2)

        if learnable_sobel:
            self.sobel_x = nn.Parameter(sobel_x)
            self.sobel_y = nn.Parameter(sobel_y)
        else:
            self.register_buffer('sobel_x', sobel_x)
            self.register_buffer('sobel_y', sobel_y)

        # Determine entry conv input channels
        if input_channels == 1:
            enc1_in_ch = 2  # Legacy: 2-ch Sobel (grad_x, grad_y)
        else:
            enc1_in_ch = input_channels  # DeepLIIF: 3-ch raw hematoxylin

        # Edge feature encoder → multi-scale features
        self.enc1 = nn.Sequential(  # 512→256, out: base_ch
            nn.Conv2d(enc1_in_ch, base_ch, 4, stride=2, padding=1),
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

    def forward(self, image):
        """
        Args:
            image: [B, C, 512, 512] in [-1, 1]
                   C=1 (legacy H-channel) or C=3 (DeepLIIF Hematoxylin)

        Returns:
            dict of edge features at each decoder resolution:
                256: [B, base_ch, 256, 256]
                128: [B, base_ch*2, 128, 128]
                64:  [B, base_ch*4, 64, 64]
                32:  [B, base_ch*4, 32, 32]
        """
        if self.input_channels == 1:
            # Legacy path: Sobel edge detection on 1-ch H-channel
            gray = (image + 1) / 2  # [B, 1, 512, 512] → [0, 1]
            gx = F.conv2d(gray, self.sobel_x, padding=1)
            gy = F.conv2d(gray, self.sobel_y, padding=1)
            x = torch.cat([gx, gy], dim=1)  # [B, 2, 512, 512]
        else:
            # DeepLIIF path: pass 3-ch hematoxylin directly
            x = (image + 1) / 2  # [B, 3, 512, 512] → [0, 1]

        # Multi-scale encoding
        e1 = self.enc1(x)   # [B, base_ch, 256, 256]
        e2 = self.enc2(e1)  # [B, base_ch*2, 128, 128]
        e3 = self.enc3(e2)  # [B, base_ch*4, 64, 64]
        e4 = self.enc4(e3)  # [B, base_ch*4, 32, 32]

        return {256: e1, 128: e2, 64: e3, 32: e4}


class MultiScaleEdgeEncoder(nn.Module):
    """Multi-scale edge encoder with independent per-scale feature extraction.

    Supports two input modes:
      - ``input_channels=1`` (legacy): 1-ch H-channel → per-scale Sobel+H = 3ch
      - ``input_channels=3`` (DeepLIIF): 3-ch hematoxylin → per-scale downsample = 3ch

    Both modes produce 3-ch input to each scale block, so the scale convs
    remain identical. Only the preprocessing changes.

    Args:
        base_ch: Base channel count.
        learnable_sobel: If True, Sobel kernels are learnable (legacy mode only).
        input_channels: Number of input channels (1 or 3).
    """

    def __init__(self, base_ch=32, learnable_sobel=False, input_channels=1):
        super().__init__()
        self.input_channels = input_channels

        # Sobel kernels for structural prior (used only when input_channels=1)
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
        # Both modes produce 3-ch input to each scale block:
        #   Legacy: 1-ch H + 2-ch Sobel = 3ch
        #   DeepLIIF: 3-ch raw hematoxylin = 3ch
        in_ch = 3

        # 512→512
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

    def _prepare_at_scale(self, image_01, size):
        """Prepare input at a given scale.

        For legacy (1-ch): downsample + Sobel → [B, 3, size, size]
        For DeepLIIF (3-ch): just downsample → [B, 3, size, size]
        """
        if self.input_channels == 1:
            # Legacy: Sobel edge extraction at this scale
            if size < image_01.shape[-1]:
                h = F.interpolate(image_01, size=size, mode='bilinear', align_corners=False)
            else:
                h = image_01
            gx = F.conv2d(h, self.sobel_x, padding=1)
            gy = F.conv2d(h, self.sobel_y, padding=1)
            return torch.cat([h, gx, gy], dim=1)  # [B, 3, size, size]
        else:
            # DeepLIIF: just downsample the 3-ch hematoxylin
            if size < image_01.shape[-1]:
                return F.interpolate(image_01, size=size, mode='bilinear', align_corners=False)
            return image_01

    def forward(self, image):
        """
        Args:
            image: [B, C, 512, 512] in [-1, 1]
                   C=1 (legacy H-channel) or C=3 (DeepLIIF Hematoxylin)

        Returns:
            dict of edge features at each decoder resolution:
                512: [B, base_ch, 512, 512]
                256: [B, base_ch, 256, 256]
                128: [B, base_ch*2, 128, 128]
                64:  [B, base_ch*4, 64, 64]
                32:  [B, base_ch*4, 32, 32]
        """
        image_01 = (image + 1) / 2  # [0, 1] for consistent processing

        return {
            512: self.scale_512(self._prepare_at_scale(image_01, 512)),
            256: self.scale_256(self._prepare_at_scale(image_01, 256)),
            128: self.scale_128(self._prepare_at_scale(image_01, 128)),
            64:  self.scale_64(self._prepare_at_scale(image_01, 64)),
            32:  self.scale_32(self._prepare_at_scale(image_01, 32)),
        }
