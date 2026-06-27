"""
Building blocks for UNIStainNet generator.

- SPADEBlock: SPADE + FiLM normalization (UNI spatial + class channel modulation)
- ResBlock: Residual block with InstanceNorm
- SelfAttention: Self-attention for global context at bottleneck
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPADEBlock(nn.Module):
    """SPADE + FiLM normalization block.

    Combines spatially-adaptive normalization from UNI features (SPADE)
    with channel-wise affine modulation from class embedding (FiLM).
    """

    def __init__(self, norm_channels, uni_channels, class_dim=64):
        super().__init__()
        self.norm = nn.InstanceNorm2d(norm_channels, affine=False)

        # SPADE: learn spatial gamma/beta from UNI features
        hidden = min(128, norm_channels)
        self.spade_shared = nn.Sequential(
            nn.Conv2d(uni_channels, hidden, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.spade_gamma = nn.Conv2d(hidden, norm_channels, 3, padding=1)
        self.spade_beta = nn.Conv2d(hidden, norm_channels, 3, padding=1)

        # FiLM: learn channel gamma/beta from class embedding
        self.film_gamma = nn.Linear(class_dim, norm_channels)
        self.film_beta = nn.Linear(class_dim, norm_channels)

        # Init SPADE gamma/beta near zero (ControlNet-style gradual activation)
        nn.init.zeros_(self.spade_gamma.weight)
        nn.init.zeros_(self.spade_gamma.bias)
        nn.init.zeros_(self.spade_beta.weight)
        nn.init.zeros_(self.spade_beta.bias)

        # Init FiLM gamma near 1, beta near 0
        nn.init.ones_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

    def forward(self, x, uni_spatial, class_emb):
        """
        Args:
            x: [B, C, H, W] feature map
            uni_spatial: [B, uni_ch, H, W] UNI features at matching resolution
            class_emb: [B, class_dim] class embedding
        """
        normalized = self.norm(x)

        # SPADE modulation from UNI features
        shared = self.spade_shared(uni_spatial)
        gamma_s = self.spade_gamma(shared)
        beta_s = self.spade_beta(shared)

        # FiLM modulation from class
        gamma_c = self.film_gamma(class_emb).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta_c = self.film_beta(class_emb).unsqueeze(-1).unsqueeze(-1)

        # Combined: (gamma_spade + gamma_film) * norm(x) + (beta_spade + beta_film)
        return (gamma_s + gamma_c) * normalized + (beta_s + beta_c)


class ResBlock(nn.Module):
    """Residual block with InstanceNorm."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class SelfAttention(nn.Module):
    """Self-attention layer for global context at bottleneck."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = (q.transpose(-1, -2) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        out = (v @ attn.transpose(-1, -2)).reshape(B, C, H, W)
        return x + self.proj(out)


# ======================================================================
# Architectural Improvements (Run 11)
# ======================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation — learned per-channel importance gating.

    Global average pool → bottleneck FC → sigmoid → reweight channels.
    Adds ~2% params per insertion, near-zero compute overhead.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class CoordAttn(nn.Module):
    """Coordinate Attention — lightweight positional encoding via factorised attention.

    Decomposes spatial attention into horizontal + vertical strips.
    Far cheaper than full self-attention (O(H+W) vs O(H×W)).
    """

    def __init__(self, channels, reduction=32):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.bn = nn.BatchNorm2d(hidden)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(hidden, channels, 1)
        self.conv_w = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        # Factorised pooling
        h_feat = self.pool_h(x)  # [B, C, H, 1]
        w_feat = self.pool_w(x)  # [B, C, 1, W]
        # Shared transform
        cat = torch.cat([h_feat, w_feat.expand(-1, -1, H, -1)], dim=3)
        shared = self.act(self.bn(self.conv1(cat)))
        h_out = self.conv_h(shared[:, :, :, :1])  # [B, C, H, 1]
        w_out = self.conv_w(shared[:, :, :, 1:1+W])  # [B, C, 1, W]
        return x * h_out.sigmoid() * w_out.sigmoid()
