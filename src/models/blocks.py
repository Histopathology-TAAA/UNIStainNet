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

    def __init__(self, norm_channels, uni_channels, class_dim=64, cls_dim=1536):
        super().__init__()
        self.cls_dim = cls_dim
        self.norm = nn.InstanceNorm2d(norm_channels, affine=False)

        # SPADE: learn spatial gamma/beta from UNI features + broadcasted CLS token
        hidden = min(128, norm_channels)
        self.spade_shared = nn.Sequential(
            nn.Conv2d(uni_channels + cls_dim, hidden, 3, padding=1),
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

    def forward(self, x, uni_spatial, class_emb, cls_emb=None):
        """
        Args:
            x: [B, C, H, W] feature map
            uni_spatial: [B, uni_ch, H, W] UNI features at matching resolution
            class_emb: [B, class_dim] class embedding
            cls_emb: [B, cls_dim] global CLS token to broadcast and concat
        """
        normalized = self.norm(x)

        # Broadcast global CLS token and concatenate with spatial UNI features
        B, _, H, W = uni_spatial.shape
        if cls_emb is not None:
            cls_spatial = cls_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        else:
            cls_spatial = torch.zeros((B, self.cls_dim, H, W), device=uni_spatial.device, dtype=uni_spatial.dtype)
            
        uni_spatial = torch.cat([uni_spatial, cls_spatial], dim=1)

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


class CrossAttention(nn.Module):
    """Cross-attention layer for attending to sub-crop CLS tokens at the bottleneck."""

    def __init__(self, channels, context_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels)
        self.norm2 = nn.LayerNorm(context_dim)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Linear(context_dim, channels)
        self.v = nn.Linear(context_dim, channels)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x, context):
        """
        x: [B, C, H, W] spatial features
        context: [B, N, context_dim] sequence of tokens to attend to
        """
        B, C, H, W = x.shape
        h = self.norm1(x)
        ctx = self.norm2(context)
        
        q = self.q(h).reshape(B, C, H * W).transpose(-1, -2)  # [B, HW, C]
        k = self.k(ctx)  # [B, N, C]
        v = self.v(ctx)  # [B, N, C]
        
        attn = (q @ k.transpose(-1, -2)) * self.scale  # [B, HW, N]
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(-1, -2).reshape(B, C, H, W)  # [B, C, H, W]
        return x + self.proj(out)

