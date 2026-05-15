"""
Building blocks for UNIStainNet generator.

- SPADEBlock: SPADE + FiLM normalization (UNI spatial + class channel modulation)
- ResBlock: Residual block with InstanceNorm
- SelfAttention: Self-attention for global context at bottleneck
- CrossAttention: Cross-attention with Q from spatial features and K/V from UNI tokens
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
            uni_spatial: [B, uni_ch, H, W] UNI features at matching resolution,
                or None to disable the SPADE branch and keep only FiLM
            class_emb: [B, class_dim] class embedding
        """
        normalized = self.norm(x)

        # SPADE modulation from UNI features; allow label-only ablation by
        # skipping the UNI branch entirely.
        if uni_spatial is not None:
            shared = self.spade_shared(uni_spatial)
            gamma_s = self.spade_gamma(shared)
            beta_s = self.spade_beta(shared)
        else:
            gamma_s = torch.zeros_like(normalized)
            beta_s = torch.zeros_like(normalized)

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
    """Cross-attention: Q from spatial features, K/V from UNI tokens."""

    def __init__(self, channels, uni_dim=1024, heads=4):
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by heads ({heads})")

        self.heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(32, channels)
        self.token_norm = nn.LayerNorm(uni_dim)

        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Linear(uni_dim, channels)
        self.v_proj = nn.Linear(uni_dim, channels)
        self.out_proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x, uni_tokens):
        """
        Args:
            x: [B, C, H, W] spatial features
            uni_tokens: [B, N, D] or [B, D, H, W] UNI tokens
        """
        B, C, H, W = x.shape
        if uni_tokens.dim() == 4:
            uni_tokens = uni_tokens.permute(0, 2, 3, 1).reshape(B, -1, uni_tokens.shape[1])

        uni_tokens = self.token_norm(uni_tokens)

        q = self.q_proj(self.norm(x)).reshape(B, self.heads, self.head_dim, H * W)
        q = q.permute(0, 1, 3, 2)  # [B, heads, HW, head_dim]

        k = self.k_proj(uni_tokens).reshape(B, -1, self.heads, self.head_dim)
        v = self.v_proj(uni_tokens).reshape(B, -1, self.heads, self.head_dim)
        k = k.permute(0, 2, 3, 1)  # [B, heads, head_dim, N]
        v = v.permute(0, 2, 1, 3)  # [B, heads, N, head_dim]

        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v  # [B, heads, HW, head_dim]
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        return x + self.out_proj(out)
