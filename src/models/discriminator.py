"""
PatchGAN discriminator for HER2 image realism scoring.

Supports both unconditional (3ch HER2 only) and conditional (6ch H&E+HER2) modes.
Returns patch-level logits AND intermediate features for feature matching loss.

Architecture:
    C64(SN) -> C128(SN+IN) -> C256(SN+IN) -> C512(SN+IN,s1) -> 1ch(SN,s1)
    70x70 receptive field, output [B, 1, 30, 30] for 512x512 input.
    ~2.8M params (3ch) or ~2.8M params (6ch).

References:
  - Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks" (CVPR 2017)
  - Miyato et al., "Spectral Normalization for GANs" (ICLR 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils import spectral_norm


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator with spectral normalization.

    Returns both logits and intermediate features (for feature matching loss).

    Args:
        in_channels: 3 for unconditional (HER2 only), 6 for conditional (H&E + HER2)
        ndf: base number of discriminator filters
        n_layers: number of intermediate conv layers
    """

    def __init__(self, in_channels=3, ndf=64, n_layers=3):
        super().__init__()
        self.n_layers = n_layers

        # Build layers as a list (not sequential) so we can extract features
        self.layers = nn.ModuleList()

        # First layer: spectral norm, no instance norm
        self.layers.append(nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ))

        # Intermediate layers: spectral norm + instance norm
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            self.layers.append(nn.Sequential(
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=2, padding=1)),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ))

        # Penultimate layer: stride 1
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.layers.append(nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=1, padding=1)),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
        ))

        # Final layer: 1-channel output, no activation (hinge loss uses raw logits)
        self.layers.append(nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * nf_mult, 1, 4, stride=1, padding=1)),
        ))

    def forward(self, x, return_features=False):
        """
        Args:
            x: [B, C, H, W] in [-1, 1]. C=3 (unconditional) or C=6 (conditional).
            return_features: if True, also return intermediate features for FM loss.

        Returns:
            logits: [B, 1, H', W'] patch-level real/fake logits
            features: list of intermediate feature maps (only if return_features=True)
        """
        features = []
        h = x
        for layer in self.layers:
            h = layer(h)
            if return_features:
                features.append(h)

        if return_features:
            return h, features
        return h


# ======================================================================
# Loss functions
# ======================================================================

def hinge_loss_d(d_real, d_fake):
    """Discriminator hinge loss."""
    return (torch.relu(1.0 - d_real).mean() + torch.relu(1.0 + d_fake).mean()) / 2


def hinge_loss_g(d_fake):
    """Generator hinge loss."""
    return -d_fake.mean()


def r1_gradient_penalty(discriminator, real_images, weight=10.0):
    """R1 gradient penalty (Mescheder et al., 2018).

    Regularizes discriminator to have small gradients on real data,
    which prevents the discriminator from becoming too confident and
    stabilizes GAN training.
    """
    real_images = real_images.detach().requires_grad_(True)
    d_real = discriminator(real_images)
    grad_real = autograd.grad(
        outputs=d_real.sum(),
        inputs=real_images,
        create_graph=True,
    )[0]
    penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return weight * penalty


def feature_matching_loss(d_feats_fake, d_feats_real):
    """Feature matching loss: L1 between intermediate discriminator features.

    Excludes the final logit layer ([:-1]) — comparing raw scores is redundant
    with the adversarial loss and blurs the texture-matching signal.
    Alignment-free because it compares feature distributions, not pixel correspondence.
    """
    feats_fake = d_feats_fake[:-1]
    feats_real = d_feats_real[:-1]
    if not feats_fake:
        return torch.tensor(0.0, device=d_feats_fake[0].device)
    loss = 0.0
    for feat_fake, feat_real in zip(feats_fake, feats_real):
        loss += torch.nn.functional.l1_loss(feat_fake, feat_real.detach())
    return loss / len(feats_fake)


class ProjectionDiscriminator(nn.Module):
    """Stain-conditioned PatchGAN using projection conditioning (Miyato & Koyama, 2018).

    Receives IHC image only (no H&E) + stain label. The stain embedding is
    projected onto the penultimate feature map so the discriminator develops
    stain-specific realism criteria:
        HER2   → membrane-localized DAB is real; nuclear DAB is penalized
        Ki67   → nuclear DAB is real
        ER/PR  → nuclear DAB is real

    Using IHC image alone (not paired with H&E) makes this fully
    misalignment-safe: no spatial correspondence with the structural input
    is required.

    Disable: set proj_disc_weight=0.0 in trainer — the discriminator is
    never instantiated and adds zero VRAM or compute cost.
    """

    def __init__(self, in_channels=3, ndf=64, n_layers=3, num_stains=5, stain_dim=64):
        super().__init__()

        self.layers = nn.ModuleList()

        # First layer: spectral norm, no instance norm
        self.layers.append(nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ))

        # Intermediate layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            self.layers.append(nn.Sequential(
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=2, padding=1)),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ))

        # Penultimate: stride 1 — output is the projection target
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        feat_ch = ndf * nf_mult
        self.layers.append(nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, feat_ch, 4, stride=1, padding=1)),
            nn.InstanceNorm2d(feat_ch),
            nn.LeakyReLU(0.2, inplace=True),
        ))

        # 1×1 final conv (keeps spatial alignment with projection logits)
        self.final_conv = spectral_norm(nn.Conv2d(feat_ch, 1, 1))

        # Stain projection: phi(y) → feat_ch weights per spatial location
        self.stain_embed = nn.Embedding(num_stains, stain_dim)
        self.stain_proj = spectral_norm(nn.Linear(stain_dim, feat_ch))

    def forward(self, x, labels, return_features=False):
        """
        Args:
            x: [B, 3, H, W] IHC image in [-1, 1]
            labels: [B] stain labels (0–num_stains-1)
            return_features: if True, return intermediate feature list too

        Returns:
            logits: [B, 1, H', W']
            features: list (only if return_features=True)
        """
        features = []
        h = x
        for layer in self.layers:
            h = layer(h)
            if return_features:
                features.append(h)

        # Base logits
        base_logits = self.final_conv(h)         # [B, 1, H', W']

        # Projection: inner product of features with stain vector
        stain_emb = self.stain_embed(labels)     # [B, stain_dim]
        stain_proj = self.stain_proj(stain_emb)  # [B, feat_ch]
        B = labels.shape[0]
        proj_logits = (h * stain_proj.view(B, -1, 1, 1)).sum(dim=1, keepdim=True)

        logits = base_logits + proj_logits

        if return_features:
            features.append(logits)
            return logits, features
        return logits


class MultiScaleDiscriminator(nn.Module):
    """Two PatchGAN discriminators at different scales."""

    def __init__(self, in_channels=6, ndf=64, n_layers=3):
        super().__init__()
        self.disc_512 = PatchDiscriminator(in_channels, ndf, n_layers)
        self.disc_256 = PatchDiscriminator(in_channels, ndf, n_layers)

    def forward(self, x, return_features=False):
        """
        Args:
            x: [B, 6, 512, 512] concat(output, H&E)

        Returns:
            list of (logits, [features]) from each scale
        """
        x_256 = F.interpolate(x, size=256, mode='bilinear', align_corners=False)

        if return_features:
            out_512, feats_512 = self.disc_512(x, return_features=True)
            out_256, feats_256 = self.disc_256(x_256, return_features=True)
            return [(out_512, feats_512), (out_256, feats_256)]
        else:
            out_512 = self.disc_512(x)
            out_256 = self.disc_256(x_256)
            return [out_512, out_256]
