"""
DeepLIIF Virtual Hematoxylin Stainer — frozen pretrained wrapper.

Loads a pretrained DeepLIIF ResNet-9blocks generator and extracts the
3-channel Reconstructed Hematoxylin (channels 0:3) from the 15-channel output.

The model is permanently frozen (.eval(), requires_grad=False) and excluded
from checkpoint saving (like the UNI model).

DeepLIIF output channels (5 images × 3 RGB each):
    [0:3]   Hematoxylin (Reconstructed)
    [3:6]   DAPI
    [6:9]   Lap2
    [9:12]  Ki67
    [12:15] Segmentation mask
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLIIFStainer(nn.Module):
    """Frozen DeepLIIF generator for on-the-fly virtual Hematoxylin extraction.

    Args:
        weights_path: Path to ``latest_net_G.pth`` pretrained weights.
        device: Target device (resolved lazily if ``None``).
    """

    def __init__(self, weights_path: str, device=None):
        super().__init__()
        self._weights_path = weights_path
        self._device = device
        self._model = None  # lazy-loaded on first forward

    # ------------------------------------------------------------------
    # Lazy loading (same pattern as UNI model in trainer.py)
    # ------------------------------------------------------------------
    def _load_model(self, device):
        """Instantiate and load DeepLIIF generator weights."""
        if self._model is not None:
            return self._model

        # Load pretrained weights first to inspect the shape
        if not os.path.isfile(self._weights_path):
            raise FileNotFoundError(
                f"DeepLIIF weights not found at: {self._weights_path}\n"
                "Download from: https://zenodo.org/record/4751737"
            )

        state_dict = torch.load(self._weights_path, map_location='cpu')
        
        # Detect whether this is a 15-channel multi-task generator or a 3-channel single-modality generator (like latest_net_G1.pth)
        # We look at the final convolution weight shape: [out_channels, in_channels, k, k]
        final_layer_key = 'model.26.weight' if 'model.26.weight' in state_dict else list(state_dict.keys())[-2]
        output_nc = state_dict[final_layer_key].shape[0]
        self._output_nc = output_nc

        # Import from the cloned deepliif package (bloated imports have been disabled upstream)
        from deepliif.models.networks import define_G

        model = define_G(
            input_nc=3, output_nc=output_nc, ngf=64,
            netG='resnet_9blocks', norm='batch',
            use_dropout=True, init_type='normal',
            init_gain=0.02, gpu_ids=[],
            padding_type='zero'
        )

        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        model.requires_grad_(False)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"[DeepLIIFStainer] Loaded pretrained generator ({output_nc} output channels): {n_params:,} params")
        print(f"[DeepLIIFStainer] Weights: {self._weights_path}")

        self._model = model
        return model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract_hematoxylin(self, he_rgb):
        """Extract 3-channel virtual Hematoxylin from H&E RGB images.

        Args:
            he_rgb: ``[B, 3, H, W]`` H&E images in ``[-1, 1]`` range.

        Returns:
            hema_rgb: ``[B, 3, H, W]`` virtual Hematoxylin in ``[-1, 1]`` range.
        """
        model = self._load_model(he_rgb.device)

        _, _, H, W = he_rgb.shape
        needs_resize = (H != 512 or W != 512)

        if needs_resize:
            x = F.interpolate(he_rgb, size=(512, 512),
                              mode='bilinear', align_corners=False)
        else:
            x = he_rgb

        # Forward pass
        out = model(x)  # [B, output_nc, 512, 512]

        # Slice channels 0:3 if it's the 15-channel model, otherwise return as is
        if self._output_nc >= 15:
            hema_rgb = out[:, 0:3, :, :]  # [B, 3, 512, 512] in [-1, 1]
        else:
            hema_rgb = out[:, 0:3, :, :]  # G1 is 1 or 3 channels, slice safely

        if needs_resize:
            hema_rgb = F.interpolate(hema_rgb, size=(H, W),
                                     mode='bilinear', align_corners=False)

        return hema_rgb

    def forward(self, he_rgb):
        """Alias for extract_hematoxylin (nn.Module interface)."""
        return self.extract_hematoxylin(he_rgb)
