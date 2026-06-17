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
            return self._model.to(device)

        if not os.path.exists(self._weights_path):
            raise FileNotFoundError(f"DeepLIIF weights not found at {self._weights_path}")

        # Disable bloating stdout from DeepLIIF
        import sys
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            from deepliif.models.networks import define_G
            state_dict = torch.load(self._weights_path, map_location=device)
            
            # Find output_nc by looking at the last convolution weight
            # U-Net usually ends with model.model.8.weight or similar, which is ConvTranspose2d
            # Conv2d weights: [out, in, k, k]
            # ConvTranspose2d weights: [in, out, k, k]
            weight_keys = [k for k in state_dict.keys() if k.endswith('.weight')]
            final_layer_key = weight_keys[-1] if weight_keys else list(state_dict.keys())[-2]
            shape = state_dict[final_layer_key].shape
            output_nc = shape[0] if shape[0] in [1, 3, 15] else shape[1]
            self._output_nc = output_nc

            keys_str = " ".join(state_dict.keys())
            if "model.model.1.model" in keys_str:
                # Highly nested structure implies U-Net
                candidates = ['unet_512_attention', 'unet_512', 'unet_256', 'unet_128']
            else:
                candidates = ['resnet_9blocks', 'resnet_6blocks']

            model = None
            last_err = None
            for netG in candidates:
                try:
                    temp_model = define_G(
                        input_nc=3, output_nc=output_nc, ngf=64,
                        netG=netG, norm='batch',
                        use_dropout=True, init_type='normal', init_gain=0.02,
                        gpu_ids=[], padding_type='zero'
                    )
                    temp_model.load_state_dict(state_dict)
                    model = temp_model
                    break
                except Exception as e:
                    last_err = e
                    continue
            
            if model is None:
                raise RuntimeError(f"Could not load state_dict with any candidate architecture. Last Error: {last_err}")

            model.eval()
            model.to(device)
            
        finally:
            # Restore stdout
            sys.stdout = old_stdout

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
        hema_rgb = out[:, 0:3, :, :]  # [B, 3, 512, 512] in [-1, 1]

        # Collapse to 1-channel structural density to match hema_channels=1
        hema_rgb = hema_rgb.mean(dim=1, keepdim=True)

        if needs_resize:
            hema_rgb = F.interpolate(hema_rgb, size=(H, W),
                                     mode='bilinear', align_corners=False)

        return hema_rgb

    @torch.no_grad()
    def extract_segmentation(self, img_rgb):
        """Extract the clinical segmentation mask from the 15-channel output.
        
        Args:
            img_rgb: ``[B, 3, H, W]`` RGB images in ``[-1, 1]`` range.
            
        Returns:
            seg_rgb: ``[B, 3, H, W]`` Segmentation mask in ``[-1, 1]`` range.
        """
        model = self._load_model(img_rgb.device)
        
        _, _, H, W = img_rgb.shape
        needs_resize = (H != 512 or W != 512)

        if needs_resize:
            x = F.interpolate(img_rgb, size=(512, 512),
                              mode='bilinear', align_corners=False)
        else:
            x = img_rgb

        # Forward pass
        out = model(x)  # [B, output_nc, 512, 512]
        
        if self._output_nc == 15:
            # 15-channel combined model
            seg_rgb = out[:, 12:15, :, :]
        else:
            # 3-channel separated model (e.g. latest_net_G4.pth)
            seg_rgb = out[:, 0:3, :, :]

        if needs_resize:
            # Nearest neighbor for segmentation masks
            seg_rgb = F.interpolate(seg_rgb, size=(H, W),
                                     mode='nearest')

        return seg_rgb

    def forward(self, he_rgb):
        """Alias for extract_hematoxylin (nn.Module interface)."""
        return self.extract_hematoxylin(he_rgb)
