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
        """Instantiate and load the full DeepLIIF pipeline using native API."""
        if self._model is not None:
            return
            
        if os.path.isfile(self._weights_path):
            self.model_dir = os.path.dirname(self._weights_path)
        else:
            self.model_dir = self._weights_path

        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"DeepLIIF model directory not found at {self.model_dir}")

        # Disable bloating stdout from DeepLIIF
        import sys
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            from deepliif.models import init_nets, get_opt
            self.opt = get_opt(self.model_dir, mode='test')
            self.opt.gpu_ids = [0] if device.type == 'cuda' else []
            self.nets = init_nets(self.model_dir, eager_mode=True, opt=self.opt)
            self._model = True
        finally:
            # Restore stdout
            sys.stdout = old_stdout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract_hematoxylin(self, he_rgb):
        """Extract the virtual Hematoxylin channel.
        
        Args:
            he_rgb: ``[B, 3, H, W]`` RGB images in ``[-1, 1]`` range.
            
        Returns:
            hema_rgb: ``[B, 1, H, W]`` virtual Hematoxylin in ``[-1, 1]`` range.
        """
        self._load_model(he_rgb.device)

        _, _, H, W = he_rgb.shape
        needs_resize = (H != 512 or W != 512)

        if needs_resize:
            x = F.interpolate(he_rgb, size=(512, 512),
                              mode='bilinear', align_corners=False)
        else:
            x = he_rgb
            
        hema_rgb = self.nets['G1'](x)  # [B, 3, 512, 512]

        # Collapse to 1-channel structural density
        hema_rgb = hema_rgb.mean(dim=1, keepdim=True)

        if needs_resize:
            hema_rgb = F.interpolate(hema_rgb, size=(H, W),
                                     mode='bilinear', align_corners=False)

        return hema_rgb

    @torch.no_grad()
    def extract_segmentation(self, img_rgb):
        """Extract the clinical segmentation mask using the full ensemble.
        
        Args:
            img_rgb: ``[B, 3, H, W]`` RGB images in ``[-1, 1]`` range.
            
        Returns:
            seg_rgb: ``[B, 3, H, W]`` Segmentation mask in ``[-1, 1]`` range.
        """
        self._load_model(img_rgb.device)
        
        _, _, H, W = img_rgb.shape
        needs_resize = (H != 512 or W != 512)

        if needs_resize:
            x = F.interpolate(img_rgb, size=(512, 512),
                              mode='bilinear', align_corners=False)
        else:
            x = img_rgb
        # The full DeepLIIF pipeline uses a 5-model ensemble for segmentation
        fake_B_1 = self.nets['G1'](x) # Hematoxylin
        fake_B_2 = self.nets['G2'](x) # DAPI
        fake_B_3 = self.nets['G3'](x) # Lap2
        fake_B_4 = self.nets['G4'](x) # Ki67
        
        # Segmentation masks from each modality
        fake_B_5_1 = self.nets['G51'](x)        # From original IHC
        fake_B_5_2 = self.nets['G52'](fake_B_1) # From Hematoxylin
        fake_B_5_3 = self.nets['G53'](fake_B_2) # From DAPI
        fake_B_5_4 = self.nets['G54'](fake_B_3) # From Lap2
        fake_B_5_5 = self.nets['G55'](fake_B_4) # From Ki67
        
        # Average the masks (DeepLIIF defaults to uniform 0.2 weights)
        seg_rgb = (fake_B_5_1 * 0.2 + 
                   fake_B_5_2 * 0.2 + 
                   fake_B_5_3 * 0.2 + 
                   fake_B_5_4 * 0.2 + 
                   fake_B_5_5 * 0.2)

        if needs_resize:
            # Nearest neighbor for segmentation masks
            seg_rgb = F.interpolate(seg_rgb, size=(H, W),
                                     mode='nearest')

        return seg_rgb

    def forward(self, he_rgb):
        """Alias for extract_hematoxylin (nn.Module interface)."""
        return self.extract_hematoxylin(he_rgb)
