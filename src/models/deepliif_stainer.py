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

        _, B, H, W = he_rgb.shape[0], he_rgb.shape[1] if len(he_rgb.shape) > 3 else 3, he_rgb.shape[-2], he_rgb.shape[-1]
        
        is_1024 = (H == 1024 and W == 1024)
        needs_resize = (not is_1024) and (H != 512 or W != 512)

        if is_1024:
            # Chunk into 4 quadrants
            top_left = he_rgb[:, :, 0:512, 0:512]
            top_right = he_rgb[:, :, 0:512, 512:1024]
            bottom_left = he_rgb[:, :, 512:1024, 0:512]
            bottom_right = he_rgb[:, :, 512:1024, 512:1024]
            x = torch.cat([top_left, top_right, bottom_left, bottom_right], dim=0)
        elif needs_resize:
            x = F.interpolate(he_rgb, size=(512, 512), mode='bilinear', align_corners=False)
        else:
            x = he_rgb
            
        fake_H = self.nets['G1'](x)

        if is_1024:
            C = fake_H.shape[1]
            out = torch.zeros(1, C, 1024, 1024, device=fake_H.device)
            out[:, :, 0:512, 0:512] = fake_H[0:1]
            out[:, :, 0:512, 512:1024] = fake_H[1:2]
            out[:, :, 512:1024, 0:512] = fake_H[2:3]
            out[:, :, 512:1024, 512:1024] = fake_H[3:4]
            fake_H = out
        elif needs_resize:
            fake_H = F.interpolate(fake_H, size=(H, W), mode='bilinear', align_corners=False)
        
        hema_rgb = fake_H.mean(dim=1, keepdim=True)

        return hema_rgb

    @torch.no_grad()
    def extract_segmentation(self, img_rgb):
        """Extract the clinical segmentation mask using the full ensemble.
        
        Args:
            img_rgb: ``[B, 3, H, W]`` RGB images in ``[-1, 1]`` range.
            
        Returns:
            seg_rgb: ``[B, 3, H, W]`` Segmentation mask in ``[-1, 1]`` range.
        """
        return self.extract_all_modalities(img_rgb)['Segmentation']

    def extract_all_modalities(self, img_rgb):
        """
        Extract all intermediate modalities (G1-G4) and the final segmentation mask.
        Useful for visualization and debugging to see what each part of the ensemble is doing.
        """
        self._load_model(img_rgb.device)
        
        _, B, H, W = img_rgb.shape[0], img_rgb.shape[1] if len(img_rgb.shape) > 3 else 3, img_rgb.shape[-2], img_rgb.shape[-1]
        
        is_1024 = (H == 1024 and W == 1024)
        needs_resize = (not is_1024) and (H != 512 or W != 512)

        if is_1024:
            # Chunk into 4 quadrants (512x512 each) to preserve native 40x magnification
            top_left = img_rgb[:, :, 0:512, 0:512]
            top_right = img_rgb[:, :, 0:512, 512:1024]
            bottom_left = img_rgb[:, :, 512:1024, 0:512]
            bottom_right = img_rgb[:, :, 512:1024, 512:1024]
            # Process them all as a single batch for speed
            x = torch.cat([top_left, top_right, bottom_left, bottom_right], dim=0)
        elif needs_resize:
            x = F.interpolate(img_rgb, size=(512, 512), mode='bilinear', align_corners=False)
        else:
            x = img_rgb
            
        # Run through the ensemble
        fake_B_1 = self.nets['G1'](x) # Hematoxylin
        fake_B_2 = self.nets['G2'](x) # mpIHC DAPI
        fake_B_3 = self.nets['G3'](x) # mpIHC Lap2
        fake_B_4 = self.nets['G4'](x) # mpIHC Ki67
        
        fake_B_5_1 = self.nets['G51'](x)
        fake_B_5_2 = self.nets['G52'](fake_B_1)
        fake_B_5_3 = self.nets['G53'](fake_B_2)
        fake_B_5_4 = self.nets['G54'](fake_B_3)
        fake_B_5_5 = self.nets['G55'](fake_B_4)
        
        seg_rgb = (fake_B_5_1 * 0.2 + fake_B_5_2 * 0.2 + fake_B_5_3 * 0.2 + fake_B_5_4 * 0.2 + fake_B_5_5 * 0.2)

        # Reconstruct or resize back
        if is_1024:
            def stitch(batch_tensor):
                C = batch_tensor.shape[1]
                # Assuming original input batch size is 1 for evaluation
                out = torch.zeros(1, C, 1024, 1024, device=batch_tensor.device)
                out[:, :, 0:512, 0:512] = batch_tensor[0:1]
                out[:, :, 0:512, 512:1024] = batch_tensor[1:2]
                out[:, :, 512:1024, 0:512] = batch_tensor[2:3]
                out[:, :, 512:1024, 512:1024] = batch_tensor[3:4]
                return out
                
            fake_B_1 = stitch(fake_B_1)
            fake_B_2 = stitch(fake_B_2)
            fake_B_3 = stitch(fake_B_3)
            fake_B_4 = stitch(fake_B_4)
            seg_rgb = stitch(seg_rgb)
        elif needs_resize:
            fake_B_1 = F.interpolate(fake_B_1, size=(H, W), mode='bilinear', align_corners=False)
            fake_B_2 = F.interpolate(fake_B_2, size=(H, W), mode='bilinear', align_corners=False)
            fake_B_3 = F.interpolate(fake_B_3, size=(H, W), mode='bilinear', align_corners=False)
            fake_B_4 = F.interpolate(fake_B_4, size=(H, W), mode='bilinear', align_corners=False)
            seg_rgb = F.interpolate(seg_rgb, size=(H, W), mode='nearest')

        return {
            'Hematoxylin': fake_B_1,
            'mpIHC_DAPI': fake_B_2,
            'mpIHC_Lap2': fake_B_3,
            'mpIHC_Ki67': fake_B_4,
            'Segmentation': seg_rgb
        }

    def forward(self, he_rgb):
        """Alias for extract_hematoxylin (nn.Module interface)."""
        return self.extract_hematoxylin(he_rgb)
