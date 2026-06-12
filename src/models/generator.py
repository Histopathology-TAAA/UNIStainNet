"""
SPADEUNetGenerator: H&E → IHC translation generator.

SPADE-UNet conditioned on UNI pathology features + HER2 class embedding.
Encoder processes H&E input, decoder uses SPADE conditioning from UNI features
+ FiLM from class embedding, with skip connections.

~30M params at 512, supports 1024 with extra encoder/decoder levels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def warp_features(x, flow, mask=None):
    """Warps feature map x using the normalized displacement flow and applies a confidence mask.
    
    Args:
        x: [B, C, H, W] features
        flow: [B, 2, H_orig, W_orig] displacement field at any resolution
              (values in normalized [-1, 1] range relative to image size)
        mask: [B, 1, H_orig, W_orig] optional confidence mask (0 to 1)
    Returns:
        warped_x: [B, C, H, W] features shifted by the flow field and optionally masked.
    """
    B, C, H, W = x.shape
    
    # Downsample flow to match feature map resolution if needed
    if flow.shape[-1] != W or flow.shape[-2] != H:
        flow_down = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
    else:
        flow_down = flow

    # Create base grid [-1, 1] for this resolution
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=x.device), 
        torch.linspace(-1, 1, W, device=x.device), 
        indexing='ij'
    )
    base_grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
    base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1) # [B, H, W, 2]
    
    # flow_down is [B, 2, H, W]. Permute to [B, H, W, 2] and add
    warped_grid = base_grid + flow_down.permute(0, 2, 3, 1)
    
    warped_x = F.grid_sample(x, warped_grid, mode='bilinear', padding_mode='reflection', align_corners=True)
    
    if mask is not None:
        if mask.shape[-1] != W or mask.shape[-2] != H:
            mask_down = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)
        else:
            mask_down = mask
        warped_x = warped_x * mask_down
        
    return warped_x


class AlignmentNetwork(nn.Module):
    """Spatial Transformer Network to align features to a target structure.
    
    Predicts a dense optical-flow-like displacement field from a source
    and target image. Final layer is zero-initialized so it starts as
    an identity transformation (zero flow).
    """
    def __init__(self, in_channels=2):
        super().__init__()
        # Input: concat(source_h, target_h) = in_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 5, padding=2, stride=2), # downsample
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # upsample
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, 3, padding=1) # 3 channels: x flow, y flow, confidence mask
        )
        # Zero-initialize the final convolution weights
        nn.init.zeros_(self.net[-1].weight)
        if self.net[-1].bias is not None:
            nn.init.zeros_(self.net[-1].bias)
            # Initialize the confidence mask bias so it starts near 1.0 (sigmoid(3) = 0.95)
            self.net[-1].bias.data[2] = 3.0

    def forward(self, src_h, tgt_h):
        # Scale input H-channels to [0, 1] for better flow processing
        src = (src_h + 1) / 2
        tgt = (tgt_h + 1) / 2
        out = self.net(torch.cat([src, tgt], dim=1))
        
        flow = out[:, :2, :, :]
        mask = torch.sigmoid(out[:, 2:, :, :])
        return flow, mask

from src.models.blocks import SPADEBlock, ResBlock, SelfAttention
from src.models.edge_encoder import EdgeEncoder, MultiScaleEdgeEncoder
from src.models.uni_processor import UNIFeatureProcessor, UNIFeatureProcessorHighRes


class SPADEUNetGenerator(nn.Module):
    """SPADE-UNet generator for H&E → HER2 translation.

    Encoder processes H&E input into multi-scale features.
    Decoder uses SPADE conditioning from UNI features + FiLM from class embedding.
    Skip connections from encoder to decoder.

    Args:
        hema_channels: Number of channels for the hematoxylin conditioning input.
            1 = legacy analytical H-channel (1ch).
            3 = DeepLIIF virtual Hematoxylin (3ch RGB).
            Main encoder input becomes 3 (H&E RGB) + hema_channels.

    ~30M params.
    """

    def __init__(self, num_classes=5, class_dim=64, uni_dim=1024,
                 input_skip=False, edge_encoder=False, edge_base_ch=32,
                 uni_spatial_size=4, image_size=512, uni_spade_at_512=False,
                 use_alignment=False, learnable_sobel=False, hema_channels=1):
        super().__init__()
        self.num_classes = num_classes
        self.class_dim = class_dim
        self.input_skip = input_skip
        self.edge_encoder_flag = edge_encoder
        self.uni_spatial_size = uni_spatial_size
        self.image_size = image_size
        self.uni_spade_at_512 = uni_spade_at_512
        self.use_alignment = use_alignment
        self.hema_channels = hema_channels

        # Class embedding (5 classes: 0, 1+, 2+, 3+, null)
        self.class_embed = nn.Embedding(num_classes, class_dim)

        # UNI feature processor — choose based on spatial resolution
        if uni_spatial_size >= 16:
            # High-res patch tokens (e.g., 32x32 = 1024 tokens)
            self.uni_processor = UNIFeatureProcessorHighRes(
                uni_dim=uni_dim, base_channels=512, spatial_size=uni_spatial_size,
                output_512=(uni_spade_at_512 and image_size == 1024),
            )
        else:
            # Original CLS-token features (4x4 = 16 tokens)
            self.uni_processor = UNIFeatureProcessor(
                uni_dim=uni_dim, base_channels=512,
            )

        # Edge encoder (parallel structure pathway)
        # Note: edge encoder always operates at 512 resolution.
        # For 1024 input, H&E is downsampled to 512 before edge extraction.
        self.edge_encoder_type = edge_encoder  # False, 'v1', or 'v2'
        if edge_encoder == 'v2':
            self.edge_encoder = MultiScaleEdgeEncoder(
                base_ch=edge_base_ch, learnable_sobel=learnable_sobel,
                input_channels=hema_channels,
            )
            edge_ch = {512: edge_base_ch, 256: edge_base_ch, 128: edge_base_ch * 2,
                       64: edge_base_ch * 4, 32: edge_base_ch * 4}
        elif edge_encoder:  # True or 'v1'
            self.edge_encoder = EdgeEncoder(
                base_ch=edge_base_ch, learnable_sobel=learnable_sobel,
                input_channels=hema_channels,
            )
            edge_ch = {512: 0, 256: edge_base_ch, 128: edge_base_ch * 2,
                       64: edge_base_ch * 4, 32: edge_base_ch * 4}
        else:
            self.edge_encoder = None
            edge_ch = {512: 0, 256: 0, 128: 0, 64: 0, 32: 0}

        # === 1024 support: extra encoder/decoder levels ===
        # Encoder input: 3ch H&E RGB + hema_channels (1ch H-channel or 3ch DeepLIIF)
        encoder_in_ch = 3 + hema_channels  # 4 (legacy) or 6 (DeepLIIF)
        if image_size == 1024:
            # enc0: 1024→512 (lightweight, just spatial downsample)
            self.enc0 = nn.Sequential(
                nn.Conv2d(encoder_in_ch, 32, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )
            enc1_in_ch = 32  # enc1 takes enc0 output, not raw input
        else:
            self.enc0 = None
            enc1_in_ch = encoder_in_ch  # enc1 takes raw input at 512

        # Encoder
        self.enc1 = nn.Sequential(  # 512→256
            nn.Conv2d(enc1_in_ch, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(  # 256→128
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc3 = nn.Sequential(  # 128→64
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc4 = nn.Sequential(  # 64→32
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc5 = nn.Sequential(  # 32→16
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Bottleneck (at 16×16)
        self.bottleneck = nn.Sequential(
            ResBlock(512),
            SelfAttention(512),
            ResBlock(512),
        )

        # Decoder with SPADE conditioning
        # Channel counts: main_skip + edge_skip (if enabled) + upsampled
        # D5: 512 (up) + 512 (skip e4) + edge_ch[32] → 512
        self.dec5_conv = nn.Conv2d(512 + 512 + edge_ch[32], 512, 3, padding=1)
        self.dec5_spade = SPADEBlock(512, uni_channels=512, class_dim=class_dim)
        self.dec5_act = nn.LeakyReLU(0.2, inplace=True)

        # D4: 512 (up) + 256 (skip e3) + edge_ch[64] → 256
        self.dec4_conv = nn.Conv2d(512 + 256 + edge_ch[64], 256, 3, padding=1)
        self.dec4_spade = SPADEBlock(256, uni_channels=256, class_dim=class_dim)
        self.dec4_act = nn.LeakyReLU(0.2, inplace=True)

        # D3: 256 (up) + 128 (skip e2) + edge_ch[128] → 128
        self.dec3_conv = nn.Conv2d(256 + 128 + edge_ch[128], 128, 3, padding=1)
        self.dec3_spade = SPADEBlock(128, uni_channels=128, class_dim=class_dim)
        self.dec3_act = nn.LeakyReLU(0.2, inplace=True)

        # D2: 128 (up) + 64 (skip e1) + edge_ch[256] → 64
        self.dec2_conv = nn.Conv2d(128 + 64 + edge_ch[256], 64, 3, padding=1)
        self.dec2_spade = SPADEBlock(64, uni_channels=64, class_dim=class_dim)
        self.dec2_act = nn.LeakyReLU(0.2, inplace=True)

        if image_size == 1024:
            # D1 (new): upsample 256→512, skip from enc0 (32ch) + edge@512
            dec1_in_ch = 64 + 32 + edge_ch[512]
            if uni_spade_at_512:
                # UNI SPADE conditioning at 512 level (uni_ch=32 at this scale)
                self.dec1_conv = nn.Conv2d(dec1_in_ch, 64, 3, padding=1)
                self.dec1_spade = SPADEBlock(64, uni_channels=32, class_dim=class_dim)
                self.dec1_act = nn.LeakyReLU(0.2, inplace=True)
            else:
                self.dec1_conv = nn.Sequential(
                    nn.Conv2d(dec1_in_ch, 64, 3, padding=1),
                    nn.InstanceNorm2d(64),
                    nn.LeakyReLU(0.2, inplace=True),
                )
                self.dec1_spade = None
                self.dec1_act = None
            # Output: upsample 512→1024, optional H&E input skip
            output_in_ch = 64 + (3 if input_skip else 0)
            self.output = nn.Sequential(
                nn.Conv2d(output_in_ch, 64, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 3, 3, padding=1),
                nn.Tanh(),
            )
        else:
            self.dec1_conv = None
            # Output: concat H&E input (3ch if input_skip) + edge@512 (if v2)
            output_in_ch = 64 + (3 if input_skip else 0) + edge_ch[512]
            self.output = nn.Sequential(
                nn.Conv2d(output_in_ch, 64, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 3, 3, padding=1),
                nn.Tanh(),
            )

        if self.use_alignment:
            self.alignment_net = AlignmentNetwork(in_channels=self.hema_channels * 2)

    def encode(self, images, h_channel=None):
        """Extract intermediate encoder features for PatchNCE loss.

        Args:
            images: [B, 3, H, H] in [-1, 1] (H&E or generated IHC)
            h_channel: [B, C, H, H] in [-1, 1] or None.
                       C=1 (legacy H-channel) or C=3 (DeepLIIF Hematoxylin).
                       If None, grayscale fallback (expanded to hema_channels).

        Returns:
            dict mapping layer index to feature tensor:
                {1: [B, 64, 256, 256], 2: [B, 128, 128, 128],
                 3: [B, 256, 64, 64], 4: [B, 512, 32, 32]}
        """
        if h_channel is None:
            gray = images.mean(dim=1, keepdim=True)  # [B, 1, H, H]
            h_channel = gray.expand(-1, self.hema_channels, -1, -1)
        encoder_input = torch.cat([images, h_channel], dim=1)  # [B, 3+hema_ch, H, H]

        if self.enc0 is not None:
            e0 = self.enc0(encoder_input)
            enc1_input = e0
        else:
            enc1_input = encoder_input

        e1 = self.enc1(enc1_input)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        return {1: e1, 2: e2, 3: e3, 4: e4}

    def forward(self, he_images, uni_features, labels, edge_input=None, he_h=None, h_channel=None):
        """
        Args:
            he_images:    [B, 3, H, H] in [-1, 1] where H=512 or H=1024
            uni_features: [B, N, 1024] where N=16 (4x4 CLS) or N=1024 (32x32 patch)
            labels:       [B] int class labels (0-4)
            edge_input:   [B, 1, H, H] in [-1, 1] or None. Target H-channel for edge encoder.
            he_h:         [B, 1, H, H] in [-1, 1] or None. Source H&E H-channel for alignment.
            h_channel:    [B, 1, H, H] in [-1, 1] or None. H-channel concatenated to encoder input.
                          If None, grayscale of he_images is used as fallback.

        Returns:
            output: [B, 3, H, H] in [-1, 1]
        """
        class_emb = self.class_embed(labels)
        uni_maps = self.uni_processor(uni_features)

        # Edge encoder (parallel structure pathway).
        # Always receives a 1-ch H-channel tensor [B, 1, H, H].
        # edge_input (provided by trainer): either IHC H-channel (Case A) or
        # H&E H-channel (Case B). When None (legacy), grayscale of he_images is used.
        if self.edge_encoder_type:
            if edge_input is not None:
                edge_src = edge_input  # [B, 1, H, H] — pass directly
            else:
                # Fallback: derive 1-ch grayscale from H&E RGB
                edge_src = he_images.mean(dim=1, keepdim=True)  # [B, 1, H, H]

            if self.image_size == 1024:
                edge_src_512 = F.interpolate(edge_src, size=512, mode='bilinear', align_corners=False)
                edge_maps = self.edge_encoder(edge_src_512)
            else:
                edge_maps = self.edge_encoder(edge_src)
        else:
            edge_maps = None

        # Build encoder input: [B, 3+hema_ch, H, H]
        if h_channel is None:
            gray = he_images.mean(dim=1, keepdim=True)
            h_channel = gray.expand(-1, self.hema_channels, -1, -1)
        encoder_input = torch.cat([he_images, h_channel], dim=1)  # [B, 3+hema_ch, H, H]

        # === 1024: extra encoder level ===
        if self.enc0 is not None:
            e0 = self.enc0(encoder_input)   # [B, 32, 512, 512]
            enc1_input = e0
        else:
            e0 = None
            enc1_input = encoder_input

        # Encoder
        e1 = self.enc1(enc1_input)  # [B, 64, 256, 256]
        e2 = self.enc2(e1)          # [B, 128, 128, 128]
        e3 = self.enc3(e2)          # [B, 256, 64, 64]
        e4 = self.enc4(e3)          # [B, 512, 32, 32]
        e5 = self.enc5(e4)          # [B, 512, 16, 16]

        # Alignment Network (STN) Feature Warping
        if self.use_alignment and hasattr(self, 'alignment_net') and he_h is not None and edge_input is not None:
            # Predict flow from he_h (source) to edge_input (target ihc_h or he_h)
            flow_field, confidence_mask = self.alignment_net(he_h, edge_input)
            
            # Warp main encoder features
            if e0 is not None:
                e0 = warp_features(e0, flow_field, confidence_mask)
            e1 = warp_features(e1, flow_field, confidence_mask)
            e2 = warp_features(e2, flow_field, confidence_mask)
            e3 = warp_features(e3, flow_field, confidence_mask)
            e4 = warp_features(e4, flow_field, confidence_mask)
            e5 = warp_features(e5, flow_field, confidence_mask)
            
            # Warp input images for skip connection
            if self.input_skip:
                he_images = warp_features(he_images, flow_field, confidence_mask)
            
            # Warp UNI semantic features
            # uni_maps is a dict {32: [B, ch, 32, 32], 64: [B, ch, 64, 64], ...}
            for k in uni_maps.keys():
                uni_maps[k] = warp_features(uni_maps[k], flow_field, confidence_mask)

        # Bottleneck at 16×16
        x = self.bottleneck(e5)     # [B, 512, 16, 16]

        # D5: upsample 16→32, skip from e4 + edge@32, UNI at 32
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        skip5 = [x, e4] + ([edge_maps[32]] if edge_maps else [])
        x = torch.cat(skip5, dim=1)
        x = self.dec5_conv(x)
        x = self.dec5_spade(x, uni_maps[32], class_emb)
        x = self.dec5_act(x)

        # D4: upsample 32→64, skip from e3 + edge@64, UNI at 64
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        skip4 = [x, e3] + ([edge_maps[64]] if edge_maps else [])
        x = torch.cat(skip4, dim=1)
        x = self.dec4_conv(x)
        x = self.dec4_spade(x, uni_maps[64], class_emb)
        x = self.dec4_act(x)

        # D3: upsample 64→128, skip from e2 + edge@128, UNI at 128
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        skip3 = [x, e2] + ([edge_maps[128]] if edge_maps else [])
        x = torch.cat(skip3, dim=1)
        x = self.dec3_conv(x)
        x = self.dec3_spade(x, uni_maps[128], class_emb)
        x = self.dec3_act(x)

        # D2: upsample 128→256, skip from e1 + edge@256, UNI at 256
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        skip2 = [x, e1] + ([edge_maps[256]] if edge_maps else [])
        x = torch.cat(skip2, dim=1)
        x = self.dec2_conv(x)
        x = self.dec2_spade(x, uni_maps[256], class_emb)
        x = self.dec2_act(x)

        if self.image_size == 1024:
            # D1: upsample 256→512, skip from e0 (32ch) + edge@512
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            skip1 = [x, e0] + ([edge_maps[512]] if edge_maps else [])
            x = torch.cat(skip1, dim=1)
            x = self.dec1_conv(x)
            if self.dec1_spade is not None:
                x = self.dec1_spade(x, uni_maps[512], class_emb)
                x = self.dec1_act(x)
            # [B, 64, 512, 512]

            # Output: upsample 512→1024, optional H&E input skip
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            if self.input_skip:
                x = torch.cat([x, he_images], dim=1)
            x = self.output(x)  # [B, 3, 1024, 1024]
        else:
            # D1: upsample 256→512, optional skip from H&E input + edge@512
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            skip1 = [x]
            if self.input_skip:
                skip1.append(he_images)
            if edge_maps and 512 in edge_maps:
                skip1.append(edge_maps[512])
            x = torch.cat(skip1, dim=1) if len(skip1) > 1 else x
            x = self.output(x)  # [B, 3, 512, 512]

        return x
