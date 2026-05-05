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

from src.models.blocks import CrossAttention, ResBlock, SelfAttention
from src.models.edge_encoder import EdgeEncoder, MultiScaleEdgeEncoder
from src.models.uni_processor import UNIFeatureProcessor, UNIFeatureProcessorHighRes


class SPADEUNetGenerator(nn.Module):
    """Structure-Semantic Decoupled UNet generator.

    Encoder processes 1-channel structural input (H-map).
    Decoder uses Cross-Attention with UNI tokens (semantic) as K/V.
    Skip connections from encoder to decoder.
    """

    def __init__(self, num_classes=5, class_dim=64, uni_dim=1024,
                 input_skip=False, edge_encoder=False, edge_base_ch=32,
                 uni_spatial_size=4, image_size=512, uni_spade_at_512=False):
        super().__init__()
        self.num_classes = num_classes
        self.class_dim = class_dim
        self.input_skip = input_skip
        self.edge_encoder_flag = edge_encoder
        self.uni_spatial_size = uni_spatial_size
        self.image_size = image_size
        self.uni_spade_at_512 = uni_spade_at_512

        # Class embedding (kept for compatibility, currently unused)
        self.class_embed = nn.Embedding(num_classes, class_dim)

        # UNI processor kept for backward compatibility (not used with cross-attn)
        if uni_spatial_size >= 16:
            self.uni_processor = UNIFeatureProcessorHighRes(
                uni_dim=uni_dim, base_channels=512, spatial_size=uni_spatial_size,
                output_512=(uni_spade_at_512 and image_size == 1024),
            )
        else:
            self.uni_processor = UNIFeatureProcessor(
                uni_dim=uni_dim, base_channels=512,
            )

        # Edge encoder (parallel structure pathway)
        # Note: edge encoder always operates at 512 resolution.
        # For 1024 input, H&E is downsampled to 512 before edge extraction.
        self.edge_encoder_type = edge_encoder  # False, 'v1', or 'v2'
        if edge_encoder == 'v2':
            self.edge_encoder = MultiScaleEdgeEncoder(base_ch=edge_base_ch)
            edge_ch = {512: edge_base_ch, 256: edge_base_ch, 128: edge_base_ch * 2,
                       64: edge_base_ch * 4, 32: edge_base_ch * 4}
        elif edge_encoder:  # True or 'v1'
            self.edge_encoder = EdgeEncoder(base_ch=edge_base_ch)
            edge_ch = {512: 0, 256: edge_base_ch, 128: edge_base_ch * 2,
                       64: edge_base_ch * 4, 32: edge_base_ch * 4}
        else:
            self.edge_encoder = None
            edge_ch = {512: 0, 256: 0, 128: 0, 64: 0, 32: 0}

        # === 1024 support: extra encoder/decoder levels ===
        if image_size == 1024:
            # enc0: 1024→512 (lightweight, just spatial downsample)
            self.enc0 = nn.Sequential(
                nn.Conv2d(1, 32, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )
            enc1_in_ch = 32  # enc1 takes enc0 output, not raw H&E
        else:
            self.enc0 = None
            enc1_in_ch = 1  # enc1 takes raw H-map at 512

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

        # Decoder with cross-attention conditioning
        # Channel counts: main_skip + edge_skip (if enabled) + upsampled
        # D5: 512 (up) + 512 (skip e4) + edge_ch[32] → 512
        self.dec5_conv = nn.Conv2d(512 + 512 + edge_ch[32], 512, 3, padding=1)
        self.dec5_attn = CrossAttention(512, uni_dim=uni_dim, heads=4)
        self.dec5_act = nn.LeakyReLU(0.2, inplace=True)

        # D4: 512 (up) + 256 (skip e3) + edge_ch[64] → 256
        self.dec4_conv = nn.Conv2d(512 + 256 + edge_ch[64], 256, 3, padding=1)
        self.dec4_attn = CrossAttention(256, uni_dim=uni_dim, heads=4)
        self.dec4_act = nn.LeakyReLU(0.2, inplace=True)

        # D3: 256 (up) + 128 (skip e2) + edge_ch[128] → 128
        self.dec3_conv = nn.Conv2d(256 + 128 + edge_ch[128], 128, 3, padding=1)
        self.dec3_attn = CrossAttention(128, uni_dim=uni_dim, heads=4)
        self.dec3_act = nn.LeakyReLU(0.2, inplace=True)

        # D2: 128 (up) + 64 (skip e1) + edge_ch[256] → 64
        self.dec2_conv = nn.Conv2d(128 + 64 + edge_ch[256], 64, 3, padding=1)
        self.dec2_attn = CrossAttention(64, uni_dim=uni_dim, heads=4)
        self.dec2_act = nn.LeakyReLU(0.2, inplace=True)

        if image_size == 1024:
            # D1 (new): upsample 256→512, skip from enc0 (32ch) + edge@512
            dec1_in_ch = 64 + 32 + edge_ch[512]
            self.dec1_conv = nn.Sequential(
                nn.Conv2d(dec1_in_ch, 64, 3, padding=1),
                nn.InstanceNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.dec1_attn = CrossAttention(64, uni_dim=uni_dim, heads=4)
            self.dec1_act = nn.LeakyReLU(0.2, inplace=True)
            # Output: upsample 512→1024, optional H-map input skip
            output_in_ch = 64 + (1 if input_skip else 0)
            self.output = nn.Sequential(
                nn.Conv2d(output_in_ch, 64, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 3, 3, padding=1),
                nn.Tanh(),
            )
        else:
            self.dec1_conv = None
            # Output: concat H-map input (1ch if input_skip) + edge@512 (if v2)
            output_in_ch = 64 + (1 if input_skip else 0) + edge_ch[512]
            self.output = nn.Sequential(
                nn.Conv2d(output_in_ch, 64, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 3, 3, padding=1),
                nn.Tanh(),
            )

    def encode(self, images):
        """Extract intermediate encoder features for PatchNCE loss."""
        if self.enc0 is not None:
            e0 = self.enc0(images)
            enc1_input = e0
        else:
            enc1_input = images

        e1 = self.enc1(enc1_input)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        return {1: e1, 2: e2, 3: e3, 4: e4}

    def forward(self, h_maps, uni_features, labels):
        """
        Args:
            h_maps: [B, 1, H, H] in [-1, 1] where H=512 or H=1024
            uni_features: [B, N, 1024] where N=16 (4x4 CLS) or N=1024 (32x32 patch)
            labels: [B] int class labels (0-4)

        Returns:
            output: [B, 3, H, H] in [-1, 1]
        """
        _ = self.class_embed(labels)

        # Edge encoder (parallel structure pathway)
        # Edge encoder always operates at 512 resolution.
        # Both v1 and v2 now accept 1-channel H-map natively — no repeat needed.
        if self.edge_encoder_type:
            if self.image_size == 1024:
                edge_input = F.interpolate(h_maps, size=512, mode='bilinear', align_corners=False)
            else:
                edge_input = h_maps
            edge_maps = self.edge_encoder(edge_input)
        else:
            edge_maps = None

        # === 1024: extra encoder level ===
        if self.enc0 is not None:
            e0 = self.enc0(h_maps)   # [B, 32, 512, 512]
            enc1_input = e0
        else:
            e0 = None
            enc1_input = h_maps

        # Encoder
        e1 = self.enc1(enc1_input)  # [B, 64, 256, 256]
        e2 = self.enc2(e1)          # [B, 128, 128, 128]
        e3 = self.enc3(e2)          # [B, 256, 64, 64]
        e4 = self.enc4(e3)          # [B, 512, 32, 32]
        e5 = self.enc5(e4)          # [B, 512, 16, 16]

        # Bottleneck at 16×16
        x = self.bottleneck(e5)     # [B, 512, 16, 16]

        # D5: upsample 16→32, skip from e4 + edge@32, UNI at 32
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        skip5 = [x, e4] + ([edge_maps[32]] if edge_maps else [])
        x = torch.cat(skip5, dim=1)
        x = self.dec5_conv(x)
        x = self.dec5_attn(x, uni_features)
        x = self.dec5_act(x)

        # D4: upsample 32→64, skip from e3 + edge@64, UNI at 64
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        skip4 = [x, e3] + ([edge_maps[64]] if edge_maps else [])
        x = torch.cat(skip4, dim=1)
        x = self.dec4_conv(x)
        x = self.dec4_attn(x, uni_features)
        x = self.dec4_act(x)

        # D3: upsample 64→128, skip from e2 + edge@128, UNI at 128
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        skip3 = [x, e2] + ([edge_maps[128]] if edge_maps else [])
        x = torch.cat(skip3, dim=1)
        x = self.dec3_conv(x)
        x = self.dec3_attn(x, uni_features)
        x = self.dec3_act(x)

        # D2: upsample 128→256, skip from e1 + edge@256, UNI at 256
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        skip2 = [x, e1] + ([edge_maps[256]] if edge_maps else [])
        x = torch.cat(skip2, dim=1)
        x = self.dec2_conv(x)
        x = self.dec2_attn(x, uni_features)
        x = self.dec2_act(x)

        if self.image_size == 1024:
            # D1: upsample 256→512, skip from e0 (32ch) + edge@512
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            skip1 = [x, e0] + ([edge_maps[512]] if edge_maps else [])
            x = torch.cat(skip1, dim=1)
            x = self.dec1_conv(x)
            x = self.dec1_attn(x, uni_features)
            x = self.dec1_act(x)
            # [B, 64, 512, 512]

            # Output: upsample 512→1024, optional H&E input skip
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            if self.input_skip:
                x = torch.cat([x, h_maps], dim=1)
            x = self.output(x)  # [B, 3, 1024, 1024]
        else:
            # D1: upsample 256→512, optional skip from H&E input + edge@512
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            skip1 = [x]
            if self.input_skip:
                skip1.append(h_maps)
            if edge_maps and 512 in edge_maps:
                skip1.append(edge_maps[512])
            x = torch.cat(skip1, dim=1) if len(skip1) > 1 else x
            x = self.output(x)  # [B, 3, 512, 512]

        return x
