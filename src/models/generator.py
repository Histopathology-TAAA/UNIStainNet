"""
SPADEUNetGenerator: H&E → IHC translation generator.

SPADE-UNet conditioned on UNI pathology features + HER2 class embedding.
Encoder processes H-map input, decoder uses Cross-Attention over UNI tokens
+ SPADE + FiLM conditioning from UNI spatial maps and stain embedding, with
skip connections.

~30M params at 512, supports 1024 with extra encoder/decoder levels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks import CrossAttention, ResBlock, SelfAttention, SPADEBlock
from src.models.edge_encoder import EdgeEncoder, EosinEncoder, MultiScaleEdgeEncoder
from src.models.uni_processor import UNIFeatureProcessor, UNIFeatureProcessorHighRes


class SPADEUNetGenerator(nn.Module):
    """Structure-Semantic Decoupled UNet generator.

    Encoder processes 1-channel structural input (H-map).
    Decoder uses Cross-Attention with UNI tokens (semantic) as K/V and
    SPADE + FiLM from UNI spatial maps and stain labels.
    Skip connections from encoder to decoder.
    """

    def __init__(self, num_classes=5, class_dim=64, uni_dim=1024,
                 input_skip=False, edge_encoder=False, edge_base_ch=32,
                 uni_spatial_size=4, image_size=512, uni_spade_at_512=False,
                 use_eosin_encoder=False, eosin_out_ch=64, eosin_multi_scale=False,
                 use_attention_for_spade=False, enable_attention_residual=True,
                 spade_use_uni=True):
        super().__init__()
        self.num_classes = num_classes
        self.class_dim = class_dim
        self.input_skip = input_skip
        self.edge_encoder_flag = edge_encoder
        self.uni_spatial_size = uni_spatial_size
        self.image_size = image_size
        self.uni_spade_at_512 = uni_spade_at_512
        self.eosin_multi_scale = eosin_multi_scale
        self.use_attention_for_spade = use_attention_for_spade
        self.enable_attention_residual = enable_attention_residual
        self.spade_use_uni = spade_use_uni

        # Class embedding used by FiLM in the decoder.
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

        # Eosin bottleneck encoder: H&E E-map → [B, eosin_out_ch, 16, 16]
        # Injected after bottleneck. Misalignment-safe: ~30px slice drift → <1px at 16×16.
        #
        # eosin_multi_scale=True (Option B — new runs only):
        #   Also injects Eosin 32×32 features at decoder level D5 (after first upsample).
        #   At 32×32 the 30px misalignment is ~1px — still safely negligible.
        #   Gives the decoder a finer membrane signal than the bottleneck alone.
        #   Enable via --eosin_multi_scale. Incompatible with checkpoints trained
        #   without it (state_dict keys differ). Default: False.
        if use_eosin_encoder:
            self.eosin_encoder = EosinEncoder(out_channels=eosin_out_ch,
                                              multi_scale=eosin_multi_scale)
            self.eosin_proj = nn.Conv2d(512 + eosin_out_ch, 512, 1)
            if eosin_multi_scale:
                # stage4 always outputs 64ch regardless of eosin_out_ch
                self.eosin_proj_32 = nn.Conv2d(512 + 64, 512, 1)
            else:
                self.eosin_proj_32 = None
        else:
            self.eosin_encoder = None
            self.eosin_proj = None
            self.eosin_proj_32 = None

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
        self.dec5_spade = SPADEBlock(512, uni_channels=512, class_dim=class_dim)
        self.dec5_act = nn.LeakyReLU(0.2, inplace=True)

        # D4: 512 (up) + 256 (skip e3) + edge_ch[64] → 256
        self.dec4_conv = nn.Conv2d(512 + 256 + edge_ch[64], 256, 3, padding=1)
        self.dec4_attn = CrossAttention(256, uni_dim=uni_dim, heads=4)
        self.dec4_spade = SPADEBlock(256, uni_channels=256, class_dim=class_dim)
        self.dec4_act = nn.LeakyReLU(0.2, inplace=True)

        # D3: 256 (up) + 128 (skip e2) + edge_ch[128] → 128
        self.dec3_conv = nn.Conv2d(256 + 128 + edge_ch[128], 128, 3, padding=1)
        self.dec3_attn = CrossAttention(128, uni_dim=uni_dim, heads=4)
        self.dec3_spade = SPADEBlock(128, uni_channels=128, class_dim=class_dim)
        self.dec3_act = nn.LeakyReLU(0.2, inplace=True)

        # D2: 128 (up) + 64 (skip e1) + edge_ch[256] → 64
        self.dec2_conv = nn.Conv2d(128 + 64 + edge_ch[256], 64, 3, padding=1)
        self.dec2_attn = CrossAttention(64, uni_dim=uni_dim, heads=4)
        self.dec2_spade = SPADEBlock(64, uni_channels=64, class_dim=class_dim)
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
            self.dec1_spade = SPADEBlock(64, uni_channels=32, class_dim=class_dim)
            self.dec1_attn_to_spade = nn.Conv2d(64, 32, 1)
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
            self.dec1_spade = None
            self.dec1_attn_to_spade = None
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

    def forward(self, h_maps, uni_features, labels, e_maps=None):
        """
        Args:
            h_maps: [B, 1, H, H] in [-1, 1] where H=512 or H=1024
            uni_features: [B, N, 1024] where N=16 (4x4 CLS) or N=1024 (32x32 patch)
            labels: [B] int class labels (0-4)

        Returns:
            output: [B, 3, H, H] in [-1, 1]
        """
        class_emb = self.class_embed(labels)
        uni_maps = self.uni_processor(uni_features)

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

        # Eosin injection at bottleneck (16×16): membrane topology from H&E E-map
        e_feat_32 = None
        if self.eosin_encoder is not None and e_maps is not None:
            eosin_out = self.eosin_encoder(e_maps)
            if self.eosin_multi_scale:
                e_feat_16, e_feat_32 = eosin_out   # [B, out_ch, 16, 16], [B, 64, 32, 32]
            else:
                e_feat_16 = eosin_out               # [B, out_ch, 16, 16]
            x = self.eosin_proj(torch.cat([x, e_feat_16], dim=1))  # [B, 512, 16, 16]

        # D5: upsample 16→32, skip from e4 + edge@32, UNI at 32
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        skip5 = [x, e4] + ([edge_maps[32]] if edge_maps else [])
        x = torch.cat(skip5, dim=1)
        x = self.dec5_conv(x)
        x_conv = x
        need_attn = self.use_attention_for_spade or self.enable_attention_residual
        x_attn = self.dec5_attn(x, uni_features) if need_attn else None
        spade_map_32 = None
        if self.spade_use_uni:
            spade_map_32 = x_attn if (self.use_attention_for_spade and x_attn is not None) else uni_maps[32]
        x = self.dec5_spade(x, spade_map_32, class_emb)
        if self.enable_attention_residual and x_attn is not None:
            x = x + (x_attn - x_conv)
        x = self.dec5_act(x)

        # Eosin injection at D5 (32×32) — Option B, only when eosin_multi_scale=True.
        # At 32×32 the 30px misalignment is <2px — still negligible.
        # Gives decoder a finer membrane signal one level above the bottleneck.
        if self.eosin_proj_32 is not None and e_feat_32 is not None:
            x = self.eosin_proj_32(torch.cat([x, e_feat_32], dim=1))  # [B, 512, 32, 32]

        # D4: upsample 32→64, skip from e3 + edge@64, UNI at 64
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        skip4 = [x, e3] + ([edge_maps[64]] if edge_maps else [])
        x = torch.cat(skip4, dim=1)
        x = self.dec4_conv(x)
        x_conv = x
        x_attn = self.dec4_attn(x, uni_features) if need_attn else None
        spade_map_64 = None
        if self.spade_use_uni:
            spade_map_64 = x_attn if (self.use_attention_for_spade and x_attn is not None) else uni_maps[64]
        x = self.dec4_spade(x, spade_map_64, class_emb)
        if self.enable_attention_residual and x_attn is not None:
            x = x + (x_attn - x_conv)
        x = self.dec4_act(x)

        # D3: upsample 64→128, skip from e2 + edge@128, UNI at 128
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        skip3 = [x, e2] + ([edge_maps[128]] if edge_maps else [])
        x = torch.cat(skip3, dim=1)
        x = self.dec3_conv(x)
        x_conv = x
        x_attn = self.dec3_attn(x, uni_features) if need_attn else None
        spade_map_128 = None
        if self.spade_use_uni:
            spade_map_128 = x_attn if (self.use_attention_for_spade and x_attn is not None) else uni_maps[128]
        x = self.dec3_spade(x, spade_map_128, class_emb)
        if self.enable_attention_residual and x_attn is not None:
            x = x + (x_attn - x_conv)
        x = self.dec3_act(x)

        # D2: upsample 128→256, skip from e1 + edge@256, UNI at 256
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        skip2 = [x, e1] + ([edge_maps[256]] if edge_maps else [])
        x = torch.cat(skip2, dim=1)
        x = self.dec2_conv(x)
        x_conv = x
        x_attn = self.dec2_attn(x, uni_features) if need_attn else None
        spade_map_256 = None
        if self.spade_use_uni:
            spade_map_256 = x_attn if (self.use_attention_for_spade and x_attn is not None) else uni_maps[256]
        x = self.dec2_spade(x, spade_map_256, class_emb)
        if self.enable_attention_residual and x_attn is not None:
            x = x + (x_attn - x_conv)
        x = self.dec2_act(x)

        if self.image_size == 1024:
            # D1: upsample 256→512, skip from e0 (32ch) + edge@512
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            skip1 = [x, e0] + ([edge_maps[512]] if edge_maps else [])
            x = torch.cat(skip1, dim=1)
            x = self.dec1_conv(x)
            x_conv = x
            x_attn = self.dec1_attn(x, uni_features) if need_attn else None
            if self.dec1_spade is not None and 512 in uni_maps:
                if self.spade_use_uni:
                    if self.use_attention_for_spade and x_attn is not None:
                        spade_map_512 = self.dec1_attn_to_spade(x_attn)
                    else:
                        spade_map_512 = uni_maps[512]
                else:
                    spade_map_512 = None
                x = self.dec1_spade(x, spade_map_512, class_emb)
            if self.enable_attention_residual and x_attn is not None:
                x = x + (x_attn - x_conv)
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
