#!/usr/bin/env python3
"""
Extract and print all hyperparameters from a checkpoint for ablation study reference.

Usage:
    python scripts/debug/extract_hparams.py --checkpoint checkpoints/destaining_v2/last.ckpt
"""

import argparse
import json
from pathlib import Path

import torch
from src.models.trainer import UNIStainNetTrainer


def main():
    parser = argparse.ArgumentParser(description='Extract hparams from checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    
    # Load checkpoint
    try:
        model = UNIStainNetTrainer.load_from_checkpoint(ckpt_path, strict=False, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    hparams = model.hparams

    print("\n" + "=" * 80)
    print("TRAINING HYPERPARAMETERS (ABLATION STUDY REFERENCE)")
    print("=" * 80)

    # Categorize hparams
    categories = {
        'Model Architecture': [
            'image_size', 'z_dim', 'edge_encoder', 'use_eosin_encoder', 'eosin_multi_scale',
            'uni_spatial_size', 'uni_freeze', 'uni_in_decoder_levels'
        ],
        'Attention & Residuals': [
            'attn_res_in_decoder', 'cross_attn_cond_type', 'cross_attn_layers',
            'self_attn_layers', 'attn_kernel_size'
        ],
        'Discriminator': [
            'proj_disc_weight', 'uncond_disc_weight', 'uncond_disc_norm',
            'disc_loss_weight', 'r1_weight', 'r1_every'
        ],
        'Loss Weights': [
            'l1_fullres_weight', 'lpips_fullres_weight', 'lpips_weight', 'lpips_fine_weight',
            'spectral_misalign_weight', 'ihc_edge_weight', 'he_edge_weight',
            'dab_intensity_weight', 'dab_histo_weight', 'dab_block_weight', 'dab_sparsity_weight',
            'dab_sparsity_margin', 'feat_match_weight', 'uncond_adv_g_weight', 'proj_adv_g_weight'
        ],
        'Training': [
            'case_b_prob', 'cfg_dropout_class_only', 'cfg_dropout_uni_only', 'cfg_dropout_both',
            'ema_decay', 'adversarial_start_step', 'lr', 'accum_steps'
        ],
        'Data': [
            'batch_size', 'num_workers', 'crop_size'
        ]
    }

    for category, keys in categories.items():
        print(f"\n{category}:")
        print("-" * 80)
        found_any = False
        for key in keys:
            if hasattr(hparams, key):
                val = getattr(hparams, key)
                # Highlight zero/disabled losses
                marker = " [DISABLED]" if (isinstance(val, (int, float)) and val == 0) else ""
                print(f"  {key:<40s} = {val}{marker}")
                found_any = True
        if not found_any:
            print(f"  (no parameters in this category)")

    # Print remaining hparams not categorized
    all_categorized_keys = set()
    for keys in categories.values():
        all_categorized_keys.update(keys)
    
    remaining = {k: v for k, v in hparams.items() if k not in all_categorized_keys}
    if remaining:
        print(f"\nOther Parameters:")
        print("-" * 80)
        for key, val in sorted(remaining.items()):
            print(f"  {key:<40s} = {val}")

    # Summary of active losses
    print("\n" + "=" * 80)
    print("ACTIVE LOSSES SUMMARY")
    print("=" * 80)
    
    loss_weights = {
        'l1_fullres': hparams.get('l1_fullres_weight', 0),
        'lpips_fullres': hparams.get('lpips_fullres_weight', 0),
        'lpips': hparams.get('lpips_weight', 0),
        'lpips_fine': hparams.get('lpips_fine_weight', 0),
        'spectral_misalign': hparams.get('spectral_misalign_weight', 0),
        'ihc_edge': hparams.get('ihc_edge_weight', 0),
        'he_edge': hparams.get('he_edge_weight', 0),
        'dab_intensity': hparams.get('dab_intensity_weight', 0),
        'dab_histo': hparams.get('dab_histo_weight', 0),
        'dab_block': hparams.get('dab_block_weight', 0),
        'dab_sparsity': hparams.get('dab_sparsity_weight', 0),
        'feat_match': hparams.get('feat_match_weight', 0),
        'uncond_adv_g': hparams.get('uncond_adv_g_weight', 0),
        'proj_adv_g': hparams.get('proj_adv_g_weight', 0),
    }

    active = {k: v for k, v in loss_weights.items() if v > 0}
    disabled = {k: v for k, v in loss_weights.items() if v == 0}

    print(f"\n✓ ACTIVE ({len(active)} losses):")
    for name, weight in sorted(active.items(), key=lambda x: -x[1]):
        print(f"    {name:<25s} weight={weight}")

    if disabled:
        print(f"\n✗ DISABLED ({len(disabled)} losses):")
        for name, weight in sorted(disabled.items()):
            print(f"    {name:<25s} weight={weight}")

    # Attention summary
    print("\n" + "=" * 80)
    print("ATTENTION & DECODER SUMMARY")
    print("=" * 80)
    
    print(f"  Residual attention in decoder: {hparams.get('attn_res_in_decoder', 'N/A')}")
    print(f"  Cross-attention conditioning:  {hparams.get('cross_attn_cond_type', 'N/A')}")
    print(f"  Cross-attention layers:       {hparams.get('cross_attn_layers', 'N/A')}")
    print(f"  Self-attention layers:        {hparams.get('self_attn_layers', 'N/A')}")
    print(f"  UNI frozen:                   {hparams.get('uni_freeze', 'N/A')}")
    print(f"  Eosin encoder:                {hparams.get('use_eosin_encoder', 'N/A')}")
    if hparams.get('use_eosin_encoder'):
        print(f"    └─ Multi-scale:             {hparams.get('eosin_multi_scale', False)}")

    # Training settings summary
    print("\n" + "=" * 80)
    print("TRAINING SETTINGS SUMMARY")
    print("=" * 80)
    
    print(f"  CFG Dropout probability:")
    print(f"    - Class only (10%):         {hparams.get('cfg_dropout_class_only', 0.1)}")
    print(f"    - UNI only (10%):           {hparams.get('cfg_dropout_uni_only', 0.1)}")
    print(f"    - Both (5%):                {hparams.get('cfg_dropout_both', 0.05)}")
    print(f"  Domain mixing (case_b_prob):  {hparams.get('case_b_prob', 0.25)} (misaligned fraction)")
    print(f"  EMA decay:                    {hparams.get('ema_decay', 0.999)}")
    print(f"  Adversarial warmup:           step {hparams.get('adversarial_start_step', 2000)}")
    print(f"  Gradient accumulation:        {hparams.get('accum_steps', 1)} steps")
    print(f"  Learning rate:                {hparams.get('lr', 'N/A')}")

    print("\n" + "=" * 80)
    print("END OF HPARAMS")
    print("=" * 80 + "\n")

    # Also save as JSON for reference
    output_json = Path(args.checkpoint).parent / 'hparams_extracted.json'
    hparams_dict = dict(hparams)
    with open(output_json, 'w') as f:
        json.dump(hparams_dict, f, indent=2, default=str)
    print(f"Hparams also saved to: {output_json}\n")


if __name__ == '__main__':
    main()
