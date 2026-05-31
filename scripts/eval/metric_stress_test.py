#!/usr/bin/env python3
"""
Metric Robustness Stress Test for Histopathology Generative Models
Evaluates the sensitivity of traditional vs. foundation model metrics to continuous, non-biological mutations.

Metrics Evaluated:
- Pixel MSE
- SSIM (Structural Similarity via skimage)
- LPIPS (Learned Perceptual Image Patch Similarity via torchmetrics)
- ImageNet-FID (Feature Distance equivalent for single image)
- Path-FID (CONCH/UNI Feature Distance without projection/normalization)

Mutations:
- Translation (Shift)
- Rotation (Tilt)
- Gaussian Noise
- Gaussian Blur
"""

import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.models import inception_v3, Inception_V3_Weights
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import timm

# Try to import torchmetrics for LPIPS
try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_models(device):
    models = {}
    
    print("Loading InceptionV3 for ImageNet-FID...")
    # Using standard inception v3 and stripping classification head for 2048-d features
    inc_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    inc_model.fc = nn.Identity()
    inc_model.eval().to(device)
    models['inception'] = inc_model

    print("Loading Foundation Model for Path-FID...")
    try:
        # Attempt to load CONCH as requested
        fm_model = timm.create_model("hf-hub:MahmoodLab/conch", pretrained=True, num_classes=0)
        print("Successfully loaded CONCH.")
    except Exception as e:
        print(f"Warning: Failed to load CONCH. Error: {e}")
        print("Falling back to UNI model (MahmoodLab/uni)...")
        fm_model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True, num_classes=0)
        print("Successfully loaded UNI.")
        
    fm_model.eval().to(device)
    models['path_fm'] = fm_model
    
    if HAS_LPIPS:
        print("Loading LPIPS (AlexNet)...")
        lpips_model = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
        models['lpips'] = lpips_model

    return models

def extract_inception_features(model, img_t):
    """Extract 2048-d features from InceptionV3."""
    # Resize to 299x299 and normalize to [-1, 1] as expected by standard FID preprocessing
    img_resized = F.interpolate(img_t, size=(299, 299), mode='bilinear', align_corners=False)
    img_norm = (img_resized - 0.5) * 2.0
    with torch.no_grad():
        feat = model(img_norm)
    return feat

def extract_path_features(model, img_t):
    """Extract features from pathology foundation model (CONCH/UNI) without projection/normalization."""
    # Resize to 224x224 and use ImageNet normalization
    img_resized = F.interpolate(img_t, size=(224, 224), mode='bilinear', align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_t.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_t.device).view(1, 3, 1, 1)
    img_norm = (img_resized - mean) / std
    
    with torch.no_grad():
        if hasattr(model, 'forward_features'):
            feat = model.forward_features(img_norm)
            # If the model returns unpooled sequence (e.g. [1, 197, 1024]), grab CLS token
            if feat.ndim == 3:
                feat = feat[:, 0]
        else:
            feat = model(img_norm)
    return feat

def compute_feature_distance(feat1, feat2):
    """
    Computes squared L2 distance between features.
    For a single sample (N=1), Frechet Distance simplifies exactly to ||mu1 - mu2||^2
    since the covariance matrices are zero.
    """
    return F.mse_loss(feat1, feat2, reduction='sum').item()

def run_stress_test(image_path, output_path):
    device = get_device()
    
    # 1. Input Handling
    if not os.path.exists(image_path):
        print(f"Image not found at '{image_path}'. Generating a random noise patch for demonstration.")
        # Create a dummy histopathology-like patch (pinkish-purple base)
        base = torch.tensor([0.8, 0.6, 0.7]).view(1, 3, 1, 1)
        noise = torch.rand(1, 3, 512, 512) * 0.4 - 0.2
        img_tensor = torch.clamp(base + noise, 0.0, 1.0)
    else:
        print(f"Loading image from {image_path}")
        img = Image.open(image_path).convert('RGB')
        # Resize/Crop to 512x512 to ensure standard patch size
        img = img.resize((512, 512))
        img_tensor = TF.to_tensor(img).unsqueeze(0)  # [1, 3, H, W] in [0, 1]
    
    img_tensor = img_tensor.to(device)
    
    models = load_models(device)
    
    # Precompute reference features
    ref_inc_feat = extract_inception_features(models['inception'], img_tensor)
    ref_path_feat = extract_path_features(models['path_fm'], img_tensor)
    ref_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # 2. Mutation Generators
    mutations = {
        'Shift': {
            'values': list(range(0, 31, 5)), # 0 to 30 in steps of 5
            'func': lambda img, v: TF.affine(img, angle=0.0, translate=[v, v], scale=1.0, shear=0.0)
        },
        'Tilt': {
            'values': list(range(0, 16, 2)), # 0 to 14 (we go up to 14, or up to 16 if we do np.arange)
            'func': lambda img, v: TF.rotate(img, float(v))
        },
        'Noise': {
            'values': np.linspace(0.0, 0.1, 11).tolist(), # Variance from 0.0 to 0.1
            'func': lambda img, v: torch.clamp(img + torch.randn_like(img) * math.sqrt(v), 0.0, 1.0)
        },
        'Blur': {
            'values': [1, 3, 5, 7, 9, 11], # Kernel sizes
            'func': lambda img, v: TF.gaussian_blur(img, kernel_size=[int(v), int(v)]) if v > 1 else img
        }
    }
    
    # Initialize metric storage
    metrics_list = ['MSE', 'SSIM', 'ImageNet-FID', 'Path-FID']
    if HAS_LPIPS:
        metrics_list.insert(2, 'LPIPS')
        
    results = {m: {metric: [] for metric in metrics_list} for m in mutations}
    
    # 3. Metric Evaluation Loop
    for mut_name, mut_data in mutations.items():
        print(f"\nEvaluating mutation: {mut_name}")
        for val in tqdm(mut_data['values'], desc=f"Steps ({mut_name})"):
            # Apply mutation
            mut_tensor = mut_data['func'](img_tensor, val)
            mut_np = mut_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # Pixel MSE
            mse_val = F.mse_loss(img_tensor, mut_tensor).item()
            results[mut_name]['MSE'].append(mse_val)
            
            # SSIM (via skimage, expects HWC)
            ssim_val = ssim(ref_np, mut_np, data_range=1.0, channel_axis=2)
            results[mut_name]['SSIM'].append(ssim_val)
            
            # LPIPS
            if HAS_LPIPS:
                with torch.no_grad():
                    # LPIPS expects input in range [-1, 1]
                    lpips_val = models['lpips'](mut_tensor * 2 - 1, img_tensor * 2 - 1).item()
                results[mut_name]['LPIPS'].append(lpips_val)
            
            # ImageNet-FID (Feature Distance)
            mut_inc_feat = extract_inception_features(models['inception'], mut_tensor)
            inc_fid_val = compute_feature_distance(ref_inc_feat, mut_inc_feat)
            results[mut_name]['ImageNet-FID'].append(inc_fid_val)
            
            # Path-FID
            mut_path_feat = extract_path_features(models['path_fm'], mut_tensor)
            path_fid_val = compute_feature_distance(ref_path_feat, mut_path_feat)
            results[mut_name]['Path-FID'].append(path_fid_val)
            
    # Helper to normalize metrics to [0, 1] for fair comparison
    # Important: MSE, LPIPS, and FIDs are distances (lower is better), SSIM is similarity (higher is better)
    # We will use (1 - SSIM) for structural dissimilarity to align the axes direction
    def normalize(arr, invert=False):
        arr = np.array(arr)
        if invert:
            arr = 1.0 - arr
        
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            return (arr - arr_min) / (arr_max - arr_min)
        return np.zeros_like(arr)

    # 4. Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Metric Robustness to Continuous Non-Biological Mutations', fontsize=16)
    axes = axes.flatten()
    
    # Setup styling based on metrics
    colors = ['#d62728', '#ff7f0e', '#8c564b', '#1f77b4', '#2ca02c']
    markers = ['o', 's', 'v', '^', 'D']
    
    for idx, (mut_name, mut_data) in enumerate(mutations.items()):
        ax = axes[idx]
        x_vals = mut_data['values']
        
        for m_idx, metric_name in enumerate(metrics_list):
            raw_vals = results[mut_name][metric_name]
            
            # Invert SSIM so it behaves like a distance metric (0 = no change)
            invert = (metric_name == 'SSIM')
            norm_vals = normalize(raw_vals, invert=invert)
            
            label = f"{metric_name} (Dist: 1-SSIM)" if metric_name == 'SSIM' else metric_name
                
            ax.plot(x_vals, norm_vals, label=label, color=colors[m_idx], marker=markers[m_idx], linewidth=2, markersize=8)
            
        ax.set_title(f"{mut_name}", fontsize=14)
        ax.set_xlabel('Mutation Magnitude', fontsize=12)
        ax.set_ylabel('Normalized Score (Degradation) [0, 1]', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        if idx == 0:
            ax.legend(loc='best', fontsize=11)
            
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved visualization to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stress Test Image Metrics')
    parser.add_argument('--image_path', type=str, default='dummy', help='Path to test image (will generate random if "dummy")')
    parser.add_argument('--output', type=str, default='metric_robustness_test.png', help='Output plot filename')
    args = parser.parse_args()
    
    run_stress_test(args.image_path, args.output)
