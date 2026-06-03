"""
Consolidated evaluation metrics for UNIStainNet.

Provides standardized metric computation for both BCI and MIST datasets.

Metrics:
    - Image quality: FID (Inception + UNI), KID, LPIPS (128+512), SSIM, PSNR
    - DAB staining: KL divergence, JSD, Pearson-r, MAE (per-pair, 256-bin histograms)
    - Optical density: IOD, mIOD, FOD (PSPStain, MICCAI 2024)
    - Downstream: AUROC, SFS via UNI linear probe (Star-Diff, arXiv 2025)

References:
    - FID: Heusel et al., "GANs Trained by a Two Time-Scale Update Rule" (NeurIPS 2017)
    - KID: Binkowski et al., "Demystifying MMD GANs" (ICLR 2018)
    - LPIPS: Zhang et al., "The Unreasonable Effectiveness of Deep Features" (CVPR 2018)
    - DAB KL: Liu et al., "ODA-GAN" (Med Image Anal 2024) — per-pair 256-bin histograms
    - IOD/mIOD/FOD: Zhan et al., "PSPStain" (MICCAI 2024) — Beer-Lambert optical density
    - AUROC/SFS: Wu et al., "Star-Diff" (arXiv 2025) — UNI linear probe downstream task
    - DAB deconvolution: Ruifrok & Johnston, Anal Quant Cytol Histol (2001)
"""

import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from scipy.stats import entropy, pearsonr

from src.utils.dab import DABExtractor


# ======================================================================
# p90 DAB score (canonical: mean of top-10%)
# ======================================================================

def compute_p90_scores(dab_maps):
    """Compute canonical p90 DAB scores: mean of pixels >= 90th percentile.

    This is the canonical p90 metric used throughout the paper. For each image,
    we find the 90th percentile threshold and return the mean of all pixels
    at or above that threshold — i.e., the mean of the top-10%.

    Args:
        dab_maps: [B, H, W] raw DAB intensity maps (normalize="none")

    Returns:
        scores: numpy array of shape [B] with per-image p90 scores
    """
    scores = []
    for i in range(dab_maps.shape[0]):
        flat = dab_maps[i].flatten()
        p90 = torch.quantile(flat, 0.9)
        mask = flat >= p90
        scores.append(flat[mask].mean().item() if mask.sum() > 0 else flat.mean().item())
    return np.array(scores)


# ======================================================================
# Image quality metrics
# ======================================================================

def compute_image_quality_metrics(generated, real):
    """FID, KID, LPIPS (full + 128px), SSIM, PSNR.

    Args:
        generated: [N, 3, H, W] in [-1, 1]
        real: [N, 3, H, W] in [-1, 1]

    Returns:
        dict with metric values
    """
    from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    N = len(generated)
    results = {}

    # SSIM
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    ssim_vals = []
    for i in range(0, N, 16):
        batch_gen_01 = ((generated[i:i+16].float() + 1) / 2).clamp(0, 1)
        batch_real_01 = ((real[i:i+16].float() + 1) / 2).clamp(0, 1)
        batch_vals = ssim(batch_gen_01, batch_real_01)
        ssim_vals.append(batch_vals.item())
    results['ssim_mean'] = float(np.mean(ssim_vals))
    results['ssim_std'] = float(np.std(ssim_vals))

    # PSNR
    psnr = PeakSignalNoiseRatio(data_range=1.0)
    psnr_vals = []
    for i in range(0, N, 16):
        batch_gen_01 = ((generated[i:i+16].float() + 1) / 2).clamp(0, 1)
        batch_real_01 = ((real[i:i+16].float() + 1) / 2).clamp(0, 1)
        batch_vals = psnr(batch_gen_01, batch_real_01)
        psnr_vals.append(batch_vals.item())
    results['psnr_mean'] = float(np.mean(psnr_vals))
    results['psnr_std'] = float(np.std(psnr_vals))

    # LPIPS (full resolution)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').cuda()
    lpips_vals = []
    for i in range(0, N, 8):
        batch_gen = generated[i:i+8].float().clamp(-1, 1).cuda()
        batch_real = real[i:i+8].float().clamp(-1, 1).cuda()
        with torch.no_grad():
            val = lpips_metric(batch_gen, batch_real).item()
        if not np.isnan(val):
            lpips_vals.append(val)
    results['lpips_mean'] = float(np.mean(lpips_vals)) if lpips_vals else float('nan')

    # LPIPS downsampled (128x128)
    lpips_ds_vals = []
    for i in range(0, N, 8):
        batch_gen = F.interpolate(generated[i:i+8].float().clamp(-1, 1).cuda(), size=128, mode='bilinear', align_corners=False)
        batch_real = F.interpolate(real[i:i+8].float().clamp(-1, 1).cuda(), size=128, mode='bilinear', align_corners=False)
        with torch.no_grad():
            val = lpips_metric(batch_gen, batch_real).item()
        if not np.isnan(val):
            lpips_ds_vals.append(val)
    results['lpips_128_mean'] = float(np.mean(lpips_ds_vals)) if lpips_ds_vals else float('nan')

    del lpips_metric
    torch.cuda.empty_cache()

    # FID (Inception)
    fid = FrechetInceptionDistance(feature=2048, normalize=True).cuda()
    for i in range(0, N, 16):
        batch_gen_01 = ((generated[i:i+16].float() + 1) / 2).clamp(0, 1).cuda()
        batch_real_01 = ((real[i:i+16].float() + 1) / 2).clamp(0, 1).cuda()
        with torch.no_grad():
            fid.update(batch_real_01, real=True)
            fid.update(batch_gen_01, real=False)
    results['fid_inception'] = float(fid.compute().item())
    
    del fid
    torch.cuda.empty_cache()

    # KID (Kernel Inception Distance)
    kid = KernelInceptionDistance(feature=2048, normalize=True, subset_size=min(N, 100)).cuda()
    for i in range(0, N, 16):
        batch_gen_01 = ((generated[i:i+16].float() + 1) / 2).clamp(0, 1).cuda()
        batch_real_01 = ((real[i:i+16].float() + 1) / 2).clamp(0, 1).cuda()
        with torch.no_grad():
            kid.update(batch_real_01, real=True)
            kid.update(batch_gen_01, real=False)
    kid_mean, kid_std = kid.compute()
    results['kid_mean'] = float(kid_mean.item())
    results['kid_std'] = float(kid_std.item())
    results['kid_mean_x1000'] = float(kid_mean.item() * 1000)
    results['kid_std_x1000'] = float(kid_std.item() * 1000)

    del kid
    torch.cuda.empty_cache()
    
    import gc
    gc.collect()

    return results


# ======================================================================
# H&E structure similarity vs generated IHC
# ======================================================================

def compute_he_structure_metrics(generated, he_reference, resize_to=256):
    """Structural similarity between generated IHC and H&E reference.

    Converts both images to grayscale, extracts Sobel edge maps, and computes
    SSIM between the normalized edge maps. This measures whether generated
    IHC preserves the tissue structure of the input H&E.

    Args:
        generated: [N, 3, H, W] in [-1, 1]
        he_reference: [N, 3, H, W] in [-1, 1]
        resize_to: spatial size used before edge extraction

    Returns:
        dict with key:
            - he_structure_ssim: SSIM between normalized edge maps
    """
    from torchmetrics.image import StructuralSimilarityIndexMeasure

    ssim_vals = []
    for i in range(0, len(generated), 16):
        gen = F.interpolate(generated[i:i+16].float(), size=resize_to, mode='bilinear', align_corners=False)
        he = F.interpolate(he_reference[i:i+16].float(), size=resize_to, mode='bilinear', align_corners=False)

        gen_gray = gen.mean(dim=1, keepdim=True)
        he_gray = he.mean(dim=1, keepdim=True)

        gen_edge = _edge_map(gen_gray)
        he_edge = _edge_map(he_gray)

        ssim_vals.append(ssim(gen_edge, he_edge).item())

    return {
        'he_structure_ssim': float(np.mean(ssim_vals)),
    }


# ======================================================================
# UNI-FID (pathology-native Frechet distance)
# ======================================================================

def compute_uni_fid(generated, real):
    """Frechet distance in UNI ViT-L/16 feature space.

    Uses CLS token features from UNI (Chen et al., Nature Medicine 2024)
    as a pathology-specific alternative to Inception FID.

    Args:
        generated, real: [N, 3, H, W] in [-1, 1]

    Returns:
        float: UNI-FID value
    """
    import timm
    import torchvision.transforms as transforms
    from scipy.linalg import sqrtm

    uni_model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True,
                                   init_values=1e-5, dynamic_img_size=True)
    uni_model = uni_model.cuda().eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def extract_cls_features(images):
        feats = []
        for i in range(0, len(images), 16):
            batch = images[i:i+16].float()
            batch_01 = ((batch + 1) / 2).clamp(0, 1)
            batch_norm = torch.stack([transform(img) for img in batch_01])
            with torch.no_grad():
                out = uni_model.forward_features(batch_norm.cuda())
                feats.append(out[:, 0, :].cpu())
        return torch.cat(feats).numpy()

    feats_gen = extract_cls_features(generated)
    feats_real = extract_cls_features(real)

    mu_gen, mu_real = feats_gen.mean(0), feats_real.mean(0)
    sigma_gen = np.cov(feats_gen, rowvar=False)
    sigma_real = np.cov(feats_real, rowvar=False)

    diff = mu_gen - mu_real
    covmean = sqrtm(sigma_gen @ sigma_real)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    del uni_model
    torch.cuda.empty_cache()

    return float(diff @ diff + np.trace(sigma_gen + sigma_real - 2 * covmean))


# ======================================================================
# DAB metrics
# ======================================================================

def compute_dab_metrics(generated, real, labels=None, dab_extractor=None):
    """DAB intensity metrics: Pearson-r, MAE, KL, JSD.

    Uses canonical p90 scoring (mean of top-10%) for Pearson-r and MAE.
    Uses per-pair 256-bin histograms for KL/JSD (ODA-GAN methodology).

    Args:
        generated, real: [N, 3, H, W] in [-1, 1]
        labels: [N] int class labels, or None for classless evaluation
        dab_extractor: DABExtractor instance (created if None)

    Returns:
        dict with DAB metric values
    """
    if dab_extractor is None:
        dab_extractor = DABExtractor(device='cpu')

    dab_gen_list, dab_real_list = [], []
    for i in range(0, len(generated), 16):
        dab_gen_list.append(dab_extractor.extract_dab_intensity(generated[i:i+16].float(), normalize="none"))
        dab_real_list.append(dab_extractor.extract_dab_intensity(real[i:i+16].float(), normalize="none"))
        
    dab_gen = torch.cat(dab_gen_list)
    dab_real = torch.cat(dab_real_list)

    gen_scores = compute_p90_scores(dab_gen)
    real_scores = compute_p90_scores(dab_real)

    results = {}
    results['dab_mae_overall'] = float(np.mean(np.abs(gen_scores - real_scores)))

    # Pearson-R
    if len(gen_scores) > 2:
        r, p_val = pearsonr(gen_scores, real_scores)
        results['dab_pearson_r'] = float(r)
        results['dab_pearson_p'] = float(p_val)

    # Per-pair DAB KL/JSD (ODA-GAN: 256-bin histogram per pair, averaged)
    n_bins = 256
    eps = 1e-10
    pair_kls = []
    pair_jsds = []
    for i in range(dab_gen.shape[0]):
        g = dab_gen[i].flatten().numpy()
        r = dab_real[i].flatten().numpy()
        hist_range = (0, max(g.max(), r.max()) + 1e-6)
        hg, _ = np.histogram(g, bins=n_bins, range=hist_range, density=True)
        hr, _ = np.histogram(r, bins=n_bins, range=hist_range, density=True)
        hg = hg + eps; hr = hr + eps
        hg = hg / hg.sum(); hr = hr / hr.sum()
        pair_kls.append(float(entropy(hg, hr)))
        m = 0.5 * (hg + hr)
        pair_jsds.append(float(0.5 * entropy(hg, m) + 0.5 * entropy(hr, m)))
    results['dab_kl'] = float(np.mean(pair_kls))
    results['dab_kl_std'] = float(np.std(pair_kls))
    results['dab_jsd'] = float(np.mean(pair_jsds))
    results['dab_jsd_std'] = float(np.std(pair_jsds))

    # Pooled DAB KL (for reference)
    dab_gen_flat = dab_gen.flatten().numpy()
    dab_real_flat = dab_real.flatten().numpy()
    hist_range = (0, max(dab_gen_flat.max(), dab_real_flat.max()) + 1e-6)
    hist_gen, _ = np.histogram(dab_gen_flat, bins=n_bins, range=hist_range, density=True)
    hist_real, _ = np.histogram(dab_real_flat, bins=n_bins, range=hist_range, density=True)
    hist_gen = hist_gen + eps; hist_real = hist_real + eps
    hist_gen = hist_gen / hist_gen.sum(); hist_real = hist_real / hist_real.sum()
    results['dab_kl_pooled'] = float(entropy(hist_gen, hist_real))

    # Mean DAB levels
    results['dab_gen_mean'] = float(np.mean(gen_scores))
    results['dab_real_mean'] = float(np.mean(real_scores))

    # Per-class metrics (BCI only — MIST passes labels=None)
    if labels is not None:
        class_names = {0: '0', 1: '1+', 2: '2+', 3: '3+'}
        within_rs = []
        for cls, name in class_names.items():
            mask = (labels == cls).numpy() if isinstance(labels, torch.Tensor) else (labels == cls)
            if mask.sum() > 0:
                results[f'dab_real_class_{name}'] = float(np.mean(real_scores[mask]))
                results[f'dab_gen_class_{name}'] = float(np.mean(gen_scores[mask]))
                results[f'dab_mae_class_{name}'] = float(np.mean(np.abs(
                    gen_scores[mask] - real_scores[mask])))
                results[f'n_samples_class_{name}'] = int(mask.sum())
                # Within-class Pearson-R
                if mask.sum() > 5:
                    r_cls, _ = pearsonr(gen_scores[mask], real_scores[mask])
                    results[f'dab_pearson_r_class_{name}'] = float(r_cls)
                    within_rs.append(r_cls)
        if within_rs:
            results['dab_pearson_r_within_class'] = float(np.mean(within_rs))

        # Ordering violation rate
        class_gen_means = {}
        for cls in range(4):
            mask = (labels == cls).numpy() if isinstance(labels, torch.Tensor) else (labels == cls)
            if mask.sum() > 0:
                class_gen_means[cls] = float(np.mean(gen_scores[mask]))
        ordered_pairs = [(3, 2), (3, 1), (3, 0), (2, 1), (2, 0), (1, 0)]
        violations, total_pairs = 0, 0
        for high_cls, low_cls in ordered_pairs:
            if high_cls in class_gen_means and low_cls in class_gen_means:
                total_pairs += 1
                if class_gen_means[high_cls] < class_gen_means[low_cls]:
                    violations += 1
        results['ordering_violations'] = violations
        results['ordering_total_pairs'] = total_pairs

    return results


# ======================================================================
# IOD / mIOD / FOD metrics (PSPStain)
# ======================================================================

def compute_iod_metrics(generated, real, labels=None):
    """Compute Integrated Optical Density metrics (PSPStain methodology).

    Beer-Lambert law: OD = -log10(I / I_0), I_0 = 255.
    IOD = sum(OD), mIOD = mean(OD), FOD = OD^alpha with alpha=1.8.

    Args:
        generated, real: [N, 3, H, W] in [-1, 1]
        labels: [N] optional class labels for per-class breakdown

    Returns:
        dict with IOD metric values
    """
    miod_gen_list, miod_real_list = [], []
    iod_gen_list, iod_real_list = [], []
    fod_gen_list, fod_real_list = [], []
    
    for i in range(0, len(generated), 16):
        g_batch = generated[i:i+16].float()
        r_batch = real[i:i+16].float()
        
        g_255 = (((g_batch + 1) / 2).clamp(0, 1) * 255.0).clamp(min=1.0)
        r_255 = (((r_batch + 1) / 2).clamp(0, 1) * 255.0).clamp(min=1.0)

        od_gen = -torch.log10(g_255 / 255.0)
        od_real = -torch.log10(r_255 / 255.0)

        miod_gen_list.append(od_gen.mean(dim=(1, 2, 3)).numpy())
        miod_real_list.append(od_real.mean(dim=(1, 2, 3)).numpy())

        iod_gen_list.append(od_gen.sum(dim=(1, 2, 3)).numpy())
        iod_real_list.append(od_real.sum(dim=(1, 2, 3)).numpy())

        alpha = 1.8
        fod_gen_list.append(od_gen.pow(alpha).mean(dim=(1, 2, 3)).numpy())
        fod_real_list.append(od_real.pow(alpha).mean(dim=(1, 2, 3)).numpy())

    miod_gen = np.concatenate(miod_gen_list)
    miod_real = np.concatenate(miod_real_list)
    iod_gen = np.concatenate(iod_gen_list)
    iod_real = np.concatenate(iod_real_list)
    fod_gen = np.concatenate(fod_gen_list)
    fod_real = np.concatenate(fod_real_list)



    results = {}
    results['miod_diff'] = float(np.mean(miod_gen) - np.mean(miod_real))
    results['miod_abs_diff'] = float(np.mean(np.abs(miod_gen - miod_real)))
    results['miod_gen_mean'] = float(np.mean(miod_gen))
    results['miod_real_mean'] = float(np.mean(miod_real))
    results['iod_diff'] = float(np.mean(iod_gen) - np.mean(iod_real))
    results['iod_diff_1e7'] = float(results['iod_diff'] / 1e7)
    results['mfod_diff'] = float(np.mean(fod_gen) - np.mean(fod_real))
    results['mfod_abs_diff'] = float(np.mean(np.abs(fod_gen - fod_real)))

    if len(miod_gen) > 2:
        r, p = pearsonr(miod_gen, miod_real)
        results['iod_pearson_r'] = float(r)

    # Per-class mIOD (BCI only)
    if labels is not None:
        class_names = {0: '0', 1: '1+', 2: '2+', 3: '3+'}
        for cls, name in class_names.items():
            mask = (labels == cls).numpy() if isinstance(labels, torch.Tensor) else (labels == cls)
            if mask.sum() > 0:
                results[f'miod_gen_class_{name}'] = float(np.mean(miod_gen[mask]))
                results[f'miod_real_class_{name}'] = float(np.mean(miod_real[mask]))
                results[f'miod_diff_class_{name}'] = float(
                    np.mean(miod_gen[mask]) - np.mean(miod_real[mask]))

    return results


# ======================================================================
# Downstream classifier (AUROC / SFS)
# ======================================================================

def compute_downstream_metrics(generated, real, labels, train_ihc_dir):
    """AUROC and SFS via UNI linear probe (Star-Diff methodology).

    1. Extract UNI CLS features from real HER2 training images
    2. Train logistic regression on real train features
    3. Evaluate on generated test images (AUROC, SFS)
    4. Evaluate on real test images as reference

    Args:
        generated, real: [N, 3, H, W] in [-1, 1]
        labels: [N] class labels
        train_ihc_dir: path to real HER2 IHC training images

    Returns:
        dict with downstream metric values
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    from sklearn.preprocessing import label_binarize
    import timm
    import torchvision.transforms as transforms
    from PIL import Image

    results = {}

    # Load UNI model
    print("  Loading UNI model for downstream evaluation...")
    uni_model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True,
                                   init_values=1e-5, dynamic_img_size=True)
    uni_model = uni_model.cuda().eval()

    uni_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def extract_features_from_tensors(images):
        feats = []
        for i in range(0, len(images), 16):
            batch = images[i:i+16]
            batch_01 = ((batch + 1) / 2).clamp(0, 1)
            batch_norm = torch.stack([uni_transform(img) for img in batch_01])
            with torch.no_grad():
                out = uni_model.forward_features(batch_norm.cuda())
                feats.append(out[:, 0, :].cpu())
        return torch.cat(feats).numpy()

    def extract_features_from_dir(img_dir):
        import torchvision.transforms.functional as TF
        img_dir = Path(img_dir)
        filenames = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        feats, labs = [], []
        label_map = {'0': 0, '1+': 1, '2+': 2, '3+': 3}
        batch_imgs, batch_labs = [], []
        for fn in filenames:
            img = Image.open(img_dir / fn).convert('RGB')
            img_t = transforms.ToTensor()(img)
            img_n = uni_transform(img_t)
            batch_imgs.append(img_n)
            parts = fn.replace('.png', '').split('_')
            batch_labs.append(label_map[parts[2]])
            if len(batch_imgs) == 16:
                batch = torch.stack(batch_imgs)
                with torch.no_grad():
                    out = uni_model.forward_features(batch.cuda())
                    feats.append(out[:, 0, :].cpu())
                labs.extend(batch_labs)
                batch_imgs, batch_labs = [], []
        if batch_imgs:
            batch = torch.stack(batch_imgs)
            with torch.no_grad():
                out = uni_model.forward_features(batch.cuda())
                feats.append(out[:, 0, :].cpu())
            labs.extend(batch_labs)
        return torch.cat(feats).numpy(), np.array(labs)

    # Extract features from real training images
    print(f"  Extracting features from training IHC images...")
    train_feats, train_labels = extract_features_from_dir(train_ihc_dir)

    # Train linear probe
    print(f"  Training linear probe on {len(train_labels)} samples...")
    clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                             multi_class='multinomial', random_state=42)
    clf.fit(train_feats, train_labels)
    results['probe_train_acc'] = float(clf.score(train_feats, train_labels))

    # Evaluate on generated
    gen_feats = extract_features_from_tensors(generated)
    test_labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    gen_probs = clf.predict_proba(gen_feats)
    gen_preds = clf.predict(gen_feats)

    test_labels_bin = label_binarize(test_labels, classes=[0, 1, 2, 3])
    try:
        results['auroc'] = float(roc_auc_score(test_labels_bin, gen_probs,
                                                multi_class='ovr', average='macro'))
    except ValueError:
        pass

    results['sfs'] = float(balanced_accuracy_score(test_labels, gen_preds))

    # Evaluate on real (reference baseline)
    real_feats = extract_features_from_tensors(real)
    real_probs = clf.predict_proba(real_feats)
    real_preds = clf.predict(real_feats)
    try:
        results['auroc_real_baseline'] = float(roc_auc_score(
            test_labels_bin, real_probs, multi_class='ovr', average='macro'))
    except ValueError:
        pass
    results['sfs_real_baseline'] = float(balanced_accuracy_score(test_labels, real_preds))

    del uni_model
    torch.cuda.empty_cache()

    return results


# ======================================================================
# H-map (Hematoxylin channel) structure metrics
# ======================================================================

def compute_he_h_ssim(generated, target_img, is_target_he=False, dab_extractor=None, he_extractor=None):
    """Compute SSIM on Hematoxylin channel between generated and target image.

    Measures structural preservation against a provided ground-truth RGB image.

    Args:
        generated: [N, 3, H, W] generated IHC in [-1, 1]
        target_img: [N, 3, H, W] target structural RGB image (IHC or H&E)
        is_target_he: True if target_img is H&E, False if IHC
        dab_extractor: DABExtractor instance (will create if None)
        he_extractor: HEExtractor instance (will create if None and needed)

    Returns:
        dict with ssim_mean, ssim_std
    """
    from torchmetrics.image import StructuralSimilarityIndexMeasure

    if dab_extractor is None:
        dab_extractor = DABExtractor(device='cpu')
    if is_target_he and he_extractor is None:
        from src.utils.dab import HEExtractor
        he_extractor = HEExtractor(device='cpu')

    # Extract H-maps from generated RGB
    h_gen_list = []
    h_target_list = []
    for i in range(0, len(generated), 16):
        h_gen_list.append(dab_extractor.extract_hematoxylin_intensity(generated[i:i+16].float().cpu(), normalize="max"))
        
        target_batch = target_img[i:i+16].float().cpu()
        if is_target_he:
            h_target_list.append(he_extractor.extract_hematoxylin_intensity(target_batch, normalize="max"))
        else:
            h_target_list.append(dab_extractor.extract_hematoxylin_intensity(target_batch, normalize="max"))
        
    h_gen = torch.cat(h_gen_list)
    h_target = torch.cat(h_target_list)

    # Ensure [0, 1] range
    h_gen = (h_gen - h_gen.min()) / (h_gen.max() - h_gen.min() + 1e-6)
    h_target = (h_target - h_target.min()) / (h_target.max() - h_target.min() + 1e-6)

    # Stack to [N, 1, H, W] for SSIM
    h_gen_1ch = h_gen.unsqueeze(1)
    h_target_1ch = h_target.unsqueeze(1)

    results = {}
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    ssim_vals = []
    for i in range(0, len(h_gen), 16):
        batch_val = ssim_metric(h_gen_1ch[i:i+16], h_target_1ch[i:i+16])
        ssim_vals.append(batch_val.item())

    results['he_h_ssim_mean'] = float(np.mean(ssim_vals))
    results['he_h_ssim_std'] = float(np.std(ssim_vals))

    return results


def compute_he_nmi(generated, target_img, is_target_he=False, dab_extractor=None, he_extractor=None, n_bins=256):
    """Compute Normalized Mutual Information (NMI) against target image.

    NMI = (2 * MI(H_gen, H_real)) / (H(H_gen) + H(H_real))
    where H = entropy, MI = mutual information.

    Measures information overlap / alignment between generated and target H-channel.
    Range: [0, 1]. Higher = better alignment.

    Args:
        generated: [N, 3, H, W] generated IHC in [-1, 1]
        target_img: [N, 3, H, W] target structural RGB image
        is_target_he: True if target_img is H&E, False if IHC
        dab_extractor: DABExtractor instance (will create if None)
        he_extractor: HEExtractor instance (will create if None and needed)
        n_bins: number of histogram bins for entropy estimation

    Returns:
        dict with nmi_mean, nmi_std
    """
    if dab_extractor is None:
        dab_extractor = DABExtractor(device='cpu')
    if is_target_he and he_extractor is None:
        from src.utils.dab import HEExtractor
        he_extractor = HEExtractor(device='cpu')

    # Extract H-maps from generated RGB
    h_gen_list = []
    h_target_list = []
    for i in range(0, len(generated), 16):
        h_gen_list.append(dab_extractor.extract_hematoxylin_intensity(generated[i:i+16].float().cpu(), normalize="none"))
        
        target_batch = target_img[i:i+16].float().cpu()
        if is_target_he:
            h_target_list.append(he_extractor.extract_hematoxylin_intensity(target_batch, normalize="none"))
        else:
            h_target_list.append(dab_extractor.extract_hematoxylin_intensity(target_batch, normalize="none"))
        
    h_gen = torch.cat(h_gen_list)
    h_target = torch.cat(h_target_list)

    nmi_vals = []
    for i in range(len(h_gen)):
        h_g = h_gen[i].flatten().numpy()
        h_r = h_target[i].flatten().numpy()

        # Normalize to [0, 1] for consistent histogram binning
        h_g_norm = (h_g - h_g.min()) / (h_g.max() - h_g.min() + 1e-6)
        h_r_norm = (h_r - h_r.min()) / (h_r.max() - h_r.min() + 1e-6)

        # Joint histogram
        hist_2d, _, _ = np.histogram2d(h_g_norm, h_r_norm, bins=n_bins, range=[[0, 1], [0, 1]])
        hist_2d = hist_2d / hist_2d.sum()  # normalize to probability

        # Marginal histograms
        p_g = hist_2d.sum(axis=1)
        p_r = hist_2d.sum(axis=0)

        # Entropies
        H_g = -np.sum(p_g[p_g > 0] * np.log2(p_g[p_g > 0]))
        H_r = -np.sum(p_r[p_r > 0] * np.log2(p_r[p_r > 0]))

        # Mutual information
        MI = 0.0
        for gi in range(len(p_g)):
            for ri in range(len(p_r)):
                if hist_2d[gi, ri] > 0 and p_g[gi] > 0 and p_r[ri] > 0:
                    MI += hist_2d[gi, ri] * np.log2(hist_2d[gi, ri] / (p_g[gi] * p_r[ri]))

        # NMI
        if (H_g + H_r) > 1e-6:
            nmi = 2.0 * MI / (H_g + H_r)
        else:
            nmi = 0.0

        nmi_vals.append(float(np.clip(nmi, 0, 1)))

    results = {}
    results['he_nmi_mean'] = float(np.mean(nmi_vals))
    results['he_nmi_std'] = float(np.std(nmi_vals))

    return results

def save_sample_grid(he, real, generated, path, n=16):
    """Save H&E | Real IHC | Generated grid for visual inspection."""
    n = min(n, len(he))
    grid_images = []
    for i in range(n):
        grid_images.extend([
            ((he[i] + 1) / 2).clamp(0, 1),
            ((real[i] + 1) / 2).clamp(0, 1),
            ((generated[i] + 1) / 2).clamp(0, 1),
        ])
    grid = torchvision.utils.make_grid(grid_images, nrow=3, padding=2)
    torchvision.utils.save_image(grid, path)
    print(f"  Sample grid ({n} samples): {path}")


def composite_background(generated, he_images, threshold=0.85):
    """Replace background regions in generated images with white.

    Uses H&E brightness to identify background (glass slide), then
    forces those regions to white in the generated output.
    """
    he_01 = ((he_images + 1) / 2).clamp(0, 1)
    brightness = he_01.mean(dim=1, keepdim=True)
    tissue = (brightness < threshold).float()
    tissue = F.avg_pool2d(tissue, kernel_size=7, stride=1, padding=3)
    tissue = (tissue > 0.3).float()
    tissue = F.avg_pool2d(tissue, kernel_size=11, stride=1, padding=5)
    return generated * tissue + 1.0 * (1.0 - tissue)
