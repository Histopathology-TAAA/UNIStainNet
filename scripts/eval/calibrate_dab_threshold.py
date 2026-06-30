#!/usr/bin/env python3
"""
Calibrate the optimal DAB positivity threshold via Gaussian Mixture Model.

Fits a 2-component GMM to per-nucleus DAB OD values from real Ki67 IHC
images. The intersection of the two Gaussians gives the Bayes-optimal
threshold for separating Ki67- from Ki67+ nuclei — unsupervised, no
pathologist labels needed.

Usage:
    PYTHONPATH=. python scripts/eval/calibrate_dab_threshold.py \
        --data_dir /path/to/Destained_MIST --n_samples 200

Output:
    Optimal dab_threshold for StarDist/Cellpose evaluators.
    Histogram plot saved to eval_output/dab_threshold_calibration.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.optimize import bisect

from src.data.mist_dataset import MISTMultiStainCropDataModule, STAIN_TO_LABEL
from src.utils.dab import DABExtractor


def extract_per_nucleus_dab_od(ihc_image_01, hema_threshold=0.05):
    """Extract DAB OD for each nucleus in an IHC image.

    Simple nuclear segmentation: threshold the hematoxylin channel + watershed.
    For calibration, we don't need perfect segmentation — just enough nuclei
    to fit a GMM. A rough threshold + connected components is sufficient.

    Returns:
        numpy array of per-nucleus mean DAB OD values
    """
    from scipy import ndimage
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    dab_ext = DABExtractor(device='cpu')

    # Convert [-1,1] → [0,1] if needed
    if isinstance(ihc_image_01, np.ndarray) and ihc_image_01.min() < 0:
        ihc_image_01 = (ihc_image_01 + 1) / 2

    if hasattr(ihc_image_01, 'cpu'):
        import torch
        ihc_image_01 = ihc_image_01.cpu().numpy()
        if ihc_image_01.shape[0] == 3:
            ihc_image_01 = np.transpose(ihc_image_01, (1, 2, 0))
        if ihc_image_01.min() < 0:
            ihc_image_01 = (ihc_image_01 + 1) / 2

    ihc_image_01 = np.clip(ihc_image_01, 0, 1).astype(np.float32)

    # Extract H and DAB channels
    od = -np.log10(np.clip(ihc_image_01, 1e-6, 1.0))
    stain_matrix = np.array([[0.650, 0.704, 0.286], [0.268, 0.570, 0.776]], dtype=np.float64)
    deconv = np.linalg.pinv(stain_matrix)
    H, W = od.shape[:2]
    conc = np.maximum(od.reshape(-1, 3) @ deconv.T, 0).reshape(H, W, 2)
    hema = conc[:, :, 0]
    dab = conc[:, :, 1]

    # Simple nuclear segmentation via H-channel threshold + watershed
    hema_norm = (hema - hema.min()) / (hema.max() - hema.min() + 1e-6)
    nuclear_mask = hema_norm > 0.15  # hematoxylin = nuclei
    nuclear_mask = ndimage.binary_opening(nuclear_mask, iterations=1)
    nuclear_mask = ndimage.binary_closing(nuclear_mask, iterations=2)

    labelled, n_nuclei = ndimage.label(nuclear_mask)
    if n_nuclei < 10:
        return np.array([])

    # Per-nucleus mean DAB OD
    per_nucleus_dab = ndimage.mean(dab, labelled, index=np.arange(1, n_nuclei + 1))
    return per_nucleus_dab


def find_gmm_threshold(values, plot_path=None):
    """Fit 2-component GMM and find the intersection (optimal threshold).

    Args:
        values: 1D array of per-nucleus DAB OD
        plot_path: if provided, save diagnostic plot

    Returns:
        optimal threshold (float), gmm object
    """
    values = values.reshape(-1, 1)
    values = values[np.isfinite(values)]
    values = values[values > 0]  # remove zeros (background)

    if len(values) < 100:
        return 0.10, None  # fallback to QuPath default

    # Fit GMM
    gmm = GaussianMixture(n_components=2, random_state=42,
                          init_params='kmeans', max_iter=200)
    gmm.fit(values)

    # Sort components by mean (component 0 = negative, component 1 = positive)
    means = gmm.means_.ravel()
    if means[0] > means[1]:
        gmm.means_ = gmm.means_[::-1]
        gmm.covariances_ = gmm.covariances_[::-1]
        gmm.weights_ = gmm.weights_[::-1]
        means = gmm.means_.ravel()

    # Find intersection: where pdf1(x) = pdf2(x), i.e., log ratio = 0
    def log_pdf_ratio(x):
        from scipy.stats import norm
        p0 = norm.logpdf(x, means[0], np.sqrt(gmm.covariances_[0, 0, 0]))
        p1 = norm.logpdf(x, means[1], np.sqrt(gmm.covariances_[1, 0, 0]))
        w0, w1 = gmm.weights_
        return (p1 + np.log(w1)) - (p0 + np.log(w0))

    try:
        threshold = bisect(log_pdf_ratio, means[0], means[1], xtol=1e-4)
    except ValueError:
        # If no sign change, use midpoint
        threshold = (means[0] + means[1]) / 2

    threshold = float(np.clip(threshold, 0.05, 0.30))

    # Diagnostic plot
    if plot_path:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        x = np.linspace(values.min(), values.max(), 500)
        from scipy.stats import norm
        pdf0 = gmm.weights_[0] * norm.pdf(x, means[0], np.sqrt(gmm.covariances_[0, 0, 0]))
        pdf1 = gmm.weights_[1] * norm.pdf(x, means[1], np.sqrt(gmm.covariances_[1, 0, 0]))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values.ravel(), bins=100, density=True, alpha=0.5, color='gray', label='Data')
        ax.plot(x, pdf0, 'b-', label=f'Neg (μ={means[0]:.3f}, w={gmm.weights_[0]:.2f})')
        ax.plot(x, pdf1, 'r-', label=f'Pos (μ={means[1]:.3f}, w={gmm.weights_[1]:.2f})')
        ax.plot(x, pdf0 + pdf1, 'k--', label='Mixture', alpha=0.7)
        ax.axvline(threshold, color='green', linestyle='--', label=f'Threshold = {threshold:.4f}')
        ax.set_xlabel('DAB OD (per nucleus mean)')
        ax.set_ylabel('Density')
        ax.set_title('GMM Calibration — Ki67 DAB Threshold')
        ax.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Diagnostic plot saved to {plot_path}")

    return threshold, gmm


def main():
    parser = argparse.ArgumentParser(description='Calibrate DAB threshold via GMM')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--n_samples', type=int, default=200,
                        help='Number of real IHC images to sample')
    parser.add_argument('--output_dir', type=str, default='eval_output')
    args = parser.parse_args()

    print("=" * 60)
    print("  DAB Threshold Calibration (GMM)")
    print("=" * 60)

    # Load data
    stain_label = STAIN_TO_LABEL['Ki67']
    dm = MISTMultiStainCropDataModule(
        base_dir=args.data_dir, stains=['Ki67'],
        batch_size=4, num_workers=2,
        image_size=(512, 512), crop_size=512, null_class=stain_label,
    )
    dm.setup('test')
    loader = dm.val_dataloader()

    # Collect per-nucleus DAB OD values
    all_dab = []
    n_collected = 0

    print(f"  Extracting per-nucleus DAB OD from real Ki67 IHC...")
    for batch_idx, batch in enumerate(loader):
        _, her2, _, _, _, _, _ = batch

        for i in range(her2.shape[0]):
            if n_collected >= args.n_samples:
                break
            dab_vals = extract_per_nucleus_dab_od(her2[i])
            if len(dab_vals) > 0:
                all_dab.append(dab_vals)
                n_collected += 1

        if n_collected >= args.n_samples:
            break

        if (batch_idx + 1) % 20 == 0:
            print(f"    {n_collected}/{args.n_samples} images...")

    if n_collected == 0:
        print("[ERROR] No nuclei found. Check data path.")
        sys.exit(1)

    all_dab = np.concatenate(all_dab)
    print(f"\n  Collected {len(all_dab):,} nuclei from {n_collected} images")
    print(f"  DAB OD range: [{all_dab.min():.4f}, {all_dab.max():.4f}]")
    print(f"  DAB OD mean: {all_dab.mean():.4f}, std: {all_dab.std():.4f}")

    # Fit GMM
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / 'dab_threshold_calibration.png'

    threshold, gmm = find_gmm_threshold(all_dab, plot_path=plot_path)

    if gmm is not None:
        means = gmm.means_.ravel()
        print(f"\n  GMM components:")
        print(f"    Negative (Ki67-): μ={means[0]:.4f}, σ={np.sqrt(gmm.covariances_[0,0,0]):.4f}, w={gmm.weights_[0]:.3f}")
        print(f"    Positive (Ki67+): μ={means[1]:.4f}, σ={np.sqrt(gmm.covariances_[1,0,0]):.4f}, w={gmm.weights_[1]:.3f}")
        print(f"\n  Optimal threshold (Bayes decision boundary): {threshold:.4f}")
        print(f"  (QuPath clinical default: 0.10)")

        expected_positive_pct = gmm.weights_[1] * 100
        print(f"  Expected Ki67+ fraction: {expected_positive_pct:.1f}%")
    else:
        print(f"\n  Using fallback threshold: {threshold:.4f}")

    print(f"\n  To use this threshold in evaluation:")
    print(f"    --dab_threshold {threshold:.4f}")


if __name__ == "__main__":
    main()
