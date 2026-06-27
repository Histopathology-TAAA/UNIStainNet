"""
Ki67 Clinical Evaluator — StarDist-based cell-level IHC scoring.

Uses a pre-trained StarDist nuclear segmentation model as an automated
"Silver Standard Referee" to compute the Ki67 Labeling Index (LI) on
both real and synthesized IHC images.  Comparing the two LI distributions
measures whether the generator preserves diagnostic utility.

Pipeline per image:
    1. StarDist nuclear instance segmentation  →  label map
    2. Ruifrok & Johnston H-DAB color deconvolution  →  DAB channel
    3. Per-nucleus mean DAB intensity  →  positive / negative call
    4. LI = (Positive / Total) × 100

References:
    - StarDist: Schmidt et al., "Cell Detection with Star-Convex Polygons"
      (MICCAI 2018)
    - Color Deconvolution: Ruifrok & Johnston, Anal Quant Cytol Histol (2001)
    - Ki67 scoring: Dowsett et al., JNCI (2011) — clinical LI thresholds
"""

import numpy as np


class Ki67ClinicalEvaluator:
    """Automated Ki67 Labeling Index scorer using StarDist + DAB deconvolution.

    The StarDist model is lazy-loaded on the first call to avoid GPU memory
    pressure until actually needed.

    Args:
        dab_threshold: Mean DAB optical-density threshold above which a
            nucleus is classified as Ki67-positive.  Lower values are more
            sensitive (more cells called positive).  Adjust per-dataset to
            account for white-balance / contrast differences.
        star_model_name: Name of the pre-trained StarDist model to load
            (default: ``'2D_versatile_he'``).
    """

    # ── Ruifrok & Johnston H-DAB stain matrix ────────────────────────
    _STAIN_MATRIX = np.array([
        [0.650, 0.704, 0.286],   # Hematoxylin
        [0.268, 0.570, 0.776],   # DAB
    ], dtype=np.float64)

    def __init__(self, dab_threshold: float = 0.15,
                 star_model_name: str = '2D_versatile_fluo',
                 deepliif_stainer=None,
                 eval_method: str = 'deepliif',
                 seg_thresh: int = 130,
                 marker_thresh: str = 'default'):
        self.dab_threshold = dab_threshold
        self._star_model_name = star_model_name
        self._star_model = None
        self._deepliif_stainer = deepliif_stainer
        self.eval_method = eval_method
        self.seg_thresh = seg_thresh
        self.marker_thresh = None if marker_thresh == 'None' else (marker_thresh if marker_thresh == 'default' else int(marker_thresh))

        # Pre-compute the pseudo-inverse of the stain matrix once
        # stain_matrix is (2, 3); we need (3, 2) pinv for deconvolution
        self._deconv_matrix = np.linalg.pinv(self._STAIN_MATRIX)  # (3, 2)

    # ------------------------------------------------------------------
    # Lazy-load StarDist
    # ------------------------------------------------------------------
    def _load_star_model(self):
        """Instantiate StarDist model on first use."""
        if self._star_model is not None:
            return self._star_model

        from stardist.models import StarDist2D
        self._star_model = StarDist2D.from_pretrained(self._star_model_name)
        print(f"[Ki67Evaluator] Loaded StarDist model: {self._star_model_name}")
        return self._star_model

    # ------------------------------------------------------------------
    # DAB channel extraction (numpy, CPU-only)
    # ------------------------------------------------------------------
    def _extract_dab_channel(self, rgb_01: np.ndarray) -> np.ndarray:
        """Isolate the DAB optical-density channel via color deconvolution.

        Args:
            rgb_01: (H, W, 3) float image in [0, 1].

        Returns:
            dab_od: (H, W) DAB optical density (non-negative).
        """
        # Optical density: OD = -log10(I), clamp to avoid log(0)
        od = -np.log10(np.clip(rgb_01, 1e-6, 1.0))          # (H, W, 3)

        # Deconvolve → stain concentrations
        # od_flat @ deconv_matrix → (N, 2) where col 1 = DAB
        H, W, _ = od.shape
        od_flat = od.reshape(-1, 3)                          # (N, 3)
        concentrations = od_flat @ self._deconv_matrix        # (N, 2)

        dab = concentrations[:, 1].reshape(H, W)             # DAB column
        return np.clip(dab, 0.0, None)                       # non-negative

    def _extract_hema_channel(self, rgb_01: np.ndarray) -> np.ndarray:
        """Isolate the Hematoxylin optical-density channel."""
        od = -np.log10(np.clip(rgb_01, 1e-6, 1.0))
        H, W, _ = od.shape
        od_flat = od.reshape(-1, 3)
        concentrations = od_flat @ self._deconv_matrix
        hema = concentrations[:, 0].reshape(H, W)
        return np.clip(hema, 0.0, None)

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------
    def compute_labeling_index(self, ihc_image):
        """Score a single IHC image for Ki67 Labeling Index.

        Args:
            ihc_image: One of the following formats:
                - ``torch.Tensor`` of shape ``[3, H, W]`` in ``[-1, 1]``
                - ``np.ndarray`` of shape ``(H, W, 3)`` in ``[0, 1]`` or
                  ``[0, 255]``

        Returns:
            total_cells (int):   Total nuclei detected by StarDist.
            positive_cells (int): Nuclei exceeding the DAB threshold.
            labeling_index (float): ``(positive / total) * 100`` (%).
                Returns 0.0 if no cells are detected.
        """
        # ── 1. Normalise to numpy (H, W, 3) float32 in [0, 1] ────────
        import torch
        if isinstance(ihc_image, torch.Tensor):
            img_np = ihc_image.detach().cpu().float().numpy()
            # (C, H, W) → (H, W, C)
            if img_np.ndim == 3 and img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            # [-1, 1] → [0, 1]
            if img_np.min() < 0:
                img_np = (img_np + 1.0) / 2.0
        else:
            img_np = np.asarray(ihc_image, dtype=np.float32)
            if img_np.max() > 1.5:          # likely [0, 255]
                img_np = img_np / 255.0

        img_np = np.clip(img_np, 0.0, 1.0).astype(np.float32)

        # ── 2. DeepLIIF Segmentation vs StarDist ──────────────────────
        if self.eval_method == 'deepliif' and self._deepliif_stainer is not None:
            # Try to use DeepLIIF Segmentation
            try:
                import torch
                from deepliif.postprocessing import compute_final_results
                
                # DeepLIIF expects [1, 3, H, W] in [-1, 1]
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
                if torch.cuda.is_available():
                    img_tensor = img_tensor.cuda()
                
                # Extract all modalities
                modalities = self._deepliif_stainer.extract_all_modalities(img_tensor)
                seg_mask = modalities['Segmentation']
                marker_mask = modalities['mpIHC_Ki67']
                
                # Convert back to numpy [0, 255]
                seg_np = ((seg_mask.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
                marker_np = ((marker_mask.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
                img_np_255 = (img_np * 255).astype(np.uint8)
                
                # Use DeepLIIF's official post-processing
                _, _, scoring = compute_final_results(
                    orig=img_np_255,
                    seg=seg_np,
                    marker=marker_np,
                    resolution='40x', # MIST dataset is 40x
                    size_thresh='default',
                    size_thresh_upper=None,
                    seg_thresh=self.seg_thresh,
                    marker_thresh=self.marker_thresh
                )
                
                total_cells = scoring['num_total']
                positive_cells = scoring['num_pos']
                
                if total_cells == 0:
                    return 0, 0, 0.0
                    
                labeling_index = scoring['percent_pos']
                return total_cells, positive_cells, labeling_index
                
            except Exception as e:
                # DeepLIIF ensemble failed — permanently fall back to StarDist
                print(f"\n[Ki67Evaluator] *** DeepLIIF FAILED — falling back to StarDist ***")
                print(f"[Ki67Evaluator] Error: {type(e).__name__}: {e}")
                self.eval_method = 'stardist'  # Permanently disable for remaining images

        # ── 3. StarDist Fallback (Grayscale Fluorescence Pipeline) ────
        star_model = self._load_star_model()
        
        # Extract individual stain densities
        dab_map = self._extract_dab_channel(img_np)
        hema_map = self._extract_hema_channel(img_np)
        
        # Total structural density = Hematoxylin + DAB
        # This converts a pale color image into a high-contrast grayscale image
        # where ALL nuclei (blue and brown) are bright blobs on a dark background.
        structural_density = hema_map + dab_map
        
        # Normalize for StarDist fluorescence model (expecting values roughly 0 to 1 with outliers)
        from csbdeep.utils import normalize
        img_norm = normalize(structural_density, 1, 99.8, axis=(0,1))
        
        labels, _ = star_model.predict_instances(img_norm)
        
        unique_ids = np.unique(labels)
        unique_ids = unique_ids[unique_ids != 0]
        total_cells = len(unique_ids)

        if total_cells == 0:
            return 0, 0, 0.0

        positive_cells = 0
        for cell_id in unique_ids:
            mask = labels == cell_id
            mean_dab = dab_map[mask].mean()
            if mean_dab > self.dab_threshold:
                positive_cells += 1

        labeling_index = (positive_cells / total_cells) * 100.0
        return total_cells, positive_cells, labeling_index

    # ------------------------------------------------------------------
    # Batch convenience
    # ------------------------------------------------------------------
    def compute_batch(self, images):
        """Score a batch of IHC images.

        Args:
            images: ``torch.Tensor`` of shape ``[B, 3, H, W]`` in ``[-1, 1]``
                or a list / iterable of single-image tensors / arrays.

        Returns:
            list of ``(total_cells, positive_cells, labeling_index)`` tuples.
        """
        import torch
        results = []
        if isinstance(images, torch.Tensor) and images.ndim == 4:
            for i in range(images.shape[0]):
                results.append(self.compute_labeling_index(images[i]))
        else:
            for img in images:
                results.append(self.compute_labeling_index(img))
        return results


# ======================================================================
# Summary statistics (called once after full eval pass)
# ======================================================================

def compute_ki67_summary(real_li_scores, fake_li_scores):
    """Compute global Ki67 clinical agreement metrics.

    Args:
        real_li_scores: list/array of per-image Real LI percentages.
        fake_li_scores: list/array of per-image Synthesized LI percentages.

    Returns:
        dict: JSON-serialisable metric dictionary suitable for W&B / TB.
    """
    from scipy.stats import pearsonr
    from sklearn.metrics import cohen_kappa_score

    real = np.asarray(real_li_scores, dtype=np.float64)
    fake = np.asarray(fake_li_scores, dtype=np.float64)
    n = len(real)

    results = {}

    # ── 1. Mean Absolute Error ────────────────────────────────────────
    abs_diffs = np.abs(real - fake)
    results['ki67_li_mae'] = float(np.mean(abs_diffs))
    results['ki67_li_mae_std'] = float(np.std(abs_diffs))

    # ── 2. Pearson correlation ────────────────────────────────────────
    if n >= 3:
        r, p = pearsonr(real, fake)
        results['ki67_li_pearson_r'] = float(r)
        results['ki67_li_pearson_p'] = float(p)
    else:
        results['ki67_li_pearson_r'] = float('nan')
        results['ki67_li_pearson_p'] = float('nan')

    # ── 3. Clinical tier concordance ──────────────────────────────────
    def _to_tier(li):
        """Map LI% → clinical risk tier (0 = Low, 1 = Intermediate, 2 = High)."""
        if li < 10.0:
            return 0
        elif li <= 20.0:
            return 1
        else:
            return 2

    real_tiers = np.array([_to_tier(x) for x in real])
    fake_tiers = np.array([_to_tier(x) for x in fake])

    concordance = float(np.mean(real_tiers == fake_tiers))
    results['ki67_tier_concordance'] = concordance

    # Weighted Cohen's kappa (ordinal agreement)
    if len(np.unique(real_tiers)) >= 2 or len(np.unique(fake_tiers)) >= 2:
        try:
            kappa = cohen_kappa_score(real_tiers, fake_tiers, weights='linear')
            results['ki67_tier_kappa'] = float(kappa)
        except Exception:
            results['ki67_tier_kappa'] = float('nan')
    else:
        results['ki67_tier_kappa'] = float('nan')

    # ── 4. ICC — Intraclass Correlation Coefficient ────────────────────
    # ICC(2,1): two-way random, single rater — measures absolute agreement.
    # Widely used in 2024-2025 Ki67 DIA papers as the primary metric.
    if n >= 3:
        try:
            import pandas as pd
            import pingouin as pg

            df = pd.DataFrame({
                'Subject': np.tile(np.arange(n), 2),
                'Rater': ['Real'] * n + ['Fake'] * n,
                'LI': np.concatenate([real, fake]),
            })
            icc_result = pg.intraclass_corr(
                data=df, targets='Subject', raters='Rater', ratings='LI',
            )
            icc_row = icc_result[icc_result['Type'] == 'ICC2']
            if len(icc_row) > 0:
                results['ki67_icc'] = float(icc_row['ICC'].values[0])
                try:
                    ci = icc_row['CI95%'].values[0]
                    if ci is not None and len(ci) == 2:
                        results['ki67_icc_ci95_low'] = float(ci[0])
                        results['ki67_icc_ci95_high'] = float(ci[1])
                except Exception:
                    pass
            else:
                results['ki67_icc'] = float('nan')
        except Exception:
            results['ki67_icc'] = float('nan')

    # ── 5. Multi-tier concordance (3-tier + 4-tier) ──────────────────
    def _to_tier_3(li):
        if li < 10.0:   return 0
        elif li <= 20.0: return 1
        else:            return 2

    def _to_tier_4(li):
        if li < 6.0:     return 0
        elif li <= 20.0: return 1
        elif li <= 50.0: return 2
        else:            return 3

    for name, fn in [('3tier', _to_tier_3), ('4tier', _to_tier_4)]:
        rt = np.array([fn(x) for x in real])
        ft = np.array([fn(x) for x in fake])
        results[f'ki67_{name}_concordance'] = float(np.mean(rt == ft))
        if len(np.unique(rt)) >= 2 or len(np.unique(ft)) >= 2:
            try:
                from sklearn.metrics import cohen_kappa_score
                results[f'ki67_{name}_kappa'] = float(cohen_kappa_score(rt, ft, weights='linear'))
            except Exception:
                results[f'ki67_{name}_kappa'] = float('nan')
        else:
            results[f'ki67_{name}_kappa'] = float('nan')

    # Backward compatibility aliases
    results['ki67_tier_concordance'] = results['ki67_3tier_concordance']
    results['ki67_tier_kappa'] = results['ki67_3tier_kappa']

    # ── 6. Distributional summaries ───────────────────────────────────
    results['ki67_real_li_mean'] = float(np.mean(real))
    results['ki67_real_li_std'] = float(np.std(real))
    results['ki67_fake_li_mean'] = float(np.mean(fake))
    results['ki67_fake_li_std'] = float(np.std(fake))
    results['ki67_n_images'] = int(n)

    return results


def compute_hotspot_analysis(real_li_scores, fake_li_scores, n_patches_per_image=4):
    """Per-image hotspot concordance analysis.

    For each image, the LI was computed per 512×512 patch. If multiple patches
    belong to the same WSI, identify the hotspot (highest-LI patch) per image
    and compare hotspot metrics.

    Args:
        real_li_scores: [N] real LI percentages (per-patch)
        fake_li_scores: [N] fake LI percentages (per-patch)
        n_patches_per_image: number of patches per WSI (default 4 = 2×2 grid)

    Returns:
        dict with hotspot metrics, or empty dict if grouping not applicable
    """
    real = np.asarray(real_li_scores, dtype=np.float64)
    fake = np.asarray(fake_li_scores, dtype=np.float64)
    n = len(real)

    if n % n_patches_per_image != 0 or n_patches_per_image <= 1:
        return {}  # can't group into per-image hotspots

    n_images = n // n_patches_per_image
    real_2d = real.reshape(n_images, n_patches_per_image)
    fake_2d = fake.reshape(n_images, n_patches_per_image)

    # Hotspot = patch with highest LI per image
    real_hotspots = real_2d.max(axis=1)
    fake_hotspots = fake_2d.max(axis=1)

    # Hotspot location: which patch index has the max?
    real_hotspot_idx = real_2d.argmax(axis=1)
    fake_hotspot_idx = fake_2d.argmax(axis=1)
    location_match = float(np.mean(real_hotspot_idx == fake_hotspot_idx))

    # Heterogeneity: hotspot / mean ratio
    real_hetero = real_hotspots / (real_2d.mean(axis=1) + 1e-6)
    fake_hetero = fake_hotspots / (fake_2d.mean(axis=1) + 1e-6)

    # Hotspot metrics
    from scipy.stats import pearsonr
    r_hs, p_hs = pearsonr(real_hotspots, fake_hotspots)
    mae_hs = float(np.mean(np.abs(real_hotspots - fake_hotspots)))

    results = {
        'ki67_hotspot_pearson_r': float(r_hs),
        'ki67_hotspot_mae': mae_hs,
        'ki67_hotspot_location_match': location_match,
        'ki67_hotspot_real_mean': float(np.mean(real_hotspots)),
        'ki67_hotspot_fake_mean': float(np.mean(fake_hotspots)),
        'ki67_heterogeneity_mae': float(np.mean(np.abs(real_hetero - fake_hetero))),
        'ki67_n_images_hotspot': int(n_images),
    }
    return results


def print_ki67_summary(summary: dict, method: str = 'stardist'):
    """Pretty-print Ki67 clinical metrics to the terminal."""
    title = "DeepLIIF Gold Standard" if method == 'deepliif' else "StarDist Silver Standard"

    print("\n" + "=" * 60)
    print(f"  Ki67 CLINICAL EVALUATION  ({title})")
    print("=" * 60)
    print(f"  Images evaluated      : {summary['ki67_n_images']}")
    print(f"  Real  LI (mean ± std) : {summary['ki67_real_li_mean']:.2f}% ± {summary['ki67_real_li_std']:.2f}%")
    print(f"  Fake  LI (mean ± std) : {summary['ki67_fake_li_mean']:.2f}% ± {summary['ki67_fake_li_std']:.2f}%")
    print("-" * 60)
    print(f"  MAE (LI %)            : {summary['ki67_li_mae']:.2f} ± {summary['ki67_li_mae_std']:.2f}")
    print(f"  Pearson r             : {summary['ki67_li_pearson_r']:.4f}  (p={summary['ki67_li_pearson_p']:.2e})")
    if 'ki67_icc' in summary and not np.isnan(summary['ki67_icc']):
        print(f"  ICC(2,1)              : {summary['ki67_icc']:.4f}  (95% CI: {summary.get('ki67_icc_ci95_low', np.nan):.4f}–{summary.get('ki67_icc_ci95_high', np.nan):.4f})")
    if 'ki67_3tier_concordance' in summary:
        print(f"  Tier Concordance (3)  : {summary['ki67_3tier_concordance'] * 100:.1f}%")
        print(f"  Weighted Kappa (3)    : {summary.get('ki67_3tier_kappa', summary.get('ki67_tier_kappa', float('nan'))):.4f}")
    if 'ki67_4tier_concordance' in summary and not np.isnan(summary.get('ki67_4tier_concordance', float('nan'))):
        print(f"  Tier Concordance (4)  : {summary['ki67_4tier_concordance'] * 100:.1f}%")
        print(f"  Weighted Kappa (4)    : {summary.get('ki67_4tier_kappa', float('nan')):.4f}")
    # Hotspot if available
    if 'ki67_hotspot_pearson_r' in summary and not np.isnan(summary['ki67_hotspot_pearson_r']):
        print(f"  Hotspot Pearson-r     : {summary['ki67_hotspot_pearson_r']:.4f}")
        print(f"  Hotspot Location Match: {summary['ki67_hotspot_location_match'] * 100:.1f}%")
        print(f"  Hotspot MAE (LI %)    : {summary['ki67_hotspot_mae']:.2f}")
    print("=" * 60 + "\n")
