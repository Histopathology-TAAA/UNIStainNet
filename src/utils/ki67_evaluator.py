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
                 star_model_name: str = '2D_versatile_he',
                 deepliif_stainer=None):
        self.dab_threshold = dab_threshold
        self._star_model_name = star_model_name
        self._star_model = None
        self._deepliif_stainer = deepliif_stainer

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
        if self._deepliif_stainer is not None:
            # Try to use DeepLIIF Segmentation
            try:
                import torch
                # DeepLIIF expects [1, 3, H, W] in [-1, 1]
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
                
                # Extract segmentation
                seg_mask = self._deepliif_stainer.extract_segmentation(img_tensor)
                
                # Convert back to numpy [0, 255]
                seg_np = ((seg_mask.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
                
                # In DeepLIIF Seg masks:
                # Positive cells are distinctly Red/Brown/Cyan depending on the training setup.
                # Usually: Blue channel > 150 = Negative (Hematoxylin)
                # Red channel > 150 = Positive (Ki67/DAB)
                
                red_channel = seg_np[:, :, 0]
                blue_channel = seg_np[:, :, 2]
                
                pos_mask = (red_channel > 150) & (blue_channel < 100)
                neg_mask = (blue_channel > 150) & (red_channel < 100)
                
                import cv2
                _, pos_labels = cv2.connectedComponents(pos_mask.astype(np.uint8))
                _, neg_labels = cv2.connectedComponents(neg_mask.astype(np.uint8))
                
                positive_cells = len(np.unique(pos_labels)) - 1 # Subtract background
                negative_cells = len(np.unique(neg_labels)) - 1
                total_cells = positive_cells + negative_cells
                
                if total_cells == 0:
                    return 0, 0, 0.0
                    
                labeling_index = (positive_cells / total_cells) * 100.0
                return total_cells, positive_cells, labeling_index
                
            except RuntimeError as e:
                # If they passed the 3-channel G1 model, it will raise a RuntimeError.
                # We fallback to StarDist.
                print(f"[Ki67Evaluator] Falling back to StarDist: {e}")
                pass

        # ── 3. StarDist Fallback ──────────────────────────────────────
        star_model = self._load_star_model()
        labels, _ = star_model.predict_instances(img_np)
        
        unique_ids = np.unique(labels)
        unique_ids = unique_ids[unique_ids != 0]
        total_cells = len(unique_ids)

        if total_cells == 0:
            return 0, 0, 0.0

        dab_map = self._extract_dab_channel(img_np)

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

    # ── 4. Distributional summaries ───────────────────────────────────
    results['ki67_real_li_mean'] = float(np.mean(real))
    results['ki67_real_li_std'] = float(np.std(real))
    results['ki67_fake_li_mean'] = float(np.mean(fake))
    results['ki67_fake_li_std'] = float(np.std(fake))
    results['ki67_n_images'] = int(n)

    return results


def print_ki67_summary(summary: dict):
    """Pretty-print Ki67 clinical metrics to the terminal."""
    tier_labels = {0: 'Low (<10%)', 1: 'Intermediate (10-20%)', 2: 'High (>20%)'}

    print("\n" + "=" * 60)
    print("  Ki67 CLINICAL EVALUATION  (StarDist Silver Standard)")
    print("=" * 60)
    print(f"  Images evaluated      : {summary['ki67_n_images']}")
    print(f"  Real  LI (mean ± std) : {summary['ki67_real_li_mean']:.2f}% ± {summary['ki67_real_li_std']:.2f}%")
    print(f"  Fake  LI (mean ± std) : {summary['ki67_fake_li_mean']:.2f}% ± {summary['ki67_fake_li_std']:.2f}%")
    print("-" * 60)
    print(f"  MAE (LI %)            : {summary['ki67_li_mae']:.2f} ± {summary['ki67_li_mae_std']:.2f}")
    print(f"  Pearson r             : {summary['ki67_li_pearson_r']:.4f}  (p={summary['ki67_li_pearson_p']:.2e})")
    print(f"  Tier Concordance      : {summary['ki67_tier_concordance'] * 100:.1f}%")
    print(f"  Weighted Kappa        : {summary['ki67_tier_kappa']:.4f}")
    print("=" * 60 + "\n")
