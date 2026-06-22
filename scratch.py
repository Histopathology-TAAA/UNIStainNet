import torch
import numpy as np
from src.utils.dab import DABExtractor, HEExtractor

def compute_nmi(h_g, h_r, n_bins=256):
    h_g_norm = (h_g - h_g.min()) / (h_g.max() - h_g.min() + 1e-6)
    h_r_norm = (h_r - h_r.min()) / (h_r.max() - h_r.min() + 1e-6)
    hist_2d, _, _ = np.histogram2d(h_g_norm, h_r_norm, bins=n_bins, range=[[0, 1], [0, 1]])
    hist_2d = hist_2d / hist_2d.sum()
    p_g = hist_2d.sum(axis=1)
    p_r = hist_2d.sum(axis=0)
    H_g = -np.sum(p_g[p_g > 0] * np.log2(p_g[p_g > 0]))
    H_r = -np.sum(p_r[p_r > 0] * np.log2(p_r[p_r > 0]))
    MI = 0.0
    for gi in range(len(p_g)):
        for ri in range(len(p_r)):
            if hist_2d[gi, ri] > 0 and p_g[gi] > 0 and p_r[ri] > 0:
                MI += hist_2d[gi, ri] * np.log2(hist_2d[gi, ri] / (p_g[gi] * p_r[ri]))
    return 2.0 * MI / (H_g + H_r + 1e-6)

# create synthetic correlated data with outliers
h_r = np.random.normal(0.5, 0.1, 512*512).clip(0, 1)
h_g = h_r + np.random.normal(0, 0.05, 512*512)

print("NMI without outliers:", compute_nmi(h_g, h_r))

# add an extreme outlier
h_g[0] = 100.0
h_r[0] = 100.0
print("NMI with outliers (min-max norm):", compute_nmi(h_g, h_r))

# test with robust normalization
def compute_nmi_robust(h_g, h_r, n_bins=256):
    p1_g, p99_g = np.percentile(h_g, [1, 99])
    p1_r, p99_r = np.percentile(h_r, [1, 99])
    h_g_norm = np.clip((h_g - p1_g) / (p99_g - p1_g + 1e-6), 0, 1)
    h_r_norm = np.clip((h_r - p1_r) / (p99_r - p1_r + 1e-6), 0, 1)
    hist_2d, _, _ = np.histogram2d(h_g_norm, h_r_norm, bins=n_bins, range=[[0, 1], [0, 1]])
    hist_2d = hist_2d / hist_2d.sum()
    p_g = hist_2d.sum(axis=1)
    p_r = hist_2d.sum(axis=0)
    H_g = -np.sum(p_g[p_g > 0] * np.log2(p_g[p_g > 0]))
    H_r = -np.sum(p_r[p_r > 0] * np.log2(p_r[p_r > 0]))
    MI = 0.0
    for gi in range(len(p_g)):
        for ri in range(len(p_r)):
            if hist_2d[gi, ri] > 0 and p_g[gi] > 0 and p_r[ri] > 0:
                MI += hist_2d[gi, ri] * np.log2(hist_2d[gi, ri] / (p_g[gi] * p_r[ri]))
    return 2.0 * MI / (H_g + H_r + 1e-6)

print("NMI with outliers (robust norm):", compute_nmi_robust(h_g, h_r))

