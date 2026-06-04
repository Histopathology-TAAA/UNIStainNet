#!/usr/bin/env python3
"""
Compute per-patient, per-stain DAB statistics from an ACROBAT HDF5 file.

Uses numpy color deconvolution (Ruifrok & Johnston, 2001) for fast batch
computation without GPU/PyTorch dependency.

Usage:
    python scripts/analysis/dab_statistics.py \
        --h5_path /path/to/acrobat_patches.h5 \
        --train_patients 1 10 12 ... \
        --val_patients 0 11 ...

Outputs a table of per-patient DAB means and a train/val distribution summary.
"""

import argparse
import numpy as np
import h5py

# Standard H-DAB stain vectors (Ruifrok & Johnston)
STAIN_MATRIX = np.array([
    [0.268, 0.570, 0.776],  # DAB (brown)
    [0.650, 0.704, 0.286],  # Hematoxylin (blue)
])
DECONV = np.linalg.pinv(STAIN_MATRIX.T)  # (2, 3)

STAINS = ['her2', 'ki67', 'er', 'pgr']


def compute_dab_score(img):
    """Compute p90 DAB score for a (H, W, 3) uint8 IHC image."""
    rgb = img.astype(np.float32) / 255.0
    rgb = np.clip(rgb, 1e-6, 1.0)
    od = -np.log10(rgb)
    conc = od.reshape(-1, 3) @ DECONV.T
    dab = conc[:, 0].reshape(img.shape[0], img.shape[1])
    dab = np.log1p(np.exp(np.clip(dab * 5.0, -20, 20))) / 5.0
    flat = dab.flatten()
    p90 = np.quantile(flat, 0.9)
    mask = flat >= p90
    return flat[mask].mean() if mask.sum() > 0 else flat.mean()


def get_per_patient_scores(h5_path, train_patients, val_patients,
                           max_samples_per_stain=30, seed=42):
    """Return dict: all_data[pid][stain] = {'n': int, 'mean': float}"""
    f = h5py.File(h5_path, 'r')
    all_pids = f['index/patient_id'][:]
    unique_pids = sorted(set(int(p) for p in all_pids))
    rng = np.random.default_rng(seed)

    all_data = {}
    for pid in sorted(unique_pids):
        he_indices = set(np.where(all_pids == pid)[0])
        pid_data = {}
        for stain in STAINS:
            pairs = f[f'index/pair_{stain}'][:]
            mask = np.isin(pairs[:, 0], list(he_indices))
            stain_pairs = pairs[mask]
            n_total = len(stain_pairs)
            if n_total > 0:
                n_sample = min(max_samples_per_stain, n_total)
                sampled = rng.choice(stain_pairs, size=n_sample, replace=False)
                scores = [compute_dab_score(f[stain][idx]) for _, idx in sampled]
                mean_s = float(np.mean(scores))
            else:
                mean_s = float('nan')
            pid_data[stain] = {'n': n_total, 'mean': mean_s}
        all_data[pid] = pid_data

    f.close()
    return all_data


def print_table(all_data, train_patients, val_patients):
    """Print per-patient table."""
    print('=' * 90)
    print('PER-PATIENT, PER-STAIN DAB STATISTICS')
    print('=' * 90)
    header = f'{"Patient":<10} {"Set":<6}'
    for s in STAINS:
        header += f' {"n_"+s:<6} {"DAB_"+s:<10}'
    print(header)
    print('-' * 90)
    for pid in sorted(all_data):
        split = 'TRAIN' if pid in train_patients else ('VAL' if pid in val_patients else 'OTHER')
        row = f'{pid:<10} {split:<6}'
        for s in STAINS:
            d = all_data[pid][s]
            row += f' {d["n"]:<6} {d["mean"]:<10.4f}'
        print(row)


def print_summary(all_data, train_patients, val_patients):
    """Print train/val distribution comparison."""
    print()
    print('=' * 90)
    print('TRAIN vs VAL POPULATION SUMMARY')
    print('=' * 90)
    for split_name, split_pids in [('TRAIN', train_patients), ('VAL', val_patients)]:
        print(f'\n--- {split_name} ({len(split_pids)} patients) ---')
        for stain in STAINS:
            means = [all_data[p][stain]['mean'] for p in split_pids
                     if p in all_data and all_data[p][stain]['n'] > 0]
            ns = [all_data[p][stain]['n'] for p in split_pids
                  if p in all_data and all_data[p][stain]['n'] > 0]
            if means:
                means = np.array(means)
                print(f'  {stain.upper():6s}: {len(means)} pts, {sum(ns)} patches, '
                      f'μ={means.mean():.4f} σ={means.std():.4f} '
                      f'min={means.min():.4f} max={means.max():.4f}')

    print()
    print('=' * 90)
    print('STAIN-BY-STAIN: TRAIN vs VAL DISTRIBUTION')
    print('=' * 90)
    for stain in STAINS:
        t = np.array([all_data[p][stain]['mean'] for p in train_patients
                      if p in all_data and all_data[p][stain]['n'] > 0])
        v = np.array([all_data[p][stain]['mean'] for p in val_patients
                      if p in all_data and all_data[p][stain]['n'] > 0])
        if len(t) and len(v):
            print(f'\n{stain.upper()}:')
            print(f'  Train: μ={t.mean():.4f}, σ={t.std():.4f}, '
                  f'range=[{t.min():.4f}, {t.max():.4f}]')
            print(f'  Val:   μ={v.mean():.4f}, σ={v.std():.4f}, '
                  f'range=[{v.min():.4f}, {v.max():.4f}]')
            delta = (v.mean() - t.mean()) / max(t.mean(), 1e-6) * 100
            print(f'  Δμ = {v.mean()-t.mean():.4f} ({delta:.1f}%)')
            below = [x for x in v if x < t.min()]
            above = [x for x in v if x > t.max()]
            if below:
                print(f'  *** {len(below)} val pts BELOW train min ({t.min():.4f})')
            if above:
                print(f'  *** {len(above)} val pts ABOVE train max ({t.max():.4f})')


def main():
    parser = argparse.ArgumentParser(
        description='Compute per-patient per-stain DAB statistics from ACROBAT HDF5')
    parser.add_argument('--h5_path', type=str, required=True,
                        help='Path to ACROBAT HDF5 file')
    parser.add_argument('--train_patients', nargs='+', type=int, required=True,
                        help='Training patient IDs')
    parser.add_argument('--val_patients', nargs='+', type=int, required=True,
                        help='Validation patient IDs')
    parser.add_argument('--max_samples', type=int, default=30,
                        help='Max patches to sample per patient per stain (default: 30)')
    args = parser.parse_args()

    train_set = set(args.train_patients)
    val_set = set(args.val_patients)

    all_data = get_per_patient_scores(
        args.h5_path, train_set, val_set,
        max_samples_per_stain=args.max_samples)

    print_table(all_data, train_set, val_set)
    print_summary(all_data, train_set, val_set)


if __name__ == '__main__':
    main()
