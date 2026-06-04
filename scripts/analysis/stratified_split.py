#!/usr/bin/env python3
"""
Compute a stratified train/val patient split for ACROBAT dataset.

Stratifies by per-patient per-stain DAB intensity to ensure:
1. Validation patients' DAB values are fully contained within the training range
2. Both splits have reasonable coverage of each stain's intensity distribution
3. Patch counts are approximately 80/20 train/val

Outputs --train_patients and --val_patients arguments for train_acrobat.sh.

Usage:
    python scripts/analysis/stratified_split.py \
        --h5_path /path/to/acrobat_patches.h5 \
        --n_val 6
"""

import argparse
import numpy as np
import h5py

# Standard H-DAB stain vectors (Ruifrok & Johnston, 2001)
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


def load_per_patient_scores(h5_path, max_samples_per_stain=30, seed=42):
    """Return dict: scores[pid][stain] = {'n': int, 'mean': float}"""
    f = h5py.File(h5_path, 'r')
    all_pids = f['index/patient_id'][:]
    unique_pids = sorted(set(int(p) for p in all_pids))
    rng = np.random.default_rng(seed)
    scores = {}
    for pid in unique_pids:
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
                dab_scores = [compute_dab_score(f[stain][idx]) for _, idx in sampled]
                mean_s = float(np.mean(dab_scores))
            else:
                mean_s = float('nan')
            pid_data[stain] = {'n': n_total, 'mean': mean_s}
        scores[pid] = pid_data
    f.close()
    return scores


def stratified_split(scores, n_val=6):
    """Compute a stratified split.

    Algorithm:
    1. For each stain, identify the min and max patients — these MUST be in train
       to prevent validation patients from falling outside the training range.
    2. Rank remaining patients by DAB percentile across available stains.
    3. Pick n_val patients evenly spread across the percentile range.
    4. Validate: no val patient's DAB outside train range for any stain.
    """
    unique_pids = sorted(scores.keys())

    # Step 1: identify must-train patients (min and max for each stain)
    must_train = set()
    for stain in STAINS:
        ranked = sorted(
            [(pid, scores[pid][stain]['mean']) for pid in unique_pids
             if scores[pid][stain]['n'] > 0],
            key=lambda x: x[1]
        )
        if len(ranked) >= 2:
            must_train.add(ranked[0][0])   # min
            must_train.add(ranked[-1][0])  # max

    # Step 2: prefer patients with data for all 4 stains for val
    patients_with_4 = [p for p in unique_pids
                       if sum(1 for s in STAINS if scores[p][s]['n'] > 0) == 4]
    val_pool = [p for p in patients_with_4 if p not in must_train]

    # Step 3: compute mean DAB percentile per candidate
    stain_percentiles = {}
    for stain in STAINS:
        ranked = sorted(
            [(pid, scores[pid][stain]['mean']) for pid in unique_pids
             if scores[pid][stain]['n'] > 0],
            key=lambda x: x[1]
        )
        n = len(ranked)
        for i, (pid, _) in enumerate(ranked):
            stain_percentiles.setdefault(pid, {})[stain] = i / max(1, n - 1)

    pct_scores = {}
    for pid in val_pool:
        pcts = [stain_percentiles[pid][s] for s in STAINS
                if s in stain_percentiles.get(pid, {})]
        if pcts:
            pct_scores[pid] = np.mean(pcts)

    # Step 4: pick n_val patients evenly across the percentile range
    val_sorted = sorted(val_pool, key=lambda p: pct_scores.get(p, 1.0))
    if len(val_sorted) <= n_val:
        val_patients = set(val_sorted)
    else:
        step = len(val_sorted) / n_val
        indices = sorted(set(min(int(i * step), len(val_sorted) - 1)
                             for i in range(n_val)))
        val_patients = {val_sorted[i] for i in indices[:n_val]}

    train_patients = set(unique_pids) - val_patients

    # Step 5: validate
    issues = []
    for stain in STAINS:
        t = [scores[p][stain]['mean'] for p in train_patients
             if scores[p][stain]['n'] > 0]
        v = [scores[p][stain]['mean'] for p in val_patients
             if scores[p][stain]['n'] > 0]
        if t and v:
            t = np.array(t)
            v_arr = np.array(v)
            v_pids = [p for p in val_patients if scores[p][stain]['n'] > 0]
            for i, (pid, x) in enumerate(zip(v_pids, v_arr)):
                if x < t.min():
                    issues.append(f'{stain.upper()}: val pid={pid} DAB={x:.4f} '
                                  f'BELOW train min={t.min():.4f}')
                if x > t.max():
                    issues.append(f'{stain.upper()}: val pid={pid} DAB={x:.4f} '
                                  f'ABOVE train max={t.max():.4f}')

    return train_patients, val_patients, issues


def print_split_details(scores, train, val):
    """Print patch counts and per-stain distribution comparison."""
    print('\n' + '=' * 80)
    print('PATCH COUNTS')
    print('=' * 80)
    print(f'{"Stain":<8} {"Train":<10} {"Val":<10} {"Ratio":<10}'
          f'{"Train μ":<12} {"Val μ":<12}')
    print('-' * 65)
    for stain in STAINS:
        train_n = sum(scores[p][stain]['n'] for p in train
                      if scores[p][stain]['n'] > 0)
        val_n = sum(scores[p][stain]['n'] for p in val
                    if scores[p][stain]['n'] > 0)
        ratio = train_n / max(1, val_n)
        t_means = [scores[p][stain]['mean'] for p in train
                   if scores[p][stain]['n'] > 0]
        v_means = [scores[p][stain]['mean'] for p in val
                   if scores[p][stain]['n'] > 0]
        t_mu = np.mean(t_means) if t_means else 0
        v_mu = np.mean(v_means) if v_means else 0
        print(f'{stain.upper():<8} {train_n:<10} {val_n:<10} {ratio:<10.1f}'
              f'{t_mu:<12.4f} {v_mu:<12.4f}')

    t_total = sum(sum(scores[p][s]['n'] for s in STAINS
                      if scores[p][s]['n'] > 0) for p in train)
    v_total = sum(sum(scores[p][s]['n'] for s in STAINS
                      if scores[p][s]['n'] > 0) for p in val)
    print('-' * 65)
    print(f'{"TOTAL":<8} {t_total:<10} {v_total:<10} '
          f'{t_total/max(1,v_total):<10.1f}')
    print(f'Train/Val: {t_total/(t_total+v_total)*100:.1f}% / '
          f'{v_total/(t_total+v_total)*100:.1f}%')


def main():
    parser = argparse.ArgumentParser(
        description='Compute stratified ACROBAT patient split')
    parser.add_argument('--h5_path', type=str, required=True,
                        help='Path to ACROBAT HDF5 file')
    parser.add_argument('--n_val', type=int, default=15,
                        help='Number of validation patients (default: 15)')
    parser.add_argument('--max_samples', type=int, default=15,
                        help='Max IHC samples per patient per stain (default: 15)')
    args = parser.parse_args()

    print(f'Loading per-patient DAB scores (max_samples={args.max_samples})...')
    scores = load_per_patient_scores(args.h5_path, max_samples_per_stain=args.max_samples)

    train, val, issues = stratified_split(scores, n_val=args.n_val)

    if issues:
        print('\n*** WARNING: Split has validation patients outside '
              'training range ***')
        for issue in issues:
            print(f'  - {issue}')
    else:
        print('\nSplit validated: all val patients within training range for '
              'every stain.')

    print_split_details(scores, train, val)

    print('\n' + '=' * 80)
    print('ARGUMENTS FOR train_acrobat.sh')
    print('=' * 80)
    print(f'--train_patients {" ".join(str(p) for p in sorted(train))} \\')
    print(f'--val_patients {" ".join(str(p) for p in sorted(val))}')


if __name__ == '__main__':
    main()
