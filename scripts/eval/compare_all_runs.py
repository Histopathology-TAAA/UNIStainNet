#!/usr/bin/env python3
"""
Compile all evaluation results into a single comprehensive comparison table.

Usage:
    PYTHONPATH=. python scripts/eval/compare_all_runs.py

Reads every results.json under eval_output/ and produces a table showing
every metric across every run, plus the configuration that produced it.
"""

import json
from pathlib import Path
from collections import OrderedDict

EVAL_DIR = Path("eval_output")

# Which eval directories to include (sorted chronologically)
RUNS = OrderedDict([
    # (display_name, path, config_line)
    ("Baseline", EVAL_DIR / "baseline_ki67_new_metrics", "L1=1.0, Edge=0.5, No DeepLIIF, No Align"),
    ("Run 5 ⚠",  EVAL_DIR / "ki67_deepliif_run5",       "Warmup=60, Anneal=40, End=0.05, No DeepLIIF, Align"),
    ("Run 6 ⚠",  EVAL_DIR / "ki67_deepliif_run6",       "Run5 + DeepLIIF"),
    ("Run 7 ⚠",  EVAL_DIR / "ki67_deepliif_run7",       "Run6 + Adv(0.5) + Gram(0.5) — COLLAPSED"),
    ("Run 8",    EVAL_DIR / "ki67_deepliif_run8_final", "Run6 − L1(1.0→0.5) + Edge(0.5→2.0) + LPIPS512(0.25) + RGBdrop(10%)"),
    ("PV1",     EVAL_DIR / "ki67_pv1",                   "Run8 + MLPA(0.5) + class_dim(256)"),
    ("PV2",     EVAL_DIR / "ki67_pv2",                   "PV1 + CTPC(2.5) + DABint(0) + MLPA(1.0)"),
    ("PV4",     EVAL_DIR / "ki67_pv4",                   "PV1 + Gram(0.5) + class_dim(256)"),
    ("Run 9",   EVAL_DIR / "ki67_run9",                  "Run8 + MLPA(0.5) + GP(1.0) − L1_lowres − DABint(0)"),
    ("Run 10",  EVAL_DIR / "ki67_run10",                 "Run9 + PatchNCE(0.1)"),
    ("Run 11",  EVAL_DIR / "ki67_run11",                 "Run10 + GPweights(misalign-aware) + L1(0.05)"),
    ("Run 12",  None,                                     "Run11 + SE-Net + CoordAttn + MLPA(0.75) + L1(0.1) + GP512(0.6) — RUNNING"),
])

# Metrics to extract, grouped by category
METRICS = OrderedDict([
    # -- Image Quality --
    ("FID ↓",              "image_quality", "fid_inception", False),
    ("UNI-FID ↓",          "image_quality", "fid_uni",       False),
    ("KID ↓",              "image_quality", "kid_mean_x1000", True),
    ("LPIPS ↓",            "image_quality", "lpips_mean",    True),
    ("SSIM ↑",             "image_quality", "ssim_mean",     True),
    ("PSNR ↑",             "image_quality", "psnr_mean",     True),
    # -- Structure --
    ("HE-H-SSIM ↑",        "he_structure",  "he_h_ssim_mean", True),
    ("HE-NMI ↑",           "he_structure",  "he_nmi_mean",    True),
    # -- DAB / Protein --
    ("DAB MAE ↓",          "dab",           "dab_mae_overall", True),
    ("DAB Pearson-r ↑",    "dab",           "dab_pearson_r",  True),
    ("DAB KL ↓",           "dab",           "dab_kl",         True),
    ("DAB JSD ↓",          "dab",           "dab_jsd",        True),
    ("IOD Diff ↓",         "iod",           "miod_diff",      False),
    ("IOD Abs Diff ↓",     "iod",           "miod_abs_diff",  True),
    # -- Clinical (DeepLIIF 80/80/80) --
    ("ICC ↑",              "ki67_clinical", "ki67_icc",       True),
    ("MAE (LI%) ↓",        "ki67_clinical", "ki67_li_mae",   True),
    ("Pearson-r (LI) ↑",   "ki67_clinical", "ki67_li_pearson_r", True),
    ("3-Tier Concord ↑",   "ki67_clinical", "ki67_3tier_concordance", True),
    ("4-Tier Concord ↑",   "ki67_clinical", "ki67_4tier_concordance", True),
    ("Kappa (3) ↑",        "ki67_clinical", "ki67_3tier_kappa", True),
])


def load_metrics(run_path):
    """Load metrics from a results.json file."""
    json_path = run_path / "results.json"
    if not json_path.exists():
        return None

    with open(json_path) as f:
        data = json.load(f)

    ki67 = data.get("per_stain", {}).get("Ki67", {})
    if not ki67:
        return None

    metrics = {}
    for display_name, section, key, precise in METRICS.values():
        # Actually, we need to track the mapping differently
        pass

    return ki67


def main():
    rows = []

    for name, path, config in RUNS.items():
        if path is None:
            # Running — placeholder
            rows.append({"Run": name, "Config": config, "Status": "⏳ RUNNING"})
            continue

        if not path.exists():
            rows.append({"Run": name, "Config": config, "Status": "⚠ NO EVAL"})
            continue

        ki67 = load_metrics(path)
        if ki67 is None:
            rows.append({"Run": name, "Config": config, "Status": "⚠ NO JSON"})
            continue

        row = {"Run": name, "Config": config, "Status": "✅"}

        for display_name, section, key, is_better_if_higher in METRICS.items():
            val = ki67.get(section, {}).get(key)
            if val is None:
                # Try top-level key (for UNI-FID etc nested differently)
                val = ki67.get(key)
            if val is None:
                row[display_name] = "—"
            elif isinstance(val, float):
                if abs(val) < 0.001 and val != 0.0:
                    row[display_name] = f"{val:.2e}"
                elif abs(val) < 1:
                    row[display_name] = f"{val:.4f}"
                elif abs(val) < 10:
                    row[display_name] = f"{val:.3f}"
                else:
                    row[display_name] = f"{val:.1f}"
            else:
                row[display_name] = str(val)

        rows.append(row)

    # -- Print Image Quality --
    print("\n" + "=" * 120)
    print("  IMAGE QUALITY")
    print("=" * 120)
    iq_metrics = ["FID ↓", "UNI-FID ↓", "KID ↓", "LPIPS ↓", "SSIM ↑", "PSNR ↑"]
    header = f"{'Run':<14}"
    for m in iq_metrics:
        header += f" {m:>12}"
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row['Run']:<14}"
        for m in iq_metrics:
            line += f" {row.get(m, '—'):>12}"
        print(line)

    # -- Print Structure --
    print("\n" + "=" * 120)
    print("  STRUCTURE")
    print("=" * 120)
    st_metrics = ["HE-H-SSIM ↑", "HE-NMI ↑"]
    header = f"{'Run':<14}"
    for m in st_metrics:
        header += f" {m:>16}"
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row['Run']:<14}"
        for m in st_metrics:
            line += f" {row.get(m, '—'):>16}"
        print(line)

    # -- Print DAB / Protein --
    print("\n" + "=" * 120)
    print("  DAB / PROTEIN")
    print("=" * 120)
    dab_metrics = ["DAB MAE ↓", "DAB Pearson-r ↑", "DAB KL ↓", "DAB JSD ↓", "IOD Abs Diff ↓"]
    header = f"{'Run':<14}"
    for m in dab_metrics:
        header += f" {m:>16}"
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row['Run']:<14}"
        for m in dab_metrics:
            line += f" {row.get(m, '—'):>16}"
        print(line)

    # -- Print Clinical --
    print("\n" + "=" * 120)
    print("  CLINICAL (DeepLIIF 80/80/80)")
    print("=" * 120)
    clin_metrics = ["ICC ↑", "MAE (LI%) ↓", "Pearson-r (LI) ↑", "3-Tier Concord ↑", "4-Tier Concord ↑", "Kappa (3) ↑"]
    header = f"{'Run':<14}"
    for m in clin_metrics:
        header += f" {m:>16}"
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row['Run']:<14}"
        for m in clin_metrics:
            line += f" {row.get(m, '—'):>16}"
        print(line)

    # -- Print config summary --
    print("\n" + "=" * 120)
    print("  CONFIGURATION")
    print("=" * 120)
    for row in rows:
        print(f"  {row['Run']:<14} {row['Config']}")

    # -- Print best values --
    print("\n" + "=" * 120)
    print("  BEST VALUE PER METRIC")
    print("=" * 120)
    all_metrics = iq_metrics + st_metrics + dab_metrics + clin_metrics
    for m in all_metrics:
        best_run = None
        best_val = None
        for row in rows:
            val = row.get(m, "—")
            if val == "—" or val is None:
                continue
            try:
                fval = float(val)
            except ValueError:
                continue
            # Determine direction
            is_high = "↑" in m
            if best_val is None or (is_high and fval > best_val) or (not is_high and fval < best_val):
                best_val = fval
                best_run = row["Run"]
        if best_run:
            print(f"  {m:<20} {best_val}  ({best_run})")

    print()


if __name__ == "__main__":
    main()
