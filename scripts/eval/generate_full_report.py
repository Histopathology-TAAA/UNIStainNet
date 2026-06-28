#!/usr/bin/env python3
"""
Scan ALL eval_output directories and generate a complete comparison report.

Handles:
- Multiple stains per dir (ER, PR, Ki67, HER2 in any case)
- Aligned vs normal evaluation
- Various result JSON formats across code versions
- Missing metrics gracefully

Usage:
    PYTHONPATH=. python scripts/eval/generate_full_report.py
    Output: reports/full_eval_report.md
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict

EVAL_DIR = Path("eval_output")
OUTPUT = Path("reports/full_eval_report.md")

# Directories to skip
SKIP = {
    "bootstrap_cache", "deepliif_sweep", "visual_tuning",
    "visual_tuning_deepliif", "visual_tuning_stardist_fluo",
}

# Stain name normalisation
STAIN_ALIASES = {
    "ki67": "Ki67", "ki-67": "Ki67", "ki_67": "Ki67",
    "her2": "HER2", "her-2": "HER2", "her_2": "HER2",
    "er": "ER", "pr": "PR",
}


def normalise_stain(name):
    return STAIN_ALIASES.get(name.lower(), name.upper())


def is_aligned_dir(dirname):
    return "_aligned" in dirname.lower() or "aligned" in dirname.lower().split("_")


# Metric definitions: (json_section, json_key, is_lower_better)
METRICS = [
    # Image Quality
    ("FID",        "image_quality", "fid_inception",       True),
    ("UNI-FID",    "image_quality", "fid_uni",             True),
    ("KID",        "image_quality", "kid_mean_x1000",      True),
    ("LPIPS",      "image_quality", "lpips_mean",          True),
    ("SSIM",       "image_quality", "ssim_mean",           False),
    ("PSNR",       "image_quality", "psnr_mean",           False),
    # Structure
    ("HE-H-SSIM",  "he_structure",  None,                   False),  # dynamic key
    ("HE-NMI",     "he_structure",  None,                   False),  # dynamic key
    # DAB
    ("DAB MAE",    "dab",           "dab_mae_overall",     True),
    ("DAB r",      "dab",           "dab_pearson_r",       False),
    ("DAB KL",     "dab",           "dab_kl",              True),
    ("DAB JSD",    "dab",           "dab_jsd",             True),
    ("mIOD Diff",  "iod",           "miod_diff",           True),
    # Clinical
    ("ICC",        "ki67_clinical", "ki67_icc",            False),
    ("MAE LI%",    "ki67_clinical", "ki67_li_mae",         True),
    ("r LI",       "ki67_clinical", "ki67_li_pearson_r",   False),
    ("3-Tier %",   "ki67_clinical", "ki67_3tier_concordance", False),
    ("Kappa",      "ki67_clinical", "ki67_3tier_kappa",    False),
    ("N Images",   "ki67_clinical", "ki67_n_images",       False),
]


def get_metric(per_stain_data, stain, section, key, is_lower):
    """Extract a metric, handling various JSON formats and missing keys."""
    stain_data = per_stain_data.get(stain, {})
    if not stain_data:
        return None

    if section == "he_structure" and key is None:
        # Dynamic key: try he_h_ssim_mean and ihc_h_ssim_mean
        struct = stain_data.get("he_structure", {})
        # Look for any key ending with _h_ssim_mean or _nmi_mean
        ssim_keys = [k for k in struct if k.endswith("_h_ssim_mean")]
        nmi_keys  = [k for k in struct if k.endswith("_nmi_mean")]
        return struct.get(ssim_keys[0]) if ssim_keys else struct.get("he_h_ssim_mean") if "HE-H-SSIM" in str(section) else struct.get(nmi_keys[0]) if nmi_keys else None

    if section not in stain_data:
        return None

    sec = stain_data[section]
    if key not in sec:
        # Try alternate keys
        alt_keys = {
            "ki67_3tier_concordance": "ki67_tier_concordance",
            "ki67_3tier_kappa": "ki67_tier_kappa",
            "fid_inception": None,  # no alt
        }
        alt = alt_keys.get(key)
        if alt and alt in sec:
            return sec[alt]
        return None

    return sec[key]


def fmt(val):
    """Format a metric value for display."""
    if val is None:
        return "—"
    if isinstance(val, str):
        return val
    if abs(val) < 0.0001:
        return f"{val:.2e}"
    if abs(val) < 1:
        return f"{val:.4f}"
    if abs(val) < 10:
        return f"{val:.3f}"
    return f"{val:.1f}"


def fmt_li(val):
    """Format LI percentage."""
    if val is None:
        return "—"
    return f"{val:.1f}%"


def get_aligned_pair(dirname):
    """If this is an aligned dir, return the name of the normal counterpart."""
    for suffix in ["_aligned", "_Aligned", "-aligned"]:
        if dirname.endswith(suffix):
            return dirname[: -len(suffix)]
    return None


def main():
    all_eval_dirs = sorted([
        d for d in EVAL_DIR.iterdir()
        if d.is_dir() and d.name not in SKIP
    ])

    # Phase 1: Discover stains, aligned status, and collect all metric rows
    rows = []  # list of dicts

    for eval_dir in all_eval_dirs:
        json_path = eval_dir / "results.json"
        if not json_path.exists():
            continue

        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception:
            continue

        per_stain = data.get("per_stain", {})
        if not per_stain:
            continue

        aligned = is_aligned_dir(eval_dir.name)
        display_name = eval_dir.name
        if aligned:
            display_name += " (A)"  # mark aligned

        for stain_raw in per_stain:
            stain = normalise_stain(stain_raw)

            row = {
                "eval_dir": display_name,
                "stain": stain,
                "aligned": aligned,
            }
            for name, section, key, is_lower in METRICS:
                if name in ("HE-H-SSIM", "HE-NMI"):
                    # Handle dynamic he_structure keys
                    struct = per_stain[stain_raw].get("he_structure", {})
                    if name == "HE-H-SSIM":
                        val = None
                        for k in struct:
                            if k.endswith("_h_ssim_mean"):
                                val = struct[k]
                                break
                    else:
                        val = None
                        for k in struct:
                            if k.endswith("_nmi_mean"):
                                val = struct[k]
                                break
                    row[name] = val
                else:
                    row[name] = get_metric(per_stain, stain_raw, section, key, is_lower)

            rows.append(row)

    if not rows:
        print("No eval data found.")
        return

    # Phase 2: Find which stains are present and which metrics have data
    all_stains = sorted(set(r["stain"] for r in rows))
    metrics_with_data = []
    for name, section, key, is_lower in METRICS:
        has_data = any(r.get(name) is not None for r in rows)
        if has_data:
            metrics_with_data.append((name, section, key, is_lower))

    # Phase 3: Group rows by eval_dir+stain for aligned pairs
    # Identify normal/aligned pairs
    run_groups = defaultdict(dict)  # run_name -> {stain: {"normal": row, "aligned": row}}

    for row in rows:
        name = row["eval_dir"]
        stain = row["stain"]
        if stain not in run_groups[name]:
            run_groups[name][stain] = {"normal": None, "aligned": None}
        key = "aligned" if row["aligned"] else "normal"
        run_groups[name][stain][key] = row

    # Phase 4: Generate markdown
    md = []
    md.append("# UNIStainNet — Complete Evaluation Report")
    md.append("")
    md.append(f"**Total eval dirs scanned:** {len(all_eval_dirs)}")
    md.append(f"**Runs with results:** {len(run_groups)}")
    md.append(f"**Stains found:** {', '.join(all_stains)}")
    md.append("")
    md.append("---")
    md.append("")

    # One table per stain
    for stain in all_stains:
        md.append(f"## {stain}")
        md.append("")

        # Only include runs that have this stain
        stain_runs = {
            name: data[stain]
            for name, data in run_groups.items()
            if stain in data and data[stain].get("normal")
        }

        if not stain_runs:
            md.append("*(no data)*")
            md.append("")
            continue

        # Check if any have aligned data
        has_aligned = any(
            data.get("aligned") is not None
            for data in stain_runs.values()
        )

        # Determine which metrics to show for this stain
        # For Ki67: show clinical. For others: skip clinical.
        show_clinical = (stain == "Ki67")
        metrics_to_show = []
        for name, section, key, is_lower in metrics_with_data:
            if section == "ki67_clinical" and not show_clinical:
                continue
            metrics_to_show.append(name)

        # Build header
        header = "| Run | "
        for m in metrics_to_show:
            arrow = "↓" if any(m2[0] == m and m2[3] for m2 in METRICS) else "↑"
            header += f" {m} |"
        md.append(header)

        sep = "|---|"
        for _ in metrics_to_show:
            sep += "---|"
        md.append(sep)

        # Sort runs: baseline first, then by name
        sorted_runs = sorted(stain_runs.items(), key=lambda x: (
            not x[0].startswith("baseline"),
            not x[0].startswith("ki67_deepliif_run"),
            not x[0].startswith("ki67_pv"),
            not x[0].startswith("ki67_run"),
            x[0],
        ))

        for run_name, pair in sorted_runs:
            row = pair["normal"]
            if row is None:
                continue

            line = f"| {run_name:<45} |"
            for m in metrics_to_show:
                val = row.get(m)
                if m in ("MAE LI%",):
                    line += f" {fmt_li(val)} |"
                elif m == "3-Tier %":
                    if val is not None:
                        line += f" {val*100:.1f}% |"
                    else:
                        line += " — |"
                elif m == "N Images":
                    line += f" {int(val) if val else '—'} |"
                else:
                    line += f" {fmt(val)} |"
            md.append(line)

        md.append("")

    # Phase 5: Ki67-only focused comparison (the most important table)
    md.append("---")
    md.append("")
    md.append("## Ki67 — Focused Comparison")
    md.append("")
    md.append("Only runs with full metrics available:")
    md.append("")

    ki67_runs = {}
    for name, data in run_groups.items():
        if "Ki67" in data and data["Ki67"].get("normal"):
            ki67_runs[name] = data["Ki67"]["normal"]

    # Filter to runs that have FID and clinical
    ki67_complete = {
        name: row for name, row in ki67_runs.items()
        if row.get("FID") is not None
    }

    # Key metrics for comparison
    key_metrics = ["FID", "UNI-FID", "KID", "LPIPS", "HE-H-SSIM", "HE-NMI",
                   "DAB MAE", "DAB r", "DAB KL", "DAB JSD",
                   "ICC", "MAE LI%", "r LI", "3-Tier %", "Kappa"]

    header = "| Run | "
    for m in key_metrics:
        header += f" {m} |"
    md.append(header)
    sep = "|---|"
    for _ in key_metrics:
        sep += "---|"
    md.append(sep)

    # Sort intelligently
    def sort_key(item):
        name = item[0]
        if name.startswith("baseline"): return (0, name)
        if "deepliif_run8" in name: return (1, name)
        if "deepliif_run5" in name: return (2, name)
        if "deepliif_run6" in name: return (2, name)
        if "deepliif_run7" in name: return (2, name)
        if "pv1" in name.lower(): return (3, name)
        if "pv2" in name.lower(): return (3, name)
        if "pv4" in name.lower(): return (3, name)
        if "ki67_run9" in name: return (4, name)
        if "ki67_run10" in name: return (4, name)
        if "ki67_run11" in name: return (4, name)
        if "ki67_run12" in name: return (4, name)
        return (5, name)

    for run_name, row in sorted(ki67_complete.items(), key=sort_key):
        line = f"| {run_name:<50} |"
        for m in key_metrics:
            val = row.get(m)
            if m == "3-Tier %" and val is not None:
                line += f" {val*100:.1f}% |"
            elif m == "MAE LI%" and val is not None:
                line += f" {fmt_li(val)} |"
            else:
                line += f" {fmt(val)} |"
        md.append(line)

    md.append("")
    md.append(f"*Report generated from {len(rows)} stain-runs across {len(run_groups)} eval directories.*")

    # Write
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        f.write("\n".join(md))

    print(f"Report written to {OUTPUT}")
    print(f"  {len(run_groups)} eval dirs, {len(rows)} stain-eval rows")


if __name__ == "__main__":
    main()
