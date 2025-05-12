#!/usr/bin/env python3
"""
compare_rot_lsd_logs.py

For each fold:
  • collect ALL rotLSD entries across epochs,
  • compute the mean rotLSD for that fold,
then average those 93 fold-means to get an overall rotLSD.

Outputs a CSV with columns: fold, baseline_mean, equivariant_mean
and prints the two overall means.
"""

import re, csv, argparse, numpy as np

# matches lines like:
# [fold 84] epoch  990 | train  ... | rotLSD   0.46
LINE_RE = re.compile(r"\[fold\s+(\d+)\].*?rotLSD\s+([\d\.]+)")

def parse_log(path):
    per_fold = {}
    with open(path, "r") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m: continue
            fold = int(m.group(1))
            val  = float(m.group(2))
            per_fold.setdefault(fold, []).append(val)
    # compute mean per fold
    means = {fold: np.mean(vals) for fold, vals in per_fold.items()}
    return means

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline",   required=True)
    p.add_argument("--equivariant",required=True)
    p.add_argument("--out",        required=True,
                   help="CSV path for per‐fold comparison")
    args = p.parse_args()

    base_means = parse_log(args.baseline)
    eq_means   = parse_log(args.equivariant)

    # ensure same set of folds
    folds = sorted(set(base_means) & set(eq_means))
    if len(folds) == 0:
        raise RuntimeError("No matching folds found in logs")

    # write CSV
    with open(args.out, "w", newline="") as csvf:
        w = csv.writer(csvf)
        w.writerow(["fold","baseline_mean_rotLSD","equivariant_mean_rotLSD"])
        for fold in folds:
            w.writerow([fold, base_means[fold], eq_means[fold]])

    # overall means (average of the per‐fold means)
    overall_base = np.mean([base_means[f] for f in folds])
    overall_eq   = np.mean([eq_means[f]   for f in folds])

    print(f"✔ parsed {len(folds)} folds")
    print(f"Baseline overall mean rotLSD    : {overall_base:.4f}")
    print(f"Equivariant overall mean rotLSD : {overall_eq:.4f}")
    print(f"Wrote per-fold comparison → {args.out}")

if __name__ == "__main__":
    main()
