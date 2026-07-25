# aggregate_random_theory.py
# Run: python aggregate_random_theory.py --random-root gpt/random_output --out artifacts/gpt_random_theory_agg.csv

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import subprocess
import sys

import pandas as pd

def list_run_dirs(random_root: str):
    dirs = sorted([d for d in glob.glob(os.path.join(random_root, "run_*")) if os.path.isdir(d)])
    return dirs

def mean_ci95(vals):
    # plain normal approx; repo uses Bonferroni CI in some scripts, but this is fine for plotting.
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = sum(vals) / n
    if n == 1:
        return m, m, m
    var = sum((x - m) ** 2 for x in vals) / (n - 1)
    se = math.sqrt(var / n)
    lo = m - 1.96 * se
    hi = m + 1.96 * se
    return m, lo, hi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random-root", required=True, help="e.g. gpt/random_output or claude/random_output")
    ap.add_argument("--out", required=True, help="output CSV with mean/CI per node")
    ap.add_argument("--max-nodes", type=int, default=12)
    ap.add_argument("--seed-count", type=int, default=120)
    ap.add_argument("--papers-per-node", type=int, default=120)
    args = ap.parse_args()

    run_dirs = list_run_dirs(args.random_root)
    if not run_dirs:
        raise SystemExit(f"No run_* directories found under {args.random_root}")

    # Collect per-run CSVs in-memory
    per_node_vals = {t: {"HHI_t": [], "mean_overlap_O_pq_t": []} for t in range(args.max_nodes)}

    for d in run_dirs:
        # Call theory_metrics.py as a module-style subprocess-free approach would be nicer,
        # but keeping it simple: read directly by importing would require packaging.
        tmp_csv = os.path.join(d, "_tmp_theory_metrics.csv")
        cmd = [
            sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "theory_metrics.py"),
            "--root", d,
            "--out", tmp_csv,
            "--max-nodes", str(args.max_nodes),
            "--seed-count", str(args.seed_count),
            "--papers-per-node", str(args.papers_per_node),
        ]
        subprocess.check_call(cmd)

        df = pd.read_csv(tmp_csv)
        for _, row in df.iterrows():
            t = int(row["node"])
            per_node_vals[t]["HHI_t"].append(float(row["HHI_t"]))
            per_node_vals[t]["mean_overlap_O_pq_t"].append(float(row["mean_overlap_O_pq_t"]))

        try:
            os.remove(tmp_csv)
        except OSError:
            pass

    out_rows = []
    for t in range(args.max_nodes):
        if len(per_node_vals[t]["HHI_t"]) == 0:
            continue   # node has no random-run data; skip empty/nan row
        hhi_mean, hhi_lo, hhi_hi = mean_ci95(per_node_vals[t]["HHI_t"])
        ov_mean, ov_lo, ov_hi = mean_ci95(per_node_vals[t]["mean_overlap_O_pq_t"])
        out_rows.append({
            "node": t,
            "n_runs": len(per_node_vals[t]["HHI_t"]),
            "HHI_mean": hhi_mean,
            "HHI_ci95_low": hhi_lo,
            "HHI_ci95_high": hhi_hi,
            "overlap_mean": ov_mean,
            "overlap_ci95_low": ov_lo,
            "overlap_ci95_high": ov_hi,
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()