# aggregate_random_jaccard.py
# Run:
#   pip install pandas
#   python aggregate_random_jaccard.py --random-root gpt/random_output --out artifacts/gpt_random_jaccard_agg.csv
#   python aggregate_random_jaccard.py --random-root claude/random_output --out artifacts/claude_random_jaccard_agg.csv

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
    return sorted([d for d in glob.glob(os.path.join(random_root, "run_*")) if os.path.isdir(d)])

def mean_ci95(vals):
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = sum(vals) / n
    if n == 1:
        return m, m, m
    var = sum((x - m) ** 2 for x in vals) / (n - 1)
    se = math.sqrt(var / n)
    return m, m - 1.96 * se, m + 1.96 * se

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random-root", required=True, help="e.g. gpt/random_output or claude/random_output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-nodes", type=int, default=12)
    args = ap.parse_args()

    run_dirs = list_run_dirs(args.random_root)
    if not run_dirs:
        raise SystemExit(f"No run_* dirs under {args.random_root}")

    per_node = {t: [] for t in range(args.max_nodes)}

    for d in run_dirs:
        tmp = os.path.join(d, "_tmp_jaccard.csv")
        cmd = [
            sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "theory_metrics_jaccard.py"),
            "--root", d,
            "--out", tmp,
            "--max-nodes", str(args.max_nodes),
        ]
        subprocess.check_call(cmd)
        df = pd.read_csv(tmp)
        for _, r in df.iterrows():
            t = int(r["node"])
            per_node[t].append(float(r["mean_jaccard_O_pq_t"]))
        try:
            os.remove(tmp)
        except OSError:
            pass

    out_rows = []
    for t in range(args.max_nodes):
        vals = per_node[t]
        m, lo, hi = mean_ci95(vals)
        out_rows.append({
            "node": t,
            "n_runs": len(vals),
            "jaccard_mean": m,
            "jaccard_ci95_low": lo,
            "jaccard_ci95_high": hi,
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