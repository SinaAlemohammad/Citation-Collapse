#!/usr/bin/env python
"""
Merge every model's node-0 selection CSV into ONE combined CSV at the repo root.

Reads selections/<model>/<model>-node0.csv for every model that has one (these are
produced by extract_node_selections.py) and concatenates them into a single file,
by default <repo root>/node0-all-models.csv. Each row already carries `model` and
`node` columns, so rows stay identifiable after merging.

Run extract_node_selections.py for your models first, e.g.:
    python extract_node_selections.py --all --node 0

Then:
    python merge_node0_selections.py                 # -> node0-all-models.csv at repo root
    python merge_node0_selections.py --out combined.csv
    python merge_node0_selections.py --node 0        # any node; default 0
"""
import argparse
import csv
from pathlib import Path

from citation_collapse.core import config


def merge(node, out_path):
    # find each model's node-<node> selection csv under selections/<model>/
    files = []
    if config.SELECTIONS_DIR.exists():
        for model_dir in sorted(p for p in config.SELECTIONS_DIR.iterdir() if p.is_dir()):
            f = model_dir / f"{model_dir.name}-node{node}.csv"
            if f.exists():
                files.append(f)

    if not files:
        raise SystemExit(
            f"no node-{node} selection CSVs found under {config.SELECTIONS_DIR}.\n"
            f"Run `python extract_node_selections.py --all --node {node}` first.")

    # union the headers (preserve first file's order, append any extras) and collect rows
    fieldnames, rows = [], []
    for f in files:
        with open(f, newline="") as fh:
            r = csv.DictReader(fh)
            for fn in (r.fieldnames or []):
                if fn not in fieldnames:
                    fieldnames.append(fn)
            n = 0
            for row in r:
                rows.append(row)
                n += 1
        print(f"  + {f.relative_to(config.SELECTIONS_DIR)}: {n} rows")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)   # missing fields -> ""
        w.writeheader()
        w.writerows(rows)

    models = sorted({row.get("model") for row in rows if row.get("model")})
    print(f"\nwrote {len(rows)} rows from {len(files)} model(s) -> {out_path}")
    print(f"models: {', '.join(models)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", type=int, default=0,
                    help="which node's selections to merge (default 0)")
    ap.add_argument("--out", default=None,
                    help="output CSV path (default <repo root>/node<N>-all-models.csv)")
    args = ap.parse_args()
    out = Path(args.out) if args.out else config.ROOT / f"node{args.node}-all-models.csv"
    merge(args.node, out)


if __name__ == "__main__":
    main()
