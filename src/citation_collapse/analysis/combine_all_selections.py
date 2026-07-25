#!/usr/bin/env python
"""
Stack every model's all-nodes selection CSV into ONE giant CSV at the repo root.

Each model already has a single CSV with all its nodes at
selections/<model>/<model>-all.csv (produced by
`extract_node_selections.py --node all --combined`). This reads each of those and
concatenates them into one file -- but ONLY for models whose CSV covers the full
set of nodes (default 12). Models missing nodes are skipped and reported. Every row
keeps its `model` and `node` columns, so the giant file stays fully identifiable.

If a model has no <model>-all.csv, it falls back to its per-node CSVs
(<model>-node*.csv) so it still works.

Usage:
  python combine_all_selections.py                       # -> all-models-all-nodes-selections.csv
  python combine_all_selections.py --out combined.csv
  python combine_all_selections.py --nodes 12            # nodes required for a "full" run
  python combine_all_selections.py --models gpt-5 gemini-2.5-flash   # restrict the list
"""
import argparse
import csv
from pathlib import Path

from citation_collapse.core import config


def discover_models():
    if not config.SELECTIONS_DIR.exists():
        return []
    return sorted(p.name for p in config.SELECTIONS_DIR.iterdir() if p.is_dir())


def read_model_rows(model):
    """(rows, source) for a model's combined selections. Prefer <model>-all.csv,
    else concatenate <model>-node*.csv. rows == [] if nothing found."""
    mdir = config.SELECTIONS_DIR / model
    if not mdir.is_dir():
        return [], None
    allcsv = mdir / f"{model}-all.csv"
    if allcsv.exists():
        with open(allcsv, newline="") as f:
            return list(csv.DictReader(f)), allcsv.name
    node_csvs = sorted(mdir.glob(f"{model}-node*.csv"))
    rows = []
    for nc in node_csvs:
        with open(nc, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows, (f"{len(node_csvs)} per-node CSVs" if node_csvs else None)


def combine(models, n_nodes, out_path):
    combined, fieldnames = [], []
    included, skipped = [], []

    for m in models:
        rows, src = read_model_rows(m)
        if not rows:
            skipped.append((m, "no selection CSV"))
            continue
        nodes_present = {int(r["node"]) for r in rows if str(r.get("node", "")).strip() != ""}
        if not set(range(n_nodes)) <= nodes_present:
            skipped.append((m, f"{len(nodes_present)}/{n_nodes} nodes"))
            continue
        included.append(m)
        for r in rows:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        combined.extend(rows)
        print(f"  + {m}: {len(rows)} rows ({src})")

    if skipped:
        print("\nskipped (not a full run):")
        for m, why in skipped:
            print(f"  - {m}: {why}")

    if not combined:
        raise SystemExit(
            f"\nno models with all {n_nodes} nodes found under {config.SELECTIONS_DIR}.\n"
            f"Make each model's combined CSV first, e.g.:\n"
            f"  python extract_node_selections.py --all --node all --combined")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(combined)

    print(f"\nwrote {len(combined)} rows from {len(included)} full-run model(s) -> {out_path}")
    print(f"models included: {', '.join(included)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=config.TOTAL_NODES,
                    help=f"number of nodes a model must have to be included (default {config.TOTAL_NODES})")
    ap.add_argument("--models", nargs="*", default=None,
                    help="restrict to these model names (default: every model under selections/)")
    ap.add_argument("--out", default=None,
                    help="output CSV path (default <repo root>/all-models-all-nodes-selections.csv)")
    args = ap.parse_args()

    models = args.models if args.models else discover_models()
    if not models:
        raise SystemExit(f"no model folders under {config.SELECTIONS_DIR}.")
    out = Path(args.out) if args.out else config.ROOT / "all-models-all-nodes-selections.csv"
    combine(models, args.nodes, out)


if __name__ == "__main__":
    main()
