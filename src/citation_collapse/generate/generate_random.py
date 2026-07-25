#!/usr/bin/env python3
"""
Matched random-null generator (the measurement floor), unified for all models.

    python generate_random.py --model gpt-5-mini
    python generate_random.py --model claude-haiku-4-5 --iters 50

For each real paper it KEEPS the per-paper citation count and the shown set, but
re-draws WHICH shown ids are cited, uniformly at random. Each iteration i is a
seeded replicate written to runs/<model>/random_output/run_<i>/. This is the
null that HHI / overlap are compared against (humans land on it).
"""

import argparse
import json
import random
from collections import defaultdict

from citation_collapse.core import config


def run_iter(model, i, max_nodes):
    random.seed(i)
    out = config.random_root(model) / f"run_{i}"
    (out / "master").mkdir(parents=True, exist_ok=True)
    wrote = 0
    for node in range(max_nodes):
        nf = config.node_file(model, node)
        if not nf.exists():
            break
        papers = [json.loads(l) for l in open(nf)]
        (out / f"node_{node}").mkdir(parents=True, exist_ok=True)
        node_cit = defaultdict(int)
        rows = []
        for p in papers:
            shown = p.get("papers_seen_id", []) or []
            k = min(len(p.get("citation_ids", []) or []), len(shown))
            cited = random.sample(shown, k) if k else []
            for c in cited:
                node_cit[c] += 1
            rows.append({"id": p["id"], "author": p.get("author"), "year": p.get("year"),
                         "type": p.get("type"), "papers_seen_id": shown, "citation_ids": cited})
        with open(out / f"node_{node}" / f"node_{node}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with open(out / f"node_{node}" / f"node_{node}_stats.jsonl", "w") as f:
            for pid, c in sorted(node_cit.items()):
                f.write(json.dumps({pid: c}) + "\n")
        wrote += 1
    # copy the model's OWN master kv (always consistent with its node files);
    # fall back to the canonical root kv only if master is absent.
    master_kv = config.master_dir(model) / "kv_pairs.jsonl"
    kv = master_kv if master_kv.exists() else config.KV_PAIRS
    if kv.exists():
        (out / "master" / "kv_pairs.jsonl").write_text(kv.read_text())
    return wrote


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=config.list_models())
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--max-nodes", type=int, default=config.TOTAL_NODES)
    args = ap.parse_args()
    for i in range(args.iters):
        n = run_iter(args.model, i, args.max_nodes)
        print(f"run_{i}: {n} nodes")
    print(f"Done. Null in {config.random_root(args.model)}")


if __name__ == "__main__":
    main()
