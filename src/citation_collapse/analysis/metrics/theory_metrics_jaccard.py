# theory_metrics_jaccard.py
# Run:
#   python theory_metrics_jaccard.py --root gpt/output --out artifacts/gpt_theory_metrics_jaccard.csv
#   python theory_metrics_jaccard.py --root claude/output --out artifacts/claude_theory_metrics_jaccard.csv
#
# This computes mean pairwise Jaccard overlap by node:
#   J_{pq,t} = |C_p ∩ C_q| / |C_p ∪ C_q|
# averaged over all p<q within node t.

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import Counter
from typing import Iterable, List, Set, Tuple

MAX_NODES_DEFAULT = 12
SEED_COUNT_DEFAULT = 120
PAPERS_PER_NODE_DEFAULT = 120

def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def read_jsonl(path: str) -> List[dict]:
    return list(iter_jsonl(path))

def load_valid_ids(root: str) -> Set[str]:
    kv = os.path.join(root, "master", "kv_pairs.jsonl")
    if not os.path.isfile(kv):
        return set()
    return {row["id"] for row in iter_jsonl(kv) if "id" in row}

def load_node_prompt_rows(root: str, node: int) -> List[dict]:
    path = os.path.join(root, f"node_{node}", f"node_{node}.jsonl")
    return read_jsonl(path)

def extract_C(row: dict, valid_ids: Set[str] | None) -> Set[str]:
    C = set(row.get("citation_ids", []) or [])
    if valid_ids and len(valid_ids) > 0:
        C = {x for x in C if x in valid_ids}
    return C

def mean_pairwise_jaccard(C_sets: List[Set[str]]) -> float:
    M = len(C_sets)
    if M < 2:
        return 0.0
    tot = 0.0
    n = 0
    for i in range(M):
        A = C_sets[i]
        for j in range(i + 1, M):
            B = C_sets[j]
            inter = len(A & B)
            uni = len(A | B)
            tot += (inter / uni) if uni > 0 else 0.0
            n += 1
    return tot / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="e.g. gpt/output or claude/output or gpt/random_output/run_2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-nodes", type=int, default=MAX_NODES_DEFAULT)
    args = ap.parse_args()

    valid_ids = load_valid_ids(args.root)

    rows = []
    for t in range(args.max_nodes):
        node_file = os.path.join(args.root, f"node_{t}", f"node_{t}.jsonl")
        if not os.path.isfile(node_file):
            continue

        prompt_rows = load_node_prompt_rows(args.root, t)
        C_sets = [extract_C(r, valid_ids if valid_ids else None) for r in prompt_rows]
        mean_j = mean_pairwise_jaccard(C_sets)

        rows.append({
            "node": t,
            "M_prompts": len(C_sets),
            "mean_jaccard_O_pq_t": mean_j,
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()