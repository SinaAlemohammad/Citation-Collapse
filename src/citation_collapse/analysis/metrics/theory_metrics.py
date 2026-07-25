# theory_metrics.py
# Run: python theory_metrics.py --root gpt/output --out gpt_theory_metrics.csv
#      python theory_metrics.py --root claude/output --out claude_theory_metrics.csv
#      python theory_metrics.py --root gpt/random_output/run_2 --out gpt_rnd_run2_theory_metrics.csv

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Set, Tuple

MAX_NODES_DEFAULT = 12
SEED_COUNT_DEFAULT = 120
PAPERS_PER_NODE_DEFAULT = 120

def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def get_available_papers(node: int, seed_count: int, papers_per_node: int) -> int:
    # N_t = S + M t  (repo uses seed_count + papers_per_node*node)
    return seed_count + papers_per_node * node

def load_valid_ids(root: str) -> Set[str]:
    # used in repo metric scripts; keeps the "canonical" paper-id universe
    kv_path = os.path.join(root, "master", "kv_pairs.jsonl")
    valid = set()
    if not os.path.isfile(kv_path):
        # For random runs, kv_pairs.jsonl should exist because random_citer copies it,
        # but if not present we fall back to "no filtering".
        return valid
    for row in iter_jsonl(kv_path):
        pid = row.get("id")
        if pid is not None:
            valid.add(pid)
    return valid

def load_node_prompt_rows(root: str, node: int) -> List[dict]:
    # This is the per-paper (prompt-instance) file produced by experiment and random_citer.
    path = os.path.join(root, f"node_{node}", f"node_{node}.jsonl")
    return read_jsonl(path)

def extract_E_C_from_prompt_row(row: dict, valid_ids: Set[str] | None) -> Tuple[List[str], List[str]]:
    """
    E_pt: papers_seen_id
    C_pt: citation_ids
    """
    E = row.get("papers_seen_id", []) or []
    C = row.get("citation_ids", []) or []

    if valid_ids is not None and len(valid_ids) > 0:
        E = [x for x in E if x in valid_ids]
        C = [x for x in C if x in valid_ids]

    return E, C

def node_counts_from_Cpt(prompts: List[Tuple[List[str], List[str]]]) -> Counter:
    """
    C_jt = sum_p Y_pjt, with Y_pjt = 1{j in C_pt}.
    Here we treat each JSONL row as one prompt-instance p at node t.
    """
    counts = Counter()
    for _, C in prompts:
        # Treat C as a set for "subset choice" interpretation (no duplicates)
        for j in set(C):
            counts[j] += 1
    return counts

def compute_hhi(counts: Counter) -> Tuple[float, int]:
    """
    HHI_t = sum_j (C_jt / K_t)^2, where K_t = sum_j C_jt.
    Returns (HHI, K_t).
    """
    K = sum(counts.values())
    if K == 0:
        return 0.0, 0
    return sum((c / K) ** 2 for c in counts.values()), K

def expected_hhi_null(N: int, k_list: List[int]) -> float:
    """
    Protocol-matched null approximation on the *catalog*:
      - each prompt p selects k_p papers uniformly from N
      - prompts independent
    Then C_j = sum_p I_{j in C_p} with P(I=1)=k_p/N.

    Derivation:
      E[HHI] = (1/K) + (sum_p k_p^2 - K) / (K^2 N)
    where K = sum_p k_p.
    """
    K = sum(k_list)
    if K <= 0 or N <= 0:
        return 0.0
    sum_k2 = sum(k * k for k in k_list)
    return (1.0 / K) + ((sum_k2 - K) / (K * K * N))

def mean_pair_overlap(C_sets: List[Set[str]]) -> float:
    """
    O_{pqt} = |C_p ∩ C_q|
    Return mean over all unordered prompt pairs.
    """
    M = len(C_sets)
    if M < 2:
        return 0.0
    total = 0
    n_pairs = 0
    for i in range(M):
        Ci = C_sets[i]
        for j in range(i + 1, M):
            total += len(Ci.intersection(C_sets[j]))
            n_pairs += 1
    return total / n_pairs

def expected_mean_pair_overlap_null(N: int, k_list: List[int]) -> float:
    """
    Under the same null as Prop 1 (uniform subset), for p != q:
      E[|C_p ∩ C_q|] = k_p k_q / N.
    So mean over pairs is mean_{p<q} k_p k_q / N.
    """
    M = len(k_list)
    if M < 2 or N <= 0:
        return 0.0
    total = 0.0
    n_pairs = 0
    for i in range(M):
        for j in range(i + 1, M):
            total += (k_list[i] * k_list[j]) / N
            n_pairs += 1
    return total / n_pairs

def node_level_shown_but_never_cited_rate(prompts: List[Tuple[List[str], List[str]]], valid_ids: Set[str], N: int) -> Tuple[float, int, int]:
    """
    Repo-style "shown_ignored_rate_of_shown":
      shown_ignored = #{papers with exposures>0 and citations==0}
      cited = #{papers with citations>0}
      shown_papers = cited + shown_ignored
      rate = shown_ignored / shown_papers
    Where exposures = count of prompts where paper is in E_pt.
    """
    exposures = Counter()
    citations = Counter()
    for E, C in prompts:
        Eset = set(E)
        Cset = set(C)
        for j in Eset:
            exposures[j] += 1
        for j in Cset:
            citations[j] += 1

    # restrict to valid_ids if provided
    if valid_ids:
        exposures = Counter({k: v for k, v in exposures.items() if k in valid_ids})
        citations = Counter({k: v for k, v in citations.items() if k in valid_ids})

    # shown-but-never-cited among shown
    shown_ids = set(exposures.keys())
    cited_ids = {j for j, c in citations.items() if c > 0}
    shown_ignored = len([j for j in shown_ids if citations.get(j, 0) == 0])
    cited = len(cited_ids)
    shown_papers = shown_ignored + cited
    rate = (shown_ignored / shown_papers) if shown_papers > 0 else 0.0
    return rate, shown_ignored, shown_papers

def per_paper_FN_jt(prompts: List[Tuple[List[str], List[str]]], N: int) -> Dict[str, float]:
    """
    A direct empirical estimate of FN_{jt} at node t:
      FN_{jt} = 1{shown>=1 and cited==0} / 1{shown>=1}
    which is the realized analogue of Prop 6’s conditional event definition.

    Note: Prop 6 gives a probability formula in terms of q_{pjt};
    here we compute the realized indicator version from data.
    """
    exposures = Counter()
    citations = Counter()
    for E, C in prompts:
        for j in set(E):
            exposures[j] += 1
        for j in set(C):
            citations[j] += 1

    out = {}
    for j, ex in exposures.items():
        if ex > 0:
            out[j] = 1.0 if citations.get(j, 0) == 0 else 0.0
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="e.g., gpt/output or claude/output or gpt/random_output/run_2")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--max-nodes", type=int, default=MAX_NODES_DEFAULT)
    ap.add_argument("--seed-count", type=int, default=SEED_COUNT_DEFAULT)
    ap.add_argument("--papers-per-node", type=int, default=PAPERS_PER_NODE_DEFAULT)
    ap.add_argument("--write-fn-per-paper", default="", help="optional JSON output of per-paper FN_{jt} by node")
    args = ap.parse_args()

    valid_ids = load_valid_ids(args.root)

    rows = []
    fn_per_paper_by_node = {}

    for t in range(args.max_nodes):
        node_dir = os.path.join(args.root, f"node_{t}")
        node_file = os.path.join(node_dir, f"node_{t}.jsonl")
        stats_file = os.path.join(node_dir, f"node_{t}_stats.jsonl")

        if not os.path.isfile(node_file):
            # Some roots may not contain all nodes; skip missing nodes
            continue

        N_t = get_available_papers(t, args.seed_count, args.papers_per_node)

        prompt_rows = load_node_prompt_rows(args.root, t)
        prompts: List[Tuple[List[str], List[str]]] = [
            extract_E_C_from_prompt_row(r, valid_ids if valid_ids else None)
            for r in prompt_rows
        ]

        # k_pt list and C_pt sets
        k_list = [len(set(C)) for _, C in prompts]
        C_sets = [set(C) for _, C in prompts]

        # Node-level citation counts C_jt, total K_t, HHI
        counts = node_counts_from_Cpt(prompts)
        hhi_t, K_t = compute_hhi(counts)

        # Null expectations from theory (Prop 1 baseline)
        hhi_null = expected_hhi_null(N_t, k_list)
        overlap_mean = mean_pair_overlap(C_sets)
        overlap_null = expected_mean_pair_overlap_null(N_t, k_list)

        # Repo-style node-level false-negative summary
        shown_ignored_rate, shown_ignored, shown_papers = node_level_shown_but_never_cited_rate(
            prompts, valid_ids, N_t
        )

        # Per-paper FN_{jt} realized indicator (optional output)
        fn_j = per_paper_FN_jt(prompts, N_t)
        fn_per_paper_by_node[str(t)] = fn_j

        rows.append({
            "node": t,
            "N_t": N_t,
            "M_prompts": len(prompts),
            "K_t_total_citations": K_t,
            "mean_k_pt": (sum(k_list) / len(k_list)) if k_list else 0.0,
            "HHI_t": hhi_t,
            "E_HHI_null_t": hhi_null,
            "excess_HHI_t": hhi_t - hhi_null,
            "mean_overlap_O_pq_t": overlap_mean,
            "E_overlap_null_t": overlap_null,
            "excess_overlap_t": overlap_mean - overlap_null,
            "shown_ignored_rate_of_shown": shown_ignored_rate,
            "shown_ignored": shown_ignored,
            "shown_papers": shown_papers,
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    if args.write_fn_per_paper:
        os.makedirs(os.path.dirname(args.write_fn_per_paper) or ".", exist_ok=True)
        with open(args.write_fn_per_paper, "w") as f:
            json.dump(fn_per_paper_by_node, f)

    print(f"Wrote {args.out}")
    if args.write_fn_per_paper:
        print(f"Wrote {args.write_fn_per_paper}")

if __name__ == "__main__":
    main()