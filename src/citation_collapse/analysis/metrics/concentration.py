# --- RELOCATED to utils/. Paths now come from config + --model -------------
import argparse as _argparse
from citation_collapse.core import config as _config
_ap = _argparse.ArgumentParser(add_help=False)
_ap.add_argument("--model", required=True, choices=_config.list_models())
_args, _ = _ap.parse_known_args()
MODEL = _args.model
OUTPUT_ROOT = str(_config.model_dir(MODEL))        # runs/<model>/
RANDOM_ROOT = str(_config.random_root(MODEL))      # runs/<model>/random_output/
# ---------------------------------------------------------------------------

import json
import os
import numpy as np
import matplotlib.pyplot as plt

from citation_collapse.analysis.metrics.random_aggregate import bonferroni_mean_ci, list_random_run_dirs


def _sort_key_count_desc(item):
    paper_id, count = item
    return (-count, paper_id)

BASE = "."
MAX_NODES = 12
SEED_COUNT = 120
PAPERS_PER_NODE = 120
# OUTPUT_ROOT defined by shim above
# RANDOM_ROOT defined by shim above
BONFERRONI_M_NODES = 12

valid_ids = set()


def get_available_papers(node: int) -> int:
    return SEED_COUNT + (PAPERS_PER_NODE * node)


def load_node_stats(node: int, data_root: str) -> dict:
    stats = {}
    path = f"{data_root}/node_{node}/node_{node}_stats.jsonl"
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            for k, v in data.items():
                if k in valid_ids:
                    stats[k] = v
    return stats


def top_n_share_fraction(sorted_stats: list, n: int, total_citations: int) -> float:
    if total_citations <= 0:
        return 0.0
    top = sorted_stats[:n]
    top_citations = 0
    for id, count in top:
        top_citations += count
    return top_citations / total_citations


def top_percent_share(
    sorted_stats: list, percent: float, total_citations: int, available: int
) -> tuple[float, int]:
    if total_citations <= 0 or available <= 0 or percent <= 0.0:
        return 0.0, 0
    k = int(available * percent)
    if k < 1:
        k = 1
    k_effective = min(k, len(sorted_stats))
    frac = top_n_share_fraction(sorted_stats, k_effective, total_citations)
    return frac, k_effective


def load_valid_ids(data_root: str) -> None:
    valid_ids.clear()
    kv_path = f"{data_root}/master/kv_pairs.jsonl"
    with open(kv_path) as f:
        for line in f:
            data = json.loads(line.strip())
            valid_ids.add(data["id"])


def analyze_node(node: int, data_root: str) -> dict:
    stats = load_node_stats(node, data_root)
    sorted_stats = sorted(stats.items(), key=_sort_key_count_desc)
    total_citations = sum(stats.values())
    available = get_available_papers(node)

    frac_10, k10 = top_percent_share(
        sorted_stats, 0.10, total_citations, available
    )

    return {
        "node": node,
        "total_citations": total_citations,
        "unique_papers_cited": len(stats),
        "available_papers": available,
        "top_10pct_share": round(100.0 * frac_10, 2),
        "top_10pct_count": k10,
    }


def collect_top10_fractions_per_node(data_root: str) -> list[float]:
    load_valid_ids(data_root)
    out = []
    for node in range(MAX_NODES):
        stats = load_node_stats(node, data_root)
        sorted_stats = sorted(stats.items(), key=_sort_key_count_desc)
        total_citations = sum(stats.values())
        available = get_available_papers(node)
        frac_10, _ = top_percent_share(
            sorted_stats, 0.10, total_citations, available
        )
        out.append(100.0 * frac_10)
    return out


def run_llm(data_root: str, label: str) -> list[dict]:
    output_dir = f"{data_root}/master"
    os.makedirs(output_dir, exist_ok=True)
    load_valid_ids(data_root)

    results = []
    for node in range(MAX_NODES):
        result = analyze_node(node, data_root)
        results.append(result)
        print(
            f"[{label}] Node {node}: total={result['total_citations']}, "
            f"top10pct={result['top_10pct_share']}% (n={result['top_10pct_count']})"
        )

    with open(f"{output_dir}/concentration.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {output_dir}/concentration.jsonl")
    return results


def aggregate_random_top10_by_node(run_dirs: list[str]) -> tuple[list[list[float]], int]:
    """Returns (values_per_node, n_runs) where values_per_node[node] lists run values in %."""
    per_node: list[list[float]] = [[] for _ in range(MAX_NODES)]
    valid_run_count = 0
    for run_dir in run_dirs:
        try:
            fracs = collect_top10_fractions_per_node(run_dir)
        except (OSError, json.JSONDecodeError, KeyError) as e:
            print(f"[Random] skip {run_dir}: {e}")
            continue
        for node in range(MAX_NODES):
            per_node[node].append(fracs[node])
        valid_run_count += 1
    return per_node, valid_run_count


def main():
    exp_results = run_llm(OUTPUT_ROOT, "Experiment")

    run_dirs = list_random_run_dirs(RANDOM_ROOT)
    if not run_dirs:
        print(f"No run_* dirs under {RANDOM_ROOT}")
        return

    per_node_vals, n_runs = aggregate_random_top10_by_node(run_dirs)
    random_rows = []
    rnd_mean: list[float] = []
    rnd_lo: list[float] = []
    rnd_hi: list[float] = []
    for node in range(MAX_NODES):
        vals = per_node_vals[node]
        mean, lo, hi = bonferroni_mean_ci(vals, BONFERRONI_M_NODES)
        rnd_mean.append(mean)
        rnd_lo.append(max(0.0, lo))
        rnd_hi.append(min(100.0, hi))
        random_rows.append({
            "node": node,
            "n_runs": len(vals),
            "top_10pct_share_mean": round(mean, 4),
            "top_10pct_share_ci_low": round(max(0.0, lo), 4),
            "top_10pct_share_ci_high": round(min(100.0, hi), 4),
        })

    master_dir = f"{OUTPUT_ROOT}/master"
    os.makedirs(master_dir, exist_ok=True)
    with open(f"{master_dir}/concentration_random_baseline.jsonl", "w") as f:
        for row in random_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {master_dir}/concentration_random_baseline.jsonl ({n_runs} runs)")

    nodes = [r["node"] for r in exp_results]
    exp_y = [r["top_10pct_share"] for r in exp_results]
    x = np.arange(len(nodes))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        x,
        exp_y,
        color="#10A37F",
        marker="o",
        linewidth=2,
        label="LLM (top 10% share)",
    )
    ax.plot(
        x,
        rnd_mean,
        color="#3498DB",
        marker="s",
        linewidth=2,
        linestyle="--",
        label=f"Mean of the random citers (n={n_runs} runs)",
    )

    ax.set_xlabel("Node (Generation)")
    ax.set_ylabel("Top 10% Papers' Share of Total Citations (%)")
    ax.set_title(
        "Citation concentration (top 10% of papers) GPT 5 Mini vs random baseline"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"Node {n}" for n in nodes])
    all_hi = [y for y in (exp_y + rnd_hi) if np.isfinite(y)]
    y_top = min(100, max(all_hi) * 1.22) if all_hi else 100
    ax.set_ylim(0, y_top)
    ax.legend(loc="upper left", fontsize=9)

    for i, (ve, vr) in enumerate(zip(exp_y, rnd_mean)):
        exp_off = (0, 7) if ve >= vr else (0, -14)
        rnd_off = (0, -14) if ve >= vr else (0, 7)
        ax.annotate(f"{ve:.1f}%", (x[i], ve), textcoords="offset points",
                    xytext=exp_off, ha="center", fontsize=8, color="#10A37F")
        ax.annotate(f"{vr:.1f}%", (x[i], vr), textcoords="offset points",
                    xytext=rnd_off, ha="center", fontsize=8, color="#3498DB")

    fig.tight_layout()
    fig.savefig(f"{master_dir}/concentration.png", dpi=150, bbox_inches="tight")
    print(f"Saved {master_dir}/concentration.png")


if __name__ == "__main__":
    main()
