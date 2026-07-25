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
import matplotlib.pyplot as plt
import numpy as np
from citation_collapse.analysis.metrics.random_aggregate import list_random_run_dirs

BASE = "."
MAX_NODES = 12
# OUTPUT_ROOT defined by shim above
# RANDOM_ROOT defined by shim above


def load_exposure(node: int, data_root: str) -> list[dict]:
    path = f"{data_root}/node_{node}/node_{node}_exposure.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_rates(entries: list[dict]) -> dict:
    seed_citations = seed_exposures = 0
    llm_citations = llm_exposures = 0

    for e in entries:
        pid = e["id"]
        c, ex = e["citations"], e["exposures"]
        if pid.startswith("SEED_"):
            seed_citations += c
            seed_exposures += ex
        else:
            llm_citations += c
            llm_exposures += ex

    seed_rate = seed_citations / seed_exposures if seed_exposures > 0 else None
    llm_rate = llm_citations / llm_exposures if llm_exposures > 0 else None

    return {
        "seed_citations": seed_citations,
        "seed_exposures": seed_exposures,
        "seed_rate": seed_rate,
        "llm_citations": llm_citations,
        "llm_exposures": llm_exposures,
        "llm_rate": llm_rate,
    }


def rates_for_node(node: int, data_root: str) -> dict:
    entries = load_exposure(node, data_root)
    return compute_rates(entries)


def main():
    run_dirs = list_random_run_dirs(RANDOM_ROOT)
    if not run_dirs:
        print(f"No run_* dirs under {RANDOM_ROOT}")
        return

    per_node_seed_rates: list[list[float]] = [[] for _ in range(MAX_NODES)]
    per_node_llm_rates: list[list[float]] = [[] for _ in range(MAX_NODES)]

    valid_run_count = 0
    for run_dir in run_dirs:
        try:
            for node in range(MAX_NODES):
                r = rates_for_node(node, run_dir)
                if r["seed_rate"] is not None:
                    per_node_seed_rates[node].append(r["seed_rate"])
                if r["llm_rate"] is not None:
                    per_node_llm_rates[node].append(r["llm_rate"])
            valid_run_count += 1
        except (OSError, json.JSONDecodeError, KeyError) as e:
            print(f"[Random] skip {run_dir}: {e}")
            continue

    print(f"Aggregated {valid_run_count} random runs")

    results = []
    for node in range(MAX_NODES):
        seed_vals = per_node_seed_rates[node]
        llm_vals = per_node_llm_rates[node]
        avg_seed_rate = float(np.mean(seed_vals)) if seed_vals else None
        avg_llm_rate = float(np.mean(llm_vals)) if llm_vals else None
        results.append({
            "node": node,
            "n_runs_seed": len(seed_vals),
            "n_runs_llm": len(llm_vals),
            "avg_seed_rate": round(avg_seed_rate, 6) if avg_seed_rate is not None else None,
            "avg_llm_rate": round(avg_llm_rate, 6) if avg_llm_rate is not None else None,
        })

    print(f"{'Node':>4}  {'Avg Seed Rate':>14}  {'Avg LLM Rate':>14}")
    print("-" * 40)
    for r in results:
        seed_rate_str = f"{r['avg_seed_rate']:.4f}" if r["avg_seed_rate"] is not None else "N/A"
        llm_rate_str = f"{r['avg_llm_rate']:.4f}" if r["avg_llm_rate"] is not None else "N/A"
        print(f"{r['node']:>4}  {seed_rate_str:>14}  {llm_rate_str:>14}")

    out_path = f"{OUTPUT_ROOT}/master/random_avg_seed_vs_llm_citation_rate.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {out_path}")

    nodes = [r["node"] for r in results]
    avg_seed_rates = [r["avg_seed_rate"] for r in results]
    avg_llm_rates = [r["avg_llm_rate"] if r["avg_llm_rate"] is not None else 0 for r in results]

    x = np.arange(len(nodes))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        x,
        avg_seed_rates,
        color="#E74C3C",
        marker="o",
        linewidth=2,
        label="Random citer — seed papers",
    )
    ax.plot(
        x,
        avg_llm_rates,
        color="#3498DB",
        marker="s",
        linewidth=2,
        linestyle="--",
        label="Random citer — LLM papers",
    )

    ax.set_xlabel("Node (Generation)")
    ax.set_ylabel("Citation Rate (citations / exposures)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Node {n}" for n in nodes])
    ax.set_ylim(0, None)
    ax.legend(loc="best", fontsize=8)

    y_lo, y_hi = ax.get_ylim()
    y_range = y_hi - y_lo
    for i, v in enumerate(avg_seed_rates):
        if v is not None:
            offset = (0, -14) if (y_hi - v) < 0.08 * y_range else (0, 8)
            ax.annotate(f"{v * 100:.1f}%", (x[i], v), textcoords="offset points",
                        xytext=offset, ha="center", fontsize=8, color="#E74C3C")
    for i, v in enumerate(avg_llm_rates):
        if v is not None:
            offset = (0, 8) if v < 0.08 * y_range else (0, -14)
            ax.annotate(f"{v * 100:.1f}%", (x[i], v), textcoords="offset points",
                        xytext=offset, ha="center", fontsize=8, color="#3498DB")
    fig.suptitle(
        f"Citation Rate: Seed vs LLM-Generated Papers\n"
        f"(averaged over {valid_run_count} random citer runs)",
        fontsize=12,
    )
    fig.tight_layout()

    fig_path = f"{OUTPUT_ROOT}/master/random_avg_seed_vs_llm_citation_rate.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
