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
import sys

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

BASE = "."
MAX_NODES = 12
# OUTPUT_ROOT defined by shim above


def _label_offsets(val, other_val, prev_val, next_val, y_lo, y_hi):
    y_range = y_hi - y_lo
    margin = 0.06 * y_range

    want_above = val >= other_val

    if prev_val is not None and (val - prev_val) > 0.04 * y_range:
        want_above = True

    if not want_above and val < y_lo + margin:
        want_above = True
    if want_above and val > y_hi - margin:
        want_above = False

    gap = abs(val - other_val) / y_range if y_range > 0 else 1
    above_pts = 7 if gap > 0.04 else 11
    below_pts = -14 if gap > 0.04 else -18

    if want_above:
        return (0, above_pts), (0, below_pts)
    else:
        return (0, below_pts), (0, above_pts)


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

    seed_rate = (
        seed_citations / seed_exposures if seed_exposures > 0 else None
    )
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
    results = []
    for node in range(MAX_NODES):
        r = rates_for_node(node, OUTPUT_ROOT)
        results.append({
            "node": node,
            "seed_citations": r["seed_citations"],
            "seed_exposures": r["seed_exposures"],
            "seed_rate": round(r["seed_rate"], 6) if r["seed_rate"] is not None else None,
            "llm_citations": r["llm_citations"],
            "llm_exposures": r["llm_exposures"],
            "llm_rate": round(r["llm_rate"], 6) if r["llm_rate"] is not None else None,
        })

    print(f"{'Node':>4}  {'Seed Rate':>10}  {'Seed C/E':>14}  {'LLM Rate':>10}  {'LLM C/E':>14}")
    print("-" * 60)
    for r in results:
        seed_str = f"{r['seed_citations']}/{r['seed_exposures']}"
        llm_str = f"{r['llm_citations']}/{r['llm_exposures']}"
        seed_rate = f"{r['seed_rate']:.4f}" if r["seed_rate"] is not None else "N/A"
        llm_rate = f"{r['llm_rate']:.4f}" if r["llm_rate"] is not None else "N/A"
        print(
            f"{r['node']:>4}  {seed_rate:>10}  {seed_str:>14}  {llm_rate:>10}  {llm_str:>14}"
        )

    out_path = f"{OUTPUT_ROOT}/master/seed_vs_llm_citation_rate.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {out_path}")

    nodes = [r["node"] for r in results]
    seed_rates = [r["seed_rate"] for r in results]
    llm_rates = [r["llm_rate"] if r["llm_rate"] is not None else 0 for r in results]

    x = np.arange(len(nodes))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        x,
        seed_rates,
        color="#10A37F",
        marker="o",
        linewidth=2,
        label="LLM citer — seed papers",
    )
    ax.plot(
        x,
        llm_rates,
        color="#4A90D9",
        marker="s",
        linewidth=2,
        linestyle="--",
        label="LLM citer — LLM papers",
    )

    ax.set_xlabel("Node (Generation)")
    ax.set_ylabel("Citation Rate (citations / exposures) %")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Node {n}" for n in nodes])
    all_vals = [v for v in seed_rates + llm_rates if v is not None]
    y_top = max(all_vals) * 1.22 if all_vals else 0.1
    ax.set_ylim(0, y_top)
    ax.legend(loc="best", fontsize=8)

    y_lo, y_hi = ax.get_ylim()
    for i, (vs, vl) in enumerate(zip(seed_rates, llm_rates)):
        vs_val = vs if vs is not None else 0

        prev_s = (seed_rates[i - 1] if seed_rates[i - 1] is not None else 0) if i > 0 else None
        next_s = (seed_rates[i + 1] if seed_rates[i + 1] is not None else 0) if i < len(seed_rates) - 1 else None
        seed_off, _ = _label_offsets(vs_val, vl, prev_s, next_s, y_lo, y_hi)

        prev_l = llm_rates[i - 1] if i > 0 else None
        next_l = llm_rates[i + 1] if i < len(llm_rates) - 1 else None
        llm_off, _ = _label_offsets(vl, vs_val, prev_l, next_l, y_lo, y_hi)

        if vs is not None:
            ax.annotate(f"{vs * 100:.1f}%", (x[i], vs), textcoords="offset points",
                        xytext=seed_off, ha="center", fontsize=8, color="#10A37F")
        ax.annotate(f"{vl * 100:.1f}%", (x[i], vl), textcoords="offset points",
                    xytext=llm_off, ha="center", fontsize=8, color="#4A90D9")
    fig.suptitle(
        "GPT 5 Mini Citation Rate of Seed vs LLM-Generated Papers",
        fontsize=12,
    )
    fig.tight_layout()

    fig_path = f"{OUTPUT_ROOT}/master/seed_vs_llm_citation_rate.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
