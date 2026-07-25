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
from collections import defaultdict

from citation_collapse.analysis.metrics.random_aggregate import bonferroni_mean_ci, list_random_run_dirs


def _get_rate(entry):
    return entry["rate"]


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


def load_node_exposure(node: int, data_root: str) -> list:
    path = f"{data_root}/node_{node}/node_{node}_exposure.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_exposure_for_node(node: int, data_root: str) -> list:
    citations = defaultdict(int)
    exposures = defaultdict(int)

    stats_path = f"{data_root}/node_{node}/node_{node}_stats.jsonl"
    with open(stats_path) as f:
        for line in f:
            data = json.loads(line.strip())
            for paper_id, count in data.items():
                if paper_id in valid_ids:
                    citations[paper_id] += count

    node_path = f"{data_root}/node_{node}/node_{node}.jsonl"
    with open(node_path) as f:
        for line in f:
            data = json.loads(line.strip())
            papers_seen = data.get("papers_seen_id", [])
            for paper_id in papers_seen:
                if paper_id in valid_ids:
                    exposures[paper_id] += 1

    all_ids = set(citations.keys()) | set(exposures.keys())
    results = []
    for paper_id in all_ids:
        cite_count = citations[paper_id]
        expose_count = exposures[paper_id]
        rate = cite_count / expose_count if expose_count > 0 else 0.0
        results.append({
            "id": paper_id,
            "citations": cite_count,
            "exposures": expose_count,
            "rate": round(rate, 4),
        })
    results.sort(key=_get_rate, reverse=True)
    return results


def write_exposure_jsonl(node: int, results: list, data_root: str) -> None:
    path = f"{data_root}/node_{node}/node_{node}_exposure.jsonl"
    with open(path, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved {path}")


def compute_exclusion(node: int, available: int, entries: list) -> dict:
    cited = sum(1 for e in entries if e["citations"] > 0)
    shown_ignored = sum(1 for e in entries if e["exposures"] > 0 and e["citations"] == 0)
    shown_papers = cited + shown_ignored
    rate_shown = shown_ignored / shown_papers if shown_papers > 0 else 0.0
    never_shown = available - len(entries)
    uncited = shown_ignored + never_shown
    exclusion_rate = round(uncited / available, 4) if available > 0 else 0.0
    shown_ignored_rate = round(shown_ignored / available, 4) if available > 0 else 0.0
    never_shown_rate = round(never_shown / available, 4) if available > 0 else 0.0
    total_citations = sum(e["citations"] for e in entries)
    return {
        "node": node,
        "available": available,
        "cited": cited,
        "shown_ignored": shown_ignored,
        "shown_papers": shown_papers,
        "shown_ignored_rate_of_shown": round(rate_shown, 6),
        "never_shown": never_shown,
        "uncited": uncited,
        "exclusion_rate": exclusion_rate,
        "shown_ignored_rate": shown_ignored_rate,
        "never_shown_rate": never_shown_rate,
        "total_citations": total_citations,
    }


def run_for_root(
    data_root: str,
    label: str,
    *,
    skip_existing_exposure: bool = True,
) -> list:
    output_dir = f"{data_root}/analysis_figures"
    os.makedirs(output_dir, exist_ok=True)

    valid_ids.clear()
    kv_path = f"{data_root}/master/kv_pairs.jsonl"
    with open(kv_path) as f:
        for line in f:
            data = json.loads(line.strip())
            valid_ids.add(data["id"])

    excl_results = []
    for node in range(MAX_NODES):
        exp_path = f"{data_root}/node_{node}/node_{node}_exposure.jsonl"
        if skip_existing_exposure and os.path.isfile(exp_path):
            entries = load_node_exposure(node, data_root)
        else:
            entries = build_exposure_for_node(node, data_root)
            write_exposure_jsonl(node, entries, data_root)
        available = get_available_papers(node)
        row = compute_exclusion(node, available, entries)
        excl_results.append(row)
        print(
            f"[{label}] Node {node}: shown_ignored={row['shown_ignored']}, "
            f"shown_papers={row['shown_papers']}, "
            f"shown_ignored_rate_of_shown={row['shown_ignored_rate_of_shown']}, "
            f"cited={row['cited']}, total_citations={row['total_citations']}"
        )

    with open(f"{output_dir}/exclusion_rate_by_node.jsonl", "w") as f:
        for r in excl_results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {output_dir}/exclusion_rate_by_node.jsonl")

    return excl_results


def main():
    exp_results = run_for_root(
        OUTPUT_ROOT, "Experiment", skip_existing_exposure=True
    )

    run_dirs = list_random_run_dirs(RANDOM_ROOT)
    if not run_dirs:
        print(f"No run_* dirs under {RANDOM_ROOT}")
        return

    per_node: list[list[float]] = [[] for _ in range(MAX_NODES)]
    valid_run_count = 0
    for run_dir in run_dirs:
        try:
            excl = run_for_root(
                run_dir, f"Random({os.path.basename(run_dir)})", skip_existing_exposure=True
            )
        except (OSError, json.JSONDecodeError, KeyError) as e:
            print(f"[Random] skip {run_dir}: {e}")
            continue
        for node in range(MAX_NODES):
            per_node[node].append(excl[node]["shown_ignored_rate_of_shown"])
        valid_run_count += 1

    rnd_mean = []
    rnd_lo = []
    rnd_hi = []
    random_rows = []
    for node in range(MAX_NODES):
        vals = per_node[node]
        mean, lo, hi = bonferroni_mean_ci(vals, BONFERRONI_M_NODES)
        rnd_mean.append(mean)
        rnd_lo.append(max(0.0, lo))
        rnd_hi.append(min(1.0, hi))
        random_rows.append({
            "node": node,
            "n_runs": len(vals),
            "shown_ignored_rate_of_shown_mean": round(mean, 6),
            "shown_ignored_rate_of_shown_ci_low": round(max(0.0, lo), 6),
            "shown_ignored_rate_of_shown_ci_high": round(min(1.0, hi), 6),
        })

    out_master = f"{OUTPUT_ROOT}/master/false_negatives_random_baseline.jsonl"
    os.makedirs(os.path.dirname(out_master), exist_ok=True)
    with open(out_master, "w") as f:
        for row in random_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {out_master} ({valid_run_count} runs)")

    nodes = [r["node"] for r in exp_results]
    exp_rates = [r["shown_ignored_rate_of_shown"] for r in exp_results]
    x = np.arange(len(nodes))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, exp_rates, color="#10A37F", marker="o", linewidth=2, label="LLM")
    ax.plot(
        x,
        rnd_mean,
        color="#3498DB",
        marker="s",
        linewidth=2,
        linestyle="--",
        label=f"Mean of the random citers (n={valid_run_count} runs)",
    )

    ax.set_xlabel("Node (Generation)")
    ax.set_ylabel("Shown-but-Uncited Papers / Shown Papers (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Node {n}" for n in nodes])
    all_y = exp_rates + rnd_mean + rnd_hi
    max_rate = max(all_y) if all_y else 0.05
    ax.set_ylim(0, max_rate * 1.35)
    ax.legend(loc="upper left", fontsize=9)

    y_lo, y_hi = ax.get_ylim()
    for i, (ve, vr) in enumerate(zip(exp_rates, rnd_mean)):
        prev_e = exp_rates[i - 1] if i > 0 else None
        next_e = exp_rates[i + 1] if i < len(exp_rates) - 1 else None
        exp_off, _ = _label_offsets(ve, vr, prev_e, next_e, y_lo, y_hi)

        prev_r = rnd_mean[i - 1] if i > 0 else None
        next_r = rnd_mean[i + 1] if i < len(rnd_mean) - 1 else None
        rnd_off, _ = _label_offsets(vr, ve, prev_r, next_r, y_lo, y_hi)

        ax.annotate(f"{ve * 100:.1f}%", (x[i], ve), textcoords="offset points",
                    xytext=exp_off, ha="center", fontsize=8, color="#10A37F")
        ax.annotate(f"{vr * 100:.1f}%", (x[i], vr), textcoords="offset points",
                    xytext=rnd_off, ha="center", fontsize=8, color="#3498DB")
    fig.suptitle(
        "GPT 5 Mini False Negatives",
        fontsize=12,
    )
    fig.tight_layout()
    out_dir = f"{OUTPUT_ROOT}/analysis_figures"
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/exclusion_rate_by_node.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out_dir}/exclusion_rate_by_node.png")


if __name__ == "__main__":
    main()
