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

BASE = "."
MAX_NODES = 12
CITATION_CAP = 10
# OUTPUT_ROOT defined by shim above


def load_node_fp_counts(node: int, data_root: str) -> dict:
    total_raw = 0
    total_valid = 0
    path = f"{data_root}/node_{node}/node_{node}.jsonl"
    with open(path) as f:
        for line in f:
            data = json.loads(line.strip())
            raw = data.get("citations", data.get("citation_ids", []))
            valid = data.get("citation_ids", [])
            total_raw += len(raw)
            total_valid += len(valid)

    hallucinations = 0
    hall_path = f"{data_root}/citation_counts/hallucinations.jsonl"
    if os.path.exists(hall_path):
        with open(hall_path) as f:
            for line in f:
                data = json.loads(line.strip())
                for paper_id in data.keys():
                    if paper_id.startswith(f"N{node}P"):
                        hallucinations += 1

    over_cap = 0
    ocv_path = f"{data_root}/citation_counts/over_cap_violations.jsonl"
    if os.path.exists(ocv_path):
        with open(ocv_path) as f:
            for line in f:
                data = json.loads(line.strip())
                if data.get("node") == node:
                    over_cap += data["original_count"] - CITATION_CAP

    hallucination_rate = round(hallucinations / total_raw, 4) if total_raw > 0 else 0.0
    over_cap_rate = round(over_cap / total_raw, 4) if total_raw > 0 else 0.0
    return {
        "node": node,
        "total_raw": total_raw,
        "valid": total_valid,
        "hallucinations": hallucinations,
        "over_cap_truncated": over_cap,
        "hallucination_rate": hallucination_rate,
        "over_cap_rate": over_cap_rate,
    }


def run_for_root(data_root: str, label: str):
    output_dir = f"{data_root}/master"
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for node in range(MAX_NODES):
        row = load_node_fp_counts(node, data_root)
        results.append(row)
        print(f"[{label}] Node {row['node']}: total_raw={row['total_raw']}, valid={row['valid']}, "
              f"hallucinations={row['hallucinations']}, over_cap={row['over_cap_truncated']}, "
              f"hallucination_rate={row['hallucination_rate']}, over_cap_rate={row['over_cap_rate']}")

    with open(f"{output_dir}/false_positives.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {output_dir}/false_positives.jsonl")

    headers = ["Node", "Total Raw", "Valid", "Hallucinations", "Over-Cap Truncated", "Hallucination Rate", "Over-Cap Rate"]
    rows = []
    for r in results:
        rows.append([
            r["node"],
            r["total_raw"],
            r["valid"],
            r["hallucinations"],
            r["over_cap_truncated"],
            f"{r['hallucination_rate']:.4f}",
            f"{r['over_cap_rate']:.4f}",
        ])

    fig, ax = plt.subplots(figsize=(15, 4))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#D9E2F3")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/false_positives_table.png", dpi=200, bbox_inches="tight")
    print(f"Saved {output_dir}/false_positives_table.png")

    nodes = [r["node"] for r in results]
    halluc = [r["hallucinations"] for r in results]
    over_cap = [r["over_cap_truncated"] for r in results]
    halluc_rates = [r["hallucination_rate"] for r in results]
    over_cap_rates = [r["over_cap_rate"] for r in results]

    x = np.arange(len(nodes))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 5))
    bars1 = ax1.bar(x - width / 2, halluc, width, color="#E74C3C", alpha=0.8,
                    edgecolor="black", linewidth=0.5, label="Hallucinations")
    bars2 = ax1.bar(x + width / 2, over_cap, width, color="#E67E22", alpha=0.8,
                    edgecolor="black", linewidth=0.5, label="Over-Cap Truncated")
    ax1.bar_label(bars1, padding=3, fontsize=9)
    ax1.bar_label(bars2, padding=3, fontsize=9)
    ax1.set_xlabel("Node (Generation)")
    ax1.set_ylabel("Citation Count")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Node {n}" for n in nodes])

    ax2 = ax1.twinx()
    ax2.plot(x, halluc_rates, color="#2C3E50", marker="o", linewidth=2, label="Hallucination Rate (FP)")
    for i, rate in enumerate(halluc_rates):
        ax2.annotate(f"{rate:.3f}", (x[i], halluc_rates[i]), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8, color="#2C3E50")
    ax2.plot(x, over_cap_rates, color="#8E44AD", marker="s", linewidth=2,
             linestyle="--", label="Over-Cap Rate")
    for i, rate in enumerate(over_cap_rates):
        ax2.annotate(f"{rate:.3f}", (x[i], over_cap_rates[i]), textcoords="offset points",
                     xytext=(0, -14), ha="center", fontsize=8, color="#8E44AD")
    ax2.set_ylabel("Rate", color="#2C3E50")
    all_rates = halluc_rates + over_cap_rates
    max_rate = max(all_rates) if any(r > 0 for r in all_rates) else 0.05
    ax2.set_ylim(0, max_rate * 1.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.suptitle(f"Figure 4: False Positives by Node — {label}", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/false_positives.png", dpi=150, bbox_inches="tight")
    print(f"Saved {output_dir}/false_positives.png")


def main():
    run_for_root(OUTPUT_ROOT, "Experiment")


if __name__ == "__main__":
    main()
