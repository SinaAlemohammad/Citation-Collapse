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
from scipy import stats

BASE = "."
# OUTPUT_ROOT defined by shim above
ALPHA = 0.05
M = 12
ALPHA_BONF = ALPHA / M

CONC_LLM_PATH = f"{OUTPUT_ROOT}/master/concentration.jsonl"
CONC_RND_PATH = f"{OUTPUT_ROOT}/master/concentration_random_baseline.jsonl"
FN_LLM_PATH = f"{OUTPUT_ROOT}/analysis_figures/exclusion_rate_by_node.jsonl"
FN_RND_PATH = f"{OUTPUT_ROOT}/master/false_negatives_random_baseline.jsonl"

CONC_OUT_PATH = f"{OUTPUT_ROOT}/master/concentration_ztest_table.png"
FN_OUT_PATH = f"{OUTPUT_ROOT}/analysis_figures/false_negatives_ztest_table.png"
SEED_LLM_PATH = f"{OUTPUT_ROOT}/master/seed_vs_llm_citation_rate.jsonl"
SEED_LLM_OUT_PATH = f"{OUTPUT_ROOT}/master/seed_vs_llm_ztest_table.png"

COL_LABELS = [
    "Node",
    "LLM Rate",
    "Random Rate",
    "z",
    "p-value",
    f"α (Bonf) = {ALPHA_BONF:.5f}",
]

M_SEED_LLM = 11
ALPHA_BONF_SEED_LLM = ALPHA / M_SEED_LLM

SEED_LLM_COL_LABELS = [
    "Node",
    "Seed Rate",
    "LLM-Paper Rate",
    "z",
    "p-value",
    f"α (Bonf) = {ALPHA_BONF_SEED_LLM:.5f}",
]

HEADER_COLOR = "#4472C4"
ALT_ROW_COLOR = "#D9E2F3"
REJECT_COLOR = "#70AD47"
FAIL_COLOR = "#C0392B"


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_pvalue(p):
    if not np.isfinite(p):
        return "N/A"
    if p < 0.0001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def z_test_right(p1, p2, n):
    """Right-tailed two-proportion z-test (H1: p1 > p2) with n1 = n2 = n."""
    p_hat = (p1 + p2) / 2
    se = (p_hat * (1 - p_hat) * (2 / n)) ** 0.5
    if se == 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / se
    p_value = float(1 - stats.norm.cdf(z))
    return z, p_value


def z_test_right_unequal(p1, n1, p2, n2):
    """Right-tailed two-proportion z-test (H1: p1 > p2) with unequal sample sizes."""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p_hat = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = (p_hat * (1 - p_hat) * (1 / n1 + 1 / n2)) ** 0.5
    if se == 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / se
    p_value = float(1 - stats.norm.cdf(z))
    return z, p_value


def render_table(cell_rows, decisions, title, out_path, col_labels=None):
    """
    cell_rows : list of 5-element string lists [Node, Rate1, Rate2, z, p-value]
    decisions : parallel list of "Reject H0" or "Fail to Reject H0"
    col_labels : optional custom column headers (defaults to COL_LABELS)
    """
    if col_labels is None:
        col_labels = COL_LABELS
    full_rows = [row + [decision] for row, decision in zip(cell_rows, decisions)]
    n_cols = len(col_labels)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis("off")

    table = ax.table(
        cellText=full_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor(HEADER_COLOR)
            cell.set_text_props(color="white", weight="bold")
        elif col_idx == n_cols - 1:
            decision = decisions[row_idx - 1]
            if decision == "Reject H0":
                cell.set_facecolor(REJECT_COLOR)
            else:
                cell.set_facecolor(FAIL_COLOR)
            cell.set_text_props(color="white", weight="bold")
        elif row_idx % 2 == 0:
            cell.set_facecolor(ALT_ROW_COLOR)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")


def concentration_table():
    llm_by_node = {r["node"]: r for r in load_jsonl(CONC_LLM_PATH)}
    rnd_by_node = {r["node"]: r for r in load_jsonl(CONC_RND_PATH)}

    cell_rows = []
    decisions = []

    for node in sorted(llm_by_node):
        llm = llm_by_node[node]
        rnd = rnd_by_node[node]

        p1 = llm["top_10pct_share"] / 100.0
        p2 = rnd["top_10pct_share_mean"] / 100.0
        n = llm["total_citations"]

        z, p_value = z_test_right(p1, p2, n)
        decision = "Reject H0" if np.isfinite(p_value) and p_value < ALPHA_BONF else "Fail to Reject H0"

        cell_rows.append([
            str(node),
            f"{p1 * 100:.2f}%",
            f"{p2 * 100:.2f}%",
            f"{z:.3f}",
            format_pvalue(p_value),
        ])
        decisions.append(decision)

    title = (
        "GPT 5 Mini Citation Concentration: Top-10% Share by Node\n"
        "Right-tailed two-proportion z-test  |  Ha: p_LLM > p_random"
    )
    render_table(cell_rows, decisions, title, CONC_OUT_PATH)


def false_negatives_table():
    llm_by_node = {r["node"]: r for r in load_jsonl(FN_LLM_PATH)}
    rnd_by_node = {r["node"]: r for r in load_jsonl(FN_RND_PATH)}

    cell_rows = []
    decisions = []

    for node in sorted(llm_by_node):
        llm = llm_by_node[node]
        rnd = rnd_by_node[node]

        p1 = llm["shown_ignored_rate_of_shown"]
        p2 = rnd["shown_ignored_rate_of_shown_mean"]
        n = llm["shown_papers"]

        z, p_value = z_test_right(p1, p2, n)
        decision = "Reject H0" if np.isfinite(p_value) and p_value < ALPHA_BONF else "Fail to Reject H0"

        cell_rows.append([
            str(node),
            f"{p1 * 100:.2f}%",
            f"{p2 * 100:.2f}%",
            f"{z:.3f}",
            format_pvalue(p_value),
        ])
        decisions.append(decision)

    title = (
        "GPT 5 Mini False Negatives\n"
        "Right-tailed two-proportion z-test  |  Ha: p_LLM > p_random"
    )
    render_table(cell_rows, decisions, title, FN_OUT_PATH)


def seed_vs_llm_table():
    rows_by_node = {r["node"]: r for r in load_jsonl(SEED_LLM_PATH)}

    cell_rows = []
    decisions = []

    for node in sorted(rows_by_node):
        if node == 0:
            continue

        r = rows_by_node[node]

        p1 = r["seed_rate"]
        n1 = r["seed_exposures"]
        p2 = r["llm_rate"] if r["llm_rate"] is not None else 0.0
        n2 = r["llm_exposures"]

        if p1 is None or n1 == 0:
            z, p_value = float("nan"), float("nan")
        else:
            z, p_value = z_test_right_unequal(p1, n1, p2, n2)

        decision = "Reject H0" if np.isfinite(p_value) and p_value < ALPHA_BONF_SEED_LLM else "Fail to Reject H0"

        cell_rows.append([
            str(node),
            f"{p1 * 100:.2f}%" if p1 is not None else "N/A",
            f"{p2 * 100:.2f}%",
            f"{z:.3f}" if np.isfinite(z) else "N/A",
            format_pvalue(p_value),
        ])
        decisions.append(decision)

    title = (
        "GOT 5 Mini Citation Rate by Node\n"
        "Right-tailed two-proportion z-test  |  Ha: p_seed > p_llm"
    )
    render_table(cell_rows, decisions, title, SEED_LLM_OUT_PATH, col_labels=SEED_LLM_COL_LABELS)


if __name__ == "__main__":
    concentration_table()
    false_negatives_table()
    seed_vs_llm_table()
