"""Shared loading + node-0 concentration metrics for The AI Citation Trap.

All computations start from four CSVs expected in the working directory:
    node0-all-models.csv                round-0 selections, 11 models
    all-models-all-nodes-selections.csv recursion selections, 8 models x 12 rounds
    node0_human_responses.csv           expert annotations (53 prompts)
    node0_seed_categories.csv           the 120 seed papers with titles/abstracts

Conventions (identical to the paper):
    - a bibliography is the SET of distinct resolved shown papers
    - the null inherits each prompt's panel and realized budget k_p, uniform picks
    - Monte Carlo nulls use NREDRAW=150 redraws, seed 0
"""
import json
import numpy as np
import pandas as pd

NREDRAW = 150
RNG = np.random.default_rng(0)

DISP = {  # display names
    "gpt-4.1-mini": "GPT-4.1 mini", "gpt-4.1": "GPT-4.1",
    "gpt-5-mini": "GPT-5 mini", "gpt-5": "GPT-5",
    "gemini-2.5-flash": "Gemini 2.5 Flash", "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "claude-haiku-4-5": "Claude Haiku 4.5", "claude-opus-4-8": "Claude Opus 4.8",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
}
VENDOR = lambda m: ("OpenAI" if m.startswith("gpt") else
                    "Google" if m.startswith("gemini") else "Anthropic")


def split_ids(cell):
    if not isinstance(cell, str) or not cell.strip():
        return []
    return [s.strip() for s in cell.split(";") if s.strip()]


def load_node0(path="node0-all-models.csv"):
    df = pd.read_csv(path)
    df["shown"] = df["papers_shown"].map(split_ids)
    df["cited"] = df["papers_selected"].map(split_ids)
    df["k"] = df["cited"].map(len)
    return df


def load_recursion(path="all-models-all-nodes-selections.csv"):
    return load_node0(path)


# ---- concentration metrics on one selector's table --------------------------

def counts(sub, papers):
    """citations, shows per paper over the rows of `sub` (one selector)."""
    c = {p: 0 for p in papers}
    n = {p: 0 for p in papers}
    for _, r in sub.iterrows():
        for p in r["shown"]:
            if p in n:
                n[p] += 1
        for p in r["cited"]:
            if p in c:
                c[p] += 1
    return c, n


def top10_hhi_excl(c, n):
    v = np.array(sorted(c.values(), reverse=True), float)
    tot = v.sum()
    ntop = max(1, int(round(0.10 * len(v))))
    top10 = v[:ntop].sum() / tot
    share = v / tot
    hhi = float((share ** 2).sum())
    excl = sum(1 for p in c if c[p] == 0 and n[p] > 0)
    return float(top10), hhi, excl


def null_redraws(sub, papers, nred=NREDRAW, rng=None):
    """Matched-null distribution of (top10, hhi, excl): uniform k_p-subsets of
    each prompt's own panel."""
    rng = rng or np.random.default_rng(0)
    outs = []
    panels = [(r["shown"], r["k"]) for _, r in sub.iterrows()]
    for _ in range(nred):
        c = {p: 0 for p in papers}
        n = {p: 0 for p in papers}
        for shown, k in panels:
            for p in shown:
                if p in n:
                    n[p] += 1
            k = min(k, len(shown))
            if k <= 0:
                continue
            for p in rng.choice(shown, size=k, replace=False):
                if p in c:
                    c[p] += 1
        outs.append(top10_hhi_excl(c, n))
    a = np.array(outs)
    return dict(top10=float(a[:, 0].mean()), top10_sd=float(a[:, 0].std()),
                hhi=float(a[:, 1].mean()),
                excl=float(a[:, 2].mean()), excl_sd=float(a[:, 2].std()))


def node0_summary(df, papers):
    out = {}
    for m, sub in df.groupby("model"):
        c, n = counts(sub, papers)
        t, h, e = top10_hhi_excl(c, n)
        out[m] = dict(top10=t, hhi=h, nexcl=e,
                      hall=float(sub["hallucination_rate"].mean()),
                      rates={p: (c[p] / n[p] if n[p] else np.nan) for p in papers},
                      shows={p: n[p] for p in papers},
                      cites={p: c[p] for p in papers})
    return out
