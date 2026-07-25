#!/usr/bin/env python3
"""Recursion paper figures: F6 (collapse persists + funnel), F7 (within-panel
competition test). White bg, STIX, vector PDF, textwidth 6.5in. Seed 0."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

OUT = "figures/"
rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 7.5, "axes.labelsize": 7.4, "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
    "legend.fontsize": 5.4, "axes.linewidth": 0.55,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})
NUL = "#8F8D86"
def ptitle(ax, s): ax.set_title(s, loc="left", fontsize=7.8, fontweight="bold", pad=3)

A = json.load(open("recursion_metrics.json"))
REG = json.load(open("competitor_regression.json"))
ROUNDS = list(range(12))
SERIES = [("gpt-4.1-mini", "GPT-4.1 mini", "#D9A441", "o"),
          ("gpt-4.1", "GPT-4.1", "#C2703A", "o"),
          ("gpt-5-mini", "GPT-5 mini", "#5B8FC9", "o"),
          ("gpt-5", "GPT-5", "#12355B", "o"),
          ("gemini-2.5-flash", "Gemini 2.5 Flash", "#7FB086", "s"),
          ("gemini-2.5-pro", "Gemini 2.5 Pro", "#2F6B3A", "s"),
          ("gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite", "#8FC4C4", "^"),
          ("gemini-3.1-pro-preview", "Gemini 3.1 Pro", "#175E5E", "^")]
def V(m, key, scale=100):
    return [(x * scale if x is not None else np.nan) for x in A[m][key]]

# ============================================================ F6 (1x3)
fig, axes = plt.subplots(1, 3, figsize=(6.5, 1.95),
                         gridspec_kw=dict(wspace=0.34, left=0.06, right=0.985,
                                          top=0.86, bottom=0.21))
ax = axes[0]
for m, lab, c, mk in SERIES:
    ax.plot(ROUNDS, V(m, "top10"), "-", marker=mk, color=c, lw=1.1, ms=2.2)
ax.plot(ROUNDS, V("gpt-5", "top10_null"), "--", color=NUL, lw=1.2, zorder=5)
ax.text(10.6, 27.2, "null", fontsize=6.0, color="#4A4844", style="italic", ha="right")
ax.set_xlabel("round", labelpad=1); ax.set_ylabel("top-10% share (%)", labelpad=1)
ax.set_xticks([0, 3, 6, 9, 11])
ptitle(ax, "(a) Above a rising null at every round")

ax = axes[1]
for m, lab, c, mk in SERIES:
    ax.plot(ROUNDS, V(m, "gen_above_null"), "-", marker=mk, color=c, lw=1.1, ms=2.2, label=lab)
ax.axhline(0, color="#3A3834", lw=0.7)
ax.set_ylim(-2.5, 22)
ax.set_yticks([0, 5, 10])
ax.legend(frameon=False, ncol=2, loc="upper left", handlelength=1.1, fontsize=5.0,
          borderaxespad=0.15, labelspacing=0.26, columnspacing=0.6)
ax.set_xlabel("round", labelpad=1); ax.set_ylabel("generated-only excess (pp)", labelpad=1)
ax.set_xticks([0, 3, 6, 9, 11])
ptitle(ax, "(b) Survives with seeds removed")

ax = axes[2]
for m, lab, c, mk in SERIES:
    ax.plot(ROUNDS, V(m, "seed_cr"), "-", marker=mk, color=c, lw=1.1, ms=2.2)
ax.plot(ROUNDS, V("gpt-5", "seed_cr_null"), "--", color=NUL, lw=1.2, zorder=5)
ax.text(10.6, 25.5, "null (flat)", fontsize=6.0, color="#4A4844", style="italic", ha="right")
ax.set_xlabel("round", labelpad=1); ax.set_ylabel("shown seed cited (%)", labelpad=1)
ax.set_xticks([0, 3, 6, 9, 11]); ax.set_ylim(15, 100)
ptitle(ax, "(c) The seed funnel")
fig.savefig(OUT + "F6_recursion.pdf")
fig.savefig("F6_paper_preview.png", dpi=180)
plt.close(fig); print("wrote F6_recursion.pdf")

# ============================================================ F7 (1x3)
big = pd.read_csv("all-models-all-nodes-selections.csv")
def parse_l(s): return [x.strip() for x in str(s).split(";") if x.strip() and x.strip() != "nan"]

fig, axes = plt.subplots(1, 3, figsize=(6.5, 1.95),
                         gridspec_kw=dict(wspace=0.60, width_ratios=[1.06, 0.98, 0.92],
                                          left=0.065, right=0.985, top=0.86, bottom=0.21))
ax = axes[0]
REP = [("gemini-3.1-pro-preview", "Gemini 3.1 Pro", "#175E5E"),
       ("gpt-5", "GPT-5", "#12355B"),
       ("gpt-5-mini", "GPT-5 mini", "#5B8FC9"),
       ("gpt-4.1-mini", "GPT-4.1 mini", "#D9A441")]
for m, lab, c in REP:
    dfm = big[(big.model == m) & (big.node >= 1)]
    xs_all, ys_all = [], []
    for shown, sel in zip(dfm.papers_shown, dfm.papers_selected):
        sh = parse_l(shown); se = set(parse_l(sel))
        seeds = [s for s in sh if s.startswith("SEED")]; ns = len(seeds)
        for s in seeds:
            xs_all.append(ns - 1); ys_all.append(1.0 if s in se else 0.0)
    xs_all = np.array(xs_all); ys_all = np.array(ys_all)
    bx = np.arange(0, 18)
    by = [ys_all[xs_all == b].mean() * 100 if (xs_all == b).sum() > 30 else np.nan for b in bx]
    ax.plot(bx, by, "-o", color=c, lw=1.2, ms=2.4, label=lab)
knull = np.mean([np.mean(A[m]["k"][1:]) for m, _, _, _ in SERIES]) / 30 * 100
ax.axhline(knull, color=NUL, ls="--", lw=1.1)
ax.text(16.7, knull - 7, "null", fontsize=6.0, color="#4A4844", style="italic", ha="right")
ax.set_xlabel("seed competitors in panel ($x$)", labelpad=1)
ax.set_ylabel("shown seed cited (%)", labelpad=1)
ax.set_xticks([0, 5, 10, 15])
ax.legend(frameon=False, handlelength=1.2, borderaxespad=0.15, labelspacing=0.3)
ptitle(ax, "(a) Same seed, more rivals")

ax = axes[1]
order = np.argsort([REG[m]["beta"] for m, _, _, _ in SERIES])
for row, oi in enumerate(order):
    m, lab, c, mk = SERIES[oi]; r = REG[m]
    ax.errorbar(r["beta"], row, xerr=1.96 * r["se"], fmt=mk, color=c, ms=3.6,
                capsize=2.0, lw=1.0, elinewidth=0.9)
ax.axvline(0, color="#3A3834", lw=0.7)
ax.set_yticks(range(8))
ax.set_yticklabels([SERIES[oi][1] for oi in order], fontsize=5.4)
ax.set_xlabel(r"$\beta$ (pp per seed competitor)", labelpad=1)
ptitle(ax, "(b) All eight negative")

ax = axes[2]
for m, lab, c, mk in SERIES:
    ax.scatter(REG[m]["beta"], A[m]["seed_excess"][11] * 100, color=c, s=26,
               marker=mk, zorder=4, edgecolor="white", lw=0.4)
ax.annotate("Gemini 2.5 pair\nreverses on both",
            (REG["gemini-2.5-pro"]["beta"], A["gemini-2.5-pro"]["seed_excess"][11] * 100),
            xytext=(-5, 8), textcoords="offset points", fontsize=5.8, color="#2F6B3A", ha="right")
ax.set_xlabel(r"$\beta$ (pp per seed competitor)", labelpad=1)
ax.set_ylabel("round-11 seed excess (pp)", labelpad=1)
ptitle(ax, "(c) Micro matches macro")
fig.savefig(OUT + "F7_mechanism.pdf")
fig.savefig("F7_paper_preview.png", dpi=180)
plt.close(fig); print("wrote F7_mechanism.pdf")
