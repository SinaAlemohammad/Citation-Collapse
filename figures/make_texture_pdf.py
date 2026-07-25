#!/usr/bin/env python3
"""FI1: distributional texture of the recursion (Gini, top-1% share, openness)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                 "font.size": 7.6, "axes.labelsize": 8, "xtick.labelsize": 7,
                 "ytick.labelsize": 7, "axes.linewidth": 0.6,
                 "axes.spines.top": False, "axes.spines.right": False,
                 "pdf.fonttype": 42, "figure.facecolor": "white"})

A = json.load(open("appI_texture.json"))
CO, CG = "#2E5A87", "#3D7A47"
col = lambda m: CO if m.startswith("gpt") else CG
rounds = list(range(1, 12))

fig, axes = plt.subplots(1, 3, figsize=(6.5, 1.9))
fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.22, wspace=0.3)
panels = [("gini", "Gini (citations)", "(a)"),
          ("top1", "top-1% share", "(b)"),
          ("open", "newest-cohort share", "(c)")]
for ax, (key, ylab, tag) in zip(axes, panels):
    for m, g in A.items():
        ax.plot(rounds, [v for v in g[key]], lw=1.0, color=col(m), alpha=0.75)
    ax.set_xlabel("round")
    ax.set_ylabel(ylab)
    ax.set_title(tag, loc="left", fontsize=8, fontweight="bold")
axes[2].axhline(0.092, color="#3A3834", ls=(0, (4, 2.5)), lw=0.8)
axes[2].text(11, 0.094, "exposure", ha="right", fontsize=6.4, color="#3A3834")
fig.savefig("figures/FI1_texture.pdf")
print("wrote figures/FI1_texture.pdf")
