#!/usr/bin/env python3
"""F1 teaser: humans at the null, every model above. Paper-quality vector PDF.
On the user's machine with full texlive, switch USETEX=True for Times/usetex.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

USETEX = False
if USETEX:
    rcParams.update({"text.usetex": True, "font.family": "serif",
                     "text.latex.preamble": r"\usepackage{times}"})
else:
    rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix"})

rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 7.6, "axes.labelsize": 7.9, "xtick.labelsize": 6.8, "ytick.labelsize": 7.1,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.0,
    "xtick.major.size": 2.2, "ytick.major.size": 0,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "pdf.fonttype": 42,
})

M = json.load(open("node0_metrics.json"))
H = M["human"]; N = M["N"]
sub = H["sub"]

CO, CG, CA = "#2E5A87", "#3D7A47", "#B4653A"       # OpenAI / Google / Anthropic
CH, CN = "#9E1B1B", "#8F8D86"                       # humans / null
def vc(name):
    return CO if name.startswith("GPT") else CG if name.startswith("Gemini") else CA

models = [(name, sub[name]["top10"]*100, sub[name]["nexcl"]/N*100, vc(name))
          for name in sub]
models.sort(key=lambda r: r[1])                     # ascending -> top bar = largest
null_a, null_b = H["nullH"]["top10"]*100, H["nullH"]["excl"]/N*100
hum_a,  hum_b  = H["top10"]*100, H["nexcl"]/N*100

rows = [("Matched null", null_a, null_b, CN, "null"),
        ("Domain experts (pooled)", hum_a, hum_b, CH, "hum")] + \
       [(n, a, b, c, "model") for (n, a, b, c) in models]
ys = [0.0, 1.0] + [2.15 + i for i in range(len(models))]

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.18), sharey=True,
                         gridspec_kw=dict(wspace=0.07))
fig.subplots_adjust(left=0.175, right=0.992, top=0.885, bottom=0.185)

for ax, col, xlab, xmax, ttl in (
    (axes[0], 1, "top-10% citation share (%)", 34.5, "(a) Concentration"),
    (axes[1], 2, "papers never cited when shown (%)", 24.5, "(b) Exclusion"),
):
    nullv = rows[0][col]
    ax.axhspan(-0.62, 1.62, color="#F3F1EC", zorder=0)                 # humans+null band
    ax.axvline(nullv, color="#3A3834", ls=(0, (4, 2.5)), lw=0.9, zorder=6)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#DDDBD4", lw=0.45, zorder=0)
    for (name, a, b, c, kind), y in zip(rows, ys):
        v = (a, b)[col - 1]
        ax.barh(y, v, height=0.62, color=c, zorder=3,
                alpha=1.0 if kind != "null" else 0.9)
        ax.text(v + xmax*0.012, y, f"{v:.1f}", va="center", ha="left",
                fontsize=6.3, color="#5A5852", zorder=4)
    ax.axhline(1.72, color="#B9B6AE", lw=0.5, zorder=2)                # separator
    ax.set_xlim(0, xmax); ax.set_ylim(-0.75, ys[-1] + 0.75)
    ax.set_xlabel(xlab, labelpad=2)
    ax.set_title(ttl, loc="left", fontsize=8.6, fontweight="bold", pad=4)

axes[0].set_yticks(ys)
labels = axes[0].set_yticklabels([r[0] for r in rows])
labels[0].set_color(CN); labels[0].set_style("italic")
labels[1].set_color(CH); labels[1].set_fontweight("bold")

handles = [mpatches.Patch(color=CO, label="OpenAI"),
           mpatches.Patch(color=CG, label="Google"),
           mpatches.Patch(color=CA, label="Anthropic"),
           Line2D([0], [0], color="#4A4844", ls=(0, (4, 3)), lw=0.9, label="null")]
axes[1].legend(handles=handles, loc="lower right", frameon=False,
               fontsize=6.4, handlelength=1.35, handleheight=0.62,
               borderaxespad=0.3, labelspacing=0.35)

fig.savefig("figures/F1_teaser.pdf")
fig.savefig("teaser_preview.png", dpi=200)
print("wrote F1_teaser.pdf + preview")
