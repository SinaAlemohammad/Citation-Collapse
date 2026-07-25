#!/usr/bin/env python3
"""Round-0 paper figures: F3 (one map), F5 (symmetry), F4 (map identity).
White background, STIX serif, vector PDF, textwidth = 6.5in. Seed 0."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib import rcParams
from scipy import stats

rng = np.random.default_rng(0)
OUT = "figures/"

rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 7.5, "axes.labelsize": 7.6, "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
    "legend.fontsize": 6.0, "axes.linewidth": 0.55,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})
CO, CG, CA = "#2E5A87", "#3D7A47", "#B4653A"
CH, CN = "#9E1B1B", "#8F8D86"
def vc(n): return CO if n.startswith("GPT") else CG if n.startswith("Gemini") else CA
def ptitle(ax, s): ax.set_title(s, loc="left", fontsize=7.8, fontweight="bold", pad=3)

M = json.load(open("node0_metrics.json"))
models = M["models"]; disp = M["disp"]; names = {m: disp[m] for m in models}
N = M["N"]; papers = M["papers"]; H = M["human"]; S = M["summary"]

# ---------- shared machinery (maps, rates) ----------
df = pd.read_csv("node0-all-models.csv")
def parse(s): return frozenset(x.strip() for x in str(s).split(";") if x.strip().startswith("SEED"))
df["shown"] = df.papers_shown.apply(parse); df["sel"] = df.papers_selected.apply(parse)
prompts = sorted(df.prompt_id.unique()); P = len(prompts)
data = {m: df[df.model == m].set_index("prompt_id") for m in models}
ref = data[models[0]]; pidx = {j: i for i, j in enumerate(papers)}
X = np.zeros((P, N), bool)
for a, p in enumerate(prompts):
    for j in ref.loc[p, "shown"]: X[a, pidx[j]] = True
ne = X.sum(0)
Rmat = np.zeros((len(models), N)); Dmaps = []
for mi, m in enumerate(models):
    Y = np.zeros((P, N), bool); k = np.zeros(P, int)
    for a, p in enumerate(prompts):
        for j in data[m].loc[p, "sel"]:
            if j in pidx: Y[a, pidx[j]] = True
        k[a] = int(Y[a].sum())
    r = (Y & X).sum(0) / np.maximum(ne, 1); Rmat[mi] = r
    with np.errstate(invalid="ignore"):
        rn = np.array([(k[X[:, j]] / 30).mean() if ne[j] > 0 else np.nan for j in range(N)])
    Dmaps.append(r - rn)
D = np.stack(Dmaps)
covc = np.cov(D)
for i in range(len(models)):
    covc[i, i] -= float(np.nanmean(Rmat[i] * (1 - Rmat[i]) / np.maximum(ne, 1)))
w, _ = np.linalg.eigh(covc); w = w[::-1]; spec = w / w.sum() * 100

def pb_pmf(ps):
    pmf = np.zeros(len(ps) + 1); pmf[0] = 1.0
    for p_ in ps:
        pmf[1:] = pmf[1:] * (1 - p_) + pmf[:-1] * p_; pmf[0] *= (1 - p_)
    return pmf
pred = np.zeros(12)
for j in range(N): pred += ne[j] * pb_pmf(Rmat[:, j])
pred /= ne.sum()
agree = np.array(M["agree"]); L = len(models); p0 = M["p0"]
obs = np.array([np.mean(agree == c) for c in range(12)])
ind = np.array([stats.binom.pmf(c, L, p0) for c in range(12)])

# ============================================================ F3a: heatmap (wrapfigure, 0.40\textwidth)
HEAT = ["GPT-5 mini", "GPT-5", "Gemini 3.1 Pro", "Claude Sonnet 4.6", "Claude Opus 4.8",
        "Gemini 2.5 Pro", "Gemini 2.5 Flash", "GPT-4.1", "GPT-4.1 mini", "Claude Haiku 4.5",
        "Gemini 3.1 Flash-Lite"]
dis = np.array(M["dis"]); inv = {names[m]: m for m in models}
ix = [models.index(inv[h]) for h in HEAT]; Dm = dis[np.ix_(ix, ix)]

# axes box made exactly square (width 0.565*2.6in == height 0.695*2.113in),
# so the equal-aspect heatmap fills it and the colorbar matches its height
fig, ax = plt.subplots(figsize=(2.6, 2.113),
                       gridspec_kw=dict(left=0.315, right=0.88, top=0.985, bottom=0.29))
im = ax.imshow(Dm, vmin=.5, vmax=1, cmap="YlGnBu")
ax.set_xticks(range(11)); ax.set_xticklabels(HEAT, rotation=55, ha="right", fontsize=5.3)
ax.set_yticks(range(11)); ax.set_yticklabels(HEAT, fontsize=5.3)
for i in range(11):
    for j in range(11):
        ax.text(j, i, f"{Dm[i, j]:.2f}".lstrip("0") if Dm[i, j] < 1 else "1.0",
                ha="center", va="center", fontsize=4.0,
                color="white" if Dm[i, j] > .82 else "#111111")
ax.add_patch(mp.Rectangle((-.5, -.5), 4, 4, fill=False, edgecolor=CH, lw=1.2))
ax.tick_params(length=0)
cax = fig.add_axes([0.905, 0.29, 0.032, 0.695])
cb = fig.colorbar(im, cax=cax)
cb.ax.tick_params(labelsize=5.2, length=1.5, width=0.4)
cb.outline.set_linewidth(0.4)
fig.savefig(OUT + "F3_heatmap.pdf")
fig.savefig("F3a_paper_preview.png", dpi=180)
plt.close(fig); print("wrote F3_heatmap.pdf")

# ============================================================ F3c: vote histogram (wrapfigure, 0.40\textwidth)
fig, ax = plt.subplots(figsize=(2.6, 1.95),
                       gridspec_kw=dict(left=0.145, right=0.985, top=0.96, bottom=0.21))
xs = np.arange(12); w_ = .28
ax.bar(xs - w_, obs * 100, w_, color=CO, label="observed")
ax.bar(xs, pred * 100, w_, color="#7FA5C6", label="predicted (shared maps)")
ax.bar(xs + w_, ind * 100, w_, color="#C9C4B4", label="independent selectors")
ax.set_xlabel("models (of 11) citing the shown paper", labelpad=1, fontsize=6.8)
ax.set_ylabel("% of shown candidates", labelpad=1, fontsize=6.8)
ax.set_xticks([0, 3, 6, 9, 11]); ax.set_ylim(0, 26); ax.set_yticks([0, 10, 20])
ax.legend(frameon=False, handlelength=1.0, borderaxespad=0.1, labelspacing=0.3, fontsize=5.7)
fig.savefig(OUT + "F3_votes.pdf")
fig.savefig("F3c_paper_preview.png", dpi=180)
plt.close(fig); print("wrote F3_votes.pdf")

# ============================================================ F5
h = pd.read_csv("node0_human_responses.csv"); h["seed"] = h.seed_id
hp = sorted(h.prompt_id.unique())
hs = {p: frozenset(h[(h.prompt_id == p) & (h.cited == 1)].seed) for p in hp}
Xh = np.zeros((len(hp), N), bool)
for a, p in enumerate(hp):
    for j in ref.loc[p, "shown"]: Xh[a, pidx[j]] = True
Yh = np.zeros((len(hp), N), bool); kh = np.zeros(len(hp), int)
for a, p in enumerate(hp):
    for j in hs[p]:
        if j in pidx: Yh[a, pidx[j]] = True
    kh[a] = int(Yh[a].sum())
def rel_of(Y, k, Xs, iters=60):
    Pn = len(k); rs = []
    for _ in range(iters):
        pm = rng.permutation(Pn); halves = (pm[:Pn // 2], pm[Pn // 2:]); ds = []
        for hh in halves:
            Xs2, Ys, ks = Xs[hh], Y[hh], k[hh]; nes = Xs2.sum(0)
            with np.errstate(invalid="ignore"):
                rr = np.where(nes > 0, (Ys & Xs2).sum(0) / np.maximum(nes, 1), np.nan)
                rn = np.array([(ks[Xs2[:, j]] / 30).mean() if nes[j] > 0 else np.nan
                               for j in range(N)])
            ds.append(rr - rn)
        ok = ~(np.isnan(ds[0]) | np.isnan(ds[1]))
        if ok.sum() > 10:
            r = np.corrcoef(ds[0][ok], ds[1][ok])[0, 1]; rs.append(2 * r / (1 + r))
    return float(np.mean(rs))
hrel = rel_of(Yh, kh, Xh)
print(f"  human reliability {hrel:.3f}")

fig, ax = plt.subplots(figsize=(2.73, 2.42),
                       gridspec_kw=dict(left=0.155, right=0.975, top=0.97, bottom=0.16))
s_, Mn = 0.25, 120
Rg = np.linspace(.0005, .5, 600)
bd = ((1 - s_ * Rg) ** Mn - (1 - s_) ** Mn) / (1 - (1 - s_) ** Mn)
ax.fill_between(Rg, bd, 0.3, color="#EDF2E8", zorder=0)
ax.plot(Rg, bd, "--", color="#3A3834", lw=1.0, zorder=2)
for m in models:
    e = M["exch"][m]
    ax.scatter(e["R"], e["obs"], color=vc(names[m]), s=26, zorder=4,
               edgecolor="white", lw=0.4)
ax.scatter(H["R"], H["nexcl"] / N, marker="o", color=CH, s=26, zorder=5,
           edgecolor="white", lw=0.4)
ax.text(.185, .135, "infeasible under\nexchangeability:\na stable paper-level\nmap is forced",
        color="#3B6D11", fontsize=6.0, style="italic", ha="center")
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], ls="--", color="#3A3834", lw=1.0, label="exchangeable bound"),
           Line2D([0], [0], marker="o", ls="", color=CO, ms=4.5, label="OpenAI"),
           Line2D([0], [0], marker="o", ls="", color=CG, ms=4.5, label="Google"),
           Line2D([0], [0], marker="o", ls="", color=CA, ms=4.5, label="Anthropic"),
           Line2D([0], [0], marker="o", ls="", color=CH, ms=4.5, label="domain experts")]
ax.legend(handles=handles, frameon=False, loc="upper right",
          borderaxespad=0.15, fontsize=5.9, handlelength=1.3, labelspacing=0.35)
ax.set_xlabel("citation rate when shown, $R$", labelpad=1)
ax.set_ylabel("fraction never cited when shown", labelpad=1)
ax.set_xlim(0, 0.5); ax.set_ylim(0, 0.3)
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5]); ax.set_yticks([0, 0.1, 0.2, 0.3])
fig.savefig(OUT + "F5_symmetry.pdf")
fig.savefig("F5_paper_preview.png", dpi=180)
plt.close(fig); print("wrote F5_symmetry.pdf")

# ============================================================ F4 (1x4)
CORE = ["teacher", "student", "soft", "logit", "temperature", "dark knowledge", "mimic",
        "distill", "feature", "layer", "hint", "representation"]
APP = ["video", "3d", "speech", "federated", "reinforcement", "policy", "agent",
       "translation", "recommend", "medical", "graph", "detection", "segmentation",
       "cross-modal", "multimodal"]
cat = pd.read_csv("node0_seed_categories.csv")
if "seed" not in cat.columns: cat["seed"] = "SEED_" + cat.seed_id.astype(str)
cat = cat.set_index("seed").reindex(papers)
def dens(t_, wl):
    t_ = str(t_).lower(); return sum(t_.count(x) for x in wl) / max(len(t_.split()), 1) * 100
txt = cat.title.fillna("") + " " + cat.abstract.fillna("")
cor = (txt.apply(lambda t_: dens(t_, CORE)) - txt.apply(lambda t_: dens(t_, APP))).values
dbar = np.array(M["dbar"]); dh = np.array([np.nan if x is None else x for x in H["d"]])
cat["dom"] = cat.domain.fillna("Other").apply(lambda d: str(d).split(";")[0].strip())
dom_dum = pd.get_dummies(cat["dom"]).values.astype(float)
def r2(Xd):
    Xd = np.column_stack([np.ones(N), Xd])
    bb, _, _, _ = np.linalg.lstsq(Xd, dbar, rcond=None); pr = Xd @ bb
    return 1 - ((dbar - pr) ** 2).sum() / ((dbar - dbar.mean()) ** 2).sum()
R2c, R2d, R2b = r2(cor), r2(dom_dum), r2(np.column_stack([cor, dom_dum]))
print(f"  R2 {R2c:.2f}/{R2d:.2f}/{R2b:.2f}")
b = np.polyfit(cor, dbar, 1); resid = dbar - (b[0] * cor + b[1])
rr = []
for g, subc in cat.groupby("dom"):
    idx = [papers.index(s) for s in subc.index]
    if len(idx) >= 5: rr.append((g, len(idx), float(np.mean(resid[idx]))))
rr.sort(key=lambda r_: r_[2])

fig, axes = plt.subplots(1, 3, figsize=(6.5, 1.85),
                         gridspec_kw=dict(wspace=0.42, width_ratios=[1, 1, 1.18],
                                          left=0.065, right=0.975,
                                          top=0.84, bottom=0.27))
for ax, y, labl, c, tag in [(axes[0], dbar, "LLM consensus", CO, "a"),
                            (axes[1], dh, "Human experts", CH, "b")]:
    ok = ~np.isnan(y)
    ax.scatter(cor[ok], y[ok], s=6, color=c, alpha=.55, edgecolor="none")
    sl, ic, rv, _, _ = stats.linregress(cor[ok], y[ok])
    xs = np.linspace(cor.min(), cor.max(), 10)
    ax.plot(xs, ic + sl * xs, color="#111111", lw=0.9)
    ax.axhline(0, color="#E2DFD8", lw=0.7)
    ax.set_xlabel("coreness index", labelpad=1); ax.set_ylim(-.42, .58)
    ptitle(ax, f"({tag}) {labl}   $r = {rv:.2f}$")
axes[0].set_ylabel("deviation from null", labelpad=1)
axes[1].set_yticklabels([])

ax = axes[2]
ax.barh(range(len(rr)), [r_[2] for r_ in rr],
        color=[CG if r_[2] > 0.03 else "#B9B6AE" for r_ in rr], height=.58,
        edgecolor="white", lw=0.3)
ax.set_yticks(range(len(rr)))
ax.set_yticklabels([f"{r_[0]} ({r_[1]})" for r_ in rr], fontsize=5.2)
ax.axvline(0, color="#3A3834", lw=0.7)
ax.set_xlabel("domain mean\nafter coreness removed", labelpad=1, fontsize=6.6)
ptitle(ax, "(c) Only Theory survives")
ax.text(0.99, -0.33, "($n$) papers per domain", transform=ax.transAxes,
        ha="right", fontsize=4.8, color="#8A8880")
fig.savefig(OUT + "F4_map_identity.pdf")
fig.savefig("F4_paper_preview.png", dpi=180)
plt.close(fig); print("wrote F4_map_identity.pdf")
