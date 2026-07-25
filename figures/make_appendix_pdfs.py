#!/usr/bin/env python3
"""Appendix figures: FD1 identity check, FE1 human study (1x2), FF1 text model."""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
OUT = "figures/"
rcParams.update({
    "font.family":"STIXGeneral","mathtext.fontset":"stix",
    "figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white",
    "font.size":7.5,"axes.labelsize":7.4,"xtick.labelsize":6.6,"ytick.labelsize":6.6,
    "legend.fontsize":6.0,"axes.linewidth":0.55,
    "xtick.major.width":0.5,"ytick.major.width":0.5,"xtick.major.size":2.0,"ytick.major.size":2.0,
    "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
CO,CG,CA,CH,CN="#2E5A87","#3D7A47","#B4653A","#9E1B1B","#8F8D86"
def vc(n): return CO if n.startswith("GPT") else CG if n.startswith("Gemini") else CA
def ptitle(ax,s): ax.set_title(s,loc="left",fontsize=7.8,fontweight="bold",pad=3)

# ---------- FD1: identity check ----------
D=json.load(open("appD_identity.json"))
fig,ax=plt.subplots(figsize=(3.05,2.55),gridspec_kw=dict(left=0.155,right=0.97,top=0.97,bottom=0.155))
lim=[2.2,7.2]
ax.plot(lim,lim,color="#B9B6AE",lw=0.8,zorder=1)
for n,v in D.items():
    ax.scatter(v["obs"],v["rec"],color=vc(n),s=26,zorder=4,edgecolor="white",lw=0.4)
obs=np.array([v["obs"] for v in D.values()]); rec=np.array([v["rec"] for v in D.values()])
r=np.corrcoef(obs,rec)[0,1]
ax.text(0.05,0.93,f"$r = {r:.3f}$\nmean rel. dev. 1.6%",transform=ax.transAxes,fontsize=6.6,va="top")
ax.set_xlabel(r"observed excess HHI ($\times 10^{3}$)",labelpad=1)
ax.set_ylabel(r"reconstructed from $(\|D\|_F^2,\rho)$ ($\times 10^{3}$)",labelpad=1)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xticks([3,4,5,6,7]); ax.set_yticks([3,4,5,6,7])
fig.savefig(OUT+"FD1_identity.pdf"); fig.savefig("FD1_preview.png",dpi=180); plt.close(fig)
print("wrote FD1_identity.pdf")

# ---------- FE1: human study (1x2) ----------
sc=np.load("appE_scatter.npy"); R=json.load(open("appE_robust.json"))
fig,axes=plt.subplots(1,2,figsize=(6.0,2.15),
    gridspec_kw=dict(wspace=0.38,width_ratios=[1.05,1],left=0.085,right=0.985,top=0.88,bottom=0.20))
ax=axes[0]
ax.scatter(sc[0],sc[1],s=7,color=CH,alpha=.55,edgecolor="none")
b=np.polyfit(sc[0],sc[1],1); xs=np.linspace(sc[0].min(),sc[0].max(),10)
ax.plot(xs,b[1]+b[0]*xs,color="#111111",lw=0.9)
ax.axhline(0,color="#E2DFD8",lw=0.7); ax.axvline(0,color="#E2DFD8",lw=0.7)
ax.text(0.03,0.94,f"$r = 0.19$; slope $= {b[0]:.2f}$",transform=ax.transAxes,fontsize=6.6,va="top")
ax.set_xlabel("consensus map (deviation from null)",labelpad=1)
ax.set_ylabel("human map",labelpad=1)
ptitle(ax,"(a) Near-orthogonal maps")
ax=axes[1]
labs=[r["label"] for r in R]; t10=[r["top10"] for r in R]; nul=[r["null"] for r in R]
xs=np.arange(3)
ax.bar(xs,t10,0.5,color=CH,alpha=0.85)
ax.scatter(xs,nul,marker="_",s=460,color="#3A3834",lw=1.4,zorder=5,label="matched null")
for i,r_ in enumerate(R):
    ax.text(i,t10[i]+1.1,f"$p={r_['p']:.2f}$",ha="center",fontsize=6.2)
ax.set_xticks(xs); ax.set_xticklabels(["all 8\nexperts","principal\nonly","without\nprincipal"],fontsize=6.2)
ax.set_ylabel("top-10% share (%)",labelpad=1); ax.set_ylim(0,40)
ax.legend(frameon=False,loc="upper left",borderaxespad=0.15)
ptitle(ax,"(b) Robust to the principal annotator")
fig.savefig(OUT+"FE1_human.pdf"); fig.savefig("FE1_preview.png",dpi=180); plt.close(fig)
print("wrote FE1_human.pdf")

# ---------- FF1: text model ----------
pr=np.load("appF_preds.npy")
fig,ax=plt.subplots(figsize=(3.05,2.55),gridspec_kw=dict(left=0.15,right=0.97,top=0.97,bottom=0.155))
ax.scatter(pr[1],pr[0],s=8,color=CO,alpha=.6,edgecolor="none")
b=np.polyfit(pr[1],pr[0],1); xs=np.linspace(pr[1].min(),pr[1].max(),10)
ax.plot(xs,b[1]+b[0]*xs,color="#111111",lw=0.9)
ax.axhline(0,color="#E2DFD8",lw=0.7); ax.axvline(0,color="#E2DFD8",lw=0.7)
r=np.corrcoef(pr[0],pr[1])[0,1]
ax.text(0.04,0.94,f"out-of-sample $r = {r:.2f}$",transform=ax.transAxes,fontsize=6.8,va="top")
ax.set_xlabel("consensus deviation (actual)",labelpad=1)
ax.set_ylabel("predicted from title + abstract",labelpad=1)
fig.savefig(OUT+"FF1_textmodel.pdf"); fig.savefig("FF1_preview.png",dpi=180); plt.close(fig)
print("wrote FF1_textmodel.pdf")
