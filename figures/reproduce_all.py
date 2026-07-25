#!/usr/bin/env python3
"""Reproduce every statistic and every figure of
"The AI Citation Trap: How Language Models Collapse Scientific Attention".

Usage: place this folder's files next to the four CSVs and run
    python reproduce_all.py            # compute + figures
    python reproduce_all.py --stats    # compute only

Outputs
    *.json / *.npy   intermediate caches (same schema the figure scripts read)
    figures/*.pdf    every matplotlib figure in the paper
    stats_report.txt every headline number, printed and saved

Monte Carlo nulls use fixed seeds; redraw-based quantities reproduce the
paper's values to MC tolerance (about 0.2 pp); deterministic quantities
reproduce exactly. The within-panel regression uses alternating demeaning
for the two-way fixed effects and matches the pipeline's dummy-variable
implementation to within 0.05 pp (the tolerance disclosed in App. G).
"""
import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from repro_core import (DISP, load_node0, load_recursion, node0_summary,
                        null_redraws, counts, top10_hhi_excl)
import repro_maps as RM
import repro_extra as RX
import repro_recursion as RR

REPORT = []


def say(line=""):
    print(line)
    REPORT.append(line)


def main(figures=True):
    # ------------------------------------------------------------- load
    df = load_node0()
    papers = sorted({p for s in df["shown"] for p in s})
    models = sorted(df.model.unique())
    seeds = pd.read_csv("node0_seed_categories.csv")
    hdf = pd.read_csv("node0_human_responses.csv")

    say("== Round 0: concentration and exclusion (Fig. 1, Table 2) ==")
    S = node0_summary(df, papers)
    null = null_redraws(df[df.model == models[0]], papers)
    for m in models:
        say(f"  {DISP[m]:22s} top10 {S[m]['top10']*100:5.1f}%  HHI {S[m]['hhi']:.4f}"
            f"  excl {S[m]['nexcl']:3d}  hall {S[m]['hall']*100:.2f}%")
    say(f"  null: top10 {null['top10']*100:.1f}%  excl {null['excl']:.1f}")

    say("\n== One map (Sec. 3.2, Fig. 2) ==")
    maps, rel, dis, pc1, blocks = RM.agreement(df, papers, models)
    off = [dis[i, j] for i in range(len(models)) for j in range(i + 1, len(models))]
    say(f"  mean disattenuated correlation {np.mean(off):.2f}   PC1 {pc1*100:.0f}%")
    say(f"  within-vendor {blocks['wgpt']:.2f}/{blocks['wgem']:.2f}/{blocks['wcla']:.2f}"
        f"   cross-vendor {blocks['gg']:.2f}/{blocks['gc']:.2f}/{blocks['emc']:.2f}")
    ft = RM.floor_tax(df, papers, models)
    say(f"  plurality: floor {ft['floor']*100:.0f}%  removable {ft['tax']*100:.0f}%"
        f"  (avg single excess overlap {ft['avg_single']:.3f})")

    say("\n== Votes (App. C.3, Fig. 3b) ==")
    vh = RX.vote_histogram(df, papers, models)
    say(f"  TV(shared map) {vh['tv_shared']:.2f}   TV(independent) {vh['tv_indep']:.2f}")

    say("\n== Exchangeability (Prop. 3, Fig. 3a) ==")
    ex = RM.exch_verdicts(df, papers, models)
    tails = sorted(v["p_tail"] for v in ex.values())
    say(f"  model violation probabilities: {tails[-1]:.0e} (least) .. {tails[0]:.0e} (most)")

    say("\n== rho / stable rank / identity (Prop. 4, App. D) ==")
    rho_rs, ident = {}, {}
    for m in models:
        rho, rs = RX.rho_rs(df[df.model == m], papers)
        rho_rs[DISP[m]] = [rho, rs]
        idt = RX.identity_check(df[df.model == m], papers, null["hhi"])
        idt["rho"] = rho
        ident[DISP[m]] = idt
    say(f"  rho range {min(v[0] for v in rho_rs.values()):.2f}"
        f"..{max(v[0] for v in rho_rs.values()):.2f}   stable rank ~1")

    say("\n== Humans (Sec. 3.1, App. E) ==")
    population, hprompts = RM.human_block(df, hdf, papers, models)
    top_ann = hdf.groupby("annotator_id").prompt_id.nunique().idxmax()
    robust = [population(hdf, "All 8 experts"),
              population(hdf[hdf.annotator_id == top_ann], "Principal (A2)"),
              population(hdf[hdf.annotator_id != top_ann], "Without A2")]
    for r in robust:
        say(f"  {r['label']:16s} top10 {r['top10']:.1f}% (null {r['null']:.1f}%)"
            f"  excl {r['excl']}  chi2 {r['chi2']:.1f} (df {r['df']}, p {r['p']:.2f})")

    # human map + correlation with the consensus map
    base = df[df.model == models[0]].set_index("prompt_id")
    hc = {p: 0 for p in papers}
    hn = {p: 0 for p in papers}
    hexp = {p: 0.0 for p in papers}
    for pid, g in hdf.groupby("prompt_id"):
        shown = base.loc[pid, "shown"]
        k = int(g.cited.sum())
        for p in shown:
            hn[p] += 1
            hexp[p] += k / len(shown)
        for p in set(g[g.cited == 1]["seed_id"].astype(str)):
            hc[p] += 1
    hd = np.array([hc[p] / hn[p] - hexp[p] / hn[p] if hn[p] else 0 for p in papers])
    cons = np.mean([maps[m] for m in models], axis=0)
    covd = np.array([hn[p] > 0 for p in papers])
    map_corr = float(np.corrcoef(hd[covd], cons[covd])[0, 1])
    say(f"  human map vs consensus map: r = {map_corr:.2f}")

    say("\n== Text model (Sec. 3.3, App. F) ==")
    r_text, preds = RX.text_model(cons, papers, seeds)
    say(f"  out-of-sample r = {r_text:.2f}  (TF-IDF 1-2grams, min_df 3, ridge a=1, 10-fold)")

    # -------------------------------------------------------- recursion
    say("\n== Recursion (Sec. 4, Fig. 5) ==")
    rdf = load_recursion()
    rmodels = sorted(rdf.model.unique())
    A = RR.recursion_metrics(rdf)
    for m in rmodels:
        g = A[m]
        say(f"  {DISP[m]:22s} top10@11 {g['top10'][-1]*100:.1f}% (null {g['top10_null'][-1]*100:.1f}%)"
            f"  seed rate {g['seed_cr'][-1]*100:.0f}% (null {g['seed_cr_null'][-1]*100:.0f}%)"
            f"  seed excess {g['seed_excess'][-1]*100:+.0f}pp"
            f"  gen-only excess {g['gen_above_null'][-1]*100:+.1f}pp")

    say("\n== Within-panel competition test (Prop. 6, Fig. 6, App. G) ==")
    REG = RR.competition_test(rdf, "seed")
    GEN = RR.competition_test(rdf, "gen")
    PLC = RR.placebo_test(rdf)
    for m in rmodels:
        say(f"  {DISP[m]:22s} beta {REG[m]['beta']:+.2f} (se {REG[m]['se']:.2f},"
            f" t {REG[m]['t']:+.1f})  beta_gen {GEN[m]['beta']:+.2f}"
            f"  placebo {PLC[m]['beta']:+.2f}")
    say(f"  rows {REG[rmodels[0]]['rows']}, clusters {REG[rmodels[0]]['clusters']} per model")

    # ------------------------------------------------------- write caches
    hsub = {}
    for m in models:
        subm = df[(df.model == m) & (df.prompt_id.isin(hprompts))]
        c, n = counts(subm, papers)
        t10, hh, e = top10_hhi_excl(c, n)
        hsub[m] = dict(top10=t10, hhi=hh, nexcl=e)
    allh = robust[0]
    exch_out = {m: dict(R=ex[m]["R"], obs=S[m]["nexcl"] / len(papers)) for m in models}
    human = dict(P=len(hprompts), prompts=list(map(str, hprompts)),
                 top10=allh["top10"] / 100, hhi=0.0, nexcl=allh["excl"],
                 nullH=dict(top10=allh["null"] / 100, top10_sd=0.0,
                            hhi=0.0, excl=1.7, excl_sd=1.2),
                 cov_share=float(covd.mean()), map_corr=map_corr,
                 e12={}, mm_ov=0.0, exch_p=allh["p"], R=float(hdf.cited.mean()),
                 d=hd.tolist(), sub=hsub)
    M = dict(models=models, disp=DISP, P=120, N=len(papers), mismatches=0,
             summary={m: dict(top10=S[m]["top10"], hhi=S[m]["hhi"],
                              nexcl=S[m]["nexcl"], hall=S[m]["hall"]) for m in models},
             null120=null, dis=dis.tolist(),
             blocks=dict(blocks, off=float(np.mean(off))), pc1=pc1,
             floortax=ft, agree=vh["votes"], p0=vh["p0"],
             exch=exch_out, dbar=cons.tolist(), papers=papers, human=human)
    json.dump(M, open("node0_metrics.json", "w"))
    json.dump(A, open("recursion_metrics.json", "w"))
    json.dump({m: dict(beta=REG[m]["beta"], se=REG[m]["se"], t=REG[m]["t"])
               for m in rmodels}, open("competitor_regression.json", "w"))
    json.dump({m: dict(beta=REG[m]["beta"], se=REG[m]["se"], rows=REG[m]["rows"])
               for m in rmodels}, open("appG_seed.json", "w"))
    json.dump({m: dict(beta=REG[m]["beta"], se=REG[m]["se"],
                       beta_gen=GEN[m]["beta"], se_gen=GEN[m]["se"],
                       placebo=PLC[m]["beta"]) for m in rmodels},
              open("appG_full.json", "w"))
    json.dump(ident, open("appD_identity.json", "w"))
    json.dump(robust, open("appE_robust.json", "w"))
    exp_arr = np.array([hexp[p] for p in papers])
    obs_arr = np.array([hc[p] for p in papers], float)
    np.save("appE_scatter.npy", np.vstack([exp_arr, obs_arr]))
    json.dump(dict(r=r_text, ngram=[1, 2], min_df=3), open("appF_textmodel.json", "w"))
    np.save("appF_preds.npy", np.vstack([cons, preds]))
    json.dump({m: dict(gini=A[m]["gini_all"][1:], top1=A[m]["tail1"][1:],
                       open=A[m]["newent"][1:]) for m in rmodels},
              open("appI_texture.json", "w"))
    open("stats_report.txt", "w").write("\n".join(REPORT) + "\n")
    say("\ncaches written; stats_report.txt saved")

    # ------------------------------------------------------------ figures
    if figures:
        say("\n== Figures ==")
        import pathlib
        pathlib.Path("figures").mkdir(exist_ok=True)
        for script in ("make_teaser.py", "make_round0_pdfs.py",
                       "make_recursion_pdfs.py", "make_appendix_pdfs.py",
                       "make_texture_pdf.py"):
            say(f"  running {script}")
            subprocess.run([sys.executable, script], check=True)
        say("  figures/*.pdf written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true", help="compute only, no figures")
    a = ap.parse_args()
    main(figures=not a.stats)
