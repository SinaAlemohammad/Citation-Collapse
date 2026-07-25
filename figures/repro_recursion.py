"""Recursion metrics (Section 4, App I), the within-panel competition test
(Prop 6 / App G), and the placebo, all from all-models-all-nodes-selections.csv."""
import numpy as np
import pandas as pd

NRED_R = 60  # redraws per round for the recursion null (MC tolerance ~0.2 pp)


def is_seed(p):
    return p.startswith("SEED")


def round_tables(df):
    """{model: {node: sub}} with shown/cited lists preparsed."""
    return {m: {t: g for t, g in sub.groupby("node")}
            for m, sub in df.groupby("model")}


def _conc_metrics(cnt, shown_cnt, catalogue):
    v = np.array(sorted((cnt.get(p, 0) for p in catalogue), reverse=True), float)
    tot = v.sum()
    if tot == 0:
        return dict(top10=np.nan, hhi=np.nan, excl=np.nan, gini=np.nan, tail1=np.nan)
    ntop = max(1, int(round(0.10 * len(v))))
    n1 = max(1, int(round(0.01 * len(v))))
    share = v / tot
    # Gini over shown papers' citation counts
    shown = np.array(sorted(cnt.get(p, 0) for p in catalogue if shown_cnt.get(p, 0) > 0),
                     float)
    g = (2 * np.sum((np.arange(1, len(shown) + 1)) * shown) /
         (len(shown) * shown.sum()) - (len(shown) + 1) / len(shown)) if shown.sum() else np.nan
    nshown = sum(1 for p in catalogue if shown_cnt.get(p, 0) > 0)
    excl = sum(1 for p in catalogue if shown_cnt.get(p, 0) > 0 and cnt.get(p, 0) == 0)
    return dict(top10=float(v[:ntop].sum() / tot), hhi=float((share ** 2).sum()),
                excl=float(excl / nshown) if nshown else np.nan,
                gini=float(g), tail1=float(v[:n1].sum() / tot))


def _round_counts(sub, catalogue):
    cnt, shn = {}, {}
    for _, r in sub.iterrows():
        for p in r["shown"]:
            shn[p] = shn.get(p, 0) + 1
        for p in r["cited"]:
            cnt[p] = cnt.get(p, 0) + 1
    return cnt, shn


def _null_round(sub, catalogue, rng, subset=None):
    """One MC average of concentration metrics + seed rate under uniform picks."""
    outs, seed_rates = [], []
    panels = [(r["shown"], r["k"]) for _, r in sub.iterrows()]
    for _ in range(NRED_R):
        cnt, shn = {}, {}
        sshow = scit = 0
        for shown, k in panels:
            for p in shown:
                shn[p] = shn.get(p, 0) + 1
                if is_seed(p):
                    sshow += 1
            pick = rng.choice(shown, size=min(k, len(shown)), replace=False)
            for p in pick:
                cnt[p] = cnt.get(p, 0) + 1
                if is_seed(p):
                    scit += 1
        cat = subset if subset is not None else catalogue
        if subset is not None:
            cnt = {p: c for p, c in cnt.items() if p in subset}
            shn = {p: c for p, c in shn.items() if p in subset}
        outs.append(_conc_metrics(cnt, shn, cat))
        seed_rates.append(scit / sshow if sshow else np.nan)
    keys = outs[0].keys()
    agg = {k: float(np.nanmean([o[k] for o in outs])) for k in keys}
    agg["seed_cr"] = float(np.nanmean(seed_rates))
    return agg


def recursion_metrics(df):
    out = {}
    for m, bynode in round_tables(df).items():
        rng = np.random.default_rng(7)
        rec = {k: [] for k in
               ("node Ncat top10 top10_null excl excl_null hhi hhi_null "
                "gini_all gini_gen top10_gen gen_above_null seed_cr seed_cr_null "
                "seed_excess gen_cr seed_attn seed_cat newent reach_seed churn "
                "tail1 k hall overlap overlap_null eff_support eff_frac").split()}
        prev_top, seen_seeds = None, set()
        for t in sorted(bynode):
            sub = bynode[t]
            catalogue = sorted({p for s in sub["shown"] for p in s})
            gen = [p for p in catalogue if not is_seed(p)]
            cnt, shn = _round_counts(sub, catalogue)
            met = _conc_metrics(cnt, shn, catalogue)
            null = _null_round(sub, catalogue, rng)
            # generated-only ledger
            if gen:
                gcnt = {p: cnt.get(p, 0) for p in gen}
                gshn = {p: shn.get(p, 0) for p in gen}
                gmet = _conc_metrics(gcnt, gshn, gen)
                gnull = _null_round(sub, catalogue, rng, subset=set(gen))
            else:
                gmet = dict(top10=np.nan, gini=np.nan)
                gnull = dict(top10=np.nan)
            # seed / generated citation rates when shown
            def rate(group):
                sh = sum(shn.get(p, 0) for p in group)
                ci = sum(cnt.get(p, 0) for p in group)
                return ci / sh if sh else np.nan
            seeds = [p for p in catalogue if is_seed(p)]
            scr, gcr = rate(seeds), rate(gen) if gen else np.nan
            # newest cohort openness (papers introduced in the previous round)
            newest = [p for p in catalogue if p.startswith(f"N{t-1}P")]
            tot_c = sum(cnt.values())
            newent = (sum(cnt.get(p, 0) for p in newest) / tot_c) if (newest and tot_c) else np.nan
            # top-decile churn vs previous round
            cited_pool = [p for p in catalogue if cnt.get(p, 0) > 0]
            v = sorted(cited_pool, key=lambda p: -cnt.get(p, 0))
            top = set(v[: max(1, int(round(0.10 * len(v))))])
            churn = (len(top & prev_top) / len(top)) if prev_top else np.nan
            prev_top = top
            seen_seeds |= {p for p in seeds if cnt.get(p, 0) > 0}
            # mean pairwise bibliography overlap (observed and null)
            bibs = [set(r["cited"]) for _, r in sub.iterrows()]
            def mean_ov(bs):
                s = c = 0
                for i in range(len(bs)):
                    for j in range(i + 1, min(i + 40, len(bs))):  # subsampled pairs
                        s += len(bs[i] & bs[j]); c += 1
                return s / c
            ov = mean_ov(bibs)
            rng2 = np.random.default_rng(11 + t)
            nb = [set(rng2.choice(r["shown"], size=min(r["k"], len(r["shown"])),
                                  replace=False)) for _, r in sub.iterrows()]
            ov0 = mean_ov(nb)
            cited_any = sum(1 for p in catalogue if cnt.get(p, 0) > 0)
            row = dict(node=int(t), Ncat=len(catalogue),
                       top10=met["top10"], top10_null=null["top10"],
                       excl=met["excl"], excl_null=null["excl"],
                       hhi=met["hhi"], hhi_null=null["hhi"],
                       gini_all=met["gini"], gini_gen=gmet["gini"],
                       top10_gen=gmet["top10"],
                       gen_above_null=(gmet["top10"] - gnull["top10"])
                       if gen else np.nan,
                       seed_cr=scr, seed_cr_null=null["seed_cr"],
                       seed_excess=scr - null["seed_cr"], gen_cr=gcr,
                       seed_attn=sum(cnt.get(p, 0) for p in seeds) / tot_c,
                       seed_cat=len(seeds) / len(catalogue),
                       newent=newent, reach_seed=len(seen_seeds) / len(seeds),
                       churn=churn, tail1=met["tail1"],
                       k=float(sub["k"].mean()),
                       hall=float(sub["hallucination_rate"].mean()),
                       overlap=ov, overlap_null=ov0,
                       eff_support=cited_any,
                       eff_frac=cited_any / len(catalogue))
            for k, v_ in row.items():
                rec[k].append(v_ if not (isinstance(v_, float) and np.isnan(v_)) else None)
        out[m] = rec
    return out


# ---- within-panel competition test (Prop 6 / App G) --------------------------

def _twoway_fe_cluster(y, x, paper, roundv, cluster, iters=60):
    """Slope of y on x with paper + round fixed effects (alternating
    demeaning), cluster-robust SE."""
    y = y.astype(float).copy(); x = x.astype(float).copy()
    for _ in range(iters):
        for f in (paper, roundv):
            df_ = pd.DataFrame(dict(y=y, x=x, f=f))
            g = df_.groupby("f").transform("mean")
            y = (df_["y"] - g["y"]).to_numpy()
            x = (df_["x"] - g["x"]).to_numpy()
    beta = float(np.dot(x, y) / np.dot(x, x))
    resid = y - beta * x
    # cluster-robust (by prompt) sandwich
    dfc = pd.DataFrame(dict(x=x, u=resid, c=cluster))
    s = dfc.assign(xu=dfc.x * dfc.u).groupby("c")["xu"].sum()
    se = float(np.sqrt((s ** 2).sum()) / np.dot(x, x))
    return beta, se


def build_rows(df, group="seed"):
    """(paper, prompt) rows for rounds 1-11: cited indicator and number of
    co-shown competitors of the paper's own group."""
    rows = []
    for _, r in df[df.node >= 1].iterrows():
        shown = r["shown"]
        nseed = sum(is_seed(p) for p in shown)
        cited = set(r["cited"])
        for p in shown:
            if (is_seed(p) if group == "seed" else not is_seed(p)):
                # seed rows: within-group competition (other seeds shown);
                # generated rows: cross-competition with seeds (App G convention)
                comp = (nseed - 1) if group == "seed" else nseed
                rows.append((p, r.prompt_id, int(r.node), comp, int(p in cited)))
    return pd.DataFrame(rows, columns=["paper", "prompt", "round", "comp", "cited"])


def competition_test(df, group="seed"):
    out = {}
    for m, sub in df.groupby("model"):
        R = build_rows(sub, group)
        beta, se = _twoway_fe_cluster(R.cited.to_numpy(), R.comp.to_numpy(),
                                      R.paper.to_numpy(), R["round"].to_numpy(),
                                      R.prompt.to_numpy())
        out[m] = dict(beta=beta * 100, se=se * 100, t=beta / se,
                      rows=len(R), clusters=R.prompt.nunique())
    return out


def placebo_test(df, ndraw=40, seed=3):
    """Identical regression with the outcome replaced by the average of
    `ndraw` uniform redraws of each bibliography."""
    rng = np.random.default_rng(seed)
    out = {}
    for m, sub in df.groupby("model"):
        R = build_rows(sub, "seed")
        # average null outcome per (paper, prompt): P(cited) estimated over draws
        kbyp = sub.set_index("prompt_id")["k"].to_dict()
        shbyp = sub.set_index("prompt_id")["shown"].to_dict()
        ymc = np.zeros(len(R))
        for d in range(ndraw):
            picks = {pid: set(rng.choice(sh, size=min(kbyp[pid], len(sh)),
                                         replace=False))
                     for pid, sh in shbyp.items()}
            ymc += np.array([p in picks[pid] for p, pid in zip(R.paper, R.prompt)])
        ymc /= ndraw
        beta, se = _twoway_fe_cluster(ymc, R.comp.to_numpy(), R.paper.to_numpy(),
                                      R["round"].to_numpy(), R.prompt.to_numpy())
        out[m] = dict(beta=beta * 100, se=se * 100)
    return out
