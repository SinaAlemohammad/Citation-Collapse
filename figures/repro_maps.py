"""Maps, reliability, cross-model agreement, plurality floor, vote histogram,
exchangeability tests, and the human analyses (Sections 3.1-3.2, Apps C-E)."""
import numpy as np
import pandas as pd
from scipy import stats
from repro_core import counts, top10_hhi_excl, VENDOR


# ---- per-model maps (deviation from matched null) ---------------------------

def null_rate_per_paper(sub, papers):
    """E[cite | shown] under the null for each paper = mean over its shows of k_p/m."""
    num = {p: 0.0 for p in papers}
    den = {p: 0 for p in papers}
    for _, r in sub.iterrows():
        m = len(r["shown"])
        for p in r["shown"]:
            num[p] += r["k"] / m
            den[p] += 1
    return {p: (num[p] / den[p] if den[p] else np.nan) for p in papers}


def model_map(sub, papers):
    c, n = counts(sub, papers)
    nr = null_rate_per_paper(sub, papers)
    return np.array([c[p] / n[p] - nr[p] for p in papers])


def split_half_reliability(sub, papers, nsplit=60, rng=None):
    rng = rng or np.random.default_rng(1)
    idx = np.arange(len(sub))
    rows = sub.reset_index(drop=True)
    rs = []
    for _ in range(nsplit):
        perm = rng.permutation(idx)
        a, b = rows.iloc[perm[: len(idx) // 2]], rows.iloc[perm[len(idx) // 2:]]
        da, db = model_map(a, papers), model_map(b, papers)
        ok = np.isfinite(da) & np.isfinite(db)
        r = np.corrcoef(da[ok], db[ok])[0, 1]
        rs.append(2 * r / (1 + r))  # Spearman-Brown
    return float(np.mean(rs))


def agreement(df, papers, models):
    maps = {m: model_map(df[df.model == m], papers) for m in models}
    rel = {m: split_half_reliability(df[df.model == m], papers) for m in models}
    K = len(models)
    dis = np.eye(K)
    for i in range(K):
        for j in range(i + 1, K):
            r = np.corrcoef(maps[models[i]], maps[models[j]])[0, 1]
            dis[i, j] = dis[j, i] = r / np.sqrt(rel[models[i]] * rel[models[j]])
    # noise-corrected spectrum: covariance of maps with per-map sampling variance
    # removed from the diagonal (binomial variance of rate estimates, averaged)
    M = np.vstack([maps[m] for m in models])
    C = np.cov(M)
    for i, m in enumerate(models):
        sub = df[df.model == m]
        c, n = counts(sub, papers)
        v = np.mean([c[p] / n[p] * (1 - c[p] / n[p]) / n[p] for p in papers])
        C[i, i] = max(C[i, i] - v, 1e-12)
    w = np.linalg.eigvalsh(C)
    pc1 = float(w[-1] / w.sum())
    # vendor blocks of the disattenuated matrix
    vend = [VENDOR(m) for m in models]
    def blk(a, b):
        vals = [dis[i, j] for i in range(K) for j in range(K)
                if i < j and {vend[i], vend[j]} == ({a, b} if a != b else {a})]
        return float(np.mean(vals))
    blocks = dict(wgpt=blk("OpenAI", "OpenAI"), wgem=blk("Google", "Google"),
                  wcla=blk("Anthropic", "Anthropic"), gg=blk("OpenAI", "Google"),
                  gc=blk("OpenAI", "Anthropic"), emc=blk("Google", "Anthropic"))
    off = [dis[i, j] for i in range(K) for j in range(i + 1, K)]
    blocks["off"] = float(np.mean(off))
    return maps, rel, dis, pc1, blocks


# ---- plurality floor (Corollary 5, computed as excess-overlap shares) --------

def floor_tax(df, papers, models):
    """Excess pairwise bibliography overlap: single models vs the all-model mixture."""
    def excess_overlap(rates_by_prompt):
        # Lemma A.2 in estimator form: mean over prompt pairs of sum_j r_pj r_qj
        # minus the same with null rates; equivalently use deviations directly.
        D = rates_by_prompt
        G = D @ D.T
        n = G.shape[0]
        return float((G.sum() - np.trace(G)) / (n * (n - 1)))

    def prompt_dev_matrix(sub):
        prompts = sorted(sub.prompt_id.unique())
        pidx = {p: i for i, p in enumerate(papers)}
        D = np.zeros((len(prompts), len(papers)))
        for i, pid in enumerate(prompts):
            r = sub[sub.prompt_id == pid].iloc[0]
            m = len(r["shown"])
            for p in r["shown"]:
                D[i, pidx[p]] = -r["k"] / m
            for p in r["cited"]:
                D[i, pidx[p]] += 1.0
        return D

    singles = []
    stacks = []
    for m in models:
        D = prompt_dev_matrix(df[df.model == m])
        singles.append(excess_overlap(D))
        stacks.append(D)
    mix = excess_overlap(np.vstack(stacks))
    avg = float(np.mean(singles))
    return dict(avg_single=avg, mix_all=mix, floor=mix / avg, tax=1 - mix / avg)


# ---- vote histogram (App C.3) ------------------------------------------------

def vote_histogram(df, papers, models):
    """Observed votes per (prompt, shown paper) instance vs the shared-map
    Poisson-binomial prediction and the matched-marginal binomial baseline."""
    rates = {}
    for m in models:
        sub = df[df.model == m]
        c, n = counts(sub, papers)
        rates[m] = {p: c[p] / n[p] for p in papers}
    base = df[df.model == models[0]]
    obs, pred, K = [], np.zeros(len(models) + 1), len(models)
    for _, r in base.iterrows():
        for p in r["shown"]:
            v = sum(p in set(df[(df.model == m) & (df.prompt_id == r.prompt_id)
                                ].iloc[0]["cited"]) for m in models)
            obs.append(v)
            # Poisson-binomial over the models' own paper rates
            probs = np.array([rates[m][p] for m in models])
            pmf = np.array([1.0])
            for q in probs:
                pmf = np.convolve(pmf, [1 - q, q])
            pred += pmf
    obs = np.bincount(obs, minlength=K + 1).astype(float)
    obs /= obs.sum()
    pred /= pred.sum()
    pbar = float(np.dot(np.arange(K + 1), obs) / K)
    binom = stats.binom.pmf(np.arange(K + 1), K, pbar)
    tv_shared = 0.5 * np.abs(obs - pred).sum()
    tv_indep = 0.5 * np.abs(obs - binom).sum()
    return obs, pred, binom, float(tv_shared), float(tv_indep), pbar


# ---- exchangeability (Prop 3) -------------------------------------------------

def exch_verdicts(df, papers, models):
    """Per model: exchangeability upper bound on its exclusion count."""
    out = {}
    for m in models:
        sub = df[df.model == m]
        c, n = counts(sub, papers)
        R = sum(c.values()) / sum(n.values())     # group citation rate when shown
        # P(one paper never cited in n_j shows) = (1-R)^{n_j}; exclusions ~ PoisBin
        ps = np.array([(1 - R) ** n[p] for p in papers])
        pmf = np.array([1.0])
        for q in ps:
            pmf = np.convolve(pmf, [1 - q, q])
        e = sum(1 for p in papers if c[p] == 0)
        out[m] = dict(R=float(R), nexcl=e, p_tail=float(pmf[e:].sum()))
    return out


# ---- humans (53-prompt slice; App E) ------------------------------------------

def human_block(df, hdf, papers, models, rng=None):
    rng = rng or np.random.default_rng(2)
    prompts = sorted(hdf.prompt_id.unique())
    base = df[df.model == models[0]].set_index("prompt_id")

    def population(sub_h, label):
        c = {p: 0 for p in papers}
        n = {p: 0 for p in papers}
        panels = []
        for pid, g in sub_h.groupby("prompt_id"):
            shown = base.loc[pid, "shown"]
            cited = set(g[g.cited == 1]["seed_id"].astype(str))
            k = len(cited)
            panels.append((shown, k))
            for p in shown:
                n[p] += 1
            for p in cited:
                c[p] += 1
        t, h, e = top10_hhi_excl(c, n)
        # matched null (Monte Carlo over the same panels/budgets)
        outs = []
        for _ in range(NRED_H):
            cc = {p: 0 for p in papers}
            for shown, k in panels:
                if k:
                    for p in np.random.default_rng(rng.integers(1 << 30)
                                                   ).choice(shown, size=min(k, len(shown)),
                                                            replace=False):
                        cc[p] += 1
            outs.append(top10_hhi_excl(cc, n))
        a = np.array(outs)
        # chi-square exchangeability: observed vs expected citations per shown paper
        exp = {p: 0.0 for p in papers}
        var = {p: 0.0 for p in papers}
        for shown, k in panels:
            q = k / len(shown)
            for p in shown:
                exp[p] += q
                var[p] += q * (1 - q)
        shownp = [p for p in papers if n[p] > 0]
        chi2 = sum((c[p] - exp[p]) ** 2 / var[p] for p in shownp if var[p] > 0)
        dfree = len(shownp) - 1
        pval = 1 - stats.chi2.cdf(chi2, dfree)
        return dict(label=label, n=len(panels), top10=t * 100,
                    null=float(a[:, 0].mean()) * 100, excl=e,
                    chi2=float(chi2), df=dfree, p=float(pval))
    return population, prompts


NRED_H = 150
