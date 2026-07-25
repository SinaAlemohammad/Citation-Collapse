"""Prompt-alignment (rho, stable rank), the HHI identity check (App D),
the vote histogram (App C.3), and the text model (App F)."""
import numpy as np
from scipy import stats
from repro_core import counts


def prompt_dev(sub, papers):
    """Per-prompt deviation rows: cited indicator minus k_p/m on shown papers,
    plus the per-prompt null-noise trace (sum of binomial variances)."""
    pidx = {p: i for i, p in enumerate(papers)}
    rows = sub.reset_index(drop=True)
    D = np.zeros((len(rows), len(papers)))
    noise = np.zeros(len(rows))
    T = 0
    for i, r in rows.iterrows():
        m = len(r["shown"])
        q = r["k"] / m
        for p in r["shown"]:
            D[i, pidx[p]] = -q
        for p in r["cited"]:
            D[i, pidx[p]] += 1.0
        noise[i] = m * q * (1 - q)
        T += r["k"]
    return D, noise, T


def rho_rs(sub, papers):
    """Noise-corrected alignment ratio and stable rank (Prop 4 estimators).
    The diagonal of the prompt Gram matrix is inflated by selection noise;
    the plug-in correction uses the model's own paper-level rates (App D)."""
    D, _, _ = prompt_dev(sub, papers)
    c, n = counts(sub, papers)
    rate = {p: c[p] / n[p] for p in papers}
    rows = sub.reset_index(drop=True)
    noise = np.array([sum(rate[p] * (1 - rate[p]) for p in r["shown"])
                      for _, r in rows.iterrows()])
    G = D @ D.T
    P = G.shape[0]
    N = len(papers)
    s = len(rows.iloc[0]["shown"]) / N          # panel share m/N
    off = (G.sum() - np.trace(G)) / (P * (P - 1))
    sig = (np.diag(G) - noise).mean()
    rho = float(off / (s * sig))                 # overlap-corrected alignment
    Gc = G / s                                   # put off-diagonals on the
    np.fill_diagonal(Gc, np.diag(G) - noise)     # full-catalogue scale
    lam = np.linalg.eigvalsh(Gc)[-1]
    rs = float(np.trace(Gc) / lam)
    return rho, rs


def identity_check(sub, papers, null_hhi):
    """Lemma A.1: observed excess HHI vs its reconstruction from the
    cross-prompt Gram matrix (off-diagonal sum over squared total citations)."""
    from repro_core import top10_hhi_excl
    c, n = counts(sub, papers)
    _, hhi, _ = top10_hhi_excl(c, n)
    D, _, T = prompt_dev(sub, papers)
    G = D @ D.T
    rec = float((G.sum() - np.trace(G)) / T ** 2)
    return dict(obs=float(hhi - null_hhi), rec=rec)


def vote_histogram(df, papers, models):
    """Observed votes per (prompt, shown paper) vs shared-map Poisson-binomial
    prediction and matched-marginal binomial baseline (TV distances)."""
    rates = {}
    citedsets = {}
    for m in models:
        sub = df[df.model == m]
        c, n = counts(sub, papers)
        rates[m] = {p: c[p] / n[p] for p in papers}
        citedsets[m] = {r.prompt_id: set(r["cited"]) for _, r in sub.iterrows()}
    base = df[df.model == models[0]]
    K = len(models)
    votes, pred = [], np.zeros(K + 1)
    for _, r in base.iterrows():
        for p in r["shown"]:
            votes.append(sum(p in citedsets[m][r.prompt_id] for m in models))
            pmf = np.array([1.0])
            for m in models:
                q = rates[m][p]
                pmf = np.convolve(pmf, [1 - q, q])
            pred += pmf
    obs = np.bincount(votes, minlength=K + 1).astype(float)
    obs /= obs.sum()
    pred /= pred.sum()
    pbar = float(np.dot(np.arange(K + 1), obs) / K)
    binom = stats.binom.pmf(np.arange(K + 1), K, pbar)
    return dict(votes=[int(v) for v in votes],
                obs=obs.tolist(), pred=pred.tolist(), binom=binom.tolist(),
                tv_shared=float(0.5 * np.abs(obs - pred).sum()),
                tv_indep=float(0.5 * np.abs(obs - binom).sum()), p0=pbar)


def text_model(consensus_map, papers, seeds, ngram=(1, 2), min_df=3, alpha=1.0,
               folds=10, seed=0):
    """Out-of-sample prediction of the consensus deviation from title+abstract
    (TF-IDF + ridge, K-fold CV). Returns r and per-paper predictions."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    txt = {row.seed: f"{row.title} {row.abstract}" for row in seeds.itertuples()}
    X_txt = [txt[p] for p in papers]
    y = np.asarray(consensus_map)
    preds = np.zeros_like(y)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(X_txt):
        vec = TfidfVectorizer(ngram_range=ngram, min_df=min_df,
                              stop_words="english", sublinear_tf=True)
        Xtr = vec.fit_transform([X_txt[i] for i in tr])
        Xte = vec.transform([X_txt[i] for i in te])
        preds[te] = Ridge(alpha=alpha).fit(Xtr, y[tr]).predict(Xte)
    r = float(np.corrcoef(preds, y)[0, 1])
    return r, preds
