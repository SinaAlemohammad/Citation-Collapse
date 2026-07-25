#!/usr/bin/env python3
"""
Turn the shipped experiment data into the paper's stats tables and figures.
Reads only local files -- NO API calls, NO keys needed.

Two independent analyses:

1. Paraphrase QC  (data/seed.jsonl  vs  paraphrased_papers/seed0_<x>.jsonl)
   For every paraphrasing model, did the rewrite (a) destroy surface form while
   (b) preserving meaning? A good paraphrase must do both; they pull against each
   other. Metrics per model:
     overlap        mean 5-gram overlap with the original abstract   (want <= 0.05)
     coreness_r     corr. of the App.F coreness index before/after   (want  > 0.95)
     len_change     mean relative abstract-length change             (want within +-15%)
     diag           % of records whose paraphrase matches its OWN original best
                    (a 1:1 alignment check; <100% means content drifted into a
                    different paper -- a quality failure, not a mapping bug)
   -> results/paraphrase_qc.csv

2. Node-0 selection summary  (selections/<run_model>/<run_model>-all-runs.csv)
   For the model that RAN node 0 against each paraphrased seed, how did it cite?
     n_selected     mean papers cited AND shown (cap = 10)
     halluc_rate    cited-not-shown / (in-list + cited-not-shown)
     n_fabricated   citations resolving to NO real paper
   -> results/node0_selection_summary.csv

Figures (results/figures/): fig1_paraphrase_qc.{pdf,png}, fig2_node0_selection.{pdf,png}

Usage:
    python analyze.py                       # run-model gpt-5-mini
    python analyze.py --run-model gpt-5-mini
"""

import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")                 # headless: write files, never open a window
import matplotlib.pyplot as plt

from citation_collapse.core import config
from citation_collapse.paraphrase.paraphrase_seed import coreness, _ngrams   # single source of truth for the metrics

# QC thresholds (documented in the README); used for pass/fail colouring only.
OVERLAP_MAX = 0.05
CORENESS_MIN = 0.95
LEN_TOL = 0.15


def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den else float("nan")


def _jaccard_words(a, b):
    return len(a & b) / (len(a | b) or 1)


# --------------------------------------------------------------------------- #
# 1. paraphrase QC
# --------------------------------------------------------------------------- #
def paraphrase_qc():
    canon = [json.loads(l) for l in open(config.SEED_STANDARDIZED)]
    canon_ids = [r["id"] for r in canon]
    canon_by_id = {r["id"]: r for r in canon}
    canon_wsets = [set(_flat_words(r["abstract"])) for r in canon]

    rows = []
    for slug, path in config.paraphrased_seeds():
        para = {r["id"]: r for r in (json.loads(l) for l in open(path))}
        overlaps, co, cp, dl = [], [], [], []
        for pid in canon_ids:
            o, n = canon_by_id[pid], para[pid]
            og, pg = _ngrams(o["abstract"]), _ngrams(n["abstract"])
            overlaps.append(len(og & pg) / len(og) if og else 0.0)
            co.append(coreness(o))
            cp.append(coreness(n))
            lo, lp = len(o["abstract"].split()), len(n["abstract"].split())
            dl.append((lp - lo) / lo if lo else 0.0)

        # diagonal / 1:1 alignment: does paraphrase i resemble original i best?
        diag_hits = 0
        for i, pid in enumerate(canon_ids):
            pw = set(_flat_words(para[pid]["abstract"]))
            sims = [_jaccard_words(pw, cw) for cw in canon_wsets]
            if max(range(len(sims)), key=lambda j: sims[j]) == i:
                diag_hits += 1

        overlap = sum(overlaps) / len(overlaps)
        r = _pearson(co, cp)
        len_change = sum(dl) / len(dl)
        diag = diag_hits / len(canon_ids)
        passes = (overlap <= OVERLAP_MAX and r > CORENESS_MIN
                  and abs(len_change) <= LEN_TOL and diag == 1.0)
        rows.append({
            "paraphraser": slug,
            "n": len(canon_ids),
            "overlap": round(overlap, 4),
            "coreness_r": round(r, 4),
            "len_change": round(len_change, 4),
            "diag": round(diag, 4),
            "pass": int(passes),
        })
    rows.sort(key=lambda r: (-r["pass"], r["overlap"]))
    return rows


_WORD_STOP = set("""a an the of and or to in for with on by is are was were be been being as at from that this
these those it its we our their they which who what when where how than then thus so such can could may might
will would shall should must not no nor but if while during via using use used based""".split())


def _flat_words(t):
    import re
    return [w for w in re.findall(r"[a-z]{3,}", (t or "").lower()) if w not in _WORD_STOP]


# --------------------------------------------------------------------------- #
# 2. node-0 selection summary
# --------------------------------------------------------------------------- #
def selection_summary(run_model):
    pooled = config.SELECTIONS_PARAPHRASED_DIR / run_model / f"{run_model}-all-runs.csv"
    if not pooled.exists():
        raise SystemExit(
            f"{pooled} not found. Build it first:\n"
            f"  python extract_node_selections.py --model {run_model} --paraphrased all")
    by_p = {}
    for r in csv.DictReader(open(pooled)):
        by_p.setdefault(r["paraphraser"], []).append(r)

    rows = []
    for slug, recs in sorted(by_p.items()):
        n = len(recs)
        mean_sel = sum(int(r["n_selected"]) for r in recs) / n
        tot_in = sum(int(r["n_selected"]) for r in recs)
        tot_out = sum(int(r["n_cited_not_shown"]) for r in recs)
        fab = sum(int(r["n_fabricated"]) for r in recs)
        rows.append({
            "paraphraser": slug,
            "n_prompts": n,
            "mean_n_selected": round(mean_sel, 3),
            "halluc_rate": round(tot_out / (tot_in + tot_out), 4) if (tot_in + tot_out) else 0.0,
            "n_fabricated": fab,
        })
    return rows


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path.relative_to(config.ROOT)}  ({len(rows)} rows)")


def _print_table(title, rows):
    print(f"\n{title}")
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  " + "  ".join(c.rjust(widths[c]) for c in cols))
    for r in rows:
        print("  " + "  ".join(str(r[c]).rjust(widths[c]) for c in cols))


def fig_qc(qc_rows, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in qc_rows:
        good = r["pass"] == 1
        ax.scatter(r["overlap"], r["coreness_r"],
                   s=90, c="#2a7", edgecolor="#145" if good else "#911",
                   marker="o" if good else "X", zorder=3,
                   linewidths=1.5, alpha=0.9)
        ax.annotate(r["paraphraser"], (r["overlap"], r["coreness_r"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axvline(OVERLAP_MAX, color="#911", ls="--", lw=1, label=f"overlap = {OVERLAP_MAX}")
    ax.axhline(CORENESS_MIN, color="#911", ls=":", lw=1, label=f"coreness r = {CORENESS_MIN}")
    ax.set_xlabel("mean 5-gram overlap with original  (lower = surface form destroyed)")
    ax.set_ylabel("coreness r before/after  (higher = meaning preserved)")
    ax.set_title("Paraphrase QC: the good region is bottom-right\n(low overlap AND high coreness r)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=150)
    plt.close(fig)
    print(f"  wrote {out.with_suffix('.pdf').relative_to(config.ROOT)} (+.png)")


def fig_selection(sel_rows, run_model, out):
    labels = [r["paraphraser"] for r in sel_rows]
    sel = [r["mean_n_selected"] for r in sel_rows]
    hal = [r["halluc_rate"] for r in sel_rows]
    x = range(len(labels))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    a1.bar(x, sel, color="#3a7", zorder=3)
    a1.axhline(config.CITATION_CAP, color="#911", ls="--", lw=1,
               label=f"citation cap = {config.CITATION_CAP}")
    a1.set_ylabel("mean papers selected")
    a1.set_ylim(0, config.CITATION_CAP + 0.5)
    a1.set_title(f"{run_model} node 0: selection vs paraphrased seed")
    a1.legend(fontsize=8)
    a1.grid(axis="y", alpha=0.25)
    a2.bar(x, hal, color="#a55", zorder=3)
    a2.set_ylabel("hallucination rate")
    a2.grid(axis="y", alpha=0.25)
    a2.set_xticks(list(x))
    a2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=150)
    plt.close(fig)
    print(f"  wrote {out.with_suffix('.pdf').relative_to(config.ROOT)} (+.png)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-model", default="gpt-5-mini",
                    help="model that RAN node 0 (default gpt-5-mini)")
    args = ap.parse_args()

    res = config.RESULTS_DIR
    figs = res / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    print("[1/2] paraphrase QC")
    qc = paraphrase_qc()
    _write_csv(res / "paraphrase_qc.csv", qc)
    _print_table("paraphrase QC (sorted: passing first, then by overlap)", qc)
    fig_qc(qc, figs / "fig1_paraphrase_qc")

    print(f"\n[2/2] node-0 selection summary  ({args.run_model})")
    sel = selection_summary(args.run_model)
    _write_csv(res / "node0_selection_summary.csv", sel)
    _print_table(f"{args.run_model} node-0 selection", sel)
    fig_selection(sel, args.run_model, figs / "fig2_node0_selection")

    n_pass = sum(r["pass"] for r in qc)
    print(f"\nDone. {n_pass}/{len(qc)} paraphrasers pass QC. "
          f"Tables + figures in {res.relative_to(config.ROOT)}/")


if __name__ == "__main__":
    main()
