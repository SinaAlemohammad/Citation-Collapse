#!/usr/bin/env python3
"""
Extract per-node selection CSVs for any node, all models, with a hallucination
breakdown.

    python extract_node_selections.py --model gpt-5-mini            # node 0
    python extract_node_selections.py --model gpt-5-mini --node 3
    python extract_node_selections.py --all --node all              # every node found
    python extract_node_selections.py --all                         # node 0, every model
    python extract_node_selections.py --model gpt-5-mini --node all --combined
                                                                    # per-node CSVs + one combined CSV

    # paraphrased runs (run_paraphrased/ -> selections_paraphrased/)
    python extract_node_selections.py --model gpt-5-mini --paraphrased claude-sonnet-4-6
    python extract_node_selections.py --model gpt-5-mini --paraphrased all

Per-node output: selections/<model>/<model>-node<node>.csv
Combined output (with --combined): selections/<model>/<model>-all.csv  (all processed nodes)

With --paraphrased, both go under selections_paraphrased/<model>/<paraphraser>/
instead -- same folder shape as run_paraphrased/. They MUST stay separate: the
model name is the same, so writing them to selections/<model>/ would overwrite
the canonical (original-seed) CSVs.

--paraphrased all ALSO writes one pooled CSV across every paraphrased run:
    selections_paraphrased/<model>/<model>-all-runs.csv
Paraphrased rows carry an extra `paraphraser` column (absent from canonical
CSVs); it is what keeps the pooled rows separable.

Columns (same for both):
    model, node, prompt_id, prompt_type, n_shown, papers_shown,
    papers_selected, n_selected,                 # cited AND shown (authoritative, capped)
    cited_not_shown, n_cited_not_shown,          # cited but NOT in the shown list
    n_fabricated,                                # subset of the above that match no real paper
    hallucination_rate                           # n_cited_not_shown / (n_in_list + n_cited_not_shown)

Citation detection uses the comprehensive, format-agnostic matcher against the
real-paper table (robust to every bracket/narrative style); "fabricated" tokens
(author/year that match no real paper at all) are found with the regex parser.
A model-level hallucination rate is printed at the end of each file.
"""

import argparse
import csv
import json

from citation_collapse.core import config
from citation_collapse.core.citation_parser import find_cited_candidates, extract_citations_from_body

SEP = "; "


def _read_kv(path):
    inv, kv = {}, {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                inv[r["id"]] = (r["author"], r["year"])
                kv[(r["author"], int(r["year"]))] = r["id"]
    return inv, kv


def load_kv_for_run(model, node):
    """Pick the kv that actually covers this run's papers_seen_id. Tries the
    canonical root kv and the model's own master kv, and uses whichever resolves
    more of the ids the run references (they must all be resolvable). Warns if
    the best available kv is still incomplete, since unresolved citations would
    otherwise masquerade as hallucinations."""
    need = set()
    for line in open(config.node_file(model, node)):
        need.update(json.loads(line).get("papers_seen_id", []) or [])

    candidates = []
    if config.KV_PAIRS.exists():
        candidates.append(("root kv_pairs.jsonl", config.KV_PAIRS))
    master = config.master_dir(model) / "kv_pairs.jsonl"
    if master.exists():
        candidates.append((f"runs/{model}/master/kv_pairs.jsonl", master))
    if not candidates:
        raise SystemExit(f"No kv_pairs found for {model} (root or master).")

    best = None
    for name, path in candidates:
        inv, kv = _read_kv(path)
        missing = len(need - set(inv))
        if best is None or missing < best[0]:
            best = (missing, name, inv, kv)
        if missing == 0:
            break
    missing, name, inv, kv = best
    if missing:
        print(f"  WARNING: best kv ({name}) is missing {missing}/{len(need)} ids "
              f"referenced by {model} node {node} -- results may be incomplete and "
              f"citations to unresolved papers will look like hallucinations. "
              f"Regenerate the canonical kv (run_nodes ... --write-root-kv).")
    return inv, kv


def selections_for(model, node, cap, paraphraser=None):
    nf = config.node_file(model, node)
    if not nf.exists():
        return None
    inv, kv = load_kv_for_run(model, node)
    all_cands = [(a, y, pid) for pid, (a, y) in inv.items()]

    rows = []
    tot_in = tot_not_shown = 0
    for line in open(nf):
        rec = json.loads(line)
        body = rec.get("body", "") or ""
        shown = rec.get("papers_seen_id", []) or []
        shown_set = set(shown)

        # prefilter candidates actually mentioned in this body, then match robustly
        present = [c for c in all_cands if c[0] in body and str(c[1]) in body]
        all_real_cited = find_cited_candidates(body, present)        # ordered ids, any format
        in_list = [c[2] for c in all_real_cited if c[2] in shown_set]
        out_real = [c[2] for c in all_real_cited if c[2] not in shown_set]

        # fabricated = apparent author/year that resolves to NO real paper
        fabricated = [f"{a}({y})" for (a, y) in extract_citations_from_body(body)
                      if kv.get((a, int(y))) is None]

        selected = in_list[:cap]
        cited_not_shown = [str(i) for i in out_real] + fabricated
        denom = len(in_list) + len(cited_not_shown)
        rate = (len(cited_not_shown) / denom) if denom else 0.0

        tot_in += len(in_list)
        tot_not_shown += len(cited_not_shown)
        rows.append({
            "model": model,
            # which seed this run read; identifies the row once pairs are pooled
            **({"paraphraser": paraphraser} if paraphraser is not None else {}),
            "node": node,
            "prompt_id": rec.get("id"),
            "prompt_type": rec.get("type"),
            "n_shown": len(shown),
            "papers_shown": SEP.join(map(str, shown)),
            "papers_selected": SEP.join(selected),
            "n_selected": len(selected),
            "cited_not_shown": SEP.join(cited_not_shown),
            "n_cited_not_shown": len(cited_not_shown),
            "n_fabricated": len(fabricated),
            "hallucination_rate": round(rate, 4),
        })

    rows.sort(key=lambda r: str(r["prompt_id"]))
    out = config.selection_path(model, node)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["model", "node", "prompt_id", "prompt_type", "n_shown", "papers_shown",
                            "papers_selected", "n_selected", "cited_not_shown",
                            "n_cited_not_shown", "n_fabricated", "hallucination_rate"])
        w.writeheader()
        w.writerows(rows)
    agg = (tot_not_shown / (tot_in + tot_not_shown)) if (tot_in + tot_not_shown) else 0.0
    print(f"  wrote {out}  ({len(rows)} prompts)  model-node hallucination rate = {agg:.4f}")
    return rows


def nodes_present(model):
    ns = []
    n = 0
    while config.node_file(model, n).exists():
        ns.append(n)
        n += 1
    return ns


def process_model(model, args, paraphraser=None):
    """Write per-node CSVs (+ optional combined) for whatever run config currently
    points at -- canonical runs/<model>/ or a retargeted paraphrased pair."""
    nodes = nodes_present(model) if args.node == "all" else [int(args.node)]
    combined = []
    for n in nodes:
        rows = selections_for(model, n, args.cap, paraphraser)
        if rows is None:
            print(f"  [skip] {model} node {n}: no node file")
        else:
            combined.extend(rows)
    if args.combined and combined:
        out = config.selections_dir(model) / f"{model}-all.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(combined[0].keys()))
            w.writeheader()
            w.writerows(combined)
        n_nodes = len({r["node"] for r in combined})
        print(f"  wrote combined {out}  ({len(combined)} rows across {n_nodes} node(s))")
    return combined


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", choices=config.list_models())
    g.add_argument("--all", action="store_true", help="every model under runs/")
    ap.add_argument("--paraphrased", metavar="PARAPHRASER",
                    help="extract a PARAPHRASED run instead of the canonical one: reads "
                         "run_paraphrased/<model>/<PARAPHRASER>/ and writes "
                         "selections_paraphrased/<model>/<PARAPHRASER>/. "
                         "Use '--paraphrased all' for every paraphraser of <model>. "
                         "Requires --model (the model that RAN node 0).")
    ap.add_argument("--node", default="0", help="node index, or 'all' (default 0)")
    ap.add_argument("--cap", type=int, default=config.CITATION_CAP)
    ap.add_argument("--combined", action="store_true",
                    help="also write one CSV per model containing all processed "
                         "nodes, as selections/<model>/<model>-all.csv "
                         "(pair with --node all for every node)")
    args = ap.parse_args()

    # ---- paraphrased mode: run_paraphrased/ -> selections_paraphrased/ -------
    if args.paraphrased:
        if not args.model:
            raise SystemExit("--paraphrased needs --model (the model that RAN node 0), "
                             "not --all.")
        targets = (config.paraphrased_runs(args.model)
                   if args.paraphrased == "all" else [args.paraphrased])
        if not targets:
            raise SystemExit(
                f"No paraphrased runs under {config.RUN_PARAPHRASED_DIR / args.model}.\n"
                f"Run: python batch_node0.py --model {args.model} --all --live")
        print(f"{args.model}: {len(targets)} paraphrased run(s): {', '.join(targets)}")
        every = []
        for t in targets:
            config.reset_run_dir()
            d = config.use_paraphrased_run(args.model, t)
            print(f"\n=== {args.model} x {t}\n    run -> {d}")
            every.extend(process_model(args.model, args, paraphraser=t))
        config.reset_run_dir()

        # one CSV pooling every paraphraser x every node processed. The
        # `paraphraser` column is what makes the pooled rows separable.
        if every:
            out = (config.SELECTIONS_PARAPHRASED_DIR / args.model /
                   f"{args.model}-all-runs.csv")
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(every[0].keys()))
                w.writeheader()
                w.writerows(every)
            n_p = len({r["paraphraser"] for r in every})
            n_n = len({r["node"] for r in every})
            print(f"\nwrote combined {out}\n  ({len(every)} rows: "
                  f"{n_p} paraphraser(s) x {n_n} node(s))")
        print(f"\noutput root: {config.SELECTIONS_PARAPHRASED_DIR / args.model}")
        return

    # ---- canonical mode (unchanged) -----------------------------------------
    models = config.discovered_models() if args.all else [args.model]
    for m in models:
        process_model(m, args)


if __name__ == "__main__":
    main()
