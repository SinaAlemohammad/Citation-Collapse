# Citation-Collapse

A benchmark for **recursive citation concentration** in large language models, with a
**paraphrase control** that tests whether the effect is driven by memorization of the
real seed papers.

Models play a recursive citation game: each "node" generates 120 papers that may cite
a shown candidate set; the output of one node seeds the next. We measure how citation
mass concentrates over the recursion, against a matched random-null floor. The
paraphrase control rewrites the seed's surface form while preserving meaning and re-runs
node 0 — if behaviour is unchanged, memorization is not what drives node 0.

This repository is self-contained: derived data ships with it, so the paraphrase tables,
the paper figures, and the metric CSVs regenerate **without any API calls**.

---

## Install

```sh
python -m venv .venv && source .venv/bin/activate      # or conda
pip install -e .                                       # installs the citation_collapse package + CLIs
pip install -e ".[metrics]"                            # + scipy/pandas/scikit-learn for the metrics scripts
cp .env.example .env                                   # add API key(s); only needed to GENERATE data
```

API keys are only needed to *generate* data (run models / paraphrase). Extraction,
metrics, figures and analysis read local files and need no keys.

## Layout

```
src/citation_collapse/       the installable package (see src/citation_collapse/README.md)
  core/       config, vendors, citation_parser, run_nodes   — the engine
  generate/   batch_node0, batch_gemini_nodes, generate_random
  analysis/   extract_selections, analyze, combine/merge  + metrics/
  paraphrase/ paraphrase_seed
  data_prep/  crawler (rebuild the seed from arXiv)
figures/                     standalone paper-figure reproduction package (F1–FI1)
data/                        seed.jsonl, seed_initial.jsonl, kv_pairs.jsonl, buckets/
paraphrased_papers/          11 paraphrased seeds (memorization control)
run_paraphrased/             node-0 generations for the paraphrase control
selections/ selections_paraphrased/   extracted citation-selection CSVs
results/                     stats tables + figures from analyze.py
```

Every directory has its own `README.md`. The raw 12-node model output `runs/` (~650 MB)
is **gitignored** — regenerate it with `cc-run` (below).

## Console commands

`pip install -e .` installs these entry points (all also runnable as
`python -m citation_collapse.<...>`):

| command | what |
|---|---|
| `cc-run --model <m>` | full recursive run → `runs/<m>/` |
| `cc-batch-node0 --model <m>` | node 0 via Batch API (or `--live`); `--paraphrased`/`--all` for the control |
| `cc-random --model <m>` | matched random-null replicates |
| `cc-extract --model <m>` | citation-selection CSVs (`--paraphrased all` for the control) |
| `cc-paraphrase --model <m>` | rewrite the seed with a model (`--report` for QC) |
| `cc-analyze` | paraphrase stats + figures from the shipped CSVs (no keys) |

## Reproduce, from cheapest to fullest

```sh
# A. paraphrase stats + figures from shipped CSVs — no keys, seconds
cc-analyze

# B. re-extract selections from shipped node-0 generations, then analyse — no keys
cc-extract --model gpt-5-mini --paraphrased all
cc-analyze

# C. the paper figures (F1–FI1) from the shipped CSVs — no keys
cd figures && python reproduce_all.py

# D. full pipeline from scratch — needs keys, costs credit
cc-paraphrase --model gpt-4.1 --fields title,abstract --live      # 1. paraphrase the seed
cc-batch-node0 --model gpt-5-mini --all --live                    # 2. run node 0 on every seed
cc-extract --model gpt-5-mini --paraphrased all                   # 3. extract
cc-analyze                                                        # 4. stats + figures
```

## The paraphrase control result

A paraphrase must both destroy surface form (mean 5-gram overlap ≤ 0.05) and preserve
meaning (coreness-index correlation > 0.95). `cc-analyze` scores all 11 paraphrasers;
**6 pass**. Across every arm, gpt-5-mini selected ~9.9 of a possible 10 shown papers and
essentially never fabricated citations — indistinguishable from the original-seed
baseline. This is evidence *against* node-0 behaviour being driven by memorization.
Caveat: node 0 pins most arms at the citation cap, so this is weak evidence of absence;
the discriminating signal is deeper in the recursion. See `results/README.md`.

## Determinism

Our-side randomness is fully fixed: `run_nodes` seeds Python's `random` and Faker with
`SEED=42`, so prompt construction, candidate sampling, and author/year assignment are
reproducible. `config.API_SEED` is passed as the OpenAI/Google generation seed (Anthropic
has none). APIs are best-effort, not bit-exact, but everything we control is pinned, and a
resumed run reproduces an uninterrupted one exactly.

Add a model in one line: `MODEL_REGISTRY` in `src/citation_collapse/core/config.py`.
