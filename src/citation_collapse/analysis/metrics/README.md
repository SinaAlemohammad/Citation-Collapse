# metrics — theory metrics and per-model analyses

Two kinds of script. All are CLIs; run as `python -m citation_collapse.analysis.metrics.<name>`.

**Path-argument, standalone** (take `--root <run dir> --out <csv>`, no config needed):

| module | metric |
|---|---|
| `theory_metrics.py` | HHI / overlap / shown-ignored / per-paper false-negative, by node |
| `theory_metrics_jaccard.py` | mean pairwise Jaccard, by node |
| `aggregate_random_theory.py` | runs `theory_metrics` over every `random_output/run_*`, aggregates mean ± CI |
| `aggregate_random_jaccard.py` | same for the Jaccard metric |

**Model-argument** (take `--model <m>`, read `runs/<m>/…` via config):

| module | produces |
|---|---|
| `concentration.py` | top-10% citation share, LLM vs random baseline (+ png) |
| `false_negatives.py` | per-node exposure + exclusion rate (writes the `*_exposure.jsonl` the seed-rate scripts need) |
| `false_positives.py` | hallucination + over-cap counts per node |
| `seed_vs_llm_citation_rate.py` | seed-paper vs generated-paper citation rate (needs `false_negatives` first) |
| `random_seed_vs_llm_citation_rate.py` | same, averaged over the random-null runs |
| `ztest_tables.py` | z-test table PNGs from the jsonl outputs above |
| `update_node0_citations.py` | re-parse every `node_0.jsonl` with the current parser (`--root/--kv/--dry-run`) |

`random_aggregate.py` is a shared library (Bonferroni mean-CI, random-run discovery),
not a CLI. Dependency order for the model-argument track:
`false_negatives` → `seed_vs_llm_citation_rate`; `concentration`/`false_negatives`/`seed_vs_llm`
→ `ztest_tables`. These need `runs/<model>/` output present (gitignored — regenerate first).
