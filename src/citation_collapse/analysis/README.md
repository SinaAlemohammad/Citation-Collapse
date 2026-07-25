# analysis — turn generations into numbers

Reads `runs/…` (or `run_paraphrased/…`) and produces the CSVs, metrics and figures the
paper is built on. No API calls except where a script regenerates model output.

| module | command | what |
|---|---|---|
| `extract_selections.py` | `cc-extract --model <m>` | the core measurement: per prompt, which shown papers were cited (`papers_selected`), which cited papers weren't shown, and which resolve to no real paper (`n_fabricated`), plus a per-prompt `hallucination_rate`. `--all` for every model; `--paraphrased all` for the control (→ `selections_paraphrased/` + a pooled `<model>-all-runs.csv`). |
| `analyze.py` | `cc-analyze` | the paraphrase-control stats + figures from the shipped CSVs (no keys). Writes `results/paraphrase_qc.csv`, `results/node0_selection_summary.csv`, and two figures. |
| `merge_node0_selections.py` | `python -m citation_collapse.analysis.merge_node0_selections` | concatenate every model's node-0 CSV → one repo-root CSV. |
| `combine_all_selections.py` | `python -m citation_collapse.analysis.combine_all_selections` | stack every model's all-nodes CSV → one CSV. |

The last two build the pooled CSVs the `figures/` package consumes. `metrics/` holds the
theory-metric and figure-input scripts — see `metrics/README.md`.
