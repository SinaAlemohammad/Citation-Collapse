# selections_paraphrased — selections for the paraphrase control

Same measurement as `selections/`, but for the paraphrased runs, kept separate so it can
never overwrite the main-experiment CSVs (same model name, different experiment).

```
selections_paraphrased/<run_model>/<paraphraser>/<run_model>-node0.csv   per pair (120 rows)
selections_paraphrased/<run_model>/<run_model>-all-runs.csv              pooled (11 × 120 = 1320 rows)
```

Produced by `cc-extract --model gpt-5-mini --paraphrased all`. The pooled file carries an
extra **`paraphraser`** column so the runs stay separable once concatenated; `cc-analyze`
reads it to build `results/node0_selection_summary.csv`. Columns are otherwise identical to
`selections/` (see that README).
