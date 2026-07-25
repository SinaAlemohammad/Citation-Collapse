# paraphrase — the memorization control

`paraphrase_seed.py` rewrites the seed corpus with a model, so node 0 can be re-run on
text the model has not memorized. Reuses `generate.batch_node0.ADAPTERS` for transport.

```sh
cc-paraphrase --model gpt-4.1 --fields title,abstract --live      # write paraphrased_papers/seed0_gpt_4_1.jsonl
cc-paraphrase --model gpt-4.1 --fields title,abstract --report    # two-sided QC, no API calls
```

It rewrites only the chosen fields (`title,abstract`) and keeps `id`, `author`, `year`
and record order identical to `data/seed.jsonl` — so the shared `kv_pairs` map and the RNG
replay stay valid. `--live` calls the model directly; `--report` scores the paraphrase on
5-gram overlap (surface form destroyed?) and the coreness index (meaning preserved?).

This module is also the single source of the `coreness()` and `_ngrams()` metric functions
that `analysis.analyze` imports — the QC and the analysis compute identical numbers.

Full control loop: `cc-paraphrase` → `cc-batch-node0 --all` → `cc-extract --paraphrased all`
→ `cc-analyze`.
