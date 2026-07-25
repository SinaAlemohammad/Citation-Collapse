# run_paraphrased — node-0 generations for the paraphrase control

Node-0 output, one directory per (run-model, paraphraser) pair:

```
run_paraphrased/<run_model>/<paraphraser>/
├── node_0/       node_0.jsonl (120 generated papers) + stats + completion marker
├── master/       kv_pairs.jsonl, token_usage.jsonl
└── citation_counts/  citation_counts / hallucinations / over_cap_violations
```

Here `run_model = gpt-5-mini` and `<paraphraser>` is one of the 11 slugged paraphrasers
(e.g. `claude_sonnet_4_6`), mirroring `selections/`. The load-bearing fields in
`node_0.jsonl` are `body` (the generated text citations are parsed from) and
`papers_seen_id` (the 30 candidates shown).

Produced by `cc-batch-node0 --model gpt-5-mini --all --live`. Extraction reads these with
no keys:

```sh
cc-extract --model gpt-5-mini --paraphrased all
```

> The per-pair `prompts/` (deterministically regenerable, ~5 MB each) are gitignored.
