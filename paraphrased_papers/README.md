# paraphrased_papers — rewritten seeds (memorization control)

One file per paraphrasing model: `seed0_<paraphraser>.jsonl` (11 total). Each is
`data/seed.jsonl` with **`title` and `abstract` rewritten** by that model, and
`id`/`author`/`year` left untouched and **in the same order**. `SEED_i` here is the
paraphrase of `SEED_i` in the canonical seed — a strict 1:1 mapping.

Produced by `cc-paraphrase` (see `src/citation_collapse/paraphrase/`).

## Quality varies — check before trusting an arm

A paraphrase must both destroy surface form (5-gram overlap ≤ 0.05) and preserve meaning
(coreness r > 0.95). `cc-analyze` scores all 11 (`results/paraphrase_qc.csv`); **6 pass**.
Two failure modes: **too literal** (`gpt-5`, `gpt-5-mini` kept ~3× too much wording) and
**over-compressed** (`claude-haiku-4-5`, `gemini-3.1-flash-lite` summarized instead of
paraphrasing). A file existing here does not mean it passed QC.

> `batch_*_input.jsonl` batch-submission scratch files are gitignored.
