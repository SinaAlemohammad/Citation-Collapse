# data — canonical inputs

Read-only inputs shared by every run. Nothing here is model-specific.

| file | records | schema | role |
|---|---|---|---|
| `seed.jsonl` | 120 | `id, author, year, title, abstract` | the canonical seed: 120 real knowledge-distillation papers with **original** abstracts. Everything is measured against this. |
| `kv_pairs.jsonl` | 120 | `author, year, id` | the `(author, year) → id` map used to resolve citations during extraction. |
| `seed_initial.jsonl` | 120 | `id, citation_count` | ground-truth seed citation counts; copied into each run for self-containment. |
| `buckets/` | — | — | optional raw seed source; only `data_prep/crawler.py` writes here, only to rebuild the seed from scratch. |

## Why kv_pairs is shared, not per-run

Seed ids (`SEED_0` … `SEED_119`) and their `(author, year)` labels are assigned by a fixed
RNG (`SEED=42`) in an order that does **not** depend on the paper text. So the same map is
correct for the original seed and for every paraphrase of it — paraphrasing rewrites
`title`/`abstract` only and preserves id order and labels. That is what lets one
`kv_pairs.jsonl` resolve citations for every run, including the paraphrase control. It is
model-independent by construction; one copy serves all models.

`author`/`year` are synthetic labels for the citation game, not the papers' real authors or
publication years.
