# selections — extracted citation selections (main experiment)

The measurement layer for the recursive benchmark: `runs/.../node_<n>.jsonl` → one CSV row
per prompt describing which shown papers each generated paper cited.

```
selections/<model>/<model>-node<n>.csv     per model, per node
selections/<model>/<model>-all.csv         all nodes for that model (with --combined)
```

Produced by `cc-extract --model <m>` (or `--all`). Columns:

| column | meaning |
|---|---|
| `prompt_id`, `prompt_type`, `n_shown`, `papers_shown` | the prompt and its 30 candidates |
| `papers_selected`, `n_selected` | cited **and** shown — the real signal (capped at 10) |
| `cited_not_shown`, `n_cited_not_shown` | cited a real paper that wasn't offered |
| `n_fabricated` | citations resolving to **no** real paper |
| `hallucination_rate` | `n_cited_not_shown / (n_selected + n_cited_not_shown)` |

Citations are matched by `(author, year) → id` via `data/kv_pairs.jsonl`; an unresolvable id
triggers a warning rather than being silently counted as a hallucination. The paraphrase
control's selections live in `selections_paraphrased/`.
