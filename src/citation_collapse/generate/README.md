# generate — produce model output

Everything that calls the models to create `runs/<model>/…` (or the paraphrased
equivalent). All reuse `core.run_nodes.Experiment` so prompts, RNG, parsing and output
schema are identical across paths.

| module | command | what |
|---|---|---|
| `run_nodes` (in core) | `cc-run --model <m>` | the full recursive N-node run |
| `batch_node0.py` | `cc-batch-node0 --model <m>` | node 0 via the vendor Batch API, incremental/merge. `--live` calls the model directly (required for OpenAI where the Batch upload endpoint is blocked). `--paraphrased <x>` / `--all` retarget to paraphrased seeds → `run_paraphrased/`. Skips completed pairs, so `--all` is safe to re-run. |
| `batch_gemini_nodes.py` | `python -m citation_collapse.generate.batch_gemini_nodes` | the **full recursion** via one batch per node (Gemini only, since the recursion is sequential). Node 0 for other vendors → use `batch_node0`. |
| `generate_random.py` | `cc-random --model <m> --iters 50` | the matched random-null: per real paper keeps its citation count and shown set, re-draws which shown ids are cited. This is the concentration floor every metric compares against. |

Batch jobs can take up to ~24 h; run under tmux. Re-running a batch submits only the
prompts still missing and merges them in.
