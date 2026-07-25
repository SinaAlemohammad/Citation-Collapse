# core — the engine

The tightly-coupled foundation every other subpackage builds on.

| module | what |
|---|---|
| `config.py` | Single source of truth: all paths, experiment constants (`NODE_SIZE=120`, `CITATION_CAP=10`, `SEED=42`, `TOTAL_NODES=12`), the `MODEL_REGISTRY` (run-name → vendor/api_model), and the paraphrase-run retarget (`use_paraphrased_seed` / `use_paraphrased_run` / `reset_run_dir`). `ROOT` is the repo root; change a path here and everything follows. |
| `citation_parser.py` | The one format-agnostic citation matcher. `find_cited_candidates(body, candidates)` (authoritative, shown-restricted) and `extract_citations_from_body(body)` (regex fallback for hallucination QC). Handles parenthetical, narrative, `et al.`, `[Author, Year]`, accents, particles. Stdlib only. Run it directly for self-tests. |
| `vendors.py` | One `async complete(vendor, api_model, system, user, ...)` hiding OpenAI / Anthropic / Google differences (client, call, usage fields, JSON enforcement). Lazy SDK imports; reads keys from env. |
| `run_nodes.py` | The generation harness. `Experiment` holds RNG-faithful seed standardization, deterministic prompt generation, parsing, a retry-until-complete loop per node, and resume (replays finished nodes with no API calls). `cc-run` / `python -m citation_collapse.core.run_nodes`. |

Add a model by adding one line to `MODEL_REGISTRY` in `config.py` — nothing else changes.
