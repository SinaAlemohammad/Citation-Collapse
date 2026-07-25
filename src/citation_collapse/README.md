# citation_collapse — the package

Installable Python package (`pip install -e .` from the repo root). Import graph,
leaves first:

```
core/       config, vendors, citation_parser, run_nodes     — no intra-package deps except within core
generate/   batch_node0, batch_gemini_nodes, generate_random  → core
analysis/   extract_selections, analyze, combine/merge        → core, paraphrase
  metrics/  theory/concentration/false_*/seed-rate/ztest      → core
paraphrase/ paraphrase_seed                                   → core, generate
data_prep/  crawler                                           — standalone (arXiv → seed)
```

All imports are package-absolute (`from citation_collapse.core import config`), so any
module runs from anywhere once the package is installed. Each subpackage has its own
README. The paper-figure code lives **outside** the package in the top-level `figures/`
directory (it is a standalone, cwd-based reproduction bundle).

Entry points (declared in `pyproject.toml`): `cc-run`, `cc-batch-node0`, `cc-random`,
`cc-extract`, `cc-paraphrase`, `cc-analyze`. Everything is also runnable as
`python -m citation_collapse.<subpkg>.<module>`.
