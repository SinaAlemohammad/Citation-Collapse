"""
Central configuration: every path and shared constant lives HERE, once.

Import this instead of hard-coding "gpt/output/..." anywhere. If a folder name
or constant changes, change it in this file only.

Layout produced by the pipeline
--------------------------------
    repo/
      kv_pairs.jsonl              <-- canonical (author, year) -> id map, SHARED
                                      by every model (it is model-independent by
                                      construction; see note below)
      buckets/bucket_50_500.jsonl <-- seed paper source (input data)
      runs/<model>/               <-- everything a model run produces
        prompts/                  N_<node>_inputs.jsonl
        seed/                     seed.jsonl, seed_initial.jsonl
        node_<k>/                 node_<k>.jsonl (+ stats, token files)
        master/                   kv_pairs.jsonl, token_usage.jsonl, concentration.jsonl
        citation_counts/          citation_counts.jsonl, hallucinations.jsonl, over_cap_violations.jsonl
        random_output/run_<i>/    matched random-null replicates
      results/                    extracted final results (selection CSVs, metric CSVs, aggregates)
      utils/                      metric-extraction + figure scripts
      figures/                    rendered figures

Why kv_pairs is shared
----------------------
Seed ids (SEED_i) and generated ids (N{node}P{i}) and their fake (author, year)
labels are assigned from a fixed RNG (SEED=42) in a deterministic order that does
NOT depend on which model wrote the text. So the (author, year) -> id table is
identical across models. Only the *shown sets* (papers_seen_id) and the written
bodies differ past node 0. Hence one canonical kv_pairs at the repo root.
"""

from pathlib import Path

# This file lives at <repo>/src/citation_collapse/core/config.py, so the repo
# root is four levels up. All data/output paths are resolved relative to it.
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"
SELECTIONS_DIR = ROOT / "selections"
FIGURES_DIR = ROOT / "figures"
BUCKETS_DIR = DATA_DIR / "buckets"

SEED_BUCKET = BUCKETS_DIR / "bucket_50_500.jsonl"      # raw seed source (only for regenerating from scratch)
SEED_STANDARDIZED = DATA_DIR / "seed.jsonl"            # CANONICAL standardized seed (shared by all models)
SEED_INITIAL = DATA_DIR / "seed_initial.jsonl"         # seed ground-truth citation counts
KV_PAIRS = DATA_DIR / "kv_pairs.jsonl"                 # canonical shared map

PARAPHRASED_DIR = ROOT / "paraphrased_papers"          # seed0_<paraphraser>.jsonl written by paraphrase_seed.py
RUN_PARAPHRASED_DIR = ROOT / "run_paraphrased"         # run_paraphrased/<run_model>/<paraphraser>/
SELECTIONS_PARAPHRASED_DIR = ROOT / "selections_paraphrased"   # selections_paraphrased/<run_model>/<paraphraser>/

# ---- shared experiment constants (identical across all models / vendors) ----
NODE_SIZE = 120            # papers generated per node
STRATUM_SIZE = 60          # how many of those are "position paper" prompts
CITATION_CAP = 10          # max unique citations kept per paper
PAPER_SET_LENGTH = 30      # candidates shown per prompt
TOTAL_NODES = 12           # recursion depth for a full run
SEED = 42
TOPIC = "Knowledge distillation or model compression in deep learning"
MAX_CONCURRENT = 60        # concurrent API calls

# ---- per-node completion (retry until all prompts answered) ----------------
MAX_NODE_PASSES = 50       # max retry passes over the unanswered prompts in a node
NODE_RETRY_BACKOFF = 10    # base seconds between passes (grows per pass, capped at 120)
STUCK_LIMIT = 20            # give up after this many consecutive passes with zero progress

# ---- determinism -----------------------------------------------------------
# Our-side randomness is fully fixed: run_nodes seeds the `random` module and
# Faker with SEED, so prompt construction, sampling, author/year assignment and
# the random-null are all reproducible. API_SEED / TEMPERATURE pin the vendor
# side as far as each API allows (best-effort; APIs are not guaranteed bit-exact).
API_SEED = SEED            # passed to OpenAI (seed=) and Google (seed=); Anthropic has no seed param
TEMPERATURE = None         # None = vendor default. Set 0.0 to push determinism further.
                           # WARNING: OpenAI gpt-5 / o-series reject non-default temperature.

# ---- model registry: run-name -> (vendor, api model string) -----------------
# The run-name is the folder under runs/. The api_model is the SDK string.
# Add a model by adding one line here; nothing else needs editing.
MODEL_REGISTRY = {
    # OpenAI
    "gpt-5-mini":        {"vendor": "openai",    "api_model": "gpt-5-mini"},
    "gpt-5":             {"vendor": "openai",    "api_model": "gpt-5"},
    "gpt-4.1":           {"vendor": "openai",    "api_model": "gpt-4.1"},
    "gpt-4.1-mini":      {"vendor": "openai",    "api_model": "gpt-4.1-mini"},
    # Anthropic
    "claude-haiku-4-5":  {"vendor": "anthropic", "api_model": "claude-haiku-4-5-20251001"},
    "claude-opus-4-8":   {"vendor": "anthropic", "api_model": "claude-opus-4-8"},
    "claude-sonnet-4-6": {"vendor": "anthropic", "api_model": "claude-sonnet-4-6"},
    # Google
    "gemini-2.5-pro":    {"vendor": "google",    "api_model": "models/gemini-2.5-pro"},
    "gemini-2.5-flash":  {"vendor": "google",    "api_model": "models/gemini-2.5-flash"},
    "gemini-3.1-pro-preview":    {"vendor": "google",    "api_model": "models/gemini-3.1-pro-preview"},
    "gemini-3.1-flash-lite": {"vendor": "google", "api_model": "models/gemini-3.1-flash-lite"},
}


def resolve_model(model: str) -> dict:
    if model not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{model}'. Known: {', '.join(sorted(MODEL_REGISTRY))}.\n"
            f"Add it to MODEL_REGISTRY in config.py."
        )
    return MODEL_REGISTRY[model]


def list_models() -> list:
    return sorted(MODEL_REGISTRY)


# ---- optional retarget: run a model against a PARAPHRASED seed --------------
# A paraphrased run is the same experiment with two things swapped: the seed it
# reads and the directory it writes. Both are resolved through this module at
# call time, so redirecting them here redirects every path helper and
# run_nodes.standardize_seed() at once -- no other file needs to know.
_RUN_DIR_OVERRIDE = None       # None -> runs/<model>;  set -> that dir, verbatim
_PARAPHRASE_PAIR = None        # None -> canonical;     set -> (run_model, paraphraser)


def paraphrased_seeds() -> list:
    """[(paraphraser, path)] for every paraphrased_papers/seed0_<paraphraser>.jsonl.
    The batch_*_input.jsonl scratch files do not match the glob and are skipped."""
    if not PARAPHRASED_DIR.exists():
        return []
    return [(p.stem[len("seed0_"):], p)
            for p in sorted(PARAPHRASED_DIR.glob("seed0_*.jsonl"))]


def paraphrased_runs(run_model: str) -> list:
    """[paraphraser] for every run_paraphrased/<run_model>/<paraphraser>/ that exists.
    These are runs that HAVE been done, unlike paraphrased_seeds() which lists seeds."""
    d = RUN_PARAPHRASED_DIR / run_model
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir())


def use_paraphrased_seed(run_model: str, paraphraser: str) -> Path:
    """Read seed0_<paraphraser>.jsonl instead of the canonical seed, and write to
    run_paraphrased/<run_model>/<paraphraser>/ instead of runs/<run_model>/.
    Call BEFORE constructing run_nodes.Experiment (its __init__ makes the dirs).
    `paraphraser` accepts either 'claude-sonnet-4-6' or 'claude_sonnet_4_6'."""
    global SEED_STANDARDIZED, _RUN_DIR_OVERRIDE, _PARAPHRASE_PAIR
    slug = paraphraser.replace("-", "_").replace(".", "_")
    p = PARAPHRASED_DIR / f"seed0_{slug}.jsonl"
    if not p.exists():
        known = ", ".join(s for s, _ in paraphrased_seeds()) or "(none)"
        raise SystemExit(f"No paraphrased seed for '{paraphraser}': {p} not found.\n"
                         f"Available: {known}")
    SEED_STANDARDIZED = p
    _RUN_DIR_OVERRIDE = RUN_PARAPHRASED_DIR / run_model / slug
    _PARAPHRASE_PAIR = (run_model, slug)
    return p


def use_paraphrased_run(run_model: str, paraphraser: str) -> Path:
    """Read an ALREADY-RUN paraphrased pair for analysis: no seed is needed, so
    unlike use_paraphrased_seed() this does not require the seed file to exist.
    -> run_paraphrased/<run_model>/<paraphraser>/ (raises if that run is absent)."""
    global _RUN_DIR_OVERRIDE, _PARAPHRASE_PAIR
    slug = paraphraser.replace("-", "_").replace(".", "_")
    d = RUN_PARAPHRASED_DIR / run_model / slug
    if not d.exists():
        known = ", ".join(paraphrased_runs(run_model)) or "(none)"
        raise SystemExit(f"No paraphrased run at {d}.\n"
                         f"Available for {run_model}: {known}")
    _RUN_DIR_OVERRIDE = d
    _PARAPHRASE_PAIR = (run_model, slug)
    return d


def reset_run_dir() -> None:
    """Undo use_paraphrased_seed()/use_paraphrased_run() (needed when looping)."""
    global SEED_STANDARDIZED, _RUN_DIR_OVERRIDE, _PARAPHRASE_PAIR
    SEED_STANDARDIZED = ROOT / "seed.jsonl"
    _RUN_DIR_OVERRIDE = None
    _PARAPHRASE_PAIR = None


# ---- path helpers (use these everywhere) ------------------------------------
def model_dir(model: str) -> Path:
    if _RUN_DIR_OVERRIDE is not None:     # paraphrased run: <run_model>/<paraphraser>
        return _RUN_DIR_OVERRIDE
    return RUNS_DIR / model
def prompts_dir(model: str) -> Path:      return model_dir(model) / "prompts"
def seed_dir(model: str) -> Path:         return model_dir(model) / "seed"
def master_dir(model: str) -> Path:       return model_dir(model) / "master"
def citation_counts_dir(model: str) -> Path: return model_dir(model) / "citation_counts"
def node_dir(model: str, node: int) -> Path: return model_dir(model) / f"node_{node}"
def node_file(model: str, node: int) -> Path: return node_dir(model, node) / f"node_{node}.jsonl"
def random_root(model: str) -> Path:      return model_dir(model) / "random_output"
def seed_file(model: str) -> Path:        return seed_dir(model) / "seed.jsonl"
def selections_dir(model: str) -> Path:
    """Where this run's selection CSVs go. A paraphrased run MUST land under
    selections_paraphrased/<run_model>/<paraphraser>/, or it would overwrite the
    canonical selections/<model>/ CSVs -- same model name, different experiment."""
    if _PARAPHRASE_PAIR is not None:
        run_model, slug = _PARAPHRASE_PAIR
        return SELECTIONS_PARAPHRASED_DIR / run_model / slug
    return SELECTIONS_DIR / model


def selection_path(model: str, node: int) -> Path:
    return selections_dir(model) / f"{model}-node{node}.csv"


def kv_pairs_path(model: str | None = None) -> Path:
    """Canonical root kv_pairs if present, else the model's own master copy."""
    if KV_PAIRS.exists():
        return KV_PAIRS
    if model is not None:
        return master_dir(model) / "kv_pairs.jsonl"
    return KV_PAIRS


def discovered_models() -> list:
    """Models that actually have a runs/<model>/ directory on disk."""
    if not RUNS_DIR.exists():
        return []
    return sorted(p.name for p in RUNS_DIR.iterdir() if p.is_dir())