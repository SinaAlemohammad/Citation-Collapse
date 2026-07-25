#!/usr/bin/env python
"""
Paraphrase the seed corpus with any model, via that vendor's Batch API.

This builds the stimulus for the separating test of Section 6: a map this
consistent across vendors could be an absorbed citation consensus memorized from
overlapping training corpora, or a convergent content-level preference for how a
citable paper reads. A paraphrase preserves the meaning and destroys the surface
form, so re-running node 0 on paraphrased records separates the two:

    map survives paraphrase  -> content-level preference (meaning drives it)
    map collapses            -> surface recognition / memorized identifiers

Reads the canonical seed.jsonl (120 blinded records) and writes a paraphrased
copy in the SAME format to paraphrased_papers/seed0_<model>.jsonl. The id,
author, year, and record order are preserved byte-for-byte, which is what keeps
the rerun comparable: run_nodes.standardize_seed() replays its RNG off that file,
so identical ids/authors/years reproduce the identical kv_pairs, the identical
panels, and the identical budgets. Only the text changes.

By default only the abstract is paraphrased (--fields abstract). Note the title
is by far the strongest identifier in the record -- see the --fields note in
--help and the README before choosing.

Transport is reused from batch_node0.py: the same three BatchAdapter subclasses,
with this script's own system prompt and JSON schema injected.

Usage
-----
  python paraphrase_seed.py --model gpt-4.1-mini                 # submit+poll+merge
  python paraphrase_seed.py --model gpt-4.1-mini --fields title,abstract
  python paraphrase_seed.py --model claude-sonnet-4-6 --submit   # or split it
  python paraphrase_seed.py --model claude-sonnet-4-6 --collect
  python paraphrase_seed.py --model gpt-4.1-mini --report        # QC, no API calls

Incremental: every run paraphrases ONLY the records missing from the output file
and merges. Requires the API key for the model's vendor.
"""
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from citation_collapse.core import config
from citation_collapse.generate.batch_node0 import ADAPTERS, _EMPTY_USAGE

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OUT_DIR = config.ROOT / "paraphrased_papers"
MAX_OUTPUT_TOKENS = 4000        # abstracts are ~200-400 words; generous headroom


# --------------------------------------------------------------------------- #
# The prompt.
#
# Every clause below is doing experimental work, because this prompt IS the
# manipulation. The design target is: destroy the surface form (the thing a
# memorizing model could recognize) while leaving the semantic content (the thing
# a content-level preference could respond to) untouched.
#
#  - Domain vocabulary is PRESERVED on purpose. "teacher", "student", "distill",
#    "logit", "temperature", "feature", "layer" are exactly the terms App. F's
#    coreness index counts. Paraphrasing them away would move the coreness score
#    and destroy the content variable the test is trying to hold fixed, so the
#    rerun would confound "surface form removed" with "content changed".
#  - Author-coined method names ARE removed. "TED", "MEAL", "BERT-of-Theseus" are
#    arbitrary labels, not content: the meaning is "task-aware layer-wise
#    distillation with task-aware filters", not the string "TED". They are the
#    purest memorization cue in the record, so leaving them in would let
#    recognition fire through a channel that carries no meaning.
#  - Length is pinned so panels stay comparable; a systematically shorter
#    paraphrase arm would change the panel's token profile, not just its wording.
#  - The 5-consecutive-word ceiling operationalizes "style not preserved" and is
#    checkable after the fact -- --report measures it.
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You rewrite scientific abstracts for a controlled experiment. You preserve "
    "meaning exactly and surface form not at all. You reply with JSON only, and "
    "never with commentary."
)

_INSTRUCTIONS = """Rewrite the {what} below so that it conveys exactly the same technical content in completely different words.

Preserve, without exception:
- every technical claim, method, result, number, and comparison, with the same emphasis and the same hedging
- the technical vocabulary of the field (teacher, student, distillation, logits, temperature, features, layers, and so on): these words carry the meaning and must NOT be swapped for synonyms
- the approximate length (within about 15% of the original)
- the register and tone of a scientific abstract

Change as completely as you can:
- sentence structure, clause order, and the order in which the points are made
- every distinctive phrase, construction, and stylistic tic
- any name or acronym the authors coined for their own method or system: replace it with a plain description such as "the proposed method". Keep the names of things they did not coin (datasets, prior methods, architectures like BERT or ResNet).

Do not:
- add, remove, strengthen, or soften any claim
- add citations, commentary, preamble, or a summary of what you changed
- reuse any span of more than 5 consecutive words from the original

Return JSON with exactly this shape: {shape}
"""


def build_user_prompt(rec, fields):
    what = " and ".join(fields)
    shape = "{" + ", ".join(f'"{f}": "..."' for f in fields) + "}"
    parts = [_INSTRUCTIONS.format(what=what, shape=shape)]
    for f in fields:
        parts.append(f"\nOriginal {f}:\n<<<\n{rec[f]}\n>>>")
    return "\n".join(parts)


def schema_for(fields):
    return {"type": "object",
            "properties": {f: {"type": "string"} for f in fields},
            "required": list(fields)}


# --------------------------------------------------------------------------- #
# paths + IO
# --------------------------------------------------------------------------- #
def slug(model):
    return re.sub(r"[^0-9a-z]+", "_", model.lower()).strip("_")


def out_file(model):
    return OUT_DIR / f"seed0_{slug(model)}.jsonl"


def state_file(model):
    return OUT_DIR / f"batch_paraphrase_{slug(model)}_job.json"


def load_seed():
    if not config.SEED_STANDARDIZED.exists():
        raise SystemExit(f"Canonical seed not found: {config.SEED_STANDARDIZED}")
    return [json.loads(l) for l in open(config.SEED_STANDARDIZED) if l.strip()]


def load_existing(model):
    p = out_file(model)
    if not p.exists():
        return {}
    return {r["id"]: r for r in (json.loads(l) for l in open(p) if l.strip())}


class _Spec:
    """Minimal stand-in for run_nodes.Experiment: the adapters only read these."""
    def __init__(self, model):
        s = config.resolve_model(model)
        self.model, self.vendor, self.api_model = model, s["vendor"], s["api_model"]


def _adapter(spec, fields):
    cls = ADAPTERS.get(spec.vendor)
    if cls is None:
        raise SystemExit(f"No batch adapter for vendor '{spec.vendor}'.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return cls(system=SYSTEM_PROMPT, schema=schema_for(fields),
               workdir=OUT_DIR, tag=f"paraphrase_{slug(spec.model)}")


# --------------------------------------------------------------------------- #
def submit(model, fields, max_output_tokens):
    spec = _Spec(model)
    seed = load_seed()
    done = load_existing(model)
    pending = [r for r in seed if r["id"] not in done]
    if not pending:
        print(f"all {len(seed)} records already paraphrased; nothing to submit.")
        return None
    print(f"{len(done)} already done; submitting the {len(pending)} remaining record(s).")

    records = [{"id": r["id"], "user_prompt": build_user_prompt(r, fields)}
               for r in pending]
    job_id = _adapter(spec, fields).submit(spec, records, max_output_tokens)

    state_file(model).write_text(json.dumps({
        "model": model, "vendor": spec.vendor, "job_id": job_id, "fields": fields,
        "submitted_ids": [r["id"] for r in pending],
        "created": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"saved job info -> {state_file(model)}")
    return job_id


def _apply(rec, text, fields):
    """-> (new_record, None) on success, or (None, reason). Shared by both paths."""
    if not text:
        return None, "no usable response"
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None, "response was not valid JSON"
    bad = [f for f in fields if not (obj.get(f) or "").strip()]
    if bad:
        return None, f"empty field(s): {','.join(bad)}"
    new = dict(rec)                          # preserves id/author/year + key order
    for f in fields:
        new[f] = obj[f].strip()
    return new, None


# --------------------------------------------------------------------------- #
# live path: one concurrent call per record, no file upload.
#
# The Batch API needs a multipart upload to /v1/files, which some networks block
# outright (openresty 400 "Request Header Or Cookie Too Large") while leaving
# /v1/chat/completions working. This path goes through vendors.complete() -- the
# same call run_nodes.py makes for every paper -- so it works wherever the
# benchmark itself works. It costs 2x batch pricing, which on 120 abstracts is
# negligible.
# --------------------------------------------------------------------------- #
async def _live_one(sem, spec, rec, fields, max_tokens):
    from citation_collapse.core import vendors
    async with sem:
        text, usage = await vendors.complete(
            spec.vendor, spec.api_model, SYSTEM_PROMPT, rec["user_prompt"],
            max_tokens=max_tokens, retry_tokens=max_tokens * 2,
            schema=schema_for(fields))
        return rec, text, usage


async def _live_pass(spec, pending, fields, max_tokens, concurrency):
    import asyncio
    sem = asyncio.Semaphore(concurrency)
    tasks = [_live_one(sem, spec, r, fields, max_tokens) for r in pending]
    out = []
    for coro in asyncio.as_completed(tasks):
        out.append(await coro)
    return out


def run_live(model, fields, max_output_tokens, concurrency, passes=4):
    import asyncio
    spec = _Spec(model)
    seed = load_seed()
    done = load_existing(model)
    todo = [{"id": r["id"], "user_prompt": build_user_prompt(r, fields), "_rec": r}
            for r in seed if r["id"] not in done]
    if not todo:
        print(f"all {len(seed)} records already paraphrased.")
        _write(model, seed, done)
        return
    print(f"{len(done)} already done; paraphrasing {len(todo)} record(s) live "
          f"({concurrency} concurrent, fields={fields})")

    reasons = {}

    # Every pass shares one event loop: vendors._clients caches a client per
    # vendor, and its connection pool binds to the loop that is live on first
    # use. A per-pass asyncio.run() closes that loop, so a retry pass would
    # reuse the client against a dead loop -> "Event loop is closed".
    async def _drive(todo):
        for attempt in range(passes):
            if not todo:
                break
            results = await _live_pass(spec, todo, fields, max_output_tokens, concurrency)
            still = []
            for rec, text, _usage in results:
                new, reason = _apply(rec["_rec"], text, fields)
                if new is None:
                    reasons[rec["id"]] = reason
                    still.append(rec)
                else:
                    done[rec["id"]] = new
                    reasons.pop(rec["id"], None)
                    print(f"  {rec['id']}: ok")
            todo = still
            if todo:
                print(f"  pass {attempt + 1}: {len(todo)} record(s) unanswered; retrying")
        return todo

    todo = asyncio.run(_drive(todo))
    _write(model, seed, done)
    if todo:
        print(f"\nWARNING: {len(todo)} record(s) not paraphrased:")
        for r in todo:
            print(f"    {r['id']}: {reasons.get(r['id'], 'unknown')}")
        print("Re-run to retry ONLY these -- new answers merge into the existing file.")


def collect(model, poll_interval):
    spec = _Spec(model)
    seed = load_seed()
    done = load_existing(model)
    pending = [r for r in seed if r["id"] not in done]
    if not pending:
        print(f"all {len(seed)} records already paraphrased.")
        _write(model, seed, done)
        return

    info_path = state_file(model)
    if not info_path.exists():
        raise SystemExit(f"{len(pending)} record(s) pending but no saved job at "
                         f"{info_path}. Run --submit first.")
    info = json.loads(info_path.read_text())
    fields = info.get("fields", ["abstract"])
    print(f"collecting {model} paraphrase job: {info['job_id']}  (fields: {fields})")

    by_key = _adapter(spec, fields).fetch(spec, info["job_id"], poll_interval)

    still = []
    for rec in pending:
        got = by_key.get(rec["id"])
        if not got:
            continue
        text, _usage, reason = got
        new, why = _apply(rec, text, fields)
        if new is None:
            still.append((rec["id"], reason or why))
        else:
            done[rec["id"]] = new

    _write(model, seed, done)
    if still:
        print(f"\nWARNING: {len(still)} record(s) not paraphrased:")
        for pid, reason in still:
            print(f"    {pid}: {reason}")
        print("Re-run to retry ONLY these -- new answers merge into the existing file.")


def _write(model, seed, done):
    """Write in canonical seed order. Order and ids must match seed.jsonl or the
    RNG replay in run_nodes.standardize_seed() diverges and the panels change."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = out_file(model)
    with open(p, "w") as f:
        for rec in seed:
            if rec["id"] in done:
                f.write(json.dumps(done[rec["id"]]) + "\n")
    n = sum(1 for rec in seed if rec["id"] in done)
    print(f"\nwrote {n}/{len(seed)} paraphrased records -> {p}")
    if n == len(seed):
        print("paraphrased seed COMPLETE. QC it with --report before rerunning node 0.")


# --------------------------------------------------------------------------- #
# report: did the paraphrase do what the experiment needs?
#
# Two things must both hold, and they pull against each other:
#   surface form destroyed -> low n-gram overlap with the original
#   meaning preserved      -> coreness index (App. F) essentially unchanged
# A low-overlap, coreness-shifted arm is not a clean paraphrase: it moved the
# content variable too, and a collapsed map would be uninterpretable.
# --------------------------------------------------------------------------- #
CORE = ("teacher", "student", "soft", "logit", "temperature", "dark knowledge",
        "mimic", "distill", "feature", "layer", "hint", "representation")
APPLIED = ("video", "3d", "speech", "federated", "reinforcement", "policy", "agent",
           "translation", "recommend", "medical", "graph", "detection",
           "segmentation", "cross-modal", "multimodal")


def coreness(rec):
    """App. F's index: core-methods density minus applications density, per 100 words,
    over title+abstract lowercased, counting raw substring occurrences."""
    text = f"{rec.get('title','')} {rec.get('abstract','')}".lower()
    n = len(text.split()) or 1
    c = sum(text.count(k) for k in CORE)
    a = sum(text.count(k) for k in APPLIED)
    return 100.0 * (c - a) / n


def _ngrams(s, n=5):
    w = re.findall(r"[a-z0-9]+", s.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def report(model, fields):
    seed = {r["id"]: r for r in load_seed()}
    done = load_existing(model)
    if not done:
        raise SystemExit(f"nothing to report: {out_file(model)} is missing or empty")

    rows = []
    for pid, new in done.items():
        old = seed[pid]
        o, p = _ngrams(old["abstract"]), _ngrams(new["abstract"])
        overlap = len(o & p) / len(o) if o else 0.0
        lo, lp = len(old["abstract"].split()), len(new["abstract"].split())
        rows.append((pid, overlap, coreness(old), coreness(new),
                     (lp - lo) / lo if lo else 0.0))

    n = len(rows)
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    ov = [r[1] for r in rows]
    dc = [r[3] - r[2] for r in rows]
    dl = [r[4] for r in rows]
    co, cp = [r[2] for r in rows], [r[3] for r in rows]
    # correlation of coreness before/after: the content variable must survive
    mo, mp = mean(co), mean(cp)
    num = sum((a - mo) * (b - mp) for a, b in zip(co, cp))
    den = (sum((a - mo) ** 2 for a in co) * sum((b - mp) ** 2 for b in cp)) ** 0.5
    r = num / den if den else float("nan")

    print(f"\nparaphrase QC: {model}  ({n}/120 records, fields={fields})\n")
    print("  surface form destroyed?")
    print(f"    mean 5-gram overlap with original   {mean(ov):.3f}   (want ~0.00-0.05)")
    print(f"    records with any 5-gram reused      {sum(1 for x in ov if x > 0)}/{n}")
    print(f"    worst record                        {max(ov):.3f}")
    print("\n  meaning preserved?")
    print(f"    coreness index r(before, after)     {r:.3f}   (want > 0.95)")
    print(f"    mean coreness shift                 {mean(dc):+.3f}  (want ~0)")
    print(f"    mean length change                  {mean(dl):+.1%}  (want within +-15%)")
    worst = sorted(rows, key=lambda x: -abs(x[3] - x[2]))[:5]
    print("\n  largest coreness shifts (inspect these by hand):")
    for pid, _o, c0, c1, _d in worst:
        print(f"    {pid:9s} {c0:+6.2f} -> {c1:+6.2f}   {seed[pid]['title'][:52]}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="NOTE on --fields: the title is the strongest identifier in the record. "
               "With --fields abstract (the default) a model that memorized the paper "
               "can still recognize it from the verbatim title, so a surviving map is "
               "NOT clean evidence of content-level preference. --fields title,abstract "
               "is the stronger separating arm. Running both gives a dose-response on "
               "surface form.")
    ap.add_argument("--model", required=True, choices=config.list_models())
    ap.add_argument("--fields", default="abstract",
                    help="comma-separated fields to paraphrase: abstract | title,abstract "
                         "(default: abstract)")
    ap.add_argument("--live", action="store_true",
                    help="skip the Batch API and make one concurrent live call per "
                         "record via vendors.complete(). Use when the network blocks "
                         "the multipart upload to /v1/files that OpenAI batches "
                         "require. 2x the cost of batch; negligible at 120 records.")
    ap.add_argument("--concurrency", type=int, default=config.MAX_CONCURRENT,
                    help=f"--live only: concurrent calls (default {config.MAX_CONCURRENT})")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--report", action="store_true",
                    help="QC an existing paraphrased file; makes no API calls")
    ap.add_argument("--poll-interval", type=int, default=30)
    ap.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    args = ap.parse_args()

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    bad = [f for f in fields if f not in ("title", "abstract")]
    if bad:
        raise SystemExit(f"--fields must be 'abstract' and/or 'title'; got {bad}")

    if args.report:
        report(args.model, fields)
    elif args.live:
        run_live(args.model, fields, args.max_output_tokens, args.concurrency)
    elif args.submit and not args.collect:
        submit(args.model, fields, args.max_output_tokens)
    elif args.collect and not args.submit:
        collect(args.model, args.poll_interval)
    else:
        submit(args.model, fields, args.max_output_tokens)
        collect(args.model, args.poll_interval)


if __name__ == "__main__":
    main()
