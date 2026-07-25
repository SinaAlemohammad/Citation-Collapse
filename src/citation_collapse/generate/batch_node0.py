#!/usr/bin/env python
"""
Run node 0 for ANY model via its vendor's Batch API, INCREMENTALLY.

All 120 node-0 prompts are sent as one async batch job (~50% cheaper on every
vendor, separate high rate limits -> no 429 grind). The prompts stay independent
(each is its own request inside the job). Output is written exactly like
run_nodes.py's node 0 (same node_0.jsonl schema, kv_pairs, completion marker), so
the rest of the pipeline works unchanged and a later `run_nodes.py --nodes N`
resumes at node 1.

This generalizes batch_gemini_node0.py to all three vendors. The experiment is
identical across them -- prompts, RNG, parsing, and output all come from
run_nodes.Experiment. Only the batch transport differs, and that variation is
isolated in the three BatchAdapter subclasses below, mirroring how vendors.py
isolates the live-call variation. Each adapter enforces JSON exactly the way
vendors.py does for that vendor, so a batched node 0 is interchangeable with a
live one:

    openai      response_format={"type": "json_object"}, max_completion_tokens
    anthropic   single forced tool (vendors._ANTHROPIC_SCHEMA), max_tokens
    google      response_schema constrained decoding, max_output_tokens

INCREMENTAL: every run only submits the prompts that are NOT already present in
node_0.jsonl, and merges the new answers into the ones already collected. So the
first run submits all 120; if some come back unanswered, the next run submits
ONLY those stragglers and adds them to the existing file -- never re-sending the
whole 120.

FAILURE REASONS: when a prompt has no usable response, the per-request reason
(Gemini finishReason such as RECITATION / SAFETY / MAX_TOKENS, an OpenAI error
body, an Anthropic errored/expired result) is printed, and the raw results file
is saved for inspection.

Prompts are generated with the SAME deterministic logic as run_nodes.py, so
node 0 stays byte-identical across models regardless of transport.

Usage
-----
  # submit remaining + poll-to-finish + merge, in one process (run under tmux):
  python batch_node0.py --model gpt-5-mini
  python batch_node0.py --model claude-haiku-4-5
  python batch_node0.py --model gemini-2.5-flash

  # or split it (submit, disconnect, collect later from any session):
  python batch_node0.py --model gpt-5-mini --submit
  python batch_node0.py --model gpt-5-mini --collect

Re-running after a partial result automatically retries ONLY the missing prompts.
Requires the API key for the model's vendor (OPENAI_API_KEY / ANTHROPIC_API_KEY /
GEMINI_API_KEY or GOOGLE_API_KEY) in the environment or a repo-root .env.
"""
import argparse
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

from citation_collapse.core import config
from citation_collapse.core import run_nodes
from citation_collapse.core.run_nodes import Experiment, SYSTEM_PROMPT
from citation_collapse.core.vendors import _ANTHROPIC_SCHEMA

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

NODE = 0
MAX_OUTPUT_TOKENS = 10000      # generous: batch has no per-call retry, avoid truncation

_EMPTY_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# --------------------------------------------------------------------------- #
# deterministic prompts + existing-state helpers  (vendor-independent)
# --------------------------------------------------------------------------- #
def build_prompts(model):
    """Construct the Experiment and generate node-0 prompts deterministically.
    -> (exp, records). Reproduces identical RNG state / self.seen / prompts on every call."""
    import random
    exp = Experiment(model, max_concurrent=1)
    random.seed(run_nodes.SEED)
    exp.pooled.extend(exp.standardize_seed())
    records = exp.generate_prompts(NODE)          # writes N_0_inputs.jsonl, fills exp.seen
    return exp, records


def state_file(model):
    return config.model_dir(model) / "batch_node0_job.json"


def load_existing_records(model):
    """Already-collected node-0 papers (so we only retry the missing prompts)."""
    p = config.node_file(model, NODE)
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


# --------------------------------------------------------------------------- #
# paraphrased-seed runs: <model> on seed0_<paraphraser>.jsonl
#   -> run_paraphrased/<model>/<paraphraser>/
# The experiment is untouched; config.use_paraphrased_seed() only swaps which
# seed is read and which directory is written.
# --------------------------------------------------------------------------- #
def is_complete(model):
    """Node 0 finished? Uses run_nodes' own completion marker, so a run started
    by run_nodes.py and one started here agree on what 'done' means."""
    return (config.node_dir(model, NODE) / f"node_{NODE}_token_totals.jsonl").exists()


def run_live_node0(model, max_concurrent, write_root_kv=False):
    """Node 0 via the LIVE path instead of the Batch API.

    Required for OpenAI on this network: batching needs POST /v1/files (multipart
    upload), which an openresty proxy rejects with 400. Chat completions work, so
    this drives run_nodes.Experiment -- the same code path a normal run uses, with
    its per-node retry loop -- and stops after node 0."""
    import asyncio
    exp = Experiment(model, max_concurrent)
    asyncio.run(exp.run(1, write_root_kv))        # nodes=1 -> node 0 only


def run_paraphrased(model, paraphraser, args):
    """One (model, paraphraser) pair. -> 'skipped' | 'done' | 'failed: ...'"""
    config.reset_run_dir()
    seed_path = config.use_paraphrased_seed(model, paraphraser)
    out = config.model_dir(model)

    if is_complete(model):
        print(f"\n=== {model} x {paraphraser}: already complete -> {out}  (skipping)")
        return "skipped"

    print(f"\n=== {model} x {paraphraser}")
    print(f"    seed -> {seed_path}")
    print(f"    out  -> {out}")
    try:
        if args.live:
            run_live_node0(model, args.max_concurrent, args.write_root_kv)
        else:
            submit(model, args.max_output_tokens)
            collect(model, args.poll_interval, args.write_root_kv)
    except SystemExit as e:                       # keep --all going past one bad seed
        print(f"    FAILED: {e}")
        return f"failed: {e}"
    return "done" if is_complete(model) else "incomplete"


# --------------------------------------------------------------------------- #
# vendor adapters: submit a batch, then poll + fetch its per-prompt results
#
# Each adapter implements
#     submit(exp, records, max_output_tokens) -> job_id
#     fetch(exp, job_id, poll_interval)       -> {prompt_id: (text|None, usage, reason|None)}
# `text` is a JSON string or None, exactly like vendors.complete(), so
# Experiment._record_paper consumes it unchanged.
#
# The system prompt, JSON schema, and scratch-file location are set at
# construction rather than hardcoded, so the same three transports serve both
# node-0 generation (SYSTEM_PROMPT + the title/abstract/body schema) and seed
# paraphrasing (paraphrase_seed.py's own prompt + schema).
# --------------------------------------------------------------------------- #
class BatchAdapter:
    def __init__(self, system, schema=None, workdir=None, tag="node0"):
        self.system = system
        self.schema = schema            # vendor-native schema; unused for OpenAI
        self.workdir = workdir
        self.tag = tag

    def input_file(self):
        return self.workdir / f"batch_{self.tag}_input.jsonl"

    def results_file(self):
        return self.workdir / f"batch_{self.tag}_results.jsonl"

    def _save_raw(self, lines):
        self.results_file().write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"saved raw results -> {self.results_file()}")

    def submit(self, exp, records, max_output_tokens):
        raise NotImplementedError

    def fetch(self, exp, job_id, poll_interval):
        raise NotImplementedError

    @staticmethod
    def _wait(get_state, terminal, poll_interval, label):
        st = get_state()
        while st not in terminal:
            print(f"  {datetime.now().strftime('%H:%M:%S')}  state={st} "
                  f"(waiting {poll_interval}s)")
            time.sleep(poll_interval)
            st = get_state()
        print(f"{label} finished: state={st}")
        return st


# ---- Google / Gemini --------------------------------------------------------
COMPLETED_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
                    "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


def _state(job):
    """Gemini job state name. Kept at module level: batch_gemini_nodes.py imports it."""
    s = getattr(job, "state", None)
    return getattr(s, "name", None) or str(s)


def _usage_from(resp):
    um = (resp.get("usageMetadata") or resp.get("usage_metadata") or {}) if resp else {}
    return {"prompt_tokens": um.get("promptTokenCount") or um.get("prompt_token_count") or 0,
            "completion_tokens": um.get("candidatesTokenCount") or um.get("candidates_token_count") or 0,
            "total_tokens": um.get("totalTokenCount") or um.get("total_token_count") or 0}


def _parse_line(line):
    """Gemini result line -> (key, text|None, usage, reason|None). reason is set only
    when text is None, and records why (finishReason / error / blocked) for diagnosis.
    Kept at module level: batch_gemini_nodes.py imports it."""
    obj = json.loads(line)
    key = obj.get("key")
    err = obj.get("error") or obj.get("status")
    resp = obj.get("response")
    if resp is None and "candidates" in obj:
        resp = obj
    usage = _usage_from(resp)

    text, finish = None, None
    if resp:
        cands = resp.get("candidates") or []
        if cands:
            finish = cands[0].get("finishReason") or cands[0].get("finish_reason")
            parts = ((cands[0].get("content") or {}).get("parts")) or []
            t = "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
            text = t or None

    reason = None
    if text is None:
        if err:
            reason = f"error: {json.dumps(err)[:200]}"
        elif finish:
            reason = f"finishReason={finish}"
        elif resp is None:
            reason = "no response object"
        elif not resp.get("candidates"):
            reason = "no candidates (blocked)"
        else:
            reason = "empty text"
    return key, text, usage, reason


class GoogleBatch(BatchAdapter):
    """Gemini Batch API: upload a JSONL of requests, poll the job, download results.

    A response_schema is set so Gemini uses constrained decoding and ALWAYS returns
    valid JSON. Without it, response_mime_type alone lets smaller models emit broken
    JSON on long outputs (unterminated strings, repeated fragments), which then fail
    json parsing and look like 'unanswered' prompts."""

    def _build_jsonl(self, records, max_output_tokens):
        gen_cfg = {"response_mime_type": "application/json",
                   "response_schema": self.schema,
                   "max_output_tokens": max_output_tokens,
                   "seed": config.API_SEED}
        if config.TEMPERATURE is not None:
            gen_cfg["temperature"] = config.TEMPERATURE

        path = self.input_file()
        with open(path, "w") as f:
            for rec in records:
                f.write(json.dumps({
                    "key": rec["id"],
                    "request": {
                        "contents": [{"parts": [{"text": rec["user_prompt"]}],
                                      "role": "user"}],
                        "system_instruction": {"parts": [{"text": self.system}]},
                        "generation_config": gen_cfg,
                    }}) + "\n")
        return path

    def submit(self, exp, records, max_output_tokens):
        from google import genai
        from google.genai import types

        path = self._build_jsonl(records, max_output_tokens)
        print(f"wrote {len(records)} request(s) -> {path}")

        client = genai.Client()
        uploaded = client.files.upload(
            file=str(path),
            config=types.UploadFileConfig(display_name=f"{exp.model}-node0-input",
                                          mime_type="jsonl"))
        print(f"uploaded input file: {uploaded.name}")

        job = client.batches.create(
            model=exp.api_model,
            src=uploaded.name,
            config={"display_name": f"{exp.model}-node0"})
        print(f"created batch job: {job.name}  (state: {_state(job)})")
        return job.name

    def _download(self, client, job):
        dest = getattr(job, "dest", None)
        fname = getattr(dest, "file_name", None) if dest else None
        lines = []
        if fname:
            raw = client.files.download(file=fname)
            text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            lines = [ln for ln in text.splitlines() if ln.strip()]
        else:
            inl = getattr(dest, "inlined_responses", None) if dest else None
            if inl:
                for r in inl:
                    resp = getattr(r, "response", None)
                    d = resp.to_json_dict() if hasattr(resp, "to_json_dict") else {}
                    lines.append(json.dumps({"response": d}))
        self._save_raw(lines)
        return lines

    def fetch(self, exp, job_id, poll_interval):
        from google import genai
        client = genai.Client()
        job = client.batches.get(name=job_id)

        def _get():
            nonlocal job
            job = client.batches.get(name=job_id)
            return _state(job)

        st = self._wait(_get, COMPLETED_STATES, poll_interval, "job")
        if st != "JOB_STATE_SUCCEEDED":
            raise SystemExit(f"batch did not succeed ({st}). "
                             f"error: {getattr(job, 'error', None)}")

        lines = self._download(client, job)
        print(f"downloaded {len(lines)} result line(s)")
        out = {}
        for ln in lines:
            try:
                k, text, usage, reason = _parse_line(ln)
            except Exception as e:
                print(f"  WARN: could not parse a result line: {e}")
                continue
            if k is not None:
                out[k] = (text, usage, reason)
        return out


# ---- OpenAI -----------------------------------------------------------------
OPENAI_TERMINAL = {"completed", "failed", "expired", "cancelled"}


class OpenAIBatch(BatchAdapter):
    """OpenAI Batch API: upload a JSONL of /v1/chat/completions requests, poll, download.

    The body mirrors vendors._openai exactly (response_format json_object,
    max_completion_tokens, optional seed/temperature), so a batched response is
    the same object a live call would have returned."""

    def _build_jsonl(self, exp, records, max_output_tokens):
        body_base = {
            "model": exp.api_model,
            "max_completion_tokens": max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        if config.API_SEED is not None:
            body_base["seed"] = config.API_SEED
        if config.TEMPERATURE is not None:
            # NOTE: gpt-5/o-series reject a non-default temperature (see vendors.py)
            body_base["temperature"] = config.TEMPERATURE

        path = self.input_file()
        with open(path, "w") as f:
            for rec in records:
                body = dict(body_base)
                body["messages"] = [{"role": "system", "content": self.system},
                                    {"role": "user", "content": rec["user_prompt"]}]
                f.write(json.dumps({"custom_id": rec["id"],
                                    "method": "POST",
                                    "url": "/v1/chat/completions",
                                    "body": body}) + "\n")
        return path

    def submit(self, exp, records, max_output_tokens):
        from openai import OpenAI

        path = self._build_jsonl(exp, records, max_output_tokens)
        print(f"wrote {len(records)} request(s) -> {path}")

        client = OpenAI()
        with open(path, "rb") as fh:
            uploaded = client.files.create(file=fh, purpose="batch")
        print(f"uploaded input file: {uploaded.id}")

        job = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"{exp.model}-node0"})
        print(f"created batch job: {job.id}  (state: {job.status})")
        return job.id

    @staticmethod
    def _parse(line):
        """-> (custom_id, text|None, usage, reason|None)"""
        obj = json.loads(line)
        key = obj.get("custom_id")
        err = obj.get("error")
        resp = obj.get("response") or {}
        code = resp.get("status_code")
        body = resp.get("body") or {}

        u = body.get("usage") or {}
        usage = {"prompt_tokens": u.get("prompt_tokens", 0) or 0,
                 "completion_tokens": u.get("completion_tokens", 0) or 0,
                 "total_tokens": u.get("total_tokens", 0) or 0}

        text, finish = None, None
        choices = body.get("choices") or []
        if choices:
            finish = choices[0].get("finish_reason")
            content = (choices[0].get("message") or {}).get("content")
            text = (content or "").strip() or None

        reason = None
        if text is None:
            if err:
                reason = f"error: {json.dumps(err)[:200]}"
            elif body.get("error"):
                reason = f"error: {json.dumps(body['error'])[:200]}"
            elif code and code != 200:
                reason = f"status_code={code}"
            elif finish:
                reason = f"finish_reason={finish}"
            elif not choices:
                reason = "no choices"
            else:
                reason = "empty text"
        return key, text, usage, reason

    def fetch(self, exp, job_id, poll_interval):
        from openai import OpenAI
        client = OpenAI()
        job = client.batches.retrieve(job_id)

        def _get():
            nonlocal job
            job = client.batches.retrieve(job_id)
            return job.status

        st = self._wait(_get, OPENAI_TERMINAL, poll_interval, "job")

        lines = []
        if getattr(job, "output_file_id", None):
            text = client.files.content(job.output_file_id).text
            lines = [ln for ln in text.splitlines() if ln.strip()]
        # per-request failures land in a separate file; keep them for the reasons
        if getattr(job, "error_file_id", None):
            text = client.files.content(job.error_file_id).text
            lines += [ln for ln in text.splitlines() if ln.strip()]
        self._save_raw(lines)

        if st != "completed" and not lines:
            raise SystemExit(f"batch did not succeed ({st}). "
                             f"errors: {getattr(job, 'errors', None)}")
        print(f"downloaded {len(lines)} result line(s)")

        out = {}
        for ln in lines:
            try:
                k, text, usage, reason = self._parse(ln)
            except Exception as e:
                print(f"  WARN: could not parse a result line: {e}")
                continue
            if k is not None:
                out[k] = (text, usage, reason)
        return out


# ---- Anthropic --------------------------------------------------------------
ANTHROPIC_TERMINAL = {"ended", "canceled"}


class AnthropicBatch(BatchAdapter):
    """Anthropic Message Batches API: requests are sent INLINE (no file upload),
    then results are streamed back once processing_status == "ended".

    JSON is forced with the same single-tool schema vendors._anthropic uses, so a
    batched response carries the same tool_use block a live call would."""

    def _requests(self, exp, records, max_output_tokens):
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        out = []
        for rec in records:
            params = {
                "model": exp.api_model,
                "system": self.system,
                "messages": [{"role": "user", "content": rec["user_prompt"]}],
                "max_tokens": max_output_tokens,
                "tools": [{"name": "json",
                           "description": "Output valid JSON only",
                           "input_schema": self.schema}],
                "tool_choice": {"type": "tool", "name": "json"},
            }
            if config.TEMPERATURE is not None:   # Anthropic has no seed param
                params["temperature"] = config.TEMPERATURE
            out.append(Request(custom_id=rec["id"],
                               params=MessageCreateParamsNonStreaming(**params)))
        return out

    def submit(self, exp, records, max_output_tokens):
        import anthropic
        client = anthropic.Anthropic()

        reqs = self._requests(exp, records, max_output_tokens)
        # keep a local copy of what we sent, mirroring the other two vendors
        self.input_file().write_text(
            "".join(json.dumps({"custom_id": r["custom_id"]}) + "\n" for r in reqs))
        print(f"prepared {len(reqs)} inline request(s)")

        job = client.messages.batches.create(requests=reqs)
        print(f"created batch job: {job.id}  (state: {job.processing_status})")
        return job.id

    @staticmethod
    def _extract(message):
        """Mirror vendors._anthropic: the forced tool's input IS the JSON payload."""
        content = None
        for block in message.content:
            if block.type == "tool_use":
                content = json.dumps(block.input)
                break
        if content == "{}" or message.stop_reason == "max_tokens":
            content = None
        return content

    def fetch(self, exp, job_id, poll_interval):
        import anthropic
        client = anthropic.Anthropic()

        def _get():
            b = client.messages.batches.retrieve(job_id)
            return b.processing_status

        self._wait(_get, ANTHROPIC_TERMINAL, poll_interval, "job")

        out, lines = {}, []
        # results arrive in ANY order -- key by custom_id, never by position
        for r in client.messages.batches.results(job_id):
            lines.append(r.to_json() if hasattr(r, "to_json") else str(r))
            kind = r.result.type
            if kind == "succeeded":
                msg = r.result.message
                u = msg.usage
                usage = {"prompt_tokens": u.input_tokens if u else 0,
                         "completion_tokens": u.output_tokens if u else 0,
                         "total_tokens": (u.input_tokens + u.output_tokens) if u else 0}
                text = self._extract(msg)
                reason = None if text else f"stop_reason={msg.stop_reason}"
                out[r.custom_id] = (text, usage, reason)
            else:
                err = getattr(r.result, "error", None)
                detail = f": {err.type}" if err is not None and hasattr(err, "type") else ""
                out[r.custom_id] = (None, dict(_EMPTY_USAGE), f"{kind}{detail}")
        self._save_raw(lines)
        print(f"downloaded {len(out)} result(s)")
        return out


ADAPTERS = {"openai": OpenAIBatch, "anthropic": AnthropicBatch, "google": GoogleBatch}

# Node-0 schemas, per vendor, exactly as the live path in vendors.py sends them.
# They are NOT interchangeable: the Anthropic one carries per-field descriptions
# ("no citations", "cite ... as (Surname, Year)") that are part of the prompt the
# model sees, so collapsing the two would silently change the experiment.
NODE0_SCHEMAS = {
    "openai": None,                       # json_object mode takes no schema
    "anthropic": _ANTHROPIC_SCHEMA,       # imported, never copied
    "google": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "abstract": {"type": "string"},
            "body": {"type": "string"},
        },
        "required": ["title", "abstract", "body"],
    },
}


def _adapter(exp):
    cls = ADAPTERS.get(exp.vendor)
    if cls is None:
        raise SystemExit(f"No batch adapter for vendor '{exp.vendor}'. "
                         f"Use run_nodes.py for {exp.model}.")
    return cls(system=SYSTEM_PROMPT, schema=NODE0_SCHEMAS[exp.vendor],
               workdir=config.model_dir(exp.model), tag="node0")


# --------------------------------------------------------------------------- #
# submit (only the prompts not yet answered)
# --------------------------------------------------------------------------- #
def submit(model, max_output_tokens):
    exp, records = build_prompts(model)
    answered = {r["id"] for r in load_existing_records(model)}
    pending = [r for r in records if r["id"] not in answered]
    if not pending:
        print(f"node 0 already has all {len(records)} prompts answered; nothing to submit.")
        return None
    print(f"{len(answered)} already answered; submitting the {len(pending)} remaining prompt(s).")

    job_id = _adapter(exp).submit(exp, pending, max_output_tokens)

    state_file(model).write_text(json.dumps({
        "model": model, "vendor": exp.vendor, "job_id": job_id,
        "submitted_ids": [r["id"] for r in pending],
        "created": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"saved job info -> {state_file(model)}")
    return job_id


# --------------------------------------------------------------------------- #
# collect (poll, merge new answers into existing, finalize)
# --------------------------------------------------------------------------- #
def collect(model, poll_interval, write_root_kv):
    exp, records = build_prompts(model)
    existing = load_existing_records(model)
    answered = {r["id"] for r in existing}
    pending_ids = [r["id"] for r in records if r["id"] not in answered]

    # nothing left to do -> just finalize / mark complete (no polling)
    if not pending_ids:
        print(f"all {len(records)} prompts already answered; finalizing/marking node 0.")
        _finalize_merged(exp, existing, [], [], [], len(records), write_root_kv,
                         is_first=False)
        return

    info_path = state_file(model)
    if not info_path.exists():
        raise SystemExit(f"{len(pending_ids)} prompt(s) pending but no saved job at "
                         f"{info_path}. Run --submit first.")
    info = json.loads(info_path.read_text())
    job_id = info.get("job_id") or info.get("job_name")   # job_name: pre-rename state files
    print(f"collecting {model} node-0 batch job: {job_id}")

    by_key = _adapter(exp).fetch(exp, job_id, poll_interval)

    inv = {pid: ay for ay, pid in exp.seen.items()}
    new_results, new_hall, new_over, new_tok = [], [], [], []
    nc_tmp = defaultdict(int)
    still = []
    for rec in records:
        if rec["id"] in answered:          # keep already-collected papers untouched
            continue
        if rec["id"] not in by_key:        # not part of this job's results
            continue
        text, usage, reason = by_key[rec["id"]]
        if not exp._record_paper(rec, text, usage, inv, NODE, new_results,
                                 nc_tmp, new_hall, new_over, new_tok):
            still.append((rec["id"], reason or "no usable response"))

    merged = existing + new_results
    print(f"merged: {len(existing)} existing + {len(new_results)} new = {len(merged)} papers")
    _finalize_merged(exp, merged, new_hall, new_over, still, len(records),
                     write_root_kv, is_first=(len(existing) == 0))


# --------------------------------------------------------------------------- #
# finalize (same node-0 outputs as run_nodes; merge-aware for QC files)
# --------------------------------------------------------------------------- #
def _finalize_merged(exp, merged, new_hall, new_over, still_missing, n_prompts,
                     write_root_kv, is_first):
    model = exp.model
    config.node_dir(model, NODE).mkdir(parents=True, exist_ok=True)

    with open(config.node_file(model, NODE), "w") as f:
        for r in merged:
            f.write(json.dumps(r) + "\n")

    node_citations = Counter()
    for r in merged:
        for cid in r.get("citation_ids", []):
            node_citations[cid] += 1
    with open(config.node_dir(model, NODE) / f"node_{NODE}_stats.jsonl", "w") as f:
        for k, v in node_citations.items():
            f.write(json.dumps({k: v}) + "\n")

    # QC files: overwrite on the first batch, append only the NEW papers on retries
    qc_mode = "w" if is_first else "a"
    for name, items in (("hallucinations.jsonl", new_hall),
                        ("over_cap_violations.jsonl", new_over)):
        with open(config.citation_counts_dir(model) / name, qc_mode) as f:
            for it in items:
                f.write(json.dumps(it) + "\n")

    with open(config.citation_counts_dir(model) / "citation_counts.jsonl", "w") as f:
        for k, v in node_citations.items():
            f.write(json.dumps({k: v}) + "\n")
    kv_model = config.master_dir(model) / "kv_pairs.jsonl"
    with open(kv_model, "w") as f:
        for (a, y), pid in exp.seen.items():
            f.write(json.dumps({"author": a, "year": y, "id": pid}) + "\n")
    tot = {"node": NODE,
           "total_prompt_tokens": sum(r.get("prompt_tokens", 0) for r in merged),
           "total_completion_tokens": sum(r.get("completion_tokens", 0) for r in merged),
           "total_tokens": sum(r.get("total_tokens", 0) for r in merged)}
    with open(config.master_dir(model) / "token_usage.jsonl", "w") as f:
        f.write(json.dumps(tot) + "\n")

    print(f"\nwrote {len(merged)}/{n_prompts} papers -> {config.node_file(model, NODE)}")

    if still_missing:
        print(f"\nWARNING: {len(still_missing)} prompt(s) still unanswered:")
        for pid, reason in still_missing:
            print(f"    {pid}: {reason}")
        print("No completion marker written (node 0 is NOT done). Re-run to retry ONLY "
              "these remaining prompts -- new answers merge into the existing ones. "
              "kv_pairs were NOT pushed to root.")
        return

    with open(config.node_dir(model, NODE) / f"node_{NODE}_token_totals.jsonl", "w") as f:
        f.write(json.dumps(tot) + "\n")
    if write_root_kv or not config.KV_PAIRS.exists():
        config.KV_PAIRS.write_text(kv_model.read_text())
        print(f"wrote canonical {config.KV_PAIRS}")
    print(f"node 0 COMPLETE ({len(merged)}/{n_prompts}) and marked done. "
          f"Output in {config.model_dir(model)}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=config.list_models(),
                    help="the model that RUNS node 0")
    ap.add_argument("--paraphrased", metavar="PARAPHRASER",
                    help="run against paraphrased_papers/seed0_<PARAPHRASER>.jsonl "
                         "(the model that DID the paraphrasing); output goes to "
                         "run_paraphrased/<model>/<PARAPHRASER>/")
    ap.add_argument("--all", action="store_true",
                    help="run against every paraphrased seed on disk; already-"
                         "completed ones are skipped, so this is safe to re-run")
    ap.add_argument("--live", action="store_true",
                    help="bypass the Batch API and call the model directly. REQUIRED "
                         "for OpenAI on this network (POST /v1/files is blocked).")
    ap.add_argument("--max-concurrent", type=int, default=config.MAX_CONCURRENT,
                    help=f"concurrent calls for --live (default {config.MAX_CONCURRENT})")
    ap.add_argument("--submit", action="store_true",
                    help="submit a batch for the remaining (unanswered) prompts and exit")
    ap.add_argument("--collect", action="store_true",
                    help="poll the saved job and merge its results into node_0.jsonl")
    ap.add_argument("--write-root-kv", action="store_true",
                    help="overwrite the canonical root kv_pairs.jsonl once node 0 is complete")
    ap.add_argument("--poll-interval", type=int, default=60,
                    help="seconds between status polls while collecting (default 60)")
    ap.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS,
                    help=f"max output tokens per request (default {MAX_OUTPUT_TOKENS})")
    args = ap.parse_args()

    # ---- paraphrased-seed mode ----------------------------------------------
    if args.all or args.paraphrased:
        if args.all and args.paraphrased:
            raise SystemExit("--all and --paraphrased are mutually exclusive.")
        if args.all:
            targets = [s for s, _ in config.paraphrased_seeds()]
            if not targets:
                raise SystemExit(f"No seed0_*.jsonl in {config.PARAPHRASED_DIR}. "
                                 f"Run paraphrase_seed.py first.")
            print(f"{len(targets)} paraphrased seed(s): {', '.join(targets)}")
        else:
            targets = [args.paraphrased]

        results = [(t, run_paraphrased(args.model, t, args)) for t in targets]

        print(f"\n{'=' * 60}\nsummary: {args.model} on {len(results)} paraphrased seed(s)")
        for t, r in results:
            print(f"  {t:28s} {r}")
        done = sum(1 for _, r in results if r == "done")
        skipped = sum(1 for _, r in results if r == "skipped")
        print(f"  -> {done} run, {skipped} already done, "
              f"{len(results) - done - skipped} failed/incomplete")
        print(f"  output root: {config.RUN_PARAPHRASED_DIR / args.model}")
        return

    # ---- original canonical-seed mode (runs/<model>/), unchanged ------------
    if args.live:
        run_live_node0(args.model, args.max_concurrent, args.write_root_kv)
    elif args.submit and not args.collect:
        submit(args.model, args.max_output_tokens)
    elif args.collect and not args.submit:
        collect(args.model, args.poll_interval, args.write_root_kv)
    else:
        # default full flow: submit the remaining prompts, then poll + merge.
        # (if nothing is pending, collect() just finalizes/marks node 0.)
        submit(args.model, args.max_output_tokens)
        collect(args.model, args.poll_interval, args.write_root_kv)


if __name__ == "__main__":
    main()
