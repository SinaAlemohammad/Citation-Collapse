#!/usr/bin/env python
"""
Run the FULL node recursion for a Gemini model via the Batch API -- one batch per node.

The recursion is sequential: node k's candidate pool is built from node k-1's
generated papers, so this submits one batch job per node, in order (node 0 -> wait ->
build node 1 -> batch it -> ...). Each node's 120 prompts go in a single batch with a
response_schema, so the JSON is always valid. If node 0 is already done (e.g. via
batch_gemini_node0.py), running this submits 11 batches for nodes 1..11.

It reuses run_nodes.py's recursion + resume machinery by subclassing Experiment and
overriding only generate_node, so finished nodes are replayed with NO API calls and
the output is byte-identical to run_nodes.py (same files, kv, markers). The rest of
the pipeline (extract_node_selections, etc.) works unchanged.

Because each batch can take up to ~24h and the run is sequential, run it under tmux.
If interrupted, re-run the SAME command: finished nodes are cached/replayed, and an
unfinished node retries ONLY its missing prompts (results merge in).

Usage:
  python batch_gemini_nodes.py --model gemini-2.5-flash                  # nodes 0..11
  python batch_gemini_nodes.py --model gemini-2.5-flash --nodes 12
  python batch_gemini_nodes.py --model gemini-2.5-flash --write-root-kv  # also push root kv
  python batch_gemini_nodes.py --model gemini-2.5-flash --fresh          # ignore cached nodes

Requires GEMINI_API_KEY (or GOOGLE_API_KEY). Reuses _parse_line/_state from
batch_gemini_node0.py, which must be in the repo.
"""
import argparse
import asyncio
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

from citation_collapse.core import config
from citation_collapse.core import run_nodes
from citation_collapse.core.run_nodes import Experiment, SYSTEM_PROMPT
from citation_collapse.generate.batch_node0 import _parse_line, _state

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MAX_OUTPUT_TOKENS = 10000
COMPLETED_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
                    "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
_SCHEMA = {"type": "object",
           "properties": {"title": {"type": "string"},
                          "abstract": {"type": "string"},
                          "body": {"type": "string"}},
           "required": ["title", "abstract", "body"]}


class BatchExperiment(Experiment):
    """Experiment whose per-node generation goes through one Gemini batch job."""

    def __init__(self, model, poll_interval=60, max_output_tokens=MAX_OUTPUT_TOKENS):
        super().__init__(model, max_concurrent=1)
        if self.vendor != "google":
            raise SystemExit(f"{model} is not a Gemini/google model; this script is "
                             f"Gemini-only. Use run_nodes.py for {self.vendor} models.")
        self.poll_interval = poll_interval
        self.max_output_tokens = max_output_tokens
        self._client = None

    def _genai(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client()
        return self._client

    def _gen_cfg(self):
        cfg = {"response_mime_type": "application/json",
               "response_schema": _SCHEMA,
               "max_output_tokens": self.max_output_tokens,
               "seed": config.API_SEED}
        if config.TEMPERATURE is not None:
            cfg["temperature"] = config.TEMPERATURE
        return cfg

    def _load_node_records(self, node):
        p = config.node_file(self.model, node)
        if not p.exists():
            return []
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

    async def _run_batch(self, node, pending):
        """Submit one batch for `pending` records, poll to completion, and return
        {key: (text, usage, reason)}."""
        from google.genai import types
        client = self._genai()
        gen_cfg = self._gen_cfg()

        jsonl = config.model_dir(self.model) / f"batch_node{node}_input.jsonl"
        with open(jsonl, "w") as f:
            for rec in pending:
                f.write(json.dumps({"key": rec["id"], "request": {
                    "contents": [{"parts": [{"text": rec["user_prompt"]}], "role": "user"}],
                    "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                    "generation_config": gen_cfg}}) + "\n")

        uploaded = client.files.upload(
            file=str(jsonl),
            config=types.UploadFileConfig(display_name=f"{self.model}-node{node}",
                                          mime_type="jsonl"))
        job = client.batches.create(model=self.api_model, src=uploaded.name,
                                    config={"display_name": f"{self.model}-node{node}"})
        (config.model_dir(self.model) / f"batch_node{node}_job.json").write_text(json.dumps(
            {"model": self.model, "node": node, "job_name": job.name,
             "input_file": uploaded.name, "n_requests": len(pending),
             "created": datetime.now(timezone.utc).isoformat()}, indent=2))
        print(f"  submitted node {node} batch: {job.name}  "
              f"({len(pending)} prompts, state {_state(job)})")

        while _state(job) not in COMPLETED_STATES:
            print(f"    {datetime.now().strftime('%H:%M:%S')} node {node} "
                  f"state={_state(job)} (waiting {self.poll_interval}s)")
            await asyncio.sleep(self.poll_interval)
            job = client.batches.get(name=job.name)
        st = _state(job)
        if st != "JOB_STATE_SUCCEEDED":
            raise SystemExit(f"node {node} batch did not succeed ({st}). "
                             f"error: {getattr(job, 'error', None)}")

        dest = getattr(job, "dest", None)
        fname = getattr(dest, "file_name", None) if dest else None
        lines = []
        if fname:
            raw = client.files.download(file=fname)
            text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            lines = [l for l in text.splitlines() if l.strip()]
        (config.model_dir(self.model) / f"batch_node{node}_results.jsonl").write_text(
            "\n".join(lines) + ("\n" if lines else ""))

        by_key = {}
        for l in lines:
            try:
                k, t, u, r = _parse_line(l)
            except Exception as e:
                print(f"    WARN: unparseable result line: {e}")
                continue
            if k is not None:
                by_key[k] = (t, u, r)
        return by_key

    async def generate_node(self, node):
        print(f"\n{'='*30}\nnode {node}  [{self.model}]  (batch)\n{'='*30}")
        records = self.generate_prompts(node)             # advances RNG + fills self.seen
        config.node_dir(self.model, node).mkdir(parents=True, exist_ok=True)

        existing = self._load_node_records(node)
        answered = {r["id"] for r in existing}
        pending = [r for r in records if r["id"] not in answered]
        if existing:
            print(f"  {len(answered)} already collected; {len(pending)} remaining to batch")

        new_results, new_hall, new_over, new_tok, still = [], [], [], [], []
        if pending:
            by_key = await self._run_batch(node, pending)
            inv = {pid: ay for ay, pid in self.seen.items()}
            nc_tmp = defaultdict(int)
            for rec in pending:
                if rec["id"] not in by_key:
                    still.append((rec["id"], "no result returned"))
                    continue
                t, u, reason = by_key[rec["id"]]
                if not self._record_paper(rec, t, u, inv, node, new_results,
                                          nc_tmp, new_hall, new_over, new_tok):
                    still.append((rec["id"], reason or "no usable response"))

        # advance the pool with EXISTING papers (recorded in a prior process);
        # newly recorded ones were already appended by _record_paper.
        for r in existing:
            self.pooled.append({"author": r["author"], "year": r["year"],
                                "title": r.get("title"), "abstract": r.get("abstract")})

        merged = existing + new_results
        with open(config.node_file(self.model, node), "w") as f:
            for r in merged:
                f.write(json.dumps(r) + "\n")

        node_citations = Counter()
        for r in merged:
            for cid in r.get("citation_ids", []):
                node_citations[cid] += 1
        with open(config.node_dir(self.model, node) / f"node_{node}_stats.jsonl", "w") as f:
            for k, v in node_citations.items():
                f.write(json.dumps({k: v}) + "\n")
        for k, v in node_citations.items():
            self.citation_counts[k] += v
        for name, items in (("hallucinations.jsonl", new_hall),
                            ("over_cap_violations.jsonl", new_over)):
            with open(config.citation_counts_dir(self.model) / name, "a") as f:
                for it in items:
                    f.write(json.dumps(it) + "\n")
        self.pooled.sort(key=lambda p: p["author"])
        tot = {"node": node,
               "total_prompt_tokens": sum(r.get("prompt_tokens", 0) for r in merged),
               "total_completion_tokens": sum(r.get("completion_tokens", 0) for r in merged),
               "total_tokens": sum(r.get("total_tokens", 0) for r in merged)}

        if still:
            print(f"\n  node {node}: {len(still)} prompt(s) still unanswered:")
            for pid, reason in still:
                print(f"      {pid}: {reason}")
            raise SystemExit(
                f"node {node} incomplete ({len(merged)}/{len(records)}); stopping so the "
                f"recursion does not build on a partial pool. Re-run the SAME command: "
                f"finished nodes are cached and this node retries only its missing prompts.")

        with open(config.node_dir(self.model, node) / f"node_{node}_token_totals.jsonl", "w") as f:
            f.write(json.dumps(tot) + "\n")
        print(f"  node {node} complete ({len(merged)}/{len(records)})")
        return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=config.list_models())
    ap.add_argument("--nodes", type=int, default=config.TOTAL_NODES,
                    help=f"number of nodes to run (default {config.TOTAL_NODES})")
    ap.add_argument("--write-root-kv", action="store_true",
                    help="overwrite the canonical root kv_pairs.jsonl with this run's map")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore completed nodes and regenerate from node 0")
    ap.add_argument("--poll-interval", type=int, default=60,
                    help="seconds between status polls while a node's batch runs (default 60)")
    ap.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS,
                    help=f"max output tokens per request (default {MAX_OUTPUT_TOKENS})")
    args = ap.parse_args()
    exp = BatchExperiment(args.model, args.poll_interval, args.max_output_tokens)
    asyncio.run(exp.run(args.nodes, args.write_root_kv, args.fresh))


if __name__ == "__main__":
    main()
