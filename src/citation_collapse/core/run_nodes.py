#!/usr/bin/env python3
"""
Unified recursive generation harness for ALL vendors.

    python run_nodes.py --model gpt-5-mini
    python run_nodes.py --model claude-haiku-4-5 --nodes 12
    python run_nodes.py --model gemini-2.5-pro --nodes 1      # node-0 only

One harness, one code path. The vendor (OpenAI / Anthropic / Google) is looked
up from config.MODEL_REGISTRY and dispatched through vendors.complete(); there
are no per-vendor scripts. Output goes to runs/<model>/.

Determinism: the RNG call sequence is preserved exactly (seed 42), so the
(author, year) -> id assignment is identical across models and matches the
canonical kv_pairs.jsonl. Only the written bodies and the shown sets past node 0
differ between models.
"""

import argparse
import asyncio
import json
import random
import re
from collections import defaultdict

from faker import Faker

from citation_collapse.core import config
from citation_collapse.core.citation_parser import find_cited_candidates, extract_citations_from_body
from citation_collapse.core.vendors import complete

# Load API keys from a .env in the working directory if present (matches the
# original harness, which used python-dotenv). Harmless if no .env / no dotenv.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---- shared constants (from config) ----------------------------------------
NODE_SIZE = config.NODE_SIZE
STRATUM_SIZE = config.STRATUM_SIZE
CITATION_CAP = config.CITATION_CAP
PAPER_SET_LENGTH = config.PAPER_SET_LENGTH
SEED = config.SEED
TOPIC = config.TOPIC

_SIMPLE_NAME = re.compile(r"^[A-Z][a-z]+$")

SYSTEM_PROMPT = f"""
    You are a researcher writing about a topic using a provided set of articles.
    Output valid JSON only:
    {{
        "title": "...",
        "abstract": "... no citations ...",
        "body": "... prose with inline citations ..."
    }}

    Citation format (follow exactly):
    - Cite each article inline as (Surname, Year) -- for example, (Smith, 2019).
    - Every article has exactly ONE author surname. Never write "et al.", initials,
      first names, or more than one name in a citation.
    - Do NOT use square brackets, numbered citations such as [1], footnotes, or a
      reference/bibliography list. Citations appear only inline, in the body.
    - Copy the surname and the year exactly as they appear in the article list.
    - The abstract must contain no citations.

    Citation limit (strict):
    - You may cite AT MOST {CITATION_CAP} distinct articles in the whole body.
    - Do NOT cite more than {CITATION_CAP} distinct articles under any circumstances.
      If more than {CITATION_CAP} seem relevant, choose only the {CITATION_CAP} most relevant and cite no others.
    - Only cite articles from the provided list; never cite anything not in the list.
    """


class Experiment:
    """Holds all run state for one model. Keeps the RNG stream identical to the
    original harness so kv_pairs stays compatible across models."""

    def __init__(self, model, max_concurrent):
        self.model = model
        spec = config.resolve_model(model)
        self.vendor = spec["vendor"]
        self.api_model = spec["api_model"]
        self.sem = asyncio.Semaphore(max_concurrent)

        self.pooled = []                       # candidate papers (author/year/title/abstract)
        self.seen = {}                         # (author, year) -> id   == kv_pairs
        self.possible_ids = set()
        self.citation_counts = defaultdict(int)
        self.fake = Faker()
        Faker.seed(SEED)

        for d in (config.prompts_dir(model), config.seed_dir(model),
                  config.master_dir(model), config.citation_counts_dir(model)):
            d.mkdir(parents=True, exist_ok=True)

    # ---- RNG-faithful helpers (do not reorder draws) ----
    def simple_surname(self):
        name = self.fake.last_name()
        while not _SIMPLE_NAME.match(name):
            name = self.fake.last_name()
        return name

    def create_set(self):
        return random.sample(self.pooled, PAPER_SET_LENGTH)

    def standardize_seed(self):
        """Read the CANONICAL standardized seed (seed.jsonl) for content + ids,
        and REPLAY the original fake-name RNG draws so the RNG state matches what
        the first run reached before node 0 (this is what keeps every model and
        the canonical kv_pairs matched). The bucket is no longer read here.
        """
        if not config.SEED_STANDARDIZED.exists():
            raise SystemExit(
                f"Canonical seed not found: {config.SEED_STANDARDIZED}\n"
                f"Place your seed.jsonl at the repo root (see KV_PAIRS_README.md / README)."
            )
        papers = [json.loads(l) for l in open(config.SEED_STANDARDIZED)]

        ret, mismatches = [], 0
        for paper in papers:
            pid = paper["id"]                       # canonical SEED_i from the file
            # replay the same draws standardize() made -> advances RNG identically
            surname = self.simple_surname()
            year = random.randint(2017, 2022)
            while (surname, year) in self.seen:
                surname = self.simple_surname()
                year = random.randint(2017, 2022)
            # guard: replayed labels should equal the pinned ones (else Faker drift)
            if (paper.get("author"), paper.get("year")) != (surname, year):
                mismatches += 1
            self.seen[(surname, year)] = pid
            self.possible_ids.add(pid)
            ret.append({"author": surname, "year": year,
                        "title": paper.get("title"), "abstract": paper.get("abstract")})

        if mismatches:
            print(f"  WARNING: {mismatches}/{len(papers)} seed (author,year) replayed "
                  f"differently from seed.jsonl. Faker version differs from the original "
                  f"run; the RNG state (and thus generated-paper ids / kv_pairs) may not "
                  f"match your existing runs. Pin faker to the original version to fix.")

        # copy canonical seed + initial counts into this model's run for self-containment
        config.seed_dir(self.model).mkdir(parents=True, exist_ok=True)
        (config.seed_dir(self.model) / "seed.jsonl").write_text(config.SEED_STANDARDIZED.read_text())
        if config.SEED_INITIAL.exists():
            (config.seed_dir(self.model) / "seed_initial.jsonl").write_text(config.SEED_INITIAL.read_text())
        return ret

    def generate_prompts(self, node, write=True):
        pos = set(random.sample(range(NODE_SIZE - 1), STRATUM_SIZE))
        records = []
        for i in range(NODE_SIZE):
            pid = f"N{node}P{i}"
            random_papers = self.create_set()
            surname = self.simple_surname()
            year = random.randint(2017, 2025)
            while (surname, year) in self.seen:
                surname = self.simple_surname()
                year = random.randint(2017, 2025)
            self.seen[(surname, year)] = pid

            shown_ids = [self.seen[(p["author"], p["year"])]
                         for p in random_papers if (p["author"], p["year"]) in self.seen]

            kind = "position paper" if i in pos else "literature review"
            verb = ("argue a position on" if kind == "position paper"
                    else "synthesize what is known about")
            user = (f"Using only the articles provided, {verb} the following topic.\n"
                    f"Support your claims by citing relevant articles inline, written as (Surname, Year).\n"
                    f"Cite only articles from the provided list, copying each author surname and year exactly.\n"
                    f'Do not write "et al.", initials, square brackets, or numbered citations.\n'
                    f"IMPORTANT: cite AT MOST {CITATION_CAP} distinct articles in total -- never more than {CITATION_CAP}.\n\n"
                    f"Topic: {TOPIC}\n\nArticles:\n{random_papers}\n")

            records.append({
                "id": pid, "type": kind, "author": surname, "year": year,
                "papers_seen_id": shown_ids, "user_prompt": user,
            })

        if write:
            path = config.prompts_dir(self.model) / f"N_{node}_inputs.jsonl"
            with open(path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
        return records

    async def _one_paper(self, rec):
        async with self.sem:
            text, usage = await complete(self.vendor, self.api_model,
                                         SYSTEM_PROMPT, rec["user_prompt"])
            return rec, text, usage

    def _record_paper(self, rec, text, usage, inv, node,
                      results, node_citations, hallucinations, over_cap, token_records):
        """Parse and record one model response. Returns True on success, or False
        if the call failed / produced unusable output (so the prompt is retried)."""
        pid = rec["id"]
        if not text:
            return False
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return False

        title = (obj.get("title") or "").strip()
        abstract = (obj.get("abstract") or "").strip()
        body = (obj.get("body") or "").strip()

        shown_ids = rec.get("papers_seen_id", []) or []
        shown_cands = [(inv[i][0], inv[i][1], i) for i in shown_ids if i in inv]

        # authoritative, shown-restricted, ordered citation ids
        matched = find_cited_candidates(body, shown_cands)
        matched_pairs = [(m[0], int(m[1])) for m in matched]
        cited_ids = [m[2] for m in matched]
        if len(cited_ids) > CITATION_CAP:
            over_cap.append({"paper_id": pid, "node": node,
                             "original_count": len(cited_ids),
                             "original_citation_ids": list(cited_ids)})
            cited_ids = cited_ids[:CITATION_CAP]

        # raw apparent citations: robust shown matches UNION any other apparent
        # pair (hallucinated / not-shown). Guarantees raw >= valid for every
        # citation style, while still capturing hallucinations for QC.
        apparent = extract_citations_from_body(body)
        seen_pairs = set(matched_pairs)
        raw_pairs = matched_pairs + [(a, int(y)) for (a, y) in apparent
                                     if (a, int(y)) not in seen_pairs]
        for a, y in apparent:
            sid = self.seen.get((a, int(y)))
            if sid is None:
                hallucinations.append({pid: [a, y]})
            elif sid not in set(shown_ids):
                hallucinations.append({pid: [a, y], "reason": "not_shown"})

        for cid in cited_ids:
            node_citations[cid] += 1

        u = usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        results.append({
            "id": pid, "author": rec["author"], "year": rec["year"],
            "type": rec["type"], "title": title, "abstract": abstract, "body": body,
            "papers_seen_id": shown_ids,
            "citations": [[a, y] for (a, y) in raw_pairs],
            "citation_ids": cited_ids,
            "n_citations_raw": len(matched),
            "prompt_tokens": u["prompt_tokens"],
            "completion_tokens": u["completion_tokens"],
            "total_tokens": u["total_tokens"],
        })
        token_records.append(u)
        self.pooled.append({"author": rec["author"], "year": rec["year"],
                            "title": title, "abstract": abstract})
        print(f"  {pid}: {len(cited_ids)} cites")
        return True

    async def generate_node(self, node):
        print(f"\n{'='*30}\nnode {node}  [{self.model}]\n{'='*30}")
        self.generate_prompts(node)
        config.node_dir(self.model, node).mkdir(parents=True, exist_ok=True)

        prompts = []
        with open(config.prompts_dir(self.model) / f"N_{node}_inputs.jsonl") as f:
            for line in f:
                prompts.append(json.loads(line))

        inv = {pid: ay for ay, pid in self.seen.items()}   # id -> (author, year)
        node_citations = defaultdict(int)
        hallucinations, over_cap, token_records, results = [], [], [], []
        node_path = config.node_file(self.model, node)

        # Retry loop: re-attempt ONLY the prompts that failed, pass after pass, until
        # every prompt in the node is answered (or no progress is made for
        # STUCK_LIMIT consecutive passes). Each call already gets one internal retry
        # inside complete(); this loop handles calls that still come back empty.
        pending = list(prompts)
        no_progress = 0
        for attempt in range(config.MAX_NODE_PASSES):
            if not pending:
                break
            before = len(pending)
            tasks = [self._one_paper(r) for r in pending]
            still = []
            for coro in asyncio.as_completed(tasks):
                rec, text, usage = await coro
                if not self._record_paper(rec, text, usage, inv, node, results,
                                          node_citations, hallucinations,
                                          over_cap, token_records):
                    still.append(rec)
            pending = still
            if not pending:
                break
            no_progress = no_progress + 1 if len(pending) == before else 0
            if no_progress >= config.STUCK_LIMIT:
                print(f"  node {node}: no progress for {config.STUCK_LIMIT} passes; "
                      f"stopping with {len(pending)}/{len(prompts)} unanswered")
                break
            wait = min(config.NODE_RETRY_BACKOFF * (attempt + 1), 120)
            print(f"  node {node}: {len(pending)}/{len(prompts)} prompt(s) unanswered "
                  f"after pass {attempt + 1}; retrying in {wait}s")
            await asyncio.sleep(wait)

        if pending:
            # Save what we have for inspection, but do NOT finalize the node (no side
            # files, no completion marker), then stop. Re-running the same command
            # resumes: earlier nodes are cached and this node is retried from scratch.
            with open(node_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            raise SystemExit(
                f"\nnode {node} [{self.model}]: only {len(results)}/{len(prompts)} "
                f"prompts answered after {config.MAX_NODE_PASSES} passes "
                f"(unanswered: {[r['id'] for r in pending]}).\n"
                f"No completion marker was written, so re-running the SAME command "
                f"retries this node (earlier nodes are cached).\n"
                f"If it keeps failing, check API keys / rate limits, or raise "
                f"MAX_NODE_PASSES / STUCK_LIMIT in config.py.")

        # node complete -> write node file (completion order across passes)
        with open(node_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        # per-node side files (written only once the node is complete)
        with open(config.node_dir(self.model, node) / f"node_{node}_stats.jsonl", "w") as f:
            for k, v in node_citations.items():
                f.write(json.dumps({k: v}) + "\n")
        for name, items in (("hallucinations.jsonl", hallucinations),
                            ("over_cap_violations.jsonl", over_cap)):
            with open(config.citation_counts_dir(self.model) / name, "a") as f:
                for it in items:
                    f.write(json.dumps(it) + "\n")

        for k, v in node_citations.items():
            self.citation_counts[k] += v
        self.pooled.sort(key=lambda p: p["author"])   # RNG-faithful pool ordering
        tot = {"node": node,
               "total_prompt_tokens": sum(t["prompt_tokens"] for t in token_records),
               "total_completion_tokens": sum(t["completion_tokens"] for t in token_records),
               "total_tokens": sum(t["total_tokens"] for t in token_records)}
        # completion marker (its existence marks node as fully done -> resume skips it)
        with open(config.node_dir(self.model, node) / f"node_{node}_token_totals.jsonl", "w") as f:
            f.write(json.dumps(tot) + "\n")
        return tot

    # ---- resume support ----
    def adopt_existing(self):
        """One-time: write completion markers for already-present node files (e.g.
        runs made by the old harness) so resume treats them as done. Walks node_0,
        node_1, ... contiguously and creates node_<k>_token_totals.jsonl where missing,
        filling totals from the per-paper token fields if available."""
        k, adopted = 0, []
        while config.node_file(self.model, k).exists():
            marker = config.node_dir(self.model, k) / f"node_{k}_token_totals.jsonl"
            if not marker.exists():
                recs = [json.loads(l) for l in open(config.node_file(self.model, k))]
                tot = {"node": k,
                       "total_prompt_tokens": sum(r.get("prompt_tokens", 0) for r in recs),
                       "total_completion_tokens": sum(r.get("completion_tokens", 0) for r in recs),
                       "total_tokens": sum(r.get("total_tokens", 0) for r in recs)}
                marker.write_text(json.dumps(tot) + "\n")
                adopted.append(k)
            k += 1
        if adopted:
            print(f"Adopted {len(adopted)} existing node(s) {adopted} for {self.model}: "
                  f"wrote completion markers. A later run will resume after node {max(adopted)}.\n"
                  f"NOTE: citation_ids in adopted nodes were parsed by whatever harness made them; "
                  f"re-run utils/update_node0_citations.py if you want them re-parsed consistently.")
        else:
            print(f"Nothing to adopt for {self.model} (markers already present, or no node files).")
        return adopted

    def _completed_nodes(self):
        """Contiguous nodes 0..L that finished (have a token_totals marker)."""
        done, k = [], 0
        while (config.node_dir(self.model, k) / f"node_{k}_token_totals.jsonl").exists():
            done.append(k)
            k += 1
        return done

    def _replay_node(self, k):
        """Rebuild RNG state, seen-map, pool and citation_counts for an already
        finished node WITHOUT calling the API, so a later run continues identically.
        """
        self.generate_prompts(k, write=False)        # advances RNG + registers N{k}Pi in seen
        drift = 0
        for line in open(config.node_file(self.model, k)):
            r = json.loads(line)
            if self.seen.get((r["author"], r["year"])) != r["id"]:
                drift += 1
            self.pooled.append({"author": r["author"], "year": r["year"],
                                "title": r.get("title"), "abstract": r.get("abstract")})
            for cid in r.get("citation_ids", []) or []:
                self.citation_counts[cid] += 1
        self.pooled.sort(key=lambda p: p["author"])
        if drift:
            print(f"  WARNING: node {k} replay: {drift} author/year mismatches vs the saved "
                  f"node file (faker drift). Resume would be inconsistent -- rerun with --fresh "
                  f"or pin faker to the original version.")
        marker = config.node_dir(self.model, k) / f"node_{k}_token_totals.jsonl"
        try:
            return json.loads(open(marker).readline())
        except Exception:
            return {"node": k, "total_prompt_tokens": 0,
                    "total_completion_tokens": 0, "total_tokens": 0}

    async def run(self, nodes, write_root_kv, fresh=False):
        random.seed(SEED)
        self.pooled.extend(self.standardize_seed())

        completed = set() if fresh else set(self._completed_nodes())
        if not completed:   # fresh start (or nothing done): truncate append-mode QC files
            for name in ("hallucinations.jsonl", "over_cap_violations.jsonl"):
                open(config.citation_counts_dir(self.model) / name, "w").close()
        if completed:
            print(f"Found {len(completed)} completed node(s) {sorted(completed)}; "
                  f"replaying their state and resuming at node {max(completed) + 1}.")

        totals = []
        for n in range(nodes):
            if n in completed:
                print(f"node {n}  [{self.model}]: already done -> replaying state (no API calls)")
                totals.append(self._replay_node(n))
            else:
                totals.append(await self.generate_node(n))

        with open(config.citation_counts_dir(self.model) / "citation_counts.jsonl", "w") as f:
            for k, v in self.citation_counts.items():
                f.write(json.dumps({k: v}) + "\n")
        kv_model = config.master_dir(self.model) / "kv_pairs.jsonl"
        with open(kv_model, "w") as f:
            for (a, y), pid in self.seen.items():
                f.write(json.dumps({"author": a, "year": y, "id": pid}) + "\n")
        with open(config.master_dir(self.model) / "token_usage.jsonl", "w") as f:
            for t in totals:
                f.write(json.dumps(t) + "\n")

        # refresh the canonical root kv_pairs (model-independent map)
        if write_root_kv or not config.KV_PAIRS.exists():
            config.KV_PAIRS.write_text(kv_model.read_text())
            print(f"wrote canonical {config.KV_PAIRS}")
        print(f"\nDone. Output in {config.model_dir(self.model)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=config.list_models())
    ap.add_argument("--nodes", type=int, default=config.TOTAL_NODES)
    ap.add_argument("--max-concurrent", type=int, default=config.MAX_CONCURRENT)
    ap.add_argument("--write-root-kv", action="store_true",
                    help="overwrite the canonical root kv_pairs.jsonl with this run's map")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore completed nodes and regenerate from node 0")
    ap.add_argument("--adopt", action="store_true",
                    help="write completion markers for existing node files (old runs), then exit")
    args = ap.parse_args()
    exp = Experiment(args.model, args.max_concurrent)
    if args.adopt:
        exp.adopt_existing()
        return
    asyncio.run(exp.run(args.nodes, args.write_root_kv, args.fresh))


if __name__ == "__main__":
    main()
