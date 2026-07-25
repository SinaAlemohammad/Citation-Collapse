#!/usr/bin/env python3
"""
Re-extract citations for every node_0.jsonl in a repo.

What it does
------------
1. Walks --root and finds every  node_0.jsonl  (any folder / any depth).
2. Loads a candidate table  (author, year) -> id  from kv_pairs.jsonl
   (or seed.jsonl as a fallback) so it can fill in citation_ids.
3. For each record, re-reads `body` and detects which candidates it cites with
   the FORMAT-AGNOSTIC matcher in robust_citation_parser.py -- handling
   (Author, Year), Author (Year), [Author, Year], and [Author], [Year].
4. Overwrites that record's `citations` and `citation_ids`.
5. Backs up each original to <file>.bak, and prints a per-file report:
   records changed, records recovered from zero, and any long body that still
   parses to zero (a possible 5th format to inspect).

Run it
------
    # keep this file next to robust_citation_parser.py (e.g. at the repo root)
    python update_node0_citations.py --root . --dry-run     # preview, writes nothing
    python update_node0_citations.py --root .               # writes, with .bak backups

Options
    --root PATH   repo root to search (default: .)
    --kv PATH     explicit candidate table; otherwise auto-detected
    --dry-run     report only, do not modify files
    --no-backup   skip writing .bak files
"""

import argparse
import glob
import json
import os
import shutil

from citation_collapse.core.citation_parser import find_cited_candidates


# --------------------------------------------------------------------------- #
# readers
# --------------------------------------------------------------------------- #

def iter_jsonl(path):
    """Strict one-record-per-line reader. Used for node_0.jsonl."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _is_mapping(d):
    """A dict whose values are all scalars -> looks like {key: id}."""
    return bool(d) and all(not isinstance(v, (dict, list)) for v in d.values())


def load_table_objects(path):
    """Flexible reader for the candidate table: jsonl of records, a JSON array,
    or a {key: id} mapping. Returns a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    # try jsonl (each non-empty line a JSON value)
    try:
        objs = [json.loads(l) for l in text.splitlines() if l.strip()]
        if len(objs) == 1 and isinstance(objs[0], list):
            return objs[0]
        if len(objs) == 1 and isinstance(objs[0], dict) and _is_mapping(objs[0]):
            return [{"_key": k, "_id": v} for k, v in objs[0].items()]
        return objs
    except json.JSONDecodeError:
        pass
    # fallback: whole-file JSON
    obj = json.loads(text)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if _is_mapping(obj):
            return [{"_key": k, "_id": v} for k, v in obj.items()]
        return [v if isinstance(v, dict) else {"_key": k, "_id": v}
                for k, v in obj.items()]
    return []


# --------------------------------------------------------------------------- #
# candidate table:  (author, year) -> id
# --------------------------------------------------------------------------- #

def _get(d, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def load_candidates(path):
    pairs = {}
    for r in load_table_objects(path):
        if not isinstance(r, dict):
            continue
        author = _get(r, "author", "Author", "fake_author", "surname")
        year = _get(r, "year", "Year", "fake_year")
        pid = _get(r, "id", "paper_id", "pid", "seed_id", "_id")
        if (author is None or year is None) and "_key" in r:
            key = str(r["_key"]).replace("|", ",").replace("_", ",")
            if "," in key:
                a, _, y = key.partition(",")
                author = author or a.strip().strip("[]() ")
                y = "".join(ch for ch in y if ch.isdigit())
                year = year or (y or None)
        if author is None or year is None or pid is None:
            continue
        try:
            year = int(year)
        except (TypeError, ValueError):
            continue
        pairs[(str(author).strip(), year)] = pid
    candidates = [(a, y, i) for (a, y), i in pairs.items()]
    return candidates


def autodetect_table(root):
    kv = sorted(glob.glob(os.path.join(root, "**", "kv_pairs*.jsonl"),
                          recursive=True))
    if kv:
        return kv[0]
    seed = sorted(glob.glob(os.path.join(root, "**", "seed*.jsonl"),
                            recursive=True))
    return seed[0] if seed else None


# --------------------------------------------------------------------------- #
# citation field formatting (mirror whatever the clean records already use)
# --------------------------------------------------------------------------- #

def detect_citation_style(node0_files):
    for path in node0_files:
        for r in iter_jsonl(path):
            c = r.get("citations")
            if c:
                el = c[0]
                if isinstance(el, str):
                    return "str"
                if isinstance(el, (list, tuple)):
                    return "list"
                if isinstance(el, dict):
                    return "dict"
    return "list"


def format_citation(author, year, style):
    if style == "str":
        return f"{author}, {year}"
    if style == "dict":
        return {"author": author, "year": year}
    return [author, year]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def process_file(path, candidates, style, dry_run, backup, cap=10):
    rows = list(iter_jsonl(path))
    changed = recovered = 0
    suspicious = []
    for r in rows:
        body = r.get("body") or ""
        old_ids = r.get("citation_ids") or []
        matched = find_cited_candidates(body, candidates)  # ordered by appearance
        n_raw = len(matched)
        if cap and cap > 0:
            matched = matched[:cap]  # keep first `cap` by order of appearance
        new_ids = [m[2] for m in matched]
        new_cites = [format_citation(m[0], m[1], style) for m in matched]
        if new_ids != list(old_ids):
            changed += 1
            if len(old_ids) == 0 and len(new_ids) > 0:
                recovered += 1
        if len(new_ids) == 0 and len(body) >= 200:
            suspicious.append(r.get("id"))
        r["citations"] = new_cites
        r["citation_ids"] = new_ids
        r["n_citations_raw"] = n_raw

    if not dry_run:
        if backup:
            shutil.copyfile(path, path + ".bak")
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return len(rows), changed, recovered, suspicious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--kv", default=None, help="explicit candidate table path")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--cap", type=int, default=10,
                    help="keep first N citations by order of appearance "
                         "(default 10; use 0 for no cap)")
    args = ap.parse_args()

    table_path = args.kv or autodetect_table(args.root)
    if not table_path or not os.path.exists(table_path):
        raise SystemExit("No candidate table found. Pass one with --kv "
                         "(kv_pairs.jsonl or seed.jsonl).")
    candidates = load_candidates(table_path)
    print(f"candidate table: {table_path}")
    print(f"  loaded {len(candidates)} (author, year) -> id mappings")
    if candidates:
        a, y, i = candidates[0]
        print(f"  sample: ({a!r}, {y}) -> {i!r}")
    else:
        raise SystemExit("Candidate table parsed to 0 mappings -- check schema; "
                         "paste me the first line of that file.")

    node0_files = sorted(set(
        glob.glob(os.path.join(args.root, "**", "node_0.jsonl"), recursive=True)))
    if not node0_files:
        raise SystemExit("No node_0.jsonl found under --root.")
    print(f"\nfound {len(node0_files)} node_0.jsonl file(s)")

    style = detect_citation_style(node0_files)
    cap_msg = f"cap {args.cap}" if args.cap and args.cap > 0 else "no cap"
    print(f"citation element style: {style} | {cap_msg}"
          + ("  (DRY RUN -- nothing written)\n" if args.dry_run else "\n"))

    tot_changed = tot_recovered = 0
    for path in node0_files:
        n, changed, recovered, suspicious = process_file(
            path, candidates, style, args.dry_run, not args.no_backup, args.cap)
        tot_changed += changed
        tot_recovered += recovered
        parts = path.split(os.sep)
        arm = parts[-4] if len(parts) >= 4 else path
        msg = (f"  {arm:22s} {n:4d} records | {changed:4d} changed | "
               f"{recovered:4d} recovered from 0")
        if suspicious:
            msg += f" | {len(suspicious)} still-empty long: {suspicious[:5]}"
        print(msg)

    print(f"\ntotal: {tot_changed} records changed, {tot_recovered} recovered from zero.")
    if not args.dry_run:
        print("originals backed up as <file>.bak" if not args.no_backup
              else "no backups written (--no-backup).")


if __name__ == "__main__":
    main()
