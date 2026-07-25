"""
Comprehensive, format-agnostic citation extraction.

Models cite in many incompatible styles, and the style is a per-model habit, so
the parser must not assume any single delimiter convention. Observed styles:

    (Smith, 2019)                  parenthetical, comma
    (Smith 2019)                   parenthetical, no comma
    Smith (2019)                   narrative
    Smith et al. (2019)            narrative, et al.
    Smith and Jones (2020)         narrative, multi-author
    Smith & Jones (2020)           narrative, ampersand
    (Smith, 2019a)                 year suffix letter
    (Smith, 2019; Jones, 2020)     multiple in one group, ';'-separated
    ([Cameron, 2021]; [Conner, 2018])   bracketed pair list      (GPT-4.1)
    ([Lynch], [2021]; [Ramirez], [2017]) separately bracketed    (GPT-5)
    [Smith] [2019]                 separate brackets, no comma
    O'Neil / Garcia / van Berg     hyphen / apostrophe / particles in surname

Two entry points:

  find_cited_candidates(body, candidates)   <-- AUTHORITATIVE. Table-driven.
      Given the (author, year) pairs actually SHOWN for a prompt, it detects
      which ones the body cites by surname+year proximity, ignoring every
      bracket/paren/comma convention. Immune to prose false positives ("In
      2019, ...") because it only looks for known author+year pairs. Use this
      wherever the shown candidate set is available (the harness and the
      extractors both have it).

  extract_citations_from_body(body)         <-- FALLBACK. No candidate table.
      Returns ordered, de-duplicated (author, year) found by regex. Tolerant of
      all the styles above. Used only for hallucination QC (tokens that look
      like citations but map to no shown paper).
"""

import re
import unicodedata

# Surname: starts uppercase (incl. accented), may contain letters, hyphen,
# apostrophe; allows a few lowercase particle prefixes (van, de, der, von, ...).
_PARTICLE = r"(?:van|von|de|der|den|del|della|di|da|du|la|le|el|al|bin|ibn|st|st\.|mc|mac)"
_SURNAME = rf"(?:{_PARTICLE}\s+)?[A-Z\u00C0-\u024F][A-Za-z\u00C0-\u024F.'\u2019\-]+"
_YEAR = r"(?:1[5-9]\d{2}|20\d{2})"


# ---- table-driven (AUTHORITATIVE) -------------------------------------------

def find_cited_candidates(body, candidates, max_gap=8):
    """Detect which shown (author, year) candidates the body cites.

    A candidate counts as cited when its surname is followed by its specific year
    in a CITATION structure: surname, then an optional possessive and an optional
    "et al." / "and X" / "& X" clause, then only citation punctuation (spaces,
    commas, semicolons, parens, brackets -- never letters) up to `max_gap`, then
    the year (optionally with a disambiguation letter, e.g. 2019a). This matches
    every style the models use --
        (Smith, 2019)  Smith 2019  Smith (2019)  [Smith, 2019]  [Smith] [2019]
        ([Smith], [2019])  Smith et al. (2019)  Smith and Jones (2019)  Smith's 2019
    -- while ignoring prose like "In 2019, Smith proposed ..." (year before name,
    or words between name and year), because the surname must lead and only
    punctuation may sit between it and the year. Word boundaries stop 'Smith'
    matching 'Smithson' and '2019' matching '12019' / '20190'.

    candidates : iterable of (author, year) or (author, year, id, ...).
    Returns the matched candidates, ordered by first appearance in the body.
    """
    if not body:
        return []
    # optional "et al." or one-or-more " and X" / " & X" / ", X" author tails
    etal = r"(?:\s+et\s+al\.?|\s*(?:&|and)\s+[A-Z][A-Za-z.'\u2019\u00C0-\u024F\-]+)*"
    punct = rf"[\s,;:.\]\)\(\[]{{0,{max_gap}}}"   # citation punctuation only (no letters/digits)
    found = []
    for cand in candidates:
        author, year = cand[0], str(cand[1])
        a, y = re.escape(author), re.escape(year)
        pat = re.compile(rf"\b{a}\b(?:'s|\u2019s)?{etal}{punct}{y}[a-z]?\b")
        m = pat.search(body)
        if m:
            found.append((m.start(), cand))
    found.sort(key=lambda x: x[0])
    return [c for _, c in found]


# ---- regex fallback (body only) ---------------------------------------------

# Narrative: Author (Year) | Author et al. (Year) | Author and/& Author (Year)
_NARR = re.compile(
    rf"({_SURNAME})"
    rf"(?:\s+et\s+al\.?)?"
    rf"(?:\s+(?:and|&)\s+{_SURNAME})*"
    rf"\s*[\(\[]\s*({_YEAR})[a-z]?\s*[\)\]]"
)
# A parenthetical group that contains at least one year; parsed item-by-item.
_PAREN_GROUP = re.compile(r"[\(\[]([^()\[\]]*?(?:1[5-9]\d{2}|20\d{2})[a-z]?[^()\[\]]*)[\)\]]")
# One author+year item inside a group, any bracket/comma styling between them.
_PAREN_ITEM = re.compile(
    rf"({_SURNAME})"
    rf"(?:\s+et\s+al\.?)?"
    rf"(?:\s+(?:and|&)\s+{_SURNAME})*"
    rf"[\]\)]?\s*,?\s*[\[\(]?\s*({_YEAR})[a-z]?"
)


def _norm(s):
    return unicodedata.normalize("NFC", s).strip()


def extract_citations_from_body(body):
    """Ordered, de-duplicated list of (author, year) ints, tolerant of all
    known styles. Over-matching is harmless downstream because callers keep
    only pairs that map to a shown paper."""
    if not body:
        return []
    hits = []
    for m in _NARR.finditer(body):
        hits.append((m.start(), _norm(m.group(1)), int(m.group(2))))
    for g in _PAREN_GROUP.finditer(body):
        base = g.start()
        for sub in g.group(1).split(";"):
            for mm in _PAREN_ITEM.finditer(sub):
                hits.append((base, _norm(mm.group(1)), int(mm.group(2))))
    hits.sort(key=lambda h: h[0])
    out, seen = [], set()
    for _, a, y in hits:
        if (a, y) not in seen:
            seen.add((a, y))
            out.append((a, y))
    return out


# ---- mapping helper ---------------------------------------------------------

def map_to_ids(pairs, kv, shown_ids=None, cap=None):
    """Map (author, year) pairs -> ids via kv {(author,int year): id},
    optionally restricting to shown_ids, de-duping, and capping (order kept)."""
    shown = set(shown_ids) if shown_ids is not None else None
    ids, seen = [], set()
    for a, y in pairs:
        pid = kv.get((a, int(y)))
        if pid is None or (shown is not None and pid not in shown) or pid in seen:
            continue
        seen.add(pid)
        ids.append(pid)
        if cap is not None and len(ids) >= cap:
            break
    return ids


if __name__ == "__main__":
    cases = {
        "paren_comma":   "see (Smith, 2019) and (Jones, 2020).",
        "paren_nocomma": "as in (Smith 2019).",
        "narrative":     "Smith (2019) showed; Jones (2020) confirmed.",
        "etal":          "Smith et al. (2019) and Garcia and Lopez (2021).",
        "amp":           "Smith & Jones (2020) argued.",
        "suffix":        "(Smith, 2019a; Smith, 2019b).",
        "multi_group":   "(Smith, 2019; Jones, 2020; Lee, 2018).",
        "bracket_pair":  "([Cameron, 2021]; [Conner, 2018]); also ([Ross, 2022]).",
        "bracket_split": "([Lynch], [2021]; [Ramirez], [2017]) and ([Mayo], [2019]).",
        "sep_nocomma":   "[Smith] [2019] and [Jones] [2020].",
        "accent":        "(García, 2020) and (Müller, 2018).",
        "apostrophe":    "O'Neil (2019) and (D'Angelo, 2020).",
        "particle":      "(van Berg, 2019) and de Vries (2021).",
        "prose_trap":    "In 2019, methods improved; by 2021, more so.",  # -> none
    }
    print("regex fallback (author, year):")
    for k, t in cases.items():
        print(f"  {k:14s} -> {extract_citations_from_body(t)}")

    print("\ntable-driven (shown candidates only):")
    cands = [("Lynch", 2021), ("Ramirez", 2017), ("Mayo", 2019),
             ("Cameron", 2021), ("Smith", 2019), ("van Berg", 2019)]
    for k in ("bracket_split", "sep_nocomma", "particle"):
        print(f"  {k:14s} -> {find_cited_candidates(cases[k], cands)}")
