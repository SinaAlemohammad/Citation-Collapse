"""
Vendor dispatch layer.

The three vendors differ in exactly four ways: the client object, the call
signature, the usage field names, and how JSON is enforced. Everything else in
the experiment is identical, so all of that variation is isolated HERE. The rest
of the codebase calls one function:

    text, usage = await complete(vendor, api_model, system, user)

`text` is a JSON string (title/abstract/body) or None on failure. `usage` is
{"prompt_tokens", "completion_tokens", "total_tokens"}.

SDKs are imported lazily, so you only need the SDK for the vendor you run.
"""

import asyncio
import json
import os

from citation_collapse.core import config

_clients = {}   # vendor -> client (created once, reused)


def _strip_json_fence(t: str) -> str:
    """Remove ```json ... ``` fences some models wrap around JSON."""
    if not t:
        return t
    t = t.strip()
    if t.startswith("```"):
        t = t[3:]
        if t.lower().startswith("json"):
            t = t[4:]
        if "```" in t:
            t = t[: t.rindex("```")]
    return t.strip().strip("`").strip()


# ---- client factories (lazy) ------------------------------------------------

def _get_client(vendor: str):
    if vendor in _clients:
        return _clients[vendor]
    if vendor == "openai":
        from openai import AsyncOpenAI
        c = AsyncOpenAI(max_retries=4)          # reads OPENAI_API_KEY
    elif vendor == "anthropic":
        from anthropic import AsyncAnthropic
        c = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), max_retries=8)
    elif vendor == "google":
        from google import genai
        c = genai.Client()                       # reads GEMINI_API_KEY / GOOGLE_API_KEY
    else:
        raise ValueError(f"Unknown vendor '{vendor}'")
    _clients[vendor] = c
    return c


# Anthropic JSON is forced via a single-tool schema (guarantees valid JSON).
_ANTHROPIC_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "abstract": {"type": "string", "description": "no citations"},
        "body": {"type": "string",
                 "description": "support claims by citing relevant articles inline, e.g. (Surname, Year)"},
    },
    "required": ["title", "abstract", "body"],
}


# ---- per-vendor calls -------------------------------------------------------

async def _openai(api_model, system, user, max_tokens, seed, temperature, schema=None):
    c = _get_client("openai")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    kw = {}
    if seed is not None:
        kw["seed"] = seed                       # best-effort determinism
    if temperature is not None:
        kw["temperature"] = temperature         # NOTE: gpt-5/o-series reject non-default temperature
    resp = await c.chat.completions.create(
        model=api_model, messages=messages,
        max_completion_tokens=max_tokens,
        response_format={"type": "json_object"},
        **kw,
    )
    u = resp.usage
    usage = {"prompt_tokens": u.prompt_tokens if u else 0,
             "completion_tokens": u.completion_tokens if u else 0,
             "total_tokens": u.total_tokens if u else 0}
    return resp.choices[0].message.content, usage


async def _anthropic(api_model, system, user, max_tokens, seed, temperature, schema=None):
    c = _get_client("anthropic")
    kw = {}
    if temperature is not None:                 # Anthropic has no seed param
        kw["temperature"] = temperature
    resp = await c.messages.create(
        model=api_model, system=system,
        tools=[{"name": "json", "description": "Output valid JSON only",
                "input_schema": schema or _ANTHROPIC_SCHEMA}],
        tool_choice={"type": "tool", "name": "json"},
        messages=[{"role": "user", "content": user}],
        max_tokens=max_tokens,
        **kw,
    )
    u = resp.usage
    usage = {"prompt_tokens": u.input_tokens if u else 0,
             "completion_tokens": u.output_tokens if u else 0,
             "total_tokens": (u.input_tokens + u.output_tokens) if u else 0}
    content = next((json.dumps(b.input) for b in resp.content if b.type == "tool_use"), None)
    if content == "{}" or resp.stop_reason == "max_tokens":
        content = None
    return content, usage


async def _google(api_model, system, user, max_tokens, seed, temperature, schema=None):
    from google.genai import types
    c = _get_client("google")
    cfg = dict(system_instruction=system, response_mime_type="application/json",
               max_output_tokens=max_tokens)
    if schema is not None:
        cfg["response_schema"] = schema
    if seed is not None:
        cfg["seed"] = seed                      # Gemini supports a generation seed
    if temperature is not None:
        cfg["temperature"] = temperature
    resp = await c.aio.models.generate_content(
        model=api_model, contents=user,
        config=types.GenerateContentConfig(**cfg),
    )
    m = getattr(resp, "usage_metadata", None)
    usage = {"prompt_tokens": getattr(m, "prompt_token_count", 0) or 0,
             "completion_tokens": getattr(m, "candidates_token_count", 0) or 0,
             "total_tokens": getattr(m, "total_token_count", 0) or 0}
    return _strip_json_fence(resp.text), usage


_DISPATCH = {"openai": _openai, "anthropic": _anthropic, "google": _google}


async def complete(vendor, api_model, system, user,
                   max_tokens=5000, retry_tokens=10000, retry_sleep=5,
                   seed=None, temperature="__cfg__", schema=None):
    """Generate one paper. Returns (json_text|None, usage). Retries once with a
    larger token budget on an empty result. seed/temperature default to the
    values in config (API_SEED / TEMPERATURE) for best-effort reproducibility.

    schema overrides the JSON shape the vendor is forced to emit. Leave it None
    for the benchmark: that keeps the title/abstract/body contract above, which
    is what run_nodes expects. paraphrase_seed.py passes its own."""
    if seed is None:
        seed = config.API_SEED
    if temperature == "__cfg__":
        temperature = config.TEMPERATURE
    fn = _DISPATCH.get(vendor)
    if fn is None:
        raise ValueError(f"Unknown vendor '{vendor}'")
    try:
        content, usage = await fn(api_model, system, user, max_tokens, seed, temperature, schema)
        if content:
            return content, usage
        await asyncio.sleep(retry_sleep)
        content2, usage2 = await fn(api_model, system, user, retry_tokens, seed, temperature, schema)
        for k in usage:
            usage[k] += usage2.get(k, 0)
        return (content2 or None), usage
    except Exception as e:
        print(f"[api error] {vendor}/{api_model}: {e}")
        await asyncio.sleep(retry_sleep)
        return None, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
