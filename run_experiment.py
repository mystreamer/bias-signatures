"""Run a prompt template set against a model through the response cache.

Every (model, temperature, prompt) is served from data/cache/ when present and
topped up to n samples otherwise (ADR 0006), so interrupted runs resume for
free. Emits the long-format response file downstream scripts work on:
data/responses/{experiment}/{model}.csv — one row per (template, sample).
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

import llm

RESPONSES_DIR = llm.DATA_DIR / "responses"


def response_format_for(row) -> dict | None:
    """Under --structured, Likert rows (option_map present, i.e. e3) are
    provider-constrained to {"label": <option letter>}; all other rows call
    unconstrained."""
    option_map = getattr(row, "option_map", None)
    if not isinstance(option_map, str) or not option_map:
        return None
    letters = list(json.loads(option_map))
    return {"type": "json_schema", "json_schema": {
        "name": "likert_label", "strict": True,
        "schema": {"type": "object",
                   "properties": {"label": {"type": "string", "enum": letters}},
                   "required": ["label"], "additionalProperties": False}}}


def bust_cache(df: pd.DataFrame, model: str, temperature: float,
               rf_for=response_format_for) -> None:
    """Delete this model's cached entries for the template set's prompts."""
    paths = [llm.cache_path(model, row.prompt, temperature, rf_for(row))
             for row in df.itertuples()]
    busted = sum(p.exists() for p in paths)
    for p in paths:
        p.unlink(missing_ok=True)
    print(f"busted {busted} cached entries for {model}")


def run(template_set: Path, model: str, temperature: float, n: int,
        workers: int, limit: int | None, max_tokens: int, bust: bool = False,
        structured: bool = False) -> None:
    rf_for = response_format_for if structured else (lambda row: None)
    df = pd.read_csv(template_set)
    if limit:
        df = df.head(limit)
    if bust:
        bust_cache(df, model, temperature, rf_for)
    experiment = template_set.stem
    out = RESPONSES_DIR / experiment / f"{model}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    failures = []

    def sample(row) -> list[dict]:
        try:
            completions = llm.cached_completions(
                model, row.prompt, temperature, n, max_tokens=max_tokens,
                response_format=rf_for(row),
            )
        except Exception as exc:  # leave the row to the next run
            failures.append((row.template_id, repr(exc)))
            return []
        base = row._asdict()
        del base["Index"]
        return [{**base, "sample_idx": i, "response": c} for i, c in enumerate(completions)]

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(sample, df.itertuples()), 1):
            rows.extend(result)
            if i % 50 == 0 or i == len(df):
                print(f"  {i}/{len(df)} templates")

    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{out}: {len(rows)} responses ({len(failures)} templates failed)")
    for tid, err in failures[:5]:
        print(f"  failed: {tid}: {err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template_set", type=Path)
    parser.add_argument("--model", required=True, choices=list(llm.MODELS))
    parser.add_argument("--temperature", type=float, default=llm.EXPERIMENT_TEMPERATURE)
    parser.add_argument("--n", type=int, default=llm.N_SAMPLES)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="only the first K templates (smoke tests)")
    parser.add_argument("--max-tokens", type=int, default=llm.MAX_TOKENS)
    parser.add_argument("--bust-cache", action="store_true",
                        help="delete this model's cached responses for these prompts before sampling")
    parser.add_argument("--structured", action="store_true",
                        help='force {"label": <letter>} json_schema output on option_map rows (e3)')
    args = parser.parse_args()
    run(args.template_set, args.model, args.temperature, args.n,
        args.workers, args.limit, args.max_tokens, args.bust_cache, args.structured)
