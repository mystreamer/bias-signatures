"""Run a prompt template set against a model through the response cache.

Every (model, temperature, prompt) is served from data/cache/ when present and
topped up to n samples otherwise (ADR 0006), so interrupted runs resume for
free. Emits the long-format response file downstream scripts work on:
data/responses/{experiment}/{model}.csv — one row per (template, sample).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

import llm

RESPONSES_DIR = llm.DATA_DIR / "responses"


def run(template_set: Path, model: str, temperature: float, n: int,
        workers: int, limit: int | None, max_tokens: int) -> None:
    df = pd.read_csv(template_set)
    if limit:
        df = df.head(limit)
    experiment = template_set.stem
    out = RESPONSES_DIR / experiment / f"{model}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    failures = []

    def sample(row) -> list[dict]:
        try:
            completions = llm.cached_completions(
                model, row.prompt, temperature, n, max_tokens=max_tokens
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
    args = parser.parse_args()
    run(args.template_set, args.model, args.temperature, args.n,
        args.workers, args.limit, args.max_tokens)
