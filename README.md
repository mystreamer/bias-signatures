# bias_signatures

Self-contained pipeline for the thesis *The Moral Consistency of Large Language
Models*: assemble typed prompt components from source datasets, build
partitioned prompt template sets, run them against LLMs, and analyse the
response distributions into bias signatures (JSD + permutation tests) and
consistency correlations (Spearman + bootstrap).

Vocabulary lives in [CONTEXT.md](CONTEXT.md); design decisions in
[docs/adr/](docs/adr/).

## Setup

```sh
uv sync
```

Credentials in `.env` (here or in the repo root): `HF_OPENAI_API_KEY`,
`HF_OPENAI_BASE_URL` (HuggingFace router; experiment models and the worker
model), optionally `OR_OPENROUTER_API_KEY`, `OR_OPENROUTER_BASE_URL`.

## Pipeline

```sh
uv run python download_raw.py                  # IssueBench + SC101 -> data/raw/ (gitignored)
uv run python assemble_components.py           # mechanical features -> data/components/
uv run python augment_components.py            # variants: AITA, priming, framings, rewrites (LLM)
uv run python annotate_components.py --generate                                        # LLM proposals
uv run python load_annotations.py             # human workload -> data/annotation_tasks/*.csv
uv run python annotate_components.py --import data/raw/manual/political_topic_annotations.csv
uv run python annotate_components.py --evaluate                                        # reify features
uv run python build_prompt_template_sets.py    # -> data/template_sets/{e1,e2,e3}.csv

for m in llama-3.1-8b llama-3.3-70b qwen-2.5-72b deepseek-r1 kimi-k2.5; do
  uv run python run_experiment.py data/template_sets/e1.csv --model $m
done                                           # -> data/responses/e1/{model}.csv (cached, resumable)

uv run python response_distribution.py data/responses/e1/llama-3.3-70b.csv --elicitor stance
uv run python response_distribution.py data/responses/e3/llama-3.3-70b.csv --elicitor likert-llm

uv run python make_signatures.py               # -> data/signatures/{e1,e2,e3}.csv
uv run python make_correlation.py              # -> data/signatures/correlations.csv
uv run python make_figures.py                  # -> out/figures/*.pdf, out/tables/*.tex
```

Every stage is idempotent: components merge on re-assembly (preserving
augmented variants and reified features), augmentation and elicitation skip
already-filled fields, and `run_experiment.py` serves every
(model, temperature, prompt) triple from `data/cache/` — reads capped at
exactly n samples, top-up on miss — so interrupted runs resume for free.

## Data

- **Committed** (`data/raw/manual/`): the MFQ-30/MFQ-2 item lists and the
  author's political-issue moral foundation annotations — manual artifacts
  that cannot be re-derived.
- **Downloaded** (rest of `data/raw/`): IssueBench (HF), Social Chemistry 101.
- **`data/cache/`** is gitignored but precious (paid API responses) — back it
  up out-of-band.

## Experiments

| | Template set | Elicitor | Signature |
|---|---|---|---|
| E1 (§5.2) | 8SA instruction × all issues, neutral framing | stance | moral bias + MVB per model × domain |
| E2 (§5.3) | argumentative instructions × 10 controversial topics × 6 perturbation cells | stance | per-topic perturbation biases (descriptive) |
| E3 (§5.4) | MFQ statements × option rotations | likert-llm | MVB per model × questionnaire |
