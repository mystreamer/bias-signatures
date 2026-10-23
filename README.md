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

| Path | Status | Contents |
|---|---|---|
| `data/raw/manual/` | committed | MFQ-30 and MFQ-2 item lists; the author's moral foundation annotations of the IssueBench political issues (Label Studio exports). Manual artifacts, not re-derivable. |
| `data/raw/` (rest) | downloaded by `download_raw.py` | IssueBench (Hugging Face) and Social Chemistry 101. Not redistributed in raw form; both datasets remain subject to their own licenses, which also cover the derived files below. |
| `data/components/` | committed | Typed prompt components assembled and annotated from the raw sources. |
| `data/template_sets/` | committed | The realised prompts of experiments E1, E2 and E3 (3 CSV files, 3.2 MB). |
| `data/responses/` | committed | The model responses and their stance or Likert labels on which the thesis results are based (15 CSV files, 229 MB). |
| `data/signatures/` | committed | Derived JSD tables, label counts and Spearman correlations. |
| `data/cache/` | gitignored | Per-call API response cache (one JSON file per model, temperature and prompt). Source of `data/responses/`; kept locally. |

## Reproducing the thesis figures

The committed responses are the ones behind the thesis, so the analysis can be
re-run without any API access:

```sh
uv sync
uv run python make_signatures.py               # data/responses -> data/signatures
uv run python make_correlation.py              # -> data/signatures/correlations.csv
uv run python make_figures.py                  # -> out/figures/*.pdf, out/tables/*.tex
```

Re-running `run_experiment.py` instead requires API credentials and will not
yield byte-identical responses (sampling at t=0.3).

## Experiments

| | Template set | Elicitor | Signature |
|---|---|---|---|
| E1 (§6.2) | 8SA instruction × all issues, neutral framing | stance | moral bias + MVB per model × domain |
| E2 (§6.3) | argumentative instructions × 10 controversial topics × 6 perturbation cells | stance | per-topic perturbation biases (descriptive) |
| E3 (§6.4) | MFQ statements × option rotations | likert-llm | MVB per model × questionnaire |
