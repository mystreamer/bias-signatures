"""Generate bias signatures: partition-pair JSDs from labelled response files.

Reads data/responses/{experiment}/{model}.csv (labelled), writes one
long-format file per experiment to data/signatures/:

  e1  family moral_bias (moral vs non-moral, with mean stances) and mvb
      (MF x MF), per model x domain, permutation-tested, Bonferroni within
      each (model, channel, family, scheme) signature
  e2  per-topic perturbation biases (baseline vs each perturbed cell),
      descriptive only, at wide (all instructions) and 8sa scope
  e3  mvb per model x questionnaire, permutation-tested

`scheme` is raw (source labels) or canonical (pooled onto the canonical five
per §5.2 — responses are pooled before distributions are re-estimated).
"""

from __future__ import annotations

import argparse
import hashlib
import json

import pandas as pd

import stats
from constructs import (
    CANONICAL_FIVE,
    INVALID,
    MFQ2_SUPPORT,
    MFQ30_SUPPORT,
    REFUSAL,
    STANCE_SUPPORT,
    canonical,
    parse_foundation,
)
from llm import DATA_DIR

RESPONSES_DIR = DATA_DIR / "responses"
SIGNATURES_DIR = DATA_DIR / "signatures"

N_PERMUTATIONS = 10_000

E2_COMPARISONS = {
    "primed": ("primed", "neutral"),
    "pro-framed": ("none", "pro"),
    "con-framed": ("none", "con"),
    "grounded": ("grounded", "neutral"),
    "abstracted": ("abstracted", "neutral"),
}

EIGHT_SA = "Write an 8-sentence argument about X."


def _seed(*parts) -> int:
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


MODELS: frozenset[str] | None = None  # optional subset, set via --models


def load_responses(experiment: str) -> dict[str, pd.DataFrame]:
    files = sorted((RESPONSES_DIR / experiment).glob("*.csv"))
    if MODELS is not None:
        missing = MODELS - {f.stem for f in files}
        assert not missing, f"no response files under responses/{experiment}/ for: {', '.join(sorted(missing))}"
        files = [f for f in files if f.stem in MODELS]
    assert files, f"no labelled response files under responses/{experiment}/"
    out = {}
    for f in files:
        df = pd.read_csv(f)
        assert "label" in df.columns, f"{f} is unlabelled — run response_distribution.py"
        df["label"] = df["label"].astype(str)
        out[f.stem] = df
    return out


def mf_lists(df: pd.DataFrame, scheme: str) -> pd.Series:
    """Per-row foundation lists under the given scheme."""
    def convert(raw: str) -> list[str]:
        foundations = [parse_foundation(x) for x in json.loads(raw)]
        if scheme == "canonical":
            foundations = [canonical(f) for f in foundations]
        return sorted({f.value for f in foundations if f is not None})

    return df["mf"].map(convert)


def tested_pair(labels_a, labels_b, support, seed) -> dict:
    jsd, p = stats.permutation_test(
        list(labels_a), list(labels_b), support, N_PERMUTATIONS, seed=seed
    )
    lo, hi = stats.jsd_bootstrap(list(labels_a), list(labels_b), support, seed=seed)
    return {"n_a": len(labels_a), "n_b": len(labels_b),
            "jsd": jsd, "jsd_lo": lo, "jsd_hi": hi, "p": p}


#: (channel, model) -> canonical partition label counts, for the response-level
#: bootstrap in make_correlation.py (persisted as label_counts.json)
LABEL_COUNTS: dict[tuple[str, str], dict] = {}


def mvb_rows(df: pd.DataFrame, support, scheme: str, **context) -> list[dict]:
    """MF x MF partition pairs over the morally relevant rows (multi-label:
    a row appears in every foundation partition its item carries)."""
    moral = df[df["morally_relevant"] == True]  # noqa: E712
    lists = mf_lists(moral, scheme)
    foundations = (
        [f.value for f in CANONICAL_FIVE]
        if scheme == "canonical"
        else sorted({f for row in lists for f in row})
    )
    parts = {
        f: moral[lists.map(lambda l, f=f: f in l)]["label"] for f in foundations
    }
    if scheme == "canonical" and "channel" in context and "model" in context:
        LABEL_COUNTS[(context["channel"], context["model"])] = {
            "support": list(support),
            "counts": {f: [int((labels == s).sum()) for s in support]
                       for f, labels in parts.items() if not labels.empty},
        }
    rows = []
    for i, f1 in enumerate(foundations):
        for f2 in foundations[i + 1 :]:
            a, b = parts[f1], parts[f2]
            if a.empty or b.empty:
                continue
            rows.append({
                **context, "family": "mvb", "scheme": scheme,
                "partition_a": f1, "partition_b": f2,
                **tested_pair(a, b, support, _seed(scheme, f1, f2, *context.values())),
            })
    return rows


def bonferroni_within(df: pd.DataFrame) -> pd.DataFrame:
    keys = [k for k in ("model", "channel", "family", "scheme") if k in df.columns]
    df["p_bonferroni"] = df.groupby(keys)["p"].transform(
        lambda p: (p * len(p)).clip(upper=1.0)
    )
    return df


# ---------------------------------------------------------------- experiments

def numeric_mean(labels: pd.Series) -> float:
    numeric = pd.to_numeric(labels, errors="coerce").dropna()
    return round(float(numeric.mean()), 3)


def signatures_e1() -> pd.DataFrame:
    rows = []
    for model, df in load_responses("e1").items():
        dropped = (df["label"] == INVALID).sum()
        if dropped:
            print(f"{model}: dropping {dropped} invalid stance labels")
        df = df[df["label"] != INVALID]
        for domain, d in df.groupby("domain"):
            context = {"model": model, "channel": domain}
            moral = d[d["morally_relevant"] == True]["label"]  # noqa: E712
            nonmoral = d[d["morally_relevant"] == False]["label"]  # noqa: E712
            rows.append({
                **context, "family": "moral_bias", "scheme": "raw",
                "partition_a": "moral", "partition_b": "nonmoral",
                "mean_a": numeric_mean(moral), "mean_b": numeric_mean(nonmoral),
                **tested_pair(moral, nonmoral, STANCE_SUPPORT, _seed("mb", model, domain)),
            })
            for scheme in ("raw", "canonical"):
                rows.extend(mvb_rows(d, STANCE_SUPPORT, scheme, **context))
    return bonferroni_within(pd.DataFrame(rows))


def signatures_e2() -> pd.DataFrame:
    rows = []
    for model, df in load_responses("e2").items():
        df = df[df["label"] != INVALID]
        scopes = {"wide": df, "8sa": df[df["prompt"].str.startswith(
            EIGHT_SA.replace(" X.", " ")
        )]}
        for scope, scoped in scopes.items():
            for topic, d in scoped.groupby("topic"):
                base = d[(d["perturbation"] == "none") & (d["framing"] == "neutral")]["label"]
                for name, (perturbation, framing) in E2_COMPARISONS.items():
                    perturbed = d[
                        (d["perturbation"] == perturbation) & (d["framing"] == framing)
                    ]["label"]
                    if base.empty or perturbed.empty:
                        continue
                    rows.append({
                        "model": model, "channel": d["domain"].iloc[0], "topic": topic,
                        "family": "perturbation", "scope": scope,
                        "partition_a": "baseline", "partition_b": name,
                        "n_a": len(base), "n_b": len(perturbed),
                        "jsd": stats.jsd_between(list(base), list(perturbed), STANCE_SUPPORT),
                    })
    return pd.DataFrame(rows)


def signatures_e3() -> pd.DataFrame:
    supports = {"mfq30": MFQ30_SUPPORT, "mfq2": MFQ2_SUPPORT}
    rows = []
    for model, df in load_responses("e3").items():
        kept = df[~df["label"].isin([REFUSAL, INVALID])]
        print(f"{model}: excluding {len(df) - len(kept)} refusal/invalid Likert responses")
        kept = kept[kept["control"] == False]  # noqa: E712
        for questionnaire, d in kept.groupby("questionnaire"):
            for scheme in ("raw", "canonical"):
                rows.extend(mvb_rows(
                    d, supports[questionnaire], scheme,
                    model=model, channel=questionnaire,
                ))
    return bonferroni_within(pd.DataFrame(rows))


BUILDERS = {"e1": signatures_e1, "e2": signatures_e2, "e3": signatures_e3}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="all", choices=[*BUILDERS, "all"])
    parser.add_argument("--models", help="comma-separated model names (default: all); "
                        "note: overwrites data/signatures/ with the subset")
    args = parser.parse_args()
    if args.models:
        MODELS = frozenset(args.models.split(","))
    SIGNATURES_DIR.mkdir(parents=True, exist_ok=True)
    for name in BUILDERS if args.experiment == "all" else [args.experiment]:
        df = BUILDERS[name]()
        out = SIGNATURES_DIR / f"{name}.csv"
        df.to_csv(out, index=False)
        print(f"{out.name}: {len(df)} signature rows")

    counts_file = SIGNATURES_DIR / "label_counts.json"
    merged = json.loads(counts_file.read_text()) if counts_file.exists() else {}
    merged.update({f"{channel}|{model}": v for (channel, model), v in LABEL_COUNTS.items()})
    counts_file.write_text(json.dumps(merged, indent=1))
    print(f"{counts_file.name}: {len(merged)} channel|model partition-count entries")
