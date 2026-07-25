"""Elicitation inconsistency: constrained self-labels vs. LLM-judge labels.

Compares each model's E3 tendency under the response_format-constrained run
(out/structured-run/, the model's own letter choice) against the unconstrained
rerun labelled by the likert-llm judge. Per (model, template): each of the
three resamples is binned to a tendency -- mfq30 (6-pt, no midpoint): low
(1-3) vs high (4-6); mfq2 (5-pt): low (1-2) / neutral (3) / high (4-5) --
then the bins are majority-voted (refusal/invalid dropped; ties and vote-less
items discarded and reported). An item is inconsistent when its bin differs
between the two conditions.
"""

# %%
import pandas as pd

import llm

STRUCTURED_DIR = llm.BASE_DIR / "out" / "structured-run"
RESPONSES_DIR = llm.DATA_DIR / "responses" / "e3"
BINS = {"low", "neutral", "high"}


def tendency(label, questionnaire) -> str | None:
    pos = pd.to_numeric(label, errors="coerce")  # refusal/invalid -> NaN
    if pd.isna(pos):
        return None
    if questionnaire == "mfq2":
        return "low" if pos < 3 else "neutral" if pos == 3 else "high"
    return "low" if pos <= 3 else "high"


def vote(bins: pd.Series) -> str:
    counts = bins.dropna().value_counts()
    if not len(counts):
        return "empty"
    if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
        return "tie"
    return counts.index[0]


def item_bins(path) -> pd.Series:
    """(template_id, questionnaire) -> majority tendency (or tie/empty)."""
    df = pd.read_csv(path)
    df["bin"] = [tendency(l, q) for l, q in zip(df.label, df.questionnaire)]
    return df.groupby(["template_id", "questionnaire"])["bin"].agg(vote)


# %%
rows, pairs = [], []
for m in llm.MODELS:
    pair = pd.concat([
        item_bins(STRUCTURED_DIR / f"responses_{m}.csv").rename("constrained"),
        item_bins(RESPONSES_DIR / f"{m}.csv").rename("judged"),
    ], axis=1).reset_index()
    for q, grp in pair.groupby("questionnaire"):
        tie = (grp.constrained == "tie") | (grp.judged == "tie")
        empty = (grp.constrained == "empty") | (grp.judged == "empty")
        kept = grp[grp.constrained.isin(BINS) & grp.judged.isin(BINS)]
        rows.append({
            "model": m, "questionnaire": q, "n": len(grp),
            "n_tie": int(tie.sum()), "n_empty": int((empty & ~tie).sum()),
            "p_inconsistent": (kept.constrained != kept.judged).mean(),
        })
        pairs.append(kept.assign(model=m))
report = pd.DataFrame(rows).set_index(["model", "questionnaire"]).round(3)
print(report.to_string())

# %%
# pooled bin-transition counts (constrained -> judged) per questionnaire
pooled = pd.concat(pairs, ignore_index=True)
for q, grp in pooled.groupby("questionnaire"):
    print(f"\n{q}")
    print(pd.crosstab(grp.constrained, grp.judged).to_string())

# %%
