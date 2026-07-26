"""Inter-annotator agreement for the political-issue MF annotations.

Compares the author's annotations (political_topic_annotations.csv,
annotator_1) with the two external annotators from the Label Studio export.
Each issue has exactly two raters -- the author plus one fixed external
annotator -- so agreement is Cohen's kappa per pair; Fleiss' kappa would
assume raters drawn interchangeably from a pool, which the fixed author
rater violates. The multi-label MF sets are binarised: moral relevance
(any foundation vs none) and per-foundation presence/absence.
"""

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # bias_signatures/ on path

import json

import pandas as pd

import llm
from constructs import parse_foundation

MANUAL_DIR = llm.DATA_DIR / "raw" / "manual"
EXPORT = MANUAL_DIR / "project-24-at-2026-07-25-18-07-ac66a69e-3.csv"


def parse_labels(cell) -> frozenset[str]:
    """Label cell -> foundation values. Handles the author CSV (comma-
    separated), the Label Studio export (bare label or {"choices": [...]}),
    and empty cells (morally irrelevant)."""
    if pd.isna(cell):
        return frozenset()
    text = str(cell)
    labels = json.loads(text)["choices"] if text.startswith("{") else text.split(",")
    return frozenset(parse_foundation(l).value for l in labels if l.strip())


mine = pd.read_csv(MANUAL_DIR / "political_topic_annotations.csv")
mine = mine.set_index("term")["annotator_1"].map(parse_labels)

theirs = pd.read_csv(EXPORT)
theirs = theirs[theirs["id"].str.startswith("pol-")]  # drop the SC blinds
theirs = theirs.assign(mf=theirs["mf"].map(parse_labels))


# %%
def cohen_kappa(a, b) -> float:
    """Cohen's kappa for two equal-length categorical sequences."""
    a, b = pd.Series(list(a)), pd.Series(list(b))
    po = (a == b).mean()
    pe = (a.value_counts(normalize=True) * b.value_counts(normalize=True)).sum()
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


# %%
# One row per (annotator pair, binarised label): kappa plus both raters'
# marginals -- kappa is only interpretable next to the label prevalences
# (they cap the attainable kappa and set the chance-agreement level).
foundations = sorted(set().union(*mine, *theirs["mf"]))
rows = []
for annotator, grp in theirs.groupby("annotator"):
    a, b = mine.loc[grp["term"]], grp["mf"]
    binarised = [("moral_relevance", a.map(bool), b.map(bool))] + [
        (f, a.map(lambda s: f in s), b.map(lambda s: f in s)) for f in foundations
    ]
    rows += [{
        "annotator": annotator,
        "n": len(grp),
        "label": label,
        "kappa": cohen_kappa(in_a, in_b),
        "p_mine": in_a.mean(),
        "p_theirs": in_b.mean(),
    } for label, in_a, in_b in binarised]
kappas = pd.DataFrame(rows).set_index(["annotator", "label"]).round(3)
print(kappas.to_string())

# %%
