"""Per-issue answer entropy for E1 as an agreement measure.

For each issue (topic) the base-2 stance-label entropy over the 6-bin
STANCE_SUPPORT is computed per model, then averaged across models. High mean
entropy = models spread answers across the scale (low agreement); ~0 = all
responses converge on one stance.
"""

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # bias_signatures/ on path

import pandas as pd
from scipy.stats import entropy

import llm
from constructs import STANCE_SUPPORT
from stats import distribution

# %%
e1 = pd.concat(
    pd.read_csv(llm.DATA_DIR / "responses" / "e1" / f"{m}.csv").assign(model=m)
    for m in llm.MODELS
)
e1 = e1[e1["label"].isin(STANCE_SUPPORT)]  # drop invalid / off-support

ent = lambda labels: entropy(distribution(labels, STANCE_SUPPORT, alpha=0), base=2)
per_model = e1.groupby(["topic", "model"])["label"].apply(ent)
issue_entropy = per_model.groupby("topic").mean().sort_values(ascending=False)
print(issue_entropy.to_string())

# %%
