"""Size-matched perturbation sensitivity of the 8SA instruction (E2).

The dashed 8SA median once drawn in the perturbation box plot pooled only
n=3 responses per side, where Laplace smoothing (alpha=1) over the 6-bin
support caps the JSD at ~0.154, versus ~0.923 for the 159-vs-159 wide-scope
cells. Those two scopes are therefore not comparable. The fair question is
whether the 8SA instruction is more or less perturbation-sensitive than the
OTHER argumentative instructions, each measured on the same footing:

  per (model, topic, comparison) and per instruction: JSD between that
  instruction's 3 baseline and 3 perturbed responses (same smoothing), then
  the percentile rank of the 8SA instruction among all instructions.

Prints, per comparison, the pooled medians, the median percentile rank of
8SA, how often it lies below the median of the other instructions, and a
two-sided sign test of that count against 0.5.
"""

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # bias_signatures/ on path

import pandas as pd
from scipy.stats import binomtest

import llm
from constructs import INVALID, STANCE_SUPPORT
from make_signatures import E2_COMPARISONS, EIGHT_SA
from stats import jsd_between

# %%
e2 = pd.concat(
    pd.read_csv(llm.DATA_DIR / "responses" / "e2" / f"{m}.csv").assign(model=m)
    for m in llm.MODELS
)
e2 = e2[e2["label"] != INVALID]
eight_sa = e2.loc[e2["prompt"].str.startswith(EIGHT_SA.replace(" X.", " ")), "instruction_id"].unique()
assert len(eight_sa) == 1, eight_sa
eight_sa = eight_sa[0]

print(f"JSD ceiling (alpha=1, {len(STANCE_SUPPORT)} bins): "
      f"3 vs 3 = {jsd_between(['1'] * 3, ['5'] * 3, STANCE_SUPPORT):.3f}, "
      f"159 vs 159 = {jsd_between(['1'] * 159, ['5'] * 159, STANCE_SUPPORT):.3f}")
print(f"instructions: {e2['instruction_id'].nunique()}, 8SA = {eight_sa}")

# %%
rows = []
for (model, topic), d in e2.groupby(["model", "topic"]):
    for comparison, (perturbation, framing) in E2_COMPARISONS.items():
        per_instruction = {}
        for instruction, di in d.groupby("instruction_id"):
            base = di[(di["perturbation"] == "none") & (di["framing"] == "neutral")]["label"]
            perturbed = di[(di["perturbation"] == perturbation) & (di["framing"] == framing)]["label"]
            if base.empty or perturbed.empty:
                continue
            per_instruction[instruction] = jsd_between(list(base), list(perturbed), STANCE_SUPPORT)
        s = pd.Series(per_instruction)
        sa, others = s[eight_sa], s.drop(eight_sa)
        rows.append({
            "model": model, "topic": topic, "comparison": comparison,
            "jsd_8sa": sa,
            "jsd_other_median": others.median(),
            "pct_rank_8sa": (others < sa).mean() + 0.5 * (others == sa).mean(),
            "below_median": sa < others.median(),
        })
ranks = pd.DataFrame(rows)

# %%
summary = ranks.groupby("comparison", sort=False).agg(
    median_8sa=("jsd_8sa", "median"),
    median_other_instruction=("jsd_other_median", "median"),
    median_pct_rank_8sa=("pct_rank_8sa", "median"),
    cells_8sa_below_median=("below_median", "sum"),
    n_cells=("below_median", "size"),
)
summary["sign_test_p"] = [
    binomtest(int(k), int(n)).pvalue
    for k, n in zip(summary["cells_8sa_below_median"], summary["n_cells"])
]
print(summary.round(3).to_string())
k, n = int(ranks["below_median"].sum()), len(ranks)
print(f"\nall comparisons: 8SA below the other instructions' median in {k}/{n} cells, "
      f"sign test p = {binomtest(k, n).pvalue:.2g}")
