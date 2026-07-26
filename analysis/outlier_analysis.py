"""E2 perturbation outliers: (issue, model) JSDs beyond the Figure 6.4 fences.

Mirrors perturbation_boxplot() in make_figures.py: scope=="wide", one box per
(perturbation, model), Tukey whiskers (1.5*IQR). The rows below are the dots
floating outside the whiskers, grouped by perturbation type.
"""

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # bias_signatures/ on path

import pandas as pd

import llm

# %%
wide = pd.read_csv(llm.DATA_DIR / "signatures" / "e2.csv").query("scope == 'wide'")


jsd = wide.groupby(["partition_b", "model"])["jsd"]
q1, q3 = jsd.transform(lambda s: s.quantile(.25)), jsd.transform(lambda s: s.quantile(.75))
iqr = q3 - q1
out = wide[(wide["jsd"] > q3 + 1.5 * iqr) | (wide["jsd"] < q1 - 1.5 * iqr)]
out = out.sort_values(["partition_b", "jsd"], ascending=[True, False])

# (topic, perturbation type) -> the actual perturbed instance term (topic_prompt)
from make_signatures import E2_COMPARISONS

comparison = {(p, f): name for name, (p, f) in E2_COMPARISONS.items()}
resp = pd.concat(pd.read_csv(p) for p in (llm.DATA_DIR / "responses" / "e2").glob("*.csv"))
resp["partition_b"] = [comparison.get(pf) for pf in zip(resp["perturbation"], resp["framing"])]
terms = (resp.dropna(subset=["partition_b"])
             .groupby(["topic", "partition_b"])["topic_prompt"].first()
             .reset_index(name="perturbed"))

out = out.merge(terms, on=["topic", "partition_b"], how="left").rename(columns={"topic": "issue"})
print(out[["partition_b", "model", "em", "issue", "perturbed", "jsd"]].to_string(index=False))

# %%
import textwrap

import make_figures

ESA = "Write an 8-sentence argument about "
e2r = {p.stem: pd.read_csv(p) for p in (llm.DATA_DIR / "responses" / "e2").glob("*.csv")}
domains = dict(zip(out["issue"], out["em"]))


def slice_(d, issue, partition_b, eight_sa=False):  # partition_b=None -> baseline
    pert, fram = E2_COMPARISONS.get(partition_b, ("none", "neutral"))
    d = d[(d.topic == issue) & (d.perturbation == pert) & (d.framing == fram)]
    return d[d.prompt.str.startswith(ESA)] if eight_sa else d


# %%
# Cleveland chart (8SA, per-response dots) for one issue under a perturbation.
issue, partition_b = "internet restrictions in China", None
data = {m: slice_(d, issue, partition_b, eight_sa=True)
        for m, d in e2r.items() if "label" in d.columns}
make_figures.cleveland_chart({issue: domains[issue]}, save=False, responses=True, data=data)

# %%
# Randomly sample one response for a given model, issue, and perturbation.
# stance=None samples from the whole pool, an int (1-5) restricts to that label.
# eight_sa=True restricts to the 8-sentence-argument prompts.
model, issue, partition_b = "llama-3.1-8b", "internet restrictions in China", None
stance, eight_sa = None, False 
pool = slice_(e2r[model], issue, partition_b, eight_sa=eight_sa)
if stance is not None:
    pool = pool[pd.to_numeric(pool["label"], errors="coerce") == stance]
row = pool.sample(1).iloc[0]
label = row["label"] if "label" in row else "n/a"
print(f"{row['topic_prompt']}  [label {label}]\n")
print(textwrap.fill(str(row["response"]), 100))

# %%
