"""Per-issue answer entropy for E1 as an agreement measure.

Two base-2 entropies over the 6-bin STANCE_SUPPORT per issue (topic):

  within   per-model label entropy, averaged across models — template
           sensitivity (how unstable each model is across prompt paraphrases)
  between  entropy of the per-model mean stances, rounded to the nearest
           bin — cross-model disagreement (0 = all models' means coincide,
           regardless of how spread each individual model is)
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
groups = e1.groupby(["topic", "model"])["label"]
means = groups.apply(lambda g: pd.to_numeric(g, errors="coerce").mean())
issue_entropy = (pd.DataFrame({
                     "within": groups.apply(ent).groupby("topic").mean(),
                     "between": (means.round().dropna().astype(int).astype(str)
                                 .groupby("topic").apply(ent)),
                 })
                 .assign(domain=e1.groupby("topic")["domain"].first())
                 .sort_values("between", ascending=False)
                 .reset_index()[["domain", "within", "between", "topic"]])
w = issue_entropy["topic"].str.len().max()
print(issue_entropy.to_string(index=False, formatters={"topic": lambda s: f"{s:<{w}}"}))

# %%
import make_figures

# %%
# Cleveland charts for a chosen slice of issues, reusing make_figures.cleveland_chart.
# save=False renders inline (one figure per domain) without touching the curated PDFs.
selection = issue_entropy.groupby("domain").head(5)
make_figures.cleveland_chart(dict(zip(selection["topic"], selection["domain"])), save=False)

# %%
# Cleveland for a single topic.
topic = "international relations"
make_figures.cleveland_chart({topic: issue_entropy.set_index("topic")["domain"][topic]},
                             save=False, responses=True)

# %%
# Randomly sample one response for a given model and issue.
import textwrap

model, issue = "llama-3.1-8b", "international relations"
row = e1[(e1["model"] == model) & (e1["topic"] == issue)].sample(3).iloc[1]
print(f"stance label: {row['label']}\n")
print(textwrap.fill(row["response"], 100))

# %%
