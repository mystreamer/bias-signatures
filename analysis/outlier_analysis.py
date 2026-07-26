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
print(out[["partition_b", "model", "channel", "topic", "jsd"]].to_string(index=False))

# %%
