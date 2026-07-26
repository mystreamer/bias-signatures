"""Bootstrap CIs elicited live from the canonical label counts.

Both cells resample data/signatures/label_counts.json (written by
make_signatures.py) instead of reading the persisted CSV intervals.

  (a) per-(em, model) MF-pair JSD with its percentile-bootstrap CI
  (b) cross-{model,domain,em} Spearman rho CIs, re-deriving rho from
      resampled counts (the make_correlation.py machinery)
"""

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # bias_signatures/ on path

import json
from itertools import combinations

import pandas as pd

import make_correlation as mc
from stats import jsd_between, jsd_bootstrap

N = 1_000  # resamples per interval (the pipeline uses 10_000)

# %%
# (a) JSD + CI per (em, model, MF pair), resampled from the label counts.
rows = []
for key, v in json.loads((mc.SIGNATURES_DIR / "label_counts.json").read_text()).items():
    em, model = key.split("|")
    labels = {f: [s for s, n in zip(v["support"], c) for _ in range(n)]
              for f, c in v["counts"].items()}
    for f1, f2 in combinations(sorted(labels), 2):
        lo, hi = jsd_bootstrap(labels[f1], labels[f2], v["support"], N, seed=0)
        rows.append({"em": em, "model": model, "pair": f"{f1}|{f2}",
                     "jsd": round(jsd_between(labels[f1], labels[f2], v["support"]), 3),
                     "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
print(pd.DataFrame(rows).to_string(index=False))

# %%
# (b) Spearman rho CIs (mc.row draws rho and CI from the resampled counts).
mc.PARTS, mc.N_RESAMPLES = mc.partition_counts(), N
vectors = mc.em_vectors()
models = sorted(next(iter(vectors.values())))
rows = [mc.row("cross_model", m1, m2,
               mc.elements(vectors[em][m1], vectors[em][m2], (em, m1), (em, m2)), em)
        for em in ("political", "personal") for m1, m2 in combinations(models, 2)]
rows += [mc.row("cross_domain", m, m,
                mc.elements(vectors["political"][m], vectors["personal"][m],
                            ("political", m), ("personal", m))) for m in models]
rows += [mc.row("cross_em", c1, c2,
                [e for m in models for e in mc.elements(vectors[c1][m], vectors[c2][m],
                                                        (c1, m), (c2, m))])
         for c1, c2 in combinations(sorted(vectors), 2)]
print(pd.DataFrame(rows).to_string(index=False))

# %%
