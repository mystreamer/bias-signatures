"""Ranked correlations between bias signatures (consistency measurement).

Consumes only data/signatures/{e1,e3}.csv (canonical scheme, so all channels
share the same 10 MF-pair support) and writes data/signatures/correlations.csv:

  cross_model    within-channel model x model rho          (Table 5.2)
  cross_domain   within-model political vs personal rho    (Table 5.3)
  cross_channel  channel x channel, stacked across models  (Table 5.4)

Pooled rows: cross_model pools the coefficients (mean + bootstrap CI over
them); cross_domain and cross_channel stack the per-model vectors into one
50-observation pair and bootstrap that.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from llm import DATA_DIR
from stats import spearman_bootstrap

SIGNATURES_DIR = DATA_DIR / "signatures"
N_RESAMPLES = 5_000


def channel_vectors() -> dict[str, dict[str, pd.Series]]:
    """channel -> model -> MF-pair JSD vector (canonical scheme)."""
    frames = [pd.read_csv(SIGNATURES_DIR / f"{e}.csv") for e in ("e1", "e3")]
    df = pd.concat(frames)
    df = df[(df["family"] == "mvb") & (df["scheme"] == "canonical")]
    df["pair"] = df["partition_a"] + "|" + df["partition_b"]
    out: dict[str, dict[str, pd.Series]] = {}
    for (channel, model), group in df.groupby(["channel", "model"]):
        out.setdefault(channel, {})[model] = group.set_index("pair")["jsd"].sort_index()
    return out


def aligned(a: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    shared = a.index.intersection(b.index)
    return a[shared].to_numpy(), b[shared].to_numpy()


def row(comparison, unit_a, unit_b, x, y, channel="") -> dict:
    rho, lo, hi = spearman_bootstrap(x, y, N_RESAMPLES, seed=0)
    return {
        "comparison": comparison, "channel": channel,
        "unit_a": unit_a, "unit_b": unit_b,
        "rho": round(rho, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3), "n": len(x),
    }


if __name__ == "__main__":
    vectors = channel_vectors()
    models = sorted(next(iter(vectors.values())))
    rows = []

    # cross-model within domain
    coefficients = []
    for channel in ("political", "personal"):
        for m1, m2 in combinations(models, 2):
            x, y = aligned(vectors[channel][m1], vectors[channel][m2])
            r = row("cross_model", m1, m2, x, y, channel)
            rows.append(r)
            coefficients.append(r["rho"])
    rng = np.random.default_rng(0)
    resampled = [
        np.mean(rng.choice(coefficients, len(coefficients))) for _ in range(N_RESAMPLES)
    ]
    rows.append({
        "comparison": "cross_model_pooled", "channel": "", "unit_a": "", "unit_b": "",
        "rho": round(float(np.mean(coefficients)), 3),
        "ci_lo": round(float(np.quantile(resampled, 0.025)), 3),
        "ci_hi": round(float(np.quantile(resampled, 0.975)), 3),
        "n": len(coefficients),
    })

    # cross-domain within model (+ stacked pooled row)
    stacked_x, stacked_y = [], []
    for model in models:
        x, y = aligned(vectors["political"][model], vectors["personal"][model])
        rows.append(row("cross_domain", model, model, x, y))
        stacked_x.extend(x)
        stacked_y.extend(y)
    rows.append(row("cross_domain_pooled", "political", "personal", stacked_x, stacked_y))

    # cross-channel, stacked across models
    for c1, c2 in combinations(sorted(vectors), 2):
        xs, ys = [], []
        for model in models:
            x, y = aligned(vectors[c1][model], vectors[c2][model])
            xs.extend(x)
            ys.extend(y)
        rows.append(row("cross_channel", c1, c2, xs, ys))

    out = SIGNATURES_DIR / "correlations.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{out.name}: {len(rows)} correlation rows")
