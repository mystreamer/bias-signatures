"""Pure statistical functions: JSD, permutation test, Spearman + bootstrap.

No I/O and no pipeline vocabulary — everything operates on label sequences
over an explicit support, or on plain vectors.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import spearmanr


def distribution(labels: Sequence[str], support: Sequence[str], alpha: float = 1.0) -> np.ndarray:
    """Laplace-smoothed probability distribution of labels over a fixed support.

    Labels outside the support are an error: filtering (e.g. dropping
    refusal/invalid from Likert distributions) is the caller's decision.
    """
    index = {s: i for i, s in enumerate(support)}
    counts = np.bincount([index[l] for l in labels], minlength=len(support)).astype(float)
    counts += alpha
    return counts / counts.sum()


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence, base 2, bounded [0, 1]."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = (p + q) / 2

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def jsd_between(
    labels_a: Sequence[str],
    labels_b: Sequence[str],
    support: Sequence[str],
    alpha: float = 1.0,
) -> float:
    return jsd(distribution(labels_a, support, alpha), distribution(labels_b, support, alpha))


def permutation_test(
    labels_a: Sequence[str],
    labels_b: Sequence[str],
    support: Sequence[str],
    n_permutations: int = 10_000,
    alpha: float = 1.0,
    seed: int | None = None,
) -> tuple[float, float]:
    """Unpaired Monte-Carlo permutation test with the JSD as test statistic.

    Returns (observed_jsd, p_value); p uses the add-one estimator so p > 0.
    """
    rng = np.random.default_rng(seed)
    index = {s: i for i, s in enumerate(support)}
    k = len(support)
    idx_a = np.array([index[l] for l in labels_a])
    idx_b = np.array([index[l] for l in labels_b])

    def stat(a_idx: np.ndarray, b_idx: np.ndarray) -> float:
        pa = np.bincount(a_idx, minlength=k) + alpha
        pb = np.bincount(b_idx, minlength=k) + alpha
        return jsd(pa / pa.sum(), pb / pb.sum())

    observed = stat(idx_a, idx_b)
    pooled = np.concatenate([idx_a, idx_b])
    n_a = len(idx_a)
    exceeded = 0
    for _ in range(n_permutations):
        rng.shuffle(pooled)
        if stat(pooled[:n_a], pooled[n_a:]) >= observed:
            exceeded += 1
    return observed, (exceeded + 1) / (n_permutations + 1)


def bonferroni(p: float, n_tests: int) -> float:
    return min(1.0, p * n_tests)


def jsd_bootstrap(
    labels_a: Sequence[str],
    labels_b: Sequence[str],
    support: Sequence[str],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    alpha: float = 1.0,
    seed: int | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap interval (lo, hi) for the smoothed JSD, redrawing
    each partition's counts ~ Multinomial(n, p_hat) — i.i.d. label resampling
    (prompt clustering deferred). Magnitude precision only: JSD is boundary-
    biased, so the interval never reaches 0 and must not be read as a
    significance device — that remains the permutation test's job.
    """
    rng = np.random.default_rng(seed)
    index = {s: i for i, s in enumerate(support)}
    k = len(support)
    ca = np.bincount([index[l] for l in labels_a], minlength=k)
    cb = np.bincount([index[l] for l in labels_b], minlength=k)
    resamples = []
    for _ in range(n_resamples):
        ra = rng.multinomial(ca.sum(), ca / ca.sum()) + alpha
        rb = rng.multinomial(cb.sum(), cb / cb.sum()) + alpha
        resamples.append(jsd(ra / ra.sum(), rb / rb.sum()))
    tail = (1 - confidence) / 2
    lo, hi = np.quantile(resamples, [tail, 1 - tail])
    return float(lo), float(hi)


def spearman_count_bootstrap(
    parts: dict,
    elements: Sequence[tuple],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    alpha: float = 1.0,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Response-level bootstrap CI for the Spearman rho of two JSD vectors.

    parts: partition key -> label counts over that partition's support.
    elements: one ((ka, kb), (kc, kd)) per vector element — x[i] is the
    smoothed JSD between partitions ka and kb, y[i] between kc and kd.

    Each replicate redraws every partition's counts ~ Multinomial(n, p_hat) —
    distributionally identical to i.i.d. label resampling (prompt clustering
    deliberately deferred: samples within a prompt are treated as independent,
    and partitions sharing multi-label rows are resampled independently) —
    then recomputes both JSD vectors and re-derives rho. Returns (rho, lo, hi)
    with rho computed from the original counts.
    """
    rng = np.random.default_rng(seed)
    counts = {k: np.asarray(v, float) for k, v in parts.items()}

    def rho_from(c: dict) -> float:
        def d(k):
            smoothed = c[k] + alpha
            return smoothed / smoothed.sum()
        x = [jsd(d(ka), d(kb)) for (ka, kb), _ in elements]
        y = [jsd(d(kc), d(kd)) for _, (kc, kd) in elements]
        return float(spearmanr(x, y).statistic)

    rho = rho_from(counts)
    involved = sorted({k for xy in elements for pair in xy for k in pair})
    resamples = []
    for _ in range(n_resamples):
        redrawn = {k: rng.multinomial(int(counts[k].sum()), counts[k] / counts[k].sum())
                   for k in involved}
        r = rho_from(redrawn)
        if not np.isnan(r):
            resamples.append(r)
    tail = (1 - confidence) / 2
    lo, hi = np.quantile(resamples, [tail, 1 - tail])
    return rho, float(lo), float(hi)


def spearman_bootstrap(
    x: Sequence[float],
    y: Sequence[float],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Spearman rho with a paired-resampling bootstrap CI: (rho, lo, hi)."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = float(spearmanr(x, y).statistic)
    n = len(x)
    resamples = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        r = spearmanr(x[idx], y[idx]).statistic
        if not np.isnan(r):
            resamples.append(float(r))
    tail = (1 - confidence) / 2
    lo, hi = np.quantile(resamples, [tail, 1 - tail])
    return rho, float(lo), float(hi)
