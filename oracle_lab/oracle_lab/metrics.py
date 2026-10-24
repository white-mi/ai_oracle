"""Metrics and tests used in the report. Definitions match the audit scripts, so numbers are comparable
(AURC 0.051 for the paper's composite on 1,153 rows, 0.045 for Haiku's P(YES) on 430 rows)."""
import numpy as np
from scipy.stats import binomtest, norm


def mcnemar(x, y):
    """Exact McNemar for paired correctness vectors. Returns (x-only, y-only, p)."""
    x = np.asarray(x).astype(bool)
    y = np.asarray(y).astype(bool)
    n10, n01 = int((x & ~y).sum()), int((~x & y).sum())
    p = binomtest(n10, n10 + n01).pvalue if n10 + n01 else 1.0
    return n10, n01, p


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def cluster_boot(values, clusters, B=5000, seed=0, fn=np.mean, q=(2.5, 97.5)):
    """Percentile bootstrap that resamples markets, because two rows of one market are dependent."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    idx = [np.where(inv == i)[0] for i in range(len(uniq))]
    out = np.empty(B)
    for b in range(B):
        s = rng.integers(0, len(uniq), len(uniq))
        out[b] = fn(values[np.concatenate([idx[j] for j in s])])
    return np.percentile(out, q)


def tost_paired(a_ok, b_ok, clusters, margin=0.02, alpha=0.05, B=5000, seed=0):
    """Equivalence of two paired accuracies within +-margin: the (1-2*alpha) cluster-bootstrap interval of
    mean(a)-mean(b) must lie inside (-margin, margin). Returns (diff, lo, hi, equivalent)."""
    d = np.asarray(a_ok, float) - np.asarray(b_ok, float)
    lo, hi = cluster_boot(d, clusters, B=B, seed=seed, q=(100 * alpha, 100 * (1 - alpha)))
    return d.mean(), lo, hi, bool(lo > -margin and hi < margin)


def holm(pvals, alpha=0.05):
    """Holm step-down. Returns a list of booleans (rejected) in the input order."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    rej = np.zeros(len(p), bool)
    for i, j in enumerate(order):
        if p[j] <= alpha / (len(p) - i):
            rej[j] = True
        else:
            break
    return rej.tolist()


def recall_by_outcome(decision, truth):
    """Share of correct answers separately for true YES and true NO (rule 9). Abstentions count as not correct."""
    decision = np.asarray(decision)
    truth = np.asarray(truth)
    out = {}
    for t in ("YES", "NO"):
        m = truth == t
        out[t] = float((decision[m] == t).mean()) if m.any() else np.nan
    return out


def aurc(score, ok):
    """Area under the risk-coverage curve, lower is better. score: higher = more trusted."""
    o = np.argsort(-np.asarray(score, float), kind="stable")
    err = 1 - np.asarray(ok, float)[o]
    return float((np.cumsum(err) / np.arange(1, len(err) + 1)).mean())


def coverage_at(score, ok, target=0.97, min_k=30):
    """Largest share of rows that can be auto-resolved (top by score) with accuracy >= target. In-sample."""
    score = np.asarray(score, float)
    ok = np.asarray(ok, float)
    o = np.argsort(-score, kind="stable")
    acc = np.cumsum(ok[o]) / np.arange(1, len(ok) + 1)
    ks = [k for k in range(min_k, len(ok) + 1) if acc[k - 1] >= target]
    return (max(ks) / len(ok)) if ks else 0.0


def threshold_for(score, ok, target=0.97, min_k=30):
    """Score threshold chosen on a tuning set (D1) to be frozen and applied to D2 (hypothesis H4)."""
    score = np.asarray(score, float)
    ok = np.asarray(ok, float)
    o = np.argsort(-score, kind="stable")
    acc = np.cumsum(ok[o]) / np.arange(1, len(ok) + 1)
    ks = [k for k in range(min_k, len(ok) + 1) if acc[k - 1] >= target]
    return float(score[o][max(ks) - 1]) if ks else float("inf")


def brier(p_yes, y):
    return float(np.mean((np.asarray(p_yes, float) - np.asarray(y, float)) ** 2))


def ece(conf, ok, nb=10):
    conf = np.asarray(conf, float)
    ok = np.asarray(ok, float)
    edges = np.linspace(0, 1, nb + 1)
    out = 0.0
    for i in range(nb):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= 0) & (conf <= edges[1])
        if m.any():
            out += m.mean() * abs(conf[m].mean() - ok[m].mean())
    return float(out)


def gaussian_copula_majority(p, rho, n, draws=200_000, seed=0):
    """Accuracy of an n-agent majority when each agent is right with probability p and errors share a one-factor
    Gaussian copula with correlation rho (used in the report to show why more agents barely help at rho ~ 0.85)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(draws)[:, None]
    e = rng.standard_normal((draws, n))
    x = np.sqrt(rho) * z + np.sqrt(1 - rho) * e
    right = x < norm.ppf(p)
    return float((right.sum(1) > n / 2).mean())
