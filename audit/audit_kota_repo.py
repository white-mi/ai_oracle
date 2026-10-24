"""Audit of the published per-question outputs of arXiv:2605.30802.

Usage:
    git clone https://github.com/Tkota10/Senior-Thesis-Multi-LLM-Resolution repo
    python audit_kota_repo.py repo

Reads only files committed to the repository. No API calls.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact

REPO = sys.argv[1] if len(sys.argv) > 1 else "repo"
M = ["gpt4o", "deepseek", "llama"]  # column prefixes used in the repo ("gpt4o" = GPT-5 nano)


def norm(s):
    return s.astype(str).str.strip().str.upper().values


def majority(df, cols):
    y = sum((norm(df[c]) == "YES").astype(int) for c in cols)
    return np.where(y >= 2, "YES", "NO"), y


def mcnemar(x, y):
    n10, n01 = int((x & ~y).sum()), int((~x & y).sum())
    p = binomtest(n10, n10 + n01).pvalue if n10 + n01 else 1.0
    return n10, n01, p


def ece(conf, correct, nb=10):
    edges = np.linspace(0, 1, nb + 1)
    out = 0.0
    for i in range(nb):
        mk = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= 0) & (conf <= edges[1])
        if mk.sum():
            out += mk.mean() * abs(conf[mk].mean() - correct[mk].mean())
    return out


def aurc(score, correct):
    """Area under the risk-coverage curve (lower is better)."""
    order = np.argsort(-score, kind="stable")
    err = 1 - correct[order]
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    return risk.mean()


A = pd.read_csv(f"{REPO}/results/Final_Architecture_A_results.csv")
B = pd.read_csv(f"{REPO}/results/Final_Architecture_B_results.csv")
gA, gB = norm(A.ground_truth), norm(B.ground_truth)
yA, yB = gA == "YES", gB == "YES"
R = {}

print("== 0. Shape")
print(f"rows A={len(A)} B={len(B)}; unique markets={A.question_id.nunique()}; "
      f"markets with 2 rows={(A.question_id.value_counts() > 1).sum()}; true YES share={yA.mean():.3f}")

print("\n== 1. Per-model accuracy: Architecture A vs round 1 and round 2 of B")
for m in M:
    a = (norm(A[f"{m}_decision"]) == gA)
    b1 = (norm(B[f"{m}_round1_decision"]) == gB)
    b2 = (norm(B[f"{m}_round2_decision"]) == gB)
    R[f"acc_{m}"] = (a.mean(), b1.mean(), b2.mean())
    print(f"{m:9s} A={100*a.mean():.2f}  B_R1={100*b1.mean():.2f}  B_R2={100*b2.mean():.2f}")

amaj, ay = majority(A, [f"{m}_decision" for m in M])
b1maj, b1y = majority(B, [f"{m}_round1_decision" for m in M])
fb = norm(B.final_decision)
print(f"A majority={100*(amaj==gA).mean():.2f} ({(amaj==gA).sum()})  "
      f"B R1 majority={100*(b1maj==gB).mean():.2f} ({(b1maj==gB).sum()})  "
      f"B final={100*(fb==gB).mean():.2f} ({(fb==gB).sum()})")
n10, n01, p = mcnemar(b1maj == gB, fb == gB)
print(f"Pure debate effect (B R1 majority -> B final): lost {n10}, gained {n01}, net {n01-n10}, McNemar p={p:.2f}")

print("\n== 2. Answer bias: share of YES and recall by true outcome")
for name, pred, g in [("A majority", amaj, gA), ("B R1 majority", b1maj, gB), ("B final", fb, gB)]:
    y = g == "YES"
    fn, fp = int(((pred == "NO") & y).sum()), int(((pred == "YES") & ~y).sum())
    print(f"{name:14s} YES share={100*(pred=='YES').mean():.1f}  recall YES={100*(pred[y]=='YES').mean():.1f}  "
          f"recall NO={100*(pred[~y]=='NO').mean():.1f}  errors={fn+fp} (false NO={fn}, false YES={fp})")
for m in M:
    a, b = norm(A[f"{m}_decision"]), norm(B[f"{m}_round1_decision"])
    print(f"{m:9s} YES share A={100*(a=='YES').mean():.1f} B_R1={100*(b=='YES').mean():.1f}; "
          f"recall YES A={100*(a[yA]=='YES').mean():.1f} B_R1={100*(b[yB]=='YES').mean():.1f}; "
          f"mean confidence A={A[f'{m}_confidence'].mean():.3f} B_R1={B[f'{m}_round1_confidence'].mean():.3f}")
print("rows with confidence exactly 0: " + ", ".join(
    f"{m} A={(A[f'{m}_confidence'] == 0).sum()} B_R1={(B[f'{m}_round1_confidence'] == 0).sum()}" for m in M))
allw = np.all([norm(A[f"{m}_decision"]) != gA for m in M], axis=0)
print(f"A: all three wrong on {allw.sum()} rows, of which true YES: {(allw & yA).sum()}")

print("\n== 3. Voting vs single models inside A (exact McNemar)")
w = sum(np.where(norm(A[f"{m}_decision"]) == "YES", 1, -1) * A[f"{m}_confidence"].values for m in M)
wv = np.where(w > 0, "YES", "NO")
print(f"majority correct={(amaj==gA).sum()}, confidence-weighted correct={(wv==gA).sum()}")
for lab, pred in [("majority", amaj), ("weighted", wv)]:
    for m in M:
        n10, n01, p = mcnemar(pred == gA, norm(A[f"{m}_decision"]) == gA)
        print(f"  {lab:8s} vs {m:9s}: +{n10} / -{n01}, p={p:.3f}")

print("\n== 4. Error correlations")
EA = pd.DataFrame({m: (norm(A[f"{m}_decision"]) != gA).astype(int) for m in M}).corr()
EB = pd.DataFrame({m: (norm(B[f"{m}_round1_decision"]) != gB).astype(int) for m in M}).corr()
print("A:\n", EA.round(3).to_string(), "\nB round 1:\n", EB.round(3).to_string())

print("\n== 5. Verbalized confidence in A")
for m in M:
    c = A[f"{m}_confidence"].values.astype(float)
    ok = (norm(A[f"{m}_decision"]) == gA).astype(float)
    py = np.where(norm(A[f"{m}_decision"]) == "YES", c, 1 - c)
    hi = c >= 0.9
    print(f"{m:9s} ECE={ece(c, ok):.3f}  Brier={((py - yA) ** 2).mean():.3f}  "
          f"conf>=0.9 on {100*hi.mean():.0f}% of rows, accuracy there {100*ok[hi].mean():.1f}%")
print(f"Brier of the base rate: {yA.mean()*(1-yA.mean()):.3f}")

print("\n== 6. Selective resolution baselines in A (in-sample)")
ua = (ay == 0) | (ay == 3)
mean_conf = np.mean([A[f"{m}_confidence"].values for m in M], axis=0)
comp = ua.astype(float) + mean_conf
ok = (amaj == gA).astype(float)
print(f"unanimous n={ua.sum()} acc={100*ok[ua].mean():.2f}; split n={(~ua).sum()} acc={100*ok[~ua].mean():.2f}")
print(f"AURC composite={aurc(comp, ok):.4f}; AURC mean confidence={aurc(mean_conf, ok):.4f}; "
      f"AURC random={1-ok.mean():.4f}")
order = np.argsort(-comp, kind="stable")
cum = np.cumsum(ok[order]) / np.arange(1, len(ok) + 1)
for cov in (0.25, 0.5, 0.75):
    k = int(cov * len(ok))
    print(f"  coverage {int(cov*100)}%: accuracy {100*cum[k-1]:.2f}")
best = max(k for k in range(50, len(ok) + 1) if cum[k - 1] >= 0.97)
print(f"  largest coverage with cumulative accuracy >= 97%: {100*best/len(ok):.1f}%")

print("\n== 7. Evidence packets committed to the repo")
feats_uid, feats_id = {}, {}
late_all = []
for f in sorted(glob.glob(f"{REPO}/cache/evidence/*.json")):
    e = json.load(open(f))
    rd = pd.to_datetime(e["resolution_date"], utc=True)
    ds = [pd.to_datetime(s.get("published_date"), utc=True, errors="coerce") for s in e.get("sources", [])]
    ds = [d for d in ds if not pd.isna(d)]
    late = any(d >= rd - pd.Timedelta(days=1) for d in ds)
    late_all.append(late)
    b = os.path.basename(f)[:-5]
    (feats_uid if b.startswith("highlights__") else feats_id)[b.replace("highlights__", "")] = late
print(f"packets={len(late_all)}; share with at least one source dated no earlier than 1 day before close: "
      f"{np.mean(late_all):.3f}")
F = [feats_uid.get(u, feats_id.get(q)) for u, q in zip(A.row_uid, A.question_id)]
has = np.array([f is not None for f in F])
late = np.array([bool(f) for f in F])
tab = []
for lab, mk in [("late source present", has & late), ("no late source", has & ~late)]:
    hit, tot = int((amaj[mk & yA] == "YES").sum()), int((mk & yA).sum())
    tab.append((hit, tot - hit))
    print(f"  {lab:20s} rows={mk.sum():4d}  accuracy={100*ok[mk].mean():.1f}  recall on true YES={hit}/{tot}={100*hit/tot:.1f}%")
print(f"  Fisher exact p for the recall difference: {fisher_exact(tab)[1]:.4f}")

print("\n== 8. Row overlap between the two files (paired check of section 1)")
for d in (A, B):
    d["key"] = d.question_id.astype(str) + "|" + d.question_text.astype(str) + "|" + d.resolution_criteria.astype(str)
print(f"distinct questions: A={A.key.nunique()} of {len(A)} rows, B={B.key.nunique()} of {len(B)} rows; "
      f"shared={len(set(A.key) & set(B.key))}")
J = A.drop_duplicates("key").merge(B.drop_duplicates("key"), on="key", suffixes=("_A", "_B"))
g = norm(J.ground_truth_A)
am, _ = majority(J, [f"{m}_decision" for m in M])
bm, _ = majority(J, [f"{m}_round1_decision" for m in M])
n10, n01, p = mcnemar(am == g, bm == g)
print(f"paired rows={len(J)}: A majority {100*(am==g).mean():.2f}%, B R1 majority {100*(bm==g).mean():.2f}%; "
      f"A-only correct {n10}, B-R1-only correct {n01}, McNemar p={p:.1e}")
for m in M:
    a, b = norm(J[f"{m}_decision"]), norm(J[f"{m}_round1_decision"])
    print(f"  {m:9s} decisions differ on {(a != b).sum()} rows: YES->NO {((a=='YES')&(b=='NO')).sum()}, "
          f"NO->YES {((a=='NO')&(b=='YES')).sum()}")
