"""Reproduce every table of the paper from the published per-question files."""
import numpy as np, pandas as pd
from kota import *
A, B, D = load()
P = lambda *a: print(*a)
P("== Table 1/2: dataset file")
gt = D.ground_truth.str.lower()
P(f"rows={len(D)} yes={(gt=='yes').sum()} ({(gt=='yes').mean():.3f}) no={(gt=='no').sum()} categories={D.category.nunique()} unique ids={D.id.nunique()} close range {D.close_time.min()[:10]}..{D.close_time.max()[:10]}")
P(D.category.value_counts().to_dict())
P("files: A rows", len(A), "distinct questions", A.key.nunique(), "| B rows", len(B), "distinct", B.key.nunique(), "| dataset distinct", D.key.nunique())
P("A YES rows", int(A.y.sum()), "B YES rows", int(B.y.sum()))
P("dataset questions missing from A:", len(set(D.key) - set(A.key)), "; from B:", len(set(D.key) - set(B.key)), "; duplicate rows in A:", int(A.key.duplicated().sum()), "in B:", int(B.key.duplicated().sum()))
P("\n== Table 3: accuracy")
for m in M: P(f"{NAMES[m]:14s} single={100*A[m+'_ok'].mean():.2f}  B final-round={100*B[m+'_ok2'].mean():.2f}  (B round1={100*B[m+'_ok1'].mean():.2f})")
P(f"A majority={100*A.maj_ok.mean():.2f} weighted={100*A.wtd_ok.mean():.2f}  B final={100*B.final_ok.mean():.2f}")
P("\n== Table 4/5: paired A vs B under different pairings")
def pair(name, a, b):
    for lab, col in [("A majority", "maj_ok"), ("A weighted", "wtd_ok")]:
        x = a[col].values.astype(bool); y = b.final_ok.values.astype(bool)
        n10, n01, p = mcnemar(x, y)
        P(f"  [{name}] {lab} vs B final: n={len(a)} both ok={int((x&y).sum())} A-only={n10} B-only={n01} both wrong={int((~x&~y).sum())} p={p:.1e}  accA={100*x.mean():.1f} accB={100*y.mean():.1f}")
pair("by file position", A, B)
J = A.merge(B, on="row_uid", suffixes=("", "_B")); 
JA = J[[c for c in A.columns]].copy(); JB = pd.DataFrame({"final_ok": J.final_ok.values})
pair("by row_uid", JA, JB)
Ad = A.drop_duplicates("key"); Bd = B.drop_duplicates("key"); J2 = Ad.merge(Bd[["key", "final_ok", "maj1_ok"]], on="key")
pair("by question (dedup)", J2, J2[["final_ok"]])
n10, n01, p = mcnemar(A.maj_ok, A.wtd_ok); P(f"  A majority vs A weighted: maj-only={n10} wtd-only={n01} p={p:.3f}")
P("\n== Sec 4.1.1: unique markets and Table 6")
for name, df, col in [("A", A, "maj_ok"), ("B", B, "final_ok")]:
    u = df.drop_duplicates("question_id"); k, n = int(u[col].sum()), len(u); lo, hi = wilson(k, n)
    g = df.groupby("question_id")[col].agg(["sum", "size"]); g = g[g["size"] > 1]
    P(f"{name}: unique-market acc (first row per id) {100*k/n:.1f}% [{lo:.3f},{hi:.3f}] n={n}; multi-row markets={len(g)}: all correct={int((g['sum']==g['size']).sum())} mixed={int(((g['sum']>0)&(g['sum']<g['size'])).sum())} all wrong={int((g['sum']==0).sum())}")
P("\n== Table 7: category accuracy")
rows = []
for c, n in A.category.value_counts().items():
    a = A[A.category == c]; b = B[B.category == c]
    rows.append(dict(category=c, nA=len(a), nB=len(b), A_wtd=round(100*a.wtd_ok.mean(), 1), A_maj=round(100*a.maj_ok.mean(), 1), best_single=round(100*max(a[m+'_ok'].mean() for m in M), 1), B=round(100*b.final_ok.mean(), 1), disagree=round(100*(1-a.unan.mean()), 1)))
P(pd.DataFrame(rows).to_string(index=False))
P("\n== Table 8/9: error correlation and conditional error")
E = pd.DataFrame({NAMES[m]: 1 - A[m+'_ok'] for m in M}); P(E.corr().round(3).to_string())
for a in M:
    for b in M:
        if a != b: P(f"  P({NAMES[b]} wrong | {NAMES[a]} wrong) = {E.loc[E[NAMES[a]]==1, NAMES[b]].mean():.3f}")
p = np.mean([A[m+'_ok'].mean() for m in M]); P(f"Condorcet (independent, p={p:.4f}): {p**3 + 3*p*p*(1-p):.4f}; with actual p's: {sum(np.prod([ (A[m+'_ok'].mean() if s[i] else 1-A[m+'_ok'].mean()) for i,m in enumerate(M)]) for s in [(1,1,1),(1,1,0),(1,0,1),(0,1,1)]):.4f}")
P("\n== Table 10 and revisions (B)")
u1 = B.yes1.isin([0, 3]); u2 = B.yes2.isin([0, 3])
P(f"R1 consensus n={int(u1.sum())} acc={100*B.final_ok[u1].mean():.2f}; not: n={int((~u1).sum())}; of those R2 consensus n={int((~u1&u2).sum())} acc={100*B.final_ok[~u1&u2].mean():.1f}; no consensus n={int((~u1&~u2).sum())} acc={100*B.final_ok[~u1&~u2].mean():.1f}")
tot = 0; rows_rev = np.zeros(len(B), bool)
for m in M:
    ch = B[m+'_d1'] != B[m+'_d2']; c2i = int((ch & (B[m+'_ok1']==1)).sum()); i2c = int((ch & (B[m+'_ok1']==0)).sum()); tot += int(ch.sum()); rows_rev |= ch.values
    flag = B[f"{m}_round2_revised"].astype(str).str.lower().eq("true")
    P(f"  {NAMES[m]:14s} decision changes={int(ch.sum())} correct->wrong={c2i} wrong->correct={i2c}; self-reported revised=True: {int(flag.sum())} (matches actual change in {int((flag==ch).sum())} rows)")
P(f"  total changes={tot} of {3*len(B)} ({100*tot/(3*len(B)):.2f}%), rows affected={int(rows_rev.sum())}")
P("\n== Table 11/12 and coverage curve (A)")
P(f"unanimous n={int(A.unan.sum())} acc={100*A.maj_ok[A.unan==1].mean():.2f}; split n={int((A.unan==0).sum())} acc={100*A.maj_ok[A.unan==0].mean():.2f}; median mean-confidence={np.median(A.mean_conf):.3f}")
hi = A.mean_conf >= np.median(A.mean_conf)
for u in (1, 0):
    for h in (True, False):
        mk = (A.unan == u) & (hi == h); P(f"  unan={u} highconf={h}: n={int(mk.sum())} acc={100*A.maj_ok[mk].mean():.2f}")
hi2 = A.mean_conf > np.median(A.mean_conf)
P("  (with strict > median): " + ", ".join(f"u{u}h{int(h)} n={int(((A.unan==u)&(hi2==h)).sum())} acc={100*A.maj_ok[(A.unan==u)&(hi2==h)].mean():.2f}" for u in (1,0) for h in (True,False)))
def curve(score, ok, covs=(0.10, 0.20, 0.25, 0.50, 0.75)):
    o = np.argsort(-score, kind="stable"); c = np.cumsum(ok[o]) / np.arange(1, len(ok)+1)
    return {f"{int(100*k)}%": round(100*c[int(round(k*len(ok)))-1], 2) for k in covs}
P("  A composite:", curve(A.unan.values + A.mean_conf.values, A.maj_ok.values))
B["unan2"] = B.yes2.isin([0, 3]).astype(int); B["mc2"] = np.mean([B[f"{m}_round2_confidence"].values for m in M], axis=0)
P("  B composite (round 2 signals):", curve(B.unan2.values + B.mc2.values, B.final_ok.values))
ce = pd.read_csv(f"{REPO}/results/escalation_analysis/coverage_accuracy_comparison.csv"); P("  repo file coverage_accuracy_comparison.csv head:\n", ce.head(12).to_string(index=False))
