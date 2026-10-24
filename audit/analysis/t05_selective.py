"""Selective resolution on file A: honest (cross-validated) evaluation of escalation rules."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from kota import *
def group_kfold(groups, n_splits=5):
    """GroupKFold of scikit-learn (no shuffle) with a stable sort of group sizes. The library sorts with np.argsort
    without kind, so the order of equal-sized groups, and with it the folds, depends on the numpy build."""
    uniq, gi = np.unique(groups, return_inverse=True); size = np.bincount(gi)
    load = np.zeros(n_splits); fold = np.zeros(len(uniq))
    for g in np.argsort(size, kind="stable")[::-1]:
        f = np.argmin(load); load[f] += size[g]; fold[g] = f
    rf = fold[gi]
    for f in range(n_splits): yield np.where(rf != f)[0], np.where(rf == f)[0]
A, B, D = load()
Ad = A.drop_duplicates("key").reset_index(drop=True)
P = print
ok = Ad.maj_ok.values.astype(float); n = len(Ad)
def aurc(score, ok):
    o = np.argsort(-score, kind="stable"); return (np.cumsum(1 - ok[o]) / np.arange(1, len(ok) + 1)).mean()
def cov_at(score, ok, target):
    o = np.argsort(-score, kind="stable"); c = np.cumsum(ok[o]) / np.arange(1, len(ok) + 1)
    ks = [k for k in range(30, len(ok) + 1) if c[k - 1] >= target]; return (max(ks) / len(ok)) if ks else 0.0
comp = Ad.unan.values + Ad.mean_conf.values
P("== 1. Signals for 'is the majority answer correct?' (1,153 distinct rows), in-sample ranking quality")
sig = {"composite of the paper (unanimous + mean confidence)": comp, "mean confidence only": Ad.mean_conf.values, "unanimity only": Ad.unan.values.astype(float),
       "min confidence of the three": np.min([Ad[f"{m}_confidence"].values for m in M], axis=0)}
for m in M: sig[f"confidence of {NAMES[m]} alone"] = Ad[f"{m}_confidence"].values
for k, s in sig.items(): P(f"  {k:55s} AURC={aurc(s, ok):.4f}  coverage@97%={100*cov_at(s, ok, 0.97):.1f}%  coverage@95%={100*cov_at(s, ok, 0.95):.1f}%")
P(f"  (no signal: AURC={1-ok.mean():.4f}; perfect ranking: AURC={aurc(ok, ok):.4f})")

P("\n== 2. The answer matters: reliability by majority answer and agreement")
for ans in ("YES", "NO"):
    for u in (1, 0):
        mk = (Ad.maj == ans) & (Ad.unan == u); P(f"  majority={ans} unanimous={u}: n={int(mk.sum()):4d} accuracy={100*ok[mk].mean():.1f}%")
mk = (Ad.maj == "NO") & (Ad.unan == 1)
for lo, hi in [(0, .7), (.7, .85), (.85, .92), (.92, .96), (.96, 1.01)]:
    b = mk & (Ad.mean_conf >= lo) & (Ad.mean_conf < hi); P(f"    unanimous NO, mean confidence [{lo},{hi}): n={int(b.sum()):4d} accuracy={100*ok[b].mean():.1f}%")
mk = (Ad.maj == "YES") & (Ad.unan == 1)
for lo, hi in [(0, .85), (.85, .92), (.92, .96), (.96, 1.01)]:
    b = mk & (Ad.mean_conf >= lo) & (Ad.mean_conf < hi); P(f"    unanimous YES, mean confidence [{lo},{hi}): n={int(b.sum()):4d} accuracy={100*ok[b].mean():.1f}%")
top = (Ad.unan == 1) & (Ad.mean_conf >= np.median(A.mean_conf))
P(f"  paper's best cell (unanimous & mean conf >= 0.91) on distinct rows: n={int(top.sum())} acc={100*ok[top].mean():.2f}%; of these YES answers: {int((top&(Ad.maj=='YES')).sum())} (acc {100*ok[top&(Ad.maj=='YES')].mean():.1f}%), NO answers: {int((top&(Ad.maj=='NO')).sum())} (acc {100*ok[top&(Ad.maj=='NO')].mean():.1f}%)")

P("\n== 3. Honest evaluation: threshold chosen on training folds (5-fold, grouped by market), applied to held-out fold")
groups = Ad.question_id.values
def honest(score, target):
    cov = []; hits = 0; tot = 0
    for tr, te in group_kfold(groups):
        o = np.argsort(-score[tr], kind="stable"); c = np.cumsum(ok[tr][o]) / np.arange(1, len(tr) + 1)
        ks = [k for k in range(30, len(tr) + 1) if c[k - 1] >= target]
        thr = score[tr][o][max(ks) - 1] if ks else np.inf
        sel = score[te] >= thr; hits += ok[te][sel].sum(); tot += sel.sum(); cov.append(sel.mean())
    return tot / n, (hits / tot if tot else np.nan), int(tot), int(tot - hits)
for target in (0.99, 0.97, 0.95):
    c, a, k, e = honest(comp, target); lo, hi = wilson(k - e, k)
    P(f"  composite, target {int(100*target)}% on train -> held-out coverage={100*c:.1f}% accuracy={100*a:.2f}% ({e} errors of {k}; Wilson 95% [{100*lo:.1f},{100*hi:.1f}])")

P("\n== 4. A learned gate (logistic regression on votes, confidences, answer) evaluated out-of-fold")
X = np.column_stack([Ad.unan.values, (Ad.maj == "YES").astype(int).values, Ad.yes_votes_n.values] + [Ad[f"{m}_confidence"].values for m in M] +
                    [np.where(Ad[f"{m}_d"] == Ad.maj, 1, -1) * Ad[f"{m}_confidence"].values for m in M])
oof = np.zeros(n)
for tr, te in group_kfold(groups):
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], ok[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
P(f"  out-of-fold AURC: learned gate={aurc(oof, ok):.4f} vs composite={aurc(comp, ok):.4f}; coverage@97%: gate={100*cov_at(oof, ok, 0.97):.1f}% composite={100*cov_at(comp, ok, 0.97):.1f}%; @95%: gate={100*cov_at(oof, ok, 0.95):.1f}% composite={100*cov_at(comp, ok, 0.95):.1f}%")
brier = lambda p: ((p - ok) ** 2).mean()
P(f"  Brier of P(majority correct): gate (out-of-fold)={brier(oof):.4f}; mean confidence used as probability={brier(Ad.mean_conf.values):.4f}; constant base rate={brier(np.full(n, ok.mean())):.4f}")
rng = np.random.default_rng(0); d = []
uniq = np.unique(groups); idx = {u: np.where(groups == u)[0] for u in uniq}
for _ in range(2000):
    s = rng.choice(uniq, len(uniq)); ii = np.concatenate([idx[u] for u in s]); d.append(aurc(oof[ii], ok[ii]) - aurc(comp[ii], ok[ii]))
P(f"  AURC difference (gate - composite), cluster bootstrap 95% CI: [{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}]")

P("\n== 5. Adding evidence dating (subset with committed packets), out-of-fold")
Xp = pd.read_pickle("X_packets.pkl").reset_index(drop=True)
okp = Xp.maj_ok.values.astype(float); gp = Xp.question_id.values
base = np.column_stack([Xp.unan.values, (Xp.maj == "YES").astype(int).values] + [Xp[f"{m}_confidence"].values for m in M])
ext = np.column_stack([base, Xp.late.astype(int).values, np.clip(Xp.newest_lag.fillna(-400).values, -60, 2), (Xp.maj == "NO").astype(int).values * (1 - Xp.late.astype(int).values)])
def oof_pred(Xm):
    out = np.zeros(len(Xm))
    for tr, te in group_kfold(gp):
        out[te] = LogisticRegression(max_iter=2000).fit(Xm[tr], okp[tr]).predict_proba(Xm[te])[:, 1]
    return out
o1, o2 = oof_pred(base), oof_pred(ext)
P(f"  n={len(Xp)}; AURC composite={aurc(Xp.unan.values + Xp.mean_conf.values, okp):.4f}; gate without dates={aurc(o1, okp):.4f}; gate with dates={aurc(o2, okp):.4f}")
for lab, mk in [("unanimous NO, late source present", (Xp.maj=='NO')&(Xp.unan==1)&Xp.late), ("unanimous NO, no late source", (Xp.maj=='NO')&(Xp.unan==1)&~Xp.late), ("unanimous YES, late source present", (Xp.maj=='YES')&(Xp.unan==1)&Xp.late), ("unanimous YES, no late source", (Xp.maj=='YES')&(Xp.unan==1)&~Xp.late)]:
    P(f"    {lab:36s} n={int(mk.sum()):3d} accuracy={100*okp[mk].mean():.1f}%")
