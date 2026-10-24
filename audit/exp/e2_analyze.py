"""Analysis of batch-2 experiments (whatever has finished)."""
import sys, os, glob, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
S = "."
sys.path.insert(0, f"{S}/analysis"); sys.path.insert(0, f"{S}/exp")
from kota import mcnemar, cluster_boot, wilson, load, M, NAMES
A, B, D = load()
Au = A.drop_duplicates("key").set_index("key")
E1 = pd.read_csv(f"{S}/exp/e1_haiku_nall_rep0_HA-TA-HB-TB.csv")
HA = E1[E1.cond == "HA"].set_index("key")
keys = HA.index.tolist()
y = HA.y.values.astype(int); mk = HA.question_id.values
def ece(conf, ok, nb=10):
    conf = np.asarray(conf, float); ok = np.asarray(ok, float); edges = np.linspace(0, 1, nb + 1); out = 0
    for i in range(nb):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= 0) & (conf <= edges[1])
        if m.sum(): out += m.mean() * abs(conf[m].mean() - ok[m].mean())
    return out
def aurc(score, ok):
    o = np.argsort(-np.asarray(score, float), kind="stable"); err = 1 - np.asarray(ok, float)[o]
    return (np.cumsum(err) / np.arange(1, len(err) + 1)).mean()
def auc(score, y):
    from scipy.stats import rankdata
    r = rankdata(score); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def ci(v): lo, hi = cluster_boot(np.asarray(v, float), mk, B=3000); return f"[{100*lo:.1f},{100*hi:.1f}]"
print(f"rows {len(keys)}; true YES {y.sum()} ({100*y.mean():.1f}%); always-NO accuracy {100*(1-y.mean()):.1f}%; markets {len(set(mk))}")
pubA = Au.loc[keys]
print("\n== Haiku (content, prompt A) vs published systems on the same rows")
ha_ok = HA.ok.values
for lab, col in [("A majority", "maj_ok"), ("A weighted", "wtd_ok")] + [(NAMES[m], f"{m}_ok") for m in M]:
    o = pubA[col].values; n10, n01, p = mcnemar(ha_ok, o); d = ha_ok - o
    print(f"  Haiku {100*np.nanmean(ha_ok):.1f} vs {lab:14s} {100*o.mean():.1f}: Haiku-only {n10}, other-only {n01}, diff {100*np.nanmean(d):+.1f} pp {ci(np.nan_to_num(d))} p={p:.3f}")
def load_e2(name):
    f = glob.glob(f"{S}/exp/e2_{name}_n*.csv"); f = [x for x in f if "_n6" not in x]
    if not f: return None
    d = pd.read_csv(f[0]).drop_duplicates("key").set_index("key"); return d.reindex(keys)
# ---------- closed book
for model in ("haiku", "sonnet"):
    C = load_e2(f"closed_{model}")
    if C is None: continue
    p = C.p_yes.values.astype(float); ok = ((p > 0.5).astype(int) == y); v = ~np.isnan(p)
    print(f"\n== closed book, {model}: n valid {v.sum()}")
    print(f"  accuracy {100*ok[v].mean():.1f} {ci(ok[v]) if v.all() else ''}; YES share {100*(p[v]>0.5).mean():.1f}; recall YES {100*ok[v&(y==1)].mean():.1f}; recall NO {100*ok[v&(y==0)].mean():.1f}")
    print(f"  Brier {np.mean((p[v]-y[v])**2):.3f} (base-rate Brier {y.mean()*(1-y.mean()):.3f}); AUC of P(YES) {auc(p[v], y[v]):.3f}")
    conf = np.maximum(p, 1-p); hi = conf >= 0.9
    print(f"  |p-0.5|>=0.4 on {100*hi[v].mean():.1f}% rows, accuracy there {100*ok[v&hi].mean():.1f}; p==0.5 exactly: {int((p==0.5).sum())}")
    print(f"  distribution of p_yes: " + ", ".join(f"{q:.2f}" for q in np.nanquantile(p, [0.05, 0.25, 0.5, 0.75, 0.95])))
    n10, n01, pp = mcnemar(ha_ok[v] == 1, ok[v]); print(f"  with evidence (HA) vs closed book: evidence-only {n10}, closed-only {n01}, p={pp:.2g}")
    # categories
    g = pd.DataFrame(dict(cat=HA.category.values, ok=ok, ha=ha_ok)).groupby("cat").agg(n=("ok", "size"), closed=("ok", "mean"), with_ev=("ha", "mean"))
    print((g[g.n >= 15] * [1, 100, 100]).round(1).to_string())
# ---------- repeats: test-retest
reps = {"rep0": HA}
for r in ("rep1", "rep2"):
    d = load_e2(f"{r}_haiku")
    if d is not None: reps[r] = d
if len(reps) > 1:
    print("\n== test-retest, Haiku, content + prompt A")
    dec = pd.DataFrame({k: v.decision.astype(str).str.upper().values for k, v in reps.items()}, index=keys)
    okm = pd.DataFrame({k: (dec[k].values == np.where(y == 1, "YES", "NO")).astype(float) for k in reps}, index=keys)
    for k in reps: print(f"  {k}: acc {100*okm[k].mean():.1f}; YES share {100*(dec[k]=='YES').mean():.1f}; mean conf {reps[k].confidence.mean():.3f}")
    ks = list(reps)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            print(f"  decisions differ {ks[i]} vs {ks[j]}: {int((dec[ks[i]] != dec[ks[j]]).sum())} of {len(dec)} ({100*(dec[ks[i]] != dec[ks[j]]).mean():.1f}%)")
    if len(ks) == 3:
        yes = (dec == "YES").sum(axis=1).values; maj = np.where(yes >= 2, "YES", "NO"); mok = (maj == np.where(y == 1, "YES", "NO")).astype(float)
        unan = (yes == 0) | (yes == 3)
        print(f"  majority of 3 repeats: acc {100*mok.mean():.1f} {ci(mok)}; unanimous on {100*unan.mean():.1f}% rows, acc there {100*mok[unan].mean():.1f}; split rows {int((~unan).sum())}, acc {100*mok[~unan].mean():.1f}")
        pd.DataFrame(dict(key=keys, yes_votes=yes, maj=maj, maj_ok=mok, unan=unan)).to_csv(f"{S}/exp/haiku_trio.csv", index=False)
# ---------- abstain
AB = load_e2("abstain_haiku")
if AB is not None:
    d = AB.decision.astype(str).str.upper().values; ins = d == "INSUFFICIENT"; ans = ~ins & np.isin(d, ["YES", "NO"])
    okab = (d == np.where(y == 1, "YES", "NO"))
    print(f"\n== abstention option, Haiku: answers {dict(pd.Series(d).value_counts())}")
    print(f"  INSUFFICIENT on {100*ins.mean():.1f}% rows; of those true YES {100*y[ins].mean():.1f}% (base {100*y.mean():.1f}%)")
    print(f"  accuracy on answered ({ans.sum()} rows, coverage {100*ans.mean():.1f}%): {100*okab[ans].mean():.1f}; recall YES among answered {100*okab[ans&(y==1)].mean():.1f}; recall NO {100*okab[ans&(y==0)].mean():.1f}")
    print(f"  binary run (HA) on the same answered rows: {100*ha_ok[ans].mean():.1f}; on the abstained rows: {100*np.nanmean(ha_ok[ins]):.1f}  (n={ins.sum()})")
    hd = HA.decision.astype(str).str.upper().values
    print(f"  what the binary run said on abstained rows: {dict(pd.Series(hd[ins]).value_counts())}; binary-run errors absorbed by abstention: {int(((ha_ok==0)&ins).sum())} of {int((ha_ok==0).sum())}; false NO absorbed {int(((ha_ok==0)&ins&(y==1)).sum())} of {int(((ha_ok==0)&(y==1)).sum())}; false YES absorbed {int(((ha_ok==0)&ins&(y==0)).sum())} of {int(((ha_ok==0)&(y==0)).sum())}")
    # compare to confidence gate at the same coverage
    conf = HA.confidence.values.astype(float); k = int(ans.sum()); o = np.argsort(-conf, kind="stable")[:k]
    print(f"  confidence gate of the binary run at the same coverage ({k} rows): accuracy {100*ha_ok[o].mean():.1f}")
    cab = AB.confidence.values.astype(float)
    # combined: answered & high confidence
    for thr in (0.9, 0.95):
        g = ans & (cab >= thr); print(f"  answered and confidence>={thr}: coverage {100*g.mean():.1f}%, accuracy {100*okab[g].mean():.1f} (errors {int((~okab[g]).sum())}/{g.sum()})")
# ---------- noul-style bare probabilities
print("\n== confidence signals compared (Haiku)")
sig = {}
def verb(d):
    dec = d.decision.astype(str).str.upper().values; c = d.confidence.values.astype(float)
    p = np.where(dec == "YES", c, 1 - c); ok = (dec == np.where(y == 1, "YES", "NO")).astype(float); ok[~np.isin(dec, ["YES", "NO"])] = np.nan
    return p, ok, c
sig["verbalized confidence (decision + confidence, with reasoning)"] = verb(HA)
for nm, lab in (("noul_haiku", "bare P(YES), no reasoning"), ("noulb_haiku", "bare P(evidence states it occurred), no reasoning")):
    d = load_e2(nm)
    if d is not None:
        p = d.p_yes.values.astype(float); ok = ((p > 0.5).astype(int) == y).astype(float); ok[np.isnan(p)] = np.nan
        sig[lab] = (p, ok, np.maximum(p, 1 - p))
if len(reps) == 3:
    ps = np.nanmean([verb(v)[0] for v in reps.values()], axis=0); ok = ((ps > 0.5).astype(int) == y).astype(float)
    sig["mean P(YES) of 3 verbalized repeats"] = (ps, ok, np.maximum(ps, 1 - ps))
    yes = np.sum([(v.decision.astype(str).str.upper().values == "YES") for v in reps.values()], axis=0)
    agree = np.where((yes == 0) | (yes == 3), 1.0, 0.0); mc = np.nanmean([v.confidence.values.astype(float) for v in reps.values()], axis=0)
    majok = (np.where(yes >= 2, 1, 0) == y).astype(float)
    sig["self-consistency: unanimous of 3 repeats + mean confidence"] = (np.where(yes >= 2, 0.5 + (agree + mc) / 4, 0.5 - (agree + mc) / 4), majok, agree + mc)
SIG = {}
for lab, (p, ok, conf) in sig.items():
    v = ~np.isnan(p) & ~np.isnan(ok) & ~np.isnan(conf); hi = v & (conf >= 0.9)
    SIG[lab] = dict(p=p, ok=ok, conf=conf)
    print(f"  {lab:62s} n={v.sum()} acc {100*ok[v].mean():.1f}  Brier {np.mean((p[v]-y[v])**2):.3f}  ECE {ece(conf[v], ok[v]):.3f}  AUC(P yes) {auc(p[v], y[v]):.3f}  AURC {aurc(conf[v], ok[v]):.4f}  conf>=0.9: {100*hi.mean():.0f}% rows at {100*ok[hi].mean():.1f}")
    for cov in (0.25, 0.5, 0.75):
        o = np.argsort(-np.where(v, conf, -1), kind="stable")[:int(cov * v.sum())]
        print(f"      top {int(cov*100)}% by confidence: accuracy {100*ok[o].mean():.1f} (errors {int((1-ok[o]).sum())}/{len(o)})", end="")
    print()
print(f"  reference: error rate of HA {1-np.nanmean(ha_ok):.4f}")
if "bare P(YES), no reasoning" in sig and "bare P(evidence states it occurred), no reasoning" in sig:
    a = sig["bare P(YES), no reasoning"][0]; b = sig["bare P(evidence states it occurred), no reasoning"][0]; v = ~np.isnan(a) & ~np.isnan(b)
    print(f"  bare P(YES) vs bare P(evidence states): correlation {np.corrcoef(a[v], b[v])[0,1]:.3f}; mean {a[v].mean():.3f} vs {b[v].mean():.3f}; share of exact 0 or 1: {100*np.mean((a[v]==0)|(a[v]==1)):.0f}% / {100*np.mean((b[v]==0)|(b[v]==1)):.0f}%; distinct values {len(np.unique(a[v]))} / {len(np.unique(b[v]))}")
    # combination with verbalized
    pv, okv, cv = sig["verbalized confidence (decision + confidence, with reasoning)"]
    comb = (pv + a) / 2; v = ~np.isnan(comb); okc = ((comb > 0.5).astype(int) == y).astype(float); cc = np.maximum(comb, 1 - comb)
    print(f"  average of verbalized and bare P(YES): acc {100*okc[v].mean():.1f} Brier {np.mean((comb[v]-y[v])**2):.3f} AURC {aurc(cc[v], okc[v]):.4f}")
    agree2 = ((pv > 0.5) == (a > 0.5)) & v; print(f"  verbalized decision and bare decision agree on {100*agree2.mean():.1f}% rows; accuracy when they agree {100*okv[agree2].mean():.1f}, when they disagree {100*np.nanmean(okv[v & ~agree2]):.1f} (n={int((v & ~agree2).sum())})")
import pickle; pickle.dump(dict(SIG=SIG, y=y, keys=keys, mk=mk), open(f"{S}/exp/e2_signals.pkl", "wb"))
# ---------- date ablation
DT = None
f = glob.glob(f"{S}/exp/e2_date_haiku_n*.csv")
if f:
    DT = pd.read_csv(f[0]).drop_duplicates("key").set_index("key"); kk = [k for k in DT.index if k in HA.index]
    a = HA.loc[kk]; b = DT.loc[kk]; n10, n01, p = mcnemar(b.ok.values == 1, a.ok.values == 1)
    print(f"\n== explicit current date in prompt (n={len(kk)}): with date {100*b.ok.mean():.1f} vs baseline {100*a.ok.mean():.1f}; date-only correct {n10}, baseline-only {n01}, p={p:.2f}; YES share {100*(b.decision=='YES').mean():.1f} vs {100*(a.decision=='YES').mean():.1f}")
print("\ncosts:", {os.path.basename(x): round(pd.read_csv(x).cost.sum(), 2) for x in glob.glob(f"{S}/exp/e2_*_n*.csv") if "_n6" not in x})
