"""Leakage audit: did Haiku 4.5 (reliable knowledge Feb 2025, training data to Jul 2025) know the outcomes?"""
import sys, os, json, glob, re, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
S = "."
sys.path.insert(0, f"{S}/exp"); from common import rows_with_packets, load, mcnemar, wilson
from scipy.stats import rankdata, fisher_exact
R = rows_with_packets(); A, B, D = load(); Au = A.drop_duplicates("key").set_index("key").reindex(R.key)
R["month"] = pd.to_datetime([e["resolution_date"] for e in R.packet], utc=True, format="ISO8601").strftime("%Y-%m")
R["after"] = R.month >= "2025-08"          # closes after the end of Haiku 4.5's training data (Jul 2025)
y = R.y.values; truth = R.truth.values
E1 = pd.read_csv(f"{S}/exp/e1_haiku_nall_rep0_HA-TA-HB-TB.csv"); HA = E1[E1.cond == "HA"].set_index("key").reindex(R.key)
CB = pd.read_csv(f"{S}/exp/e2_closed_haiku_nall.csv").set_index("key").reindex(R.key); p = CB.p_yes.astype(float).values
def auc(s, yy):
    r = rankdata(s); n1 = yy.sum(); n0 = len(yy) - n1; return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
print("== L1. Closed book by close month (Haiku 4.5)")
for m, g in R.groupby("month"):
    i = g.index.values; acc = ((p[i] > 0.5) == y[i]).mean(); base = max(y[i].mean(), 1 - y[i].mean())
    print(f"  {m}: n={len(i):3d} acc {100*acc:5.1f}%  majority-class {100*base:5.1f}%  AUC {auc(p[i], y[i]):.2f}  mean p_yes | YES {p[i][y[i]==1].mean():.2f} | NO {p[i][y[i]==0].mean():.2f}")
for lab, mk in (("before Aug 2025 (inside training data)", ~R.after.values), ("Aug-Nov 2025 (after training data)", R.after.values)):
    print(f"  {lab}: n={mk.sum()} AUC {auc(p[mk], y[mk]):.2f}, share of confident correct answers (p>0.9 & YES or p<0.1 & NO) {100*(((p[mk]>0.9)&(y[mk]==1))|((p[mk]<0.1)&(y[mk]==0))).mean():.1f}%")
print("\n== L2. Fixed-packet Haiku 4.5 (prompt A, full text) before vs after the training-data cutoff")
ok = HA.ok.values.astype(int)
for lab, mk in (("before Aug 2025", ~R.after.values), ("Aug-Nov 2025", R.after.values)):
    lo, hi = wilson(ok[mk].sum(), mk.sum()); pm = Au.maj_ok.values[mk]
    print(f"  {lab}: Haiku {100*ok[mk].mean():.1f}% [{100*lo:.1f},{100*hi:.1f}] (n={mk.sum()}), paper majority {100*pm.mean():.1f}%, Haiku minus paper {100*(ok[mk].mean()-pm.mean()):+.1f} pp")
an = pd.read_csv(f"{S}/exp/e3_annot.csv").set_index("key").reindex(R.key)
sil_yes = (an.label.values == "SILENT") & (y == 1)
print(f"\n== L3. Packets that do not contain the outcome (SILENT) with true YES: n={sil_yes.sum()}")
d = HA.decision.astype(str).str.upper().values
print(f"  fixed-packet Haiku answers YES on {int((d[sil_yes]=='YES').sum())} of {sil_yes.sum()} (if it remembered outcomes it would say YES here)")
print(f"  closed-book p_yes on these rows: mean {np.nanmean(p[sil_yes]):.2f}, share > 0.5: {100*np.nanmean(p[sil_yes] > 0.5):.0f}%")
for tag in ("haiku_clean",):
    G = pd.read_csv(f"{S}/exp/e4_agent_{tag}.csv"); G = G[G.error.isna()].copy()
    idx = R.reset_index().set_index("key").reindex(G.key)["index"].values
    G["cb_p"] = p[idx]; G["cb_ok"] = ((p[idx] > 0.5) == y[idx]).astype(int); G["after"] = R.after.values[idx]
    G["ok"] = (G.decision == G.truth).astype(int); G["fixed_ok"] = ok[idx]
    print(f"\n== L4. Agent with live search [{tag}] vs what Haiku knows without search")
    ct = pd.crosstab(G.cb_ok.map({1: "closed-book right", 0: "closed-book wrong"}), G.ok.map({1: "agent right", 0: "agent not right"})); print(ct.to_string())
    a = ct.values; print(f"  P(agent right | closed-book right) {a[1,1]/a[1].sum():.2f}  vs  P(agent right | closed-book wrong) {a[0,1]/a[0].sum():.2f}" if ct.shape == (2,2) else "")
    fn = G[G.a_err & G.h_err & (G.truth == "YES")]
    fixed = fn[fn.ok == 1]; notf = fn[fn.ok == 0]
    print(f"  shared false NO on fixed packets: {len(fn)}; agent fixed {len(fixed)}. Closed-book p_yes on fixed ones: mean {fixed.cb_p.mean():.2f}, >0.5 in {int((fixed.cb_p>0.5).sum())} of {len(fixed)}")
    # after-cutoff only, weighted to the 340 after-cutoff rows
    good = (Au.maj_ok.values == 1) & (ok == 1)
    N = {"err": int((~good & R.after.values).sum()), "cy": int((good & (y == 1) & R.after.values).sum()), "cn": int((good & (y == 0) & R.after.values).sum())}
    Ga = G[G.after]; s = {"err": Ga[~Ga.control], "cy": Ga[Ga.control & (Ga.truth == "YES")], "cn": Ga[Ga.control & (Ga.truth == "NO")]}
    def est(col): return sum(s[k][col].mean() * N[k] for k in s) / sum(N.values())
    Ga["ab"] = (Ga.decision == "INSUFFICIENT").astype(int); Ga["wrong"] = ((Ga.decision != Ga.truth) & (Ga.decision != "INSUFFICIENT")).astype(int)
    s = {"err": Ga[~Ga.control], "cy": Ga[Ga.control & (Ga.truth == "YES")], "cn": Ga[Ga.control & (Ga.truth == "NO")]}
    rng = np.random.default_rng(0); bo = []
    for _ in range(4000):
        t = 0
        for k, dd in s.items():
            v = dd.ok.values; t += v[rng.integers(0, len(v), len(v))].mean() * N[k]
        bo.append(t / sum(N.values()))
    lo, hi = np.percentile(bo, [2.5, 97.5])
    print(f"  Aug-Nov 2025 markets only (after training data): sampled {len(Ga)} (strata {[(k, len(v)) for k, v in s.items()]}, population {N})")
    print(f"    agent accuracy {100*est('ok'):.1f}% [{100*lo:.1f},{100*hi:.1f}], INSUFFICIENT {100*est('ab'):.1f}%, wrong {100*est('wrong'):.1f}%; fixed-packet Haiku on the same 340 rows {100*ok[R.after.values].mean():.1f}% (wrong {100*(1-ok[R.after.values].mean()):.1f}%)")
    # groundedness: cited URLs present in search results
    raw = glob.glob(f"{S}/agent/{tag}/raw/*.jsonl"); tot = inres = 0; empty = 0
    for f in raw:
        L = [json.loads(l) for l in open(f) if l.startswith("{")]
        links = set(); fin = None
        for l in L:
            if l.get("type") == "user" and isinstance(l["message"].get("content"), list):
                for c in l["message"]["content"]:
                    if c.get("type") == "tool_result": links |= set(re.findall(r'"url":"([^"]+)"', str(c.get("content"))))
            if l.get("type") == "result": fin = l
        src = ((fin or {}).get("structured_output") or {}).get("sources") or []
        if not src: empty += 1
        norm = lambda u: u.split("#")[0].rstrip("/").replace("http://", "https://").replace("://www.", "://")
        L2 = {norm(u) for u in links}
        for u in src:
            tot += 1; inres += norm(u) in L2
    print(f"  groundedness: {inres} of {tot} cited URLs ({100*inres/max(tot,1):.0f}%) appear among the returned search results; runs without cited sources {empty} of {len(raw)}")
