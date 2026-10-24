"""Key results restricted to markets that close after Jul 2025 (outside Haiku 4.5's training data)."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
S = "."
sys.path.insert(0, f"{S}/exp"); from common import rows_with_packets, load, mcnemar, cluster_boot
R = rows_with_packets(); A, B, D = load(); Au = A.drop_duplicates("key").set_index("key").reindex(R.key)
after = pd.to_datetime([e["resolution_date"] for e in R.packet], utc=True, format="ISO8601").strftime("%Y-%m").values >= "2025-08"
y = R.y.values; t = R.truth.values; q = R.question_id.values
E1 = pd.read_csv(f"{S}/exp/e1_haiku_nall_rep0_HA-TA-HB-TB.csv")
C = {c: E1[E1.cond == c].set_index("key").reindex(R.key) for c in ("HA", "TA", "HB", "TB")}
def aurc(s, ok):
    o = np.argsort(-np.asarray(s, float), kind="stable"); e = 1 - np.asarray(ok, float)[o]; return (np.cumsum(e) / np.arange(1, len(e) + 1)).mean()
for lab, mk in (("all 430", np.ones(len(R), bool)), ("Aug-Nov 340", after)):
    print(f"== {lab}")
    ok = {c: C[c].ok.values[mk].astype(float) for c in C}
    d = ok["HA"] - ok["TA"]; lo, hi = cluster_boot(d, q[mk], B=3000); print(f"  text vs titles (prompt A): {100*ok['HA'].mean():.1f} vs {100*ok['TA'].mean():.1f}, diff {100*d.mean():+.1f} [{100*lo:+.1f},{100*hi:+.1f}]")
    d = ok["HA"] - ok["HB"]; print(f"  prompt A vs B with text: {100*d.mean():+.1f} pp")
    pm = Au.maj_ok.values[mk]; n10, n01, p = mcnemar(ok["HA"], pm); print(f"  Haiku vs paper majority: {100*ok['HA'].mean():.1f} vs {100*pm.mean():.1f} ({n10} vs {n01}, p={p:.3f})")
    pub = {m: Au[f"{m}_ok"].values[mk] for m in ("gpt4o", "deepseek", "llama")}
    for m, v in pub.items():
        bb = B.drop_duplicates("key").set_index("key").reindex(R.key)
        print(f"    {m}: A {100*v.mean():.1f}%  B round 1 {100*bb[f'{m}_ok1'].values[mk].mean():.1f}%")
    N = pd.read_csv(f"{S}/exp/e2_noul_haiku_nall.csv").set_index("key").reindex(R.key); pp = N.p_yes.astype(float).values[mk]
    okn = ((pp > 0.5) == y[mk]).astype(float); comp = (Au.unan.values + Au.mean_conf.values)[mk]
    print(f"  AURC: Haiku P(YES) {aurc(np.abs(pp-0.5), okn):.3f} vs paper composite {aurc(comp, pm):.3f}")
    Ab = pd.read_csv(f"{S}/exp/e2_abstain_haiku_nall.csv").set_index("key").reindex(R.key); da = Ab.decision.astype(str).str.upper().values[mk]
    ans = np.isin(da, ["YES", "NO"]); print(f"  abstention: INSUFFICIENT {100*(~ans).mean():.1f}%, accuracy on answered {100*(da[ans]==t[mk][ans]).mean():.1f}%")
    Dd = pd.read_csv(f"{S}/exp/e3_deb.csv").set_index("key")
    full_h1 = C["HB"].decision.astype(str).str.upper().values; m2 = Dd.h2.to_dict(); h2 = np.array([m2.get(k, a) for k, a in zip(R.key, full_h1)])
    n10, n01, p = mcnemar((h2 == t)[mk], (full_h1 == t)[mk]); print(f"  debate round 2 vs round 1 (Haiku, full text): {100*(h2==t)[mk].mean():.1f} vs {100*(full_h1==t)[mk].mean():.1f} ({n10} vs {n01}, p={p:.2f})")
