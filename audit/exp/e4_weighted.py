"""Stratified estimates (weighted to all 430 rows) with bootstrap CIs for the agentic search runs."""
import sys, numpy as np, pandas as pd
S = "."
sys.path.insert(0, f"{S}/exp"); from common import rows_with_packets, load
R = rows_with_packets(); A, B, D = load(); Au = A.drop_duplicates("key").set_index("key").reindex(R.key)
E1 = pd.read_csv(f"{S}/exp/e1_haiku_nall_rep0_HA-TA-HB-TB.csv"); HA = E1[E1.cond == "HA"].set_index("key").reindex(R.key)
y = R.y.values; good = (Au.maj_ok.values == 1) & (HA.ok.values == 1)
N = {"err": int((~good).sum()), "cy": int((good & (y == 1)).sum()), "cn": int((good & (y == 0)).sum())}
HAk = HA.ok
def est(G, col):
    s = {"err": G[~G.control], "cy": G[G.control & (G.truth == "YES")], "cn": G[G.control & (G.truth == "NO")]}
    return sum(s[k][col].mean() * N[k] for k in s) / sum(N.values()), s
def boot(G, col, B=4000, seed=0):
    rng = np.random.default_rng(seed); _, s = est(G, col); out = []
    for _ in range(B):
        tot = 0
        for k, d in s.items():
            v = d[col].values; tot += v[rng.integers(0, len(v), len(v))].mean() * N[k]
        out.append(tot / sum(N.values()))
    return np.percentile(out, [2.5, 97.5])
for tag in ("haiku", "haiku_clean"):
    G = pd.read_csv(f"{S}/exp/e4_agent_{tag}.csv"); G = G[G.error.isna()].copy()
    G["ok"] = (G.decision == G.truth).astype(float); G["ab"] = (G.decision == "INSUFFICIENT").astype(float)
    G["wrong"] = ((G.decision != G.truth) & (G.decision != "INSUFFICIENT")).astype(float)
    G["fixed"] = HAk.reindex(G.key).values.astype(float); G["diff"] = G.ok - G.fixed
    print(f"== {tag}  (strata sizes {N})")
    for lab, sub in (("all sampled rows", G), ("rows that never touched a market site", G[~G.market_site])):
        a, _ = est(sub, "ok"); ab, _ = est(sub, "ab"); w, _ = est(sub, "wrong"); dd, _ = est(sub, "diff")
        lo, hi = boot(sub, "ok"); dlo, dhi = boot(sub, "diff")
        print(f"  {lab}: accuracy {100*a:.1f}% [{100*lo:.1f}, {100*hi:.1f}]; INSUFFICIENT {100*ab:.1f}%; wrong answers {100*w:.1f}%; "
              f"accuracy on answered {100*a/(1-ab):.1f}%; vs fixed-packet Haiku {100*dd:+.1f} pp [{100*dlo:+.1f}, {100*dhi:+.1f}]")
