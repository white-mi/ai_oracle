"""Analysis: debate with heterogeneous peers (deb2) and split evidence (split)."""
import sys, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
S = "."
sys.path.insert(0, f"{S}/exp"); sys.path.insert(0, f"{S}/analysis")
from common import rows_with_packets, load, mcnemar, cluster_boot
R = rows_with_packets(); A, B, D = load(); Au = A.drop_duplicates("key").set_index("key").reindex(R.key)
late = pd.to_datetime([e["resolution_date"] for e in R.packet], utc=True, format="ISO8601").strftime("%Y-%m") >= "2025-08"
E1 = pd.read_csv(f"{S}/exp/e1_haiku_nall_rep0_HA-TA-HB-TB.csv"); HA = E1[E1.cond == "HA"].set_index("key").reindex(R.key)
truth = R.truth.values
f = f"{S}/exp/e5_deb2.csv"
if os.path.exists(f):
    d = pd.read_csv(f)
    print(f"== Debate with a stronger peer (Haiku 4.5, Sonnet 4.5, GPT-5 nano), full text, Aug-Nov rows. errors {int(d.error.notna().sum())}, cost {d.cost.fillna(0).sum():.2f}")
    for ag in ("haiku", "sonnet45"):
        x = d[d.agent == ag]
        for lab, sub in (("disagree", x[~x.unanimous]), ("unanimous", x[x.unanimous])):
            ok1 = (sub.r1 == sub.truth).astype(int); ok2 = (sub.r2 == sub.truth).astype(int); fl = sub.r1 != sub.r2
            n10, n01, p = mcnemar(ok2, ok1)
            print(f"  {ag:9s} {lab:9s} n={len(sub):3d} r1 {100*ok1.mean():.1f}% r2 {100*ok2.mean():.1f}%  flips {int(fl.sum())} (c->w {int(((ok1==1)&(ok2==0)).sum())}, w->c {int(((ok1==0)&(ok2==1)).sum())}; YES->NO {int(((sub.r1=='YES')&(sub.r2=='NO')).sum())}, NO->YES {int(((sub.r1=='NO')&(sub.r2=='YES')).sum())}) p={p:.2f}")
    # majority before and after over all Aug-Nov rows (unsampled unanimous rows unchanged)
    keys = R.key[late].tolist(); tl = truth[late]
    S45 = pd.read_csv(f"{S}/exp/e2_rep1_sonnet45_nall.csv").set_index("key").reindex(keys)
    h1 = HA.reindex(keys).decision.astype(str).str.upper().values; s1 = S45.decision.astype(str).str.upper().values; g = Au.reindex(keys).gpt4o_d.values
    m2 = {(r.key, r.agent): r.r2 for r in d.itertuples()}
    h2 = np.array([m2.get((k, "haiku"), a) for k, a in zip(keys, h1)]); s2 = np.array([m2.get((k, "sonnet45"), a) for k, a in zip(keys, s1)])
    maj = lambda *v: np.where(sum((x == "YES").astype(int) for x in v) >= 2, "YES", "NO")
    M1 = maj(h1, s1, g); M2 = maj(h2, s2, g)
    for lab, v in (("Haiku r1", h1), ("Sonnet45 r1", s1), ("GPT-5 nano", g), ("majority r1", M1), ("Haiku r2", h2), ("Sonnet45 r2", s2), ("majority after debate", M2)):
        print(f"  all {len(keys)} Aug-Nov rows: {lab:22s} {100*(v == tl).mean():.1f}%")
    n10, n01, p = mcnemar(M2 == tl, M1 == tl); dd = (M2 == tl).astype(float) - (M1 == tl).astype(float)
    lo, hi = cluster_boot(dd, R.question_id[late].values, B=3000); print(f"  majority after - before: {100*dd.mean():+.2f} pp [{100*lo:+.2f}, {100*hi:+.2f}], {n10} vs {n01}, p={p:.3f}")
    n10, n01, p = mcnemar(s2 == tl, s1 == tl); print(f"  Sonnet45 after vs before: {n10} vs {n01}, p={p:.3f}")
f = f"{S}/exp/e3_split.csv"
if os.path.exists(f):
    x = pd.read_csv(f); x = x[x.error.isna()]
    print(f"\n== Split evidence: three Haiku agents, each sees a third of the sources. rows {x.key.nunique()}, calls {len(x)}")
    W = x.pivot_table(index="key", columns="agent", values="decision", aggfunc="first").reindex(R.key)
    ok = {a: (W[a].values == truth).astype(float) for a in W.columns}
    for a in W.columns: print(f"  agent {a}: {100*np.nanmean(ok[a]):.1f}%")
    yes = sum((W[a].values == "YES").astype(int) for a in W.columns); maj = np.where(yes >= 2, "YES", "NO"); okm = (maj == truth)
    print(f"  majority of the three split agents {100*okm.mean():.1f}% vs one agent with the full packet {100*HA.ok.mean():.1f}%")
    n10, n01, p = mcnemar(okm, HA.ok.values == 1); print(f"  McNemar split-majority vs full: {n10} vs {n01}, p={p:.3f}")
    E = pd.DataFrame({f"split{a}": 1 - ok[a] for a in W.columns}).corr(); print("  error correlation between split agents:"); print(E.round(2).to_string())
    r1 = pd.read_csv(f"{S}/exp/e2_rep1_haiku_nall.csv").set_index("key").reindex(R.key); r2 = pd.read_csv(f"{S}/exp/e2_rep2_haiku_nall.csv").set_index("key").reindex(R.key)
    F = pd.DataFrame({"full_rep0": 1 - HA.ok.values, "full_rep1": 1 - (r1.decision.astype(str).str.upper().values == truth), "full_rep2": 1 - (r2.decision.astype(str).str.upper().values == truth)}).astype(float)
    print("  error correlation between repeated full-packet runs:"); print(F.corr().round(2).to_string())
    unan = np.isin(yes, [0, 3]); print(f"  unanimous split agents: {100*unan.mean():.1f}% of rows, accuracy {100*okm[unan].mean():.1f}%; split: accuracy {100*okm[~unan].mean():.1f}%")
    print(f"  recall YES: split-majority {100*okm[R.y.values==1].mean():.1f}% vs full {100*HA.ok.values[R.y.values==1].mean():.1f}%; recall NO {100*okm[R.y.values==0].mean():.1f}% vs {100*HA.ok.values[R.y.values==0].mean():.1f}%")
