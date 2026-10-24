"""Analysis of batch 4 and the agentic search run."""
import sys, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
S = "."
sys.path.insert(0, f"{S}/exp"); sys.path.insert(0, f"{S}/analysis")
from common import rows_with_packets, load, mcnemar, cluster_boot, wilson
from scipy.stats import rankdata
R = rows_with_packets(); A, B, D = load(); Au = A.drop_duplicates("key").set_index("key").reindex(R.key)
month = pd.to_datetime([e["resolution_date"] for e in R.packet], utc=True, format="ISO8601").strftime("%Y-%m")
R["month"] = month; R["late"] = R.month >= "2025-08"
E1 = pd.read_csv(f"{S}/exp/e1_haiku_nall_rep0_HA-TA-HB-TB.csv"); HA = E1[E1.cond == "HA"].set_index("key").reindex(R.key)
def auc(s, y):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def aurc(score, ok):
    o = np.argsort(-np.asarray(score, float), kind="stable"); err = 1 - np.asarray(ok, float)[o]
    return (np.cumsum(err) / np.arange(1, len(err) + 1)).mean()
def load_e2(exp, model):
    f = f"{S}/exp/e2_{exp}_{model}_nall.csv"
    return pd.read_csv(f).set_index("key").reindex(R.key) if os.path.exists(f) else None
y = R.y.values; truth = R.truth.values
print(f"430 rows; close Aug-Nov 2025: {int(R.late.sum())}; base rate NO {100*(1-y.mean()):.1f}%")

print("\n== 1. Closed book (no evidence): leak check")
for m in ("haiku", "sonnet", "sonnet45", "haiku55"):
    c = load_e2("closed", m)
    if c is None or "p_yes" not in c: continue
    p = c.p_yes.astype(float).values; ok = ~np.isnan(p)
    acc = ((p > 0.5).astype(int) == y)
    s = f"  {m:9s} n={ok.sum()} acc {100*acc[ok].mean():.1f}% AUC {auc(p[ok], y[ok]):.2f} Brier {np.mean((p[ok]-y[ok])**2):.3f}"
    for lab, mk in (("May-Jul", ~R.late.values), ("Aug-Nov", R.late.values)):
        mm = ok & mk; s += f" | {lab}: acc {100*acc[mm].mean():.1f}% AUC {auc(p[mm], y[mm]):.2f} (n={mm.sum()})"
    print(s)

print("\n== 2. Sonnet 4.5 with the paper's prompt A and full text (rep1)")
S45 = load_e2("rep1", "sonnet45")
if S45 is not None:
    d = S45.decision.astype(str).str.upper().values; ok45 = (d == truth).astype(int); okH = HA.ok.values.astype(int)
    for lab, mk in (("all 430", np.ones(len(R), bool)), ("Aug-Nov only", R.late.values)):
        n10, n01, p = mcnemar(ok45[mk], okH[mk])
        print(f"  {lab}: Sonnet 4.5 {100*ok45[mk].mean():.1f}% vs Haiku 4.5 {100*okH[mk].mean():.1f}% ({n10} vs {n01}, p={p:.3f}); "
              f"recall YES {100*ok45[mk & (y==1)].mean():.1f} vs {100*okH[mk & (y==1)].mean():.1f}; recall NO {100*ok45[mk & (y==0)].mean():.1f} vs {100*okH[mk & (y==0)].mean():.1f}")
    M = {"haiku45": okH, "sonnet45": ok45, "gpt5nano": Au.gpt4o_ok.values, "deepseek": Au.deepseek_ok.values, "llama": Au.llama_ok.values}
    dec = {"haiku45": HA.decision.astype(str).str.upper().values, "sonnet45": d, "gpt5nano": Au.gpt4o_d.values, "deepseek": Au.deepseek_d.values, "llama": Au.llama_d.values}
    mk = R.late.values
    print("  error correlation (Aug-Nov rows):")
    E = pd.DataFrame({k: 1 - v[mk] for k, v in M.items()}); print(E.corr().round(2).to_string())
    import itertools
    print("  majority votes (Aug-Nov rows):")
    for combo in [("haiku45", "sonnet45", "gpt5nano"), ("haiku45", "sonnet45", "deepseek"), ("gpt5nano", "deepseek", "llama"), ("haiku45", "sonnet45", "gpt5nano", "deepseek", "llama")]:
        yes = sum((dec[c][mk] == "YES").astype(int) for c in combo); maj = np.where(yes > len(combo) / 2, "YES", "NO")
        okm = (maj == truth[mk]).astype(int); best = max(combo, key=lambda c: M[c][mk].mean())
        n10, n01, p = mcnemar(okm, M[best][mk])
        print(f"    {'+'.join(combo):45s} {100*okm.mean():.1f}%  best single {best} {100*M[best][mk].mean():.1f}%  ({n10} vs {n01}, p={p:.2f})")
N45 = load_e2("noul", "sonnet45"); NH = load_e2("noul", "haiku")
if N45 is not None and NH is not None:
    print("\n== 3. Bare P(YES) signal (noul prompt): AURC, lower is better (Aug-Nov rows)")
    mk = R.late.values
    for lab, Nn in (("haiku45", NH), ("sonnet45", N45)):
        p = Nn.p_yes.astype(float).values; dd = (p > 0.5).astype(int); okk = (dd == y).astype(float); conf = np.abs(p - 0.5)
        print(f"  {lab}: acc {100*okk[mk].mean():.1f}%  AURC {aurc(conf[mk], okk[mk]):.3f}  Brier {np.mean((p[mk]-y[mk])**2):.3f}")
AB = load_e2("abstain", "sonnet45")
if AB is not None:
    dd = AB.decision.astype(str).str.upper(); ans = dd.isin(["YES", "NO"]).values
    print(f"\n== 4. Sonnet 4.5 with abstention: INSUFFICIENT {100*(~ans).mean():.1f}%, accuracy on answered {100*(dd.values[ans] == truth[ans]).mean():.1f}%")
    AH = load_e2("abstain", "haiku"); da = AH.decision.astype(str).str.upper(); aa = da.isin(["YES", "NO"]).values
    print(f"   Haiku 4.5 for reference: INSUFFICIENT {100*(~aa).mean():.1f}%, accuracy on answered {100*(da.values[aa] == truth[aa]).mean():.1f}%")
    both = ans & aa; print(f"   abstain overlap: both abstain {int((~ans & ~aa).sum())}, only Sonnet {int((~ans & aa).sum())}, only Haiku {int((ans & ~aa).sum())}")

f = f"{S}/exp/e3_annot.csv"
if os.path.exists(f):
    d = pd.read_csv(f); d = d[d.label.notna()]
    d["grp"] = np.where(d.control, "control", np.where(d.a_err & d.h_err, "both wrong", np.where(d.a_err, "paper only", "haiku only")))
    print(f"\n== 5. Annotation of packets ({len(d)} labelled)"); print(pd.crosstab([d.grp, d.truth], d.label, margins=True).to_string())

for tag in ("haiku", "haiku_clean"):
  f = f"{S}/exp/e4_agent_{tag}.csv"
  if os.path.exists(f):
      G = pd.read_csv(f); G = G[G.error.isna()].copy()
      G["grp"] = np.where(G.control, "control", np.where(G.a_err & G.h_err, "both wrong", np.where(G.a_err, "paper only", "haiku only")))
      G["agent_ok"] = (G.decision == G.truth).astype(int); G["abst"] = (G.decision == "INSUFFICIENT").astype(int)
      hk = HA.reindex(G.key); G["fixed_haiku_ok"] = hk.ok.values; G["paper_maj_ok"] = Au.reindex(G.key).maj_ok.values
      print(f"\n== 6. Agentic search oracle [{tag}], {len(G)} rows")
      print(G.groupby(["grp", "truth"]).agg(n=("agent_ok", "size"), agent_right=("agent_ok", "sum"), insufficient=("abst", "sum"), fixed_haiku_right=("fixed_haiku_ok", "sum"),
                                            paper_right=("paper_maj_ok", "sum"), searches=("n_search", "mean"), market_site=("market_site", "sum")).round(2).to_string())
      an = pd.read_csv(f"{S}/exp/e3_annot.csv").set_index("key")
      G["label"] = an.reindex(G.key).label.values
      print("  agent result by packet label:"); print(G.groupby("label").agg(n=("agent_ok", "size"), agent_right=("agent_ok", "sum"), insufficient=("abst", "sum"), fixed_haiku_right=("fixed_haiku_ok", "sum")).to_string())
      clean = G[~G.market_site]
      print(f"  without rows that touched market sites: n={len(clean)}, agent right {int(clean.agent_ok.sum())}, fixed-packet Haiku right {int(clean.fixed_haiku_ok.sum())}")
      # population estimate for all 430: strata = error rows (all sampled) + both-correct rows split by truth (sampled 30+30)
      good_yes = int(((Au.maj_ok.values == 1) & (HA.ok.values == 1) & (y == 1)).sum()); good_no = int(((Au.maj_ok.values == 1) & (HA.ok.values == 1) & (y == 0)).sum())
      err = G[G.grp != "control"]; cy = G[(G.grp == "control") & (G.truth == "YES")]; cn = G[(G.grp == "control") & (G.truth == "NO")]
      n_err = int(((Au.maj_ok.values == 0) | (HA.ok.values == 0)).sum())
      est = (err.agent_ok.mean() * n_err + cy.agent_ok.mean() * good_yes + cn.agent_ok.mean() * good_no) / (n_err + good_yes + good_no)
      est_ab = (err.abst.mean() * n_err + cy.abst.mean() * good_yes + cn.abst.mean() * good_no) / (n_err + good_yes + good_no)
      print(f"  weighted estimate over all 430 rows: agent accuracy {100*est:.1f}% (INSUFFICIENT {100*est_ab:.1f}%) vs fixed-packet Haiku 84.2%, paper majority 81.6%")
      print(f"  strata: error rows {n_err} (sampled {len(err)}), correct YES {good_yes} (sampled {len(cy)}), correct NO {good_no} (sampled {len(cn)})")
      print("  cost", round(pd.read_csv(f).cost.fillna(0).sum(), 2), "mean searches", round(G.n_search.mean(), 2), "models used", G.models.value_counts().head(3).to_dict())
