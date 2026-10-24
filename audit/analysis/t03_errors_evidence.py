"""Error structure, trivial baselines, query design and evidence dating (Architecture A, published outputs)."""
import numpy as np, pandas as pd, re
from scipy.stats import fisher_exact
from kota import *
A, B, D = load(); pk = packets()
Ad = A.drop_duplicates("key").copy()          # 1,153 distinct questions
P = print
P("== 1. Trivial baselines on the 1,153 distinct rows of file A")
y = Ad.y.values
P(f"n={len(Ad)} YES share={y.mean():.3f}; always-NO accuracy={100*(1-y.mean()):.2f}; A majority={100*Ad.maj_ok.mean():.2f}; weighted={100*Ad.wtd_ok.mean():.2f}")
for m in M: P(f"  {NAMES[m]:14s} acc={100*Ad[m+'_ok'].mean():.2f} YES-rate={100*(Ad[m+'_d']=='YES').mean():.1f} recall YES={100*Ad.loc[Ad.y==1,m+'_ok'].mean():.1f} recall NO={100*Ad.loc[Ad.y==0,m+'_ok'].mean():.1f}")
fn = int(((Ad.maj=='NO')&(Ad.y==1)).sum()); fp = int(((Ad.maj=='YES')&(Ad.y==0)).sum())
P(f"majority errors={fn+fp}: false NO={fn} ({100*fn/(fn+fp):.0f}%), false YES={fp}; precision of YES answers={100*((Ad.maj=='YES')&(Ad.y==1)).sum()/(Ad.maj=='YES').sum():.1f}% (n={(Ad.maj=='YES').sum()}); precision of NO answers={100*((Ad.maj=='NO')&(Ad.y==0)).sum()/(Ad.maj=='NO').sum():.1f}% (n={(Ad.maj=='NO').sum()})")
P("by vote pattern (YES votes): " + "; ".join(f"{k} votes: n={len(g)} true-YES share={100*g.y.mean():.0f}%" for k, g in Ad.groupby('yes_votes_n')))

P("\n== 2. Query design: rows whose question text is shared with a sibling row (query lacks the specific outcome)")
shared = Ad.qkey.isin(D.qkey[D.qkey.duplicated(keep=False)])
for lab, mk in [("shared generic question text", shared), ("unique question text", ~shared)]:
    g = Ad[mk]; P(f"  {lab:30s} n={len(g):4d} acc={100*g.maj_ok.mean():.1f} recall YES={100*g.loc[g.y==1,'maj_ok'].mean():.1f} (n={int(g.y.sum())}) recall NO={100*g.loc[g.y==0,'maj_ok'].mean():.1f} YES share={100*g.y.mean():.0f}%")
ex = D[D.qkey.duplicated(keep=False)].sort_values("qkey").head(4)
for _, r in ex.iterrows(): P("   e.g.", r.id, "| Q:", r.question[:60], "| rules:", r.description[:90])
tab = [[int(Ad.loc[shared&(Ad.y==1),'maj_ok'].sum()), int((1-Ad.loc[shared&(Ad.y==1),'maj_ok']).sum())],[int(Ad.loc[~shared&(Ad.y==1),'maj_ok'].sum()), int((1-Ad.loc[~shared&(Ad.y==1),'maj_ok']).sum())]]
P("  Fisher p (recall on YES, shared vs unique):", round(fisher_exact(tab)[1], 4), tab)

P("\n== 3. Evidence dating in the cached packets (file A rows that map to a committed packet)")
def feats(e):
    rd = pd.to_datetime(e["resolution_date"], utc=True)
    ds = [pd.to_datetime(s.get("published_date"), utc=True, errors="coerce") for s in e["sources"]]
    ds = [d for d in ds if not pd.isna(d)]
    lag = [(d - rd).total_seconds()/86400 for d in ds]
    return dict(n_src=len(e["sources"]), n_dated=len(ds), late=any(l >= -1 for l in lag), n_late=sum(l >= -1 for l in lag), within7=sum(-7 <= l for l in lag), newest_lag=max(lag) if lag else np.nan,
                median_lag=np.median(lag) if lag else np.nan, chars=sum(len(" ".join(s.get("highlights") or [])) for s in e["sources"]))
F = pd.DataFrame([feats(pk[k]) if k in pk else {} for k in Ad.qkey], index=Ad.index)
X = pd.concat([Ad, F], axis=1); X = X[X.n_src.notna()].copy(); X["late"] = X["late"].astype(bool)
P(f"rows of A (distinct) with a committed packet: {len(X)} of {len(Ad)}; packets used: {X.qkey.nunique()}; accuracy on them {100*X.maj_ok.mean():.1f}")
P(f"packets: median sources={X.n_src.median():.0f}, median dated={X.n_dated.median():.0f}; share of rows whose packet has NO source dated >= close-1d: {100*(~X.late).mean():.1f}%; median 'newest source minus close' = {X.newest_lag.median():.1f} days; median of median lag = {X.median_lag.median():.0f} days")
for lab, mk in [("late source present", X.late), ("no late source", ~X.late)]:
    g = X[mk]; P(f"  {lab:20s} n={len(g):4d} acc={100*g.maj_ok.mean():.1f}  recall YES={int(g.loc[g.y==1,'maj_ok'].sum())}/{int(g.y.sum())}={100*g.loc[g.y==1,'maj_ok'].mean():.1f}%  recall NO={100*g.loc[g.y==0,'maj_ok'].mean():.1f}%  unanimous share={100*g.unan.mean():.0f}%")
t = [[int(X.loc[X.late&(X.y==1),'maj_ok'].sum()), int((1-X.loc[X.late&(X.y==1),'maj_ok']).sum())],[int(X.loc[~X.late&(X.y==1),'maj_ok'].sum()), int((1-X.loc[~X.late&(X.y==1),'maj_ok']).sum())]]
P("  Fisher p (recall YES, late vs none):", round(fisher_exact(t)[1], 5), t)
bins = pd.cut(X.newest_lag, [-10000, -30, -7, -1, 0, 1.01]); 
P("  by newest-source lag (days relative to close):"); P(X.groupby(bins, observed=True).agg(n=('maj_ok','size'), acc=('maj_ok','mean'), yes_share=('y','mean'), pred_yes=('yes_votes_n', lambda v: (v>=2).mean())).round(3).to_string())
P("  by category (share of rows without late source):"); P(X.groupby('category').agg(n=('late','size'), no_late=('late', lambda v: 1-v.mean()), acc=('maj_ok','mean')).query('n>=15').round(2).to_string())
X.to_pickle("X_packets.pkl")

P("\n== 4. Why NO? keyword scan of reasoning for false-NO vs true-NO (GPT-5 nano reasoning, all distinct rows)")
pat = r"no (definitive|explicit|direct|clear|concrete|conclusive)? ?(evidence|confirmation|source|indication|report)|not (explicitly )?(confirm|state|mention|report|indicate)|does not (confirm|provide|show|state|mention)|do not (confirm|provide|show|state|mention)|absence of|insufficient|cannot (confirm|verify|determine)|none of the (sources|provided)|no information|not enough|lack of"
for lab, mk in [("false NO (truth YES)", (Ad.maj=='NO')&(Ad.y==1)), ("true NO", (Ad.maj=='NO')&(Ad.y==0)), ("true YES", (Ad.maj=='YES')&(Ad.y==1))]:
    r = Ad.loc[mk, 'gpt4o_reasoning'].astype(str)
    P(f"  {lab:22s} n={len(r):4d} share with absence-of-evidence wording: {100*r.str.contains(pat, flags=re.I, regex=True).mean():.0f}%")

P("\n== 5. Two-row markets: logical consistency of the pair (distinct rows)")
g = Ad.groupby("question_id").filter(lambda x: len(x) == 2).groupby("question_id")
pairs = pd.DataFrame({"truth_yes": g.y.sum(), "pred_yes": g.apply(lambda x: (x.maj=='YES').sum()), "ok": g.maj_ok.sum()})
P(f"pairs={len(pairs)}; truth pattern (number of YES in pair): {pairs.truth_yes.value_counts().sort_index().to_dict()}; predicted pattern: {pairs.pred_yes.value_counts().sort_index().to_dict()}")
P(pd.crosstab(pairs.truth_yes, pairs.pred_yes, rownames=['true YES in pair'], colnames=['predicted YES in pair']).to_string())
