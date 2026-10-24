"""Misc checks for the problem list."""
import re, json, glob, numpy as np, pandas as pd, warnings
from urllib.parse import urlparse
warnings.filterwarnings("ignore")
from kota import *
A, B, D = load(); pk = packets()
print("== Companies / World: A weighted vs B final (paired by file position; rows aligned?)")
same = (A.key.values == B.key.values).mean(); print(f"share of file positions where A and B hold the same row: {same:.3f}")
J = A.drop_duplicates("key").merge(B.drop_duplicates("key")[["key","final_ok","maj1_ok"]], on="key")
for cat in ["Companies", "World", "Politics", "Sports"]:
    j = J[J.category == cat]; n10, n01, p = mcnemar(j.wtd_ok, j.final_ok)
    print(f"  {cat:10s} n={len(j)} A wtd {100*j.wtd_ok.mean():.1f} B final {100*j.final_ok.mean():.1f}; A-only {n10}, B-only {n01}, exact McNemar p={p:.3f}")
print("\n== Agent failures / missing values")
for m in M:
    print(f"  A {m}: decision values {dict(pd.Series(A[f'{m}_d']).value_counts())}; conf NaN {A[f'{m}_confidence'].isna().sum()}")
    print(f"  B {m}: R1 {dict(pd.Series(B[f'{m}_d1']).value_counts())} R2 {dict(pd.Series(B[f'{m}_d2']).value_counts())}")
print("  columns A:", [c for c in A.columns if "lat" in c or "token" in c or "cost" in c or "error" in c][:20])
print("  columns B:", [c for c in B.columns if "lat" in c or "token" in c or "cost" in c or "error" in c or "tie" in c or "method" in c][:20])
for c in [c for c in B.columns if "tie" in c or "method" in c]: print("   ", c, dict(B[c].value_counts()))
print("\n== Llama confidence exactly 0 in A")
z = A[A.llama_confidence == 0]; print(f"  rows {len(z)}; decisions {dict(z.llama_d.value_counts())}; Llama acc there {100*z.llama_ok.mean():.1f}; truth YES share {100*z.y.mean():.1f}")
zb = B[B.llama_round1_confidence == 0]; print(f"  B R1: rows {len(zb)}; decisions {dict(zb.llama_d1.value_counts())}; acc {100*zb.llama_ok1.mean():.1f}; truth YES share {100*zb.y.mean():.1f}")
print("\n== Market's own page or other prediction markets among sources")
dom_ct = {}; rows = []
for k, e in pk.items():
    doms = [urlparse(s.get("url") or "").netloc.replace("www.", "") for s in e["sources"]]
    for d in set(doms): dom_ct[d] = dom_ct.get(d, 0) + 1
    rows.append(dict(qkey=k, kalshi=any("kalshi" in d for d in doms), poly=any("polymarket" in d for d in doms),
                     pm=any(any(x in d for x in ("kalshi", "polymarket", "metaculus", "manifold", "predictit", "goodjudgment", "robinhood")) for d in doms), n=len(doms)))
P = pd.DataFrame(rows); print(f"  packets {len(P)}; with a kalshi.com source {P.kalshi.sum()} ({100*P.kalshi.mean():.1f}%); polymarket {P.poly.sum()} ({100*P.poly.mean():.1f}%); any prediction-market site {P.pm.sum()} ({100*P.pm.mean():.1f}%)")
top = sorted(dom_ct.items(), key=lambda x: -x[1])[:25]; print("  top domains (packets containing):", top)
Ad = A.drop_duplicates("key").merge(P, on="qkey", how="inner")
for lab, mk in [("kalshi source present", Ad.kalshi), ("absent", ~Ad.kalshi)]:
    s = Ad[mk]; print(f"  {lab:22s} rows {len(s)} A majority acc {100*s.maj_ok.mean():.1f} recall YES {100*s[s.y==1].maj_ok.mean():.1f} (n YES {int(s.y.sum())})")
print("\n== All three wrong in A (distinct rows)")
Au = A.drop_duplicates("key"); aw = (Au.gpt4o_ok + Au.deepseek_ok + Au.llama_ok) == 0
print(f"  distinct rows {len(Au)}; all three wrong {aw.sum()} ({100*aw.mean():.1f}%), of which true YES {int((aw & (Au.y==1)).sum())}")
print(f"  at least one model right among A-majority errors: {int(((Au.maj_ok==0) & ~aw).sum())} of {int((Au.maj_ok==0).sum())}")
print("\n== Confusion matrix A majority / A weighted / B final (file rows)")
for nm, df, col in [("A majority", A, "maj"), ("A weighted", A, "wtd"), ("B R1 majority", B, "maj1"), ("B final", B, "final")]:
    y = df.y.values == 1; p = df[col].values == "YES"
    print(f"  {nm:14s} TP {int((p&y).sum())} FN {int((~p&y).sum())} TN {int((~p&~y).sum())} FP {int((p&~y).sum())} | recall YES {100*(p&y).sum()/y.sum():.1f} recall NO {100*(~p&~y).sum()/(~y).sum():.1f} precision YES {100*(p&y).sum()/p.sum():.1f} YES share {100*p.mean():.1f}")
for m in M:
    y = A.y.values == 1; p = A[f"{m}_d"].values == "YES"
    print(f"  A {NAMES[m]:14s} recall YES {100*(p&y).sum()/y.sum():.1f} recall NO {100*(~p&~y).sum()/(~y).sum():.1f} YES share {100*p.mean():.1f}")
