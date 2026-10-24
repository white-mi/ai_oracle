"""Analysis of experiment 1 (Claude Haiku 4.5, 430 rows): evidence content x prompt layout."""
from common import *
import numpy as np
E = pd.read_csv(f"{S}/exp/e1_haiku_nall_rep0_HA-TA-HB-TB.csv")
R = rows_with_packets()
W = E.pivot(index="key", columns="cond", values="ok"); D = E.pivot(index="key", columns="cond", values="decision"); C = E.pivot(index="key", columns="cond", values="confidence")
meta = R.set_index("key").loc[W.index]
y = meta.y.values; grp = meta.question_id.values
print("rows", len(W), "missing decisions:", D.isna().sum().to_dict())
W = W.fillna(0)
NAME = {"HA": "content + prompt A (as published A)", "HB": "content + prompt B", "TA": "titles only + prompt A", "TB": "titles only + prompt B (as published B round 1)"}
print("\n== accuracy, YES share, recall")
for c in ["HA", "HB", "TA", "TB"]:
    ok = W[c].values; lo, hi = cluster_boot(ok, grp, B=4000)
    print(f"{NAME[c]:48s} acc={100*ok.mean():.1f}% [{100*lo:.1f},{100*hi:.1f}]  YES share={100*(D[c]=='YES').mean():.1f}%  recall YES={100*ok[y==1].mean():.1f}%  recall NO={100*ok[y==0].mean():.1f}%  mean confidence={C[c].mean():.3f}  conf==0: {(C[c]==0).sum()}")
print("true YES share", y.mean().round(3), "n YES", int(y.sum()))
print("\n== paired contrasts (exact McNemar; difference with cluster-bootstrap 95% CI)")
def contrast(a, b, lab):
    n10, n01, p = mcnemar(W[a].values, W[b].values); d = W[a].values - W[b].values; lo, hi = cluster_boot(d, grp, B=4000)
    print(f"{lab:58s} {a}-only={n10:3d} {b}-only={n01:3d} diff={100*d.mean():+.1f} pp [{100*lo:+.1f},{100*hi:+.1f}] p={p:.2g}")
contrast("HA", "TA", "content effect under prompt A")
contrast("HB", "TB", "content effect under prompt B")
contrast("HA", "HB", "prompt effect with content")
contrast("TA", "TB", "prompt effect with titles only")
contrast("HA", "TB", "published A design vs published B round-1 design")
main_c = (W.HA + W.HB - W.TA - W.TB).values / 2; main_p = (W.HA + W.TA - W.HB - W.TB).values / 2
print(f"main effect of content: {100*main_c.mean():+.1f} pp CI {np.round(100*cluster_boot(main_c, grp, B=4000),1)}; main effect of prompt A vs B: {100*main_p.mean():+.1f} pp CI {np.round(100*cluster_boot(main_p, grp, B=4000),1)}")
# recall on YES contrasts
yy = y == 1
for a, b in [("HA", "TA"), ("HB", "TB"), ("HA", "HB")]:
    n10, n01, p = mcnemar(W[a].values[yy], W[b].values[yy]); print(f"  recall on true YES, {a} vs {b}: {100*W[a].values[yy].mean():.1f}% vs {100*W[b].values[yy].mean():.1f}%  ({a}-only {n10}, {b}-only {n01}, p={p:.2g})")
print("\n== decision changes when content is removed")
for a, b in [("HA", "TA"), ("HB", "TB")]:
    print(f"  {a}->{b}: YES->NO {int(((D[a]=='YES')&(D[b]=='NO')).sum())}, NO->YES {int(((D[a]=='NO')&(D[b]=='YES')).sum())}, unchanged {int((D[a]==D[b]).sum())}")
print("\n== same rows, published runs of the paper")
print("  A: " + ", ".join(f"{NAMES[m]} {100*meta[m+'_ok'].mean():.1f}%" for m in M) + f"; majority {100*meta.maj_ok.mean():.1f}%")
print("  B round 1: " + ", ".join(f"{NAMES[m]} {100*meta['B_'+m+'_ok1'].mean():.1f}%" for m in M) + f"; majority {100*meta.B_maj1_ok.mean():.1f}%; B final {100*meta.B_final_ok.mean():.1f}%")
for m in M:
    a = meta[m+"_ok"].values; b = meta["B_"+m+"_ok1"].values; yesA = (meta[m+"_d"]=="YES").mean(); yesB = (meta["B_"+m+"_d1"]=="YES").mean()
    print(f"  {NAMES[m]:14s} A minus B round 1 = {100*(a.mean()-b.mean()):+.1f} pp; YES share {100*yesA:.1f}% -> {100*yesB:.1f}%; recall YES {100*a[yy].mean():.1f}% -> {100*b[yy].mean():.1f}%")
print(f"  Haiku 4.5: content minus titles (prompt B) = {100*(W.HB.mean()-W.TB.mean()):+.1f} pp; YES share {100*(D.HB=='YES').mean():.1f}% -> {100*(D.TB=='YES').mean():.1f}%; recall YES {100*W.HB.values[yy].mean():.1f}% -> {100*W.TB.values[yy].mean():.1f}%")
print("\n== agreement of Haiku with the published models (same evidence, prompt A)")
for m in M:
    print(f"  Haiku HA vs {NAMES[m]} (A): same decision {100*(D.HA.values==meta[m+'_d'].values).mean():.1f}%; error correlation {np.corrcoef(1-W.HA.values, 1-meta[m+'_ok'].values)[0,1]:.3f}")
print(f"  Haiku TB vs published B round-1 decisions: " + ", ".join(f"{NAMES[m]} {100*(D.TB.values==meta['B_'+m+'_d1'].values).mean():.1f}%" for m in M))
# quotation tracing for Haiku to validate the method
import re
normt = lambda s: re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
def quotes(r): return [normt(x) for x in re.findall(r"[\"“‘']([^\"”’']{25,300})[\"”’']", str(r)) if len(normt(x).split()) >= 5]
Rr = E.pivot(index="key", columns="cond", values="reasoning")
print("\n== where do Haiku's direct quotations come from (validation of the tracing method)")
for c in ["HA", "HB", "TA", "TB"]:
    nq = t = h = 0
    for k, r in Rr[c].items():
        e = meta.loc[k, "packet"]; titles = normt(" | ".join(s.get("title") or "" for s in e["sources"])); hl = normt(" | ".join(" ".join(s.get("highlights") or []) for s in e["sources"])); qc = normt(e["question_text"] + " " + e["resolution_criteria"])
        for q in quotes(r):
            nq += 1
            if q in qc: pass
            elif q in titles: t += 1
            elif q in hl: h += 1
    print(f"  {c}: quotations={nq}, from titles={t}, from source text only={h} ({100*h/max(nq,1):.1f}%)")
print("\ncost of experiment 1: $", round(E.cost.sum(), 2))
