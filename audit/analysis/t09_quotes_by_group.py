"""Quotation tracing restricted to rows where B provably used the same cache key (same packet) as A."""
import re, json, glob, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from kota import *
A, B, D = load()
cache = {os.path.basename(f)[:-5]: f for f in glob.glob(REPO + "/cache/evidence/*.json")}
Au = A.drop_duplicates("key").set_index("key"); Bu = B.drop_duplicates("key").set_index("key")
J = Au.join(Bu[["row_uid"] + [f"{m}_round{r}_reasoning" for m in M for r in (1, 2)] + [f"{m}_round1_confidence" for m in M] + [f"{m}_d1" for m in M] + [f"{m}_ok1" for m in M] + ["maj1_ok", "final_ok"]], rsuffix="_B", how="inner")
J["same_uid"] = J.row_uid == J.row_uid_B; J["fileA"] = ["highlights__" + u for u in J.row_uid]; J = J[J.fileA.isin(cache)]
norm_ = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
def quotes(r):
    q = re.findall(r"[\"“‘']([^\"”’']{25,300})[\"”’']", str(r))
    return [norm_(x) for x in q if len(norm_(x).split()) >= 5]
def trace(sub, cols):
    out = []
    for m, c in cols.items():
        n = t = h = qq = 0
        for k, r in sub.iterrows():
            e = json.load(open(cache[r.fileA]))
            titles = norm_(" | ".join((s.get("title") or "") for s in e["sources"])); hl = norm_(" | ".join(" ".join(s.get("highlights") or []) for s in e["sources"]))
            qc = norm_(e["question_text"] + " " + e["resolution_criteria"])
            for q in quotes(r[c]):
                n += 1
                if q in qc: qq += 1
                elif q in titles: t += 1
                elif q in hl: h += 1
        out.append(f"{NAMES[m]}: {h}/{n} from source text ({100*h/max(n,1):.1f}%), {t} from titles")
    return "; ".join(out)
for lab, sub in [("SAME cache key in A and B (same packet guaranteed)", J[J.same_uid]), ("DIFFERENT cache key (B re-queried Exa)", J[~J.same_uid])]:
    print(f"\n== {lab}: rows {len(sub)}")
    print("  A        :", trace(sub, {m: f"{m}_reasoning" for m in M}))
    print("  B round 1:", trace(sub, {m: f"{m}_round1_reasoning" for m in M}))
    print("  B round 2:", trace(sub, {m: f"{m}_round2_reasoning" for m in M}))
    y = sub.y.values
    print(f"  accuracy: A majority {100*sub.maj_ok.mean():.1f}; B R1 majority {100*sub.maj1_ok.mean():.1f}; B final {100*sub.final_ok.mean():.1f}; " + "; ".join(f"{NAMES[m]} A {100*sub[f'{m}_ok'].mean():.1f} -> B R1 {100*sub[f'{m}_ok1'].mean():.1f}" for m in M))
    n10, n01, p = mcnemar(sub.maj_ok, sub.maj1_ok); print(f"  A majority vs B R1 majority: A-only {n10}, B-only {n01}, p={p:.3g}")
    print(f"  YES share of majority: A {100*(sub.maj=='YES').mean():.1f} -> B R1 {100*(sum((sub[f'{m}_d1']=='YES').astype(int) for m in M)>=2).mean():.1f}; true {100*y.mean():.1f}; Llama conf==0: A {(sub.llama_confidence==0).sum()} B R1 {(sub.llama_round1_confidence==0).sum()}")
    dec_same = np.mean([(sub[f"{m}_d"] == sub[f"{m}_d1"]).mean() for m in M]); print(f"  mean share of identical decisions A vs B R1 per model: {100*dec_same:.1f}%")
# all rows, by same_uid (no cache needed)
JJ = Au.join(Bu[["row_uid"] + [f"{m}_ok1" for m in M] + ["maj1_ok", "final_ok"]], rsuffix="_B", how="inner"); JJ["same_uid"] = JJ.row_uid == JJ.row_uid_B
print("\n== all 1,153 shared rows by cache-key identity")
for lab, sub in [("same key", JJ[JJ.same_uid]), ("different key", JJ[~JJ.same_uid])]:
    n10, n01, p = mcnemar(sub.maj_ok, sub.maj1_ok)
    print(f"  {lab:13s} n={len(sub)} A majority {100*sub.maj_ok.mean():.1f}  B R1 majority {100*sub.maj1_ok.mean():.1f}  B final {100*sub.final_ok.mean():.1f}  (A-only {n10}, B-only {n01}, p={p:.2g})")
g = A.iloc[188]; print("\nGlen Powell: A uid", g.row_uid, "| B uid", B.iloc[188].row_uid)
