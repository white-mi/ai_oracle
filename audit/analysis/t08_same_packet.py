"""Did A and B see the same evidence packet per row? Check via cache keys and via source-index/date statements in reasoning."""
import re, json, glob, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from kota import *
A, B, D = load()
cache = {os.path.basename(f)[:-5]: f for f in glob.glob(REPO + "/cache/evidence/*.json")}
Au = A.drop_duplicates("key").set_index("key"); Bu = B.drop_duplicates("key").set_index("key")
J = Au.join(Bu[["row_uid"] + [f"{m}_round1_reasoning" for m in M] + [f"{m}_d1" for m in M]], rsuffix="_B", how="inner")
J["same_uid"] = J.row_uid == J.row_uid_B
print("distinct shared rows", len(J), "same row_uid in A and B:", int(J.same_uid.sum()), "different:", int((~J.same_uid).sum()))
J["fileA"] = ["highlights__" + u for u in J.row_uid]; J["fileB"] = ["highlights__" + u for u in J.row_uid_B]
J["hasA"] = J.fileA.isin(cache); J["hasB"] = J.fileB.isin(cache)
print("rows whose A cache key is in committed cache:", int(J.hasA.sum()), "| B cache key in committed cache:", int(J.hasB.sum()), "| both:", int((J.hasA & J.hasB).sum()), "| both and same file:", int((J.hasA & J.hasB & J.same_uid).sum()))
print(pd.crosstab(J.same_uid, [J.hasA, J.hasB]))
MON = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
pat = re.compile(r"Source (\d{1,2})\s*\((?:[^()]{0,40}?,\s*)?(?:published |dated |from )?(?:on )?([A-Z][a-z]{2})[a-z]*\.? (?:\d{1,2}, )?(\d{4})\)")
def check(reasoning, pkt):
    ok = bad = 0
    for mo in pat.finditer(str(reasoning)):
        k = int(mo.group(1)); mon = MON.get(mo.group(2).lower()); yr = int(mo.group(3))
        if not mon or k < 1 or k > len(pkt["sources"]): continue
        d = pd.to_datetime(pkt["sources"][k - 1].get("published_date"), utc=True, errors="coerce")
        if pd.isna(d): continue
        if d.month == mon and d.year == yr: ok += 1
        else: bad += 1
    return ok, bad
rows = []
for k, r in J[J.hasA].iterrows():
    pkt = json.load(open(cache[r.fileA]))
    a_ok = a_bad = b_ok = b_bad = 0
    for m in M:
        o, b = check(r[f"{m}_reasoning"], pkt); a_ok += o; a_bad += b
        o, b = check(r[f"{m}_round1_reasoning"], pkt); b_ok += o; b_bad += b
    rows.append(dict(key=k, same_uid=r.same_uid, hasB=r.hasB, a_ok=a_ok, a_bad=a_bad, b_ok=b_ok, b_bad=b_bad))
R = pd.DataFrame(rows)
print("\nstatements of the form 'Source k (Month Year)' checked against the packet committed for A's cache key")
print("A reasoning: consistent", R.a_ok.sum(), "inconsistent", R.a_bad.sum())
for lab, mk in [("B R1, same row_uid as A", R.same_uid), ("B R1, different row_uid", ~R.same_uid)]:
    s = R[mk]; print(f"{lab}: rows {len(s)}; consistent {s.b_ok.sum()}, inconsistent {s.b_bad.sum()}; rows with any statement {int(((s.b_ok+s.b_bad)>0).sum())}, rows with >=1 inconsistent {int((s.b_bad>0).sum())}")
# B's own cache key committed? then compare the two packets directly
both = J[J.hasA & J.hasB & ~J.same_uid]
same_urls = 0; jacc = []
for k, r in both.iterrows():
    pa = json.load(open(cache[r.fileA])); pb = json.load(open(cache[r.fileB]))
    ua = [s["url"] for s in pa["sources"]]; ub = [s["url"] for s in pb["sources"]]
    same_urls += ua == ub; jacc.append(len(set(ua) & set(ub)) / max(1, len(set(ua) | set(ub))))
if len(both): print(f"\nrows where A and B used different cache keys and both packets are committed: {len(both)}; identical ordered URL lists: {same_urls}; mean Jaccard of URL sets {np.mean(jacc):.2f}")
R.to_pickle("t08_R.pkl"); J[["row_uid", "row_uid_B", "same_uid", "hasA", "hasB"]].to_pickle("t08_J.pkl")
i = list(Au.index).index(A.iloc[381].key); print("\nMedicaid row: A uid", A.iloc[381].row_uid, "| B uid", B.iloc[381].row_uid)
