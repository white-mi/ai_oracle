import sys, os, math, glob, re
import pandas as pd, numpy as np
R=sys.argv[1]
A=pd.read_csv(f"{R}/results/Final_Architecture_A_results.csv"); B=pd.read_csv(f"{R}/results/Final_Architecture_B_results.csv")
up=lambda s: s.astype(str).str.strip().str.upper()
M=["gpt4o","deepseek","llama"]
def norm(s): return s.astype(str).str.replace(r"\s+"," ",regex=True).str.strip()
for df in (A,B): df["key"]=df.question_id.astype(str)+"|"+norm(df.question_text)+"|"+norm(df.resolution_criteria); df["t"]=up(df.ground_truth)
A["maj_ok"]=(np.where(sum((up(A[m+"_decision"])=="YES").astype(int) for m in M)>=2,"YES","NO")==A.t)
B["r1_ok"]=(np.where(sum((up(B[m+"_round1_decision"])=="YES").astype(int) for m in M)>=2,"YES","NO")==B.t)
B["fin_ok"]=up(B.final_decision)==B.t
# row_uid structure
print("row_uid pattern ok A:", A.row_uid.str.match(r"^row\d{4}_").all(), "B:", B.row_uid.str.match(r"^row\d{4}_").all())
print("A row_uid prefix==row_number:", (A.row_uid.str.slice(3,7).astype(int)==A.row_number).all(), "B:", (B.row_uid.str.slice(3,7).astype(int)==B.row_number).all())
print("A row_number resets", int((A.row_number.diff()<0).sum()), A.row_number.iloc[95:105].tolist(), "B monotonic 1..", B.row_number.is_monotonic_increasing, B.row_number.min(), B.row_number.max())
print("A distinct keys", A.key.nunique(), "B distinct", B.key.nunique(), "A dup rows", len(A)-A.key.nunique())
# duplicates in A: same row_uid?
d=A[A.key.duplicated(keep=False)]
print("A dup rows share row_uid?", d.groupby("key").row_uid.nunique().value_counts().to_dict())
Au=A.drop_duplicates("key").set_index("key"); Bu=B.drop_duplicates("key").set_index("key")
J=Au.join(Bu[["row_uid","r1_ok","fin_ok"]],rsuffix="_B",how="inner")
print("shared",len(J),"diff row_uid",int((J.row_uid!=J.row_uid_B).sum()))
# alternative: keep='last' for A dups
Au2=A.drop_duplicates("key",keep="last").set_index("key"); J2=Au2.join(Bu[["row_uid"]],rsuffix="_B",how="inner")
print("with keep=last for A dups: diff row_uid", int((J2.row_uid!=J2.row_uid_B).sum()))
# any A row (incl dups) whose row_uid equals B's for that key
anym=A.merge(B[["key","row_uid"]],on="key",suffixes=("","_B")); 
g=anym.groupby("key").apply(lambda x:(x.row_uid==x.row_uid_B).any())
print("contracts where no A copy matches B row_uid:", int((~g).sum()), "of", len(g))
pk=set(os.path.basename(f) for f in glob.glob(f"{R}/cache/evidence/highlights__*.json"))
print("A row_uids with committed packet", A.row_uid.map(lambda u:f"highlights__{u}.json" in pk).sum(), "unique", A.drop_duplicates('row_uid').row_uid.map(lambda u:f"highlights__{u}.json" in pk).sum())
print("B row_uids with committed packet", B.row_uid.map(lambda u:f"highlights__{u}.json" in pk).sum())
print("packets matching neither", len(pk - set("highlights__"+u+".json" for u in A.row_uid) - set("highlights__"+u+".json" for u in B.row_uid)))
J["same"]=J.row_uid==J.row_uid_B
J["pkA"]=J.row_uid.map(lambda u:f"highlights__{u}.json" in pk)
print("same key & committed:", int((J.same & J.pkA).sum()), " diff key & A packet committed:", int((~J.same & J.pkA).sum()), " total A-packet committed among shared:", int(J.pkA.sum()))
# paired decomposition on 1153 contracts
a=J.maj_ok.sum(); r1=J.r1_ok.sum(); f=J.fin_ok.sum()
print("paired 1153: A maj", a, "B r1 maj", r1, "B final", f, "gap", a-f, "of which r1", a-r1)
print("paired A maj only vs B final", int((J.maj_ok&~J.fin_ok).sum()), int((~J.maj_ok&J.fin_ok).sum()))
print("paired A maj only vs B r1", int((J.maj_ok&~J.r1_ok).sum()), int((~J.maj_ok&J.r1_ok).sum()))
# 1170 script pairing
a2,b2=A.copy(),B.copy()
for df in (a2,b2): df["mk"]=df.question_text.astype(str)+"||"+df.ground_truth.astype(str); df["occ"]=df.groupby("mk").cumcount()
P=a2.merge(b2,on=["mk","occ"],suffixes=("_a","_b"))
print("script pairs", len(P), "criteria differ", int((norm(P.resolution_criteria_a)!=norm(P.resolution_criteria_b)).sum()))
# Glen powell
print(A[A.row_uid.str.contains("KXSNLHOST")&A.resolution_criteria.str.contains("Glen Powell")].row_uid.tolist(), B[B.row_uid.str.contains("KXSNLHOST")&B.resolution_criteria.str.contains("Glen Powell")].row_uid.tolist())
print("SNLHOST rows A", A[A.row_uid.str.contains("KXSNLHOST")].row_uid.tolist()[:5], "B", B[B.row_uid.str.contains("KXSNLHOST")].row_uid.tolist()[:5])
