import sys, os, json, glob, math, re
import pandas as pd, numpy as np
R = sys.argv[1]
A = pd.read_csv(f"{R}/results/Final_Architecture_A_results.csv")
B = pd.read_csv(f"{R}/results/Final_Architecture_B_results.csv")
print("retrieval_mode A", A.retrieval_mode.value_counts(dropna=False).to_dict(), "B", B.retrieval_mode.value_counts(dropna=False).to_dict())
# --- 3b cache
tot=emp=0; ne_files={}; hl_tot=0; hl_emp_hl=0
for f in glob.glob(f"{R}/cache/evidence/*.json"):
    e=json.load(open(f)); bn=os.path.basename(f)
    for s in e["sources"]:
        tot+=1
        t=(s.get("text") or "")
        if t.strip()=="" : emp+=1
        else: ne_files[bn]=ne_files.get(bn,0)+1
        if bn.startswith("highlights__"):
            hl_tot+=1
            if not s.get("highlights"): hl_emp_hl+=1
print("sources",tot,"empty text",emp,"nonempty files",ne_files)
print("highlights-packet sources",hl_tot,"of which with no highlights",hl_emp_hl)
# --- my own reimplementation of both formatters
def fmtA(e):
    L=[f"Question: {e['question_text']}",f"Resolution Criteria: {e['resolution_criteria']}",f"Resolution Date: {e['resolution_date']}","",f"Retrieved {len(e['sources'])} sources:",""]
    for i,s in enumerate(e["sources"],1):
        L+= [f"--- Source {i} ---",f"Title: {s['title']}",f"URL: {s['url']}"]
        if s.get("published_date"): L.append(f"Published: {s['published_date']}")
        hl=s.get("highlights") or []
        c=("\n".join(f"- {x}" for x in hl) if hl else (s.get("text") or "")).strip()
        L.append(f"\n{c}\n" if c else "\n(No extractable content)\n")
    return "\n".join(L)
def fmtB(e):
    if not e["sources"]: return "No evidence sources were retrieved."
    L=[]
    for i,s in enumerate(e["sources"],1):
        L+= [f"--- Source {i} ---",f"Title: {s['title']}",f"URL: {s['url']}"]
        if s.get("published_date"): L.append(f"Published: {s['published_date']}")
        L.append(s.get("text") or ""); L.append("")
    return "\n".join(L)
e=json.load(open(f"{R}/cache/evidence/highlights__row0089_KXSNLHOST_ddc2b51a4839.json"))
a,b=fmtA(e),fmtB(e)
print("Powell A chars",len(a),"Powell count",a.count("Powell"),"| B chars",len(b),"Powell",b.count("Powell"))
print("Powell in titles:",[i for i,s in enumerate(e['sources'],1) if 'Powell' in s['title']], "in highlights:",[i for i,s in enumerate(e['sources'],1) if 'Powell' in ' '.join(s.get('highlights') or [])])
print("Powell in question/criteria of A block header:", a[:400].count("Powell"))
la=[];lb=[]
for f in glob.glob(f"{R}/cache/evidence/highlights__*.json"):
    e2=json.load(open(f)); la.append(len(fmtA(e2))); lb.append(len(fmtB(e2)))
print("n hl packets",len(la),"median A",np.median(la),"median B",np.median(lb))
