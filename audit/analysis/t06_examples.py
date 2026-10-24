"""Extra checks for the problem list: (1) do B agents say they got no content, (2) the paper's four worked examples."""
import re, json, numpy as np, pandas as pd
from kota import *
A, B, D = load(); pk = packets()
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 200)

print("== 1. Reasoning text that says the evidence has titles only / no content")
pat = re.compile(r"(only (the )?(titles?|headlines?|urls?|metadata)|titles? (and|or) (urls?|dates?|publication)|no (actual |source |article |full )?(content|text|excerpts?|snippets?|body)|content (is|was|are|were) not (provided|available|included)|without (the |any )?(actual |article |source |full )?(content|text)|(titles?|headlines?) alone|based (solely |only )?on (the )?(titles?|headlines?)|not (provided|included|available) in the evidence|(articles?|sources?)'? (contents?|texts?) (is|are|was|were) not)", re.I)
rows = []
for m in M:
    a = A[f"{m}_reasoning"].astype(str); b1 = B[f"{m}_round1_reasoning"].astype(str); b2 = B[f"{m}_round2_reasoning"].astype(str)
    rows.append(dict(model=NAMES[m], A=int(a.str.contains(pat).sum()), B_round1=int(b1.str.contains(pat).sum()), B_round2=int(b2.str.contains(pat).sum()),
                     A_len=int(a.str.len().median()), B1_len=int(b1.str.len().median())))
print(pd.DataFrame(rows).to_string(index=False))
anyA = np.any([A[f"{m}_reasoning"].astype(str).str.contains(pat).values for m in M], axis=0)
anyB = np.any([B[f"{m}_round1_reasoning"].astype(str).str.contains(pat).values for m in M], axis=0)
print(f"rows where at least one agent says so: A {anyA.sum()} of {len(A)} ({100*anyA.mean():.1f}%), B round 1 {anyB.sum()} of {len(B)} ({100*anyB.mean():.1f}%)")
ex = 0
for m in M:
    for r in B[f"{m}_round1_reasoning"].astype(str):
        mo = pat.search(r)
        if mo and ex < 8:
            s = max(0, mo.start() - 160); print(f"  [{NAMES[m]}] ...{r[s:mo.end()+160]}..."); ex += 1; break
for m in M:
    c = 0
    for r in B[f"{m}_round1_reasoning"].astype(str):
        mo = pat.search(r)
        if mo:
            c += 1
            if c in (5, 25):
                s = max(0, mo.start() - 160); print(f"  [{NAMES[m]}] ...{r[s:mo.end()+160]}...")

print("\n== 2. The four worked examples of the paper")
def show(sub, label):
    print(f"\n--- {label}")
    for df, nm in ((A, "A"), (B, "B")):
        hit = df[df.question_text.astype(str).str.contains(sub, case=False, regex=True)]
        for i, r in hit.head(3).iterrows():
            print(f"[{nm}] file row {i} (1-based {i+1}) id={r.question_id} truth={r.truth} cat={r.category}")
            print("    Q:", str(r.question_text)[:230])
            if nm == "A":
                print("    A decisions:", {NAMES[m]: (r[f'{m}_d'], r[f'{m}_confidence']) for m in M})
            else:
                print("    B R1:", {NAMES[m]: (r[f'{m}_d1'], r[f'{m}_round1_confidence']) for m in M}, " R2:", {NAMES[m]: r[f'{m}_d2'] for m in M}, "final", r.final)
            e = pk.get(r.qkey)
            if e is not None and nm == "A":
                rd = pd.to_datetime(e["resolution_date"], utc=True)
                print(f"    packet: resolution_date={e['resolution_date']} query={e.get('query','')[:120]!r} n_sources={len(e['sources'])}")
                for s in e["sources"]:
                    d = pd.to_datetime(s.get("published_date"), utc=True, errors="coerce")
                    hl = " ".join(s.get("highlights") or [])
                    print(f"      {str(d)[:10]:10s} (close{(d-rd).days:+5d}d) text_len={len(s.get('text') or ''):4d} hl_len={len(hl):5d} | {str(s.get('title'))[:90]}")
            elif nm == "A": print("    packet not in committed cache")
    return
show(r"Glen Powell hosts", "Retrieval failure example (p.17, Row 0189)")
show(r"S&P 500.*6699\.99", "Temporal reasoning example (p.17, Row 0726)")
show(r"\bStar\b appears on Taylor Swift", "Hallucination example (p.17)")
show(r"cuts at least \$800 billion from Medicaid", "Shared bias example (p.18)")
