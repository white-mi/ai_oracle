import sys, json
import pandas as pd
R=sys.argv[1]
f=json.load(open(sys.argv[2]))
A=pd.read_csv(f"{R}/results/Final_Architecture_A_results.csv")
up=lambda s: str(s).strip().upper()
for r in f["F8"]["final_outcome_packets"]:
    e=json.load(open(f"{R}/cache/evidence/highlights__{r['row_uid']}.json"))
    a=A[A.row_uid==r["row_uid"]]
    dec=[up(a.iloc[0][m+"_decision"]) for m in ("gpt4o","deepseek","llama")] if len(a) else None
    print(r["row_uid"], "| res", e["resolution_date"][:10], "| truth", r["truth"], "Amaj", r["a_majority"], dec, "|", (r["question"] or "")[:70])
    for u,o in r["pages"]: print("     ", o, u[:120])
