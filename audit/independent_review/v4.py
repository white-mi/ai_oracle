import sys, re
import pandas as pd
R=sys.argv[1]
A=pd.read_csv(f"{R}/results/Final_Architecture_A_results.csv"); B=pd.read_csv(f"{R}/results/Final_Architecture_B_results.csv")
up=lambda s: str(s).strip().upper()
M=["gpt4o","deepseek","llama"]
ex=[("4.4.1","Glen Powell hosts"),("4.4.2","S&P 500 index value starting Mar 19, 2025 and ending on Jan 1, 2026 is above 6699.99"),("4.4.3","If Star appears on Taylor Swift"),("4.4.4","cuts at least $800 billion from")]
phr={"4.4.2":["forecasts do not count as evidence of an actual crossing"],"4.4.3":["person/entity named","named 'Star'","named \"Star\""],"4.4.1":["Glen Powell"],"4.4.4":["become law"]}
for sec,nd in ex:
    a=A[A.resolution_criteria.str.contains(nd,regex=False)]; b=B[B.resolution_criteria.str.contains(nd,regex=False)]
    print(f"== {sec} nA={len(a)} nB={len(b)} A uid={a.row_uid.tolist()} B uid={b.row_uid.tolist()}")
    ra=a.iloc[0]; rb=b.iloc[0]
    print("  truth", up(ra.ground_truth), "A", [up(ra[m+'_decision']) for m in M], "conf", [ra[m+'_confidence'] for m in M])
    print("  B r1", [up(rb[m+'_round1_decision']) for m in M], [rb[m+'_round1_confidence'] for m in M])
    print("  B r2", [up(rb[m+'_round2_decision']) for m in M], [rb[m+'_round2_confidence'] for m in M], "final", rb.final_decision)
    for p in phr[sec]:
        hitsA=[m for m in M if p.lower() in str(ra[m+'_reasoning']).lower()]
        hitsB=[(m,r) for m in M for r in (1,2) if p.lower() in str(rb[f'{m}_round{r}_reasoning']).lower()]
        print(f"  phrase {p!r}: A {hitsA}  B {hitsB}")
