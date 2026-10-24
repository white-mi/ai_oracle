"""Prompt and answer sizes for a faithful replication of arXiv:2605.30802, from the paper's own packets and outputs."""
import sys, json, hashlib, numpy as np, pandas as pd
S = "."
sys.path.insert(0, f"{S}/exp")
from common import *  # prompts verbatim from the repo, fmt_A/fmt_B, rows_with_packets, load

R = rows_with_packets()
A, B, D = load()
Au = A.drop_duplicates("key").set_index("key"); Bu = B.drop_duplicates("key").set_index("key")

# 1. calibrate chars -> tokens on Haiku calls of experiment 1 (exact prompts, logged input tokens)
mem = {}
for line in open(f"{S}/exp/cache_calls.jsonl"):
    r = json.loads(line); mem[r["h"]] = r
xs, ys = [], []
for i, e in enumerate(R.packet):
    for mode, fn in (("highlights", prompt_A), ("titles", prompt_B), ("highlights", prompt_B)):
        s, u = fn(e, mode)
        h = hashlib.sha256(json.dumps(["haiku", s, u, SCHEMA, 0], sort_keys=True).encode()).hexdigest()
        if h in mem and mem[h].get("in_tok"):
            xs.append(len(s) + len(u)); ys.append(mem[h]["in_tok"])
xs, ys = np.array(xs), np.array(ys)
b, a = np.polyfit(xs, ys, 1)
print(f"calibration on {len(xs)} Haiku calls: tokens = {a:.0f} + {b:.4f} * chars  (chars/token = {1/b:.2f}); wrapper overhead ~{a:.0f}")

def tok(chars): return b * np.asarray(chars, float)

# 2. prompt sizes per call type on the 430 rows with packets
rows = []
for i, e in enumerate(R.packet):
    k = R.key[i]
    sA, uA = prompt_A(e, "highlights")
    s1, u1 = prompt_B(e, "highlights")
    s1t, u1t = prompt_B(e, "titles")
    # round 2: own answer + two peers' answers. Use published A reasoning (full-text world) and B reasoning (titles world)
    ra = [str(Au.loc[k, f"{m}_reasoning"]) for m in M]
    rb = [str(Bu.loc[k, f"{m}_round1_reasoning"]) if k in Bu.index else ra[j] for j, m in enumerate(M)]
    def r2(ev, rs):
        own = json.dumps({"decision": "NO", "confidence": 0.8, "reasoning": rs[0]}, indent=2)
        sec = "\n\n".join(f"AGENT {chr(65+j)}:\nAnswer: NO\nConfidence: 0.8\nReasoning: {x}" for j, x in enumerate(rs[1:]))
        return len(R2_SYSTEM) + len(R2_USER.format(question=e["question_text"], criteria=e["resolution_criteria"], exa_evidence=ev, your_round_1_response=own, other_agents_section=sec))
    closed = len(PA_SYSTEM) + len(e["question_text"]) + len(e["resolution_criteria"]) + 400
    rows.append(dict(A=len(sA)+len(uA), B1=len(s1)+len(u1), B1_titles=len(s1t)+len(u1t), R2=r2(fmt_B(e, "highlights"), ra), R2_titles=r2(fmt_B(e, "titles"), rb), closed=closed))
P = pd.DataFrame(rows)
print("\ninput tokens per call (430 rows with packets):")
print(pd.DataFrame({c: [tok(P[c]).mean(), np.median(tok(P[c])), np.percentile(tok(P[c]), 90)] for c in P}, index=["mean", "median", "p90"]).round(0).to_string())

# 3. visible output sizes from the paper's own answers (reasoning chars / chars-per-token + JSON overhead ~40)
print("\nvisible output tokens per call (mean), from published reasoning:")
for m in M:
    a_ = A[f"{m}_reasoning"].astype(str).str.len().mean()
    b1 = B[f"{m}_round1_reasoning"].astype(str).str.len().mean()
    b2 = (B[f"{m}_round2_reasoning"].astype(str).str.len() + B[f"{m}_round2_convergence_notes"].astype(str).str.len()).mean()
    print(f"  {NAMES[m]:14s} A {b*a_+40:6.0f}   B round1 {b*b1+40:6.0f}   B round2 {b*b2+60:6.0f}")
P.to_csv(f"../replication/prompt_chars.csv", index=False)
json.dump(dict(a=a, b=b, mean_tok={c: float(tok(P[c]).mean()) for c in P},
               out={m: dict(A=float(b*A[f"{m}_reasoning"].astype(str).str.len().mean()+40),
                            B1=float(b*B[f"{m}_round1_reasoning"].astype(str).str.len().mean()+40),
                            B2=float(b*(B[f"{m}_round2_reasoning"].astype(str).str.len()+B[f"{m}_round2_convergence_notes"].astype(str).str.len()).mean()+60)) for m in M}),
          open(f"../replication/sizes.json", "w"), indent=1)
