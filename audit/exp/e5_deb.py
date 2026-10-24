"""Experiment 11: debate with a stronger, different peer and full text.
Agents: Haiku 4.5 (fixed-packet answer, prompt A) and Sonnet 4.5 (prompt A), plus GPT-5 nano's published answer as third voice.
Each Claude agent gets the paper's round-2 prompt with its own answer and the two others. Rows: Aug-Nov 2025 markets
(after Sonnet 4.5's training cutoff), all rows where the three disagree + 50 unanimous controls.
"""
import sys, random
from common import *
R = rows_with_packets(); A, B, D = load(); Au = A.drop_duplicates("key").set_index("key").reindex(R.key)
E1 = pd.read_csv(f"{S}/exp/e1_haiku_nall_rep0_HA-TA-HB-TB.csv"); HA = E1[E1.cond == "HA"].set_index("key").reindex(R.key)
S45 = pd.read_csv(f"{S}/exp/e2_rep1_sonnet45_nall.csv").set_index("key").reindex(R.key)
late = pd.to_datetime([e["resolution_date"] for e in R.packet], utc=True, format="ISO8601").strftime("%Y-%m") >= "2025-08"
ans = {"haiku": (HA.decision.astype(str).str.upper().values, HA.confidence.values, HA.reasoning.astype(str).values),
       "sonnet45": (S45.decision.astype(str).str.upper().values, S45.confidence.values, S45.reasoning.astype(str).values),
       "gpt": (Au.gpt4o_d.values, Au.gpt4o_confidence.values, Au.gpt4o_reasoning.astype(str).values)}
valid = np.isin(ans["haiku"][0], ["YES", "NO"]) & np.isin(ans["sonnet45"][0], ["YES", "NO"])
unan = (ans["haiku"][0] == ans["sonnet45"][0]) & (ans["sonnet45"][0] == ans["gpt"][0])
rng = np.random.default_rng(13); pool = np.where(late & valid & unan)[0]
ctrl = rng.choice(pool, size=min(50, len(pool)), replace=False)
idx = sorted(set(np.where(late & valid & ~unan)[0]) | set(ctrl))
jobs, meta = [], []
for i in idx:
    e = R.packet[i]
    for me in ("haiku", "sonnet45"):
        own = json.dumps({"decision": ans[me][0][i], "confidence": float(ans[me][1][i]), "reasoning": ans[me][2][i]}, indent=2)
        peers = [p for p in ("haiku", "sonnet45", "gpt") if p != me]
        random.Random(f"{e['question_id']}:{me}:deb2").shuffle(peers)
        sec = "\n\n".join("\n".join([f"AGENT {chr(65 + j)}:", f"Answer: {ans[p][0][i]}", f"Confidence: {ans[p][1][i]}", f"Reasoning: {ans[p][2][i]}"]) for j, p in enumerate(peers))
        u = R2_USER.format(question=e["question_text"], criteria=e["resolution_criteria"], exa_evidence=fmt_B(e, "highlights"), your_round_1_response=own, other_agents_section=sec)
        jobs.append(dict(id=(R.key[i], me), model=me, system=R2_SYSTEM, user=u, schema=SCHEMA_R2, rep=0)); meta.append(dict(i=i, key=R.key[i], agent=me))
print(f"rows {len(idx)} (disagree {len(idx) - len(ctrl)}, unanimous controls {len(ctrl)}), calls {len(jobs)}")
res = run_many(jobs, workers=6, label="deb2")
rows = []
for m in meta:
    rec = res[(m["key"], m["agent"])]; dd, c, r = parse(rec); i = m["i"]
    rows.append(dict(key=m["key"], agent=m["agent"], r1=ans[m["agent"]][0][i], r2=dd, r2_conf=c, haiku=ans["haiku"][0][i], sonnet45=ans["sonnet45"][0][i], gpt=ans["gpt"][0][i],
                     unanimous=bool(unan[i]), truth=R.truth[i], question_id=R.question_id[i], cost=rec.get("cost"), error=rec.get("error")))
out = pd.DataFrame(rows); out.to_csv(f"{S}/exp/e5_deb2.csv", index=False)
print("deb2 rows", len(out), "errors", int(out.error.notna().sum()), "cost", round(out.cost.fillna(0).sum(), 2))
