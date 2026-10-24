"""Experiments 2-4 on the 430 rows with cached evidence.
usage: python e2_run.py <exp> <model> [n]
  exp = rep1 | rep2      : repeat of highlights + prompt A (noise, and extra samples for the same-evidence trio)
        closed           : no evidence, probability of YES from the model's own knowledge (leak / prior control)
        abstain          : highlights + prompt A with a third answer INSUFFICIENT
        noul             : highlights, bare probability of YES, no reasoning (Jev-style interface emulated with an LLM)
        noulb            : highlights, bare probability that the evidence states the rule's condition has occurred
        date             : highlights + prompt A + explicit 'current date' line (harness-date ablation)
"""
import sys, datetime
from common import *
exp, model = sys.argv[1], sys.argv[2]; n = sys.argv[3] if len(sys.argv) > 3 else "all"
R = rows_with_packets()
if n != "all": R = R.sample(int(n), random_state=11).reset_index(drop=True)
P_SCHEMA = {"type": "object", "properties": {"p_yes": {"type": "number", "minimum": 0.0, "maximum": 1.0}}, "required": ["p_yes"], "additionalProperties": False}
ABST_SCHEMA = {"type": "object", "properties": {"decision": {"type": "string", "enum": ["YES", "NO", "INSUFFICIENT"]}, "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}, "reasoning": {"type": "string"}},
               "required": ["decision", "confidence", "reasoning"], "additionalProperties": False}
ABST_SYSTEM = PA_SYSTEM.replace("should resolve to YES or NO based on the provided evidence.", "should resolve to YES or NO based on the provided evidence, or whether the evidence is INSUFFICIENT to tell.") \
    .replace("4. If evidence is ambiguous, use your best judgment", "4. If the evidence does not establish whether the outcome occurred, answer INSUFFICIENT instead of guessing") \
    .replace("- decision: YES or NO", "- decision: YES, NO or INSUFFICIENT")
assert ABST_SYSTEM != PA_SYSTEM and "INSUFFICIENT instead of guessing" in ABST_SYSTEM
NOUL_SYSTEM = "You are an expert prediction market resolution agent. You will receive a question, its resolution criteria and retrieved evidence. Output only the probability that the market resolves to YES. Do not explain."
NOULB_SYSTEM = "You are an expert prediction market resolution agent. You will receive a question, its resolution criteria and retrieved evidence. Output only the probability that the evidence states that the condition in the resolution criteria has already occurred. Do not explain."
CLOSED_SYSTEM = "You are an expert on world events. You will receive a prediction market question that has already been resolved, with its resolution criteria and close date. No documents are provided. Using only your own knowledge, output the probability that it resolved to YES. Do not explain."
jobs = []
for i, e in enumerate(R.packet):
    if exp in ("rep1", "rep2"):
        s, u = prompt_A(e); sch = SCHEMA; rep = int(exp[-1])
    elif exp == "abstain":
        s, u = ABST_SYSTEM, PA_USER.format(evidence_text=fmt_A(e)).replace("should this question resolve to YES or NO?", "should this question resolve to YES or NO, or is the evidence INSUFFICIENT?"); sch = ABST_SCHEMA; rep = 0
    elif exp == "noul":
        s, u = NOUL_SYSTEM, PA_USER.format(evidence_text=fmt_A(e)).replace("then provide your resolution decision.", "then give the probability.").replace("should this question resolve to YES or NO?", "what is the probability that this question resolves to YES?"); sch = P_SCHEMA; rep = 0
    elif exp == "noulb":
        s, u = NOULB_SYSTEM, PA_USER.format(evidence_text=fmt_A(e)).replace("then provide your resolution decision.", "then give the probability.").replace("should this question resolve to YES or NO?", "what is the probability that the evidence states that the condition in the resolution criteria has already occurred?"); sch = P_SCHEMA; rep = 0
    elif exp == "closed":
        s, u = CLOSED_SYSTEM, f"Question: {e['question_text']}\nResolution Criteria: {e['resolution_criteria']}\nMarket close date: {e['resolution_date'][:10]}\n\nWhat is the probability that this question resolved to YES?"; sch = P_SCHEMA; rep = 0
    elif exp == "date":
        d = (datetime.datetime.fromisoformat(e["resolution_date"][:19]) + datetime.timedelta(days=1)).date().isoformat()
        s, u = PA_SYSTEM + f"\n\nThe current date is {d}. Disregard any other statement about today's date.", PA_USER.format(evidence_text=fmt_A(e)); sch = SCHEMA; rep = 0
    else: raise SystemExit("unknown exp")
    jobs.append(dict(id=R.key[i], model=model, system=s, user=u, schema=sch, rep=rep))
res = run_many(jobs, workers=6, label=f"{exp} {model}")
rows = []
for k, rec in res.items():
    st = rec.get("structured") or {}
    if sch is P_SCHEMA:
        p = st.get("p_yes"); rows.append(dict(key=k, p_yes=p, decision=None if p is None else ("YES" if p > 0.5 else "NO"), confidence=None if p is None else max(p, 1 - p), reasoning="", cost=rec.get("cost"), error=rec.get("error")))
    else:
        d, c, r = parse(rec); rows.append(dict(key=k, p_yes=None, decision=d, confidence=c, reasoning=r, cost=rec.get("cost"), error=rec.get("error")))
out = pd.DataFrame(rows).merge(R[["key", "truth", "y", "category", "question_id"]], on="key"); out["exp"] = exp; out["model"] = model
out["ok"] = (out.decision == out.truth).astype(int)
out.to_csv(f"{S}/exp/e2_{exp}_{model}_n{n}.csv", index=False)
print(exp, model, "n", len(out), "acc", round(out.ok.mean(), 4), "answers", out.decision.value_counts(dropna=False).to_dict(), "cost", round(out.cost.sum(), 2), "errors", int(out.error.notna().sum()))
