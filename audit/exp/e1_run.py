"""Experiment 1: 2x2 evidence content (highlights vs titles only) x prompt layout (A vs B round 1).
usage: python e1_run.py <model> <n_rows|all> [conditions comma list] [rep]"""
import sys
from common import *
model = sys.argv[1]; n = sys.argv[2]; conds = (sys.argv[3] if len(sys.argv) > 3 else "HA,TA,HB,TB").split(","); rep = int(sys.argv[4]) if len(sys.argv) > 4 else 0
R = rows_with_packets()
if n != "all": R = R.sample(int(n), random_state=7).reset_index(drop=True)
jobs = []
for i, e in enumerate(R.packet):
    for c in conds:
        mode = "highlights" if c[0] == "H" else "titles"
        s, u = prompt_A(e, mode) if c[1] == "A" else prompt_B(e, mode)
        jobs.append(dict(id=(R.key[i], c), model=model, system=s, user=u, schema=SCHEMA, rep=rep))
res = run_many(jobs, workers=6, label=f"E1 {model} rep{rep}")
rows = []
for (k, c), rec in res.items():
    d, conf, reason = parse(rec)
    rows.append(dict(key=k, cond=c, model=model, rep=rep, decision=d, confidence=conf, reasoning=reason, cost=rec.get("cost"), in_tok=rec.get("in_tok"), out_tok=rec.get("out_tok"), error=rec.get("error")))
out = pd.DataFrame(rows).merge(R[["key", "truth", "y", "category", "question_id"]], on="key")
out["ok"] = (out.decision == out.truth).astype(int)
out.to_csv(f"{S}/exp/e1_{model}_n{n}_rep{rep}_{'-'.join(conds)}.csv", index=False)
print(out.groupby("cond").agg(n=("ok", "size"), acc=("ok", "mean"), yes_rate=("decision", lambda v: (v == "YES").mean()), conf=("confidence", "mean"), missing=("decision", lambda v: v.isna().sum()), in_tok=("in_tok", "median"), cost=("cost", "sum")).round(3))
