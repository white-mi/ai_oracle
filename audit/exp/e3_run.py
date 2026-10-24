"""Experiments 5-7 on the 430 rows with cached evidence.
usage: python e3_run.py <exp>
  deb    : second round with FULL source text. Haiku (round 1 = content + prompt B, already run) sees the published Architecture-A answers
           of GPT-5 nano and DeepSeek V3 as peers, with the paper's round-2 prompt. Rows: all non-unanimous trios + 100 unanimous controls.
  split  : three Haiku agents, each sees a disjoint third of the sources (index mod 3), prompt A.
  annot  : Sonnet labels what the packet contains with respect to the TRUE outcome, for all errors of the paper's A majority and of Haiku + 60 controls.
"""
import sys, random
from common import *
exp = sys.argv[1]
R = rows_with_packets()
E1 = pd.read_csv(f"{S}/exp/e1_haiku_nall_rep0_HA-TA-HB-TB.csv")
HB = E1[E1.cond == "HB"].set_index("key").reindex(R.key); HA = E1[E1.cond == "HA"].set_index("key").reindex(R.key)
A, B, D = load(); Au = A.drop_duplicates("key").set_index("key").reindex(R.key)

if exp == "deb":
    hd = HB.decision.astype(str).str.upper().values; g = Au.gpt4o_d.values; d = Au.deepseek_d.values
    unan = (hd == g) & (g == d); valid = np.isin(hd, ["YES", "NO"])
    rng = np.random.default_rng(5); ctrl = rng.choice(np.where(unan & valid)[0], size=100, replace=False)
    idx = sorted(set(np.where(~unan & valid)[0]) | set(ctrl))
    jobs = []; meta = []
    for i in idx:
        e = R.packet[i]; k = R.key[i]
        own = json.dumps({"decision": hd[i], "confidence": float(HB.confidence.values[i]), "reasoning": str(HB.reasoning.values[i])}, indent=2)
        peers = [("gpt", Au.gpt4o_d.values[i], Au.gpt4o_confidence.values[i], Au.gpt4o_reasoning.values[i]), ("deepseek", Au.deepseek_d.values[i], Au.deepseek_confidence.values[i], Au.deepseek_reasoning.values[i])]
        random.Random(f"{e['question_id']}:haiku").shuffle(peers)
        sec = "\n\n".join("\n".join([f"AGENT {chr(ord('A') + j)}:", f"Answer: {p[1]}", f"Confidence: {p[2]}", f"Reasoning: {p[3]}"]) for j, p in enumerate(peers))
        u = R2_USER.format(question=e["question_text"], criteria=e["resolution_criteria"], exa_evidence=fmt_B(e, "highlights"), your_round_1_response=own, other_agents_section=sec)
        jobs.append(dict(id=k, model="haiku", system=R2_SYSTEM, user=u, schema=SCHEMA_R2, rep=0)); meta.append(dict(key=k, unanimous=bool(unan[i]), h1=hd[i], gpt=g[i], deepseek=d[i]))
    res = run_many(jobs, workers=6, label="deb haiku")
    rows = []
    for m in meta:
        rec = res[m["key"]]; dd, c, r = parse(rec); st = rec.get("structured") or {}
        rows.append(dict(**m, h2=dd, h2_conf=c, h2_reasoning=r, revised=st.get("revised"), notes=st.get("convergence_notes"), cost=rec.get("cost"), error=rec.get("error")))
    out = pd.DataFrame(rows).merge(R[["key", "truth", "y", "category", "question_id"]], on="key")
    out.to_csv(f"{S}/exp/e3_deb.csv", index=False); print("deb rows", len(out), "non-unanimous", int((~out.unanimous).sum()), "cost", round(out.cost.sum(), 2), "errors", int(out.error.notna().sum()))

elif exp == "split":
    jobs = []
    for i, e in enumerate(R.packet):
        for a in range(3):
            srcs = [s for j, s in enumerate(e["sources"]) if j % 3 == a]
            s, u = prompt_A(e, "highlights", srcs)
            jobs.append(dict(id=(R.key[i], a), model="haiku", system=s, user=u, schema=SCHEMA, rep=0))
    res = run_many(jobs, workers=6, label="split haiku")
    rows = []
    for (k, a), rec in res.items():
        dd, c, r = parse(rec); rows.append(dict(key=k, agent=a, decision=dd, confidence=c, reasoning=r, cost=rec.get("cost"), error=rec.get("error")))
    out = pd.DataFrame(rows).merge(R[["key", "truth", "y", "category", "question_id"]], on="key"); out["ok"] = (out.decision == out.truth).astype(int)
    out.to_csv(f"{S}/exp/e3_split.csv", index=False); print("split rows", len(out), "acc by agent", out.groupby("agent").ok.mean().round(4).to_dict(), "cost", round(out.cost.sum(), 2), "errors", int(out.error.notna().sum()))

elif exp == "annot":
    a_err = Au.maj_ok.values == 0; h_err = HA.ok.values == 0
    rng = np.random.default_rng(9); good = np.where(~a_err & ~h_err)[0]
    ctrl_yes = rng.choice([i for i in good if R.y[i] == 1], size=30, replace=False); ctrl_no = rng.choice([i for i in good if R.y[i] == 0], size=30, replace=False)
    idx = sorted(set(np.where(a_err | h_err)[0]) | set(ctrl_yes) | set(ctrl_no))
    SYS = ("You audit evidence packets that were used to resolve prediction market questions. You receive the question, its resolution criteria, the market close date, "
           "the TRUE final outcome, and the evidence packet (source titles, publication dates and excerpts). Judge only what is written in the packet. "
           "Classify what the packet contains with respect to the true outcome:\n"
           "STATES: at least one excerpt or title explicitly reports the fact that decides the question in the direction of the true outcome "
           "(for a true YES: the event is reported as having happened or the threshold as met; for a true NO: the packet reports that it did not happen, or reports a mutually exclusive final result).\n"
           "IMPLIES: no explicit report, but the packet contains concrete facts from which the true outcome follows by a short, safe inference.\n"
           "SILENT: the packet does not show how the question ended (background, forecasts, previews, schedules made before the event, unrelated items).\n"
           "CONTRADICTS: the packet explicitly reports an outcome opposite to the true outcome.\n"
           "Also report whether any source was clearly written after the deciding event or deadline, and give one verbatim quote of at most 30 words from the packet that best supports your label (empty string if none).")
    SCH = {"type": "object", "properties": {"label": {"type": "string", "enum": ["STATES", "IMPLIES", "SILENT", "CONTRADICTS"]}, "post_event_source": {"type": "boolean"},
                                            "quote": {"type": "string"}, "note": {"type": "string"}}, "required": ["label", "post_event_source", "quote", "note"], "additionalProperties": False}
    jobs = []
    for i in idx:
        e = R.packet[i]
        u = f"TRUE OUTCOME: {R.truth[i]}\nMarket close date: {e['resolution_date'][:10]}\n\n{fmt_A(e)}\n\nClassify the packet."
        jobs.append(dict(id=R.key[i], model="sonnet", system=SYS, user=u, schema=SCH, rep=0))
    res = run_many(jobs, workers=5, label="annot sonnet")
    rows = []
    for i in idx:
        rec = res[R.key[i]]; st = rec.get("structured") or {}
        rows.append(dict(key=R.key[i], label=st.get("label"), post_event=st.get("post_event_source"), quote=st.get("quote"), note=st.get("note"), a_err=bool(a_err[i]), h_err=bool(h_err[i]),
                         control=bool(i in set(ctrl_yes) | set(ctrl_no)), cost=rec.get("cost"), error=rec.get("error")))
    out = pd.DataFrame(rows).merge(R[["key", "truth", "y", "category", "question_id"]], on="key")
    out.to_csv(f"{S}/exp/e3_annot.csv", index=False); print("annot rows", len(out), out.label.value_counts(dropna=False).to_dict(), "cost", round(out.cost.sum(), 2), "errors", int(out.error.notna().sum()))
