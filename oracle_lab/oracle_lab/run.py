"""Command line entry points.

  python -m oracle_lab.run import-paper --repo PATH --work WORK
      Convert the audited paper's dataset (D1) and its cached evidence packets into our formats
      (WORK/d1.csv, WORK/packets/paper_cache/...). Offline, no cost. Useful for dry runs and drift checks.

  python -m oracle_lab.run packets --data WORK/d1.csv --work WORK --variant baseline --templates paper
      Retrieve evidence with Exa (needs EXA_API_KEY). --templates paper,rules,result,opposite builds the merged packet of H2.

  python -m oracle_lab.run answer --data WORK/d1.csv --work WORK --variant baseline --condition prob \
        --model anthropic:claude-haiku-4-5-20251001 --out WORK/results/haiku_prob.csv [--limit 100] [--reps 1]
      Conditions: prob (P(YES)), abstain (YES/NO/INSUFFICIENT), closed (no evidence), paper_A (paper's prompt A,
      needs --repo). Use --model fake:x for a free dry run of the whole pipeline.

  python -m oracle_lab.run d2 --from 2026-10-12 --to 2026-11-15 --out WORK/d2.csv
      Export settled Kalshi markets closing in the window.
"""
import argparse
import glob
import json
import os

import pandas as pd

from .evidence import Packet, Source, build_packet, check_content, render_packet
from .llm import make_client, run_many
from .prompts import ABSTAIN, CLOSED, CLOSED_SYSTEM, PROB, SYSTEM, load_paper_prompts
from .store import CallCache, PacketStore, question_key


def render_paper_A(p: Packet) -> str:
    """EvidencePacket.format_for_prompt of the audited repo (Architecture A layout, highlights shown)."""
    lines = [f"Question: {p.question}", f"Resolution Criteria: {p.criteria}", f"Resolution Date: {p.close_time}", "",
             f"Retrieved {len(p.sources)} sources:", ""]
    for i, s in enumerate(p.sources, 1):
        lines += [f"--- Source {i} ---", f"Title: {s.title}", f"URL: {s.url}"]
        if s.published:
            lines.append(f"Published: {s.published}")
        c = s.content().strip()
        lines.append(f"\n{c}\n" if c else "\n(No extractable content)\n")
    return "\n".join(lines)


def cmd_import_paper(a):
    os.makedirs(a.work, exist_ok=True)
    d = pd.read_csv(os.path.join(a.repo, "data/kalshibench_v2_evaluation_Filtered.csv"))
    rows = pd.DataFrame(dict(market_id=d["id"], question=d["question"], criteria=d["description"], close_time=d["close_time"],
                             outcome=d["ground_truth"].str.upper(), category=d["category"]))
    rows["key"] = [question_key(m, q, c) for m, q, c in zip(rows.market_id, rows.question, rows.criteria)]
    dup = rows.key.duplicated().sum()
    rows = rows.drop_duplicates("key")
    rows.to_csv(os.path.join(a.work, "d1.csv"), index=False)
    store = PacketStore(os.path.join(a.work, "packets"))
    n = 0
    for f in sorted(glob.glob(os.path.join(a.repo, "cache/evidence/*.json"))):
        e = json.load(open(f))
        src = [Source(url=s["url"], title=s.get("title") or "", published=s.get("published_date"),
                      highlights=list(s.get("highlights") or []), text=s.get("text") or "", query=e.get("query_used", ""))
               for s in e["sources"]]
        p = Packet(e["question_id"], e["question_text"], e["resolution_criteria"], e["resolution_date"], "paper_cache",
                   [e.get("query_used", "")], src, e.get("retrieval_timestamp", ""))
        if not store.has(p.key, p.variant):
            store.save(p)
            n += 1
    print(f"d1.csv: {len(rows)} distinct questions ({dup} duplicate rows dropped); packets imported: {n}")


def cmd_packets(a):
    rows = pd.read_csv(a.data)
    store = PacketStore(os.path.join(a.work, "packets"))
    templates = tuple(a.templates.split(","))
    todo = [r for r in rows.itertuples() if not store.has(question_key(r.market_id, r.question, r.criteria), a.variant)]
    print(f"{len(todo)} packets to retrieve ({len(rows) - len(todo)} cached), {len(todo) * len(templates)} Exa searches")
    if a.limit:
        todo = todo[:a.limit]
    for i, r in enumerate(todo, 1):
        p = build_packet(r.market_id, r.question, r.criteria, r.close_time, a.variant, templates,
                         retrieved_at=pd.Timestamp.utcnow().isoformat())
        store.save(p)
        if i % 50 == 0:
            print(f"  {i}/{len(todo)}", flush=True)


def cmd_answer(a):
    rows = pd.read_csv(a.data)
    if a.limit:
        rows = rows.head(a.limit)
    store = PacketStore(os.path.join(a.work, "packets"))
    cache = CallCache(os.path.join(a.work, "calls.jsonl"))
    client = make_client(a.model, max_tokens=a.max_tokens)
    paper = load_paper_prompts(a.repo) if a.condition == "paper_A" else None
    jobs, meta, skipped = [], [], 0
    for r in rows.itertuples():
        k = question_key(r.market_id, r.question, r.criteria)
        if a.condition == "closed":
            system, user = CLOSED_SYSTEM, CLOSED.format(question=r.question, criteria=r.criteria, close_time=r.close_time)
        else:
            if not store.has(k, a.variant):
                skipped += 1
                continue
            p = store.load(k, a.variant)
            if a.condition == "paper_A":
                system, user = paper["A_SYSTEM"], paper["A_USER"].format(evidence_text=render_paper_A(p))
            else:
                tpl = {"prob": PROB, "abstain": ABSTAIN}[a.condition]
                system, user = SYSTEM, tpl.format(question=r.question, criteria=r.criteria, close_time=r.close_time,
                                                  evidence=render_packet(p))
            check_content(user, p)  # rule 1: refuse to send a prompt without source text
        for rep in range(a.reps):
            jobs.append(dict(id=(k, rep), system=system, user=user, rep=rep))
            meta.append(dict(key=k, rep=rep, market_id=r.market_id, outcome=getattr(r, "outcome", None),
                             category=getattr(r, "category", None)))
    print(f"{len(jobs)} calls ({skipped} rows without a '{a.variant}' packet skipped); model {client.name}")
    if a.dry_count:
        return
    res = run_many(client, cache, jobs, workers=a.workers, label=a.condition)
    out = []
    for m in meta:
        rec = res[(m["key"], m["rep"])]
        js = rec.get("parsed") or {}
        if a.condition == "paper_A":
            dec = str(js.get("decision", "")).upper() or None
            p_yes = None
            conf = js.get("confidence")
        else:
            p_yes = js.get("p_yes")
            dec = str(js.get("decision", "")).upper() or None
            if dec is None and p_yes is not None:
                dec = "YES" if float(p_yes) > 0.5 else "NO"
            conf = None
        out.append(dict(**m, decision=dec, p_yes=p_yes, confidence=conf, reasoning=js.get("reasoning"),
                        model_version=rec.get("model"), request_id=rec.get("request_id"), in_tok=rec.get("in_tok"),
                        out_tok=rec.get("out_tok"), error=rec.get("error")))
    df = pd.DataFrame(out)
    df["correct"] = (df.decision == df.outcome).astype(int)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    ok = df[df.error.isna()]
    print(f"wrote {a.out}: {len(df)} rows, errors {df.error.notna().sum()}, accuracy {ok.correct.mean():.3f}")


def cmd_d2(a):
    from .kalshi import settled_markets, to_rows
    rows = pd.DataFrame(to_rows(settled_markets(a.date_from, a.date_to)))
    rows["key"] = [question_key(m, q, c) for m, q, c in zip(rows.market_id, rows.question, rows.criteria)]
    rows.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(rows)} settled markets, YES share {(rows.outcome == 'YES').mean():.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oracle_lab")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("import-paper"); p.add_argument("--repo", required=True); p.add_argument("--work", required=True)
    p = sp.add_parser("packets"); p.add_argument("--data", required=True); p.add_argument("--work", required=True)
    p.add_argument("--variant", default="baseline"); p.add_argument("--templates", default="paper"); p.add_argument("--limit", type=int)
    p = sp.add_parser("answer"); p.add_argument("--data", required=True); p.add_argument("--work", required=True)
    p.add_argument("--variant", default="baseline"); p.add_argument("--condition", choices=["prob", "abstain", "closed", "paper_A"], required=True)
    p.add_argument("--model", required=True); p.add_argument("--out", required=True); p.add_argument("--repo")
    p.add_argument("--limit", type=int); p.add_argument("--reps", type=int, default=1); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=1500)
    p.add_argument("--dry-count", dest="dry_count", action="store_true", help="only count calls, send nothing")
    p = sp.add_parser("d2"); p.add_argument("--from", dest="date_from", required=True); p.add_argument("--to", dest="date_to", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"import-paper": cmd_import_paper, "packets": cmd_packets, "answer": cmd_answer, "d2": cmd_d2}[a.cmd](a)


if __name__ == "__main__":
    main()
