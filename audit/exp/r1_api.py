"""Confirmation run through the Anthropic API (Message Batches), with leakage controls.

Nothing is sent unless --send is given together with --cap-usd. Without --send the script only builds the
requests and prints the number of calls and an upper bound on the cost.

Conditions, all on the same committed evidence packet per question (rows_with_packets in common.py):
  HA        prompt A, source highlights shown              (Architecture A as published)
  TB        prompt B round 1, titles, URLs and dates only  (Architecture B as published)
  HB        prompt B round 1, source highlights shown      (Architecture B with the formatter fixed)
  closed    prompt A with the source list removed           (what the model answers from memory)
  HA_nomkt  prompt A without prediction-market and betting pages

Leakage controls in the analysis step:
  1. closed-book accuracy and the subset of questions the model does not answer correctly from memory;
  2. split by resolution date at the model's training-data cutoff;
  3. HA against HA_nomkt, which removes settled market pages ("Final outcome: ...") from the evidence;
  4. the served model is read from each response and must equal the requested model.

Usage (from the audit/ folder, with ./repo a clone of the paper's repository at f76754b):
  python3 exp/r1_api.py build                       # writes exp/r1_requests.jsonl, prints calls and cost bound
  ANTHROPIC_API_KEY=... python3 exp/r1_api.py send --cap-usd 15
  ANTHROPIC_API_KEY=... python3 exp/r1_api.py collect
  python3 exp/r1_api.py analyze
"""
import argparse, hashlib, json, os, re, sys, time
import numpy as np, pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from common import prompt_A, prompt_B, rows_with_packets, SCHEMA, mcnemar, wilson  # noqa: E402

MODEL = "claude-haiku-4-5-20251001"
TRAINING_CUTOFF = "2025-07-31"     # Haiku 4.5 training-data cutoff (July 2025); check against the model card before use
PRICE_IN, PRICE_OUT = 1.00, 5.00   # USD per 1M tokens, list price; check before use
BATCH_DISCOUNT = 0.5
MAX_TOKENS = 800                   # replies in the pilot runs were 150-450 tokens
CHARS_PER_TOKEN = 3.0              # conservative, so the bound is high
API = "https://api.anthropic.com/v1/messages/batches"
HERE = os.path.dirname(os.path.abspath(__file__))
REQ, STATE, RES = (os.path.join(HERE, n) for n in ("r1_requests.jsonl", "r1_state.json", "r1_results.jsonl"))
MARKET = re.compile(r"kalshi\.com|polymarket\.com|manifold\.markets|predictit\.org|metaculus\.com|fanduel|draftkings|"
                    r"sportsbook|covers\.com|oddsshark|oddschecker|betfair", re.I)
TOOL = {"name": "submit_resolution", "description": "Submit the resolution decision.", "input_schema": SCHEMA}
CONDITIONS = {"HA": 2, "TB": 1, "HB": 2, "closed": 1, "HA_nomkt": 1}   # condition -> repetitions


def prompts(e, cond):
    if cond == "HA":
        return prompt_A(e, "highlights")
    if cond == "TB":
        return prompt_B(e, "titles")
    if cond == "HB":
        return prompt_B(e, "highlights")
    if cond == "closed":
        return prompt_A(e, "highlights", sources=[])
    if cond == "HA_nomkt":
        return prompt_A(e, "highlights", sources=[s for s in e["sources"] if not MARKET.search(s.get("url") or "")])
    raise ValueError(cond)


def cid(row_key, cond, rep):
    return f"{cond}-r{rep}-" + hashlib.sha1(row_key.encode()).hexdigest()[:16]


def build(a):
    R = rows_with_packets()
    n_in = 0
    with open(REQ, "w") as f:
        for _, r in R.iterrows():
            for cond, reps in CONDITIONS.items():
                system, user = prompts(r.packet, cond)
                for rep in range(reps):
                    body = {"custom_id": cid(r.key, cond, rep),
                            "params": {"model": MODEL, "max_tokens": MAX_TOKENS, "temperature": 0.0, "system": system,
                                       "messages": [{"role": "user", "content": user}],
                                       "tools": [TOOL], "tool_choice": {"type": "tool", "name": TOOL["name"]}}}
                    n_in += (len(system) + len(user) + len(json.dumps(TOOL))) / CHARS_PER_TOKEN
                    f.write(json.dumps(body, ensure_ascii=False) + "\n")
    calls = sum(1 for _ in open(REQ))
    bound = BATCH_DISCOUNT * (n_in * PRICE_IN + calls * MAX_TOKENS * PRICE_OUT) / 1e6
    expected = BATCH_DISCOUNT * (n_in * PRICE_IN + calls * 400 * PRICE_OUT) / 1e6
    print(f"{len(R)} questions, {calls} calls, about {n_in / 1e6:.2f}M input tokens. "
          f"Cost bound ${bound:.2f} (every reply at max_tokens), expected about ${expected:.2f}.")
    return bound


def headers():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("ANTHROPIC_API_KEY is not set")
    return {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}


def send(a):
    bound = build(a)
    if a.cap_usd is None or bound > a.cap_usd:
        sys.exit(f"Not sent: the cost bound ${bound:.2f} exceeds --cap-usd {a.cap_usd}.")
    reqs = [json.loads(x) for x in open(REQ)]
    ids = []
    for i in range(0, len(reqs), 5000):
        r = requests.post(API, headers=headers(), json={"requests": reqs[i:i + 5000]}, timeout=120)
        r.raise_for_status()
        ids.append(r.json()["id"])
    json.dump({"batches": ids, "sent": time.time(), "model": MODEL}, open(STATE, "w"))
    print("submitted batches:", ids)


def collect(a):
    st = json.load(open(STATE))
    with open(RES, "w") as out:
        for bid in st["batches"]:
            while True:
                b = requests.get(f"{API}/{bid}", headers=headers(), timeout=60).json()
                if b.get("processing_status") == "ended":
                    break
                print(bid, b.get("processing_status"), b.get("request_counts")); time.sleep(60)
            res = requests.get(b["results_url"], headers=headers(), timeout=300)
            res.raise_for_status()
            for line in res.text.splitlines():
                out.write(line + "\n")
    print("results written to", RES)


def parse_line(d):
    r = d.get("result", {})
    if r.get("type") != "succeeded":
        return dict(error=r.get("type"))
    m = r["message"]
    tool = next((c for c in m.get("content", []) if c.get("type") == "tool_use"), None)
    inp = tool.get("input", {}) if tool else {}
    return dict(served=m.get("model"), decision=str(inp.get("decision", "")).upper() or None,
                confidence=inp.get("confidence"), in_tok=m["usage"]["input_tokens"], out_tok=m["usage"]["output_tokens"])


def analyze(a):
    R = rows_with_packets()
    got = {}
    for line in open(RES):
        d = json.loads(line)
        got[d["custom_id"]] = parse_line(d)
    served = {v.get("served") for v in got.values() if v.get("served")}
    print("served models:", served, "(must equal", MODEL + ")")
    assert served <= {MODEL}, "served model differs from the requested one"
    cost = sum(BATCH_DISCOUNT * (v.get("in_tok", 0) * PRICE_IN + v.get("out_tok", 0) * PRICE_OUT) / 1e6 for v in got.values())
    print(f"errors: {sum(1 for v in got.values() if v.get('error'))}, cost ${cost:.2f}")
    for cond, reps in CONDITIONS.items():
        for rep in range(reps):
            R[f"{cond}{rep}"] = [got.get(cid(k, cond, rep), {}).get("decision") for k in R.key]
            R[f"{cond}{rep}_ok"] = (R[f"{cond}{rep}"] == R.truth).astype(int)
    R["res_date"] = [str(e.get("resolution_date"))[:10] for e in R.packet]
    R["has_mkt"] = [any(MARKET.search(s.get("url") or "") for s in e["sources"]) for e in R.packet]
    subsets = {"all": R.index == R.index,
               "closed-book wrong": R.closed0_ok == 0,
               f"resolved after {TRAINING_CUTOFF}": R.res_date > TRAINING_CUTOFF,
               "no market page in packet": ~R.has_mkt}
    pairs = [("HB0", "TB0", "effect of the formatter fix (defect 1)"), ("HA0", "HB0", "prompt A against prompt B, both with text"),
             ("HA0", "TB0", "A as published against B round 1 as published"), ("HA0", "HA1", "repeat of the same run (noise)"),
             ("HB0", "HB1", "repeat of the same run (noise)"), ("HA0", "HA_nomkt0", "effect of removing market pages")]
    for name, sel in subsets.items():
        S = R[sel]
        print(f"\n== {name}: {len(S)} questions")
        for c in ("HA0", "TB0", "HB0", "closed0", "HA_nomkt0"):
            k = int(S[f"{c}_ok"].sum()); lo, hi = wilson(k, len(S))
            print(f"  {c:10s} accuracy {100 * k / len(S):5.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]")
        for x, y, lab in pairs:
            n10, n01, p = mcnemar(S[f"{x}_ok"], S[f"{y}_ok"])
            print(f"  {x} vs {y}: {n10} / {n01}, exact McNemar p = {p:.2g}   {lab}")
    R.drop(columns=["packet"]).to_csv(os.path.join(HERE, "r1_decisions.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "send", "collect", "analyze"])
    ap.add_argument("--cap-usd", dest="cap_usd", type=float)
    a = ap.parse_args()
    {"build": build, "send": send, "collect": collect, "analyze": analyze}[a.step](a)
