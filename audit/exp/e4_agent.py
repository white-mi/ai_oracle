"""Experiment 10: agentic oracle with live web search instead of a fixed evidence packet.
Haiku 4.5 (knowledge cutoff Feb 2025) gets the question, rules and close date, may run up to ~4 web searches,
must not use prediction-market sites, may answer INSUFFICIENT. Rows: the 150 rows of the annotation sample
(all 90 errors of the paper's majority or of Haiku on fixed packets + 60 controls).
usage: python e4_agent.py [model] [limit]
"""
import sys, re, json, hashlib, os, subprocess, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from common import *
model = sys.argv[1] if len(sys.argv) > 1 else "haiku"; lim = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "all" else None
variant = sys.argv[3] if len(sys.argv) > 3 else "default"   # "clean": web-search helper pinned to Haiku 4.5 (no model with a 2026 cutoff in the loop)
tag = model if variant == "default" else f"{model}_{variant}"
OUT = f"{S}/agent/{tag}"; os.makedirs(OUT, exist_ok=True); os.makedirs(f"{OUT}/raw", exist_ok=True)
MARKET = ("kalshi.com", "polymarket.com", "manifold.markets", "predictit.org", "metaculus.com", "polymarketanalytics", "oddschecker", "betfair", "covers.com", "sportsbook", "draftkings", "fanduel", "oddsshark", "actionnetwork")
SYS = ("You resolve prediction market questions after the market has closed. Use the WebSearch tool to find news reports of how the question ended "
       "(at most 4 searches). Do not use prediction market or betting sites (kalshi.com, polymarket.com, manifold.markets, predictit.org, metaculus.com, "
       "odds aggregators) as evidence and do not search for the market itself. Follow the resolution criteria literally. Decide only from what you find. "
       "If you cannot find how the question ended, answer INSUFFICIENT.")
SCH = {"type": "object", "properties": {"decision": {"type": "string", "enum": ["YES", "NO", "INSUFFICIENT"]}, "confidence": {"type": "number", "minimum": 0, "maximum": 1},
       "reasoning": {"type": "string"}, "sources": {"type": "array", "items": {"type": "string"}}}, "required": ["decision", "confidence", "reasoning", "sources"], "additionalProperties": False}
R = rows_with_packets(); A, B, D = load(); Au = A.drop_duplicates("key").set_index("key").reindex(R.key)
E1 = pd.read_csv(f"{S}/exp/e1_haiku_nall_rep0_HA-TA-HB-TB.csv"); HA = E1[E1.cond == "HA"].set_index("key").reindex(R.key)
a_err = Au.maj_ok.values == 0; h_err = HA.ok.values == 0
rng = np.random.default_rng(9); good = np.where(~a_err & ~h_err)[0]
ctrl_yes = rng.choice([i for i in good if R.y[i] == 1], size=30, replace=False); ctrl_no = rng.choice([i for i in good if R.y[i] == 0], size=30, replace=False)
idx = sorted(set(np.where(a_err | h_err)[0]) | set(ctrl_yes) | set(ctrl_no))
if lim: idx = idx[:lim]
STOPF = threading.Event()

def run_one(i):
    e = R.packet[i]; k = R.key[i]
    user = f"Question: {e['question_text']}\nResolution criteria: {e['resolution_criteria']}\nMarket close date: {e['resolution_date'][:10]}\nResolve the market."
    h = hashlib.sha256(json.dumps([model, SYS, user], sort_keys=True).encode()).hexdigest()[:20]
    fn = f"{OUT}/{h}.json"
    if os.path.exists(fn): return json.load(open(fn))
    if STOPF.is_set() or STOP.is_set(): return dict(key=k, error="skipped: limit")
    env = {kk: v for kk, v in os.environ.items() if kk not in DROP_ENV}
    if variant == "clean":
        env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = MODELS["haiku"]; env["ANTHROPIC_SMALL_FAST_MODEL"] = MODELS["haiku"]
    cmd = ["claude", "-p", user, "--model", MODELS.get(model, model), "--system-prompt", SYS, "--tools", "WebSearch", "--allowedTools", "WebSearch",
           "--max-turns", "10", "--output-format", "stream-json", "--verbose", "--no-session-persistence", "--strict-mcp-config", "--setting-sources", "",
           "--json-schema", json.dumps(SCH)]
    try:
        p = subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=420, cwd=f"{S}/runs")
    except subprocess.TimeoutExpired:
        return dict(key=k, error="timeout")
    open(f"{OUT}/raw/{h}.jsonl", "w").write(p.stdout)
    L = []
    for line in p.stdout.splitlines():
        if line.startswith("{"):
            try: L.append(json.loads(line))
            except ValueError: pass
    fin = [l for l in L if l.get("type") == "result"]
    if not fin: return dict(key=k, error=("no result: " + (p.stderr or p.stdout)[-200:]))
    fin = fin[-1]
    if fin.get("is_error"):
        msg = str(fin.get("result"))
        if is_limit(msg): STOPF.set(); STOP.set()
        return dict(key=k, error=msg[:200])
    queries, links = [], []
    for l in L:
        if l.get("type") == "assistant":
            for c in l["message"].get("content", []):
                if c.get("type") == "tool_use" and c.get("name") == "WebSearch": queries.append(c["input"].get("query"))
        if l.get("type") == "user" and isinstance(l["message"].get("content"), list):
            for c in l["message"]["content"]:
                if c.get("type") == "tool_result": links += re.findall(r'"url":"([^"]+)"', str(c.get("content")))
    st = fin.get("structured_output") or {}
    rec = dict(key=k, decision=st.get("decision"), confidence=st.get("confidence"), reasoning=st.get("reasoning"), sources=st.get("sources") or [],
               queries=queries, links=links, n_search=len(queries), turns=fin.get("num_turns"), cost=fin.get("total_cost_usd"),
               models=sorted((fin.get("modelUsage") or {}).keys()), ms=fin.get("duration_ms"))
    json.dump(rec, open(fn, "w"))
    return rec

from common import _is_limit as is_limit
res = []; t0 = time.time()
with ThreadPoolExecutor(max_workers=4) as ex:
    fut = {ex.submit(run_one, i): i for i in idx}
    for n, f in enumerate(as_completed(fut), 1):
        res.append((fut[f], f.result()))
        if n % 25 == 0 or n == len(idx):
            err = sum(1 for _, r in res if r.get("error")); cost = sum((r.get("cost") or 0) for _, r in res)
            print(f"[agent {tag}] {n}/{len(idx)} {time.time()-t0:.0f}s cost ${cost:.2f} errors {err}" + ("  STOPPED: subscription limit" if STOP.is_set() else ""), flush=True)
rows = []
for i, r in res:
    mk = any(any(m in u for m in MARKET) for u in (r.get("links") or []) + (r.get("sources") or []))
    rows.append(dict(key=R.key[i], truth=R.truth[i], y=R.y[i], category=R.category[i], question_id=R.question_id[i], a_err=bool(a_err[i]), h_err=bool(h_err[i]),
                     control=bool(i in set(ctrl_yes) | set(ctrl_no)), decision=r.get("decision"), confidence=r.get("confidence"), n_search=r.get("n_search"),
                     market_site=mk, cost=r.get("cost"), error=r.get("error"), models=",".join(r.get("models") or []), reasoning=r.get("reasoning")))
out = pd.DataFrame(rows); out.to_csv(f"{S}/exp/e4_agent_{tag}.csv", index=False)
ok = out[out.error.isna()]
print("rows", len(out), "done", len(ok), "decisions", ok.decision.value_counts().to_dict(), "market-site touched", int(ok.market_site.sum()), "cost", round(out.cost.fillna(0).sum(), 2))
