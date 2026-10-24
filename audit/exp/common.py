"""Harness for pilot experiments on the cached evidence packets of arXiv:2605.30802.

Model calls go through the `claude -p` command-line client (single turn, no tools, custom system prompt).
Every response is cached in a JSONL file keyed by (model, system, user, schema, rep), so runs are resumable.
"""
import glob, hashlib, json, os, re, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd

S = "."
REPO = f"{S}/repo"
sys.path.insert(0, f"{S}/analysis")
from kota import load, norm, M, NAMES, mcnemar, wilson, cluster_boot, _ws

MODELS = {"haiku": "claude-haiku-4-5-20251001", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5",
          "sonnet45": "claude-sonnet-4-5-20250929", "haiku55": "claude-haiku-5-5"}
STOP = threading.Event()          # set on the first subscription-limit message: nothing else is sent
LIMIT_MARKERS = ("hit your", "spend limit", "usage limit", "session limit", "weekly limit", "limit reached", "out of extra usage")
def _is_limit(msg):
    m = str(msg or "").lower(); return any(k in m for k in LIMIT_MARKERS)
DROP_ENV = ["SYSTEM_REMINDER_MEMORY_CONTEXT", "CLAUDE_PROJECT_UUID", "CLAUDE_PROJECT_TOOL", "CLAUDE_CODE_USER_EMAIL",
            "CLAUDE_ADDITIONAL_DIRECTORIES", "CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD"]
CACHE = f"{S}/exp/cache_calls.jsonl"
_lock = threading.Lock()
_mem = {}

def _load_cache():
    if _mem or not os.path.exists(CACHE): return
    with open(CACHE) as f:
        for line in f:
            try:
                r = json.loads(line); _mem[r["h"]] = r
            except Exception: pass

def call(model, system, user, schema=None, rep=0, timeout=240, retries=3):
    """One single-turn model call. Returns dict(result, structured, cost, usage, ...)."""
    _load_cache()
    h = hashlib.sha256(json.dumps([model, system, user, schema, rep], sort_keys=True).encode()).hexdigest()
    if h in _mem: return _mem[h]
    if STOP.is_set(): return dict(h=h, model=model, rep=rep, result=None, structured=None, error="skipped: limit reached earlier", cost=0)
    env = {k: v for k, v in os.environ.items() if k not in DROP_ENV}
    env["MAX_THINKING_TOKENS"] = "0"; env["DISABLE_PROMPT_CACHING"] = "1"
    user_s = user.replace("\x00", ""); system = system.replace("\x00", "")
    cmd = ["claude", "-p", user_s, "--model", MODELS.get(model, model), "--system-prompt", system, "--tools", "",
           "--output-format", "json", "--no-session-persistence", "--strict-mcp-config", "--setting-sources", ""]
    if schema: cmd += ["--json-schema", json.dumps(schema)]
    last = None
    for a in range(retries):
        try:
            p = subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout, cwd=f"{S}/runs")
            d = json.loads(p.stdout)
            if d.get("is_error"):
                if _is_limit(d.get("result")):
                    STOP.set(); return dict(h=h, model=model, rep=rep, result=None, structured=None, error="LIMIT: " + str(d.get("result"))[:200], cost=0)
                raise RuntimeError(str(d.get("result"))[:200])
            u = d.get("usage", {})
            rec = dict(h=h, model=model, rep=rep, result=d.get("result"), structured=d.get("structured_output"), cost=d.get("total_cost_usd"),
                       in_tok=u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0),
                       out_tok=u.get("output_tokens"), think=(u.get("output_tokens_details") or {}).get("thinking_tokens"), ms=d.get("duration_ms"), ts=time.time(),
                       models=sorted((d.get("modelUsage") or {}).keys()))
            with _lock:
                _mem[h] = rec
                with open(CACHE, "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return rec
        except Exception as e:
            last = e
            if _is_limit(e) or _is_limit(getattr(p, "stdout", "") if "p" in locals() else ""):
                STOP.set(); break
            time.sleep(3 * (a + 1))
    return dict(h=h, model=model, rep=rep, result=None, structured=None, error=str(last)[:300], cost=0)

def run_many(jobs, workers=6, label=""):
    """jobs: list of dict(id=..., model=..., system=..., user=..., schema=..., rep=...). Returns {id: rec}."""
    out = {}; t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(call, j["model"], j["system"], j["user"], j.get("schema"), j.get("rep", 0)): j["id"] for j in jobs}
        for f in as_completed(fut):
            out[fut[f]] = f.result(); done += 1
            if done % 50 == 0 or done == len(jobs):
                cost = sum((r.get("cost") or 0) for r in out.values()); err = sum(1 for r in out.values() if r.get("error"))
                print(f"[{label}] {done}/{len(jobs)} done, {time.time()-t0:.0f}s, cost ${cost:.2f}, errors {err}" + ("  STOPPED: subscription limit" if STOP.is_set() else ""), flush=True)
    return out

def parse(rec):
    """Extract (decision, confidence, reasoning) from a response."""
    if rec is None: return None, None, ""
    d = rec.get("structured")
    if d is None and rec.get("result"):
        t = rec["result"]; m = re.search(r"\{.*\}", t, flags=re.S)
        if m:
            try: d = json.loads(m.group(0))
            except Exception: d = None
    if not isinstance(d, dict): return None, None, ""
    dec = str(d.get("decision", "")).strip().upper() or None
    try: conf = float(d.get("confidence"))
    except Exception: conf = None
    return dec, conf, str(d.get("reasoning", ""))

# ---------------------------------------------------------------- prompts, verbatim from the repo
import importlib.util
def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
_pa = _load(f"{REPO}/src/agents/prompts.py", "kota_prompts_a"); _pb = _load(f"{REPO}/src/resolution/debate/prompts.py", "kota_prompts_b")
PA_SYSTEM, PA_USER = _pa.SYSTEM_PROMPT, _pa.USER_MESSAGE_TEMPLATE
PB_SYSTEM, PB_USER, R2_SYSTEM, R2_USER = _pb.ROUND_1_SYSTEM_PROMPT, _pb.ROUND_1_USER_TEMPLATE, _pb.ROUND_2_SYSTEM_PROMPT, _pb.ROUND_2_USER_TEMPLATE
SCHEMA = {"type": "object", "properties": {"decision": {"type": "string", "enum": ["YES", "NO"]},
          "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}, "reasoning": {"type": "string"}},
          "required": ["decision", "confidence", "reasoning"], "additionalProperties": False}
SCHEMA_R2 = {"type": "object", "properties": {**SCHEMA["properties"], "revised": {"type": "boolean"}, "convergence_notes": {"type": "string"}},
             "required": ["decision", "confidence", "reasoning", "revised", "convergence_notes"], "additionalProperties": False}

def content(src, mode):
    """mode 'highlights' = what Architecture A showed; 'titles' = what Architecture B showed (source.text is empty)."""
    if mode == "titles": return ""
    hl = src.get("highlights") or []
    return "\n".join(f"- {x}" for x in hl) if hl else (src.get("text") or "")

def fmt_A(e, mode="highlights", sources=None):
    """EvidencePacket.format_for_prompt of the repo (Architecture A layout)."""
    srcs = e["sources"] if sources is None else sources
    lines = [f"Question: {e['question_text']}", f"Resolution Criteria: {e['resolution_criteria']}", f"Resolution Date: {e['resolution_date']}", "", f"Retrieved {len(srcs)} sources:", ""]
    for i, s in enumerate(srcs, 1):
        lines += [f"--- Source {i} ---", f"Title: {s.get('title')}", f"URL: {s.get('url')}"]
        if s.get("published_date"): lines.append(f"Published: {s['published_date']}")
        c = content(s, mode).strip()
        if mode == "titles": lines.append("")
        else: lines.append(f"\n{c}\n" if c else "\n(No extractable content)\n")
    return "\n".join(lines)

def fmt_B(e, mode="titles", sources=None):
    """MultiLLMDebateRunner._format_exa_sources of the repo (Architecture B layout). mode='titles' reproduces the published runs."""
    srcs = e["sources"] if sources is None else sources
    if not srcs: return "No evidence sources were retrieved."
    lines = []
    for i, s in enumerate(srcs, 1):
        lines += [f"--- Source {i} ---", f"Title: {s.get('title')}", f"URL: {s.get('url')}"]
        if s.get("published_date"): lines.append(f"Published: {s['published_date']}")
        lines += [content(s, mode), ""]
    return "\n".join(lines)

def prompt_A(e, mode="highlights", sources=None):
    return PA_SYSTEM, PA_USER.format(evidence_text=fmt_A(e, mode, sources))

def prompt_B(e, mode="titles", sources=None):
    return PB_SYSTEM, PB_USER.format(question=e["question_text"], criteria=e["resolution_criteria"], exa_evidence=fmt_B(e, mode, sources))

# ---------------------------------------------------------------- rows with an exact committed evidence packet
def rows_with_packets():
    A, B, D = load()
    pk = {}
    for f in sorted(glob.glob(f"{REPO}/cache/evidence/*.json")):
        e = json.load(open(f))
        k = e["question_id"] + "|" + re.sub(r"\s+", " ", e["question_text"]).strip() + "|" + re.sub(r"\s+", " ", e["resolution_criteria"]).strip()
        pk.setdefault(k, e)
    Ad = A.drop_duplicates("key").copy()
    Ad = Ad[Ad.key.isin(pk)].reset_index(drop=True)
    Ad["packet"] = [pk[k] for k in Ad.key]
    Bd = B.drop_duplicates("key").set_index("key")
    for m in M:
        Ad[f"B_{m}_d1"] = Bd.loc[Ad.key, f"{m}_d1"].values; Ad[f"B_{m}_ok1"] = Bd.loc[Ad.key, f"{m}_ok1"].values
    Ad["B_maj1_ok"] = Bd.loc[Ad.key, "maj1_ok"].values; Ad["B_final_ok"] = Bd.loc[Ad.key, "final_ok"].values
    return Ad
