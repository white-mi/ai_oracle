"""Model clients with a shared disk cache (rule 8: raw response, model version and request id are kept).

All clients call APIs with your own keys and spend limits (OpenRouter, Anthropic Console, TypeSafe).
Do not route bulk runs through a chat subscription. None of the network clients was exercised against the
live services from the environment where this scaffold was written. FakeClient is for dry runs and tests.
"""
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .store import CallCache, call_hash


def parse_json(text):
    """First JSON object in a model reply (tolerates code fences and prose around it)."""
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text)
    depth, start = 0, None
    for i, ch in enumerate(t):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1])
                except ValueError:
                    start = None
    return None


class Client:
    provider = "base"

    def __init__(self, model, max_tokens=1500, temperature=0.0, timeout=180):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    @property
    def name(self):
        return f"{self.provider}:{self.model}"

    def _complete(self, system, user):  # -> dict(text, model, request_id, in_tok, out_tok, raw)
        raise NotImplementedError


class OpenRouterClient(Client):
    """OpenAI-compatible chat endpoint. Reasoning models (e.g. GPT-5 nano) spend output tokens on reasoning,
    so keep max_tokens generous."""
    provider = "openrouter"
    URL = "https://openrouter.ai/api/v1/chat/completions"

    def _complete(self, system, user):
        import requests
        r = requests.post(self.URL, timeout=self.timeout,
                          headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                          json={"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        r.raise_for_status()
        d = r.json()
        u = d.get("usage") or {}
        return dict(text=d["choices"][0]["message"].get("content") or "", model=d.get("model"), request_id=d.get("id"),
                    in_tok=u.get("prompt_tokens"), out_tok=u.get("completion_tokens"), raw=d)


class AnthropicClient(Client):
    """Claude Messages API with an API key from the Console (workspace spend limit recommended).
    For thousands of calls the Message Batches API halves the price; this client is the synchronous variant."""
    provider = "anthropic"
    URL = "https://api.anthropic.com/v1/messages"

    def _complete(self, system, user):
        import requests
        r = requests.post(self.URL, timeout=self.timeout,
                          headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                                   "content-type": "application/json"},
                          json={"model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
                                "system": system, "messages": [{"role": "user", "content": user}]})
        r.raise_for_status()
        d = r.json()
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        u = d.get("usage") or {}
        return dict(text=text, model=d.get("model"), request_id=d.get("id"), in_tok=u.get("input_tokens"),
                    out_tok=u.get("output_tokens"), raw=d)


class FakeClient(Client):
    """Deterministic stand-in: p_yes is a hash of the prompt. Lets the whole pipeline run without network or cost."""
    provider = "fake"

    def _complete(self, system, user):
        p = int(hashlib.sha256((system + user).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        dec = "INSUFFICIENT" if 0.45 < p < 0.55 and "INSUFFICIENT" in user else ("YES" if p > 0.5 else "NO")
        text = json.dumps({"decision": dec, "p_yes": round(p, 3), "changed": False, "reasoning": "fake"})
        return dict(text=text, model=self.model, request_id=None, in_tok=len(user) // 4, out_tok=20, raw=None)


def make_client(spec: str, **kw) -> Client:
    """'openrouter:openai/gpt-5-nano', 'anthropic:claude-haiku-4-5-20251001', 'fake:any'."""
    provider, model = spec.split(":", 1)
    return {"openrouter": OpenRouterClient, "anthropic": AnthropicClient, "fake": FakeClient}[provider](model, **kw)


def call(client: Client, cache: CallCache, system: str, user: str, rep: int = 0, retries: int = 3):
    h = call_hash(client=client.name, system=system, user=user, rep=rep, max_tokens=client.max_tokens,
                  temperature=client.temperature)
    hit = cache.get(h)
    if hit:
        return hit
    last = None
    for a in range(retries):
        try:
            out = client._complete(system, user)
            rec = dict(h=h, client=client.name, rep=rep, ts=time.time(), parsed=parse_json(out["text"]),
                       **{k: v for k, v in out.items() if k != "raw"})
            cache.put(rec)
            return rec
        except Exception as e:  # network errors, 429, 5xx
            last = e
            time.sleep(2 ** a * 3)
    return dict(h=h, client=client.name, rep=rep, error=str(last)[:300], parsed=None)


def run_many(client, cache, jobs, workers=8, label=""):
    """jobs: list of dict(id, system, user, rep). Returns {id: record}. Prints progress and failures."""
    out, t0 = {}, time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(call, client, cache, j["system"], j["user"], j.get("rep", 0)): j["id"] for j in jobs}
        for i, f in enumerate(as_completed(fut), 1):
            out[fut[f]] = f.result()
            if i % 100 == 0 or i == len(jobs):
                err = sum(1 for r in out.values() if r.get("error"))
                print(f"[{label} {client.name}] {i}/{len(jobs)} {time.time() - t0:.0f}s errors={err}", flush=True)
    return out


# ---------------------------------------------------------------- Jev (TypeSafe), per typesafe_sdk 0.7.2
def jev_p_yes(state: dict, instructions: str, model="jev-1.13.0", client=None) -> dict:
    """One Noul question. Pin the version and keep response.model and request_id (rule 8).
    Field names inside `state` are ours; the API does not prescribe them. Not run against the live API."""
    from typesafe_sdk import Noul, TypeSafeClient  # pip install typesafe-sdk==0.7.2

    def ask(c):
        r = c.system_one(state=state, model=model, questions={"yes": Noul(instructions=instructions)})
        return dict(p_yes=r.nouls["yes"].noul, model=r.model, request_id=r.request_id,
                    in_tok=r.usage.input_tokens, out_tok=r.usage.output_tokens)

    if client is not None:
        return ask(client)
    with TypeSafeClient() as c:  # reads TYPESAFE_API_KEY from the environment
        return ask(c)


def jev_state(packet) -> dict:
    """State for Jev: rules, close time and sources, with the after-close flag computed in code (rule 7)."""
    flags = packet.after_close_flags()
    return {
        "question": packet.question,
        "resolution_rule": packet.criteria,
        "market_close": packet.close_time,
        "evidence": [{"source": s.url, "title": s.title, "published": s.published,
                      "published_after_close": f, "text": s.content()} for s, f in zip(packet.sources, flags)],
    }
