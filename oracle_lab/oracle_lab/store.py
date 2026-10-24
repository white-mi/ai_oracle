"""Keys and on-disk caches.

Rule 2 of the plan: a cache key is built from the market id, the question text and the resolution rules.
The row number of an input file never enters a key (that is how the audited paper ended up with
different evidence for the same question in its two architectures).
"""
import hashlib
import json
import os
import re
import threading


def norm_ws(s) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _h(s: str, n: int = 12) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def question_key(market_id: str, question: str, criteria: str) -> str:
    """Stable identity of one market question. Same inputs give the same key in every run and every file."""
    return f"{norm_ws(market_id)}|{_h(norm_ws(question))}|{_h(norm_ws(criteria))}"


def call_hash(**fields) -> str:
    """Hash of everything that determines a model response (model, prompts, params, repetition index)."""
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


class CallCache:
    """Append-only JSONL cache of model calls. Failed calls are not cached, so a rerun retries only them."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._mem = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        self._mem[r["h"]] = r
                    except (ValueError, KeyError):
                        continue

    def get(self, h: str):
        return self._mem.get(h)

    def put(self, rec: dict):
        with self._lock:
            self._mem[rec["h"]] = rec
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def __len__(self):
        return len(self._mem)


class PacketStore:
    """One JSON file per (question key, packet variant). Every condition of an experiment reads the same file (rule 3)."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def path(self, key: str, variant: str) -> str:
        return os.path.join(self.root, variant, _h(key, 20) + ".json")

    def has(self, key: str, variant: str) -> bool:
        return os.path.exists(self.path(key, variant))

    def save(self, packet) -> str:
        p = self.path(packet.key, packet.variant)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(packet.to_dict(), f, ensure_ascii=False, indent=1)
        return p

    def load(self, key: str, variant: str):
        from .evidence import Packet
        with open(self.path(key, variant), encoding="utf-8") as f:
            return Packet.from_dict(json.load(f))
