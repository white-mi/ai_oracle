"""Evidence packets: retrieval, filtering, merging, rendering and the content check.

Rules of the plan implemented here:
  1  content check: every source shown to a model must carry text (the audited paper sent titles only in Architecture B)
  4  prediction-market sites are removed from search results
  7  dates are compared in code; the model receives a ready flag "published after market close"
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from urllib.parse import urlparse
import os

from .store import question_key

MARKET_DOMAINS = (
    "kalshi.com", "polymarket.com", "manifold.markets", "predictit.org", "metaculus.com",
    "polymarketanalytics.com", "polymarketwhales.info", "betmoose.com",
)
MIN_CHARS_PER_SOURCE = 200


class EmptyEvidenceError(RuntimeError):
    """Raised when a prompt would reach a model without source text."""


def parse_ts(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = str(s).strip().replace("Z", "+00:00")
    try:
        d = datetime.fromisoformat(s)
    except ValueError:
        try:
            d = datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


@dataclass
class Source:
    url: str
    title: str = ""
    published: Optional[str] = None
    highlights: List[str] = field(default_factory=list)
    text: str = ""
    query: str = ""

    def content(self) -> str:
        if self.highlights:
            return "\n".join(f"- {h}" for h in self.highlights)
        return self.text or ""

    def host(self) -> str:
        return (urlparse(self.url).hostname or "").lower()


@dataclass
class Packet:
    market_id: str
    question: str
    criteria: str
    close_time: str
    variant: str = "baseline"
    queries: List[str] = field(default_factory=list)
    sources: List[Source] = field(default_factory=list)
    retrieved_at: str = ""
    dropped_market_sources: int = 0

    @property
    def key(self) -> str:
        return question_key(self.market_id, self.question, self.criteria)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["key"] = self.key
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Packet":
        d = dict(d)
        d.pop("key", None)
        d["sources"] = [Source(**s) for s in d.get("sources", [])]
        return cls(**d)

    def after_close_flags(self) -> List[Optional[bool]]:
        close = parse_ts(self.close_time)
        out = []
        for s in self.sources:
            p = parse_ts(s.published)
            out.append(None if (p is None or close is None) else p > close)
        return out


# ---------------------------------------------------------------- queries
def query_templates(question: str, criteria: str, close_time: str) -> Dict[str, str]:
    """Four query templates for hypothesis H2. 'paper' reproduces the audited paper exactly.
    The other three are drafts; fix their wording in the preregistration commit before any run."""
    q = " ".join(str(question).split())
    month = ""
    c = parse_ts(close_time)
    if c:
        month = c.strftime("%B %Y")
    rules = " ".join(str(criteria).split())[:300]
    return {
        "paper": f'What is the result of this question "{q}"',
        "rules": rules,
        "result": f"News report of the final outcome, {month}: {q}",
        "opposite": f"News report that this did not happen, {month}: {q}",
    }


def end_date_for(close_time: str, days: int = 1) -> str:
    """Publication cut-off used by the paper: market close plus one day."""
    c = parse_ts(close_time)
    return (c + timedelta(days=days)).isoformat().replace("+00:00", "Z")


def drop_market_sites(sources: List[Source], domains=MARKET_DOMAINS):
    kept, dropped = [], 0
    for s in sources:
        h = s.host()
        if any(h == d or h.endswith("." + d) for d in domains):
            dropped += 1
        else:
            kept.append(s)
    return kept, dropped


def exa_search(query: str, end_date: str, n: int = 10, max_chars: int = 2000, api_key: Optional[str] = None,
               exclude=MARKET_DOMAINS) -> List[Source]:
    """Exa search in highlights mode with the paper's settings. Mirrors src/retrieval/exa_retriever.py of the
    audited repo (exa_py SDK). NOT tested against the live API from this environment."""
    from exa_py import Exa  # pip install exa_py

    exa = Exa(api_key or os.environ["EXA_API_KEY"])
    res = exa.search(
        query=query, type="auto", num_results=n, end_published_date=end_date,
        exclude_domains=list(exclude),
        contents={"highlights": {"query": query, "max_characters": max_chars}},
    )
    out = []
    for r in res.results:
        out.append(Source(url=r.url, title=r.title or "", published=getattr(r, "published_date", None),
                          highlights=list(getattr(r, "highlights", None) or []),
                          text=getattr(r, "text", "") or "", query=query))
    return out


def build_packet(market_id, question, criteria, close_time, variant="baseline", templates=("paper",),
                 search=exa_search, n=10, retrieved_at="") -> Packet:
    """Run one or more query templates and merge the results with URL de-duplication (order: template order, then rank)."""
    tq = query_templates(question, criteria, close_time)
    end = end_date_for(close_time)
    seen, merged, dropped = set(), [], 0
    for t in templates:
        kept, d = drop_market_sites(search(tq[t], end, n=n))
        dropped += d
        for s in kept:
            u = s.url.rstrip("/")
            if u not in seen:
                seen.add(u)
                merged.append(s)
    return Packet(market_id, question, criteria, close_time, variant, [tq[t] for t in templates], merged,
                  retrieved_at, dropped)


def merge_packets(packets: List[Packet], variant="merged", max_sources: Optional[int] = 20) -> Packet:
    first = packets[0]
    seen, merged = set(), []
    for p in packets:
        assert p.key == first.key, "merge only packets of the same question"
        for s in p.sources:
            u = s.url.rstrip("/")
            if u not in seen:
                seen.add(u)
                merged.append(s)
    if max_sources:
        merged = merged[:max_sources]
    return Packet(first.market_id, first.question, first.criteria, first.close_time, variant,
                  sum((p.queries for p in packets), []), merged, first.retrieved_at,
                  sum(p.dropped_market_sources for p in packets))


# ---------------------------------------------------------------- rendering and the content check
def render_packet(p: Packet, show_after_close_flag: bool = True) -> str:
    lines = [f"Retrieved {len(p.sources)} sources.", ""]
    for i, (s, late) in enumerate(zip(p.sources, p.after_close_flags()), 1):
        lines += [f"--- Source {i} ---", f"Title: {s.title}", f"URL: {s.url}"]
        if s.published:
            lines.append(f"Published: {s.published}")
        if show_after_close_flag and late is not None:
            lines.append(f"Published after market close: {'yes' if late else 'no'}")
        lines += ["", s.content().strip() or "(no extractable content)", ""]
    return "\n".join(lines)


def check_content(prompt: str, p: Packet, min_chars: int = MIN_CHARS_PER_SOURCE, min_share: float = 0.8) -> None:
    """Rule 1. Raise if the prompt lacks the text of the packet's sources.
    A source counts as present when its first 80 content characters occur in the prompt and it has >= min_chars."""
    if not p.sources:
        return  # an empty packet is a legitimate condition (closed book) and is labelled as such elsewhere
    ok = 0
    for s in p.sources:
        c = s.content().strip()
        if len(c) >= min_chars and c[:80] in prompt:
            ok += 1
    if ok / len(p.sources) < min_share:
        raise EmptyEvidenceError(f"{p.key}: only {ok} of {len(p.sources)} sources have text in the prompt")
