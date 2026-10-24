"""D2 export: settled Kalshi markets that closed inside a window.

Endpoint per docs.kalshi.com (GET /trade-api/v2/markets): status=settled is compatible with min/max_settled_ts,
while min/max_close_ts work with status=closed or no status. We therefore page with close-time filters and keep
markets whose result is 'yes' or 'no'. The docs do not state whether auth is required. Not run from this environment.
"""
import time
from datetime import datetime, timezone

BASE = "https://api.elections.kalshi.com/trade-api/v2"


def _ts(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def settled_markets(close_from: str, close_to: str, page=1000, sleep=0.3, session=None):
    import requests
    s = session or requests.Session()
    cursor, out = "", []
    while True:
        params = {"limit": page, "min_close_ts": _ts(close_from), "max_close_ts": _ts(close_to), "mve_filter": "exclude"}
        if cursor:
            params["cursor"] = cursor
        r = s.get(f"{BASE}/markets", params=params, timeout=60)
        r.raise_for_status()
        d = r.json()
        for m in d.get("markets", []):
            if m.get("result") in ("yes", "no"):
                out.append(m)
        cursor = d.get("cursor") or ""
        if not cursor:
            return out
        time.sleep(sleep)


def to_rows(markets):
    """Rows in the dataset format used by run.py: market_id, question, criteria, close_time, outcome, event_ticker."""
    rows = []
    for m in markets:
        q = m.get("title") or m.get("yes_sub_title") or m["ticker"]  # 'title' is deprecated in the docs
        crit = " ".join(x for x in (m.get("rules_primary"), m.get("rules_secondary")) if x)
        rows.append(dict(market_id=m["ticker"], event_ticker=m.get("event_ticker"), question=q, criteria=crit,
                         close_time=m.get("close_time"), outcome=m["result"].upper()))
    return rows
