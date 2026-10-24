"""How much of the live-search agent's evidence postdates the resolution moment? (no model calls)
The paper's protocol allows evidence published up to close + 1 day. The agent searched in Oct 2026 with no date bound.
Signals of hindsight per run: (a) a cited URL whose path carries a date later than close + 1 day,
(b) a cited Wikipedia page (edited continuously, so its current text is written after the event),
(c) a calendar date later than close + 1 day inside the search-result text the agent read."""
import sys, os, re, json, glob, ast, datetime as dt, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
S = "."
sys.path.insert(0, f"{S}/exp"); from common import rows_with_packets
R = rows_with_packets().set_index("key")
close = {k: pd.to_datetime(e["resolution_date"], utc=True).date() for k, e in R.packet.items()}
MON = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"], 1)}
MON.update({k[:3]: v for k, v in list(MON.items())}); MON["sept"] = 9
MP = "|".join(sorted(MON, key=len, reverse=True))
p_url = [re.compile(r"/(20[12]\d)[/-](0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])(?:/|-|$)"), re.compile(r"(?<!\d)(20[12]\d)(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)")]
p_txt = [re.compile(r"\b(" + MP + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(20[12]\d)\b", re.I), re.compile(r"\b(\d{1,2})\s+(" + MP + r")\.?,?\s+(20[12]\d)\b", re.I),
         re.compile(r"\b(20[12]\d)-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")]
def mk(y, m, d):
    try: return dt.date(int(y), int(m), int(d))
    except ValueError: return None
def url_date(u):
    for p in p_url:
        m = p.search(u)
        if m: return mk(*m.groups())
    return None
def text_dates(t):
    out = []
    for m in p_txt[0].finditer(t): out.append(mk(m.group(3), MON[m.group(1).lower()], m.group(2)))
    for m in p_txt[1].finditer(t): out.append(mk(m.group(3), MON[m.group(2).lower()], m.group(1)))
    for m in p_txt[2].finditer(t): out.append(mk(*m.groups()))
    return [d for d in out if d]
recs = []
for f in glob.glob(f"{S}/agent/haiku_clean/*.json"):
    r = json.load(open(f)); k = r["key"]
    if k not in close: continue
    h = os.path.basename(f)[:-5]; lim = close[k] + dt.timedelta(days=1)  # basename: glob returns "\" on Windows
    src = r.get("sources") or []
    src = ast.literal_eval(src) if isinstance(src, str) else src
    txt = ""
    rawf = f"{S}/agent/haiku_clean/raw/{h}.jsonl"
    try:
        for l in open(rawf):
            if not l.startswith("{"): continue
            j = json.loads(l)
            if j.get("type") == "user" and isinstance(j["message"].get("content"), list):
                for c in j["message"]["content"]:
                    if isinstance(c, dict) and c.get("type") == "tool_result":
                        s = c.get("content"); txt += (s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)) + "\n"
    except FileNotFoundError: pass
    ud = [url_date(u) for u in src]; dated = [d for d in ud if d]
    late_url = [d for d in dated if d > lim]
    wiki = any("wikipedia.org" in u for u in src)
    td = [d for d in text_dates(txt) if d.year <= 2027]
    late_txt = [d for d in td if d > lim]
    recs.append(dict(key=k, decision=r["decision"], n_src=len(src), n_dated=len(dated), late_url=len(late_url) > 0,
                     max_late_days=max([(d - close[k]).days for d in late_url], default=0), wiki=wiki,
                     late_txt=len(late_txt) > 0, late_txt_2026=any(d.year >= 2026 for d in late_txt)))
D = pd.DataFrame(recs)
G = pd.read_csv(f"{S}/exp/e4_agent_haiku_clean.csv").drop_duplicates("key").set_index("key")
D["truth"] = G.truth.reindex(D.key).values; D["ok"] = (D.decision == D.truth).astype(int)
D["wrong"] = ((D.decision != D.truth) & (D.decision != "INSUFFICIENT")).astype(int); D["ins"] = (D.decision == "INSUFFICIENT").astype(int)
D["any_hind"] = D.late_url | D.wiki | D.late_txt
n = len(D)
print(f"runs {n}; cited sources {D.n_src.sum()}, of them with a date in the URL {D.n_dated.sum()}")
print(f"(a) runs citing a URL dated after close + 1 day: {D.late_url.sum()} ({100*D.late_url.mean():.0f}%); median lag of the latest such URL {D[D.late_url].max_late_days.median():.0f} days")
print(f"(b) runs citing Wikipedia: {D.wiki.sum()} ({100*D.wiki.mean():.0f}%)")
print(f"(c) runs whose search results mention a date after close + 1 day: {D.late_txt.sum()} ({100*D.late_txt.mean():.0f}%), a date in 2026 or later: {D.late_txt_2026.sum()}")
print(f"any of (a)-(c): {D.any_hind.sum()} ({100*D.any_hind.mean():.0f}%); none detected: {(~D.any_hind).sum()}")
for lab, m in (("with a hindsight signal", D.any_hind), ("no hindsight signal detected", ~D.any_hind)):
    s = D[m]; print(f"  {lab:30s} n={len(s):3d} right {100*s.ok.mean():5.1f}%  INSUFFICIENT {100*s.ins.mean():5.1f}%  wrong {100*s.wrong.mean():4.1f}%")
print("  caution: the split is not random. Questions with rich later coverage (elections, sports) are also the easy ones.")
D.to_csv(f"{S}/exp/e7_hindsight_runs.csv", index=False)
