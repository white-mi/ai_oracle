"""Hindsight inside the committed packets: undated pages pass the publication-date filter.
For each source in the 430 packets used in the experiments: is the publication date missing, and does the text
mention a calendar date later than close + 1 day (the paper's cutoff)? No model calls."""
import sys, re, json, numpy as np, pandas as pd, datetime as dt, warnings
warnings.filterwarnings("ignore")
S = "."
sys.path.insert(0, f"{S}/exp"); from common import rows_with_packets
R = rows_with_packets()
MON = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                    "september", "october", "november", "december"], 1)}
MON.update({k[:3]: v for k, v in list(MON.items())}); MON["sept"] = 9
pat_iso = re.compile(r"\b(20\d\d)[-./](0?[1-9]|1[0-2])[-./](0?[1-9]|[12]\d|3[01])\b")
pat_mdy = re.compile(r"\b(" + "|".join(sorted(MON, key=len, reverse=True)) + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(20\d\d)\b", re.I)
pat_dmy = re.compile(r"\b(\d{1,2})\s+(" + "|".join(sorted(MON, key=len, reverse=True)) + r")\.?,?\s+(20\d\d)\b", re.I)
def dates(t):
    out = []
    for y, m, d in pat_iso.findall(t):
        try: out.append(dt.date(int(y), int(m), int(d)))
        except ValueError: pass
    for mo, d, y in pat_mdy.findall(t):
        try: out.append(dt.date(int(y), MON[mo.lower()], int(d)))
        except (ValueError, KeyError): pass
    for d, mo, y in pat_dmy.findall(t):
        try: out.append(dt.date(int(y), MON[mo.lower()], int(d)))
        except (ValueError, KeyError): pass
    return out
rows = []
for _, r in R.iterrows():
    e = r.packet; close = pd.to_datetime(e["resolution_date"], utc=True).date(); lim = close + dt.timedelta(days=1)
    for s in e.get("sources", []):
        txt = " ".join(s.get("highlights") or []) + " " + (s.get("text") or "")
        pub = s.get("published_date") or s.get("publishedDate")
        ds = [d for d in dates(txt) if d.year <= 2027]
        later = [d for d in ds if d > lim]
        rows.append(dict(key=r.key, undated=pub in (None, ""), later=len(later) > 0, max_later=max(later) if later else None,
                         after_2025=any(d.year >= 2026 for d in later), truth=r.truth))
D = pd.DataFrame(rows)
print(f"sources {len(D)} in {D.key.nunique()} packets")
print(f"  undated sources: {D.undated.sum()} ({100*D.undated.mean():.1f}%)")
print(f"  sources whose text names a date later than close + 1 day: {D.later.sum()} ({100*D.later.mean():.1f}%)")
print(f"    of which undated: {(D.later & D.undated).sum()} ; dated: {(D.later & ~D.undated).sum()}")
P = D.groupby("key").agg(undated=("undated", "any"), later=("later", "any"), y2026=("after_2025", "any"), truth=("truth", "first"))
print(f"packets with at least one undated source: {P.undated.sum()} of {len(P)} ({100*P.undated.mean():.1f}%)")
print(f"packets with a source naming a date after close + 1 day: {P.later.sum()} ({100*P.later.mean():.1f}%)")
print(f"packets with a source naming a date in 2026 or later: {P.y2026.sum()} ({100*P.y2026.mean():.1f}%)")
# does it matter for accuracy? fixed-packet Haiku by packet flag (descriptive only, flags are confounded with topic)
E1 = pd.read_csv(f"{S}/exp/e1_haiku_nall_rep0_HA-TA-HB-TB.csv"); HA = E1[E1.cond == "HA"].set_index("key")
an = pd.read_csv(f"{S}/exp/e3_annot.csv").set_index("key")
P["ok"] = HA.reindex(P.index).ok.values
for f in ("later", "y2026"):
    g = P.groupby(f).ok.agg(["count", "mean"]); print(f"  Haiku full-text accuracy by '{f}':", {k: (int(v['count']), round(100*v['mean'], 1)) for k, v in g.iterrows()})
print("  note: a later date in the text is not always hindsight (forecast pages name future dates). Manual reading needed.")
D.to_csv(f"{S}/exp/e6_dates_sources.csv", index=False)
# when were the cached packets retrieved, relative to the market close?
import glob
gaps, months = [], {}
for f in glob.glob(f"{S}/repo/cache/evidence/*.json"):
    e = json.load(open(f)); t = e.get("retrieval_timestamp")
    if not t: continue
    months[t[:7]] = months.get(t[:7], 0) + 1
    gaps.append((pd.to_datetime(t, utc=True) - pd.to_datetime(e["resolution_date"], utc=True)).days)
g = np.array(gaps)
print(f"cached packets with a retrieval timestamp: {len(g)}; months {months}; days from close to retrieval min {g.min()} median {np.median(g):.0f} max {g.max()}")
