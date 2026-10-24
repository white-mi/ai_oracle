"""Leakage audit, part 2: is the larger Haiku advantage on May-Jul markets (inside training data) a memory effect?
Difference-in-differences of (Haiku - paper majority) before vs after Aug 2025, cluster bootstrap by market,
plus conditions where memory should show most (titles only, packets without the outcome)."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
S = "."
sys.path.insert(0, f"{S}/exp"); from common import rows_with_packets, load
R = rows_with_packets(); A, B, D = load(); Au = A.drop_duplicates("key").set_index("key").reindex(R.key)
R["month"] = pd.to_datetime([e["resolution_date"] for e in R.packet], utc=True, format="ISO8601").strftime("%Y-%m")
aft = (R.month >= "2025-08").values; y = R.y.values; mk = R.question_id.astype(str).values
E1 = pd.read_csv(f"{S}/exp/e1_haiku_nall_rep0_HA-TA-HB-TB.csv")
ok = {c: E1[E1.cond == c].set_index("key").reindex(R.key).ok.values.astype(float) for c in ("HA", "TA", "HB", "TB")}
CB = pd.read_csv(f"{S}/exp/e2_closed_haiku_nall.csv").set_index("key").reindex(R.key); p = CB.p_yes.astype(float).values
ok["closed"] = ((p > 0.5) == y).astype(float)
pm = Au.maj_ok.values.astype(float)
an = pd.read_csv(f"{S}/exp/e3_annot.csv").set_index("key").reindex(R.key).label.values

def did(a, b, B=5000, seed=0):
    """(mean(a-b) before) - (mean(a-b) after), cluster bootstrap by market within each period"""
    d = a - b; rng = np.random.default_rng(seed); out = []
    groups = {}
    for per in (False, True):
        sel = np.where(aft == per)[0]; u, inv = np.unique(mk[sel], return_inverse=True)
        groups[per] = [sel[inv == i] for i in range(len(u))]
    for _ in range(B):
        m = []
        for per in (False, True):
            g = groups[per]; ii = np.concatenate([g[j] for j in rng.integers(0, len(g), len(g))]); m.append(d[ii].mean())
        out.append(m[0] - m[1])
    est = d[~aft].mean() - d[aft].mean(); lo, hi = np.percentile(out, [2.5, 97.5]); pv = 2 * min(np.mean(np.array(out) <= 0), np.mean(np.array(out) >= 0))
    return est, lo, hi, min(pv, 1.0)

print("== Accuracy by period (n before = %d, after = %d)" % ((~aft).sum(), aft.sum()))
for lab, v in (("Haiku closed book", ok["closed"]), ("Haiku titles only, prompt A", ok["TA"]), ("Haiku full text, prompt A", ok["HA"]),
               ("Haiku titles only, prompt B", ok["TB"]), ("Haiku full text, prompt B", ok["HB"]), ("paper majority (A)", pm)):
    print(f"  {lab:32s} before {100*v[~aft].mean():5.1f}%   after {100*v[aft].mean():5.1f}%")
print("\n== Difference-in-differences: (Haiku minus paper majority) before minus after, cluster bootstrap by market")
for lab, c in (("full text, prompt A", "HA"), ("titles only, prompt A", "TA"), ("full text, prompt B", "HB"), ("titles only, prompt B", "TB")):
    e, lo, hi, pv = did(ok[c], pm); print(f"  {lab:22s} {100*e:+.1f} pp [{100*lo:+.1f}, {100*hi:+.1f}], p={pv:.2f}")
e, lo, hi, pv = did(ok["HA"], ok["TA"]); print(f"  text gain (HA minus TA) before minus after: {100*e:+.1f} pp [{100*lo:+.1f}, {100*hi:+.1f}], p={pv:.2f}")
print("\n== Where Haiku's extra advantage before Aug comes from (full text, prompt A)")
for per, lab in ((False, "before"), (True, "after")):
    s = (aft == per)
    hp = ((ok["HA"] == 1) & (pm == 0) & s).sum(); ph = ((ok["HA"] == 0) & (pm == 1) & s).sum()
    print(f"  {lab}: Haiku right & paper wrong {hp}, paper right & Haiku wrong {ph}")
    for t, nm in ((1, "YES"), (0, "NO")):
        ss = s & (y == t); print(f"     true {nm}: n={ss.sum():3d}  Haiku {100*ok['HA'][ss].mean():5.1f}%  paper {100*pm[ss].mean():5.1f}%")
print("\n== Memory should help most where the packet lacks the outcome (annotated SILENT rows with true YES)")
lab_ok = pd.notna(an)
for per, lab in ((False, "before"), (True, "after")):
    s = (aft == per) & (an == "SILENT") & (y == 1)
    print(f"  {lab}: n={s.sum():2d}  fixed-packet Haiku says YES on {int(ok['HA'][s].sum())}  closed-book p>0.5 on {int((p[s] > 0.5).sum())}")
print("  annotated rows by period:", {lab: int(((aft == per) & lab_ok).sum()) for per, lab in ((False, "before"), (True, "after"))})
