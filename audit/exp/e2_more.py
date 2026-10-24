import sys, pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
S = "."
sys.path.insert(0, f"{S}/analysis")
from kota import mcnemar, cluster_boot, wilson, load, M, NAMES
P = pickle.load(open(f"{S}/exp/e2_signals.pkl", "rb")); SIG, y, keys, mk = P["SIG"], P["y"], P["keys"], P["mk"]
A, B, D = load(); Au = A.drop_duplicates("key").set_index("key").loc[keys]
E1 = pd.read_csv(f"{S}/exp/e1_haiku_nall_rep0_HA-TA-HB-TB.csv"); HA = E1[E1.cond == "HA"].set_index("key").loc[keys]
def aurc(score, ok):
    o = np.argsort(-np.asarray(score, float), kind="stable"); err = 1 - np.asarray(ok, float)[o]
    return (np.cumsum(err) / np.arange(1, len(err) + 1)).mean()
def cov_at(score, ok, target):
    o = np.argsort(-score, kind="stable"); acc = np.cumsum(ok[o]) / np.arange(1, len(ok) + 1)
    ks = [k for k in range(20, len(ok) + 1) if acc[k - 1] >= target]; return (max(ks) / len(ok)) if ks else 0.0
def top(score, ok, cov):
    o = np.argsort(-score, kind="stable")[:int(cov * len(ok))]; return ok[o].mean(), int((1 - ok[o]).sum()), len(o)
sigs = {}
comp = Au.unan.values + Au.mean_conf.values
sigs["paper: 3 models, unanimity + mean confidence (majority decision)"] = (comp, Au.maj_ok.values.astype(float))
sigs["paper: GPT-5 nano confidence alone (its own decision)"] = (Au.gpt4o_confidence.values.astype(float), Au.gpt4o_ok.values.astype(float))
v = SIG["verbalized confidence (decision + confidence, with reasoning)"]; sigs["Haiku: verbalized confidence"] = (np.nan_to_num(v["conf"], nan=0.0), np.nan_to_num(v["ok"], nan=0.0))
b = SIG["bare P(YES), no reasoning"]; sigs["Haiku: bare P(YES) without reasoning"] = (np.nan_to_num(b["conf"], nan=0.5), np.nan_to_num(b["ok"], nan=0.0))
e = SIG["bare P(evidence states it occurred), no reasoning"]; sigs["Haiku: bare P(evidence states it occurred)"] = (np.nan_to_num(e["conf"], nan=0.5), np.nan_to_num(e["ok"], nan=0.0))
sc = SIG["self-consistency: unanimous of 3 repeats + mean confidence"]; sigs["Haiku: 3 repeats, unanimity + mean confidence"] = (np.nan_to_num(sc["conf"], nan=0.0), np.nan_to_num(sc["ok"], nan=0.0))
# two-view agreement: verbalized decision vs bare decision; composite = agree + mean of the two confidences; decision = verbalized
agree = ((v["p"] > 0.5) == (b["p"] > 0.5)).astype(float); agree[np.isnan(v["p"]) | np.isnan(b["p"])] = 0
sigs["Haiku: two views agree + mean confidence"] = (agree + (np.nan_to_num(v["conf"]) + np.nan_to_num(b["conf"], nan=0.5)) / 2, np.nan_to_num(v["ok"], nan=0.0))
# 4 models: 3 published + Haiku
votes = np.stack([(Au[f"{m}_d"].values == "YES").astype(int) for m in M] + [(HA.decision.astype(str).str.upper().values == "YES").astype(int)])
yes4 = votes.sum(0); un4 = ((yes4 == 0) | (yes4 == 4)).astype(float); mc4 = (Au.mean_conf.values * 3 + np.nan_to_num(HA.confidence.values.astype(float), nan=0.5)) / 4
dec4 = np.where(yes4 >= 3, 1, np.where(yes4 <= 1, 0, (HA.decision.astype(str).str.upper().values == "YES").astype(int)))  # tie -> Haiku
ok4 = (dec4 == y).astype(float)
sigs["4 models (paper's 3 + Haiku): unanimity + mean confidence"] = (un4 + mc4, ok4)
print(f"{'signal':66s} acc    AURC    top25  top50  top75   cov@97%  cov@95%")
for lab, (s, ok) in sigs.items():
    print(f"{lab:66s} {100*ok.mean():5.1f}  {aurc(s, ok):.4f}  {100*top(s, ok, .25)[0]:5.1f}  {100*top(s, ok, .5)[0]:5.1f}  {100*top(s, ok, .75)[0]:5.1f}   {100*cov_at(s, ok, .97):5.1f}    {100*cov_at(s, ok, .95):5.1f}")
# bootstrap AURC differences vs paper composite (cluster by market)
rng = np.random.default_rng(0); uniq, inv = np.unique(mk, return_inverse=True); idx = [np.where(inv == i)[0] for i in range(len(uniq))]
base = "paper: 3 models, unanimity + mean confidence (majority decision)"
names = [k for k in sigs if k != base]; diffs = {k: [] for k in names}
for bi in range(2000):
    ii = np.concatenate([idx[j] for j in rng.integers(0, len(uniq), len(uniq))])
    a0 = aurc(sigs[base][0][ii], sigs[base][1][ii])
    for k in names: diffs[k].append(aurc(sigs[k][0][ii], sigs[k][1][ii]) - a0)
print("\nAURC difference vs the paper's 3-model composite on the same 430 rows (negative = better), cluster bootstrap 95% CI")
for k in names:
    d = np.array(diffs[k]); print(f"  {k:66s} {aurc(*sigs[k]) - aurc(*sigs[base]):+.4f}  [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")
print("\n4-model vote: unanimous rows", int(un4.sum()), f"({100*un4.mean():.1f}%), accuracy there {100*ok4[un4==1].mean():.1f}; 3-model unanimous rows {int(Au.unan.sum())} acc {100*Au.maj_ok[Au.unan==1].mean():.1f}")
print("  among 3-model-unanimous rows, Haiku disagrees on", int(((Au.unan.values == 1) & (un4 == 0)).sum()), "rows; 3-model majority accuracy there", round(100*Au.maj_ok.values[(Au.unan.values == 1) & (un4 == 0)].mean(), 1), "; Haiku accuracy there", round(100*np.nanmean(HA.ok.values[(Au.unan.values == 1) & (un4 == 0)]), 1))
uy = (yes4 == 4); un = (yes4 == 0); print(f"  4-unanimous YES: n={uy.sum()} acc {100*ok4[uy].mean():.1f}; 4-unanimous NO: n={un.sum()} acc {100*ok4[un].mean():.1f}")
c = HA.confidence.values.astype(float); print("\nHaiku rows with verbalized confidence < 0.5:", int((c < 0.5).sum()), "; accuracy there", round(100*np.nanmean(HA.ok.values[c < 0.5]), 1), "; decisions", dict(pd.Series(HA.decision.values[c < 0.5]).value_counts()))
# abstain by category
AB = pd.read_csv(f"{S}/exp/e2_abstain_haiku_nall.csv").drop_duplicates("key").set_index("key").reindex(keys)
d = AB.decision.astype(str).str.upper().values; ins = d == "INSUFFICIENT"
g = pd.DataFrame(dict(cat=HA.category.values, ins=ins, y=y, ok=(d == np.where(y == 1, "YES", "NO")))).groupby("cat").agg(n=("ins", "size"), abstain=("ins", "mean"))
print("\nabstention rate by category (n>=15):"); print((g[g.n >= 15] * [1, 100]).round(1).to_string())
# abstained rows: truth and what evidence dates look like
print("abstained rows: true YES", int(y[ins].sum()), "of", int(ins.sum()), "; answered rows true YES share", round(100*y[~ins].mean(), 1))
# reasons in abstain reasoning
r = AB.reasoning.astype(str)[ins]
import re
for lab, pat in [("mentions lack of post-event / outcome info", r"(no (source|evidence|information).{0,60}(after|post|outcome|result|actual))|(does not (state|establish|confirm|report))|(not (yet )?(occurred|happened|confirmed|reported))|predat|before the (event|resolution|deadline)"), ("mentions forecasts/predictions only", r"forecast|predict|projection|odds|preview|speculat")]:
    print(f"  abstain reasoning: {lab}: {100*r.str.contains(pat, flags=re.I, regex=True).mean():.0f}%")
