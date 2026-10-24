"""Experiment 7: second round with FULL source text (Haiku revises after seeing GPT-5 nano and DeepSeek V3 answers)."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
S = "."
sys.path.insert(0, f"{S}/analysis"); sys.path.insert(0, f"{S}/exp")
from kota import mcnemar, cluster_boot, load
D = pd.read_csv(f"{S}/exp/e3_deb.csv")
A, B, _ = load(); Au = A.drop_duplicates("key").set_index("key")
D["gpt_conf"] = Au.loc[D.key, "gpt4o_confidence"].values; D["ds_conf"] = Au.loc[D.key, "deepseek_confidence"].values
E1 = pd.read_csv(f"{S}/exp/e1_haiku_nall_rep0_HA-TA-HB-TB.csv"); HB = E1[E1.cond == "HB"].set_index("key")
D["h1_conf"] = HB.loc[D.key, "confidence"].values
t = D.truth.values
D["h1_ok"] = (D.h1 == t).astype(int); D["h2_ok"] = (D.h2 == t).astype(int); D["gpt_ok"] = (D.gpt == t).astype(int); D["ds_ok"] = (D.deepseek == t).astype(int)
yes = (D.h1 == "YES").astype(int) + (D.gpt == "YES").astype(int) + (D.deepseek == "YES").astype(int)
D["maj"] = np.where(yes >= 2, "YES", "NO"); D["maj_ok"] = (D.maj == t).astype(int)
w = np.where(D.h1 == "YES", 1, -1) * D.h1_conf + np.where(D.gpt == "YES", 1, -1) * D.gpt_conf + np.where(D.deepseek == "YES", 1, -1) * D.ds_conf
D["wtd"] = np.where(w > 0, "YES", "NO"); D["wtd_ok"] = (D.wtd == t).astype(int)
D["flip"] = (D.h1 != D.h2).astype(int)
D["c2w"] = ((D.h1_ok == 1) & (D.h2_ok == 0)).astype(int); D["w2c"] = ((D.h1_ok == 0) & (D.h2_ok == 1)).astype(int)
D["pos"] = np.where((D.h1 == D.gpt) & (D.h1 == D.deepseek), "both peers agree", np.where((D.h1 != D.gpt) & (D.h1 != D.deepseek), "lone dissenter", "peers split"))
print("rows", len(D), "| invalid h2:", int((~D.h2.isin(["YES", "NO"])).sum()), "| cost", round(D.cost.sum(), 2))
for lab, sub in [("NON-UNANIMOUS trios", D[~D.unanimous]), ("UNANIMOUS controls", D[D.unanimous])]:
    print(f"\n== {lab}: n={len(sub)}; true YES {int(sub.y.sum())}")
    print(f"  Haiku round 1 {100*sub.h1_ok.mean():.1f} | GPT-5 nano {100*sub.gpt_ok.mean():.1f} | DeepSeek {100*sub.ds_ok.mean():.1f} | majority of 3 {100*sub.maj_ok.mean():.1f} | weighted {100*sub.wtd_ok.mean():.1f} | Haiku round 2 {100*sub.h2_ok.mean():.1f}")
    print(f"  counts correct: h1 {int(sub.h1_ok.sum())}, gpt {int(sub.gpt_ok.sum())}, ds {int(sub.ds_ok.sum())}, maj {int(sub.maj_ok.sum())}, wtd {int(sub.wtd_ok.sum())}, h2 {int(sub.h2_ok.sum())}")
    print(f"  flips {int(sub.flip.sum())} ({100*sub.flip.mean():.1f}%): correct->wrong {int(sub.c2w.sum())}, wrong->correct {int(sub.w2c.sum())}; YES->NO {int(((sub.h1=='YES')&(sub.h2=='NO')).sum())}, NO->YES {int(((sub.h1=='NO')&(sub.h2=='YES')).sum())}")
    for a, b in [("h2_ok", "h1_ok"), ("h2_ok", "maj_ok"), ("h2_ok", "wtd_ok"), ("maj_ok", "h1_ok")]:
        n10, n01, p = mcnemar(sub[a], sub[b]); print(f"    {a} vs {b}: {a}-only {n10}, {b}-only {n01}, exact McNemar p={p:.3f}")
    g = sub.groupby("pos").agg(n=("flip", "size"), flips=("flip", "sum"), h1_right=("h1_ok", "sum"), h2_right=("h2_ok", "sum"), maj_right=("maj_ok", "sum"), c2w=("c2w", "sum"), w2c=("w2c", "sum"))
    print(g.to_string())
nu = D[~D.unanimous]
ld = nu[nu.pos == "lone dissenter"]
print("\nlone dissenter Haiku: right in round 1:", int(ld.h1_ok.sum()), "of", len(ld), "| kept answer:", int((ld.flip == 0).sum()),
      "| when right kept", int(((ld.h1_ok == 1) & (ld.flip == 0)).sum()), "of", int(ld.h1_ok.sum()), "| when wrong flipped", int(((ld.h1_ok == 0) & (ld.flip == 1)).sum()), "of", int((ld.h1_ok == 0).sum()))
print("  lone dissenter by Haiku's own answer:", ld.groupby("h1").agg(n=("flip", "size"), flips=("flip", "sum"), right1=("h1_ok", "sum"), right2=("h2_ok", "sum")).to_dict("index"))
ps = nu[nu.pos == "peers split"]; print("peers split: Haiku right r1", int(ps.h1_ok.sum()), "of", len(ps), "| flips", int(ps.flip.sum()), "| right r2", int(ps.h2_ok.sum()))
print("revised flag true:", int((D.revised == True).sum()), "| actual flips", int(D.flip.sum()), "| mismatch", int(((D.revised == True) != (D.flip == 1)).sum()))
print("confidence: mean h1", round(D.h1_conf.mean(), 3), "mean h2", round(D.h2_conf.mean(), 3), "| non-unanimous h1", round(nu.h1_conf.mean(), 3), "h2", round(nu.h2_conf.mean(), 3),
      "| unanimous h1", round(D[D.unanimous].h1_conf.mean(), 3), "h2", round(D[D.unanimous].h2_conf.mean(), 3))
# flip vs confidence of own answer
print("flip rate by own round-1 confidence (non-unanimous):", nu.groupby(pd.cut(nu.h1_conf, [-0.01, 0.6, 0.85, 0.95, 1.0])).agg(n=("flip", "size"), flips=("flip", "sum")).to_dict("index"))
# all 430 rows: unsampled unanimous rows keep the round-1 answer (controls show how often that is wrong)
keys = list(HB.index); allu = len(HB) - len(nu); c = D[D.unanimous]
print(f"\nall 430 rows: non-unanimous {len(nu)}, unanimous {allu} (100 sampled; flips among sampled {int(c.flip.sum())}, c2w {int(c.c2w.sum())}, w2c {int(c.w2c.sum())})")
full = pd.DataFrame({"key": keys}); full["h1"] = HB.decision.astype(str).str.upper().values; full["truth"] = HB.truth.values; full["qid"] = HB.question_id.values
full["gpt"] = Au.loc[keys, "gpt4o_d"].values; full["ds"] = Au.loc[keys, "deepseek_d"].values
yy = (full.h1 == "YES").astype(int) + (full.gpt == "YES").astype(int) + (full.ds == "YES").astype(int); full["maj"] = np.where(yy >= 2, "YES", "NO")
m = D.set_index("key").h2; full["h2"] = [m.get(k, h) for k, h in zip(full.key, full.h1)]
for col in ("h1", "gpt", "ds", "maj", "h2"): print(f"  {col:4s} accuracy {100*(full[col] == full.truth).mean():.1f} ({int((full[col] == full.truth).sum())}/430)")
for a, b in [("h2", "maj"), ("h2", "h1"), ("maj", "h1")]:
    n10, n01, p = mcnemar(full[a] == full.truth, full[b] == full.truth)
    d = ((full[a] == full.truth).astype(float) - (full[b] == full.truth).astype(float)).values; lo, hi = cluster_boot(d, full.qid.values, B=3000)
    print(f"  {a} vs {b}: {n10} vs {n01}, p={p:.3f}; diff {100*d.mean():+.2f} pp [{100*lo:+.2f}, {100*hi:+.2f}]")
# comparison with the published debate (titles only), same 430 rows: use B round1->round2 of the three paper models
Bu = B.drop_duplicates("key").set_index("key").reindex(keys)
for mm in ("gpt4o", "deepseek", "llama"):
    f = (Bu[f"{mm}_d1"] != Bu[f"{mm}_d2"]); print(f"  published B on the same 430 rows, {mm}: flips {int(f.sum())}, acc r1 {100*Bu[f'{mm}_ok1'].mean():.1f} r2 {100*Bu[f'{mm}_ok2'].mean():.1f}")
print("\nexamples of flips (non-unanimous):")
for _, r in nu[nu.flip == 1].iterrows():
    print(f"  truth {r.truth} | h1 {r.h1} ({r.h1_conf}) -> h2 {r.h2} ({r.h2_conf}) | gpt {r.gpt} ds {r.deepseek} | {r.key.split('|')[1][:80]} || {str(r.notes)[:300]}")
D.to_pickle(f"{S}/exp/e3_deb.pkl")
