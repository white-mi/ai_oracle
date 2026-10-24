"""Test-retest noise: 36 questions that appear twice in the published Architecture A file,
plus other repeated-run files in the repo."""
import numpy as np, pandas as pd, glob, os
from kota import *
A, B, D = load()
dup = A[A.key.duplicated(keep=False)].sort_values("key")
g = dup.groupby("key")
print("questions run twice in file A:", g.ngroups, "| rows:", len(dup))
first = g.nth(0).reset_index(drop=True); second = g.nth(1).reset_index(drop=True)
print("positions of copies (file index): first", sorted(g.apply(lambda x: x.index[0]).values)[:5], "... second copies min index", min(g.apply(lambda x: x.index[1]).values))
tot_dis = 0
for m in M:
    same = (first[f"{m}_d"].values == second[f"{m}_d"].values)
    dc = np.abs(first[f"{m}_confidence"].values - second[f"{m}_confidence"].values)
    tot_dis += int((~same).sum())
    print(f"{NAMES[m]:14s} same decision in {int(same.sum())}/{len(same)} ({100*same.mean():.0f}%); mean |conf diff|={dc.mean():.3f}; identical confidence in {int((dc==0).sum())}")
print(f"all models: decisions differ in {tot_dis}/{3*len(first)} = {100*tot_dis/(3*len(first)):.1f}% of repeated calls")
print("majority differs in", int((first.maj.values != second.maj.values).sum()), "of", len(first), "; weighted differs in", int((first.wtd.values != second.wtd.values).sum()))
same_reason = (first.gpt4o_reasoning.values == second.gpt4o_reasoning.values).sum(); print("identical GPT reasoning text in", same_reason, "pairs (0 => independent calls)")
# other files with overlapping questions
print("\nOther result files in repo:")
for f in sorted(glob.glob(f"{REPO}/results/*.csv") + glob.glob(f"{REPO}/results/architecture_a_results/*.csv")):
    try:
        d = pd.read_csv(f)
    except Exception as e:
        print(os.path.basename(f), "unreadable", str(e)[:60]); continue
    cols = [c for c in d.columns if c.endswith("_decision")]
    print(f"  {os.path.relpath(f, REPO):62s} rows={len(d):5d} decision cols={cols[:6]}")
