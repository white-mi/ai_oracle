"""Debate dynamics in file B and the limit of majority voting under correlated errors (file A)."""
import numpy as np, pandas as pd
from scipy.stats import norm as N
from scipy.optimize import brentq
from scipy import integrate
from kota import *
A, B, D = load()
P = print
P("== 1. Round 1 -> round 2 transitions in B, by the agent's position relative to its two peers")
rows = []
for m in M:
    others = [o for o in M if o != m]
    own = B[m+"_d1"]; peers_same = sum((B[o+"_d1"] == own).astype(int) for o in others)   # 0,1,2 peers agree with me
    flip = (B[m+"_d2"] != own).astype(int)
    for k in (2, 1, 0):
        mk = peers_same == k
        rows.append(dict(model=NAMES[m], peers_agreeing=k, n=int(mk.sum()), flips=int(flip[mk].sum()), flip_rate=round(100*flip[mk].mean(), 1),
                         was_correct=int(B.loc[mk, m+"_ok1"].sum()), correct_to_wrong=int((flip[mk] & (B.loc[mk, m+"_ok1"] == 1)).sum()), wrong_to_correct=int((flip[mk] & (B.loc[mk, m+"_ok1"] == 0)).sum())))
T = pd.DataFrame(rows); P(T.to_string(index=False))
agg = T.groupby("peers_agreeing")[["n", "flips", "correct_to_wrong", "wrong_to_correct", "was_correct"]].sum(); agg["flip_rate_%"] = (100*agg.flips/agg.n).round(1)
P("\nall models:\n", agg.to_string())
lone = T[T.peers_agreeing == 0]
P(f"lone dissenters: n={lone.n.sum()}, of them correct in round 1: {lone.was_correct.sum()} ({100*lone.was_correct.sum()/lone.n.sum():.0f}%); flips={lone.flips.sum()} ({100*lone.flips.sum()/lone.n.sum():.1f}%), of which correct->wrong {lone.correct_to_wrong.sum()}, wrong->correct {lone.wrong_to_correct.sum()}")
# direction of flips
tot_yn = sum(((B[m+"_d1"]=="YES")&(B[m+"_d2"]=="NO")).sum() for m in M); tot_ny = sum(((B[m+"_d1"]=="NO")&(B[m+"_d2"]=="YES")).sum() for m in M)
P(f"flip direction: YES->NO {tot_yn}, NO->YES {tot_ny}")
# confidence change
for m in M:
    d = B[f"{m}_round2_confidence"] - B[f"{m}_round1_confidence"]; un = B.yes1.isin([0,3])
    P(f"  {NAMES[m]:14s} mean confidence change after seeing peers: unanimous rows {d[un].mean():+.3f}, split rows {d[~un].mean():+.3f}")

P("\n== 2. Majority voting under correlated errors: one-factor (Gaussian copula) model fitted to file A")
Ad = A.drop_duplicates("key")
p = float(np.mean([Ad[m+"_ok"].mean() for m in M])); 
E = pd.DataFrame({m: 1 - Ad[m+"_ok"] for m in M}); phi = float(E.corr().values[np.triu_indices(3, 1)].mean())
P(f"mean single accuracy p={p:.4f}; mean pairwise error correlation (phi)={phi:.3f}")
# tetrachoric rho such that P(both wrong) matches
q = 1 - p; both = float(np.mean([ (E[a] & E[b]).mean() for a, b in [(M[0], M[1]), (M[0], M[2]), (M[1], M[2])]]))
from scipy.stats import multivariate_normal as mvn
f = lambda r: mvn.cdf([N.ppf(q), N.ppf(q)], mean=[0, 0], cov=[[1, r], [r, 1]]) - both
rho = brentq(f, 0.01, 0.999)
P(f"P(both wrong) observed={both:.4f} (independent would be {q*q:.4f}); latent (tetrachoric) correlation rho={rho:.3f}")
def maj_acc(n, p, rho):
    # P(majority correct) for n exchangeable voters: integrate over common factor z
    t = N.ppf(p)
    def integrand(z):
        pc = N.cdf((t - np.sqrt(rho) * z) / np.sqrt(1 - rho))   # prob. each voter correct given z
        from scipy.stats import binom
        return binom.sf(n // 2, n, pc) * N.pdf(z)
    return integrate.quad(integrand, -8, 8)[0]
for n in (1, 3, 5, 9, 15, 31, 101, 1001):
    P(f"  N={n:5d}: independent={sum(__import__('scipy.stats').stats.binom.pmf(k, n, p) for k in range(n//2+1, n+1)):.4f}   correlated model={maj_acc(n, p, rho):.4f}")
lim = N.cdf(N.ppf(p) / np.sqrt(rho)); P(f"  N->infinity limit under the model: P(z < Phi^-1(p)/sqrt(rho)) = {lim:.4f}")
P(f"  observed 3-model majority (distinct rows) = {Ad.maj_ok.mean():.4f}; model prediction for N=3 = {maj_acc(3, p, rho):.4f}")
P(f"  share of rows where all three are wrong (irreducible by voting among these models) = {((E.sum(axis=1))==3).mean():.4f}; at least two wrong = {((E.sum(axis=1))>=2).mean():.4f}")
P("  what correlation would be needed: " + "; ".join(f"rho={r}: N=3 {maj_acc(3,p,r):.3f}, N=9 {maj_acc(9,p,r):.3f}, limit {N.cdf(N.ppf(p)/np.sqrt(r)):.3f}" for r in (0.2, 0.4, 0.6, 0.8)))
