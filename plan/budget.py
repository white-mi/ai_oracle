"""Token, search and tracing budget of the live proof of concept, 3-30 November 2026, plus the power of the
preregistered tests. No model calls, no network. Run from the repository root:
    python plan/budget.py > plan/budget.out

Prices checked on 24 October 2026:
  claude-haiku-5-5   $0.10 / $0.50 per 1M input / output tokens, prompts up to 100K tokens (Claude API reference, cached 6 Oct 2026)
  gpt-5.4-mini       $0.75 / $4.50 per 1M input / output tokens (developers.openai.com/api/docs/pricing)
  Exa search         $7 per 1k requests with up to 10 results, plus $1 per 1k pages of highlights (exa.ai/pricing)
  Jev                about $0.04 per 1M input tokens, output free (open reviews cited in the proposal of 22 Oct 2026)
  LangSmith          free plan with 5,000 base traces a month, 14-day retention (langchain.com/pricing)
Token sizes: one search with 10 highlighted sources adds about as many tokens as the source block of a committed
packet of the audited paper, i.e. prompt A minus the closed-book prompt (replication/sizes.json).
"""
import json
from scipy.stats import beta, binom

SZ = json.load(open("replication/sizes.json"))
E = SZ["mean_tok"]["A"] - SZ["mean_tok"]["closed"]  # evidence tokens per search
CLOSED = SZ["mean_tok"]["closed"]
B0 = 1500         # system prompt, question, rules and tool schema (assumption)
S = 4             # searches per market per architecture, the same limit for all three
OUT_CALL = 600    # output tokens of an intermediate call, thinking included (assumption)
OUT_FINAL = 1000  # output tokens of a decision call, thinking included (assumption)
PRICE = {"claude-haiku-5-5": (0.10, 0.50), "gpt-5.4-mini": (0.75, 4.50)}
EXA = (7 + 10 * 1) / 1000
JEV_IN = 0.04
WAIT = 0.30       # share of runs answered WAIT and repeated once after 3 days (assumption of the 22 Oct proposal)
RESERVE = 0.10
CAP = 500.0


def single():
    """One agent searches S times in a loop; every call re-reads the growing context."""
    return (S + 1) * B0 + E * S * (S + 1) / 2, S * OUT_CALL + OUT_FINAL


def leader(workers=2):
    """The leader plans, each worker searches S/workers times, the leader decides from the workers' summaries."""
    s = S // workers
    w_in, w_out = (s + 1) * B0 + E * s * (s + 1) / 2, s * OUT_CALL + 800
    return workers * w_in + B0 + (B0 + workers * 800), workers * w_out + OUT_CALL + OUT_FINAL


def debate(agents=3):
    """S searches build one packet; each agent answers, then answers again after reading all first-round answers."""
    packet = B0 + S * E
    return agents * packet + agents * (packet + agents * OUT_FINAL), 2 * agents * OUT_FINAL


ARCH = {"single agent": single, "leader, 2 workers": leader, "debate, 3 agents, 2 rounds": debate}


def per_market(out_mult, archs, jev=True):
    """Cost of one market: the LLM oracles in `archs` on both models with their searches and WAIT repeats,
    one closed-book control per model, and Jev on the single agent's packet."""
    runs = control = 0.0
    for pi, po in PRICE.values():
        for a in archs:
            tin, tout = ARCH[a]()
            runs += (tin * pi + tout * out_mult * po) / 1e6 + S * EXA
        control += ((CLOSED + B0) * pi + OUT_FINAL * out_mult * po) / 1e6
    return runs * (1 + WAIT) + control + ((B0 + S * E) * JEV_IN / 1e6 if jev else 0.0)


print("== Tokens per oracle run (both models use the same token model)")
for a, f in ARCH.items():
    tin, tout = f()
    print(f"  {a:28s} input {tin:9,.0f}  output {tout:6,.0f} (x3 in the high scenario)  searches {S}")
print(f"  evidence tokens per search E = {E:,.0f}, base prompt B0 = {B0:,}, closed-book prompt {CLOSED:,.0f}")

print("\n== Cost of one run, USD (low = outputs as assumed, high = outputs x3 for longer reasoning)")
for m, (pi, po) in PRICE.items():
    for a, f in ARCH.items():
        tin, tout = f()
        lo, hi = (tin * pi + tout * po) / 1e6, (tin * pi + 3 * tout * po) / 1e6
        print(f"  {m:17s} {a:28s} LLM {lo:.4f}-{hi:.4f}  + search {S * EXA:.3f}")
print(f"  Jev on the single agent's packet: {(B0 + S * E) * JEV_IN / 1e6:.4f}")

ALL, SOLO = list(ARCH), ["single agent"]
print("\n== Whole run: single agent on both models from 3 Nov (7 days), all oracles from 10 Nov (21 days),"
      f" WAIT repeats {WAIT:.0%}, reserve {RESERVE:.0%}")
print(f"  {'markets/day':>11s} {'markets':>8s} {'low $':>8s} {'high $':>8s}  within {CAP:.0f} $ in the high scenario")
for r in (8, 10, 12, 14, 16, 20):
    n1, n2 = 7 * r, 21 * r
    tot = {k: (n1 * per_market(k, SOLO, jev=False) + n2 * per_market(k, ALL)) * (1 + RESERVE) for k in (1, 3)}
    print(f"  {r:11d} {n1 + n2:8d} {tot[1]:8.2f} {tot[3]:8.2f}  {'yes' if tot[3] <= CAP else 'no'}")
for k, lab in ((1, "low"), (3, "high")):
    pm1, pm2 = per_market(k, SOLO, jev=False), per_market(k, ALL)
    print(f"  per market, {lab}: single agent only {pm1:.3f}, all oracles {pm2:.3f}")

print("\n== Share of search in the cost of one market with all oracles")
for k, lab in ((1, "low"), (3, "high")):
    srch = 2 * len(ALL) * S * EXA * (1 + WAIT)
    print(f"  {lab}: search {srch:.3f} of {per_market(k, ALL):.3f} ({srch / per_market(k, ALL):.0%})")

print("\n== LangSmith traces a month (free plan: 5,000)")
for r in (12, 16, 20):
    llm_runs = (7 * r * 2 + 21 * r * 6) * (1 + WAIT)
    with_rest = llm_runs + 21 * r * 1 + 28 * r * 2  # plus Jev and closed-book controls
    print(f"  {r} markets/day: LLM oracle runs {llm_runs:,.0f}; tracing Jev and controls too {with_rest:,.0f}")

print("\n== Power of the architecture test (paired by market and model, both models pooled)")
print("  one-sided exact sign test on discordant pairs at alpha 0.05/3 (first Holm step) and 0.05;"
      " true improvement 5 pp; discordance = share of pairs where exactly one of the two oracles closes wrongly")
for r in (10, 12, 14):
    pairs = 21 * r * 2
    for d in (0.10, 0.15, 0.20):
        nd = round(d * pairs)
        p01 = (d + 0.05) / (2 * d)  # share of discordant pairs where the architecture is right and the single agent wrong
        out = []
        for a in (0.05 / 3, 0.05):
            kcrit = next(k for k in range(nd + 1) if binom.sf(k - 1, nd, 0.5) <= a)
            out.append(binom.sf(kcrit - 1, nd, p01))
        print(f"  {r} markets/day: {pairs} pairs, discordance {d:.0%} ({nd} pairs): power {out[0]:.2f} at 0.05/3, {out[1]:.2f} at 0.05")

print("\n== P4: most errors among automatically closed markets that keep the one-sided 95% upper bound at or below 5%")
for n in (100, 150, 200, 250, 300):
    k = max(k for k in range(n) if beta.ppf(0.95, k + 1, n - k) <= 0.05)
    print(f"  {n} closed automatically: at most {k} errors ({k / n:.1%}), upper bound {beta.ppf(0.95, k + 1, n - k):.3f}")
