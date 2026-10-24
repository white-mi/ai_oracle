"""Budget of an honest replication of arXiv:2605.30802 on D1 (1,188 distinct questions). Prices checked 11 Oct 2026."""
import json
S = "."
z = json.load(open(f"../replication/sizes.json")); T = z["mean_tok"]; O = z["out"]
N = 1188
PRICE = {  # USD per 1M tokens (input, output)
    "gpt4o":    (0.05, 0.40),   # GPT-5 nano, OpenAI list price (= OpenRouter)
    "deepseek": (0.25, 1.00),   # DeepSeek-V3-0324 weights via OpenRouter/SiliconFlow; Together serves it only as dedicated now
    "llama":    (1.04, 1.04),   # Llama-3.3-70B-Instruct-Turbo on Together, the paper's endpoint
}
EXA_PER_QUERY = (7 + 10 * 1) / 1000   # auto search up to 10 results $7/1k + highlights $1/1k pages
# arms: (name, input key, output key per model, calls per model per question)
ARMS = [
    ("A: голосование, промпт A, полный текст",            "A",        "A",  1),
    ("A: повтор того же прогона (шум)",                    "A",        "A",  1),
    ("B: раунд 1 с полным текстом",                        "B1",       "B1", 1),
    ("B: раунд 2 с полным текстом",                        "R2",       "B2", 1),
    ("B как в статье: раунд 1 только заголовки",           "B1_titles","B1", 1),
    ("B как в статье: раунд 2 только заголовки",           "R2_titles","B2", 1),
    ("Раунд 2 поверх ответов A (эффект дебатов при том же промпте)", "R2", "B2", 1),
    ("Без поиска (контроль утечки)",                       "closed",   None, 1),
]
def run(tok_mult, nano_reasoning):
    rows, tot = [], 0.0
    for name, ik, ok, c in ARMS:
        cost = 0.0; calls = 0
        for m, (pi, po) in PRICE.items():
            tin = T[ik] * tok_mult
            tout = (O[m][ok] if ok else 80) * tok_mult + (nano_reasoning if m == "gpt4o" else 0)
            cost += N * c * (tin * pi + tout * po) / 1e6; calls += N * c
        rows.append((name, calls, cost)); tot += cost
    return rows, tot
lo_rows, lo = run(0.85, 600); hi_rows, hi = run(1.15, 2500)
exa = N * EXA_PER_QUERY
print(f"{'arm':66s} {'calls':>7s} {'low $':>8s} {'high $':>8s}")
for (n, c, l), (_, _, h) in zip(lo_rows, hi_rows):
    print(f"{n:66s} {c:7d} {l:8.2f} {h:8.2f}")
calls = sum(c for _, c, _ in lo_rows)
print(f"{'LLM total':66s} {calls:7d} {lo:8.2f} {hi:8.2f}")
print(f"{'Exa: one packet per question, 1,188 queries x $0.017':66s} {N:7d} {exa:8.2f} {exa:8.2f}")
print(f"{'Reserve 25% (retries, pilot, debugging)':66s} {'':7s} {0.25*(lo+exa):8.2f} {0.25*(hi+exa):8.2f}")
print(f"{'TOTAL':66s} {'':7s} {1.25*(lo+exa):8.2f} {1.25*(hi+exa):8.2f}")
# share by model at high
for m in PRICE:
    pi, po = PRICE[m]; s = 0
    for name, ik, ok, c in ARMS:
        s += N * c * (T[ik] * 1.15 * pi + ((O[m][ok] if ok else 80) * 1.15 + (2500 if m == "gpt4o" else 0)) * po) / 1e6
    print(f"  high scenario, {m}: {s:.2f}")
# minimal core: A + B fixed + closed book, no repeat, no titles arm, no extra round 2
core = [a for a in ARMS if a[0].startswith(("A: голосование", "B: раунд", "Без поиска"))]
ARMS[:] = core; l2, lo2 = run(0.85, 600); h2, hi2 = run(1.15, 2500)
print(f"minimal core (A, B fixed, closed book): calls {sum(c for _, c, _ in l2)}, LLM {lo2:.2f}-{hi2:.2f}, with Exa and 25% reserve {1.25*(lo2+exa):.2f}-{1.25*(hi2+exa):.2f}")
