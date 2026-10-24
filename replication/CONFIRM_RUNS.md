# Confirmation runs (prepared, not yet run)

Two runs confirm the audit with fresh model calls. Both use only the committed evidence packets, so no new
retrieval happens and the evidence is the same as in the paper. Neither has been run yet.

## R1. The paper's models and code, with the formatter fixed

Needs `OPENAI_API_KEY` and `TOGETHER_API_KEY`, each with a spend limit. No Exa key is needed.

```bash
git clone https://github.com/Tkota10/Senior-Thesis-Multi-LLM-Resolution kota && cd kota
git checkout f76754b5bcff1a1d7a50c2f1073a4b6d44bd985c
git apply ../replication_fixes.patch
python3 -m pytest -q                         # 35 passed, 4 skipped, 1 failed (it fails before the patch too)
python3 -I ../migrate_cache.py --repo .      # 415 questions get A's packet under the new cache key
python3 scripts/test_architecture_a.py -n 1189 --cache-only --output results/R1_A.csv
python3 scripts/test_architecture_b.py -n 1189 --cache-only --evidence-render titles --output results/R1_B_titles.csv
python3 scripts/test_architecture_b.py -n 1189 --cache-only --evidence-render text   --output results/R1_B_text.csv
python3 scripts/analyze_architecture_results.py --arch-a results/R1_A.csv --arch-b results/R1_B_text.csv --output-dir results/R1_text
python3 scripts/analyze_architecture_results.py --arch-a results/R1_A.csv --arch-b results/R1_B_titles.csv --output-dir results/R1_titles
```

With `--cache-only`, questions without a committed packet are skipped, so the runs cover 415 questions.
The three runs make about 6,200 calls. Scaling `replication/budget.py` to 415 questions gives about
12-18 USD, or 15-22 USD with a 25% reserve.

What it tests. `R1_B_text` against `R1_B_titles` measures the effect of the formatter on the same packets and
models. `R1_B_text` against `R1_A` measures what is left of the A-B gap. The published B read this same
packet for only 100 of the 415 questions (the other 315 have different cache keys, F5), so the comparison of
`R1_B_titles` with the published B decisions is reported on those 100 separately; on them it should agree up
to run-to-run noise. A closed-book run of the three models (prompt A with the source list removed) is still
needed to bound what they answer from memory.

## R2. Mechanism check with one model through the Anthropic API

Needs `ANTHROPIC_API_KEY` with a spend limit. Script: `audit/exp/r1_api.py`, run from `audit/` with
`./repo` a clone at f76754b.

```bash
python3 exp/r1_api.py build                    # 430 questions, 3,010 calls, cost bound 13.37 USD, expected about 10 USD
python3 exp/r1_api.py send --cap-usd 15        # refuses to send if the bound exceeds the cap
python3 exp/r1_api.py collect
python3 exp/r1_api.py analyze
```

Model `claude-haiku-4-5-20251001`, temperature 0, Message Batches (half price), output forced through a tool
with the paper's schema. Conditions on the same packet: prompt A with text (twice), prompt B with titles
only (as published), prompt B with text (twice), prompt A with no sources, prompt A without market pages.

Leakage controls in `analyze`:

1. the closed-book run, and every contrast again on the questions the model gets wrong from memory;
2. every contrast again on questions resolved after the model's training-data cutoff (July 2025);
3. the contrast of A with and without prediction-market and betting pages, since 16 packets contain
   settled market pages that state "Final outcome";
4. the served model is read from every response and must equal the requested one.

The pilot of R2 was run earlier through the Claude command-line client (`exp/e1_run.py`, results in
`exp/e1_analyze.out`). R2 repeats it through the API under fixed settings so that the numbers can be cited.
