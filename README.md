# ai_oracle

**Evaluating AI oracles for prediction markets.** VEGA x AI Talent Hub project.

An AI oracle is a program that, once a prediction market has closed, searches for evidence and decides which outcome the market should settle on. Published evaluations of such oracles are mostly retrospective. They run on old markets with evidence collected months after the events, and current models often remember the outcomes. This project builds an open evaluation engine on LangGraph and measures oracles on fresh Kalshi markets in real time, where neither the model nor the web can know the answer in advance.

## Status

The repository currently holds the groundwork, namely a pilot on past markets, an offline scaffold for the engine and the research plan. The engine itself is under development.

| Period | Milestone |
| --- | --- |
| By 1 Nov 2026 | Engine v0.1 (single agent, market collector, tracing, spend caps) and a preregistration commit |
| 3-30 Nov 2026 | Live runs on fresh Kalshi markets in crypto, finance, politics and geopolitics. Two LLMs in three architectures (single agent, leader with search workers, debate), plus Jev and a numeric-source resolver |
| 15 Dec 2026 | Proof-of-concept report |
| Jan 2027 | Backtest mode with point-in-time retrieval, checked against the live results on the same markets |

During the live run every decision and the hashes of the pages it relied on will be committed here before Kalshi settles the market, so anyone can verify that an answer predates the official outcome. The full plan, with hypotheses, budget and timeline, is in [`plan/proposal.md`](plan/proposal.md) (in Russian), and the cost estimate is reproduced by [`plan/budget.py`](plan/budget.py).

## Repository layout

| Path | Contents |
| --- | --- |
| `oracle_lab/` | Offline scaffold the engine grows from, with question keys and caches, evidence retrieval and the prompt content check, prompts, model clients (Anthropic, OpenRouter, Jev), metrics and the Kalshi export. Eight offline tests |
| `audit/analysis/` | Analyses of the published outputs of arXiv:2605.30802 (`t01`-`t09`), each with its saved output |
| `audit/exp/` | Pilot runs with Claude models on the paper's committed evidence packets. Harness, model answers, raw responses, analysis scripts and their outputs |
| `audit/agent/` | Answers and cleaned transcripts of a live-search agent |
| `audit/figs/`, `audit/independent_review/` | Figure scripts and an independent re-check of the audit numbers |
| `kota_audit/` | Reproducible audit of arXiv:2605.30802. A script checks the paper's code and published results at commit f76754b without model calls, and its output is saved next to it |
| `replication/` | Patch to the paper's code with regression tests, cache migration, plan and cost of confirmation runs |
| `plan/` | Research plan and the budget script |
| `notes/` | Literature notes and a check of the paper's references |
| `scripts/` | Restore the paper's repository into `audit/repo` (bash and PowerShell) |
| `third_party/` | Git bundle of the paper's repository at the audited commit |

## Quick start

Requires Python 3.11 or newer and git. None of these commands calls a model or uses the network. The paper's repository is restored from the bundled copy.

```bash
bash scripts/setup_kota_repo.sh          # Windows: powershell -File scripts/setup_kota_repo.ps1
pip install -r requirements.txt
mkdir -p work

python -I -X utf8 kota_audit/kota_audit_proofs.py --repo audit/repo --json work/findings.json > work/proofs_output.txt
diff --strip-trailing-cr kota_audit/findings.json work/findings.json

KOTA_REPO=audit/repo python -I -X utf8 oracle_lab/tests/test_offline.py
python plan/budget.py
cd audit && python -X utf8 analysis/t05_selective.py     # scripts under audit/ run from audit/
```

`-X utf8` matters on Windows, where output redirected to a file is otherwise encoded in the system code page.

## Reproducibility notes

- The saved outputs were computed on Linux. `kota_audit_proofs.py` and most analysis scripts reproduce them byte for byte on Windows as well.
- `audit/analysis/t03_errors_evidence.py` and `t07_misc.py` depend on which of several evidence files of the same question `audit/analysis/kota.py:packets()` picks, and that follows the file system order. Their saved outputs reproduce only in the original order.
- Several scripts overwrite their own `.pkl` and `.csv` outputs (`t03`, `t08`, `e2_analyze`, `e3_deb_analyze`, `e6_dates`, `e7_hindsight`). Run them on a copy of `audit/` if you want to keep the saved files.
- Pilot numbers are preliminary. Each configuration was run once through the Claude command-line client. A repeat through the API is prepared in `audit/exp/r1_api.py` and `replication/CONFIRM_RUNS.md` and has not been run.

## License

MIT, see [`LICENSE`](LICENSE). `third_party/` contains a copy of [Tkota10/Senior-Thesis-Multi-LLM-Resolution](https://github.com/Tkota10/Senior-Thesis-Multi-LLM-Resolution) under its own MIT license (Copyright (c) 2026 Tarun Kota).
