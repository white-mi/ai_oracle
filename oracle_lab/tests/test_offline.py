"""Offline tests: no network, no keys, no cost.
Run:  python tests/test_offline.py            (or pytest tests/)
Set KOTA_REPO=/path/to/clone of the audited paper's repository to also run the tests on its real data."""
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from oracle_lab import metrics as M  # noqa: E402
from oracle_lab.evidence import (EmptyEvidenceError, Packet, Source, check_content, drop_market_sites,  # noqa: E402
                                 merge_packets, query_templates, render_packet)
from oracle_lab.store import question_key  # noqa: E402

REPO = os.environ.get("KOTA_REPO")


def _packet():
    hl = ["The Senate passed the bill on July 1, 2025 by a 51-50 vote after the Vice President broke the tie. " * 3]
    src = [Source(url=f"https://news{i}.example.com/a", title=f"Title {i}", published="2025-07-02", highlights=hl) for i in range(5)]
    return Packet("MKT1", "Will the Senate pass the bill?", "Resolves YES if the Senate passes the bill before July 4, 2025.",
                  "2025-07-04T00:00:00Z", "baseline", ["q"], src)


def test_key_ignores_whitespace_and_has_no_row_number():
    a = question_key("MKT1", "Will  X happen?", "Rule\n text")
    b = question_key("MKT1", "Will X happen?", "Rule text")
    assert a == b
    assert question_key("MKT1", "Will X happen?", "Other rule") != a


def test_content_check_catches_titles_only_prompt():
    p = _packet()
    titles_only = "\n".join(f"--- Source {i} ---\nTitle: {s.title}\nURL: {s.url}\n" for i, s in enumerate(p.sources, 1))
    try:
        check_content(titles_only, p)
        raise AssertionError("titles-only prompt was not caught")
    except EmptyEvidenceError:
        pass
    check_content(render_packet(p), p)  # full prompt passes


def test_after_close_flag_and_market_filter():
    p = _packet()
    assert p.after_close_flags() == [False] * 5
    srcs = [Source(url="https://kalshi.com/markets/x"), Source(url="https://www.polymarket.com/event/y"), Source(url="https://apnews.com/z")]
    kept, dropped = drop_market_sites(srcs)
    assert dropped == 2 and kept[0].host() == "apnews.com"


def test_merge_dedups_by_url():
    p = _packet()
    q = _packet()
    q.sources = q.sources[:2] + [Source(url="https://other.example.com/b", highlights=["x" * 300])]
    m = merge_packets([p, q])
    assert len(m.sources) == 6 and m.variant == "merged"


def test_query_templates():
    t = query_templates("Will X happen?", "Resolves YES if X.", "2025-07-04T00:00:00Z")
    assert t["paper"] == 'What is the result of this question "Will X happen?"' and "July 2025" in t["result"]


def test_metrics():
    ok = np.array([1, 1, 1, 0, 1, 0])
    assert abs(M.aurc(np.arange(6)[::-1], ok) - np.mean(np.cumsum(1 - ok) / np.arange(1, 7))) < 1e-12
    assert M.aurc(np.array([1, 1, 1, 0, 1, 0]), ok) < M.aurc(np.arange(6)[::-1], ok)
    assert M.holm([0.01, 0.04, 0.03, 0.2]) == [True, False, False, False]
    assert M.mcnemar([1, 1, 0, 1], [0, 1, 1, 0])[:2] == (2, 1)
    rng = np.random.default_rng(1)
    a = rng.integers(0, 2, 2000)
    d, lo, hi, eq = M.tost_paired(a, a, np.arange(2000), B=500)
    assert d == 0 and eq


def test_copula_matches_report():
    # report: p=0.819, rho=0.845 -> majority of 3 = 83.0%, of 9 = 83.6%
    assert abs(M.gaussian_copula_majority(0.819, 0.845, 3) - 0.830) < 0.004
    assert abs(M.gaussian_copula_majority(0.819, 0.845, 9) - 0.836) < 0.004


def test_paper_data_if_available():
    if not REPO:
        print("  (skipped: set KOTA_REPO to run on the paper's data)")
        return
    import pandas as pd
    from oracle_lab.run import main
    from oracle_lab.store import PacketStore

    work = tempfile.mkdtemp()
    try:
        main(["import-paper", "--repo", REPO, "--work", work])
        d1 = pd.read_csv(os.path.join(work, "d1.csv"))
        store = PacketStore(os.path.join(work, "packets"))
        have = [k for k in d1.key if store.has(k, "paper_cache")]
        assert len(have) >= 400, len(have)
        # rule 1 on the real cache: rendered prompts pass, titles-only prompts fail
        bad = 0
        for k in have:
            p = store.load(k, "paper_cache")
            try:
                check_content(render_packet(p), p)
            except EmptyEvidenceError:
                bad += 1
        print(f"  content check on {len(have)} cached packets: {bad} fail with full text")
        assert bad / len(have) < 0.05
        # free end-to-end dry run with the fake model
        out = os.path.join(work, "fake.csv")
        main(["answer", "--data", os.path.join(work, "d1.csv"), "--work", work, "--variant", "paper_cache",
              "--condition", "abstain", "--model", "fake:x", "--out", out, "--limit", "60"])
        r = pd.read_csv(out)
        assert len(r) > 0 and r.error.isna().all()
        # the paper's composite signal reproduces AURC 0.051 on 1,153 distinct rows
        A = pd.read_csv(os.path.join(REPO, "results/Final_Architecture_A_results.csv"))
        A["k"] = A.question_id.astype(str) + "|" + A.question_text.str.split().str.join(" ") + "|" + A.resolution_criteria.str.split().str.join(" ")
        A = A.drop_duplicates("k")
        dec = {m: A[f"{m}_decision"].str.upper().str.strip() for m in ("gpt4o", "deepseek", "llama")}
        yes = sum((v == "YES").astype(int) for v in dec.values())
        maj = np.where(yes >= 2, "YES", "NO")
        ok = (maj == A.ground_truth.str.upper().str.strip()).astype(float).values
        comp = yes.isin([0, 3]).astype(float).values + np.mean([A[f"{m}_confidence"] for m in dec], axis=0)
        val = M.aurc(comp, ok)
        print(f"  paper composite AURC on {len(A)} rows: {val:.4f}")
        assert abs(val - 0.051) < 0.0015
    finally:
        shutil.rmtree(work)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
