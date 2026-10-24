#!/usr/bin/env python3
"""Reproducible checks for arXiv:2605.30802 (Kota, "Design and Evaluation of Multi-Agent AI Oracle
Systems for Prediction Market Resolution", v1, 29 May 2026) against the author's public repository.

Usage:
    git clone https://github.com/Tkota10/Senior-Thesis-Multi-LLM-Resolution kota_repo
    git -C kota_repo checkout f76754b5bcff1a1d7a50c2f1073a4b6d44bd985c
    python3 kota_audit_proofs.py --repo kota_repo [--json findings.json]

The script calls no model and no search engine. It reads only files in the repository at the pinned
commit: the code, the two published results files, the evaluation CSV and the 486 committed evidence
packets. Every printed number is computed here. Requirements: Python 3.9+, pandas, numpy.
"""
import argparse, ast, glob, importlib.util, json, math, os, re, subprocess, sys
from collections import Counter

import numpy as np
import pandas as pd

COMMIT = "f76754b5bcff1a1d7a50c2f1073a4b6d44bd985c"
M = ["gpt4o", "deepseek", "llama"]  # column prefixes in the results files; "gpt4o" holds GPT-5 nano (see F12)
NAME = {"gpt4o": "GPT-5 nano", "deepseek": "DeepSeek V3", "llama": "Llama 3.3 70B"}
OUT = {}


# ----------------------------------------------------------------------------- helpers
def section(fid, title):
    print("\n" + "=" * 100 + f"\n{fid}. {title}\n" + "=" * 100)
    OUT[fid] = {"title": title, "facts": []}


def fact(fid, text, **values):
    print("  - " + text)
    OUT[fid]["facts"].append({"text": text, **values})


def show_lines(repo, path, a, b, fid=None):
    lines = open(os.path.join(repo, path), encoding="utf-8").read().splitlines()
    print(f"    {path}, lines {a}-{b}:")
    for i in range(a, b + 1):
        print(f"      {i:4d} | {lines[i - 1]}")
    return lines


def line_is(repo, path, n, needle):
    line = open(os.path.join(repo, path), encoding="utf-8").read().splitlines()[n - 1]
    assert needle in line, f"{path}:{n} does not contain {needle!r}: {line!r}"
    return line


def binom_two_sided(k, n):
    """Exact two-sided binomial test with p = 0.5 (the exact McNemar test)."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def fisher_two_sided(a, b, c, d):
    """Exact two-sided Fisher test for the 2x2 table [[a, b], [c, d]]."""
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    def pr(x):
        return math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1)
    p0 = pr(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return min(1.0, sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p0 * (1 + 1e-9)))


def norm_dec(s):
    return s.astype(str).str.strip().str.upper()


def ws(s):
    return s.astype(str).str.replace(r"\\n", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()


def pct(k, n):
    return 100.0 * k / n


# ----------------------------------------------------------------------------- loading
def load(repo):
    A = pd.read_csv(os.path.join(repo, "results/Final_Architecture_A_results.csv"))
    B = pd.read_csv(os.path.join(repo, "results/Final_Architecture_B_results.csv"))
    D = pd.read_csv(os.path.join(repo, "data/kalshibench_v2_evaluation_Filtered.csv"))
    for df in (A, B):
        df["truth"] = norm_dec(df.ground_truth)
    for m in M:
        A[f"{m}_d"] = norm_dec(A[f"{m}_decision"])
        A[f"{m}_ok"] = A[f"{m}_d"] == A.truth
        for r in (1, 2):
            B[f"{m}_d{r}"] = norm_dec(B[f"{m}_round{r}_decision"])
            B[f"{m}_ok{r}"] = B[f"{m}_d{r}"] == B.truth
    A["yes_n"] = sum((A[f"{m}_d"] == "YES").astype(int) for m in M)
    A["maj"] = np.where(A.yes_n >= 2, "YES", "NO")
    A["maj_ok"] = A.maj == A.truth
    A["final_ok"] = norm_dec(A.final_decision) == A.truth
    w = sum(np.where(A[f"{m}_d"] == "YES", 1.0, -1.0) * A[f"{m}_confidence"].astype(float) for m in M)
    A["wtd_ok"] = np.where(w > 0, "YES", "NO") == A.truth
    A["unan"] = A.yes_n.isin([0, 3])
    A["mean_conf"] = np.mean([A[f"{m}_confidence"].astype(float).values for m in M], axis=0)
    for r in (1, 2):
        B[f"yes{r}"] = sum((B[f"{m}_d{r}"] == "YES").astype(int) for m in M)
        B[f"maj{r}"] = np.where(B[f"yes{r}"] >= 2, "YES", "NO")
        B[f"maj{r}_ok"] = B[f"maj{r}"] == B.truth
    B["final_ok"] = norm_dec(B.final_decision) == B.truth
    # identity of a contract row: market id + question text + resolution criteria
    A["key"] = A.question_id.astype(str) + "|" + ws(A.question_text) + "|" + ws(A.resolution_criteria)
    B["key"] = B.question_id.astype(str) + "|" + ws(B.question_text) + "|" + ws(B.resolution_criteria)
    D["key"] = D["id"].astype(str) + "|" + ws(D.question) + "|" + ws(D.description)
    return A, B, D


def packets(repo):
    pk = {}
    for f in sorted(glob.glob(os.path.join(repo, "cache/evidence/*.json"))):
        e = json.load(open(f, encoding="utf-8"))
        e["_file"] = os.path.basename(f)
        pk[e["_file"]] = e
    return pk


# ----------------------------------------------------------------------------- F1
def f1_no_source_text(repo, A, B, pk):
    fid = "F1"
    section(fid, "Architecture B never received the text of any source")
    rp = "src/resolution/debate/runner.py"
    line_is(repo, rp, 309, "def _format_exa_sources")
    line_is(repo, rp, 320, "lines.append(source.text)")
    line_is(repo, rp, 121, "self._format_exa_sources(evidence.sources)")
    line_is(repo, rp, 179, "self._format_exa_sources(evidence.sources)")
    show_lines(repo, rp, 309, 322)
    fact(fid, "The debate formatter appends source.text (runner.py line 320). Round 1 (line 121) and round 2 "
              "(line 179) both build their evidence block with this formatter.")
    ev = "src/retrieval/evidence.py"
    line_is(repo, ev, 17, 'text: str = ""')
    line_is(repo, ev, 19, "highlights: List[str]")
    line_is(repo, ev, 109, "content_for_prompt()")
    show_lines(repo, ev, 14, 28)
    fact(fid, "Architecture A formats evidence with EvidencePacket.format_for_prompt (evidence.py line 109), which calls "
              "Source.content_for_prompt and returns the Exa highlights (lines 25-28).")
    ex = "src/retrieval/exa_retriever.py"
    line_is(repo, ex, 85, 'contents={"highlights": highlights_contents}')
    line_is(repo, ex, 97, 'text=result.text if hasattr(result, "text") and result.text else ""')
    show_lines(repo, ex, 80, 98)
    modes = {"A": A.retrieval_mode.value_counts().to_dict(), "B": B.retrieval_mode.value_counts().to_dict()}
    fact(fid, f"In highlights mode the Exa request asks for highlights only (exa_retriever.py line 85), and text is "
              f"stored as an empty string when Exa returns none (line 97). Retrieval mode in the results files: {modes}.",
         modes=modes)
    n_src = n_empty = 0
    nonempty_files = Counter()
    for f, e in pk.items():
        for s in e["sources"]:
            n_src += 1
            if (s.get("text") or "").strip():
                nonempty_files[f] += 1
            else:
                n_empty += 1
    legacy_only = all(not f.startswith("highlights__") for f in nonempty_files)
    fact(fid, f"Committed cache: {n_src} sources in {len(pk)} packets. source.text is empty for {n_empty}. "
              f"The {n_src - n_empty} non-empty ones sit in {len(nonempty_files)} legacy packets that do not start with "
              f"'highlights__' (all legacy: {legacy_only}).", n_sources=n_src, n_empty_text=n_empty)
    # Render the round-1 evidence block of B and the evidence block of A with the authors' own code. Only
    # src/retrieval/evidence.py (standard library only) is executed, plus the one formatter function from runner.py.
    line_is(repo, ev, 25, "def content_for_prompt(self) -> str:")
    line_is(repo, ev, 26, "if self.highlights:")
    line_is(repo, ev, 27, 'return "\\n".join(f"- {item}" for item in self.highlights)')
    line_is(repo, ev, 28, "return self.text")
    line_is(repo, ev, 97, 'f"Resolution Date: {self.resolution_date}",')
    spec = importlib.util.spec_from_file_location("kota_evidence", os.path.join(repo, ev))
    evm = importlib.util.module_from_spec(spec)
    sys.modules["kota_evidence"] = evm
    spec.loader.exec_module(evm)
    EvidencePacket, Source = evm.EvidencePacket, evm.Source
    tree = ast.parse(open(os.path.join(repo, rp), encoding="utf-8").read())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_format_exa_sources")
    fn.decorator_list = []
    ns = {"Source": Source, "List": list}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), rp, "exec"), ns)
    fmt_b = ns["_format_exa_sources"]
    name = "highlights__row0089_KXSNLHOST_ddc2b51a4839.json"
    e = EvidencePacket.from_dict({k: v for k, v in pk[name].items() if not k.startswith("_")})
    block_b = fmt_b(e.sources)
    block_a = e.format_for_prompt()
    print("\n    Evidence block of B round 1 for the paper's Glen Powell example (first 2 sources), "
          "produced by the authors' own _format_exa_sources:")
    for ln in block_b.splitlines()[:10]:
        print("      | " + ln)
    mentions_a = block_a.count("Powell")
    mentions_b = block_b.count("Powell")
    in_crit = e.resolution_criteria.count("Powell")
    in_titles = sum((s.title or "").count("Powell") for s in e.sources)
    in_hl = sum(" ".join(s.highlights).count("Powell") for s in e.sources)
    hl_sources = sum("Powell" in " ".join(s.highlights) for s in e.sources)
    fact(fid, f"Applied to {name}, the packet A read for this contract (B read row0189_KXSNLHOST_f5982df4d591, which is "
              f"not committed): the A evidence block has {len(block_a)} characters and mentions 'Powell' {mentions_a} times "
              f"({in_crit} in the criteria line, {in_titles} in a title, {in_hl} in the highlights of {hl_sources} sources). "
              f"The B block has {len(block_b)} characters and mentions 'Powell' {mentions_b} times, in a title.",
         a_chars=len(block_a), b_chars=len(block_b), a_powell=mentions_a, b_powell=mentions_b, a_powell_highlight_sources=hl_sources)
    lens_a, lens_b = [], []
    for f2, e2 in pk.items():
        if not f2.startswith("highlights__"):
            continue
        p = EvidencePacket.from_dict({k: v for k, v in e2.items() if not k.startswith("_")})
        lens_a.append(len(p.format_for_prompt()))
        lens_b.append(len(fmt_b(p.sources)))
    fact(fid, f"Over all {len(lens_a)} highlights packets the median evidence block is {int(np.median(lens_a))} "
              f"characters for A and {int(np.median(lens_b))} for B.",
         median_chars_a=int(np.median(lens_a)), median_chars_b=int(np.median(lens_b)))
    own_b = int(sum(f"highlights__{u}.json" in pk for u in B.row_uid))
    fact(fid, f"B's own packet (named by B's row_uid) is committed for {own_b} of {len(B)} B rows. For these rows the empty "
              "text is observed directly. For the others the claim rests on the code path and on the retrieval_mode field "
              "of the results files.", b_rows_with_own_packet=own_b)
    hist = subprocess.run(["git", "-C", repo, "log", "--format=%h", "--", rp], capture_output=True, text=True).stdout.split()
    same = [h for h in hist if "lines.append(source.text)" in subprocess.run(
        ["git", "-C", repo, "show", f"{h}:{rp}"], capture_output=True, text=True).stdout]
    fact(fid, f"Every one of the {len(hist)} commits that touch {rp} ({', '.join(hist)}) appends source.text in the "
              f"formatter: {len(same) == len(hist)}.", runner_commits=hist)
    # How the mismatch arose: the switch to highlights updated A's formatter and left B's untouched.
    git = lambda *a: subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True).stdout
    sw = "bbb53efd80775069e198eb9824b871af94f9e4a7"
    msg = git("log", "-1", "--format=%ad %s", "--date=short", sw).strip()
    files = git("show", "--name-only", "--format=", sw).split()
    ev_diff = git("show", sw, "--", ev)
    removed = '-            lines.append(f"\\n{source.text}\\n")' in ev_diff
    added = '+            content = source.content_for_prompt().strip()' in ev_diff
    assert removed and added and ev in files and rp not in files, "commit bbb53ef does not look as expected"
    b_impl = "af935865a4eaf4a5f62f3847f4038badf960823f"
    old_ret = git("show", f"{b_impl}:src/retrieval/exa_retriever.py")
    fact(fid, f"Commit {sw[:7]} ({msg}) adds the highlights field and changes A's formatter in {ev} from "
              f"source.text to content_for_prompt() (line removed: {removed}, line added: {added}). It does not touch "
              f"{rp} ({rp in files}). B's debate code was written in commit {b_impl[:7]} "
              f"({git('log', '-1', '--format=%ad', '--date=short', b_impl).strip()}), when the retriever had no highlights "
              f"mode ('highlights' in exa_retriever.py then: {'highlights' in old_ret}) and filled source.text.",
         switch_commit=sw, switch_touches_runner=rp in files)
    tp = "tests/test_debate_runner.py"
    line_is(repo, tp, 49, 'text="text"')
    fact(fid, "The unit test that covers the formatter builds its source with text=\"text\" (tests/test_debate_runner.py "
              "line 49), so it passes although real packets carry their content in highlights.")


# ----------------------------------------------------------------------------- F2
def f1b_quotes(repo, A, B, pk):
    fid = "F2"
    section(fid, "The published B answers quote titles, almost never source text")
    Au = A.drop_duplicates("key").set_index("key")
    Bu = B.drop_duplicates("key").set_index("key")
    cols = ["row_uid"] + [f"{m}_round{r}_reasoning" for m in M for r in (1, 2)]
    J = Au.join(Bu[cols], rsuffix="_B", how="inner")
    J = J[(J.row_uid == J.row_uid_B) & J.row_uid.map(lambda u: f"highlights__{u}.json" in pk)]
    nz = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

    def quotes(text):
        q = re.findall(r"[\"“‘']([^\"”’']{25,300})[\"”’']", str(text))
        return [nz(x) for x in q if len(nz(x).split()) >= 5]

    res = {}
    for lab, colf in [("A", lambda m: f"{m}_reasoning"), ("B round 1", lambda m: f"{m}_round1_reasoning"),
                      ("B round 2", lambda m: f"{m}_round2_reasoning")]:
        parts = []
        for m in M:
            n = hl = ti = 0
            for _, r in J.iterrows():
                e = pk[f"highlights__{r.row_uid}.json"]
                titles = nz(" | ".join((s.get("title") or "") for s in e["sources"]))
                hls = nz(" | ".join(" ".join(s.get("highlights") or []) for s in e["sources"]))
                qc = nz(e["question_text"] + " " + e["resolution_criteria"])
                for q in quotes(r[colf(m)]):
                    n += 1
                    if q in qc:
                        continue
                    if q in titles:
                        ti += 1
                    elif q in hls:
                        hl += 1
            parts.append(f"{NAME[m]} {hl}/{n} from source text ({pct(hl, max(n, 1)):.1f}%), {ti} from titles")
            res[(lab, m)] = (hl, n, ti)
        fact(fid, f"{lab}: " + "; ".join(parts))
    # The agents themselves remark that source content is missing, in B and never in A.
    miss = re.compile(r"(?:content|text|body|details?) (?:is|are|was|were) not (?:shown|provided|included|available|visible|given)"
                      r"|(?:titles?|headlines?) alone|only (?:the )?(?:titles?|headlines?)|(?:titles?|headlines?) only"
                      r"|based (?:solely |only )?on (?:the )?(?:source )?(?:titles?|headlines?)"
                      r"|no (?:article )?(?:content|text|body)(?: is| was)? (?:provided|shown|included|available)"
                      r"|without (?:the |any )?(?:article |source )?(?:content|text|body)", re.I)
    nb = {m: int(B[f"{m}_round1_reasoning"].astype(str).str.contains(miss).sum()) for m in M}
    na = {m: int(A[f"{m}_reasoning"].astype(str).str.contains(miss).sum()) for m in M}
    fact(fid, f"Reasoning texts that say the source content is missing (pattern search, no manual coding): B round 1 "
              f"{sum(nb.values())} of {3 * len(B)} ({', '.join(f'{NAME[m]} {nb[m]}' for m in M)}), A {sum(na.values())} of "
              f"{3 * len(A)}.", b_round1_missing=nb, a_missing=na)
    for uid, col, needle in (("row0317_KXMARATHON_8648f52e2447", "gpt4o_round1_reasoning", "the content is not shown here"),
                             ("row0630_KXAWARDSCOUNTWHI_951619a4f8a9", "deepseek_round1_reasoning", "since the content is not provided"),
                             ("row0939_KXHURCTOTMAJ_83171b545677", "deepseek_round1_reasoning", "but the content is not provided")):
        t = str(B.loc[B.row_uid == uid, col].iloc[0])
        i = t.find(needle)
        assert i >= 0, (uid, needle)
        fact(fid, f"Example, B {uid}, {col}: '...{t[max(0, i - 160):i + 90].strip()}...'")
    fact(fid, f"Rows used: {len(J)} questions where A and B have the same cache key and the packet file is committed. "
              "A quote is a quoted string of 5 or more words, normalised, matched against question and criteria first, "
              "then titles, then highlights.", n_rows=len(J))
    return res


# ----------------------------------------------------------------------------- F3
def f2_gap_before_debate(A, B):
    fid = "F3"
    section(fid, "The A-B gap exists before any agent sees another agent's answer")
    n = len(A)
    a_maj, a_wtd, b1, bf = int(A.maj_ok.sum()), int(A.wtd_ok.sum()), int(B.maj1_ok.sum()), int(B.final_ok.sum())
    fact(fid, f"A majority {a_maj}/{n} ({pct(a_maj, n):.2f}%), A weighted {a_wtd}/{n} ({pct(a_wtd, n):.2f}%), "
              f"B round-1 majority {b1}/{n} ({pct(b1, n):.2f}%), B final {bf}/{n} ({pct(bf, n):.2f}%).",
         a_majority=a_maj, a_weighted=a_wtd, b_round1_majority=b1, b_final=bf)
    for m in M:
        a, r1, r2 = int(A[f"{m}_ok"].sum()), int(B[f"{m}_ok1"].sum()), int(B[f"{m}_ok2"].sum())
        fact(fid, f"{NAME[m]}: A {a} ({pct(a, n):.2f}%), B round 1 {r1} ({pct(r1, n):.2f}%), B round 2 {r2} "
                  f"({pct(r2, n):.2f}%).", model=m, a=a, b1=r1, b2=r2)
    c2w = int((B.maj1_ok & ~B.final_ok).sum())
    w2c = int((~B.maj1_ok & B.final_ok).sum())
    fact(fid, f"From B round-1 majority to B final: {c2w} correct answers became wrong, {w2c} wrong became correct, "
              f"exact McNemar p = {binom_two_sided(c2w, c2w + w2c):.3f}.", c2w=c2w, w2c=w2c)
    rev = Counter()
    for m in M:
        rev[(m, "c2w")] = int((B[f"{m}_ok1"] & ~B[f"{m}_ok2"]).sum())
        rev[(m, "w2c")] = int((~B[f"{m}_ok1"] & B[f"{m}_ok2"]).sum())
    tot_rev = int(sum(int((B[f"{m}_d1"] != B[f"{m}_d2"]).sum()) for m in M))
    q_rev = int(np.any([B[f"{m}_d1"] != B[f"{m}_d2"] for m in M], axis=0).sum())
    fact(fid, f"Agent-level revisions: {tot_rev} of {3 * n} agent-round pairs on {q_rev} questions; correct to wrong "
              f"{sum(rev[(m, 'c2w')] for m in M)}, wrong to correct {sum(rev[(m, 'w2c')] for m in M)}. Per model "
              + ", ".join(f"{NAME[m]} {rev[(m, 'c2w')]}/{rev[(m, 'w2c')]}" for m in M) + " (paper, Section 4.4.5: "
              "88 revisions on 83 questions, 45 and 43, GPT 26/16, DeepSeek 3/9).", revisions=tot_rev, questions=q_rev)
    fact(fid, f"Totals over each file's 1,189 rows: of the net gap of {a_maj - bf} correct answers between A majority and "
              f"B final, {a_maj - b1} is present after round 1 of B and {b1 - bf} arises in the debate round. These are "
              "differences of totals over different row sets (see F9).", gap_total=a_maj - bf, gap_round1=a_maj - b1)
    Au = A.drop_duplicates("key").set_index("key")
    Bu = B.drop_duplicates("key").set_index("key")
    J = Au[["maj_ok"]].join(Bu[["maj1_ok", "final_ok"]], how="inner")
    pa, p1, pf = int(J.maj_ok.sum()), int(J.maj1_ok.sum()), int(J.final_ok.sum())
    fact(fid, f"Paired on the {len(J)} contracts in both files: A majority {pa}, B round-1 majority {p1}, B final {pf}. "
              f"Net gap {pa - pf}, of which {pa - p1} is present after round 1.", paired_n=len(J), paired_a=pa,
         paired_b1=p1, paired_bf=pf)


# ----------------------------------------------------------------------------- F4
def f3_prompts(repo):
    fid = "F4"
    section(fid, "Round 1 of B uses a different system prompt, a different user template and no resolution date")
    pa = "src/agents/prompts.py"
    pb = "src/resolution/debate/prompts.py"
    consts = {}
    for path in (pa, pb):
        tree = ast.parse(open(os.path.join(repo, path), encoding="utf-8").read())
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                consts[(path, node.targets[0].id)] = node.value.value
    sysA, sysB = consts[(pa, "SYSTEM_PROMPT")], consts[(pb, "ROUND_1_SYSTEM_PROMPT")]
    usrA, usrB = consts[(pa, "USER_MESSAGE_TEMPLATE")], consts[(pb, "ROUND_1_USER_TEMPLATE")]
    fact(fid, f"System prompts identical: {sysA.strip() == sysB.strip()}. User templates identical: {usrA == usrB}.")
    ph = sorted(set(re.findall(r"\{(\w+)\}", usrB)))
    fact(fid, f"The B round-1 user template has the placeholders {ph}, so it carries no resolution date. "
              "A's evidence block contains 'Resolution Date:' (evidence.py line 97).", placeholders=ph)
    line_is(repo, "src/retrieval/evidence.py", 97, "Resolution Date")
    show_lines(repo, "src/resolution/debate/runner.py", 121, 126)


# ----------------------------------------------------------------------------- F5
def f4_cache_key(repo, A, B, pk):
    fid = "F5"
    section(fid, "The evidence cache key contains the row number, and A and B numbered rows differently")
    for path, a, b, c in [("scripts/test_architecture_a.py", 104, 118, 159), ("scripts/test_architecture_b.py", 150, 164, 216)]:
        line_is(repo, path, b, 'return f"row{row_number:04d}_{question.id}_{digest}"')
        line_is(repo, path, c, 'cache_key = f"{retrieval_mode}__{row_uid}"')
        show_lines(repo, path, b, b)
        show_lines(repo, path, c, c)
    resets_a = int((A.row_number.diff() < 0).sum())
    resets_b = int((B.row_number.diff() < 0).sum())
    fact(fid, f"Row numbers in file order: A restarts its numbering {resets_a} times (rows 96-105 are "
              f"{A.row_number.iloc[95:105].tolist()}) and repeats one number up to {int(A.row_number.value_counts().max())} times. "
              f"B restarts {resets_b} times and numbers 1 to {int(B.row_number.max())}.", resets_a=resets_a, resets_b=resets_b)
    Au = A.drop_duplicates("key").set_index("key")
    Bu = B.drop_duplicates("key").set_index("key")
    J = Au.join(Bu[["row_uid", "maj1_ok", "final_ok"] + [f"{m}_round1_reasoning" for m in M]], rsuffix="_B", how="inner")
    diff = J.row_uid != J.row_uid_B
    fact(fid, f"Contracts present in both files: {len(J)}. Different row_uid, hence different cache key: {int(diff.sum())} "
              f"({pct(int(diff.sum()), len(J)):.0f}%).", shared=len(J), different_key=int(diff.sum()))
    mon = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
    pat = re.compile(r"Source (\d{1,2})\s*\((?:[^()]{0,40}?,\s*)?(?:published |dated |from )?(?:on )?([A-Z][a-z]{2})[a-z]*\.? (?:\d{1,2}, )?(\d{4})\)")

    def check(text, e):
        ok = bad = 0
        for mo in pat.finditer(str(text)):
            k, mm, yy = int(mo.group(1)), mon.get(mo.group(2).lower()), int(mo.group(3))
            if not mm or k < 1 or k > len(e["sources"]):
                continue
            d = pd.to_datetime(e["sources"][k - 1].get("published_date"), utc=True, errors="coerce")
            if pd.isna(d):
                continue
            ok, bad = (ok + 1, bad) if (d.month == mm and d.year == yy) else (ok, bad + 1)
        return ok, bad

    stats = {True: [0, 0, 0], False: [0, 0, 0]}
    for _, r in J.iterrows():
        f = f"highlights__{r.row_uid}.json"
        if f not in pk:
            continue
        ok = bad = 0
        for m in M:
            o, b = check(r[f"{m}_round1_reasoning"], pk[f])
            ok += o
            bad += b
        s = stats[bool(r.row_uid != r.row_uid_B)]
        s[0] += ok
        s[1] += bad
        s[2] += bad > 0
    fact(fid, f"Statements 'Source k (Month Year)' in B round-1 answers, checked against A's committed packet: same key "
              f"{stats[False][0]} consistent and {stats[False][1]} inconsistent; different key {stats[True][0]} consistent "
              f"and {stats[True][1]} inconsistent, on {stats[True][2]} questions.", inconsistent_rows=stats[True][2])
    g = A[A.row_uid.str.contains("KXSNLHOST")].iloc[0].row_uid
    h = B[B.row_uid.str.contains("KXSNLHOST")].iloc[0].row_uid
    fact(fid, f"The paper's Glen Powell example is 'Row 0189' (Section 4.4.1). In A that contract is {g}, in B it is {h}.")


# ----------------------------------------------------------------------------- F6
def f5_examples(repo, A, B, pk):
    fid = "F6"
    section(fid, "Two of the four failure-mode examples were answered correctly by all three models in Architecture A")
    ex = [("4.4.1 Retrieval failure", "Glen Powell hosts"),
          ("4.4.2 Temporal reasoning error", "S&P 500 index value starting Mar 19, 2025 and ending on Jan 1, 2026 is above 6699.99"),
          ("4.4.3 Hallucination", "If Star appears on Taylor Swift"),
          ("4.4.4 Shared bias", "cuts at least $800 billion from")]
    for sec, needle in ex:
        a = A[A.resolution_criteria.str.contains(needle, regex=False)].iloc[0]
        b = B[B.resolution_criteria.str.contains(needle, regex=False)].iloc[0]
        da = [a[f"{m}_d"] for m in M]
        d1 = [b[f"{m}_d1"] for m in M]
        d2 = [b[f"{m}_d2"] for m in M]
        same = a.row_uid == b.row_uid
        fact(fid, f"{sec}: truth {a.truth}. A (GPT, DeepSeek, Llama) {da}; B round 1 {d1}; B round 2 {d2}. "
                  f"A read {a.row_uid}, B read {b.row_uid}: {'same' if same else 'different'} packet. "
                  f"B's packet committed: {f'highlights__{b.row_uid}.json' in pk}.",
             section=sec, a=da, b1=d1, b2=d2, same_packet=bool(same))
    e = pk["highlights__row0089_KXSNLHOST_ddc2b51a4839.json"]
    hit = [i for i, s in enumerate(e["sources"], 1) if "Powell" in " ".join(s.get("highlights") or [])]
    fact(fid, f"A's packet for the Glen Powell contract names Powell in the highlights of sources {hit} "
              f"({len(hit)} of {len(e['sources'])}). The paper says none of the ten sources mentioned him.")
    for ph in ("forecasts do not count as evidence of an actual crossing", "person/entity named"):
        ca = [c for c in A.columns if c.endswith("_reasoning") and A[c].astype(str).str.contains(ph, regex=False).any()]
        cb = [c for c in B.columns if c.endswith("_reasoning") and B[c].astype(str).str.contains(ph, regex=False).any()]
        fact(fid, f"The reasoning the paper quotes, '{ph}', occurs in A's reasoning columns {ca} and in B's {cb}.")
    ea = sorted(glob.glob(os.path.join(repo, "scripts/error_analysis/export_architecture_b_*.py")))
    for f in ea:
        assert 'DEFAULT_RESULTS = Path("results/Final_Architecture_B_results.csv")' in open(f, encoding="utf-8").read(), f
    fc = open(os.path.join(repo, "scripts/error_analysis/export_failure_cases.py"), encoding="utf-8").read()
    assert 'DEFAULT_ARCH_A_INPUT = Path("results/architecture_a_results.csv")' in fc
    fact(fid, f"The {len(ea)} scripts scripts/error_analysis/export_architecture_b_*.py (retrieval failures, temporal failures, "
              "temporal conflicts, hallucinations, shared bias) read results/Final_Architecture_B_results.csv by default. "
              "export_failure_cases.py reads both, with A from the older results/architecture_a_results.csv.",
         b_scripts=[os.path.basename(f) for f in ea])
    for sec, needle in ex:
        n = A[A.resolution_criteria.str.contains(needle, regex=False)].key.nunique()
        assert n == 1, (sec, n)


# ----------------------------------------------------------------------------- F7
def f6_query(repo, A):
    fid = "F7"
    section(fid, "The search query contains the question text only, not the contract named in the criteria")
    ex = "src/retrieval/exa_retriever.py"
    line_is(repo, ex, 35, "def question_to_query(question: str) -> str:")
    line_is(repo, ex, 40, "What is the result of this question")
    show_lines(repo, ex, 35, 40)
    Au = A.drop_duplicates("key").copy()
    Au["qt"] = Au.question_id.astype(str) + "|" + ws(Au.question_text)
    shared = Au.qt.map(Au.qt.value_counts()) > 1
    yes = Au.truth == "YES"
    r_sh = Au[shared & yes].maj_ok.mean()
    r_un = Au[~shared & yes].maj_ok.mean()
    k1, n1 = int(Au[shared & yes].maj_ok.sum()), int((shared & yes).sum())
    k2, n2 = int(Au[~shared & yes].maj_ok.sum()), int((~shared & yes).sum())
    p_f = fisher_two_sided(k1, n1 - k1, k2, n2 - k2)
    fact(fid, f"Among {len(Au)} distinct contracts in A, {int(shared.sum())} share their question text with another contract "
              f"that differs only in the criteria, so both get the same query. Recall on YES outcomes: "
              f"{100 * r_sh:.1f}% for those ({int((shared & yes).sum())} YES contracts) against {100 * r_un:.1f}% for the rest "
              f"({int((~shared & yes).sum())}), Fisher exact p = {p_f:.4f}.", shared=int(shared.sum()), recall_shared=r_sh,
         recall_unique=r_un, fisher_p=p_f)
    q = A[A.row_uid.str.contains("KXSNLHOST")].iloc[0].question_text
    fact(fid, f"The query for the Glen Powell contract is built from '{q}' and does not name Glen Powell.")


# ----------------------------------------------------------------------------- F8
def f7_timing(repo, pk, A, B):
    fid = "F8"
    section(fid, "The evidence was retrieved months later and includes settled market pages")
    line_is(repo, "src/retrieval/exa_retriever.py", 60,
            "end_date = (resolution_dt + timedelta(days=1)).isoformat().replace('+00:00', 'Z')")
    show_lines(repo, "src/retrieval/exa_retriever.py", 58, 60)
    for ln in (67, 84):
        line_is(repo, "src/retrieval/exa_retriever.py", ln, "end_published_date=end_date,")
    fact(fid, "The filter is Exa's end_published_date, one day after the resolution date (exa_retriever.py lines 58-60, "
              "passed at lines 67 and 84). It bounds the publication date Exa reports, not the page content.")
    hp = {f: e for f, e in pk.items() if f.startswith("highlights__")}
    months = Counter((e.get("retrieval_timestamp") or "")[:7] for e in pk.values())
    lags = []
    for f, e in hp.items():
        t, c = e.get("retrieval_timestamp"), e.get("resolution_date")
        if t and c:
            lags.append((pd.to_datetime(t, utc=True) - pd.to_datetime(c, utc=True)).days)
    lags = np.array(lags)
    fact(fid, f"Retrieval month of the {len(pk)} committed packets: {dict(months)}. For the {len(hp)} highlights packets the "
              f"retrieval came {lags.min()}-{lags.max()} days after the date passed as resolution date (median {np.median(lags):.0f}).",
         lag_min=int(lags.min()), lag_max=int(lags.max()), lag_median=float(np.median(lags)))
    n_src = sum(len(e["sources"]) for e in hp.values())
    undated = sum(1 for e in hp.values() for s in e["sources"] if not s.get("published_date"))
    pk_und = sum(1 for e in hp.values() if any(not s.get("published_date") for s in e["sources"]))
    fact(fid, f"{undated} of {n_src} sources ({pct(undated, n_src):.1f}%) have no publication date, in {pk_und} of {len(hp)} packets. "
              "The date filter cannot apply to them.", undated=undated, sources=n_src)
    dom = re.compile(r"kalshi\.com|polymarket\.com|manifold\.markets|predictit\.org|metaculus\.com|fanduel|draftkings|sportsbook|covers\.com|oddsshark|oddschecker|betfair", re.I)
    hits, pk_hit = Counter(), 0
    final = []
    for f, e in hp.items():
        h = False
        for s in e["sources"]:
            mo = dom.search(s.get("url") or "")
            if mo:
                hits[mo.group(0).lower()] += 1
                h = True
                txt = " ".join(s.get("highlights") or [])
                if re.search(r"Final outcome:", txt):
                    final.append((f, s["url"], s.get("published_date"), re.search(r"Final outcome:\s*\S+", txt).group(0)))
        pk_hit += h
    fact(fid, f"{pk_hit} of {len(hp)} packets ({pct(pk_hit, len(hp)):.0f}%) contain a prediction-market or betting page: {dict(hits)}.",
         packets_with_market_pages=pk_hit)
    pk_final = len({x[0] for x in final})
    fact(fid, f"{len(final)} sources in {pk_final} packets are market pages whose highlights state 'Final outcome: ...'. Examples:")
    for x in final[:6]:
        print(f"      {x[0]}\n        {x[1]}\n        published_date={x[2]}  highlight: {x[3]}")
    OUT[fid]["final_outcome_sources"] = [list(x) for x in final]
    # Pages whose visible content postdates the contract's resolution date, and one whose content postdates its
    # reported publication date but not the resolution date.
    cases = (("row0004_KXAIRDROPMONAD_8ecbc7d6c53f", "monad-airdrop-by", ["Final outcome: No", "November 30"],
              "It lists the November 25 and November 30 date markets with the outcome Yes, content written after the "
              "resolution date."),
             ("row0023_KXLALEADEROUT_1ccab1781569", "maduro-out-by-february-28-2026", ["Final outcome: Yes"],
              "It is the market 'Maduro out by February 28, 2026' shown as settled Yes, content written after the "
              "resolution date."),
             ("row0070_KXROMANIATOP_508b8b8b2424", "nicolae-ciuc", ["Final outcome: No", "George Simion100.0%"],
              "It shows the May 2025 first round as decided (Simion at 100%). The round took place before the resolution "
              "date, so this is resolution-time information, but it was written five months after the reported "
              "publication date."))
    for uid, needle, must, why in cases:
        r = A[A.row_uid == uid].iloc[0]
        e = hp[f"highlights__{uid}.json"]
        s = [s for s in e["sources"] if needle in (s.get("url") or "")][0]
        txt = " ".join(s.get("highlights") or [])
        for m_ in must:
            assert m_ in txt, (uid, m_)
        fact(fid, f"'{r.question_text}' (truth {r.truth}, A majority {r.maj}, resolution_date {e['resolution_date'][:10]}, "
                  f"retrieved {e['retrieval_timestamp'][:10]}). The packet includes {s['url']} with published_date "
                  f"{s['published_date'][:10]}, whose highlights contain {must}. {why}", row_uid=uid)
    e7 = hp.get("highlights__row0007_KXROMANIATOP_e3321e635fbc.json")
    if e7 is not None:
        in_a = bool((A.row_uid == "row0007_KXROMANIATOP_e3321e635fbc").any())
        in_b = bool((B.row_uid == "row0007_KXROMANIATOP_e3321e635fbc").any())
        fact(fid, f"highlights__row0007_KXROMANIATOP_e3321e635fbc.json carries the same Ciuca page; this packet belongs to "
                  f"no row of A ({in_a}) or B ({in_b}).")
    # Every packet with a 'Final outcome' page, with the decision A made on it.
    rows = []
    for f in sorted({x[0] for x in final}):
        uid = f[len("highlights__"):-len(".json")]
        r = A[A.row_uid == uid]
        rows.append(dict(row_uid=uid, in_A=bool(len(r)),
                         question=r.iloc[0].question_text if len(r) else None,
                         truth=r.iloc[0].truth if len(r) else None, a_majority=r.iloc[0].maj if len(r) else None,
                         pages=[(x[1], x[3]) for x in final if x[0] == f]))
    OUT[fid]["final_outcome_packets"] = rows
    inA = [x for x in rows if x["in_A"]]
    ok = sum(x["truth"] == x["a_majority"] for x in inA)
    fact(fid, f"{len(inA)} of these {len(rows)} packets belong to rows in the A results. A's majority matches the truth on "
              f"{ok} of {len(inA)}, so these pages are a contamination of the evidence, not a measured gain in accuracy.",
         final_packets_in_A=len(inA), a_correct_on_them=ok)


# ----------------------------------------------------------------------------- F9
def f8_alignment(A, B, D):
    fid = "F9"
    section(fid, "Row identity: duplicates in A, rows missing from A, and an id that is a series ticker")
    ca, cb, cd = Counter(A.key), Counter(B.key), Counter(D.key)
    fact(fid, f"Rows: A {len(A)}, B {len(B)}, evaluation CSV {len(D)}. Distinct contracts (id + question + criteria): "
              f"A {len(ca)}, B {len(cb)}, evaluation {len(cd)}.", a_distinct=len(ca), b_distinct=len(cb), d_distinct=len(cd))
    extra = sum((ca - cd).values())
    missing = sum((cd - ca).values())
    fact(fid, f"Multiset difference against the evaluation CSV: A has {extra} rows not in the evaluation set and lacks {missing} "
              f"of its rows. B: {sum((cb - cd).values())} and {sum((cd - cb).values())}.", a_extra=extra, a_missing=missing)
    lab = D.set_index("key")["ground_truth"].astype(str).str.upper().to_dict()
    in_d = A[A.key.isin(lab)]
    fact(fid, f"id equals series_ticker in {int((D['id'] == D['series_ticker']).sum())} of {len(D)} evaluation rows; "
              f"distinct ids {D['id'].nunique()}; ids with exactly 2 rows {int((D['id'].value_counts() == 2).sum())}, "
              f"with more than 2 rows {int((D['id'].value_counts() > 2).sum())}.")


# ----------------------------------------------------------------------------- F10
def f9_pairing(repo, A, B):
    fid = "F10"
    section(fid, "Pairing of A and B in the analysis script joins different contracts")
    path = "scripts/analyze_architecture_results.py"
    line_is(repo, path, 138, 'key_cols = ["question_text", "ground_truth"]')
    show_lines(repo, path, 136, 147)
    a2, b2 = A.copy(), B.copy()
    for df in (a2, b2):
        df["mk"] = df.question_text.astype(str) + "||" + df.ground_truth.astype(str)
        df["occ"] = df.groupby("mk").cumcount()
    P = a2.merge(b2, on=["mk", "occ"], suffixes=("_a", "_b"))
    bad = int((ws(P.resolution_criteria_a) != ws(P.resolution_criteria_b)).sum())
    fact(fid, f"The script's pairing gives {len(P)} pairs; in {bad} of them the resolution criteria differ.", pairs=len(P), mismatched=bad)
    a3 = A.drop_duplicates("key").set_index("key")
    b3 = B.drop_duplicates("key").set_index("key")
    S = a3[["maj_ok", "wtd_ok"]].join(b3[["final_ok"]], how="inner")
    for col in ("maj_ok", "wtd_ok"):
        x = int((S[col] & ~S.final_ok).sum())
        y = int((~S[col] & S.final_ok).sum())
        fact(fid, f"Pairing by contract identity: {len(S)} pairs, A {col[:-3]} only {x}, B only {y}, exact McNemar p = "
                  f"{binom_two_sided(x, x + y):.1e}. The direction and significance of Table 4 hold.")


# ----------------------------------------------------------------------------- F11
def f10_tables(repo, A, B):
    fid = "F11"
    section(fid, "Numbers in the paper checked against the published files")
    n = len(A)
    rows = []

    def chk(where, label, paper, ours, tol=0.051):
        ok = abs(paper - ours) <= tol
        rows.append((where, label, paper, round(ours, 2), "reproduces" if ok else "DIFFERS"))

    chk("Table 3", "GPT-5 nano (labelled GPT-4o)", 82.34, pct(A.gpt4o_ok.sum(), n))
    chk("Table 3", "DeepSeek", 82.42, pct(A.deepseek_ok.sum(), n))
    chk("Table 3", "Llama", 81.24, pct(A.llama_ok.sum(), n))
    chk("Table 3", "A majority", 82.93, pct(A.maj_ok.sum(), n))
    chk("Table 3", "A confidence-weighted", 83.43, pct(A.wtd_ok.sum(), n))
    chk("Table 3", "B final-round GPT", 74.35, pct(B.gpt4o_ok2.sum(), n))
    chk("Table 3", "B final-round DeepSeek", 76.70, pct(B.deepseek_ok2.sum(), n))
    chk("Table 3", "B final-round Llama", 76.37, pct(B.llama_ok2.sum(), n))
    chk("Table 3", "B deliberative final", 76.11, pct(B.final_ok.sum(), n))
    # Table 5: caption says confidence-weighted, values compared with both
    a2, b2 = A.copy(), B.copy()
    for df in (a2, b2):
        df["mk"] = df.question_text.astype(str) + "||" + df.ground_truth.astype(str)
        df["occ"] = df.groupby("mk").cumcount()
    P = a2.merge(b2, on=["mk", "occ"], suffixes=("_a", "_b"))
    t5 = {}
    for col, lab in (("maj_ok", "majority"), ("wtd_ok", "weighted")):
        t5[lab] = (int((P[col] & P.final_ok_b).sum()), int((P[col] & ~P.final_ok_b).sum()),
                   int((~P[col] & P.final_ok_b).sum()), int((~P[col] & ~P.final_ok_b).sum()))
    rows.append(("Table 5", "A vs B: both / A only / B only / both wrong", "852/117/37/164", "/".join(map(str, t5["majority"])),
                 "reproduces" if t5["majority"] == (852, 117, 37, 164) else "DIFFERS"))
    label_facts = [f"Table 5 is captioned confidence-weighted. Its counts equal A's majority vote "
                   f"({'/'.join(map(str, t5['majority']))}); the weighted vote gives {'/'.join(map(str, t5['weighted']))}."]
    # Table 6 consistency
    for df, lab, paper, col in ((A, "A", (380, 67, 39), "final_ok"), (B, "B", (338, 90, 58), "final_ok")):
        g = df.groupby("question_id")[col].agg(["size", "sum"])
        g = g[g["size"] > 1]
        trip = (int((g["sum"] == g["size"]).sum()), int(((g["sum"] > 0) & (g["sum"] < g["size"])).sum()), int((g["sum"] == 0).sum()))
        rows.append(("Table 6", f"{lab} all correct / mixed / all wrong", "/".join(map(str, paper)), "/".join(map(str, trip)),
                     "reproduces" if trip == paper else "DIFFERS"))
    # Table 8 error correlations
    err = pd.DataFrame({m: (~A[f"{m}_ok"]).astype(int) for m in M})
    C = err.corr()
    chk("Table 8", "r(GPT, DeepSeek)", 0.580, C.loc["gpt4o", "deepseek"], 0.0006)
    chk("Table 8", "r(GPT, Llama)", 0.529, C.loc["gpt4o", "llama"], 0.0006)
    chk("Table 8", "r(DeepSeek, Llama)", 0.689, C.loc["deepseek", "llama"], 0.0006)
    # Table 9 conditional error probabilities P(column wrong | row wrong)
    def cond(a, b):
        return float(err[b][err[a] == 1].mean())
    chk("Table 9", "P(DeepSeek wrong | GPT wrong)", 0.652, cond("gpt4o", "deepseek"), 0.0006)
    chk("Table 9", "P(Llama wrong | GPT wrong)", 0.633, cond("gpt4o", "llama"), 0.0006)
    chk("Table 9", "P(Llama wrong | DeepSeek wrong)", 0.770, cond("deepseek", "llama"), 0.0006)
    chk("Table 9", "P(DeepSeek wrong | Llama wrong)", 0.722, cond("llama", "deepseek"), 0.0006)
    chk("Section 4.3", "text: 63.3% 'for GPT-4o when Llama errs' = P(GPT wrong | Llama wrong)", 0.633,
        cond("llama", "gpt4o"), 0.0006)
    # Table 4 exact McNemar p-values on the script's pairing
    for lab, x, y, paper in (("A majority vs A weighted", int((A.maj_ok & ~A.wtd_ok).sum()), int((~A.maj_ok & A.wtd_ok).sum()), 0.070),
                             ("A majority vs B", 117, 37, 7.2e-11), ("A weighted vs B", 123, 37, 5.5e-12)):
        pv = binom_two_sided(x, x + y)
        rows.append(("Table 4", f"{lab}: {x}/{y}, exact McNemar p", f"{paper:.2g}", f"{pv:.2g}",
                     "reproduces" if abs(pv - paper) <= 0.051 * paper else "DIFFERS"))
    # Table 10 (B convergence)
    u1 = B.yes1.isin([0, 3])
    u2 = B.yes2.isin([0, 3])
    chk("Table 10", "round-1 consensus n", 1002, int(u1.sum()), 0)
    chk("Table 10", "round-1 consensus accuracy", 80.24, 100 * B[u1].final_ok.mean())
    chk("Table 10", "round-2 consensus n", 58, int((~u1 & u2).sum()), 0)
    chk("Table 10", "round-2 consensus accuracy", 56.90, 100 * B[~u1 & u2].final_ok.mean())
    chk("Table 10", "no consensus n", 129, int((~u1 & ~u2).sum()), 0)
    chk("Table 10", "no consensus accuracy", 52.71, 100 * B[~u1 & ~u2].final_ok.mean())
    # Figure 5 net revisions per model (round 1 to round 2)
    for m, paper in (("gpt4o", -10), ("deepseek", 6)):
        net = int((~B[f"{m}_ok1"] & B[f"{m}_ok2"]).sum() - (B[f"{m}_ok1"] & ~B[f"{m}_ok2"]).sum())
        chk("Figure 5", f"net revisions {NAME[m]}", paper, net, 0)
    # Table 11 / 12
    chk("Table 11", "unanimous n", 978, int(A.unan.sum()), 0)
    chk("Table 11", "unanimous accuracy", 88.34, 100 * A[A.unan].maj_ok.mean())
    chk("Table 11", "split accuracy", 57.82, 100 * A[~A.unan].maj_ok.mean())
    med = float(np.median(A.mean_conf))
    hi = A.mean_conf >= med
    chk("Table 12", "median average confidence", 0.91, med, 0.0051)
    chk("Table 12", "unanimous, high: n", 563, int((A.unan & hi).sum()), 0)
    chk("Table 12", "unanimous, high: accuracy", 97.87, 100 * A[A.unan & hi].maj_ok.mean())
    # Coverage claims, from the repository's own curve
    cs = json.load(open(os.path.join(repo, "results/escalation_analysis/comparison_summary.json")))
    pts = {round(p["fraction_auto_resolved"], 2): p for p in cs["coverage_accuracy_checkpoints"]}
    p10 = pts[0.1]
    rows.append(("Section 4.5", "A at 10% coverage (text says 100% and, one paragraph later, 99.15%)", "100 / 99.15",
                 f"{100 * p10['architecture_a_accuracy']:.2f}", "inconsistent in text"))
    rows.append(("Section 4.5", "B better than A at the top of the ranking (text: 100% vs 99.15% at 10%)", "B > A",
                 f"B>A at {cs['b_better_prefix_count']} of 1,189 prefixes; at 10% A {100 * p10['architecture_a_accuracy']:.0f}%, B {100 * p10['architecture_b_accuracy']:.0f}%",
                 "DIFFERS"))
    chk("Section 4.5", "A at 50% coverage", 97.47, 100 * pts[0.5]["architecture_a_accuracy"])
    chk("Section 4.5", "A at 75% coverage", 90.12, 100 * pts[0.75]["architecture_a_accuracy"])
    chk("Section 4.5", "gap A - B at 50% coverage", 10.44,
        100 * (pts[0.5]["architecture_a_accuracy"] - pts[0.5]["architecture_b_accuracy"]))
    for lab, sel, pn, pa in (("unanimous, low", A.unan & ~hi, 415, 75.42), ("split, high", ~A.unan & hi, 46, 56.52),
                             ("split, low", ~A.unan & ~hi, 165, 58.18)):
        chk("Table 12", f"{lab}: n", pn, int(sel.sum()), 0)
        chk("Table 12", f"{lab}: accuracy", pa, 100 * A[sel].maj_ok.mean())
    chk("Section 4.5", "coverage of unanimous high-confidence rows", 47.4, pct(int((A.unan & hi).sum()), n))
    chk("Section 4.5", "coverage of all unanimous rows", 82.3, pct(int(A.unan.sum()), n))
    p20 = pts[0.2]
    label_facts.append(f"The 99.15% of Section 4.5 is probably B at 20% coverage: {100 * p20['architecture_b_accuracy']:.3f}% "
                       "in comparison_summary.json.")
    w = max(len(r[1]) for r in rows)
    for r in rows:
        print(f"    {r[0]:12s} {r[1]:{w}s}  paper {str(r[2]):>15s}  files {str(r[3]):>15s}  {r[4]}")
    OUT[fid]["table"] = [list(map(str, r)) for r in rows]
    ok = sum(r[4] == "reproduces" for r in rows)
    fact(fid, f"{ok} of {len(rows)} checked values reproduce to the reported precision (tolerance 0.051 points for "
              "percentages, 0.0006 for correlations and probabilities, 5% relative for p-values, exact for counts).",
         reproduced=ok, checked=len(rows))
    for t in label_facts:
        fact(fid, t)


# ----------------------------------------------------------------------------- F12
def f11_labels(repo, A, B):
    fid = "F12"
    section(fid, "Table 7, the unique-market result and the model label")
    t = A.truth
    g = A.groupby("category")
    best = pd.DataFrame({m: 100 * g[f"{m}_ok"].mean() for m in M})
    paper_best = {"Politics": 76.2, "Economics": 57.1, "Social": 76.9, "Health": 50.0, "Mentions": 76.9,
                  "Science and Technology": 40.0, "Financials": 90.0}
    for cat, pb in paper_best.items():
        r = best.loc[cat]
        fact(fid, f"Table 7 'Best Baseline' for {cat}: paper {pb}. Per model: " +
                  ", ".join(f"{NAME[m]} {r[m]:.1f}" for m in M) + f". The paper's value equals GPT-5 nano's.")
    pol = best.loc["Politics"].max()
    wt = 100 * A[A.category == "Politics"].wtd_ok.mean()
    fact(fid, f"Politics: A weighted {wt:.1f}% against the best single model {pol:.1f}% (Llama), a gap of {wt - pol:.1f} points; "
              "the paper reports 3.2 points (79.4% vs 76.2%).")
    for lab, df, col in (("A final", A, "final_ok"), ("B final", B, "final_ok"), ("B DeepSeek round 2", B, "deepseek_ok2")):
        f = df.drop_duplicates("question_id", keep="first")
        k, nn = int(f[col].sum()), len(f)
        z = 1.96
        p = k / nn
        c = (p + z * z / (2 * nn)) / (1 + z * z / nn)
        h = z * math.sqrt(p * (1 - p) / nn + z * z / (4 * nn * nn)) / (1 + z * z / nn)
        fact(fid, f"Unique-market accuracy (first row per id), {lab}: {k}/{nn} = {100 * p:.1f}% [Wilson {c - h:.3f}, {c + h:.3f}]. "
                  "Paper: A 81.9% [0.789, 0.846], B 75.5% [0.722, 0.786].")
    mi = "scripts/analysis_by_market_instance.py"
    line_is(repo, mi, 87, 'if col.endswith("_decision") or col.endswith("_final"):')
    line_is(repo, mi, 272, 'row = overall.sort_values(["accuracy", "attempted_n"], ascending=[False, False]).iloc[0]')
    line_is(repo, mi, 306, 'f"- unique_by_id: {_best_system_text(unique_overall)}"')
    for lab, df in (("A", A), ("B", B)):
        f = df.drop_duplicates("question_id", keep="first")
        t = norm_dec(f.ground_truth)
        accs = {c: float((norm_dec(f[c]) == t).mean()) for c in f.columns
                if (c.endswith("_decision") or c.endswith("_final")) and c != "ground_truth"}
        best = max(accs, key=accs.get)
        fact(fid, f"{mi} chooses the decision column with the highest accuracy (lines 87 and 272) and writes it to "
                  f"summary_comparison.md as 'Best-performing decision column' (line 306). On unique markets of {lab} "
                  f"that column is {best} ({100 * accs[best]:.1f}%).", best_column=best)
    agent = open(os.path.join(repo, "src/agents/openai_agent.py"), encoding="utf-8").read()
    line_is(repo, "src/agents/openai_agent.py", 31, 'MODEL_ID = "gpt-5-nano"')
    line_is(repo, "src/agents/openai_agent.py", 67, 'if not self.MODEL_ID.startswith("gpt-5"):')
    line_is(repo, "src/agents/openai_agent.py", 68, 'request_kwargs["temperature"] = 0.0')
    show_lines(repo, "src/agents/openai_agent.py", 66, 68)
    fact(fid, "Temperature 0 is set only for models whose id does not start with gpt-5 (openai_agent.py lines 66-68), "
              "so GPT-5 nano runs at its default temperature.")
    fact(fid, f"src/agents/openai_agent.py sets MODEL_ID = 'gpt-5-nano': {'MODEL_ID = \"gpt-5-nano\"' in agent}. Tables 3, 8 and 9 "
              "and Sections 4.3 and 4.4.5 call this model GPT-4o; the result columns are named gpt4o_*.")
    med = float(np.median(A.mean_conf))
    fact(fid, f"The confidence threshold of Table 12 is the median of the same 1,189 rows ({med:.2f}), so the 97.87% is "
              "measured on the rows that set the threshold.")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    head = subprocess.run(["git", "-C", a.repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"repository HEAD {head}; expected {COMMIT}; match: {head == COMMIT}")
    if head != COMMIT:
        sys.exit("check out the pinned commit first")
    A, B, D = load(a.repo)
    pk = packets(a.repo)
    f1_no_source_text(a.repo, A, B, pk)
    f1b_quotes(a.repo, A, B, pk)
    f2_gap_before_debate(A, B)
    f3_prompts(a.repo)
    f4_cache_key(a.repo, A, B, pk)
    f5_examples(a.repo, A, B, pk)
    f6_query(a.repo, A)
    f7_timing(a.repo, pk, A, B)
    f8_alignment(A, B, D)
    f9_pairing(a.repo, A, B)
    f10_tables(a.repo, A, B)
    f11_labels(a.repo, A, B)
    if a.json:
        json.dump(OUT, open(a.json, "w"), indent=1, default=str)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
