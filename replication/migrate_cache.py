"""Copy the committed evidence packets to the question-based cache keys of the patch.

After replication_fixes.patch the scripts look packets up by evidence_cache_key (market id, question text,
resolution criteria). The committed packets are named by row_uid, so without this step a run with
--cache-only finds nothing and a run without it retrieves new evidence today. This script copies every
committed packet whose question text and criteria match an evaluation row exactly to the new name.
Packets are copied, never moved or edited. When a question has several packets, the one Architecture A
read is used.

Usage, inside the patched clone:
  python3 -I /path/to/migrate_cache.py --repo . [--dry-run]
"""
import argparse, glob, importlib.util, json, os, shutil
import pandas as pd


def ws(s):
    return " ".join(str(s).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("keys", os.path.join(a.repo, "src/retrieval/keys.py"))
    keys = importlib.util.module_from_spec(spec); spec.loader.exec_module(keys)
    D = pd.read_csv(os.path.join(a.repo, "data/kalshibench_v2_evaluation_Filtered.csv"))
    by_q = {}
    for r in D.itertuples():
        by_q.setdefault((str(r.id), ws(r.question), ws(r.description)), r)
    cache = os.path.join(a.repo, "cache/evidence")
    A = pd.read_csv(os.path.join(a.repo, "results/Final_Architecture_A_results.csv"))
    a_uid = set(A.row_uid.astype(str))
    cand, skipped = {}, 0
    for f in sorted(glob.glob(os.path.join(cache, "highlights__row*.json"))):
        e = json.load(open(f))
        q = by_q.get((str(e["question_id"]), ws(e["question_text"]), ws(e["resolution_criteria"])))
        if q is None:
            skipped += 1
            continue
        uid = os.path.basename(f)[len("highlights__"):-len(".json")]
        cand.setdefault(keys.evidence_cache_key("highlights", q), []).append((uid not in a_uid, f))
    done = from_a = 0
    for k, fs in cand.items():
        not_a, src = sorted(fs)[0]          # prefer the packet that Architecture A read
        from_a += not not_a
        dst = os.path.join(cache, k + ".json")
        if not a.dry_run and not os.path.exists(dst):
            shutil.copyfile(src, dst)
        done += 1
    print(f"{done} questions get a packet under the new key ({from_a} of them the packet Architecture A read); "
          f"{sum(len(v) - 1 for v in cand.values())} further packets for the same questions are left unused; "
          f"{skipped} packets have no exact evaluation row" + (". Dry run, nothing written." if a.dry_run else "."))


if __name__ == "__main__":
    main()
