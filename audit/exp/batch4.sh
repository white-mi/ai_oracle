#!/bin/bash
# Sequential runs; stop the whole batch at the first subscription-limit message.
cd "$(dirname "$0")"
step() { echo "=== $(date -u +%H:%M) $*"; python3 "$@" 2>&1 | tee -a batch4.log | tail -3; if grep -q "STOPPED: subscription limit\|LIMIT:" batch4.log; then echo "BATCH4 STOPPED AT LIMIT" | tee -a batch4.log; exit 0; fi; }
: > batch4.log
step e3_run.py annot
step e2_run.py closed sonnet45
step e2_run.py closed haiku55
step e2_run.py rep1 sonnet45
step e2_run.py noul sonnet45
step e2_run.py abstain sonnet45
echo "BATCH4 DONE" | tee -a batch4.log
