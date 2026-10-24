#!/bin/bash
cd ./exp
python3 e2_run.py closed haiku
python3 e2_run.py rep1 haiku
python3 e2_run.py abstain haiku
python3 e2_run.py noul haiku
python3 e2_run.py noulb haiku
python3 e2_run.py rep2 haiku
python3 e2_run.py closed sonnet
python3 e2_run.py date haiku 150
echo BATCH2 DONE
