#!/bin/bash
S=.
cd $S/exp
while ! grep -q "BATCH2 DONE" batch2.log; do sleep 20; done
python3 e3_run.py deb
python3 e3_run.py annot
python3 e3_run.py split
echo "BATCH3 DONE"
