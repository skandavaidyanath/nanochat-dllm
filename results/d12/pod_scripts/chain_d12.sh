#!/bin/bash
cd ~/nanochat-dllm
bash ~/scripts/remote_setup.sh > ~/setup.log 2>&1
if grep -q "SETUP DONE" ~/setup.log && ! grep -qE "[0-9]+ failed" ~/setup.log; then
  echo "===== SWEEP LAUNCH $(date -u +%H:%M:%S) =====" > ~/sweep.log
  SKIP_SETUP=1 NPROC=1 bash runs/exp_d12_sweep.sh >> ~/sweep.log 2>&1
  echo "===== SWEEP EXIT $? $(date -u +%H:%M:%S) =====" >> ~/sweep.log
else
  echo "===== SETUP FAILED, SWEEP NOT STARTED =====" > ~/sweep.log
fi
