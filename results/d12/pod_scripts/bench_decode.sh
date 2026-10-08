#!/bin/bash
while pgrep -f "exp_d12_datarep|exp_d12_sweep" >/dev/null; do sleep 60; done
cd ~/nanochat-dllm && source .venv/bin/activate
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat
echo "===== DECODE BENCH START $(date -u +%H:%M:%S) ====="
python -m scripts.infer_bench_mdm --ar-tag d12_ar --mdm-tag d12_mdm --gen-tokens 256 --batch-sizes 1,8,32 2>&1 | grep -vE "INFO|Warning"
echo "===== DECODE BENCH DONE ====="
