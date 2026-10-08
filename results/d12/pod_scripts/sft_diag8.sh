#!/bin/bash
cd ~/nanochat-dllm && source .venv/bin/activate
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
run() { TAG=$1; MT=$2; FRAC=$3
  echo "===== DIAG $TAG START $(date -u +%H:%M:%S) ====="
  python -m scripts.chat_sft --model-tag=d12_$MT --output-tag=$TAG --init-lr-frac=$FRAC \
      --num-iterations=3200 --eval-every=200 --chatcore-every=200 --chatcore-max-cat=400 --chatcore-max-sample=1 \
      --device-batch-size=16 --run=$TAG 2>&1 | grep --line-buffered -E "Validation bpb|ChatCORE|Traceback|Error"
  echo "===== DIAG $TAG DONE $(date -u +%H:%M:%S) ====="; }
run diag_mdm_sft_lr0.1 mdm 0.1
run diag_mdm_sft_lr0.3 mdm 0.3
run diag_ar_sft_lr0.1 ar 0.1
echo "===== DIAG8 ALL DONE ====="
