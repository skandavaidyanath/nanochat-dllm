#!/bin/bash
cd ~/nanochat-dllm && source .venv/bin/activate
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
TAG=diag_mdm_sft_mmlu9_lr0.3
echo "===== DIAG $TAG START $(date -u +%H:%M:%S) ====="
python -m scripts.chat_sft --model-tag=d12_mdm --output-tag=$TAG --init-lr-frac=0.3 --mmlu-epochs=9 \
    --num-iterations=6400 --eval-every=200 --chatcore-every=200 --chatcore-max-cat=400 --chatcore-max-sample=1 \
    --device-batch-size=16 --run=$TAG 2>&1 | grep --line-buffered -E "Training mixture|Validation bpb|ChatCORE|Traceback|Error"
echo "===== DIAG $TAG DONE $(date -u +%H:%M:%S) ====="
echo "===== DIAG9 ALL DONE ====="
