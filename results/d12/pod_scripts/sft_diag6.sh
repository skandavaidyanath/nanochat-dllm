#!/bin/bash
cd ~/nanochat-dllm && source .venv/bin/activate
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
TAG=diag_mdm_sft_padded
echo "===== DIAG $TAG START $(date -u +%H:%M:%S) ====="
python -m scripts.chat_sft --model-tag=d12_mdm --output-tag=$TAG --mdm-sft-format=padded --max-seq-len=1024 \
    --num-iterations=6400 --eval-every=200 --chatcore-every=200 --chatcore-max-cat=400 --chatcore-max-sample=1 \
    --device-batch-size=16 --run=$TAG 2>&1 | grep --line-buffered -E "Training mixture|gradient accumulation|Validation bpb|ChatCORE|Traceback|Error"
echo "===== DIAG $TAG DONE $(date -u +%H:%M:%S) ====="
