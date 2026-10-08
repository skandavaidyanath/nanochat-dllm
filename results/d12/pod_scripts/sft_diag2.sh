#!/bin/bash
cd ~/nanochat-dllm && source .venv/bin/activate
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
TAG=diag_mdm_sft_unemb0.0004
echo "===== DIAG $TAG START $(date -u +%H:%M:%S) ====="
python -m scripts.chat_sft --model-tag=d12_mdm --output-tag=$TAG --unembedding-lr=0.0004 \
    --num-iterations=3200 --eval-every=200 --chatcore-every=200 --chatcore-max-cat=400 --chatcore-max-sample=1 \
    --device-batch-size=16 --run=$TAG 2>&1 | grep -E "Validation bpb|ChatCORE|Traceback|Error"
echo "===== DIAG $TAG DONE $(date -u +%H:%M:%S) ====="
