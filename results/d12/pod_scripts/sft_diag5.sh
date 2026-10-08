#!/bin/bash
while pgrep -f sft_diag4.sh >/dev/null; do sleep 20; done
cd ~/nanochat-dllm && source .venv/bin/activate
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
TAG=diag_ar_sft_mmluonly
echo "===== DIAG $TAG START $(date -u +%H:%M:%S) ====="
python -m scripts.chat_sft --model-tag=d12_ar --output-tag=$TAG --smoltalk-epochs=0 --gsm8k-epochs=0 --mmlu-epochs=3 \
    --num-iterations=3200 --eval-every=200 --chatcore-every=200 --chatcore-max-cat=400 --chatcore-max-sample=1 \
    --device-batch-size=16 --run=$TAG 2>&1 | grep --line-buffered -E "Training mixture|Validation bpb|ChatCORE|Traceback|Error"
echo "===== DIAG $TAG DONE $(date -u +%H:%M:%S) ====="
echo "===== DIAG5 ALL DONE ====="
