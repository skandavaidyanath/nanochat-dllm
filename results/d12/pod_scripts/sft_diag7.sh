#!/bin/bash
cd ~/nanochat-dllm && source .venv/bin/activate
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
for MT in ar mdm; do
  TAG=diag_${MT}_sft_mmluonly_lowlr
  echo "===== DIAG $TAG START $(date -u +%H:%M:%S) ====="
  python -m scripts.chat_sft --model-tag=d12_$MT --output-tag=$TAG --smoltalk-epochs=0 --gsm8k-epochs=0 --mmlu-epochs=3 --init-lr-frac=0.1 \
      --num-iterations=3200 --eval-every=200 --chatcore-every=200 --chatcore-max-cat=400 --chatcore-max-sample=1 \
      --device-batch-size=16 --run=$TAG 2>&1 | grep --line-buffered -E "Validation bpb|ChatCORE|Traceback|Error"
  echo "===== DIAG $TAG DONE $(date -u +%H:%M:%S) ====="
done
echo "===== DIAG7 ALL DONE ====="
