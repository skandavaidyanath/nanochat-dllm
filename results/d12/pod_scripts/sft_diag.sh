#!/bin/bash
# SFT saturation diagnostic: 200 optimizer steps (= 3200 micro-batches at grad-accum 16) of MDM SFT with
# three minimum mask rates, categorical ChatCORE on 400 problems per task at the end. Runs alongside the sweep.
cd ~/nanochat-dllm && source .venv/bin/activate
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
for EPS in 0.1 0.3 1.0; do
  TAG=diag_mdm_sft_eps$EPS
  echo "===== DIAG $TAG START $(date -u +%H:%M:%S) ====="
  python -m scripts.chat_sft --model-tag=d12_mdm --output-tag=$TAG --mdm-min-mask-rate=$EPS \
      --num-iterations=3200 --eval-every=200 --chatcore-every=200 --chatcore-max-cat=400 --chatcore-max-sample=1 \
      --device-batch-size=16 --run=$TAG 2>&1 | grep -E "Validation bpb|ChatCORE|ARC|MMLU|GSM8K|HumanEval|Traceback|Error"
  echo "===== DIAG $TAG DONE $(date -u +%H:%M:%S) ====="
done
echo "===== DIAG ALL DONE ====="
