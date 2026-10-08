#!/bin/bash
# End-to-end smoke test of every pipeline stage for both model types on a single GPU (tiny d6, 512 ctx).
cd ~/nanochat-dllm
source .venv/bin/activate
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
stage() { echo "===== STAGE $1 START $(date +%H:%M:%S) ====="; }
done_() { echo "===== STAGE $1 DONE $(date +%H:%M:%S) ====="; }
fail() { echo "===== STAGE $1 FAILED ====="; exit 1; }
TRAIN_ARGS="--depth=6 --head-dim=64 --max-seq-len=512 --device-batch-size=32 --total-batch-size=16384 --eval-every=20 --eval-tokens=65536 --core-metric-every=40 --core-metric-max-per-task=12 --sample-every=40 --num-iterations=40"
SFT_ARGS="--num-iterations=20 --eval-every=10 --eval-tokens=16384 --chatcore-every=20 --chatcore-max-cat=8 --chatcore-max-sample=2 --device-batch-size=8"
RL_ARGS="--num-steps=2 --examples-per-step=2 --num-samples=4 --device-batch-size=4 --max-new-tokens=32 --eval-every=1 --eval-examples=2 --save-every=1"
for MT in mdm ar; do
  TAG=smoke_${MT}_d6
  stage "$MT train";    python -m scripts.base_train --model-type=$MT $TRAIN_ARGS --run=$TAG --model-tag=$TAG || fail "$MT train"; done_ "$MT train"
  stage "$MT base_eval"; python -m scripts.base_eval --model-tag=$TAG --device-batch-size=8 --split-tokens=65536 --max-per-task=12 --mdm-mc=2 || fail "$MT base_eval"; done_ "$MT base_eval"
  stage "$MT sft";      python -m scripts.chat_sft --model-tag=$TAG $SFT_ARGS --run=${TAG}_sft || fail "$MT sft"; done_ "$MT sft"
  stage "$MT chat_eval"; python -m scripts.chat_eval -i sft -g $TAG -a "ARC-Easy|GSM8K" -x 4 -b 4 -m 48 || fail "$MT chat_eval"; done_ "$MT chat_eval"
  stage "$MT rl";       python -m scripts.chat_rl --model-tag=$TAG $RL_ARGS --run=${TAG}_rl || fail "$MT rl"; done_ "$MT rl"
done
echo "===== ALL STAGES DONE ====="
