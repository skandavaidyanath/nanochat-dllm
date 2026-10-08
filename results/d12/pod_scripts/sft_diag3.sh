#!/bin/bash
# wait for diag2, then softcap diagnostics
while pgrep -f sft_diag2.sh >/dev/null; do sleep 20; done
cd ~/nanochat-dllm && source .venv/bin/activate
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
for CAP in 0 50; do
  TAG=diag_mdm_sft_cap$CAP
  echo "===== DIAG $TAG START $(date -u +%H:%M:%S) ====="
  python -m scripts.chat_sft --model-tag=d12_mdm --output-tag=$TAG --softcap=$CAP \
      --num-iterations=3200 --eval-every=200 --chatcore-every=200 --chatcore-max-cat=400 --chatcore-max-sample=1 \
      --device-batch-size=16 --run=$TAG 2>&1 | grep -E "softcap|Validation bpb|ChatCORE|Traceback|Error"
  echo "===== DIAG $TAG DONE $(date -u +%H:%M:%S) ====="
done
echo "===== DIAG3 ALL DONE ====="
