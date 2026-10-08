#!/bin/bash
# Throughput benchmark: AR vs MDM at d12, 2048 ctx, nanochat defaults, 30 steps each (single GPU).
cd ~/nanochat-dllm
source .venv/bin/activate
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
for MT in ar mdm; do
  echo "===== BENCH $MT START $(date +%H:%M:%S) ====="
  python -m scripts.base_train --model-type=$MT --depth=12 --device-batch-size=32 --total-batch-size=524288 \
      --num-iterations=30 --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
      --run=dummy --model-tag=bench_${MT}_d12 2>&1 | grep -E "Model type|Tokens / micro|gradient accumulation|step 000(2[0-9])/|Peak memory|Total training time"
  echo "===== BENCH $MT DONE $(date +%H:%M:%S) ====="
done
