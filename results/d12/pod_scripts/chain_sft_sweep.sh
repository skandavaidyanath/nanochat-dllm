#!/bin/bash
# Phase 2a chain, runs ON a 2-GPU pod: setup -> the two post-training-2 probes (GPU 0) -> the two SFT sweep lanes
# in parallel (one GPU each) -> results bundle -> wait for the local side to acknowledge its final pull (~/ACK,
# up to ACK_WAIT_MIN minutes) -> terminate this pod through the Prime Intellect API. The pod stops itself even if
# setup fails or nobody acknowledges, so an unattended laptop cannot leave it billing.
# Needs ~/pod.env (mode 600) with PRIME_API_KEY and POD_ID, placed by the launcher. Progress: ~/chain.status.
cd ~/nanochat-dllm
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
ACK_WAIT_MIN="${ACK_WAIT_MIN:-120}"
status() { echo "$(date -u +%FT%TZ) $*" | tee -a ~/chain.status; }
status "CHAIN START"
bash ~/scripts/remote_setup_sft.sh > ~/setup.log 2>&1
if grep -q "SETUP DONE" ~/setup.log; then
  source .venv/bin/activate
  status "PROBES START"
  CUDA_VISIBLE_DEVICES=0 python -m scripts.probe_answer_slot --tag d12_ar --tag d12_mdm > ~/probe_answer_slot.log 2>&1
  CUDA_VISIBLE_DEVICES=0 python -m scripts.probe_grad_share --tag d12_ar --tag d12_mdm > ~/probe_grad_share.log 2>&1
  status "SWEEP START"
  CUDA_VISIBLE_DEVICES=0 LANE=0 bash runs/exp_d12_sft_sweep.sh > ~/lane0.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 LANE=1 bash runs/exp_d12_sft_sweep.sh > ~/lane1.log 2>&1 &
  wait
  status "SWEEP DONE"
else
  status "SETUP FAILED"
fi
tar czf ~/results_bundle.tgz --exclude='*.pt' -C ~ setup.log chain.status probe_answer_slot.log probe_grad_share.log \
    lane0.log lane1.log .cache/nanochat/logs .cache/nanochat/probes 2>/dev/null
status "CHAIN DONE (bundle ready; waiting up to ${ACK_WAIT_MIN} min for ~/ACK)"
for _ in $(seq 1 "$ACK_WAIT_MIN"); do [ -f ~/ACK ] && break; sleep 60; done
status "TERMINATING ($([ -f ~/ACK ] && echo acked || echo timeout))"
source ~/pod.env
curl -s -X DELETE -H "Authorization: Bearer $PRIME_API_KEY" "https://api.primeintellect.ai/api/v1/pods/$POD_ID" >> ~/chain.status 2>&1
