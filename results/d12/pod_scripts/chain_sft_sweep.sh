#!/bin/bash
# Phase 2a chain, runs ON a multi-GPU pod: setup -> the two post-training-2 probes (GPU 0) -> one SFT sweep lane
# per GPU in parallel -> results bundle -> wait for the local side to acknowledge its final pull (~/ACK, up to
# ACK_WAIT_MIN minutes) -> terminate this pod through the Prime Intellect API. The pod stops itself even if setup
# fails or nobody acknowledges, so an unattended laptop cannot leave it billing.
# Needs ~/pod.env (mode 600) with PRIME_API_KEY and POD_ID, placed by the launcher. Progress: ~/chain.status.
cd ~/nanochat-dllm
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
ACK_WAIT_MIN="${ACK_WAIT_MIN:-20}"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
[ "$NGPU" -ge 1 ] || NGPU=1
status() { echo "$(date -u +%FT%TZ) $*" | tee -a ~/chain.status; }
status "CHAIN START"
# RESUME=1: setup, upload and probes already done on this box (re-entry after a lane was stopped); wait for any
# training/eval process still running, then go straight to the sweep, which skips finished runs.
if [ "${RESUME:-0}" = "1" ]; then
  status "RESUME: waiting for running chat_sft/chat_eval processes"
  while pgrep -f "[s]cripts.chat_sft|[s]cripts.chat_eval" >/dev/null; do sleep 15; done
  echo "SETUP DONE (resumed)" >> ~/setup.log; touch ~/UPLOAD_DONE
else
  bash ~/scripts/remote_setup_sft.sh > ~/setup.log 2>&1
fi
have_ckpt() { ls "$NANOCHAT_BASE_DIR/base_checkpoints/$1"/model_*.pt >/dev/null 2>&1; }
if grep -q "SETUP DONE" ~/setup.log; then
  # the launcher uploads the base checkpoints in parallel with setup and touches ~/UPLOAD_DONE when finished
  status "WAITING FOR CHECKPOINT UPLOAD"
  for _ in $(seq 1 180); do [ -f ~/UPLOAD_DONE ] && break; sleep 10; done
fi
if grep -q "SETUP DONE" ~/setup.log && have_ckpt d12_ar && have_ckpt d12_mdm; then
  source .venv/bin/activate
  if [ "${RESUME:-0}" != "1" ]; then
    status "PROBES START"
    CUDA_VISIBLE_DEVICES=0 python -m scripts.probe_answer_slot --tag d12_ar --tag d12_mdm > ~/probe_answer_slot.log 2>&1
    CUDA_VISIBLE_DEVICES=0 python -m scripts.probe_grad_share --tag d12_ar --tag d12_mdm > ~/probe_grad_share.log 2>&1
  fi
  status "SWEEP START ($NGPU lanes)"
  for i in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=$i NLANES=$NGPU LANE=$i bash runs/exp_d12_sft_sweep.sh >> ~/lane$i.log 2>&1 &
  done
  wait
  status "SWEEP DONE"
else
  status "SETUP FAILED OR CHECKPOINTS MISSING (see setup.log)"
fi
tar czf ~/results_bundle.tgz --exclude='*.pt' -C ~ setup.log chain.status probe_answer_slot.log probe_grad_share.log \
    $(cd ~ && ls lane*.log 2>/dev/null) .cache/nanochat/logs .cache/nanochat/probes 2>/dev/null
status "CHAIN DONE (bundle ready; waiting up to ${ACK_WAIT_MIN} min for ~/ACK)"
for _ in $(seq 1 "$ACK_WAIT_MIN"); do [ -f ~/ACK ] && break; sleep 60; done
status "TERMINATING ($([ -f ~/ACK ] && echo acked || echo timeout))"
source ~/pod.env
curl -s -X DELETE -H "Authorization: Bearer $PRIME_API_KEY" "https://api.primeintellect.ai/api/v1/pods/$POD_ID" >> ~/chain.status 2>&1
