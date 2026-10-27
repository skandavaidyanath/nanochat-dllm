#!/bin/bash
# Phase 2b chain, runs ON a single-GPU pod: setup (with 24 pretraining shards for the AR-to-diffusion run) -> wait for
# the checkpoint upload -> runs/exp_d12_phase2b.sh -> bundle -> wait for the local ACK (up to ACK_WAIT_MIN) -> terminate
# this pod through the Prime API. The launcher's deadline watchdog is the backstop. Needs ~/pod.env. Progress: ~/chain.status
cd ~/nanochat-dllm
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
ACK_WAIT_MIN="${ACK_WAIT_MIN:-20}"
status() { echo "$(date -u +%FT%TZ) $*" | tee -a ~/chain.status; }
status "CHAIN START (phase 2b)"
SHARDS=24 bash ~/scripts/remote_setup_sft.sh > ~/setup.log 2>&1
have_ckpt() { ls "$NANOCHAT_BASE_DIR/base_checkpoints/$1"/model_*.pt >/dev/null 2>&1; }
if grep -q "SETUP DONE" ~/setup.log; then
  status "WAITING FOR CHECKPOINT UPLOAD"
  for _ in $(seq 1 180); do [ -f ~/UPLOAD_DONE ] && break; sleep 10; done
fi
if grep -q "SETUP DONE" ~/setup.log && have_ckpt d12_ar && have_ckpt d12_mdm; then
  status "RUNS START"
  bash runs/exp_d12_phase2b.sh >> ~/lane0.log 2>&1
  status "RUNS DONE"
else
  status "SETUP FAILED OR CHECKPOINTS MISSING (see setup.log)"
fi
tar czf ~/results_bundle.tgz --exclude='*.pt' -C ~ setup.log chain.status lane0.log .cache/nanochat/logs .cache/nanochat/base_eval 2>/dev/null
status "CHAIN DONE (bundle ready; waiting up to ${ACK_WAIT_MIN} min for ~/ACK)"
for _ in $(seq 1 "$ACK_WAIT_MIN"); do [ -f ~/ACK ] && break; sleep 60; done
status "TERMINATING ($([ -f ~/ACK ] && echo acked || echo timeout))"
source ~/pod.env
curl -s -X DELETE -H "Authorization: Bearer $PRIME_API_KEY" "https://api.primeintellect.ai/api/v1/pods/$POD_ID" >> ~/chain.status 2>&1
