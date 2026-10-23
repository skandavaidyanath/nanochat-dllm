#!/bin/bash
# Launch the phase-2a chain on a Prime Intellect pod with a hard spending guard.
# Order matters: the first thing that runs on the box is a detached deadline watchdog that terminates the pod
# after MAX_HOURS no matter what else happens (chain never starts, laptop goes offline, ssh breaks). Only then are
# code, tokenizer and checkpoints synced and the chain started; the chain terminates the pod itself when done.
# Usage: CLOUD_ID=gpu_1x_h100_pcie MAX_HOURS=14 bash results/launch_pod.sh
#        (reads PRIME_API_KEY / PRIME_INTELLECT_KEY from .env; needs the ssh alias block written by this script)
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
export PRIME_API_KEY="${PRIME_API_KEY:-$PRIME_INTELLECT_KEY}"
CLOUD_ID="${CLOUD_ID:?set CLOUD_ID (e.g. gpu_1x_h100_pcie, 4H200.141S.176V_SPOT)}"
MAX_HOURS="${MAX_HOURS:?set MAX_HOURS: the pod is terminated after this many hours no matter what}"
NAME="${NAME:-d12-sft-sweep}"
HOST="${HOST:-prime-sft}"
SCRATCH="${SCRATCH:-$(mktemp -d)}"
API="https://api.primeintellect.ai/api/v1/pods"
log() { echo "$(date -u +%H:%M:%S) $*"; }

# 1) create the pod
OUT=$(prime --plain pods create --cloud-id "$CLOUD_ID" --name "$NAME" --image ubuntu_22_cuda_12 -y 2>&1 | tail -6)
echo "$OUT"
POD=$(echo "$OUT" | grep -o "created pod [0-9a-f]*" | awk '{print $3}')
[ -n "$POD" ] || { echo "pod creation failed"; exit 1; }
terminate() { log "terminating $POD"; curl -s -X DELETE -H "Authorization: Bearer $PRIME_API_KEY" "$API/$POD" >/dev/null || true; }
trap 'echo "launcher failed; terminating the pod"; terminate' ERR

# 2) wait for ACTIVE and an ssh target
for _ in $(seq 1 90); do
  S=$(prime --plain pods status "$POD" -o json 2>/dev/null || true)
  ST=$(echo "$S" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('status'))" 2>/dev/null || echo "")
  SSHT=$(echo "$S" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('ssh') or '')" 2>/dev/null || echo "")
  log "pod $POD: $ST $SSHT"
  [ "$ST" = "ACTIVE" ] && [ -n "$SSHT" ] && [ "$SSHT" != "N/A" ] && break
  case "$ST" in TERMINATED|ERROR|FAILED) echo "pod ended during provisioning"; exit 1;; esac
  sleep 20
done
USER_HOST="${SSHT%% *}"; SSH_USER="${USER_HOST%@*}"; IP="${USER_HOST#*@}"; PORT=$(echo "$SSHT" | grep -o -- "-p [0-9]*" | awk '{print $2}'); PORT="${PORT:-22}"

# 3) ssh alias (IPQoS none: without it macOS ssh uploads at ~0.1 MB/s on some routes)
python3 - "$HOST" "$IP" "$SSH_USER" "$PORT" <<'EOF'
import os, re, sys
host, ip, user, port = sys.argv[1:5]
p = os.path.expanduser("~/.ssh/config"); s = open(p).read() if os.path.exists(p) else ""
block = f"Host {host}\n  HostName {ip}\n  User {user}\n  Port {port}\n  IPQoS none\n  IdentityFile ~/.ssh/id_rsa\n  StrictHostKeyChecking accept-new\n  ConnectTimeout 20\n  ServerAliveInterval 30\n"
s2, n = re.subn(rf"Host {host}\n(?:  .*\n)*", block, s)
if n == 0: s2 = s.rstrip("\n") + "\n\n" + block
open(p, "w").write(s2)
EOF
ssh-keygen -R "$IP" >/dev/null 2>&1 || true
for i in $(seq 1 30); do ssh -o BatchMode=yes "$HOST" true 2>/dev/null && break; log "waiting for sshd ($i)"; sleep 10; done
ssh -o BatchMode=yes "$HOST" true

# 4) FIRST: credentials + deadline watchdog on the box, verified
umask 077; printf 'PRIME_API_KEY=%s\nPOD_ID=%s\n' "$PRIME_API_KEY" "$POD" > "$SCRATCH/pod.env"
scp -q "$SCRATCH/pod.env" "$HOST:pod.env"
ssh "$HOST" "chmod 600 ~/pod.env; nohup bash -c 'sleep $((MAX_HOURS * 3600)); source ~/pod.env; echo \"\$(date -u) WATCHDOG TERMINATING\" >> ~/watchdog.log; curl -s -X DELETE -H \"Authorization: Bearer \$PRIME_API_KEY\" $API/\$POD_ID >> ~/watchdog.log 2>&1' </dev/null >~/watchdog.log 2>&1 & disown; sleep 1; pgrep -f 'sleep $((MAX_HOURS * 3600))' >/dev/null && echo 'WATCHDOG ARMED' || echo 'WATCHDOG NOT RUNNING'" | tee "$SCRATCH/watchdog.txt"
grep -q "WATCHDOG ARMED" "$SCRATCH/watchdog.txt" || { echo "watchdog did not start"; exit 1; }
log "deadline watchdog armed: pod $POD terminates after $MAX_HOURS h"

# 5) sync code, tokenizer, scripts, checkpoints (resumable, retried)
sync_retry() { for t in 1 2 3 4 5 6 7 8; do rsync -az --partial --inplace "$@" && return 0; log "rsync retry $t"; sleep 5; done; return 1; }
ssh "$HOST" 'mkdir -p ~/nanochat-dllm ~/.cache/nanochat/tokenizer ~/.cache/nanochat/base_checkpoints/d12_ar ~/.cache/nanochat/base_checkpoints/d12_mdm ~/scripts'
sync_retry --exclude .git --exclude .venv --exclude results/checkpoints --exclude results/logs --exclude 'results/d12/pod_logs' \
    --exclude 'results/d12/base_eval' --exclude 'results/d12/plots' --exclude '__pycache__' --exclude '.env' --exclude '*.png' ./ "$HOST:nanochat-dllm/"
sync_retry ~/.cache/nanochat/tokenizer/ "$HOST:.cache/nanochat/tokenizer/"
sync_retry results/d12/pod_scripts/ "$HOST:scripts/"
ssh "$HOST" 'chmod +x ~/scripts/*.sh'

# 6) start the chain detached, verify by its status file (never by pgrep on the script name)
ssh "$HOST" 'nohup bash ~/scripts/chain_sft_sweep.sh > ~/chain.log 2>&1 < /dev/null & disown; sleep 3; cat ~/chain.status'
ssh "$HOST" 'grep -q "CHAIN START" ~/chain.status' || { echo "chain did not start"; exit 1; }
log "chain started"

# 7) checkpoints (the chain waits for ~/UPLOAD_DONE after setup)
for tag in d12_ar d12_mdm; do
  sync_retry "results/checkpoints/base_checkpoints/$tag/"model_*.pt "results/checkpoints/base_checkpoints/$tag/"meta_*.json "$HOST:.cache/nanochat/base_checkpoints/$tag/"
done
ssh "$HOST" 'ls -la ~/.cache/nanochat/base_checkpoints/*/ | grep model_; touch ~/UPLOAD_DONE; echo UPLOAD_DONE'
trap - ERR
log "launched. Follow with: caffeinate -i bash results/poll_pod.sh $HOST d12 300   (pod $POD, hard deadline ${MAX_HOURS} h)"
echo "$POD" > "$SCRATCH/pod_id"
