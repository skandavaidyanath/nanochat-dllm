#!/bin/bash
# Poll Prime Intellect availability until one of the acceptable offers is in stock, then launch the phase-2a chain
# on it with results/launch_pod.sh (watchdog first, pod terminated on any launcher failure). Offers are listed in
# preference order with the hard deadline (hours) that keeps the worst case inside the budget, and the resource
# arguments the offer needs so the CLI does not prompt. Exits 0 once a launch succeeds.
# Usage: bash results/wait_and_launch.sh [max_polls]   (one poll per minute)
set -u
cd "$(dirname "$0")/.."
set -a; source .env; set +a
export PRIME_API_KEY="${PRIME_API_KEY:-$PRIME_INTELLECT_KEY}"
PY=$(head -1 "$(command -v prime)" | sed 's/^#!//')   # the prime CLI's own python, which has prime_cli installed
MAX_POLLS="${1:-360}"
SCRATCH="${SCRATCH:-$(mktemp -d)}"
# cloud_id | MAX_HOURS | extra env for launch_pod.sh. PROFILE=sweep (multi-GPU ok) or phase2b (single GPU, ~$20 budget).
PROFILE="${PROFILE:-sweep}"
if [ "$PROFILE" = "phase2b" ]; then
  OFFERS=(
    "gpu-h200-sxm_1gpu-16vcpu-200gb|3.5|DISK_SIZE=1500 VCPUS=16 MEMORY=200"   # Nebius 1xH200 on-demand $5.40/h, ~2 h expected, cap $19
    "gpu_1x_h100_pcie|5|"                                                    # Lambda 1xH100 PCIe $3.29/h, ~3 h expected, cap $16.5
    "1H200.141S.44V_SPOT|6|DISK_SIZE=200"                                    # DataCrunch 1xH200 spot $2.9/h, cap $17.5
  )
else
  OFFERS=(
    "gpu-h200-sxm_1gpu-16vcpu-200gb|8|DISK_SIZE=1500 VCPUS=16 MEMORY=200"   # Nebius 1xH200 on-demand, $5.40/h, ~6 h expected
    "gpu_1x_h100_pcie|11|"                                                  # Lambda 1xH100 PCIe on-demand, $3.29/h, ~8-9 h expected
    "2H100.80S.80V_SPOT|7|DISK_SIZE=200"                                    # DataCrunch 2xH100 spot, $4.5/h, ~4.5 h expected
    "4H200.141S.176V_SPOT|4|DISK_SIZE=200"                                  # DataCrunch 4xH200 spot, $11.8/h, ~2 h expected
    "1H200.141S.44V_SPOT|10|DISK_SIZE=200"                                  # DataCrunch 1xH200 spot, $2.9/h, ~6 h expected
  )
fi
COOLDOWN=""   # "cloud_id:poll_until" entries for offers whose launch just failed (e.g. provider HTTP 500), skipped for 5 polls
poll=0
for _ in $(seq 1 "$MAX_POLLS"); do
  poll=$((poll + 1))
  AVAIL=$("$PY" - <<'EOF' 2>/dev/null
from prime_cli.api.client import APIClient
from prime_cli.api.availability import AvailabilityClient
avail = AvailabilityClient(APIClient()).get()
for gtype in ("H100_80GB", "H200_141GB"):
    for g in avail.get(gtype, []):
        d = g.model_dump() if hasattr(g, "model_dump") else dict(g.__dict__)
        if d.get("stock_status") == "Available":
            print(d["cloud_id"])
EOF
)
  for o in "${OFFERS[@]}"; do
    IFS='|' read -r CID HRS ENVS <<< "$o"
    until_poll=$(echo "$COOLDOWN" | tr ' ' '\n' | grep "^$CID:" | tail -1 | cut -d: -f2)
    if [ -n "$until_poll" ] && [ "$poll" -le "$until_poll" ]; then continue; fi
    if echo "$AVAIL" | grep -qx "$CID"; then
      echo "$(date -u +%H:%M:%S) launching on $CID (hard cap $HRS h)"
      env $ENVS CLOUD_ID="$CID" MAX_HOURS="$HRS" SCRATCH="$SCRATCH" CHAIN="${CHAIN:-chain_sft_sweep.sh}" NAME="${NAME:-d12-sft-sweep}" bash results/launch_pod.sh 2>&1 | grep -v -i "userwarning\|warnings.warn"
      rc=${PIPESTATUS[0]}
      echo "$(date -u +%H:%M:%S) launcher exit $rc"
      [ "$rc" -eq 0 ] && exit 0
      COOLDOWN="$COOLDOWN $CID:$((poll + 5))"
      echo "launch on $CID failed; skipping it for 5 polls, trying the next offer"
      break
    fi
  done
  echo "$(date -u +%H:%M:%S) no acceptable offer; in stock: $(echo "$AVAIL" | tr '\n' ' ')"
  sleep 60
done
echo "gave up after $MAX_POLLS polls"
exit 1
