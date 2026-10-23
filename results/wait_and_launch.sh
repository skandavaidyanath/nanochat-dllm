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
# cloud_id | MAX_HOURS | extra env for launch_pod.sh
OFFERS=(
  "gpu-h200-sxm_1gpu-16vcpu-200gb|8|DISK_SIZE=1500 VCPUS=16 MEMORY=200"   # Nebius 1xH200 on-demand, $5.40/h, ~6 h expected
  "gpu_1x_h100_pcie|11|"                                                  # Lambda 1xH100 PCIe on-demand, $3.29/h, ~8-9 h expected
  "2H100.80S.80V_SPOT|7|DISK_SIZE=200"                                    # DataCrunch 2xH100 spot, $4.5/h, ~4.5 h expected
  "4H200.141S.176V_SPOT|4|DISK_SIZE=200"                                  # DataCrunch 4xH200 spot, $11.8/h, ~2 h expected
  "1H200.141S.44V_SPOT|10|DISK_SIZE=200"                                  # DataCrunch 1xH200 spot, $2.9/h, ~6 h expected
)
for _ in $(seq 1 "$MAX_POLLS"); do
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
    if echo "$AVAIL" | grep -qx "$CID"; then
      echo "$(date -u +%H:%M:%S) launching on $CID (hard cap $HRS h)"
      env $ENVS CLOUD_ID="$CID" MAX_HOURS="$HRS" SCRATCH="$SCRATCH" bash results/launch_pod.sh 2>&1 | grep -v -i "userwarning\|warnings.warn"
      rc=${PIPESTATUS[0]}
      echo "$(date -u +%H:%M:%S) launcher exit $rc"
      [ "$rc" -eq 0 ] && exit 0
      echo "launch on $CID failed; continuing to poll"
      break
    fi
  done
  echo "$(date -u +%H:%M:%S) no acceptable offer; in stock: $(echo "$AVAIL" | tr '\n' ' ')"
  sleep 60
done
echo "gave up after $MAX_POLLS polls"
exit 1
