#!/bin/bash
# Local side of the pod chain: pull outputs every INTERVAL seconds; when the box reports CHAIN DONE, do a final
# pull and touch ~/ACK on the box so it terminates itself. The box terminates on its own after a timeout anyway,
# so this loop only shortens the wait. Run under caffeinate so the Mac does not idle-sleep:
#   caffeinate -i bash results/poll_pod.sh prime-sft d12 300
HOST="${1:?ssh host alias}"
PHASE="${2:-d12}"
INTERVAL="${3:-300}"
cd "$(dirname "$0")/.."
fails=0
while true; do
  if ssh -o ConnectTimeout=20 "$HOST" true 2>/dev/null; then
    fails=0
    bash results/sync_pod.sh "$HOST" "$PHASE" >/dev/null 2>&1 || true
    st=$(ssh "$HOST" 'tail -1 ~/chain.status 2>/dev/null')
    echo "$(date -u +%H:%M) $st"
    if echo "$st" | grep -q "CHAIN DONE\|TERMINATING"; then
      bash results/sync_pod.sh "$HOST" "$PHASE"
      ssh "$HOST" 'touch ~/ACK'
      echo "ACK sent; the pod terminates itself within a minute"
      exit 0
    fi
  else
    fails=$((fails + 1))
    echo "$(date -u +%H:%M) ssh failed ($fails)"
    if [ "$fails" -ge 6 ]; then echo "box unreachable for 30 min (spot preempted, or it already terminated)"; exit 2; fi
  fi
  sleep "$INTERVAL"
done
