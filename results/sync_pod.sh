#!/bin/bash
# Pull phase-2a outputs from the pod: run metrics (results/logs), stage logs, probe JSONs and the chain logs
# (results/<phase>/sft_sweep). Usage: bash results/sync_pod.sh <ssh-host> [phase]
set -euo pipefail
HOST="${1:?ssh host alias}"
PHASE="${2:-d12}"
cd "$(dirname "$0")/.."
DEST="results/$PHASE/sft_sweep"
mkdir -p results/logs "$DEST/probes" "$DEST/stage_logs"
rsync -az --include='*/' --include='metrics.jsonl' --include='config.json' --exclude='*' "$HOST:.cache/nanochat/logs/" results/logs/
rsync -az --include='*.log' --exclude='*' "$HOST:.cache/nanochat/logs/" "$DEST/stage_logs/" 2>/dev/null || true
rsync -az "$HOST:.cache/nanochat/probes/" "$DEST/probes/" 2>/dev/null || true
rsync -az --include='*.log' --include='chain.status' --exclude='*' "$HOST:~/" "$DEST/" 2>/dev/null || true
echo "synced from $HOST at $(date -u +%H:%M:%S):"; tail -2 "$DEST/chain.status" 2>/dev/null || true
