#!/bin/bash
# Pull run logs, eval CSVs and stage logs from a GPU box into results/logs and results/<phase>/.
# Usage: bash results/sync_logs.sh <ssh-host> [phase]      e.g. bash results/sync_logs.sh prime-d12 d12
set -euo pipefail
HOST="${1:?ssh host alias}"
PHASE="${2:-d12}"
cd "$(dirname "$0")/.."
mkdir -p results/logs results/$PHASE
rsync -az --include='*/' --include='metrics.jsonl' --include='config.json' --exclude='*' "$HOST:.cache/nanochat/logs/" results/logs/
rsync -az "$HOST:.cache/nanochat/base_eval/" results/$PHASE/base_eval/ 2>/dev/null || true
rsync -az --include='*.log' --exclude='*' "$HOST:.cache/nanochat/logs/" results/$PHASE/stage_logs/ 2>/dev/null || true
rsync -az "$HOST:sweep.log" results/$PHASE/sweep.log 2>/dev/null || true
echo "synced from $HOST:"; find results/logs -name metrics.jsonl | sed 's|results/logs/||' | sort
