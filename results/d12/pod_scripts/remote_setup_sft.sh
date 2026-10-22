#!/bin/bash
# Runs ON the GPU box after the repo, the tokenizer and the d12 base checkpoints have been rsynced
# (~/nanochat-dllm, ~/.cache/nanochat/tokenizer, ~/.cache/nanochat/base_checkpoints/{d12_ar,d12_mdm}).
# Phase 2a needs no pretraining shards: SFT uses the task datasets, which are downloaded here once so the
# two sweep lanes do not race on the download.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH" NANOCHAT_BASE_DIR="$HOME/.cache/nanochat" OMP_NUM_THREADS=1
echo "=== host ==="; hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; nproc; free -g | head -2; df -h ~ | tail -1
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
cd ~/nanochat-dllm
[ -d .venv ] || uv venv
uv sync --extra gpu --group dev
source .venv/bin/activate
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"
echo "=== checkpoints ==="; ls -la "$NANOCHAT_BASE_DIR"/base_checkpoints/*/
echo "=== tokenizer ==="; python -c "
from nanochat.tokenizer import get_tokenizer; t=get_tokenizer(); print('vocab', t.get_vocab_size(), 'mask id', t.encode_special('<|mask|>'))"
echo "=== task data (downloaded once, before the lanes start) ==="
python - <<'EOF'
from tasks.smoltalk import SmolTalk
from tasks.mmlu import MMLU
from tasks.gsm8k import GSM8K
from tasks.arc import ARC
from tasks.humaneval import HumanEval
for t in [SmolTalk(split="train"), SmolTalk(split="test"),
          MMLU(subset="all", split="auxiliary_train"), MMLU(subset="all", split="test"),
          GSM8K(subset="main", split="train"), GSM8K(subset="main", split="test"),
          ARC(subset="ARC-Easy", split="test"), ARC(subset="ARC-Challenge", split="test"), HumanEval()]:
    print(type(t).__name__, len(t))
EOF
echo "=== fa3 ==="; python -c "from nanochat.flash_attention import HAS_FA3, USE_FA3; print('HAS_FA3', HAS_FA3, 'USE_FA3', USE_FA3)"
echo "SETUP DONE"
