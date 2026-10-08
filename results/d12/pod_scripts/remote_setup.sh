#!/bin/bash
# Runs ON the GPU box after the repo and tokenizer have been rsynced to ~/nanochat-dllm and ~/.cache/nanochat/tokenizer
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export OMP_NUM_THREADS=1
echo "=== host ==="; hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; nproc; free -g | head -2; df -h ~ | tail -1
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
cd ~/nanochat-dllm
[ -d .venv ] || uv venv
uv sync --extra gpu --group dev
source .venv/bin/activate
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "=== data ==="; python -m nanochat.dataset -n 8
ls "$NANOCHAT_BASE_DIR/tokenizer" && python -c "
from nanochat.tokenizer import get_tokenizer; t=get_tokenizer(); print('tokenizer vocab', t.get_vocab_size(), 'mask id', t.encode_special('<|mask|>'))"
echo "=== tests ==="; python -m pytest tests -q -p no:cacheprovider 2>&1 | tail -3
echo "=== fa3 ==="; python -c "from nanochat.flash_attention import HAS_FA3, USE_FA3; print('HAS_FA3', HAS_FA3, 'USE_FA3', USE_FA3)"
echo "SETUP DONE"
