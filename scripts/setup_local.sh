#!/usr/bin/env bash
# Set up the free, open-weights Wan2.2 model for local generation.
#
#   bash scripts/setup_local.sh            # TI2V-5B (~24 GB VRAM, ~35 GB download)
#   bash scripts/setup_local.sh a14b       # T2V-A14B + I2V-A14B (~80 GB VRAM, ~120 GB download)
#
# Needs: Linux, an NVIDIA GPU with a working CUDA driver, Python 3.10+, git.
# Run it inside a virtualenv; it installs PyTorch and Wan2.2's requirements.
set -euo pipefail

PRESET="${1:-ti2v-5B}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WAN_DIR="${WAN_REPO_DIR:-$ROOT/third_party/Wan2.2}"
MODELS_DIR="${WAN_MODELS_DIR:-$ROOT/models}"
PY="${PYTHON:-python3}"

case "$PRESET" in
  ti2v-5B) REPOS=(Wan-AI/Wan2.2-TI2V-5B) ;;
  a14b)    REPOS=(Wan-AI/Wan2.2-T2V-A14B Wan-AI/Wan2.2-I2V-A14B) ;;
  *) echo "unknown preset '$PRESET' (use ti2v-5B or a14b)" >&2; exit 1 ;;
esac

if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
  echo "WARNING: nvidia-smi not found. Wan2.2 needs an NVIDIA GPU with CUDA." >&2
fi

echo "==> Cloning Wan2.2 into $WAN_DIR"
if [ ! -d "$WAN_DIR/.git" ]; then
  git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git "$WAN_DIR"
else
  git -C "$WAN_DIR" pull --ff-only || true
fi

echo "==> Installing Python dependencies"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$ROOT/requirements.txt"
# flash_attn compiles against the installed torch, so install everything else first.
grep -iv '^flash[_-]attn' "$WAN_DIR/requirements.txt" > "$WAN_DIR/requirements.noflash.txt"
"$PY" -m pip install -r "$WAN_DIR/requirements.noflash.txt"
"$PY" -m pip install "huggingface_hub[cli]"
if ! "$PY" -m pip install flash-attn --no-build-isolation; then
  echo "WARNING: flash-attn failed to install, and Wan2.2 requires it." >&2
  echo "         Install a prebuilt wheel matching your torch/CUDA/Python versions from" >&2
  echo "         https://github.com/Dao-AILab/flash-attention/releases" >&2
fi

echo "==> Downloading model weights into $MODELS_DIR"
for repo in "${REPOS[@]}"; do
  "$PY" -c "from huggingface_hub import snapshot_download as d; d('$repo', local_dir='$MODELS_DIR/${repo#*/}')"
done

cat <<EOF

Done. Try a 30-second video:

  python -m wan_unlimited --preset $PRESET -p "A red fox trotting through fresh snow at dawn" -d 30

EOF
