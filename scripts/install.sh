#!/usr/bin/env bash
# Install ComfyUI + the one custom node pack this workflow needs.
#
# Unlike the Colab notebook this came from, every step is checked. A failed
# install fails loudly here instead of surfacing later as "backend failed to
# start" with no explanation.
set -Eeuo pipefail

COMFY_ROOT="${COMFY_ROOT:-/opt/ComfyUI}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
PY="${PYTHON:-python3}"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

trap 'die "install.sh failed at line $LINENO"' ERR

command -v git >/dev/null || die "git is required"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
  || die "Python 3.10+ is required (found $($PY --version 2>&1))"

log "Installing ComfyUI into ${COMFY_ROOT}"
if [ -d "${COMFY_ROOT}/.git" ]; then
  log "Already present, pulling latest"
  git -C "${COMFY_ROOT}" pull --ff-only
else
  mkdir -p "$(dirname "${COMFY_ROOT}")"
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "${COMFY_ROOT}"
fi

log "Installing PyTorch (${TORCH_INDEX})"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install torch torchvision torchaudio --index-url "${TORCH_INDEX}"

log "Installing ComfyUI requirements"
"$PY" -m pip install -r "${COMFY_ROOT}/requirements.txt"

# The upstream workflow pulled in rgthree, LayerStyle and Comfyroll purely for
# debug/preview nodes. Those were stripped from workflows/tryon_api.json, so
# ComfyUI-GGUF (for UnetLoaderGGUF) is the only pack still required.
log "Installing ComfyUI-GGUF custom node"
NODES="${COMFY_ROOT}/custom_nodes"
mkdir -p "${NODES}"
if [ ! -d "${NODES}/ComfyUI-GGUF/.git" ]; then
  git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git "${NODES}/ComfyUI-GGUF"
fi
"$PY" -m pip install gguf
[ -f "${NODES}/ComfyUI-GGUF/requirements.txt" ] && \
  "$PY" -m pip install -r "${NODES}/ComfyUI-GGUF/requirements.txt"

log "Installing API requirements"
"$PY" -m pip install -r "$(dirname "$0")/../requirements.txt"

mkdir -p "${COMFY_ROOT}"/models/{unet,clip,vae,loras}

log "Verifying CUDA is visible to torch"
"$PY" - <<'PYEOF'
import torch
if not torch.cuda.is_available():
    print("  WARNING: torch.cuda.is_available() is False — CPU-only inference "
          "will be unusably slow. Check your driver and CUDA install.")
else:
    n = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  OK: {n} ({vram:.1f} GB VRAM)")
    if vram < 16:
        print("  NOTE: under 16 GB — set COMFY_EXTRA_ARGS=--lowvram in .env")
PYEOF

log "Done. Next: scripts/download_models.sh"
