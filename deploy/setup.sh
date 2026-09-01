#!/usr/bin/env bash
# Rebuild the whole stack on a bare Ubuntu 24.04 GPU container.
#
# Written for rented GPU boxes that come back as a fresh container each time.
# /workspace usually survives a restart (models, code, outputs) but the venv and
# system packages do not, so this is safe to re-run: it skips what is present.
#
# Takes about 10 minutes cold (18 GB of weights is the long pole) or about 4
# minutes if /workspace still has the models.
set -euo pipefail

ROOT=${ROOT:-${TRYON_ROOT:-/workspace/tryon}}
VENV=$ROOT/venv
LOG=$ROOT/logs
mkdir -p "$ROOT" "$LOG" "$ROOT/hf-cache" "$ROOT/models"/{unet,text_encoders,vae,loras}
cd "$ROOT"

export HF_HOME=$ROOT/hf-cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

say() { echo "[$(date +%H:%M:%S)] $*"; }

say "creating venv"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q -U pip wheel setuptools
"$VENV/bin/pip" install -q "huggingface_hub[hf_transfer]"

# Model pull runs first and in the background so the transfer overlaps the
# CPU-bound pip installs below.
cat > "$ROOT/pull_models.sh" <<'PULL'
set -euo pipefail
ROOT=${ROOT:-${TRYON_ROOT:-/workspace/tryon}}
export HF_HOME=$ROOT/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1
HF=$ROOT/venv/bin/hf
$HF download unsloth/FLUX.2-klein-9B-GGUF flux-2-klein-9b-Q8_0.gguf \
    --local-dir "$ROOT/models/unet"
$HF download Comfy-Org/vae-text-encorder-for-flux-klein-9b \
    split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors \
    split_files/vae/flux2-vae.safetensors --local-dir "$ROOT/models/dl"
mv -f "$ROOT/models/dl/split_files/text_encoders/"*.safetensors "$ROOT/models/text_encoders/"
mv -f "$ROOT/models/dl/split_files/vae/"*.safetensors "$ROOT/models/vae/"
$HF download fal/flux-klein-9b-virtual-tryon-lora flux-klein-tryon-comfy.safetensors \
    --local-dir "$ROOT/models/loras"
# The guardrail's vision model. Loaded at 4-bit at run time; this is the bf16 repo.
# Note: each --exclude takes ONE pattern. Passing two patterns to one flag makes
# the CLI treat the second as a filename and the download 404s.
$HF download Qwen/Qwen3-VL-4B-Instruct
echo MODELS_DONE
PULL
nohup bash "$ROOT/pull_models.sh" > "$LOG/models.log" 2>&1 &
say "model pull running in background (pid $!)"

say "installing torch cu128"
"$VENV/bin/pip" install -q torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
"$VENV/bin/python" - <<'PY'
import torch
cap = torch.cuda.get_device_capability()
print(f"torch {torch.__version__} cuda {torch.version.cuda} "
      f"{torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")
assert torch.cuda.is_available(), "no CUDA device visible to torch"
PY

say "cloning ComfyUI"
[ -d "$ROOT/ComfyUI" ] || git clone --depth 1 https://github.com/comfyanonymous/ComfyUI "$ROOT/ComfyUI"
[ -d "$ROOT/ComfyUI/custom_nodes/ComfyUI-GGUF" ] || \
    git clone --depth 1 https://github.com/city96/ComfyUI-GGUF \
        "$ROOT/ComfyUI/custom_nodes/ComfyUI-GGUF"

# ComfyUI's requirements list torch. Installing them unfiltered pulls the default
# CUDA build over the cu128 one above. Filter by exact distribution name — an
# earlier startswith("torch") also stripped torchsde, which ComfyUI needs.
"$VENV/bin/python" - <<'PY'
import re, pathlib, os
root = os.environ.get("ROOT", "${TRYON_ROOT:-/workspace/tryon}")
src = pathlib.Path(root) / "ComfyUI/requirements.txt"
held = {"torch", "torchvision", "torchaudio"}
keep = [l for l in src.read_text().splitlines()
        if re.split(r"[<>=!~\[ ]", l.strip(), 1)[0].lower() not in held]
pathlib.Path("/tmp/comfy-reqs.txt").write_text("\n".join(keep) + "\n")
print("held back:", sorted(held), "| kept", len(keep), "lines")
PY
"$VENV/bin/pip" install -q -r /tmp/comfy-reqs.txt
"$VENV/bin/pip" install -q -r "$ROOT/ComfyUI/custom_nodes/ComfyUI-GGUF/requirements.txt"

say "installing app deps"
# numpy must be 2.x: ComfyUI uses np.long, which does not exist in 1.26.
"$VENV/bin/pip" install -q fastapi "uvicorn[standard]" python-multipart httpx \
    pillow transformers accelerate bitsandbytes qwen-vl-utils \
    pydantic-settings websockets "numpy>=2.1"
# rembg is not installed by default: background removal is off unless a caller
# asks for it. Install it only if you turn remove_background on.
#   "$VENV/bin/pip" install -q rembg onnxruntime

link_models() {
  mkdir -p "$ROOT/ComfyUI/models"/{unet,text_encoders,vae,loras}
  for d in unet text_encoders vae loras; do
    for f in "$ROOT/models/$d"/*; do
      [ -e "$f" ] && ln -sf "$f" "$ROOT/ComfyUI/models/$d/$(basename "$f")"
    done
  done
}
link_models

say "waiting on model pull"
while ! grep -q MODELS_DONE "$LOG/models.log" 2>/dev/null; do
  pgrep -f pull_models.sh > /dev/null || { say "pull exited early — see $LOG/models.log"; break; }
  sleep 10
done
link_models      # again: the pull may have finished after the first pass

say "done"
ls -la "$ROOT/ComfyUI/models"/{unet,text_encoders,vae,loras}
