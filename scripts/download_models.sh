#!/usr/bin/env bash
# Fetch both model sets: the generator (~18 GB) and the vision model (~16 GB).
#
# huggingface_hub, not aria2: HF's Xet CDN returns 403 for aria2's parallel
# range requests because each signed URL carries a byte-range condition in its
# policy. Every download is size-verified afterwards — a 404 writes a small
# HTML body that a downloader reports as success.
set -Eeuo pipefail
COMFY="${COMFY_ROOT:-/opt/ComfyUI}"
PY="${PYTHON:-python3}"
log(){ printf '\033[1;34m==>\033[0m %s\n' "$*"; }
trap 'echo "ERROR at line $LINENO" >&2; exit 1' ERR

log "generator weights -> ${COMFY}/models"
COMFY_ROOT="$COMFY" "$PY" - <<'PYEOF'
import os, shutil, sys
from huggingface_hub import hf_hub_download
M = os.path.join(os.environ["COMFY_ROOT"], "models")
JOBS = [
 ("unsloth/FLUX.2-klein-9B-GGUF","flux-2-klein-9b-Q8_0.gguf","unet","flux-2-klein-9b-Q8_0.gguf",6000),
 ("Comfy-Org/flux2-klein-9B","split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors","clip","qwen_3_8b_fp8mixed.safetensors",4000),
 ("Comfy-Org/flux2-dev","split_files/vae/flux2-vae.safetensors","vae","flux2-vae.safetensors",50),
 ("fal/flux-klein-9b-virtual-tryon-lora","flux-klein-tryon-comfy.safetensors","loras","flux-klein-tryon-comfy.safetensors",10),
]
for repo, remote, sub, fn, min_mb in JOBS:
    dest = os.path.join(M, sub, fn); os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest)//1048576 >= min_mb:
        print(f"[skip] {fn}", flush=True); continue
    print(f"[get ] {fn}", flush=True)
    p = hf_hub_download(repo_id=repo, filename=remote)
    shutil.copyfile(os.path.realpath(p), dest)   # hf returns a symlink into the blob store
    mb = os.path.getsize(dest)//1048576
    if mb < min_mb: sys.exit(f"!! {fn} is {mb} MB, expected >= {min_mb} MB")
    print(f"[ok  ] {fn} {mb} MB", flush=True)
PYEOF

log "vision model -> \$HF_HOME"
"$PY" - <<'PYEOF'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct",
                      allow_patterns=["*.json","*.safetensors","*.txt","*.py","tokenizer*","*.model"])
print("vision model at", p)
PYEOF
log "done"
