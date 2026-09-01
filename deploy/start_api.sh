#!/usr/bin/env bash
# Start the try-on API, which supervises ComfyUI itself.
#
# VRAM: leave RESERVE_VRAM empty when nothing else shares the card. Set it to
# 4.0 only when the guardrail's vision model is resident alongside ComfyUI, and
# never higher — on the 5090 an 11.0 reserve (sized for an 8-bit VLM) starved
# ComfyUI, pushed the 9.7 GB UNet into offload, and turned an 8 s generation
# into 215 s. Keep the VLM at 4-bit and the reserve at 4.0.
set -euo pipefail
ROOT=${ROOT:-${TRYON_ROOT:-/workspace/tryon}}
cd "$ROOT"

export COMFY_ROOT=$ROOT/ComfyUI
export COMFY_MANAGE=true
export COMFY_EXTRA_ARGS=${RESERVE_VRAM:+--reserve-vram $RESERVE_VRAM}
export OUTPUT_DIR=$ROOT/outputs
export HF_HOME=$ROOT/hf-cache
export PYTHONUNBUFFERED=1

exec ./venv/bin/uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
