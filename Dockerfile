# Virtual try-on API — generator + resident vision guardrail.
#
# CUDA 12.8 is not optional. Blackwell (RTX 5090) is sm_120, and a runtime built
# against an older toolkit either refuses to compile kernels or silently falls
# back to something slow. The build asserts the capability rather than letting
# that surface as a mysterious performance problem in production.
#
# The ~34 GB of weights are NOT baked into the image. They live on a volume, so
# rebuilding the app does not refetch them and the image stays a few GB.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    COMFY_ROOT=/opt/ComfyUI \
    HF_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip git curl ca-certificates \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------- torch
# cu128 wheels, from PyTorch's index rather than PyPI: the default PyPI wheel
# does not carry sm_120 kernels.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# ------------------------------------------------------------- ComfyUI
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI ${COMFY_ROOT} \
 && git clone --depth 1 https://github.com/city96/ComfyUI-GGUF \
      ${COMFY_ROOT}/custom_nodes/ComfyUI-GGUF

# ComfyUI pins a bare `torch`, which pip resolves to a CPU wheel and installs
# over the cu128 build. Hold back exactly the three CUDA distributions — a
# prefix match also swallows torchsde, which ComfyUI imports at startup.
RUN python - <<'PY' \
 && pip install -r ${COMFY_ROOT}/requirements.nocuda.txt gguf sentencepiece protobuf
import os, pathlib, re
HOLD = {"torch", "torchvision", "torchaudio"}
def name(line):
    s = line.strip()
    return None if not s or s.startswith("#") else re.split(r"[\s=<>!~;\[]", s, 1)[0].lower()
root = pathlib.Path(os.environ["COMFY_ROOT"])
req = (root / "requirements.txt").read_text().splitlines()
keep = [l for l in req if name(l) and name(l) not in HOLD]
(root / "requirements.nocuda.txt").write_text("\n".join(keep) + "\n")
PY

# ------------------------------------------------- app + vision deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
      "transformers>=4.57" accelerate bitsandbytes qwen-vl-utils \
      insightface onnxruntime opencv-python-headless

# Fail the build, not the first request, if the toolchain is wrong.
RUN python -c "import torch, torchsde, transformers, bitsandbytes; \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
assert torch.version.cuda and torch.version.cuda.startswith('12.8'), torch.version.cuda"

WORKDIR /app
COPY . /app

# Weights and cache are volumes; nothing large belongs in a layer.
VOLUME ["/opt/ComfyUI/models", "/models", "/app/outputs", "/app/cache"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/readyz || exit 1

# Runtime check: the image may be correct while the host driver or the
# --gpus flag is not.
COPY <<'ENTRY' /entrypoint.sh
#!/usr/bin/env bash
set -Eeuo pipefail
python - <<'PY'
import torch, sys
if not torch.cuda.is_available():
    sys.exit("no CUDA device visible — did you pass --gpus all?")
cap = torch.cuda.get_device_capability()
print(f"GPU {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}  "
      f"torch {torch.__version__}/{torch.version.cuda}")
if cap[0] < 12:
    print(f"warning: built for Blackwell sm_120, found sm_{cap[0]}{cap[1]}",
          file=sys.stderr)
PY
exec "$@"
ENTRY
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
