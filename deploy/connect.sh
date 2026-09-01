#!/usr/bin/env bash
# Probe the GPU box and report what survived a restart.
#
#   HOST=<HOST> PORT=<PORT> ./deploy/connect.sh
#
# The rented instances come back on a new port with a fresh container, so the
# two things worth knowing immediately are whether the key still works and
# whether /workspace kept the 18 GB of weights.
set -uo pipefail
: "${HOST:?set HOST}" ; : "${PORT:?set PORT}"
KEY=${KEY:-~/.ssh/id_ed25519}
ROOT=${ROOT:-${TRYON_ROOT:-/workspace/tryon}}

if ! ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
        -p "$PORT" -i "$KEY" "root@$HOST" true 2>/dev/null; then
  echo "cannot reach root@$HOST:$PORT"
  echo
  echo "if it refuses the key, paste this into the instance's web console:"
  echo "  mkdir -p ~/.ssh && chmod 700 ~/.ssh"
  echo "  echo '$(cat "${KEY}.pub" 2>/dev/null)' >> ~/.ssh/authorized_keys"
  echo "  chmod 600 ~/.ssh/authorized_keys"
  exit 1
fi

ssh -p "$PORT" -i "$KEY" "root@$HOST" bash -s <<EOSH
ROOT=$ROOT
echo "host      \$(hostname)"
echo "gpu       \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "vram used \$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo "workspace \$(du -sh \$ROOT 2>/dev/null | cut -f1)"
echo "models    \$(du -sh \$ROOT/models 2>/dev/null | cut -f1)"
test -x \$ROOT/venv/bin/python && echo "venv      present" || echo "venv      MISSING - run deploy/setup.sh"
test -d \$ROOT/ComfyUI && echo "comfyui   present" || echo "comfyui   MISSING"
curl -sf -m 4 http://127.0.0.1:8000/healthz >/dev/null \
  && echo "api       up" || echo "api       down - run deploy/start_api.sh"
EOSH
