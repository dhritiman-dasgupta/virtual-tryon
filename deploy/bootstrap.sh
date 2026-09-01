#!/usr/bin/env bash
# Bring the box back after a stop. One command, or set it as the instance's
# startup command and it needs nothing at all:
#
#     bash /workspace/swift-teal-stoat/bootstrap.sh
#
# /workspace persists across a stop; the container filesystem does not. So the
# SSH key, the tunnel credentials and the models all live on the volume and are
# restored from there.
#
# The tunnel is a NAMED Cloudflare tunnel, not a quick tunnel. A quick tunnel
# gets a new random hostname on every start, which meant re-pasting a URL into
# the page each session. This one is always https://tryon.example.com.
set -uo pipefail

ROOT=${ROOT:-/workspace/swift-teal-stoat}
cd "$ROOT" 2>/dev/null || { echo "no $ROOT - volume is empty"; exit 1; }
mkdir -p logs
export HF_HOME=$ROOT/hf-cache

say() { echo "[$(date +%H:%M:%S)] $*"; }

# ------------------------------------------------------------------ ssh key
if [ -f "$ROOT/authorized_keys" ]; then
  mkdir -p /root/.ssh && chmod 700 /root/.ssh
  # Append and de-duplicate, never overwrite: the provider injects its own key
  # and clobbering it would lock the web console out.
  cat "$ROOT/authorized_keys" >> /root/.ssh/authorized_keys 2>/dev/null || true
  sort -u /root/.ssh/authorized_keys -o /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
  say "ssh keys restored ($(wc -l < /root/.ssh/authorized_keys))"
fi
pgrep -x sshd >/dev/null 2>&1 || { mkdir -p /run/sshd; /usr/sbin/sshd 2>/dev/null; }

# -------------------------------------------------------------------- stack
if [ ! -x "$ROOT/venv/bin/python" ]; then
  say "venv missing - rebuilding (~4 min, models are already on the volume)"
  bash "$ROOT/setup.sh" > logs/setup.log 2>&1 || {
    say "rebuild FAILED - see logs/setup.log"; exit 1; }
fi

if ! curl -sf -m 4 http://127.0.0.1:8000/healthz >/dev/null 2>&1; then
  say "starting API"
  pkill -f '[u]vicorn app.main' 2>/dev/null; sleep 2
  setsid nohup ./start_api.sh > logs/api.log 2>&1 < /dev/null &
  for _ in $(seq 45); do
    curl -sf -m 4 http://127.0.0.1:8000/healthz >/dev/null 2>&1 && break
    sleep 5
  done
fi
curl -sf -m 4 http://127.0.0.1:8000/healthz >/dev/null 2>&1 \
  && say "API up" || { say "API did not start - see logs/api.log"; exit 1; }

# ------------------------------------------------------------------- tunnel
if [ ! -x ./cloudflared ]; then
  say "fetching cloudflared"
  curl -sL -o cloudflared \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x cloudflared
fi

if [ -f "$ROOT/tunnel_token" ]; then
  if ! pgrep -f '[c]loudflared tunnel run' >/dev/null; then
    say "starting named tunnel"
    setsid nohup ./cloudflared tunnel --no-autoupdate run \
      --token "$(cat "$ROOT/tunnel_token")" > logs/tunnel.log 2>&1 < /dev/null &
    sleep 8
  fi
  pgrep -f '[c]loudflared tunnel run' >/dev/null \
    && say "tunnel up" || say "tunnel failed - see logs/tunnel.log"
else
  say "no $ROOT/tunnel_token - the fixed URL will not work"
fi

# ------------------------------------------------------------------ warm-up
# The first generation after a restart loads 18 GB from disk and takes 40-90s;
# every one after is about 6s. Pay it here rather than on the user's first real
# try-on, where a minute of silence reads as a hung page.
if [ "${WARMUP:-1}" = "1" ]; then
  P=$(ls "$ROOT"/inputs/models/*.jpeg 2>/dev/null | head -1)
  G=$(ls "$ROOT"/inputs/fg/*.jpeg 2>/dev/null | head -1)
  if [ -n "$P" ] && [ -n "$G" ]; then
    say "warming the generator (40-90s, one time)"
    curl -s -m 600 -X POST \
      "http://127.0.0.1:8000/v1/tryon?wait=true&wait_timeout=580" \
      -F "person=@$P" -F "garment=@$G" -F "guardrail=false" \
      -F "megapixels=0.5" -o /dev/null
    say "generator warm - images now take about 6s"
  fi
fi

echo
echo "======================================================================"
echo "  Page:  https://<your-bucket>.s3.amazonaws.com/demo/index.html"
echo "  API :  https://tryon.example.com   (fixed, never changes)"
echo "  Login: demo"
echo "======================================================================"
