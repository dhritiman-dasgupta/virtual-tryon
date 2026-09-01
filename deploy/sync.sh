#!/usr/bin/env bash
# Push local code to the GPU box and (re)start the API.
#
#   HOST=<HOST> PORT=<PORT> ./deploy/sync.sh
#
# Only app/, pipeline/ and workflows/ go up — models and outputs live on the
# box's /workspace volume and must not be overwritten from a laptop.
set -euo pipefail
: "${HOST:?set HOST}" ; : "${PORT:?set PORT}"
KEY=${KEY:-~/.ssh/id_ed25519}
ROOT=${ROOT:-/workspace/swift-teal-stoat}
HERE=$(cd "$(dirname "$0")/.." && pwd)

ssh -p "$PORT" -i "$KEY" "root@$HOST" "mkdir -p $ROOT/{pipeline,app,workflows,pairs,logs}"

# Fresh containers do not always have rsync. Fall back to tar over ssh rather
# than failing the deploy - the payload is a few hundred KB of source either
# way, so the only thing rsync buys here is not re-sending unchanged files.
if ! ssh -p "$PORT" -i "$KEY" "root@$HOST" 'command -v rsync' >/dev/null 2>&1; then
  echo "rsync missing on the box; using tar"
  tar czf - -C "$HERE" --exclude __pycache__ app pipeline workflows \
      deploy/setup.sh deploy/start_api.sh \
    | ssh -p "$PORT" -i "$KEY" "root@$HOST" "tar xzf - -C $ROOT"
  if [ "${RESTART:-1}" = "1" ]; then
    ssh -p "$PORT" -i "$KEY" "root@$HOST" \
      "cp -f $ROOT/deploy/start_api.sh $ROOT/ 2>/dev/null; \
       pkill -f 'uvicorn app.main' || true; sleep 1; cd $ROOT && \
       RESERVE_VRAM=${RESERVE_VRAM:-} setsid nohup ./start_api.sh > logs/api.log 2>&1 < /dev/null & disown"
    echo "restarting API"
  fi
  exit 0
fi
# -c compares by checksum: mtimes are meaningless across a laptop and a container.
rsync -az -c --delete -e "ssh -p $PORT -i $KEY" \
      --exclude __pycache__ \
      "$HERE/app/" "root@$HOST:$ROOT/app/"
rsync -az -c -e "ssh -p $PORT -i $KEY" --exclude __pycache__ \
      "$HERE/pipeline/" "root@$HOST:$ROOT/pipeline/"
rsync -az -c -e "ssh -p $PORT -i $KEY" \
      "$HERE/workflows/" "root@$HOST:$ROOT/workflows/"
rsync -az -c -e "ssh -p $PORT -i $KEY" \
      "$HERE/deploy/setup.sh" "$HERE/deploy/start_api.sh" "root@$HOST:$ROOT/"

if [ "${RESTART:-1}" = "1" ]; then
  ssh -p "$PORT" -i "$KEY" "root@$HOST" \
    "pkill -f 'uvicorn app.main' || true; sleep 1; cd $ROOT && \
     RESERVE_VRAM=${RESERVE_VRAM:-} setsid nohup ./start_api.sh > logs/api.log 2>&1 < /dev/null & disown"
  echo "restarting API; waiting for health"
  for _ in $(seq 60); do
    if ssh -o ConnectTimeout=8 -p "$PORT" -i "$KEY" "root@$HOST" \
         'curl -sf -m 4 http://127.0.0.1:8000/healthz >/dev/null' 2>/dev/null; then
      echo "API up"; exit 0
    fi
    sleep 5
  done
  echo "API did not come up — check $ROOT/logs/api.log" >&2; exit 1
fi
