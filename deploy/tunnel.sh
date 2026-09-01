#!/usr/bin/env bash
# Open a public tunnel to the API and print its URL.
#
# The provider's gateway blocks port 8000 from outside, so a browser reaches
# the API through cloudflared rather than directly.
#
# A script rather than an inline background command because '&' inside an ssh
# 'cd X && ...' chain binds to the whole chain, which silently swallowed
# several restarts before this existed. The log is truncated first so the URL
# grep cannot return a dead one from a previous session.
set -uo pipefail
ROOT=${ROOT:-/workspace/swift-teal-stoat}
cd "$ROOT"

if [ ! -x ./cloudflared ]; then
  echo "downloading cloudflared" >&2
  curl -sL -o cloudflared \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x cloudflared
fi

pkill -f '[c]loudflared tunnel' 2>/dev/null
sleep 1
mkdir -p logs
: > logs/tunnel.log
setsid ./cloudflared tunnel --url http://127.0.0.1:8000 >> logs/tunnel.log 2>&1 &

for _ in $(seq 25); do
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' logs/tunnel.log | tail -1)
  [ -n "${URL:-}" ] && { echo "$URL"; exit 0; }
  sleep 2
done
echo 'tunnel did not report a URL; see logs/tunnel.log' >&2
exit 1
