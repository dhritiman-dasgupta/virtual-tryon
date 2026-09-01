#!/usr/bin/env bash
# Set the test-platform login on the box, over SSH.
#
#   HOST=1.2.3.4 PORT=<PORT> ./deploy/set-credentials.sh demo 'some password'
#
# The .env is generated locally and copied as a file rather than written with a
# remote heredoc. That matters: a password hash looks like sha256$salt$digest,
# and an unquoted remote heredoc expands $salt and $digest to nothing, leaving
# a hash that can never match. That failure cost a debugging round once already.
set -euo pipefail

: "${HOST:?set HOST}" ; : "${PORT:?set PORT}"
KEY=${KEY:-~/.ssh/id_ed25519}
ROOT=${ROOT:-${TRYON_ROOT:-/workspace/tryon}}
REPO=$(cd "$(dirname "$0")/.." && pwd)

USER_NAME=${1:-}
PASSWORD=${2:-}
if [ -z "$USER_NAME" ] || [ -z "$PASSWORD" ]; then
  echo "usage: HOST=... PORT=... $0 <username> '<password>'" >&2
  exit 2
fi

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

PYTHON=${PYTHON:-python3}
"$PYTHON" - "$USER_NAME" "$PASSWORD" "$REPO" > "$TMP" <<'PY'
import secrets, sys
sys.path.insert(0, sys.argv[3])
from app.auth import hash_password
print(f"AUTH_USERS={sys.argv[1]}:{hash_password(sys.argv[2])}")
print(f"AUTH_SECRET={secrets.token_hex(32)}")
print("CORS_ORIGINS=*")
print("GUARDRAIL_ENABLED=true")
PY

scp -q -P "$PORT" -i "$KEY" "$TMP" "root@$HOST:$ROOT/.env"
ssh -p "$PORT" -i "$KEY" "root@$HOST" "chmod 600 $ROOT/.env"

echo "credentials written to $ROOT/.env on $HOST"
echo "  username: $USER_NAME"
echo
echo "the API must be restarted to pick them up:"
echo "  ssh -p $PORT -i $KEY root@$HOST 'pkill -f \"[u]vicorn app.main\"' && \\"
echo "  ssh -p $PORT -i $KEY root@$HOST 'bash $ROOT/bootstrap.sh'"
echo
echo "then verify (auth must succeed BEFORE you trust the page):"
echo "  curl -s -X POST <API_URL>/v1/auth/token \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"username\":\"$USER_NAME\",\"password\":\"<password>\"}'"
