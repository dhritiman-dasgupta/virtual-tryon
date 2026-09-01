#!/usr/bin/env bash
# Run this ONCE on the box, while you still have access. It copies the keys
# currently authorised onto the persistent volume so bootstrap.sh can restore
# them after every stop.
set -euo pipefail
ROOT=${ROOT:-/workspace/swift-teal-stoat}
mkdir -p "$ROOT"
cp /root/.ssh/authorized_keys "$ROOT/authorized_keys"
chmod 600 "$ROOT/authorized_keys"
echo "saved $(wc -l < "$ROOT/authorized_keys") key(s) to $ROOT/authorized_keys"
echo "after every restart, run:  bash $ROOT/bootstrap.sh"
