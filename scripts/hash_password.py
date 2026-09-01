#!/usr/bin/env python3
"""Print an AUTH_USERS entry for a username and password.

    python3 scripts/hash_password.py reviewer 'some password'
    -> reviewer:sha256$a1b2...$c3d4...

Several entries go in AUTH_USERS comma separated.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from app.auth import hash_password

if len(sys.argv) != 3:
    raise SystemExit(__doc__)
print(f"{sys.argv[1]}:{hash_password(sys.argv[2])}")
