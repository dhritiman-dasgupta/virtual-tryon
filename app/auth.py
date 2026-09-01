"""Bearer tokens for the test platform.

Deliberately small: HMAC-signed tokens over the standard library, no jose, no
passlib, no session store. The API is stateless, so a signed token that carries
its own expiry is enough, and every dependency here would be one more thing to
patch on a box that already carries 18 GB of model weights.

Passwords are stored as salted SHA-256 (`sha256$salt$digest`) rather than in
the clear. That is weaker than bcrypt or argon2 against an offline attack on a
stolen file, and the right upgrade if this ever holds real customer accounts -
for a handful of reviewer logins on a test platform it is a reasonable floor,
and it is at least not plaintext.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time

log = logging.getLogger("auth")


def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(8)
    digest = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return f"sha256${salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check. Accepts a plaintext entry for local development."""
    if stored.startswith("sha256$"):
        _, salt, digest = stored.split("$", 2)
        return hmac.compare_digest(
            hashlib.sha256(f"{salt}{password}".encode()).hexdigest(), digest)
    # A plaintext entry is allowed so a dev box can be brought up quickly, but
    # it is worth a warning every time it is used.
    log.warning("password stored in plaintext; run scripts/hash_password.py")
    return hmac.compare_digest(password, stored)


def parse_users(spec: str) -> dict[str, str]:
    """`alice:sha256$..,bob:sha256$..` -> {user: stored}."""
    users: dict[str, str] = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        name, stored = entry.split(":", 1)
        users[name.strip()] = stored.strip()
    return users


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def issue(username: str, secret: str, ttl: int) -> tuple[str, int]:
    payload = {"sub": username, "exp": int(time.time()) + ttl}
    body = _b64(json.dumps(payload, separators=(",", ":")).encode())
    sig = _b64(hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}", payload["exp"]


def verify(token: str, secret: str) -> str | None:
    """Return the username, or None if the token is invalid or expired."""
    try:
        body, sig = token.split(".", 1)
    except ValueError:
        return None
    expected = _b64(hmac.new(secret.encode(), body.encode(),
                             hashlib.sha256).digest())
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        payload = json.loads(_unb64(body))
    except Exception:                                    # noqa: BLE001
        return None
    if payload.get("exp", 0) < time.time():
        return None
    return payload.get("sub")
