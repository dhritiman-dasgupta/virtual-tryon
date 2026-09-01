#!/usr/bin/env python3
"""End-to-end smoke test against a running server.

    python tests/smoke_test.py \
        --person "/path/to/model.jpeg" \
        --garment "/path/to/outfit.jpeg" \
        --preset f1_orange

With no --person/--garment it looks for the sample set in ~/Downloads.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

SAMPLES = Path.home() / "Downloads" / "samples/inputs"
DEFAULT_PERSON = SAMPLES / "Female" / "female model" / "WhatsApp Image 2026-08-12 at 1.10.03 PM (5).jpeg"
DEFAULT_GARMENT = SAMPLES / "Female" / "outfit 1 (1).jpeg"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--person", type=Path, default=DEFAULT_PERSON)
    ap.add_argument("--garment", type=Path, default=DEFAULT_GARMENT)
    ap.add_argument("--preset", default="f1_orange")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", type=Path, default=Path("smoke_result.png"))
    args = ap.parse_args()

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    c = httpx.Client(base_url=args.base, headers=headers, timeout=60.0)

    print("1. health")
    h = c.get("/healthz").json()
    print(f"   status={h['status']} comfy={h['comfy_reachable']}")
    for name, ok in (h.get("models_present") or {}).items():
        print(f"   {'OK ' if ok else 'MISSING'} {name}")
    if h["status"] != "ok":
        print(f"   detail: {h.get('detail')}")
        print("   -> fix health before generating")
        return 1

    print("2. presets")
    presets = c.get("/v1/prompts").json()
    print(f"   {len(presets)} available; using '{args.preset}'")

    for p in (args.person, args.garment):
        if not p.exists():
            print(f"   missing input: {p}")
            return 1

    print("3. submit")
    t0 = time.time()
    with args.person.open("rb") as pf, args.garment.open("rb") as gf:
        r = c.post(
            "/v1/tryon",
            files={"person": (args.person.name, pf), "garment": (args.garment.name, gf)},
            data={"preset": args.preset, "steps": str(args.steps), "seed": str(args.seed)},
        )
    if r.status_code >= 400:
        print(f"   HTTP {r.status_code}: {r.text[:500]}")
        return 1
    job_id = r.json()["job_id"]
    print(f"   job {job_id}")

    print("4. poll")
    while True:
        info = c.get(f"/v1/jobs/{job_id}").json()
        status = info["status"]
        if status == "running" and info.get("total_steps"):
            print(f"\r   step {info['step']}/{info['total_steps']}", end="", flush=True)
        if status in ("succeeded", "failed"):
            print()
            break
        time.sleep(1.5)

    if status == "failed":
        print(f"   FAILED: {info['error']}")
        return 1

    print(f"   done in {info['duration_seconds']}s (seed {info['seed']})")

    print("5. fetch image")
    img = c.get(f"/v1/jobs/{job_id}/image")
    img.raise_for_status()
    args.out.write_bytes(img.content)
    print(f"   wrote {args.out} ({len(img.content) // 1024} KB)")
    print(f"\nTotal wall clock: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
