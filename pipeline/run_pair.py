#!/usr/bin/env python
"""Generate one try-on from an arbitrary person photo and garment photo.

run_model6.py is tied to the catalogue layout on the GPU box. This one takes
two files from anywhere, crops the garment to a box given on the command line,
builds the prompt from a spec in pair_specs.py, and writes the result next to
wherever you point --out. It talks to the API over HTTP, so it works from a
laptop against a remote box as long as the port is reachable.

usage:
    run_pair.py --spec maroon_velvet_saree \
                --person ~/Downloads/model.jpg \
                --garment ~/Downloads/garment.jpg \
                --api http://HOST:8000 --out ~/Downloads/out.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from PIL import Image

from pipeline.garments import build
from pipeline.pair_specs import SPECS


def crop(src: Path, box, dst: Path) -> Path:
    im = Image.open(src).convert("RGB")
    w, h = im.size
    l, t, r, b = box
    out = im.crop((int(w * l), int(h * t), int(w * r), int(h * b)))
    out.save(dst, quality=95)
    print(f"cropped {im.size} -> {out.size}  {dst}")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="key in pair_specs.SPECS")
    ap.add_argument("--person", required=True)
    ap.add_argument("--garment", required=True)
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--megapixels", default="0.75")
    ap.add_argument("--lora-strength", default="0.4")
    a = ap.parse_args()

    spec = SPECS[a.spec]
    person = Path(a.person).expanduser()
    garment = Path(a.garment).expanduser()
    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    gcrop = crop(garment, spec["crop"], out.with_name(out.stem + "_garment_crop.jpg"))
    text = build(spec)
    out.with_name(out.stem + "_prompt.txt").write_text(text)

    client = httpx.Client(base_url=a.api, timeout=1800.0)
    t0 = time.time()
    with person.open("rb") as pf, gcrop.open("rb") as gf:
        r = client.post("/v1/tryon",
                        files={"person": (person.name, pf, "image/jpeg"),
                               "garment": (gcrop.name, gf, "image/jpeg")},
                        data={"prompt": text, "steps": str(a.steps),
                              "seed": str(a.seed),
                              "lora_strength": a.lora_strength,
                              "megapixels": a.megapixels})
    r.raise_for_status()
    job = r.json()["job_id"]

    while True:
        info = client.get(f"/v1/jobs/{job}").json()
        if info["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.3)

    if info["status"] == "failed":
        print("failed:", info.get("error"))
        return 1

    out.write_bytes(client.get(f"/v1/jobs/{job}/image").content)
    print(f"{out}  ({time.time() - t0:.1f}s, seed {a.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
