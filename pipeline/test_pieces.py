#!/usr/bin/env python
"""A/B the piece-enumeration fix on garments that have a dupatta.

The reported defects — a two-piece set fused into one, a dupatta drifting to the
wrong side or vanishing — are all the same failure: the generator drops garment
components the prompt does not force it to account for. Same shape as the
three-hands bug, which a numbered closed list fixed.

So this runs each garment twice at the same seed, changing only the prompt:

    before   TYPE + HOW_WORN prose, the dupatta named inside a sentence
    after    PIECES as a counted closed list + an explicit DUPATTA clause

Same seed both times, so any difference is the prompt and not the noise.

usage: test_pieces.py --model f6 --garments fg01,fg06,fg16
"""
from __future__ import annotations
import os

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from PIL import Image

from app.garment import GarmentCache, analyse
from app.tryon_prompt import ROLE, PRESERVE, build
from app.vision import LocalVision

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("pieces")

ROOT = Path(os.environ.get("TRYON_ROOT", Path(__file__).resolve().parent.parent / "runs"))
API = "http://127.0.0.1:8000"


def build_without_pieces(g: dict) -> str:
    """The previous prompt: no piece count, no dedicated dupatta clause."""
    parts = []
    if t := g.get("TYPE"):
        parts.append(f"The garment is a {t}.")
    if w := g.get("HOW_WORN"):
        parts.append(f"It is worn like this: {w}")
    if c := g.get("COLOURS"):
        parts.append(f"Its colours are {c} — reproduce them exactly.")
    if m := g.get("METAL"):
        parts.append(f"Metallic work is {m}.")
    if f := g.get("FABRIC"):
        parts.append(f"The fabric is {f}.")
    body = "THE GARMENT:\n" + " ".join(parts) if parts else ""
    blocks = [ROLE] + ([body] if body else []) + [
        PRESERVE, "The second image contributes the garment alone."]
    return "\n\n".join(blocks)


def crop_garment(src: Path, dst: Path) -> Path:
    if dst.exists():
        return dst
    im = Image.open(src).convert("RGB")
    w, h = im.size
    im.crop((int(w * .24), int(h * .06), int(w * .76), int(h * .94))).save(
        dst, quality=95)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="f6")
    ap.add_argument("--garments", default="", help="comma-separated stems")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--megapixels", default="1.0")
    ap.add_argument("--out", default=str(ROOT / "outputs_pieces"))
    a = ap.parse_args()

    inputs = ROOT / "inputs"
    person = inputs / "models" / f"{a.model}.jpeg"
    stems = [s.strip() for s in a.garments.split(",") if s.strip()]
    garments = ([inputs / "fg" / f"{s}.jpeg" for s in stems] if stems
                else sorted((inputs / "fg").glob("*.jpeg")))

    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = ROOT / "cache" / "crops"; crop_dir.mkdir(parents=True, exist_ok=True)
    cache = GarmentCache(ROOT / "cache" / "garments")
    client = httpx.Client(base_url=API, timeout=3600.0)

    def generate(person_p: Path, garment_p: Path, prompt: str, tag: str) -> float:
        with person_p.open("rb") as pf, garment_p.open("rb") as gf:
            r = client.post("/v1/tryon",
                            files={"person": (person_p.name, pf),
                                   "garment": (garment_p.name, gf)},
                            data={"prompt": prompt, "steps": str(a.steps),
                                  "seed": str(a.seed), "megapixels": a.megapixels})
        r.raise_for_status(); job = r.json()["job_id"]
        while True:
            info = client.get(f"/v1/jobs/{job}").json()
            if info["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.4)
        if info["status"] == "failed":
            raise RuntimeError(info.get("error", "generation failed"))
        (out_dir / f"{tag}.png").write_bytes(
            client.get(f"/v1/jobs/{job}/image").content)
        return info["duration_seconds"]

    records = []
    with LocalVision(cache_dir=str(ROOT / "hf-cache"), quantise="4bit") as vlm:
        log.info("vision model resident")
        for gpath in garments:
            g = analyse(vlm, gpath, cache)
            pieces = g.get("PIECES", "")
            dup = g.get("DUPATTA", "")
            log.info("%s  type=%-9s pieces=%-46s dupatta=%s",
                     gpath.stem, g.get("TYPE", "?"), pieces[:46], dup[:40])

            gcrop = crop_garment(gpath, crop_dir / f"{gpath.stem}.jpg")
            t_before = generate(person, gcrop, build_without_pieces(g),
                                f"{gpath.stem}_before")
            t_after = generate(person, gcrop, build(g), f"{gpath.stem}_after")
            records.append({"garment": gpath.stem, "type": g.get("TYPE"),
                            "pieces": pieces, "dupatta": dup,
                            "seconds_before": round(t_before, 1),
                            "seconds_after": round(t_after, 1)})
            log.info("  generated both  before %.1fs  after %.1fs", t_before, t_after)

    (out_dir / "report.json").write_text(json.dumps(records, indent=2))
    print(f"\n{len(records)} garments -> {out_dir}")
    n_dup = sum(1 for r in records
                if r["dupatta"] and r["dupatta"].lower() not in ("none", "-"))
    print(f"{n_dup} of {len(records)} have a dupatta the prompt now names explicitly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
