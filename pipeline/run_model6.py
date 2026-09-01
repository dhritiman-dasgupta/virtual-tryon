#!/usr/bin/env python
"""Model 6 against all 23 female garments, with hand-written prompts.

No vision model anywhere in this path: the garment readings, the piece lists,
the drape descriptions and the crop boxes are all written by hand in
garments.py, and the output is reviewed by eye afterwards rather than
by an automated guardrail. That removes ~8.5s of per-image QA and the retry
loop, so wall-clock here is generation time and nothing else.

usage:
    run_model6.py                          # all 23
    run_model6.py --garments fg01,fg06     # a subset
    run_model6.py --tag round4 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from PIL import Image

from pipeline.garments import GARMENTS, build

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("m6")

ROOT = Path("/workspace/swift-teal-stoat")
API = "http://127.0.0.1:8000"


def crop(src: Path, box: tuple[float, float, float, float], dst: Path) -> Path:
    """Cut the garment photo down to the garment, per-garment.

    The point is to remove the showroom, the props and the app UI from the
    *input*, because this text encoder does not act on negation — telling it
    "no mannequin, no sofa" had no measurable effect, so the only reliable way
    to keep a scene out of the result is to keep it out of the image. Each box
    is set by eye to keep every piece of the garment, dupattas included.
    """
    im = Image.open(src).convert("RGB")
    w, h = im.size
    l, t, r, b = box
    im.crop((int(w * l), int(h * t), int(w * r), int(h * b))).save(dst, quality=95)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="f6")
    ap.add_argument("--garments", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--megapixels", default="0.75")
    ap.add_argument("--lora-strength", default="0.4")
    ap.add_argument("--tag", default="round4")
    a = ap.parse_args()

    person = ROOT / "inputs" / "models" / f"{a.model}.jpeg"
    if not person.exists():
        log.error("no person image at %s", person)
        return 1

    stems = ([s.strip() for s in a.garments.split(",") if s.strip()]
             or sorted(GARMENTS))
    out_dir = ROOT / f"outputs_{a.tag}" / a.model
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = ROOT / "cache" / "crops" / a.tag
    crop_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir = out_dir.parent / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    client = httpx.Client(base_url=API, timeout=1800.0)
    records, t_run = [], time.time()

    for i, stem in enumerate(stems, 1):
        spec = GARMENTS.get(stem)
        if spec is None:
            log.warning("%s has no hand-written spec, skipping", stem)
            continue

        src = ROOT / "inputs" / "fg" / f"{stem}.jpeg"
        gcrop = crop(src, spec["crop"], crop_dir / f"{stem}.jpg")
        text = build(spec)
        (prompt_dir / f"{stem}.txt").write_text(text)

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
            log.error("%s failed: %s", stem, info.get("error"))
            records.append({"garment": stem, "ok": False,
                            "error": info.get("error")})
            continue

        (out_dir / f"{stem}.png").write_bytes(
            client.get(f"/v1/jobs/{job}/image").content)
        elapsed = time.time() - t0
        records.append({
            "garment": stem, "ok": True,
            "seconds": round(elapsed, 2),
            "seconds_generate": round(info.get("duration_seconds") or elapsed, 2),
            "pieces": len(spec["pieces"]),
            "summary": spec["summary"],
            "seed": a.seed, "prompt_chars": len(text),
        })
        log.info("[%2d/%2d] %s  %-52s %5.1fs",
                 i, len(stems), stem, spec["summary"][:52], elapsed)

    report = out_dir.parent / f"{a.model}_report.json"
    report.write_text(json.dumps(records, indent=2))

    done = [r for r in records if r.get("ok")]
    if done:
        times = [r["seconds"] for r in done]
        total = time.time() - t_run
        log.info("%d/%d generated in %.1f min — mean %.1fs, "
                 "fastest %.1fs, slowest %.1fs",
                 len(done), len(records), total / 60,
                 sum(times) / len(times), min(times), max(times))
    print(f"\nimages -> {out_dir}\nreport -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
