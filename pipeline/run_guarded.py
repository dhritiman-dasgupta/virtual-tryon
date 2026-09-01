#!/usr/bin/env python
"""Full guarded pipeline: analyse garment -> generate -> inspect -> retry.

Both models stay resident on the card, so each image is judged the moment it is
made and retried on the spot. On a 24 GB box this had to be batched into
generate-all / swap / inspect-all / regenerate; 32 GB removes that.

usage:
    run_guarded.py --model f6 --garments fg
    run_guarded.py --model f6 --limit 3 --max-attempts 2
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
from app.qa import run as guarded_run
from app.tryon_prompt import build as build_prompt
from app.vision import LocalVision, NIMVision

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-8s %(message)s",
                    datefmt="%H:%M:%S")
# One poll per 0.4s per job otherwise drowns the results.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("run")

ROOT = Path(os.environ.get("TRYON_ROOT", Path(__file__).resolve().parent.parent / "runs"))
API = "http://127.0.0.1:8000"


def crop_garment(src: Path, dst: Path) -> Path:
    """Remove the garment photo's own room before it reaches the model.

    This encoder ignores negation, so the showroom cannot be prompted away —
    it has to not be in the input. A fixed centre crop fixed it on 4 of 4
    images that were importing their background.
    """
    if dst.exists():
        return dst
    im = Image.open(src).convert("RGB")
    w, h = im.size
    im.crop((int(w * 0.24), int(h * 0.06), int(w * 0.76), int(h * 0.94))).save(
        dst, quality=95)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="f6")
    ap.add_argument("--garments", default="fg", choices=["fg", "mg"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--megapixels", default="0.75")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--qa-backend", default="local", choices=["local", "nim"],
                    help="who runs the guardrail; garment analysis stays local")
    ap.add_argument("--nim-model", default="meta/muse-glimmer-30b")
    ap.add_argument("--nim-tokens", type=int, default=4096)
    ap.add_argument("--only", default="",
                    help="comma-separated garment stems, e.g. fg01,fg03")
    a = ap.parse_args()

    inputs = ROOT / "inputs"
    person = inputs / "models" / f"{a.model}.jpeg"
    garments = sorted((inputs / a.garments).glob("*.jpeg"))
    if a.only:
        want = {x.strip() for x in a.only.split(",") if x.strip()}
        garments = [g for g in garments if g.stem in want]
    if a.limit:
        garments = garments[: a.limit]
    if not person.exists():
        log.error("no person image at %s", person)
        return 1

    out_dir = Path(a.out) / a.model
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = ROOT / "cache" / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    cache = GarmentCache(ROOT / "cache" / "garments")

    client = httpx.Client(base_url=API, timeout=3600.0)

    def make_generator(person_path: Path, garment_path: Path, prompt: str):
        def generate(seed: int, steps: int) -> bytes:
            with person_path.open("rb") as pf, garment_path.open("rb") as gf:
                r = client.post(
                    "/v1/tryon",
                    files={"person": (person_path.name, pf),
                           "garment": (garment_path.name, gf)},
                    data={"prompt": prompt, "steps": str(steps),
                          "seed": str(seed), "megapixels": a.megapixels})
            r.raise_for_status()
            job = r.json()["job_id"]
            while True:
                info = client.get(f"/v1/jobs/{job}").json()
                if info["status"] in ("succeeded", "failed"):
                    break
                time.sleep(0.4)
            if info["status"] == "failed":
                raise RuntimeError(info.get("error", "generation failed"))
            return client.get(f"/v1/jobs/{job}/image").content
        return generate

    records = []
    t_all = time.time()

    # One `with`, held for the whole run: the vision model loads once and stays
    # resident beside the generator. Re-entering per image would cost a full
    # load every time and defeat the point of the 32 GB card.
    with LocalVision(cache_dir=str(ROOT / "hf-cache"), quantise="4bit") as vlm:
        log.info("vision model resident")
        # Garment analysis always runs locally — it is cached per garment, so
        # paying a slow remote model for it buys nothing. Only the per-image
        # guardrail is swappable.
        qa_backend = vlm
        if a.qa_backend == "nim":
            qa_backend = NIMVision(model=a.nim_model, qa_tokens=a.nim_tokens,
                                   fallbacks=())
            log.info("guardrail backend: NIM %s (%d tokens)",
                     a.nim_model, a.nim_tokens)
        for i, gpath in enumerate(garments, 1):
            tag = f"{a.model}__{gpath.stem}"

            t0 = time.time()
            gdesc = analyse(vlm, gpath, cache)
            t_analyse = time.time() - t0

            prompt = build_prompt(gdesc)
            gcrop = crop_garment(gpath, crop_dir / f"{gpath.stem}.jpg")

            t0 = time.time()
            best, attempts = guarded_run(
                generate=make_generator(person, gcrop, prompt),
                backend=qa_backend, person=str(person), garment=str(gpath),
                out_dir=out_dir, tag=tag, base_seed=a.seed, steps=a.steps,
                max_attempts=a.max_attempts)
            t_total = time.time() - t0

            gen_s = sum(x.seconds for x in attempts)
            qa_s = t_total - gen_s
            rec = {
                "tag": tag, "ok": best.ok, "reason": best.reason,
                "stage": best.stage, "attempts": len(attempts),
                "garment_type": gdesc.get("TYPE", ""),
                "seconds_total": round(t_total, 2),
                "seconds_generate": round(gen_s, 2),
                "seconds_qa": round(qa_s, 2),
                "seconds_analyse": round(t_analyse, 2),
                "per_attempt": [x.report() for x in attempts],
            }
            records.append(rec)
            log.info("[%d/%d] %-14s %-4s %d attempt(s)  gen %.1fs  qa %.1fs  "
                     "total %.1fs  %s",
                     i, len(garments), tag, "PASS" if best.ok else "FAIL",
                     len(attempts), gen_s, qa_s, t_total,
                     gdesc.get("TYPE", "") if best.ok else best.reason[:52])

    report = Path(a.out) / f"{a.model}_report.json"
    report.write_text(json.dumps(records, indent=2))

    passed = sum(1 for r in records if r["ok"])
    gens = [r["seconds_generate"] / r["attempts"] for r in records if r["attempts"]]
    qas = [r["seconds_qa"] for r in records]
    tot = [r["seconds_total"] for r in records]
    retried = sum(1 for r in records if r["attempts"] > 1)

    print(f"\n=== {passed}/{len(records)} passed in {(time.time()-t_all)/60:.1f} min ===")
    if gens:
        print(f"generate   mean {sum(gens)/len(gens):5.1f}s   "
              f"min {min(gens):5.1f}s  max {max(gens):5.1f}s   (per attempt)")
        print(f"guardrail  mean {sum(qas)/len(qas):5.1f}s   "
              f"min {min(qas):5.1f}s  max {max(qas):5.1f}s")
        print(f"end-to-end mean {sum(tot)/len(tot):5.1f}s   "
              f"min {min(tot):5.1f}s  max {max(tot):5.1f}s")
    print(f"retried    {retried}/{len(records)}")
    print(f"report     {report}")
    for r in records:
        if not r["ok"]:
            print(f"  unresolved: {r['tag']:14} {r['reason'][:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
