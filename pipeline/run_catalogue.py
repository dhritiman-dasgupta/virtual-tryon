#!/usr/bin/env python
"""Run a whole catalogue: one model, every garment, one command.

This is the boring path, on purpose. It uses the hand-written garment specs in
garments.py, which measured better than any vision-model catalogue we
produced (12/12 dupattas present, 8/8 sarees kept their construction), so there
is no brochure model, no quantisation and none of the failure modes that come
with them.

Everything except the generation itself is exercised by tests/test_catalogue.py
against a stub server, so the code is proven before it costs GPU time.

    run_catalogue.py --model f6                     # all garments, guardrail per server default
    run_catalogue.py --model f6 --guardrail off     # generation only
    run_catalogue.py --garments fg01,fg06 --dry-run # no API calls at all
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

log = logging.getLogger("catalogue")


def crop(src: Path, box, dst: Path) -> Path:
    """Cut the garment photo down to the garment.

    Per garment, by hand. One fixed centre box used to cut dupattas that hang
    at the edge of the frame clean out of the input, and no prompt can restore
    something the model never saw.
    """
    im = Image.open(src).convert("RGB")
    w, h = im.size
    l, t, r, b = box
    out = im.crop((int(w * l), int(h * t), int(w * r), int(h * b)))
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst, quality=95)
    return dst


def generate_one(client: httpx.Client, person: Path, garment: Path, prompt: str,
                 *, steps: int, seed: int, megapixels: str, lora: str,
                 guardrail: bool | None, poll: float = 0.4) -> dict:
    """Submit one try-on and wait for it. Returns the job info plus the bytes."""
    data = {"prompt": prompt, "steps": str(steps), "seed": str(seed),
            "lora_strength": lora, "megapixels": megapixels}
    if guardrail is not None:
        data["guardrail"] = "true" if guardrail else "false"

    with person.open("rb") as pf, garment.open("rb") as gf:
        r = client.post("/v1/tryon",
                        files={"person": (person.name, pf, "image/jpeg"),
                               "garment": (garment.name, gf, "image/jpeg")},
                        data=data)
    r.raise_for_status()
    job_id = r.json()["job_id"]

    while True:
        info = client.get(f"/v1/jobs/{job_id}").json()
        if info["status"] in ("succeeded", "failed"):
            break
        time.sleep(poll)

    if info["status"] == "failed":
        raise RuntimeError(info.get("error") or "generation failed")
    info["image"] = client.get(f"/v1/jobs/{job_id}/image").content
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--root", default="/workspace/swift-teal-stoat")
    ap.add_argument("--model", default="f6")
    ap.add_argument("--garments", default="")
    ap.add_argument("--guardrail", choices=["on", "off", "default"],
                    default="default")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--megapixels", default="0.75")
    ap.add_argument("--lora-strength", default="0.4")
    ap.add_argument("--tag", default="catalogue")
    ap.add_argument("--dry-run", action="store_true",
                    help="build crops and prompts, make no API calls")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    root = Path(a.root)
    person = root / "inputs" / "models" / f"{a.model}.jpeg"
    if not a.dry_run and not person.exists():
        log.error("no person image at %s", person)
        return 2

    stems = [s.strip() for s in a.garments.split(",") if s.strip()] or sorted(GARMENTS)
    unknown = [s for s in stems if s not in GARMENTS]
    if unknown:
        log.error("no hand-written spec for: %s", ", ".join(unknown))
        return 2

    out_dir = root / f"outputs_{a.tag}" / a.model
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir.parent / "prompts").mkdir(parents=True, exist_ok=True)
    guardrail = {"on": True, "off": False, "default": None}[a.guardrail]

    client = None if a.dry_run else httpx.Client(base_url=a.api, timeout=1800.0)
    records: list[dict] = []
    t_run = time.time()

    for i, stem in enumerate(stems, 1):
        spec = GARMENTS[stem]
        src = root / "inputs" / "fg" / f"{stem}.jpeg"
        gcrop = crop(src, spec["crop"], root / "cache" / "crops" / a.tag / f"{stem}.jpg")
        prompt = build(spec)
        (out_dir.parent / "prompts" / f"{stem}.txt").write_text(prompt)

        if a.dry_run:
            log.info("[%2d/%2d] %s  crop %s  prompt %d chars",
                     i, len(stems), stem, gcrop.name, len(prompt))
            records.append({"garment": stem, "ok": True, "dry_run": True,
                            "prompt_chars": len(prompt)})
            continue

        t0 = time.time()
        try:
            info = generate_one(client, person, gcrop, prompt,
                                steps=a.steps, seed=a.seed,
                                megapixels=a.megapixels, lora=a.lora_strength,
                                guardrail=guardrail)
        except Exception as e:                            # noqa: BLE001
            log.error("[%2d/%2d] %s FAILED: %s", i, len(stems), stem, e)
            records.append({"garment": stem, "ok": False, "error": str(e)})
            continue

        (out_dir / f"{stem}.png").write_bytes(info["image"])
        elapsed = time.time() - t0
        records.append({
            "garment": stem, "ok": True,
            "seconds": round(elapsed, 2),
            "seconds_generate": info.get("duration_seconds"),
            "guardrail": info.get("guardrail", False),
            # null means never inspected - it is not a pass.
            "guardrail_ok": info.get("guardrail_ok"),
            "guardrail_reason": info.get("guardrail_reason"),
            "guardrail_seconds": info.get("guardrail_seconds"),
            "attempts": info.get("attempts", 1),
            "pieces": len(spec["pieces"]),
            "summary": spec["summary"],
            "seed": a.seed,
        })
        # null is "never inspected", which is not the same as a failure and
        # must not be displayed as one.
        if not info.get("guardrail"):
            verdict = "guardrail off"
        elif info.get("guardrail_ok") is None:
            verdict = "NOT INSPECTED - verdict missing"
        elif info["guardrail_ok"]:
            verdict = "PASS"
        else:
            verdict = f"FLAG {(info.get('guardrail_reason') or '')[:40]}"
        log.info("[%2d/%2d] %s  %-44s %5.1fs  %s", i, len(stems), stem,
                 spec["summary"][:44], elapsed, verdict)

    report = out_dir.parent / f"{a.model}_report.json"
    report.write_text(json.dumps(records, indent=2))

    done = [r for r in records if r.get("ok") and not r.get("dry_run")]
    if done:
        times = [r["seconds"] for r in done]
        retried = [r for r in done if (r.get("attempts") or 1) > 1]
        checked = [r for r in done if r.get("guardrail")]
        passed = [r for r in checked if r.get("guardrail_ok")]
        log.info("%d/%d in %.1f min - mean %.1fs, fastest %.1fs, slowest %.1fs",
                 len(done), len(records), (time.time() - t_run) / 60,
                 sum(times) / len(times), min(times), max(times))
        if checked:
            log.info("guardrail: %d/%d passed, %d needed a retry",
                     len(passed), len(checked), len(retried))
        else:
            log.info("guardrail: not run (results are unverified)")
    print(f"\nimages -> {out_dir}\nreport -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
