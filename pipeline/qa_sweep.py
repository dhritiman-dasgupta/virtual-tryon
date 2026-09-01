#!/usr/bin/env python
"""Run one guardrail model over a whole directory of results and record verdicts.

For comparing guardrails rather than generators: the images are fixed, so any
difference in the output is the model disagreeing, not the generator varying.
Run it twice with different --model and diff the JSON.

usage:
    qa_sweep.py --model Qwen/Qwen3-VL-8B-Instruct --quant bf16 \
                --results /workspace/swift-teal-stoat/outputs/f6 --out sweep_8b.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.guardrail import COMBINED_ASK, COMBINED_CHECKS, COMBINED_FIELDS, _failed
from app.vision import LocalVision, parse

logging.basicConfig(level=logging.WARNING)
ROOT = Path("/workspace/swift-teal-stoat")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default="bf16", choices=["bf16", "8bit", "4bit"])
    ap.add_argument("--results", default=str(ROOT / "outputs/f6"))
    ap.add_argument("--person", default=str(ROOT / "inputs/models/f6.jpeg"))
    ap.add_argument("--garments", default=str(ROOT / "inputs/fg"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=420)
    a = ap.parse_args()

    results = sorted(p for p in Path(a.results).glob("f6__fg*.png")
                     if "_a" not in p.stem)
    if not results:
        print("no results found"); return 1
    print(f"{len(results)} images | {a.model} {a.quant}", flush=True)

    rows, t_all = [], time.time()
    import torch
    with LocalVision(model=a.model, cache_dir=str(ROOT / "hf-cache"),
                     quantise=a.quant) as vlm:
        vram = torch.cuda.memory_allocated() / 1048576
        print(f"resident: {vram:.0f} MiB\n", flush=True)
        for p in results:
            g = p.stem.split("__")[1]
            gpath = Path(a.garments) / f"{g}.jpeg"
            t0 = time.time()
            raw = vlm.ask([a.person, str(gpath), str(p)], COMBINED_ASK,
                          max_new_tokens=a.tokens)
            dt = time.time() - t0
            checks = parse(raw, COMBINED_FIELDS)
            # Same fail-closed rule the pipeline uses: a check with no verdict
            # word is a non-answer, not a pass.
            from app.guardrail import _VERDICT_TOKEN
            missing = [c for c in COMBINED_CHECKS
                       if c not in checks or not _VERDICT_TOKEN.search(checks[c] or "")]
            failed = _failed(checks, COMBINED_CHECKS)
            ok = not failed and not missing
            rows.append({"tag": p.stem, "garment": g, "ok": ok,
                         "failed": failed, "missing": missing,
                         "seconds": round(dt, 2),
                         "garment_type": checks.get("GARMENT_TYPE", "")[:70],
                         "garment_match": checks.get("GARMENT_MATCH", "")[:70]})
            flag = "pass" if ok else ("UNPARSED" if missing else "FAIL")
            print(f"  {g:6} {dt:5.1f}s  {flag:9} {','.join(failed)[:38]:38} "
                  f"{checks.get('GARMENT_TYPE','')[:34]}", flush=True)

    Path(a.out).write_text(json.dumps(
        {"model": a.model, "quant": a.quant, "vram_mib": round(vram),
         "total_s": round(time.time() - t_all, 1), "rows": rows}, indent=2))
    n_ok = sum(1 for r in rows if r["ok"])
    n_unparsed = sum(1 for r in rows if r["missing"])
    print(f"\n{n_ok}/{len(rows)} pass | {n_unparsed} unparsed | "
          f"{(time.time()-t_all):.0f}s total | "
          f"{sum(r['seconds'] for r in rows)/len(rows):.1f}s mean")
    print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
