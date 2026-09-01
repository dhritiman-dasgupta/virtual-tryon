#!/usr/bin/env python
"""Benchmark guardrail backends: throughput, latency, and whether they still catch.

Speed alone is the wrong metric. A guardrail that answers in 2s but passes a
black lehenga rendered sage-green is worse than useless — it certifies defects.
So every candidate is scored on three known cases as well as on tokens/sec:

    fg03 at 1.0 MP   the garment is BLACK, the render is sage-green.
                     A correct guardrail FAILS this on GARMENT_MATCH.
    fg09             correct purple saree — a correct guardrail PASSES it.
    f1__fg20         two people and an imported showroom, if present.

usage: bench_vlm.py --models Qwen/Qwen3-VL-4B-Instruct:bf16 Qwen/Qwen3-VL-8B-Instruct:4bit
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


def cases() -> list[dict]:
    """Each case carries the answer we already know to be correct."""
    inp = ROOT / "inputs"
    out = []
    fg03 = ROOT / "outputs_mp10/f6/f6__fg03.png"
    if fg03.exists():
        out.append({"name": "fg03 black->sage", "person": inp / "models/f6.jpeg",
                    "garment": inp / "fg/fg03.jpeg", "result": fg03,
                    "expect_fail": True, "why": "garment is black, render is sage-green"})
    fg09 = ROOT / "outputs_fast2/f6/f6__fg09.png"
    if fg09.exists():
        out.append({"name": "fg09 correct", "person": inp / "models/f6.jpeg",
                    "garment": inp / "fg/fg09.jpeg", "result": fg09,
                    "expect_fail": False, "why": "purple saree, correct"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="repo:quant, quant in {bf16,8bit,4bit}")
    ap.add_argument("--repeats", type=int, default=2)
    a = ap.parse_args()

    trials = cases()
    if not trials:
        print("no benchmark cases found — run a generation first")
        return 1
    print(f"{len(trials)} case(s): " + ", ".join(t["name"] for t in trials))

    rows = []
    for spec in a.models:
        repo, _, quant = spec.partition(":")
        quant = quant or "4bit"
        print(f"\n===== {repo}  {quant} =====", flush=True)
        try:
            t_load = time.time()
            with LocalVision(model=repo, cache_dir=str(ROOT / "hf-cache"),
                             quantise=quant) as vlm:
                load_s = time.time() - t_load
                import torch
                vram = torch.cuda.memory_allocated() / 1048576
                print(f"loaded in {load_s:.0f}s, {vram:.0f} MiB allocated", flush=True)

                lat, toks, correct = [], [], 0
                for t in trials:
                    best = None
                    for _ in range(a.repeats):
                        t0 = time.time()
                        raw = vlm.ask([str(t["person"]), str(t["garment"]),
                                       str(t["result"])],
                                      COMBINED_ASK, max_new_tokens=420)
                        dt = time.time() - t0
                        best = dt if best is None else min(best, dt)
                    # Rough token count; enough to compare backends fairly.
                    n_tok = max(1, len(raw.split()))
                    lat.append(best)
                    toks.append(n_tok / best)
                    checks = parse(raw, COMBINED_FIELDS)
                    failed = _failed(checks, COMBINED_CHECKS)
                    got_fail = bool(failed)
                    hit = got_fail == t["expect_fail"]
                    correct += hit
                    print(f"  {t['name']:20} {best:5.1f}s  "
                          f"{n_tok/best:5.1f} tok/s  "
                          f"{'FAIL' if got_fail else 'pass':5} "
                          f"{'ok  ' if hit else 'WRONG'} "
                          f"{','.join(failed)[:44]}", flush=True)
                rows.append({"model": repo, "quant": quant,
                             "load_s": round(load_s, 1), "vram_mib": round(vram),
                             "latency_s": round(sum(lat) / len(lat), 2),
                             "tok_s": round(sum(toks) / len(toks), 1),
                             "correct": correct, "of": len(trials)})
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            rows.append({"model": repo, "quant": quant, "error": str(exc)[:120]})

    print("\n" + "=" * 78)
    print(f"{'model':34} {'quant':6} {'VRAM':>7} {'lat':>7} {'tok/s':>7} {'catches':>8}")
    for r in rows:
        if "error" in r:
            print(f"{r['model'][:34]:34} {r['quant']:6} {'—':>7} {'—':>7} {'—':>7}  {r['error'][:24]}")
            continue
        print(f"{r['model'][:34]:34} {r['quant']:6} {r['vram_mib']:6}M "
              f"{r['latency_s']:6.1f}s {r['tok_s']:6.1f} {r['correct']:4}/{r['of']}")
    Path(ROOT / "outputs" / "vlm_bench.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
