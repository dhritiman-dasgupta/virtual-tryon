#!/usr/bin/env python
"""Turn run_eval.py's JSON into the tables that decide the choice.

Three views, because one number cannot answer this:

  1. headline      catch rate, false-alarm rate, speed, VRAM
  2. per check     which of the 14 fields each model is actually good at
  3. diagnosis     for the losers, whether they failed on judgement or format

The last one matters most in practice. A model that scores badly because it
cannot hold the output format is a different problem from one that reads the
images wrongly, and only the first is fixable by changing the prompt.
"""
from __future__ import annotations

import argparse
import json
import pathlib


def bar(x: float, width: int = 10) -> str:
    return "#" * round(x * width) + "." * (width - round(x * width))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--baseline", default="qwen3vl4b")
    a = ap.parse_args()

    res = json.loads(pathlib.Path(a.results).expanduser().read_text())
    ok = [r for r in res if "error" not in r and r.get("catch_rate") is not None]
    dead = [r for r in res if r not in ok]

    print("=" * 92)
    print("HEADLINE  (catch = defects flagged; false = clean images flagged anyway)")
    print("=" * 92)
    print(f"{'model':13s} {'catch':>7s} {'':12s} {'false':>7s} {'bal':>7s} "
          f"{'s/item':>7s} {'tok/s':>7s} {'VRAM':>6s} {'parsed':>7s}")
    for r in sorted(ok, key=lambda x: -x["balanced"]):
        n_items = r.get("items", 1)
        parsed = 1 - r["unparsed_fields"] / (n_items * 14)
        tag = "  <- production" if r["key"] == a.baseline else ""
        print(f"{r['key']:13s} {r['catch_rate']:>7.3f} {bar(r['catch_rate']):12s} "
              f"{r['false_alarm_rate']:>7.3f} {r['balanced']:>7.3f} "
              f"{r['mean_seconds']:>7.1f} {r['tokens_per_second']:>7.1f} "
              f"{r['peak_vram_gb']:>6.2f} {parsed:>7.1%}{tag}")
    for r in dead:
        print(f"{r['key']:13s} {r.get('error', 'no result')[:70]}")

    if not ok:
        return 1

    checks = sorted({c for r in ok for c in r["per_check"]})
    print("\n" + "=" * 92)
    print("PER CHECK  (fraction of that check's labels answered correctly)")
    print("=" * 92)
    print(f"{'check':18s} " + " ".join(f"{r['key'][:10]:>11s}" for r in ok))
    for c in checks:
        row = f"{c:18s} "
        for r in ok:
            d = r["per_check"].get(c)
            if not d:
                row += f"{'-':>11s} "
                continue
            tot = sum(d.values())
            correct = d["caught"] + d["ok"]
            row += f"{correct/tot:>11.2f} " if tot else f"{'-':>11s} "
        print(row)

    print("\n" + "=" * 92)
    print("DIAGNOSIS")
    print("=" * 92)
    base = next((r for r in ok if r["key"] == a.baseline), None)
    for r in sorted(ok, key=lambda x: -x["balanced"]):
        n_items = r.get("items", 1)
        parsed = 1 - r["unparsed_fields"] / (n_items * 14)
        notes = []
        if parsed < 0.9:
            notes.append(f"format unreliable - only {parsed:.0%} of fields parsed, "
                         "so its score understates its judgement")
        if r["catch_rate"] < 0.5:
            notes.append("misses more than half of real defects - not usable as a gate")
        if r["false_alarm_rate"] > 0.15:
            notes.append(f"flags {r['false_alarm_rate']:.0%} of clean images, "
                         "which would burn reseeds")
        if base and r["key"] != a.baseline:
            ds = base["mean_seconds"] - r["mean_seconds"]
            dv = base["peak_vram_gb"] - r["peak_vram_gb"]
            if ds > 0.5 or dv > 0.3:
                notes.append(f"{ds:+.1f}s and {dv:+.2f} GB against production")
        if not notes:
            notes.append("no disqualifying weakness")
        print(f"\n{r['key']}")
        for n in notes:
            print(f"  - {n}")

    # A guardrail is only worth running if it beats "flag nothing" on catch rate
    # while not flagging so much that the reseeds cost more than the defects.
    print("\n" + "=" * 92)
    best = max(ok, key=lambda r: (r["catch_rate"], -r["mean_seconds"]))
    fast = min((r for r in ok if r["catch_rate"] >= 0.8), key=lambda r: r["mean_seconds"],
               default=None)
    print(f"highest catch rate : {best['key']} ({best['catch_rate']:.3f}, "
          f"{best['mean_seconds']:.1f}s)")
    if fast:
        print(f"fastest above 0.80 : {fast['key']} ({fast['catch_rate']:.3f}, "
              f"{fast['mean_seconds']:.1f}s, {fast['peak_vram_gb']:.2f} GB)")
    else:
        print("fastest above 0.80 : none reached a 0.80 catch rate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
