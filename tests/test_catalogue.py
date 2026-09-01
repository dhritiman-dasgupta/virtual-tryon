#!/usr/bin/env python3
"""Prove the catalogue pipeline works without spending GPU time.

A stub stands in for the API and records exactly what was sent, so the tests
check the things that have actually gone wrong on real runs:

  * every garment gets its own crop box, and the crop is not the whole image
  * the guardrail flag is transmitted as asked, including "off"
  * a result that was never inspected reports guardrail_ok = null, never a pass
  * retries and their cost land in the report
  * a failure of one garment does not abandon the other twenty-two

Run: python3 tests/test_catalogue.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from pipeline import run_catalogue
from pipeline.garments import GARMENTS, build

PASSED, FAILED = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASSED if cond else FAILED).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")


class StubResponse:
    def __init__(self, payload=None, content=b""):
        self._payload, self.content = payload, content

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class StubClient:
    """Records posts and returns a finished job immediately."""

    def __init__(self, *, guardrail_ok=True, attempts=1, fail_on=()):
        self.posts: list[dict] = []
        self.guardrail_ok, self.attempts, self.fail_on = guardrail_ok, attempts, fail_on
        self._n = 0

    def post(self, url, files=None, data=None):
        self._n += 1
        self.posts.append({"data": dict(data or {}),
                           "garment_name": files["garment"][0],
                           "garment_bytes": files["garment"][1].read()})
        return StubResponse({"job_id": f"job{self._n}"})

    def get(self, url):
        if url.endswith("/image"):
            return StubResponse(content=b"PNGDATA")
        job = url.rsplit("/", 1)[-1]
        idx = int(job.replace("job", "")) - 1
        stem = self.posts[idx]["garment_name"].replace(".jpg", "")
        if stem in self.fail_on:
            return StubResponse({"status": "failed", "error": "stub failure"})
        guard = self.posts[idx]["data"].get("guardrail")
        on = guard != "false"
        return StubResponse({
            "status": "succeeded", "duration_seconds": 9.9,
            "guardrail": on,
            "guardrail_ok": (self.guardrail_ok if on else None),
            "guardrail_reason": None if self.guardrail_ok or not on else "HANDS: FAIL",
            "guardrail_seconds": 8.1 if on else None,
            "attempts": self.attempts,
        })


def build_tree(root: Path, stems: list[str]) -> None:
    (root / "inputs" / "models").mkdir(parents=True, exist_ok=True)
    (root / "inputs" / "fg").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (800, 1200), "navy").save(root / "inputs" / "models" / "f6.jpeg")
    for s in stems:
        Image.new("RGB", (900, 1400), "maroon").save(root / "inputs" / "fg" / f"{s}.jpeg")


def run(root: Path, stub, argv: list[str]) -> list[dict]:
    run_catalogue.httpx.Client = lambda **kw: stub          # type: ignore[assignment]
    rc = run_catalogue.main(["--root", str(root)] + argv)
    assert rc == 0, f"exit code {rc}"
    report = next((root).glob("outputs_*/f6_report.json"))
    return json.loads(report.read_text())


def main() -> int:
    stems = ["fg01", "fg06", "fg12"]

    print("\n1. crops are per garment and actually crop")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); build_tree(root, stems)
        stub = StubClient()
        run(root, stub, ["--garments", ",".join(stems), "--tag", "t1"])
        sizes = []
        for p in sorted((root / "cache" / "crops" / "t1").glob("*.jpg")):
            sizes.append(Image.open(p).size)
        check("one crop per garment", len(sizes) == 3, str(sizes))
        check("crops are smaller than the source",
              all(w < 900 or h < 1400 for w, h in sizes))
        check("crops differ between garments (per-garment boxes)",
              len(set(sizes)) > 1, f"{len(set(sizes))} distinct sizes")

    print("\n2. the guardrail switch is transmitted")
    for flag, expect in (("on", "true"), ("off", "false"), ("default", None)):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); build_tree(root, ["fg01"])
            stub = StubClient()
            run(root, stub, ["--garments", "fg01", "--guardrail", flag, "--tag", flag])
            sent = stub.posts[0]["data"].get("guardrail")
            check(f"--guardrail {flag} sends {expect!r}", sent == expect, f"sent {sent!r}")

    print("\n3. an unchecked result is never reported as a pass")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); build_tree(root, ["fg01"])
        recs = run(root, StubClient(), ["--garments", "fg01", "--guardrail", "off",
                                        "--tag", "off2"])
        check("guardrail flag false", recs[0]["guardrail"] is False)
        check("guardrail_ok is null, not True", recs[0]["guardrail_ok"] is None,
              repr(recs[0]["guardrail_ok"]))

    print("\n4. failures and retries are recorded, not hidden")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); build_tree(root, stems)
        recs = run(root, StubClient(guardrail_ok=False, attempts=3),
                   ["--garments", ",".join(stems), "--guardrail", "on", "--tag", "t4"])
        check("every garment in the report", len(recs) == 3, str(len(recs)))
        check("failed verdict recorded", all(r["guardrail_ok"] is False for r in recs))
        check("attempts recorded", all(r["attempts"] == 3 for r in recs))
        check("reason recorded", all(r["guardrail_reason"] for r in recs))

    print("\n5. one bad garment does not abandon the rest")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); build_tree(root, stems)
        recs = run(root, StubClient(fail_on=("fg06",)),
                   ["--garments", ",".join(stems), "--tag", "t5"])
        ok = [r for r in recs if r["ok"]]
        bad = [r for r in recs if not r["ok"]]
        check("two succeeded", len(ok) == 2, str(len(ok)))
        check("one recorded as failed", len(bad) == 1 and bad[0]["garment"] == "fg06")

    print("\n6. prompts carry the pieces and the drape")
    for stem in ("fg01", "fg06", "fg12"):
        text = build(GARMENTS[stem])
        n = len(GARMENTS[stem]["pieces"])
        check(f"{stem} states its {n} pieces", f"made of {n} piece" in text)
        if GARMENTS[stem].get("drape"):
            check(f"{stem} has a drape clause", "THE DRAPE" in text)
    check("fg12 names both dupattas", "TWO dupattas" in build(GARMENTS["fg12"]))

    print("\n7. unknown garments are refused, not silently skipped")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); build_tree(root, ["fg01"])
        rc = run_catalogue.main(["--root", str(root), "--garments", "fg99",
                                 "--dry-run"])
        check("exit code 2 for an unknown garment", rc == 2, f"got {rc}")

    print("\n8. dry run touches no API")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); build_tree(root, stems)
        stub = StubClient()
        run_catalogue.httpx.Client = lambda **kw: stub      # type: ignore[assignment]
        run_catalogue.main(["--root", str(root), "--garments", ",".join(stems),
                            "--dry-run", "--tag", "dry"])
        check("no posts made", len(stub.posts) == 0, f"{len(stub.posts)} posts")
        check("prompts still written",
              len(list((root / "outputs_dry" / "prompts").glob("*.txt"))) == 3)

    print(f"\n{'=' * 60}\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        for f in FAILED:
            print(f"  FAILED: {f}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
