"""Generate, inspect, and regenerate until the result passes — or attempts run out.

This is the loop the 32 GB card makes possible. On a 24 GB box the generator and
the vision model could not both stay resident, so inspection had to be batched:
generate everything, unload, inspect everything, reload, regenerate the failures.
With both models resident an image can be judged the moment it is made and
retried immediately.

Order of checks is chosen for cost, cheapest first, short-circuiting:

    numeric     ~50 ms, no GPU     background diff, face count, face distance
    anatomy     ~2-3 s, 1 image    hands, fingers, arms, eyes, face, body
    fidelity    ~3-4 s, 3 images   same person/scene/pose, garment type, colour

A numeric rejection therefore never costs a VLM call.

On disagreements the numeric checks win. That is not a preference, it is
measured: the vision model failed one image for a replaced background that
measurably had not changed (0.082 against a 0.20 threshold), and misread a
gown as a saree. Retrying on those would burn GPU on images that were already
correct, so a VLM-only failure is advisory unless it lands in CRITICAL.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .guardrail import (ANATOMY_ASK, ANATOMY_CHECKS, ANATOMY_FIELDS,
                        COMBINED_ASK, COMBINED_CHECKS, COMBINED_FIELDS, CRITICAL,
                        FIDELITY_ASK, FIDELITY_CHECKS, FIDELITY_FIELDS,
                        numeric_gate, _failed)

log = logging.getLogger("qa")

# The numeric check that supersedes each VLM check when they disagree.
SUPERSEDED = {
    "SAME_BACKGROUND": "background_distance",
    "ONE_PERSON": "faces_after",
    "SAME_PERSON": "face_distance",
}


@dataclass
class Attempt:
    index: int
    seed: int
    steps: int
    seconds: float
    ok: bool
    reason: str
    stage: str
    measured: dict = field(default_factory=dict)
    anatomy: dict = field(default_factory=dict)
    fidelity: dict = field(default_factory=dict)
    overruled: list[str] = field(default_factory=list)
    image: bytes | None = None

    def report(self) -> dict:
        """Everything except the image bytes."""
        d = {k: v for k, v in self.__dict__.items() if k != "image"}
        return d


def inspect(backend, person: str, garment: str, result: str,
            combined: bool = True) -> dict:
    """Run every check on one result and return the verdict plus the evidence.

    combined=True asks all twelve checks in a single pass over the three
    images. The split path (anatomy alone, then fidelity) encodes the result
    image twice and asks the model to re-read the same scene for two halves of
    one question; merging removes that duplication. Set combined=False to fall
    back if a model handles the longer instruction poorly.
    """
    from .vision import parse

    ok, reason, measured = numeric_gate(person, result)
    out = {"measured": measured, "anatomy": {}, "fidelity": {},
           "overruled": [], "ok": ok, "reason": reason, "stage": "numeric"}
    if not ok:
        return out

    if combined:
        # 12 short labelled lines; 900 tokens was budgeting for prose that
        # never comes and costs decode time on every image.
        # A reasoning model needs room to think *and* answer — muse-glimmer
        # returned content=None at 700 tokens because it spent them all on
        # reasoning_content. Backends declare their own budget.
        budget = getattr(backend, "qa_tokens", 420)
        raw = backend.ask([person, garment, result], COMBINED_ASK,
                          max_new_tokens=budget)
        checks = parse(raw, COMBINED_FIELDS)
        # Fail closed. An unparseable answer is not a clean image — it is an
        # unknown one, and treating unknown as clean is how a guardrail ends up
        # certifying defects it actually detected. This exact case shipped a
        # whole 23-image run as "23/23 passed" while parsing zero fields.
        # A check with no PASS/FAIL word is not an opinion, it is a
        # non-answer — "GARMENT_TYPE: SAREE, SAREE" reads as clean to any
        # verdict test. Count those as missing so they fail closed too.
        from .guardrail import _VERDICT_TOKEN
        missing = [c for c in COMBINED_CHECKS
                   if c not in checks or not _VERDICT_TOKEN.search(checks[c] or "")]
        if missing:
            out.update(ok=False, stage="unparsed",
                       reason=f"guardrail answer unusable, {len(missing)} of "
                              f"{len(COMBINED_CHECKS)} checks missing: "
                              f"{','.join(missing[:4])}")
            out["raw"] = raw[:400]
            return out
        # Split back into the two buckets so reports stay the same shape
        # whichever path produced them.
        out["anatomy"] = {k: v for k, v in checks.items() if k in ANATOMY_CHECKS}
        out["fidelity"] = {k: v for k, v in checks.items()
                           if k in FIDELITY_CHECKS}
        bad = _failed(checks, COMBINED_CHECKS)
        stage = "combined"
    else:
        anatomy = parse(backend.ask(result, ANATOMY_ASK, max_new_tokens=700),
                        ANATOMY_FIELDS)
        out["anatomy"] = anatomy
        if bad := _failed(anatomy, ANATOMY_CHECKS):
            out.update(ok=False, stage="anatomy",
                       reason=f"{'+'.join(bad)}: "
                              f"{(anatomy.get('REASON') or anatomy.get(bad[0], ''))[:140]}")
            return out
        fidelity = parse(
            backend.ask([person, garment, result], FIDELITY_ASK,
                        max_new_tokens=900),
            FIDELITY_FIELDS)
        out["fidelity"] = fidelity
        checks = fidelity
        bad = _failed(fidelity, FIDELITY_CHECKS)
        stage = "fidelity"

    # Drop any complaint a measurement already contradicts.
    kept, overruled = [], []
    for check in bad:
        metric = SUPERSEDED.get(check)
        if metric and measured.get(metric) is not None:
            overruled.append(f"{check} (numeric {metric}={measured[metric]})")
        else:
            kept.append(check)
    out["overruled"] = overruled

    if kept:
        out.update(ok=False, stage=stage,
                   reason=f"{'+'.join(kept)}: {checks.get(kept[0], '')[:140]}")
        return out

    out.update(ok=True, stage="passed", reason="")
    return out


def worth_retrying(result: dict) -> bool:
    """Whether a failure is the kind a different seed might fix.

    Anything in CRITICAL is a defect in the image — a third hand, an invented
    person, the wrong garment type — and a new seed genuinely may not produce
    it. A soft failure (pose drift, colour a shade off) tends to reproduce
    across seeds, so retrying mostly buys a different imperfect image.
    """
    head = result.get("reason", "").split(":")[0]
    return any(c in head for c in CRITICAL)


def run(*, generate, backend, person: str, garment: str, out_dir: Path,
        tag: str, base_seed: int = 42, steps: int = 5,
        max_attempts: int = 3, speculate: bool = False
        ) -> tuple[Attempt, list[Attempt]]:
    """Generate until it passes. Returns (best, all_attempts).

    `generate(seed, steps) -> bytes` is injected so this loop is testable
    without a GPU and does not care whether the backend is ComfyUI or anything
    else.

    Always returns something: if every attempt fails, the best-scoring one comes
    back with its report attached rather than nothing at all. Callers wanting a
    hard rejection check `best.ok`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[Attempt] = []

    # Speculation is off by default, and that is a measured decision rather
    # than caution. Starting attempt N+1 while attempt N is inspected looked
    # free — different processes — but both are GPU-bound on one card, so they
    # timeshare rather than overlap. With a 23/23 first-attempt pass rate every
    # speculative generation was discarded, and contending for SMs pushed
    # guardrail time from 8.5s to 9.0s. It only pays when failures are common.
    pool = ThreadPoolExecutor(max_workers=1) if speculate else None
    pending: Future | None = None

    def seed_for(i: int) -> int:
        return base_seed + 1000 * i

    def steps_for(i: int) -> int:
        return steps + (1 if i == max_attempts - 1 and i > 0 else 0)

    try:
        for i in range(max_attempts):
            t0 = time.time()
            if pending is not None:
                image = pending.result()
            else:
                image = generate(seed=seed_for(i), steps=steps_for(i))
            gen_s = time.time() - t0

            if pool is not None and i + 1 < max_attempts:
                pending = pool.submit(generate, seed=seed_for(i + 1),
                                      steps=steps_for(i + 1))
            else:
                pending = None

            path = out_dir / f"{tag}_a{i}.png"
            path.write_bytes(image)

            verdict = inspect(backend, person, garment, str(path))
            att = Attempt(index=i, seed=seed_for(i), steps=steps_for(i),
                          seconds=round(gen_s, 2), image=image,
                          **{k: verdict[k] for k in
                             ("ok", "reason", "stage", "measured", "anatomy",
                              "fidelity", "overruled")})
            attempts.append(att)

            if att.ok:
                log.info("%s passed on attempt %d (seed %d)", tag, i + 1, att.seed)
                path.replace(out_dir / f"{tag}.png")
                return att, attempts

            log.info("%s attempt %d failed @%s: %s", tag, i + 1, att.stage,
                     att.reason)
            if not worth_retrying(verdict):
                log.info("%s: soft failure, not retrying", tag)
                break
    finally:
        # Never leave a speculative generation running into the next image.
        if pending is not None:
            pending.cancel()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    best = _best(attempts)
    (out_dir / f"{tag}.png").write_bytes(best.image or b"")
    return best, attempts


def _best(attempts: list[Attempt]) -> Attempt:
    """Least-bad attempt: passing first, then fewest failed checks, then
    the one that got furthest through the pipeline."""
    order = {"numeric": 0, "anatomy": 1, "fidelity": 2, "passed": 3}

    def score(a: Attempt) -> tuple:
        failed = len(_failed(a.anatomy, ANATOMY_CHECKS)) + \
                 len(_failed(a.fidelity, FIDELITY_CHECKS))
        return (a.ok, -failed, order.get(a.stage, 0))

    return max(attempts, key=score)
