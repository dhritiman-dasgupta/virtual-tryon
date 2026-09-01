"""Brochure: read every garment once and write its spec.

This is the expensive-model phase. It runs alone on the card with the 32B, and
its output is cached to disk, so the cost is paid once per garment ever - 34
analyses for a 172-pair catalogue, and zero on the next run.

The cache key includes the model id. Without it, re-reading the catalogue with
a better model silently returns the previous model's answers, and the upgrade
appears to do nothing. That is not hypothetical: the key already had to grow a
question-hash for the same reason when PIECES and DUPATTA were added.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .garment import ASK, FIELDS, GarmentCache, ask_version, file_hash, _normalise_type

log = logging.getLogger("brochure")


def cache_key(path: str | Path, model_id: str) -> str:
    """content + question + model. All three change the answer."""
    import hashlib
    model_tag = hashlib.sha256(model_id.encode()).hexdigest()[:6]
    return f"{file_hash(path)}-{ask_version()}-{model_tag}"


@dataclass
class BrochureJob:
    job_id: str
    total: int
    status: str = "queued"          # queued | running | succeeded | failed
    done: int = 0
    cached: int = 0
    analysed: int = 0
    failed: list[str] = field(default_factory=list)
    specs: dict[str, dict] = field(default_factory=dict)
    error: str | None = None
    started: float = field(default_factory=time.time)
    seconds: float | None = None

    def info(self) -> dict:
        return {
            "job_id": self.job_id, "status": self.status,
            "total": self.total, "done": self.done,
            "cached": self.cached, "analysed": self.analysed,
            "failed": self.failed, "error": self.error,
            "seconds": self.seconds,
            "progress": round(self.done / self.total, 3) if self.total else 1.0,
        }


class BrochureRunner:
    """Runs brochure jobs one at a time; the GPU holds a single model."""

    def __init__(self, cache_dir: Path):
        self.cache = GarmentCache(Path(cache_dir))
        self.jobs: dict[str, BrochureJob] = {}
        self._lock = asyncio.Lock()

    def get(self, job_id: str) -> BrochureJob | None:
        return self.jobs.get(job_id)

    def cached_specs(self, model_id: str, paths: list[Path]) -> dict[str, dict]:
        out = {}
        for p in paths:
            hit = self.cache.get(cache_key(p, model_id))
            if hit is not None:
                out[p.stem] = hit
        return out

    async def run(self, vision, model_id: str, paths: list[Path],
                  force: bool = False) -> BrochureJob:
        job = BrochureJob(job_id=uuid.uuid4().hex[:12], total=len(paths))
        self.jobs[job.job_id] = job
        async with self._lock:
            job.status = "running"
            try:
                for p in paths:
                    key = cache_key(p, model_id)
                    hit = None if force else self.cache.get(key)
                    if hit is not None:
                        job.specs[p.stem] = hit
                        job.cached += 1
                    else:
                        try:
                            spec = await asyncio.to_thread(
                                self._analyse, vision, p)
                        except Exception as e:           # noqa: BLE001
                            log.warning("%s failed: %s", p.stem, e)
                            job.failed.append(p.stem)
                            job.done += 1
                            continue
                        self.cache.put(key, spec)
                        job.specs[p.stem] = spec
                        job.analysed += 1
                    job.done += 1
                job.status = "succeeded"
            except Exception as e:                       # noqa: BLE001
                job.status, job.error = "failed", f"{type(e).__name__}: {e}"
                log.exception("brochure job %s failed", job.job_id)
            finally:
                job.seconds = round(time.time() - job.started, 1)
        log.info("brochure %s: %d analysed, %d cached, %d failed in %.1fs",
                 job.job_id, job.analysed, job.cached, len(job.failed),
                 job.seconds or 0.0)
        return job

    @staticmethod
    def _analyse(vision, path: Path) -> dict:
        from .vision import parse
        raw = vision.ask(str(path), ASK, max_new_tokens=400)
        fields = parse(raw, FIELDS)
        # The type is what the rest of the pipeline reasons about, so store a
        # known label rather than whatever prose came back.
        fields["TYPE"] = _normalise_type(fields.get("TYPE", ""))
        return fields


# --------------------------------------------------------------- to a prompt
# The vision model returns labelled fields; the prompt builder wants a spec of
# the shape hand-written in pipeline/garments.py. Bridging to that
# builder rather than writing a second prompt format is deliberate: it is the
# construction that measured best (12/12 dupattas present, 8/8 sarees keeping
# their construction), and it carries the piece enumeration and the waist-join
# clause that stop a lehenga rendering as a plain skirt.

_PIECE_SPLIT = __import__("re").compile(r"\(\d+\)\s*")


def parse_pieces(text: str) -> list[str]:
    """Turn "3 pieces: (1) blouse, (2) skirt, (3) dupatta" into a list.

    Falls back to splitting on commas, because models drop the numbering often
    enough that a strict parser would throw away a correct reading.
    """
    text = (text or "").strip()
    if not text:
        return []
    body = text.split(":", 1)[1] if ":" in text else text
    parts = [p.strip(" ,.;") for p in _PIECE_SPLIT.split(body) if p.strip(" ,.;")]
    if len(parts) < 2:
        parts = [p.strip(" ,.;") for p in body.split(",") if p.strip(" ,.;")]
    # Strip any surviving "(1)" markers: the comma fallback does not consume
    # them, and a piece called "(1) gown" reads as an artefact in the prompt.
    parts = [_PIECE_SPLIT.sub("", p).strip(" ,.;") for p in parts]
    return [p for p in parts if len(p) > 2][:6]


_LEAD = __import__("re").compile(r"^(a|an|the)\s+", __import__("re").I)


def short_name(detail: str) -> str:
    """The garment piece's name, taken from the head of its description.

    "A fitted blouse with long sleeves, in dusty rose fabric..." -> "fitted
    blouse". Needed because the vision model often answers PIECES with just a
    count ("3") while describing each piece perfectly, and the piece NAMES are
    what the waist-join rule keys on - without them a lehenga loses the clause
    that stops it rendering as a single panel.
    """
    text = (detail or "").strip()
    for sep in (",", " with ", " in ", " that ", " which "):
        i = text.find(sep)
        if i > 0:
            text = text[:i]
            break
    return _LEAD.sub("", text).strip(" .").lower()[:48]


def to_spec(fields: dict, crop: tuple[float, float, float, float] | None = None
            ) -> dict:
    """Vision-model fields -> a spec the prompt builder understands."""
    gtype = (fields.get("TYPE") or "garment").strip()
    pieces = parse_pieces(fields.get("PIECES", ""))
    detail_names = [short_name(d) for d in (
        (fields.get(f"PIECE_{i}") or "").strip() for i in range(1, 5))
        if d and d.lower() not in ("none", "n/a", "-")]
    detail_names = [n for n in detail_names if len(n) > 2]
    # Prefer the names taken from the descriptions whenever there are more of
    # them: the model reliably describes each piece, and only sometimes lists
    # them.
    if len(detail_names) > len(pieces):
        pieces = detail_names
    if not pieces:
        # A garment with no piece list still has to generate. One piece named
        # after its type is honest about what is known, and the closed-set
        # sentence then simply says "1 piece" rather than inventing components.
        pieces = [f"a {gtype.lower()}"]

    colours = (fields.get("COLOURS") or "").strip()
    detail = " ".join(x for x in [
        (fields.get("HOW_WORN") or "").strip(),
        f"The fabric is {fields['FABRIC'].strip()}." if fields.get("FABRIC") else "",
        f"Metallic work is {fields['METAL'].strip()}." if fields.get("METAL") else "",
        f"It is {fields['LENGTH'].strip()}." if fields.get("LENGTH") else "",
    ] if x)

    dup = (fields.get("DUPATTA") or "").strip()
    drape = ""
    if dup and dup.lower() not in ("none", "no", "n/a", "-", "there is none"):
        drape = (f"this garment includes a dupatta and it is present in the "
                 f"result, worn exactly like this: {dup}")

    # Each piece described on its own. A garment summarised as a whole loses
    # its construction; enumerated and described separately, every component
    # has to be accounted for.
    piece_details = [
        (fields.get(f"PIECE_{i}") or "").strip()
        for i in range(1, 5)
    ]
    piece_details = [d for d in piece_details
                     if d and d.lower() not in ("none", "n/a", "-")]

    from pipeline.garments import PRESERVE_GENERIC

    return {
        # Never the catalogue's model-6 block: that one names her night garden,
        # her sindoor and her brown phone, and asserting those onto someone
        # else's photograph describes a scene that is not there.
        "preserve": PRESERVE_GENERIC,
        "piece_details": piece_details,
        "summary": f"a {gtype.lower()}" + (f" in {colours}" if colours else ""),
        "pieces": pieces,
        "detail": detail or f"A {gtype.lower()}.",
        "drape": drape,
        "colours": colours or "as shown in the second image",
        "crop": crop or (0.0, 0.0, 1.0, 1.0),
        "_fields": fields,
    }


def spec_to_prompt(spec: dict) -> str:
    """Render a spec with the same builder the hand-written catalogue uses."""
    import sys
    from pathlib import Path
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from pipeline.garments import build
    return build(spec)


# ------------------------------------------------------- the person, per batch
# The catalogue's 23 results kept their background because the prompt named the
# actual surroundings - the palms, the car, the vase. That only worked because
# someone had written model 6's scene down by hand.
#
# For an unseen person nothing can be written down in advance, so the same
# vision model that reads the garments reads the person once per batch and the
# preserve block is built from what it saw. Nothing about the subject is
# hardcoded: no scene, no pose, no accessories, no assumption that both hands
# hold a phone.

def person_preserve(fields: dict) -> str:
    """Build a preserve block from a reading of this particular photograph."""
    def got(key: str) -> str:
        v = (fields.get(key) or "").strip()
        return "" if v.lower() in ("", "none", "n/a", "-", "not visible") else v

    lines = ["KEEP EXACTLY AS IN THE FIRST PHOTOGRAPH:"]
    if face := got("FACE"):
        lines.append(f"FACE: {face} It reads as the same person photographed on "
                     f"the same day.")
    else:
        lines.append("FACE: the same face and the same expression, at the same age.")
    if hair := got("HAIR"):
        lines.append(f"HAIR: {hair}")
    if build := got("BUILD"):
        lines.append(f"BODY: {build} The garment is fitted to the body already "
                     f"in the photograph.")
    if pose := got("POSE"):
        lines.append(f"POSE: {pose}")

    # Hands are enumerated as a closed set whatever they are doing. Naming what
    # each one holds is what keeps a held object held; asserting a phone in both
    # hands, as the catalogue block does, would be wrong for almost everyone.
    left, right = got("LEFT_HAND"), got("RIGHT_HAND")
    hands = ("HANDS: this person has TWO hands in total and this is the "
             "complete list of them - one left hand and one right hand, each "
             "with five separated fingers. Draw exactly those two.")
    if left:
        hands += f" The left hand: {left}"
    if right:
        hands += f" The right hand: {right}"
    lines.append(hands)

    if acc := got("ACCESSORIES"):
        lines.append(f"JEWELLERY AND ACCESSORIES: {acc} All of it stays exactly "
                     f"as it is.")
    if setting := got("SETTING"):
        lines.append(f"BACKGROUND: the same place, unchanged - {setting} Every "
                     f"object stays where it is, at the same size, in the same "
                     f"light, casting the same shadows.")
    else:
        lines.append("BACKGROUND: the same place. Every object behind and beside "
                     "the person stays where it is, in the same light.")
    if framing := got("FRAMING"):
        lines.append(f"FRAMING: {framing}")
    return "\n".join(lines)


def read_person(vision, path) -> dict:
    """Read the person photograph once. Charged per batch, not per garment."""
    from .person_prompt import MODEL_ASK, MODEL_FIELDS
    from .vision import parse
    raw = vision.ask(str(path), MODEL_ASK, max_new_tokens=520)
    return parse(raw, MODEL_FIELDS)
