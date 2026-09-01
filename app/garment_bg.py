"""Take the room out of a garment photograph automatically.

The reason this exists, stated plainly because it was expensive to learn: this
text encoder does not act on negation. Instructing the model to keep the studio
out has no effect, and a rewritten, much stronger prompt made no difference in a
controlled test - the same person, the same garment, the same prompt, and the
garment photo's gallery room still replaced the subject's event hall. Cropping
the room out of the input, changing nothing else, preserved the hall completely.

So the room must leave the image. Cropping by hand works and is what the
catalogue does, but it cannot be asked of someone uploading a photo through a
web page. This removes the background instead, and composites the garment onto
a flat neutral field.

Neutral grey rather than white: white reads as a bright studio and the model
sometimes carries that brightness into the scene, while mid-grey is closer to
"no information" and is what the garment's own reference latent is left to
speak over.

rembg is optional. If it is not installed the original image is returned
unchanged and `applied` is False - a missing dependency must degrade to "no
preprocessing", never to a silent crop that might cut a dupatta off.
"""
from __future__ import annotations

import logging
from io import BytesIO

from PIL import Image

log = logging.getLogger("garment_bg")

_SESSION = None
_UNAVAILABLE = False
NEUTRAL = (128, 128, 128)


def available() -> bool:
    """Whether background removal can run, without importing it twice."""
    global _SESSION, _UNAVAILABLE
    if _SESSION is not None:
        return True
    if _UNAVAILABLE:
        return False
    try:
        from rembg import new_session
        _SESSION = new_session("u2net")
        log.info("background removal ready (u2net)")
        return True
    except Exception as exc:                              # noqa: BLE001
        _UNAVAILABLE = True
        log.info("background removal unavailable (%s); garments pass through",
                 type(exc).__name__)
        return False


def strip(data: bytes, *, pad: int = 24, min_coverage: float = 0.04
          ) -> tuple[bytes, dict]:
    """Remove the background and centre the garment on a neutral field.

    Returns (jpeg_bytes, info). info["applied"] says whether anything changed,
    so a caller can report honestly rather than implying a step that did not run.
    """
    info: dict = {"applied": False}
    if not available():
        info["reason"] = "rembg not installed"
        return data, info

    from rembg import remove

    try:
        src = Image.open(BytesIO(data)).convert("RGBA")
        cut = remove(src, session=_SESSION)
    except Exception as exc:                              # noqa: BLE001
        log.warning("background removal failed: %s", exc)
        info["reason"] = f"{type(exc).__name__}: {exc}"
        return data, info

    alpha = cut.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        info["reason"] = "nothing found"
        return data, info

    w, h = src.size
    coverage = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / float(w * h)
    # A tiny mask means the model latched onto a prop rather than the garment.
    # Passing the original through is the safe failure: a garment with its room
    # still attached is a worse image, but a garment cropped down to a necklace
    # is not a garment at all.
    if coverage < min_coverage:
        info.update(reason=f"mask covers only {coverage:.1%}", coverage=coverage)
        return data, info

    box = (max(0, bbox[0] - pad), max(0, bbox[1] - pad),
           min(w, bbox[2] + pad), min(h, bbox[3] + pad))
    cut = cut.crop(box)
    flat = Image.new("RGB", cut.size, NEUTRAL)
    flat.paste(cut, mask=cut.getchannel("A"))

    out = BytesIO()
    flat.save(out, "JPEG", quality=95)
    info.update(applied=True, coverage=round(coverage, 3),
                original=[w, h], cropped=list(cut.size))
    log.info("garment background removed: %sx%s -> %sx%s (mask %.0f%%)",
             w, h, cut.size[0], cut.size[1], coverage * 100)
    return out.getvalue(), info
