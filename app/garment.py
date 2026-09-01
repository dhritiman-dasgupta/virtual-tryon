"""Garment analysis — what the garment is, and how it is worn.

The reference latent already carries the garment's appearance, so this is not
about describing pixels. It exists for the two things the latent does not
reliably transfer:

    construction    a saree is a continuous unstitched drape; a lehenga is a
                    stitched skirt with a waistband. They photograph almost
                    identically on a mannequin and the generator will happily
                    swap one for the other.
    colour          a black net lehenga came out gold with no text prompt.

Cached on disk by file hash: the same garment is analysed once ever, across
restarts and across every model it is paired with. For a 172-pair run that is
34 analyses instead of 172.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

log = logging.getLogger("garment")

# Construction, not silhouette — the discriminator has to be something the
# model can see and that actually separates the classes.
TAXONOMY = """Garment types, and how to tell them apart by construction:

SAREE: one long unstitched cloth wrapped around the body, pleated and tucked at
  the waist, with the loose end (pallu) carried up across the torso and over one
  shoulder. Worn with a short fitted blouse. There is NO separate stitched skirt
  and no waistband — the drape is continuous from waist to shoulder.
LEHENGA: a separate stitched flared skirt with its own waistband, worn with a
  short blouse (choli) and usually a separate dupatta scarf. The skirt and the
  top are two different garments with a visible join at the waist.
ANARKALI: a single stitched long frock-style kurta, fitted at the bust and
  flaring from below it to floor length. One continuous stitched piece.
SALWAR KAMEEZ: a stitched tunic worn over stitched trousers.
SHARARA or GHARARA: a short kurta worn over very wide flared trousers.
GOWN: a single stitched floor-length dress, western cut.
SUIT: a tailored jacket with matching trousers, worn over a shirt.
SEPARATES: shirt, kurta, jacket, polo, tee, trousers or shorts worn as
  independent pieces."""

ASK = f"""Look at this photograph of a garment and describe the garment itself.
Ignore the mannequin, hanger, room, floor, props and any on-screen graphics —
they are not part of the garment.

{TAXONOMY}

Answer with exactly these labelled lines and nothing else:

TYPE: (one type from the list above, using the construction rules)
PIECES: (count the separate pieces of clothing, then list them numbered. A
  dupatta, scarf, stole, jacket or cape is its own piece — count it. Write it
  exactly like this: "3 pieces: (1) fitted blouse, (2) flared skirt,
  (3) dupatta". If it is a single stitched garment write "1 piece: (1) gown".)
DUPATTA: (if the garment includes a dupatta, scarf or stole, say which shoulder
  it crosses, whether it is draped, pinned or held over the arm, and where its
  ends fall. If there is none, write exactly "none".)
PIECE_1: (describe the FIRST piece on its own, as if the others were not there:
  what it is, its colour, its fabric, its cut, where it sits on the body, and
  its embroidery. One sentence.)
PIECE_2: (the same for the SECOND piece, or exactly "none" if there is no
  second piece.)
PIECE_3: (the same for the THIRD piece, or exactly "none".)
PIECE_4: (the same for the FOURTH piece, or exactly "none".)
HOW_WORN: (one sentence on how it sits on the body — where it fastens or tucks,
  which shoulder any drape crosses, where the hem falls)
COLOURS: (the main colour first, then any secondary colours, in plain words)
METAL: (the tone of any metallic thread, sequins or stones — gold, silver,
  antique gold, rose gold, or none)
FABRIC: (silk, net, georgette, velvet, cotton, brocade, chiffon or similar)
LENGTH: (floor-length, ankle-length, calf-length, knee-length or short)"""

# PIECES and DUPATTA exist because the generator drops garment components the
# prompt does not force it to account for: two-piece sets fused into one, and
# dupattas vanishing outright. Same failure the hand enumeration fixed — a
# component mentioned inside a sentence is optional to the model; a numbered
# closed list is not.
# Per-piece descriptions exist because a garment described as a whole loses its
# construction: a lehenga becomes "a long embroidered dress" and renders as a
# plain skirt. Described one piece at a time, each component has to be accounted
# for on its own, which is the same closed-set principle that fixed the hands.
FIELDS = ["TYPE", "PIECES", "DUPATTA", "PIECE_1", "PIECE_2", "PIECE_3",
          "PIECE_4", "HOW_WORN", "COLOURS", "METAL", "FABRIC", "LENGTH"]


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class GarmentCache:
    """Disk-backed, keyed by content hash so renames and copies still hit."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            # A truncated cache entry must not poison every future run.
            log.warning("discarding unreadable cache entry %s", p.name)
            p.unlink(missing_ok=True)
            return None

    def put(self, key: str, value: dict) -> None:
        # Write-then-rename: a crash mid-write leaves the old entry, not a
        # half-written one.
        tmp = self._path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(value, indent=2))
        tmp.replace(self._path(key))


def ask_version() -> str:
    """Short hash of the question itself.

    Part of the cache key, because the cache is otherwise keyed only by the
    image — so changing the question would silently keep serving answers to the
    old one. Adding PIECES and DUPATTA to the ask would have read back blank on
    every garment already analysed, and the fix would have appeared not to work.
    """
    return hashlib.sha256(ASK.encode()).hexdigest()[:8]


def analyse(backend, path: str | Path, cache: GarmentCache | None = None) -> dict:
    """Read a garment photo. Returns the parsed fields, cached by content+ask."""
    from .vision import parse  # local import keeps this module import-light

    key = f"{file_hash(path)}-{ask_version()}"
    if cache and (hit := cache.get(key)) is not None:
        return hit

    raw = backend.ask(str(path), ASK, max_new_tokens=400)
    fields = parse(raw, FIELDS)

    # The type is the field the rest of the pipeline reasons about, so
    # normalise it to a known label instead of storing whatever prose came back.
    fields["TYPE"] = _normalise_type(fields.get("TYPE", ""))
    if cache:
        cache.put(key, fields)
    return fields


_KNOWN = ["SAREE", "LEHENGA", "ANARKALI", "SALWAR KAMEEZ", "SHARARA",
          "GHARARA", "GOWN", "SUIT", "SEPARATES"]


def _normalise_type(text: str) -> str:
    """Map free text onto a known type, or return it cleaned if unrecognised.

    Models answer "LEHENGA", "a lehenga", "Lehenga (with dupatta)" and
    "lehenga choli" for the same thing. Comparing raw strings would make the
    GARMENT_TYPE check fail on wording rather than on the garment.
    """
    up = (text or "").upper()
    for known in _KNOWN:
        if re.search(rf"\b{known}\b", up):
            return known
    # "lehenga choli" and "saree/sari" spellings
    if "SARI" in up:
        return "SAREE"
    if "CHOLI" in up:
        return "LEHENGA"
    return re.sub(r"\s+", " ", text or "").strip()[:40]
