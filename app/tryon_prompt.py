"""Builds the instruction sent to the generator.

Three parts, in this order:

    ROLE         which image is the subject and which is the garment. Without
                 it the model has no way to know: both are just reference
                 latents, and a promptless run renders both figures, draws the
                 garment photo's mannequin as a person, and may take either
                 photo's background. Assigning roles fixed that 5/5.
    GARMENT      what the garment is and how it is worn, from the vision model.
                 The reference latent carries the garment's appearance well
                 enough on its own, but not its *construction* — a saree can
                 come out as a lehenga — and not reliably its colour.
    PRESERVE     face, body, pose, hands, scene.

Two rules learned the hard way, both worth stating plainly because they are
easy to undo by accident:

1.  This text encoder does not act on negation. "Any mannequin, showroom, wall,
    floor, furniture, plant or lamp must not appear" had *zero* measurable
    effect — the showroom still replaced the model's background on every image
    tested. Anything the output must not contain has to be removed from the
    input instead; the garment photo is cropped for exactly this reason.

2.  Constraints work as closed-set enumeration, not prohibition. "There is no
    third hand" invites a third hand. "This person has two hands and this is
    the complete list of them" does not. An earlier prompt listed a raised
    hand, a held object and a pose as three separate clauses and the model drew
    a hand per clause.
"""
from __future__ import annotations

import re

ROLE = (
    "The first image is a photograph of one person. The second image shows a "
    "garment. Edit the first photograph so that same person is wearing that "
    "garment.\n"
    "There is exactly ONE person in the result: the person from the first "
    "photograph."
)

PRESERVE = (
    "KEEP EXACTLY AS IN THE FIRST PHOTOGRAPH:\n"
    "FACE: the same face and the same expression, at the same age. It must read "
    "as the same person on the same day.\n"
    "BODY: the same build, height and proportions. The garment fits the body "
    "that is already in the photograph; the body does not change to suit the "
    "garment.\n"
    "POSE: the same stance, the same head angle, the same gaze, the same camera "
    "distance and crop. Both arms stay where they are. Anything being held is "
    "still held, in the same hand.\n"
    "HANDS: this person has TWO hands in total and this is the complete list of "
    "them — one left hand and one right hand, each with five separated fingers. "
    "Draw exactly those two.\n"
    "BACKGROUND: the same place, the same objects in the same positions, the "
    "same light direction and colour, the same shadows."
)

# Field order matters: type and drape first, because they determine the
# silhouette, then colour, then material detail.
GARMENT_FIELDS = ("TYPE", "PIECES", "DUPATTA", "HOW_WORN", "COLOURS", "METAL",
                  "FABRIC", "LENGTH")


def describe(garment: dict | None) -> str:
    """Render the vision model's garment reading into prompt lines.

    Skips fields the analyser left blank rather than emitting an empty label —
    a stray "COLOURS:" with nothing after it reads as an instruction to invent
    one. Returns "" when nothing is known, in which case the garment comes
    entirely from its reference latent.

    PIECES is stated as a closed set, in the same construction that fixed the
    three-hands defect. A garment component named inside a sentence is optional
    to the model — two-piece sets came back fused into one, and dupattas
    disappeared outright. Counted and numbered, with an explicit statement that
    every one appears, they survive. The count is repeated after the list
    because the number is the part the model can check itself against.
    """
    if not garment:
        return ""
    parts = []
    if t := garment.get("TYPE"):
        parts.append(f"The garment is a {t}.")

    pieces = (garment.get("PIECES") or "").strip()
    if pieces:
        n = _piece_count(pieces)
        parts.append(
            f"THE PIECES — this garment is made of {pieces}. "
            f"That is the complete list. "
            f"All {n if n else 'of those'} pieces appear in the result, each one "
            f"clearly separate from the others, worn together on the person.")

    dupatta = (garment.get("DUPATTA") or "").strip()
    if dupatta and dupatta.lower() not in ("none", "no", "n/a", "-"):
        parts.append(
            f"THE DUPATTA — this garment includes a dupatta, and it is present "
            f"in the result, worn exactly like this: {dupatta}")

    if w := garment.get("HOW_WORN"):
        parts.append(f"It sits on the body like this: {w}")
    if c := garment.get("COLOURS"):
        parts.append(f"Its colours are {c} — reproduce them exactly.")
    if m := garment.get("METAL"):
        parts.append(f"Metallic work is {m}.")
    if f := garment.get("FABRIC"):
        parts.append(f"The fabric is {f}.")
    if ln := garment.get("LENGTH"):
        parts.append(f"Length: {ln}.")
    if not parts:
        return ""
    return "THE GARMENT:\n" + " ".join(parts)


def _piece_count(pieces: str) -> str:
    """The leading integer of a "3 pieces: (1) ... " string, or ""."""
    hit = re.match(r"\s*(\d+)", pieces or "")
    return hit.group(1) if hit else ""


def build(garment: dict | None = None) -> str:
    """Assemble the full instruction."""
    blocks = [ROLE]
    if body := describe(garment):
        blocks.append(body)
    blocks.append(PRESERVE)
    blocks.append(
        "The second image contributes the garment alone.")
    return "\n\n".join(blocks)
