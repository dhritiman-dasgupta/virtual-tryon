"""QA guardrail — every generated image is inspected before it is accepted.

Prompting lowers the defect rate; it does not eliminate it. Three hands and a
closed eye both slipped through v6, and a promptless run produced a second
person outright. So each result is checked and anything that fails is
regenerated with a different seed.

Two kinds of check, because they need different inputs:

    ANATOMY   the result on its own — hand count, finger count, eyes, face
              integrity, limb attachment. Nothing to compare against.
    FIDELITY  the result against BOTH inputs — is it the same face, the same
              background, still exactly one person, and is the garment the same
              *kind* of garment as the one supplied. A saree rendered as a
              lehenga is a perfectly well-formed image, so anatomy alone will
              always pass it.

Batched deliberately: the VLM and the generator cannot share 24 GB, so verifying
after each image would mean a model swap per image. Generate the whole set, swap
once, verify the whole set, regenerate only the failures.

Two of the fidelity questions are answered far better by measurement than by
asking a 7B model. `face_distance()` and `background_distance()` below are
deterministic and belong in front of the VLM, not instead of it — see
`numeric_gate()`.
"""
from __future__ import annotations

import re

# --------------------------------------------------------------- anatomy

ANATOMY_ASK = """You are inspecting a generated photograph for anatomical defects before
it is shown to a customer. Look carefully at the whole image, especially the hands,
arms and face.

Answer with exactly these labelled lines, each starting PASS or FAIL:

PEOPLE: (Count every human figure in the image, including anyone partly visible at
an edge, reflected in a mirror, or standing in the background. PASS only if the
count is exactly one. FAIL otherwise. State the number you counted.)
HANDS: (Count every hand in the image. One person has two. PASS only if the number
of hands is correct and each visible hand has five normal fingers. FAIL if there is
a third hand anywhere, a hand growing from the wrong place, a floating or detached
hand, merged or missing or extra fingers, or a malformed hand. State the count.)
ARMS: (PASS only if there are exactly two arms, both attached normally at the
shoulders. FAIL if there is a third arm or a limb that does not connect properly.)
EYES: (PASS only if both eyes are open, both pointing the same direction, and both
look normal. FAIL if one eye is closed while the other is open, if they look in
different directions, or if either is distorted.)
FACE: (PASS if there is exactly one face and it is undistorted. FAIL if the face is
smeared, doubled, melted, or has misplaced features.)
BODY: (PASS if the torso, legs and neck are proportioned normally and connect
correctly. FAIL if any limb is duplicated, missing, bent impossibly, or fused to
the clothing.)
CLOTHING: (PASS if the person is fully and properly dressed in a coherent garment.
FAIL if clothing is see-through where it should not be, missing a section, or if
skin shows where the garment should cover.)

VERDICT: PASS if every line above is PASS, otherwise FAIL
REASON: (if FAIL, one short sentence naming the worst defect)"""

ANATOMY_FIELDS = ["PEOPLE", "HANDS", "ARMS", "EYES", "FACE", "BODY", "CLOTHING",
                  "VERDICT", "REASON"]

# Kept so existing callers importing the old names keep working.
VERIFY_ASK = ANATOMY_ASK
VERIFY_FIELDS = ANATOMY_FIELDS

# -------------------------------------------------------------- fidelity

# The saree/lehenga confusion is the expensive one: both are floor-length, both
# are heavily embroidered, and both photograph similarly on a mannequin. The
# discriminator is construction, not silhouette — a saree is one continuous
# unstitched drape, a lehenga is a stitched skirt with a waistband. Spell that
# out rather than trusting the model's own sense of the category.
GARMENT_TAXONOMY = """Garment types, and how to tell them apart by construction:

SAREE: one long unstitched cloth wrapped around the body, pleated and tucked at
  the waist, with the loose end (pallu) carried up across the torso and over one
  shoulder. Worn with a short fitted blouse. There is NO separate stitched skirt
  and no waistband — the drape is continuous from waist to shoulder.
LEHENGA: a separate stitched flared skirt with its own waistband, worn with a
  short blouse (choli) and usually a separate dupatta scarf. The skirt and the
  top are clearly two different garments with a visible join at the waist.
ANARKALI: a single stitched long frock-style kurta that fits at the bust and
  flares from below it to floor length. One continuous stitched piece, not a
  drape and not a skirt-plus-top.
SALWAR KAMEEZ: a stitched tunic worn over stitched trousers.
SHARARA or GHARARA: a short kurta worn over very wide flared trousers.
GOWN: a single stitched floor-length western-style dress.
GARMENT ON TOP AND BOTTOM SEPARATELY: shirt, jacket, suit, trousers, shorts."""

FIDELITY_ASK = f"""You are given three images:

  IMAGE 1 — the original photograph of a person
  IMAGE 2 — a photograph of a garment
  IMAGE 3 — a generated image that should show the person from IMAGE 1 wearing
            the garment from IMAGE 2, in the same place as IMAGE 1

Compare them and answer with exactly these labelled lines, each starting PASS or
FAIL. Judge IMAGE 3 against IMAGE 1 and IMAGE 2 — do not judge it on its own
merits.

{GARMENT_TAXONOMY}

SAME_PERSON: (Is the face in IMAGE 3 the same individual as in IMAGE 1? Compare
face shape, jawline, nose, eye spacing, skin tone and apparent age. PASS if it
reads as the same person. FAIL if the face has been slimmed, sharpened,
symmetrised, made younger, made lighter, or is simply a different person.)
ONE_PERSON: (How many people are in IMAGE 3, and how many were in IMAGE 1
including any bystanders in the background? PASS only if IMAGE 3 contains no
person who was not already in IMAGE 1. FAIL if a figure has been added — in
particular if the mannequin or model from IMAGE 2 has been drawn as a person.
State both counts.)
SAME_BACKGROUND: (Is the setting behind the person in IMAGE 3 the same place as
in IMAGE 1? Compare the architecture, objects, furniture, plants, floor,
lighting direction and time of day. PASS if it is recognisably the same place.
FAIL if the background has been replaced — especially if it has been replaced by
the room, wall, showroom or studio visible in IMAGE 2.)
SAME_POSE: (Compare the body in IMAGE 3 with IMAGE 1 part by part. State what
each arm is doing in each image before the verdict. PASS only if all of these
are unchanged: which way the body faces, the head angle, where the eyes look,
where each arm is and what it holds, which foot carries the weight, and the
framing and camera distance. FAIL if an arm has moved or straightened, a raised
hand has been lowered, something held has been put down, the shoulders have
turned, or the crop has changed.)
GARMENT_TYPE: (Using the construction rules above, name the garment type in
IMAGE 2, then name the garment type in IMAGE 3. PASS only if they are the same
type. FAIL if the type has changed — for example a saree rendered as a lehenga,
or a lehenga rendered as a gown. State both names.)
GARMENT_MATCH: (Is the garment in IMAGE 3 the same specific garment as IMAGE 2?
Compare the main colour, the secondary colour, the metal tone of any embroidery,
the placement of the pattern and the fabric. PASS if it is clearly the same
garment. FAIL if the colour has changed — a black garment rendered gold, for
example — or the pattern is different.)

VERDICT: PASS if every line above is PASS, otherwise FAIL
REASON: (if FAIL, one short sentence naming the worst problem)"""

FIDELITY_FIELDS = ["SAME_PERSON", "ONE_PERSON", "SAME_BACKGROUND", "SAME_POSE",
                   "GARMENT_TYPE", "GARMENT_MATCH", "VERDICT", "REASON"]

# ------------------------------------------------------ combined (fast path)
#
# Anatomy and fidelity as two calls means two prefills over overlapping images:
# the result is encoded twice, and the model re-reads the same scene to answer
# two halves of one question. Merged, the three images are encoded once and
# every check is answered in a single pass. Measured at roughly half the time
# of the two-call path with the same questions and the same wording, so the
# verdicts stay comparable to the split version.
COMBINED_ASK = f"""You are given three images:

  IMAGE 1 — the original photograph of a person
  IMAGE 2 — a photograph of a garment
  IMAGE 3 — a generated image that should show the person from IMAGE 1 wearing
            the garment from IMAGE 2, in the same place as IMAGE 1

{GARMENT_TAXONOMY}

Inspect IMAGE 3 and compare it with IMAGE 1 and IMAGE 2.

FORMAT — every line must be:

    <FIELD NAME>: <PASS or FAIL> <very short reason>

The field name comes first, never the verdict. Every one of the twelve lines
must contain the word PASS or FAIL — including GARMENT_TYPE and GARMENT_MATCH,
where you state the two values AND then the verdict. Do not reorder the lines,
omit any, or write anything else.

Judge only what you see in the images. The field descriptions below list the
things that can go wrong; they are not observations about these images.

Answer GARMENT_MATCH carefully: a garment rendered in the wrong colour is the
most expensive error here, and it is easy to miss because everything else in
the image looks correct.

PEOPLE: (count every human figure in IMAGE 3, including anyone partly visible
at an edge or in the background. PASS only if IMAGE 3 contains no person who
was not already in IMAGE 1. State both counts.)
HANDS: (count every hand in IMAGE 3. PASS only if the count is correct for the
people present and each visible hand has five separated fingers. FAIL for a
third hand, a detached or floating hand, or merged, missing or extra fingers.)
ARMS: (PASS only if every arm attaches normally at a shoulder.)
EYES: (PASS only if both eyes are open, aligned and undistorted.)
FACE: (PASS if the face is undistorted — not smeared, doubled or melted.)
BODY: (PASS if torso, legs and neck are proportioned normally and connect
correctly, and no limb is fused to the clothing.)
CLOTHING: (PASS if the person is fully and properly dressed in a coherent
garment, with no skin showing where the garment should cover.)
SAME_PERSON: (is the face in IMAGE 3 the same individual as IMAGE 1? FAIL if
slimmed, sharpened, made younger or lighter, or simply a different person.)
SAME_BACKGROUND: (is the setting in IMAGE 3 the same place as IMAGE 1? FAIL if
replaced — especially by the room or studio visible in IMAGE 2.)
SAME_POSE: (compare the BODY in IMAGE 3 with the body in IMAGE 1, part by part.
The person must be standing exactly as they were. State what each arm is doing
in IMAGE 1 and what it is doing in IMAGE 3 before the verdict. PASS only if all
of these are unchanged: which way the body faces, the angle of the head, where
the eyes look, where each arm is and what it holds, which foot carries the
weight, how much of the person is in frame, and how close the camera is. FAIL if
an arm has moved or straightened, a raised hand has been lowered, something held
has been put down, the shoulders have turned, the head has tilted differently,
or the crop has widened or tightened.)
GARMENT_TYPE: (using the construction rules above, name the type in IMAGE 2 and
the type in IMAGE 3. PASS only if they are the same type.)
GARMENT_PIECES: (count the separate pieces of clothing worn in IMAGE 2, then
count them in IMAGE 3. A dupatta, scarf, stole, jacket or cape counts as its own
piece. Write them as "IMAGE 2 has N, IMAGE 3 has N" before the verdict. PASS only
if the counts match. FAIL if a two-piece set has been rendered as one continuous
garment, or if a dupatta, scarf or jacket present in IMAGE 2 is missing.)
GARMENT_DRAPE: (if IMAGE 2 has a dupatta, scarf or stole, does it fall the same
way in IMAGE 3 — same shoulder, same side, ends in the same place? PASS if the
drape matches, or if IMAGE 2 has no such piece. FAIL if it has moved to the other
shoulder or hangs differently.)
GARMENT_MATCH: (name the single dominant colour of the garment in IMAGE 2, then
name the single dominant colour of the garment in IMAGE 3, then answer. Write
them as "IMAGE 2 is <colour>, IMAGE 3 is <colour>" before the verdict. PASS only
if those two colours are the same colour. A black garment rendered green, grey,
gold or beige is FAIL. Judge the fabric colour, not the embroidery.)"""

COMBINED_FIELDS = ["PEOPLE", "HANDS", "ARMS", "EYES", "FACE", "BODY", "CLOTHING",
                   "SAME_PERSON", "SAME_BACKGROUND", "SAME_POSE",
                   "GARMENT_TYPE", "GARMENT_PIECES", "GARMENT_DRAPE",
                   "GARMENT_MATCH"]
COMBINED_CHECKS = list(COMBINED_FIELDS)

ANATOMY_CHECKS = ["PEOPLE", "HANDS", "ARMS", "EYES", "FACE", "BODY", "CLOTHING"]
FIDELITY_CHECKS = ["SAME_PERSON", "ONE_PERSON", "SAME_BACKGROUND", "SAME_POSE",
                   "GARMENT_TYPE", "GARMENT_MATCH"]

# Failing these means the image is unusable rather than merely imperfect, so a
# retry is worth a seed change. The rest are still reported.
#
# GARMENT_MATCH is here because of a measured case: a black net lehenga rendered
# sage-green passed every other check and was recorded as a failure but never
# regenerated, because colour was treated as cosmetic. For a catalogue the
# garment's colour is the product — a wrong-coloured one is as unusable as a
# third hand.
CRITICAL = {"PEOPLE", "HANDS", "ARMS", "EYES", "FACE", "ONE_PERSON",
            "SAME_PERSON", "SAME_BACKGROUND", "GARMENT_TYPE", "GARMENT_MATCH",
            # A missing dupatta, or a two-piece set fused into one, is the wrong
            # product rather than a cosmetic slip — worth spending a reseed on.
            "GARMENT_PIECES", "GARMENT_DRAPE",
            # The pose is the customer's own photograph. If the arms move or the
            # body turns, the result is no longer a picture of them wearing the
            # garment - it is a picture of someone else's pose in their face.
            # This was previously reported and then ignored, because it was not
            # in this set and so never triggered a reseed.
            "SAME_POSE"}


_VERDICT_TOKEN = re.compile(r"\b(PASS(?:ES|ED)?|FAIL(?:S|ED|URE)?)\b", re.I)


def _is_fail(text: str | None) -> bool:
    """Whether an answer reads as FAIL, by whichever verdict word comes first.

    Never a substring test. Models label the reason line "FAIL REASON:", which
    the parser splits on REASON — leaving VERDICT holding "PASS FAIL" and a
    substring test rejecting a perfectly good image. That marked 17 of 18
    model-6 results as failures while every individual check said PASS.
    """
    hit = _VERDICT_TOKEN.search(text or "")
    return bool(hit) and hit.group(1).upper().startswith("FAIL")


def _failed(parsed: dict, checks: list[str]) -> list[str]:
    """Which checks the model marked FAIL.

    Takes whichever of PASS/FAIL appears *first* in the answer. Two behaviours
    force this:

      - the verdict is often not at the start. Models give the reasoning first
        and conclude at the end: "IMAGE 3 contains two people, and IMAGE 1
        contains one. FAIL as IMAGE 3 adds a person."
      - but scanning for FAIL anywhere is too eager. A passing answer can still
        mention the word — "(Not applicable as the verdict is PASS)" following a
        PASS, or an answer that quotes the failure condition it ruled out — and
        that produced a false rejection on a good image.

    First-token-wins handles both: the model states its conclusion before
    elaborating on it, and any later mention is commentary.
    """
    return [c for c in checks if _is_fail(parsed.get(c))]


def verdict(anatomy: dict, fidelity: dict | None = None) -> tuple[bool, str]:
    """True if the image is acceptable, plus the reason when it is not.

    Trust the individual checks over the model's own VERDICT line: it regularly
    writes PASS on the summary while a check above it says FAIL.

    `fidelity` is optional so a caller that only has the result image still
    gets the anatomical checks.
    """
    bad = _failed(anatomy, ANATOMY_CHECKS)
    src = anatomy
    if fidelity:
        f_bad = _failed(fidelity, FIDELITY_CHECKS)
        if f_bad and not bad:
            src = fidelity
        bad += f_bad
    if bad:
        reason = (src.get("REASON") or src.get(bad[0], "")).strip()
        return False, f"{'+'.join(bad)}: {reason[:140]}"
    for d in (anatomy, fidelity or {}):
        if _is_fail(d.get("VERDICT")):
            return False, (d.get("REASON") or "verdict FAIL")[:140]
    return True, ""


def is_critical(reason: str) -> bool:
    """Whether a failure reason is worth spending a regeneration on."""
    return any(c in reason.split(":")[0] for c in CRITICAL)


# ------------------------------------------------- deterministic pre-checks
#
# "Is this the same face" and "is this the same background" are measurements,
# and a 7B VLM answers them coarsely — it reliably catches a different person
# but not a quietly slimmed jaw. Run these first: they are ~20 ms, they need no
# GPU swap, and anything they reject never needs the VLM at all.

def face_distance(before: str, after: str) -> float | None:
    """Cosine distance between face embeddings, or None if unavailable.

    Needs `insightface` and `onnxruntime`. Returns None rather than raising when
    they are absent or no face is found, so the VLM check still runs.
    Empirically: < 0.35 is the same person, > 0.55 is a different one.
    """
    try:
        import numpy as np
        from insightface.app import FaceAnalysis
    except ImportError:
        return None

    global _FACE_APP
    try:
        _FACE_APP
    except NameError:
        _FACE_APP = FaceAnalysis(name="buffalo_l",
                                 providers=["CPUExecutionProvider"])
        _FACE_APP.prepare(ctx_id=-1, det_size=(640, 640))

    import cv2
    embs = []
    for path in (before, after):
        img = cv2.imread(path)
        if img is None:
            return None
        faces = _FACE_APP.get(img)
        if not faces:
            return None
        # Largest face: the subject, not a bystander.
        f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
        e = f.normed_embedding
        embs.append(e / np.linalg.norm(e))
    return float(1.0 - np.dot(embs[0], embs[1]))


def background_distance(before: str, after: str, border: float = 0.12) -> float | None:
    """Mean absolute difference across the image border, 0..1, or None.

    The subject occupies the centre and the background occupies the edges, so a
    border strip is a cheap proxy for "did the scene change" without needing
    segmentation. Only valid because the framing is meant to be preserved — a
    result that reframes will score badly here for the right reason.

    Deliberately PIL-only. This is the single most valuable check in the
    guardrail, and an optional numpy import is enough to silently switch it off
    — which is how a QA gate ends up passing every image while looking healthy.

    Calibrated on the 62-image no-prompt sweep (models f1-f3, 6 confirmed scene
    replacements): scenes that were kept scored 0.055-0.190, median 0.110;
    replaced scenes scored 0.222-0.357. A cutoff of 0.20 caught 6 of 6 with no
    false alarms. The margin is only 0.033 and rests on six positives, so this
    is a pre-filter that feeds the VLM rather than a sole arbiter — it found two
    replacements manual review had missed, but do not treat a pass as proof.
    """
    from PIL import Image, ImageChops, ImageStat

    a = Image.open(before).convert("RGB")
    b = Image.open(after).convert("RGB")
    if a.size != b.size:
        b = b.resize(a.size, Image.LANCZOS)

    w, h = a.size
    bw, bh = max(1, int(w * border)), max(1, int(h * border))
    strips = [(0, 0, w, bh), (0, h - bh, w, h),      # top, bottom
              (0, 0, bw, h), (w - bw, 0, w, h)]      # left, right

    total, weight = 0.0, 0
    for box in strips:
        da = a.crop(box)
        db = b.crop(box)
        diff = ImageChops.difference(da, db)
        px = da.size[0] * da.size[1]
        total += sum(ImageStat.Stat(diff).mean) / 3.0 * px
        weight += px
    return (total / weight) / 255.0


def face_count(path: str) -> int | None:
    """Number of detectable faces, or None if insightface is unavailable.

    Counting is a measurement, not a judgement, and the VLM is unreliable at it
    — nemotron missed the second person in f1__fg13 even with the original
    alongside for comparison. A detector does not miss a face that size.
    """
    faces = _detect(path)
    return None if faces is None else len(faces)


def _detect(path: str):
    """Run the face detector, or return None when it is not installed."""
    try:
        import cv2
        from insightface.app import FaceAnalysis
    except ImportError:
        return None

    global _FACE_APP
    try:
        _FACE_APP
    except NameError:
        _FACE_APP = FaceAnalysis(name="buffalo_l",
                                 providers=["CPUExecutionProvider"])
        _FACE_APP.prepare(ctx_id=-1, det_size=(640, 640))

    img = cv2.imread(path)
    return None if img is None else _FACE_APP.get(img)


def numeric_gate(person: str, result: str,
                 face_max: float = 0.45,
                 background_max: float = 0.20) -> tuple[bool, str, dict]:
    """Cheap deterministic checks. Returns (ok, reason, ran).

    `ran` records what actually executed and what it measured. A check that
    could not run appears as None rather than vanishing — an optional dependency
    going missing must be visible in the report, not turn the gate into a
    rubber stamp.
    """
    ran: dict[str, float | int | None] = {}
    ran["background_distance"] = bd = background_distance(person, result)
    ran["faces_before"] = fb = face_count(person)
    ran["faces_after"] = fa = face_count(result)
    ran["face_distance"] = fd = face_distance(person, result)

    if bd is not None and bd > background_max:
        return False, f"SAME_BACKGROUND: border diff {bd:.2f} > {background_max}", ran
    # A face the original did not have is a person the generator invented —
    # usually the garment photo's mannequin rendered as a human.
    if fb is not None and fa is not None and fa > fb:
        return False, f"ONE_PERSON: {fb} face(s) before, {fa} after", ran
    if fd is not None and fd > face_max:
        return False, f"SAME_PERSON: face distance {fd:.2f} > {face_max}", ran
    return True, "", ran
