"""VLM prompts + builder, v6 — describe the person, don't just instruct about them.

v5 fixed the framing (edit, not generate) and that stopped the raised hand
disappearing and the body being slimmed. But its preserve list was generic
boilerplate: "the same face", "the same build". The background fix worked
because it described the wanted result positively rather than forbidding the
unwanted one, and the same logic applies to the person.

v6 has the vision model read the subject as carefully as it reads the garment —
expression, build, stance, what the hands are doing, what is being held — and
the builder quotes those specifics back into the preserve list. "Keep her
expression" becomes "keep her closed-mouth smile with her head tilted slightly
left", which is a thing the model can actually aim at.
"""
import re

# --------------------------------------------------------------- VLM asks

MODEL_ASK = """Look at this photograph of a person. I am going to regenerate it with
different clothes, so I need everything about the person and the place recorded
precisely. Describe only what you can see. If it is a phone screenshot, ignore the
interface chrome. Reply with these labelled lines only:

SETTING: (one paragraph, 40-70 words: where this is, the surfaces, furniture, plants,
walls and objects, the time of day, and the direction, colour and hardness of the light)
FRAMING: (how much of the person is in shot, where they sit in the frame, camera height
and angle, how close the camera is)
POSE: (stance and which foot carries the weight, the direction the body faces, head
angle and where the eyes look — describe the body only, NOT the hands)
LEFT_HAND: (one short sentence: where the person's left hand is and what it is doing —
raised to the face, hanging at the side, on a hip, holding something and what. If it is
hidden or out of frame, say "not visible")
RIGHT_HAND: (same, for the right hand)
BUILD: (apparent age range, body build and proportions, height impression — factual
and neutral, this is recorded so the person is not altered)
FACE: (face shape, skin tone, and above all the exact expression — smiling with teeth,
closed-mouth smile, neutral, and any head tilt)
HAIR: (length, colour, texture, how it is worn and parted)
ACCESSORIES: (everything worn or held that is not clothing — glasses, watch, bangles,
rings, bindi, headband, bag, phone — or "none")"""

GARMENT_ASK = """Catalogue this garment. Describe only what you can see; if it is a phone
screenshot ignore the interface chrome. Reply with these labelled lines only:

TYPE:
COLOUR:
FABRIC:
METAL: (gold, silver, mixed or none - look carefully, this matters)
EMBELLISHMENT:
FIT: (how this garment is meant to sit on a body: fitted or loose, where it closes,
sleeve length, hemline, and how the fabric falls)
DESCRIPTION: (one paragraph, 55-85 words, starting "wearing", precise enough to
reproduce THIS garment and not a similar one)"""

MODEL_FIELDS   = ["SETTING","FRAMING","POSE","LEFT_HAND","RIGHT_HAND","BUILD","FACE","HAIR","ACCESSORIES"]
GARMENT_FIELDS = ["TYPE","COLOUR","FABRIC","METAL","EMBELLISHMENT","FIT","DESCRIPTION"]


# --------------------------------------------------------------- builder

def build(subject: str, m: dict, g: dict) -> str:
    """m = model catalogue entry, g = garment catalogue entry."""
    her = "her" if subject == "woman" else "his"
    she = "She" if subject == "woman" else "He"
    s   = she.lower()

    def f(d, k, fallback=""):
        return (d.get(k) or fallback).strip()

    # The vision model often buries "wearing ..." mid-sentence rather than
    # opening with it, and blindly prepending produced "She is wearing  The
    # garment is cut to sit like this: ... wearing this dark grey lehenga" -- a
    # dangling verb, a doubled "wearing", and the fit clause sitting in front of
    # the garment it describes. Normalise the description first, then append fit.
    desc = f(g, "DESCRIPTION")
    hit = re.search(r"\bwearing\b", desc, re.I)   # not `m` -- that is the model dict
    if hit:
        desc = desc[hit.end():].lstrip(" ,:")      # keep what follows "wearing"
    desc = desc[0].lower() + desc[1:] if desc else desc

    # Fall back to the structured fields when the paragraph is unusable.
    if len(desc) < 15:
        bits = [f(g, k) for k in ("COLOUR", "FABRIC", "TYPE") if f(g, k)]
        desc = " ".join(bits) or "the garment shown in image 2"

    fit = f(g, "FIT")
    fit_line = f" It is cut to sit like this: {fit}" if fit else ""

    parts = [
        "This is a photograph of a real person. Edit it so the person is wearing "
        "different clothes. Change ONLY the clothing. Every other pixel of this "
        "photograph — the person, the place, the light, the framing — stays exactly "
        "as it is.",
        "",
        f"THE NEW CLOTHING: {she} is {desc}{fit_line}",
        "",
        "KEEP EXACTLY AS IN THE PHOTOGRAPH — these describe what is already there, "
        "and every one must survive unchanged:",
        "",
        f"FACE AND EXPRESSION: {f(m,'FACE')} Keep this face feature for feature and "
        f"keep this exact expression. Do not slim, sharpen, symmetrise, youthen or "
        f"beautify it. The result must read as a photograph of the same person on the "
        f"same day, not of someone who resembles {her}.",
        "",
        f"BODY: {f(m,'BUILD')} Keep this build, these proportions and this apparent "
        f"age exactly. Do not slim, lengthen, narrow or reshape the body. The clothing "
        f"fits the body already in the photograph — the body does not change to suit "
        f"the clothing.",
        "",
        f"POSE: {f(m,'POSE')} Keep this stance, this head angle and this gaze.",
        "",
        f"THE HANDS — this person has TWO hands in total and this is the complete list "
        f"of them:\n"
        f"  LEFT HAND: {f(m,'LEFT_HAND','hanging naturally at the side')}\n"
        f"  RIGHT HAND: {f(m,'RIGHT_HAND','hanging naturally at the side')}\n"
        f"Draw exactly those two hands, each doing exactly what is described, each with "
        f"five individually separated fingers. There is no third hand anywhere in the "
        f"image — not at the waist, not at the hip, not holding anything else, not "
        f"resting on the garment. Two arms, two hands, nothing more. If a described "
        f"hand is holding something, that object appears once, in that hand only.",
        "",
        f"HAIR: {f(m,'HAIR')} Same style, same parting, same stray strands.",
        "",
        f"ACCESSORIES: {f(m,'ACCESSORIES','none')} Keep every one of these and add "
        f"nothing that is not listed.",
        "",
        f"FRAMING: {f(m,'FRAMING')} Keep this framing — the same position in frame, "
        f"the same distance, the same camera height and angle, the same crop. Do not "
        f"move, re-centre, zoom or re-frame the subject.",
        "",
        f"THE SCENE: {f(m,'SETTING')} This setting stays identical: the same objects "
        f"in the same places, the same light direction, colour and intensity, the same "
        f"shadows falling the same way.",
        "",
        "WEARING IT PROPERLY: the garment must sit on the body like real cloth on a "
        "real person — correct size and fit for this body, seams and hems where they "
        "belong, fabric falling with its own weight, natural folds where the body "
        "bends, and shadow between the cloth and the skin beneath it. Nothing floats, "
        "nothing is painted flat onto the body.",
        "",
        "THE GARMENT PHOTOGRAPH SUPPLIES THE GARMENT ONLY: take its colour, fabric, "
        "cut and embellishment, and nothing else. Do not bring across its mannequin, "
        "hanger, model, showroom, studio wall, flooring, props, watermark or any "
        "interface graphics.",
    ]
    return "\n".join(parts)
