"""Prompt library.

The upstream workflow shipped with a Chinese instruction —
"将图1的女性模特服装换成图2" ("replace the clothing of the female model in
image 1 with image 2") — which suggested the fal LoRA was trained on Chinese
instruction pairs.

TESTED 2026-08-13 on an RTX 4090, seed 42, 8 steps: that terse Chinese prompt
collapses into noise. The detailed English prompts below produce clean results
on the same seed and inputs. English is therefore the default; the Chinese
variants are kept only so you can reproduce the comparison.
"""
from __future__ import annotations

# The generic default, used when a caller sends no prompt at all.
#
# The previous one-sentence version broke all three rules this pipeline learned
# from measurement, and scenes were being replaced by the garment photo's studio
# as a result:
#
#   1. It relied on negation ("do not invent detail"). This text encoder does
#      not act on negation - prohibitions had zero measurable effect across
#      every image tested.
#   2. It assigned no roles. Both inputs are just reference latents; without a
#      sentence saying which one is the subject, a promptless run renders both
#      figures and may take either photo's background.
#   3. It omitted "the second image contributes the garment alone", which is
#      the clause that actually keeps the garment photo's room out of the
#      result.
#
# It cannot name the caller's particular background the way the catalogue
# prompts do, so instead it states preservation positively, item by item, and
# enumerates the hands as a closed set - the construction that fixed the
# three-hands defect.
DEFAULT_PROMPT = (
    "The first image is a photograph of one person. The second image shows a "
    "garment. Edit the first photograph so that the same person is wearing "
    "that garment.\n"
    "There is exactly ONE person in the result: the person from the first "
    "photograph, standing where they already stand.\n\n"
    "KEEP EXACTLY AS IN THE FIRST PHOTOGRAPH:\n"
    "FACE: the same face, the same expression, the same age, the same skin "
    "tone. It reads as the same person photographed on the same day.\n"
    "HAIR: the same hair, the same length, falling the same way.\n"
    "BODY: the same build, height and proportions. The garment is fitted to "
    "the body already in the photograph.\n"
    "POSE: the same stance, the same arm positions, the same head angle and "
    "gaze, the same camera distance and crop. Anything being held stays held, "
    "in the same hand.\n"
    "HANDS: this person has TWO hands in total and this is the complete list "
    "of them - one left hand and one right hand, each with five separated "
    "fingers. Draw exactly those two.\n"
    "JEWELLERY AND ACCESSORIES: everything already worn on the head, ears, "
    "neck, wrists and fingers stays exactly as it is.\n"
    "BACKGROUND: the same place. Every object behind and beside the person "
    "stays where it is, at the same size, in the same light, casting the same "
    "shadows. The floor, the walls, the sky and the horizon are the ones "
    "already in the first photograph.\n\n"
    "THE GARMENT: reproduce its colour, its construction and its embroidery "
    "exactly as they appear in the second image, including every separate "
    "piece it is made of.\n\n"
    "The second image contributes the garment alone."
)
# Kept for A/B reproduction only -- these fail at 8 steps, see module docstring.
LEGACY_PROMPT_ZH = "将图1的女性模特服装换成图2"
LEGACY_PROMPT_ZH_MALE = "将图1的男性模特服装换成图2"

_PRESET = {
    "default_en": (
        "Generic English (default)",
        DEFAULT_PROMPT,
    ),
    "legacy_female_zh": (
        "Chinese trigger, female -- FAILS at 8 steps, kept for A/B only",
        LEGACY_PROMPT_ZH,
    ),
    "legacy_male_zh": (
        "Chinese trigger, male -- FAILS at 8 steps, kept for A/B only",
        LEGACY_PROMPT_ZH_MALE,
    ),
    # --- Female set -------------------------------------------------
    "f1_orange": (
        "Orange bustier + draped skirt",
        "Using image 1's woman — same face, hair, bandana, jewellery, pose, "
        "background and lighting — dress her in the exact bright orange set from "
        "image 2: structured sweetheart bustier with gold and copper zardozi "
        "floral embroidery and deep red accents, scalloped embroidered hem band, "
        "matching orange draped wrap skirt pleated at the front waist falling to "
        "the floor, and two long sheer orange georgette cape panels from the "
        "shoulders edged with narrow gold sequined trim. Bare midriff. Do not "
        "change her face or invent motifs absent from image 2.",
    ),
    "f2_ivory": (
        "Ivory halter + palazzo",
        "Using image 1's woman — same face, hair, bandana, jewellery, pose, "
        "background and lighting — dress her in the exact ivory set from image 2: "
        "sleeveless halter top with high banded neck and pointed handkerchief hem "
        "ending in a V, sheer net bodice with multicoloured resham floral "
        "embroidery (dusty pink blooms, sage leaves, gold vines, red berries), "
        "pearl beaded trim along the hem point, and very wide flowing ivory "
        "georgette palazzo trousers. Bare shoulders. Do not change her face, do "
        "not add sleeves, do not render the embroidery in gold only.",
    ),
    "f3_coral": (
        "Coral beaded peplum + ombre palazzo",
        "Using image 1's woman — same face, hair, bandana, jewellery, pose, "
        "background and lighting — dress her in the exact coral set from image 2: "
        "sleeveless deep-V peplum top with dense gold, copper and pearl art-deco "
        "beadwork (fan motifs at the neckline resolving into vertical beaded "
        "lines), scalloped hem with hanging bead drops, fitted waist with slight "
        "hip flare, and very wide sheer ombre coral-to-pink palazzo trousers. Do "
        "not change her face, do not add sleeves, do not flatten the ombre.",
    ),
    "f4_saree": (
        "Rose-gold saree + zardozi blouse",
        "Using image 1's woman — same face, hair, bandana, jewellery, pose, "
        "background and lighting — dress her in the exact rose-gold saree from "
        "image 2: dusty rose-gold satin-organza, structured bustier blouse with "
        "sweetheart neckline and dense GOLD zardozi floral embroidery, scalloped "
        "sleeve hems, scattered gold butis on the saree body, wide gold "
        "embroidered border. Classic saree drape — pleats at the front waist, "
        "pallu over the left shoulder with the border visible, midriff exposed, "
        "floor length. Do not change her face, do not render the embroidery in "
        "silver, do not invent motifs absent from image 2.",
    ),
    "f5_pink_suit": (
        "Pink kameez + dupatta",
        "Using image 1's woman — same face, hair, bandana, jewellery, pose, "
        "background and lighting — dress her in the exact pink suit set from "
        "image 2: onion-pink satin kameez, round neck with front placket, two "
        "vertical bands of multicolour stones (red, green, mint, coral, pearl) in "
        "SILVER zari framing, large silver paisley motif below the yoke, scalloped "
        "embroidered cuffs, matching pink trousers, and a sheer pink chiffon "
        "dupatta with scalloped silver-embroidered border over one shoulder. Do "
        "not change her face, do not render the embroidery in gold.",
    ),
    # --- Male set ---------------------------------------------------
    "m1_studded_tee": (
        "Black studded sleeveless tee",
        "Using image 1's man — same face, sunglasses, necklace, watch, earring, "
        "tattoo, shorts, sandals, pose, background and lighting — replace only his "
        "t-shirt with the exact black sleeveless tee from image 2: boxy oversized "
        "cut, ribbed high mock-crew neck, silver ball studs around the neckline and "
        "both armholes, front graphic reading 'MATIERE' in silver rhinestone block "
        "capitals with 'Desires' in pink script overlaid, above a photographic open "
        "mouth with pink rhinestone lips, crystal teeth and a glossy tongue. Do not "
        "change his face or his lower half.",
    ),
    "m2_bandhgala": (
        "Black gold-embroidered bandhgala",
        "Using image 1's man — same face, sunglasses, earring, pose, background and "
        "lighting — dress him in the exact black bandhgala from image 2: "
        "knee-length black wool-crepe coat, Nehru stand collar fully covered in "
        "gold embroidery, concealed placket, dense GOLD zardozi and bullion floral "
        "embroidery heaviest at the shoulders and chest cascading into vine sprays "
        "with hanging chain details and thinning toward the hem, antique-silver "
        "accents, two embroidered flap pockets at the hip. Plain black trousers and "
        "black leather shoes. Do not change his face, do not spread the embroidery "
        "evenly — the shoulder-to-hem gradient is the design.",
    ),
    "m3_polo": (
        "Navy POLO cable-knit + cream trousers",
        "Using image 1's man — same face, sunglasses, earring, watch, pose, "
        "background and lighting — dress him in the exact outfit from image 2: a "
        "blue-and-white striped shirt knotted at the throat with tails hanging down "
        "the chest, worn UNDER a navy cable-knit crewneck sweater reading 'POLO' in "
        "large cream serif capitals, 'Ralph Lauren' in script below, 'EST. 1967' "
        "below that, with a white polo-pony logo on the upper right chest and a "
        "small crossed-mallet crest on the upper left. Cream pleated trousers, white "
        "leather sneakers. Keep shirt and sweater as separate visible layers. Do not "
        "change his face or distort the lettering.",
    ),
    "m4_overcoat": (
        "Black overcoat + crest sweater",
        "Using image 1's man — same face, sunglasses, earring, pose, background and "
        "lighting — dress him in the exact outfit from image 2: white dress shirt "
        "collar at the neck, black knit crewneck sweater with a large gold bullion "
        "heraldic crest (gold crown, two rearing gold horses flanking a red shield "
        "with a red 'RL' monogram, 'IXVII' banner, gold laurel wreath below), worn "
        "under a long black wool overcoat with notched lapels and black velvet "
        "collar, hanging OPEN and falling below the knee. Black pleated trousers, "
        "black leather dress shoes. Keep all three layers visibly separate. Do not "
        "change his face or simplify the crest.",
    ),
    "m5_field_jacket": (
        "Black belted field jacket + shirt and tie",
        "Using image 1's man — same face, sunglasses, earring, pose, background and "
        "lighting — dress him in the exact outfit from image 2: white point-collar "
        "dress shirt and dark charcoal knit tie visible at the neck, under a black "
        "hip-length cotton field jacket with tall stand collar worn up, silver-snap "
        "epaulettes, two flap chest pockets, a self-fabric belt tied at the waist "
        "with silver eyelets, a zip detail on the lower left front, and buckled tab "
        "cuffs. Black leather belt with silver buckle, white slim cropped trousers, "
        "black suede loafers with silver buckles. Keep the shirt and tie visible. "
        "Do not change his face.",
    ),
}


# --- Scene anchors -----------------------------------------------------------
# The 2026-08-13 run showed the LoRA adopts whatever scene the GARMENT photo
# supplies -- the showroom from the mannequin shots overrode her bedroom. A
# negative instruction ("do not use image 2's background") did nothing: these
# text encoders weight early tokens heavily and handle negation poorly.
#
# The fix is to lead with a POSITIVE, concrete description of the wanted scene
# so the model has something to generate rather than something to avoid.
_SCENE_FEMALE = (
    "A photograph taken inside a hotel bedroom. The room has a large bed made up "
    "with white linen, a dark wood four-poster bed frame, a framed picture on the "
    "wall, warm beige walls, a doorway on the left, and warm indoor lighting. "
    "The woman from image 1 is standing in the centre of this bedroom, in front "
    "of the bed, facing the camera. She has exactly two arms and two hands. Keep "
    "this hotel bedroom exactly as the setting, with the same warm indoor light "
    "and the same camera framing. "
)
_SCENE_MALE = (
    "A photograph taken outdoors on a rural asphalt road with dense green "
    "tropical vegetation on both sides, under flat overcast daylight. The man "
    "from image 1 is standing on this road, facing the camera, arms relaxed at "
    "his sides. Keep this outdoor road setting exactly, with the same daylight "
    "and the same camera framing. "
)

# Applied after the garment description. Each clause fixes a defect observed in
# the 2026-08-13 run: merged/extra fingers, slimmed faces, invented jewellery.
_CONSTRAINTS = (
    "\n\nCRITICAL CONSTRAINTS.\n"
    "SETTING: the finished photograph must stay in the setting described above. "
    "Image 2 supplies the garment only — take nothing else from it: no mannequin, "
    "no showroom, no studio wall, no potted plants, no wooden flooring, no arched "
    "window, no screenshot borders or interface graphics.\n"
    "ANATOMY: the person has exactly two arms and exactly two hands in the whole "
    "image — one left, one right. Do not draw a second pair of arms or hands, do "
    "not place hands both at the waist and at the sides, and do not add any "
    "detached or floating limb. Both arms hang down and outward from the "
    "shoulders exactly as in image 1, and each hand has exactly five fingers, "
    "individually separated and clearly defined, with no merged, extra, missing "
    "or deformed digits.\n"
    "FACE: reproduce the face from image 1 exactly — same bone structure, jaw "
    "width, nose and eye shape. Do not slim, sharpen, symmetrise or beautify it.\n"
    "ACCESSORIES: do not add earrings, necklaces, rings, watches or any jewellery "
    "that is not already visible in image 1."
)

_GARMENT_PREFIXES = ("f", "m")


def _is_garment(preset: str) -> bool:
    return preset[:1] in _GARMENT_PREFIXES and preset[1:2].isdigit()


def _scene(preset: str) -> str:
    return _SCENE_FEMALE if preset.startswith("f") else _SCENE_MALE


def _compose(preset: str, text: str) -> str:
    """Scene first, garment second, constraints last.

    Order matters: the scene anchor has to land in the early tokens or the
    garment photo's own setting wins.
    """
    if not _is_garment(preset):
        return text
    return _scene(preset) + "\n\n" + text + _CONSTRAINTS


def all_presets() -> list[dict[str, str]]:
    return [
        {"id": k, "label": label, "prompt": _compose(k, text)}
        for k, (label, text) in _PRESET.items()
    ]


def resolve(prompt: str | None, preset: str | None) -> str:
    """Explicit prompt wins; then preset (scene + garment + constraints)."""
    if prompt:
        return prompt
    if preset:
        if preset not in _PRESET:
            raise KeyError(preset)
        return _compose(preset, _PRESET[preset][1])
    return DEFAULT_PROMPT
