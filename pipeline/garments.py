"""Hand-written garment readings for the 23 female garments, model 6.

Written by eye from the source photographs rather than by a vision model. Two
things come out of that which an automated reading did not give us:

1.  Per-garment crop boxes. The old preprocessing used one fixed centre crop
    (24%-76% of the width) for every garment. On fg01 the dupatta hangs at
    78-97% of the width and on fg06 it spreads across the left 45% — the crop
    was cutting the dupatta out of the input. A missing dupatta in the output
    was not the generator dropping it; it never saw it. Each box below is
    chosen to keep every piece of the garment and to drop as much of the room,
    the props and the app UI as possible.

2.  Piece lists that are actually right. A dupatta hung on a separate hook
    reads as part of the wall to a captioner; fg12 has two dupattas, which no
    single-sentence description was ever going to carry.

Prompt construction follows the two rules established earlier: state
constraints as closed sets rather than prohibitions, and never rely on
negation — this text encoder does not act on it.
"""
from __future__ import annotations

# --- the person -----------------------------------------------------------
# Model 6: a woman photographed at night in a garden, flash-lit, holding a
# phone in both hands. Every detail here is something the generator has been
# observed to alter when it is not named.
ROLE = (
    "The first image is a photograph of one woman. The second image shows a "
    "garment on its own. Edit the first photograph so that the same woman is "
    "wearing that garment.\n"
    "There is exactly ONE person in the result: the woman from the first "
    "photograph, standing where she already stands."
)

PRESERVE = (
    "KEEP EXACTLY AS IN THE FIRST PHOTOGRAPH:\n"
    "FACE: the same woman — the same face, the same warm smile with dark "
    "berry lipstick, the same small dark bindi, the same red sindoor in her "
    "centre hair parting, the same age, the same skin tone. She reads as the "
    "same person photographed on the same evening.\n"
    "HAIR: long straight black hair, parted in the centre, falling forward "
    "over both shoulders and down past her waist.\n"
    "BODY: the same build, the same height, the same proportions. The garment "
    "is fitted to the body already in the photograph.\n"
    "POSE: standing, facing the camera, head very slightly tilted, looking "
    "just past the lens. Her forearms come together in front of her waist and "
    "she is holding a brown smartphone in both hands, still held in exactly "
    "that way.\n"
    "HANDS: this woman has TWO hands in total and this is the complete list "
    "of them — one left hand and one right hand, each with five separated "
    "fingers, both closed around the phone at her waist. Her right hand wears "
    "henna.\n"
    "JEWELLERY: stacked white pearl bangles and gold bangles on both wrists, "
    "a fine chain necklace, small drop earrings, a ring on each hand.\n"
    "BACKGROUND: the same night garden, unchanged — tall areca palm fronds "
    "filling the frame behind her, a white car parked in the dark on the "
    "left, a pale mint-green wooden side table at the left edge holding a "
    "brass vase of white and pink chrysanthemums, a raised red-brick planter "
    "on the right, an orange-and-white patterned tile floor, a blue cable "
    "running across the tiles, and black night sky above the leaves. The same "
    "camera flash on her from the front and the same darkness behind her."
)

# For anyone who is not model 6. The block above names her actual surroundings,
# which is why her 23 results kept their background - but attaching it to a
# different photograph describes a scene that is not there, and a prompt that
# asserts palm fronds and a white car into someone's living room is worse than
# one that says nothing specific. This states what to preserve without
# inventing what it is.
PRESERVE_GENERIC = (
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
    "already in the first photograph."
)

# Source attribution, stated as an explicit split rather than a prohibition.
# The encoder does not act on negation, so "do not copy the second image's
# room" achieves nothing; naming what each image is allowed to contribute is
# the same instruction in a form the model does use. This helps and is worth
# having, but it is not sufficient on its own - the room still has to be
# cropped out of the garment photograph, which is measured, not assumed.
CLOSING = (
    "WHERE EACH THING COMES FROM — this is the complete division between the "
    "two images:\n"
    "From the FIRST image: the person, her face, her hair, her body, her pose, "
    "her hands, her jewellery, the room or place she is standing in, "
    "everything behind her and beside her, the floor she stands on, the light "
    "falling on her, and the shadows she casts. The result is that photograph, "
    "with her clothing changed.\n"
    "From the SECOND image: the garment, and only the garment - its colour, "
    "its fabric, its cut, its embroidery, its every piece. The second image "
    "contributes the garment alone.\n"
    "The background of the result is the background of the FIRST image. "
    "Whatever room, wall, floor, furniture or backdrop appears in the second "
    "image belongs to the garment's photograph and stays there."
)


def _waist_clause(pieces: list[str], summary: str = "") -> str:
    """A default waist sentence for any outfit with a separate top and skirt.

    Returns "" for sarees, gowns and suits: a saree's drape is continuous by
    construction and a gown has no join, so asserting a seam there would be
    describing a garment that does not exist. Gowns are excluded by summary
    rather than by piece list, because a gown is often described as a bodice
    and a skirt - fg14 and fg15 are exactly that, and claiming a waist seam on
    them would introduce the defect this clause exists to remove.
    """
    if "gown" in summary.lower():
        return ""
    joined = " ".join(pieces).lower()
    top = any(w in joined for w in ("blouse", "choli", "bodice"))
    skirt = any(w in joined for w in ("skirt", "lehenga"))
    if not (top and skirt):
        return ""
    return ("The blouse ends just below the bust and the flared skirt starts "
            "at her waist below it.")


def build(g: dict) -> str:
    """ROLE + garment + PRESERVE, in that order."""
    pieces = g["pieces"]
    n = len(pieces)
    listed = ", ".join(f"({i}) {p}" for i, p in enumerate(pieces, 1))

    body = [f"THE GARMENT — {g['summary']}."]
    # The counted closed list. Naming a component inside a sentence makes it
    # optional to the model; counting it does not. This is the construction
    # that fixed the three-hands defect.
    body.append(
        f"It is made of {n} piece{'s' if n > 1 else ''}: {listed}. "
        f"That is the complete list. All {n} appear in the result, worn "
        f"together on the woman, each one clearly distinct from the others.")
    # Each piece described on its own, before the garment is described as a
    # whole. A garment summarised in one breath loses its construction - a
    # lehenga becomes "a long embroidered dress" and renders as a plain skirt.
    # Describing the blouse separately forces it to exist.
    for i, d in enumerate(g.get("piece_details") or [], 1):
        body.append(f"PIECE {i} — {d}")
    body.append(g["detail"])
    # Counting the pieces keeps them all present, but it does not stop the
    # generator fusing a blouse and a skirt into one continuous anarkali — it
    # did that on 5 of 12 lehengas at 1.0 MP and again at 0.75 MP. Naming the
    # seam gives it a concrete thing to draw at the waist instead of a smooth
    # panel, and that fixed fg04, fg16 and fg17 when it was applied by hand.
    #
    # Derived from the piece list rather than set per garment: any outfit whose
    # pieces include both a blouse and a skirt has a waist join by definition,
    # and hand-maintaining the flag is how most of them came to be missing it.
    waist = g.get("waist") or _waist_clause(pieces, g.get("summary", ""))
    if waist:
        body.append(
            "THE WAIST — the blouse and the skirt are two separate garments, "
            f"not one dress. {waist} The join between them is visible at "
            "her waist as a horizontal edge, with the skirt's waistband below "
            "it and the hem of the blouse above it.")
    if g.get("drape"):
        body.append(f"THE DRAPE — {g['drape']}")
    body.append(f"COLOURS — {g['colours']}. Reproduce these colours exactly.")

    # One-off pairs (pipeline/pair_specs.py) carry their own person, so they
    # override these. The catalogue's block describes model 6 in her night
    # garden, which would be actively wrong for anyone else.
    return "\n\n".join([g.get("role") or ROLE, " ".join(body),
                        g.get("preserve") or PRESERVE, CLOSING])


# --- the garments ---------------------------------------------------------
# crop is (left, top, right, bottom) as fractions of the source image.
GARMENTS: dict[str, dict] = {

    "fg01": {
        "crop": (0.02, 0.16, 1.00, 1.00),
        "summary": "a dusty-rose bridal lehenga",
        "pieces": ["a long-sleeved fitted blouse",
                   "a floor-length flared lehenga skirt",
                   "a sheer dupatta"],
        "detail": "The blouse and the skirt are dusty rose net carpeted in "
                  "dense silver-white and pearl thread embroidery — fine "
                  "floral vines, scattered sequins, a deep scalloped "
                  "embroidered border around the hem. The skirt flares wide "
                  "from a fitted waist to the floor.",
        "waist": "The blouse is a fitted choli that ends just below the bust, and the flared skirt starts at her natural waist below it.",
        "drape": "the sheer pale-rose net dupatta, scattered with small "
                 "silver flowers, is draped over her left shoulder and its "
                 "long end falls down her back to below the knee.",
        "colours": "dusty rose pink, with silver and pearl-white embroidery",
    },

    "fg02": {
        "crop": (0.00, 0.10, 0.94, 1.00),
        "summary": "a fuchsia bridal lehenga in heavy traditional hand work",
        "pieces": ["an elbow-sleeved blouse",
                   "a flared floor-length lehenga skirt",
                   "a dupatta"],
        "detail": "Every surface is worked in kundan stones, mirror discs, "
                  "pearl beads and multicoloured thread — pastel green and "
                  "orange florals and paisleys over a fuchsia silk ground, "
                  "with a very heavy encrusted border and a scalloped "
                  "beaded hem.",
        "drape": "the fuchsia dupatta has the same scalloped embroidered "
                 "edge and is draped over her left shoulder, falling to "
                 "the knee.",
        "colours": "deep fuchsia pink, with gold, mirror-silver, pastel "
                   "green and orange embroidery",
    },

    "fg03": {
        "crop": (0.00, 0.03, 1.00, 1.00),
        "summary": "a charcoal-grey net lehenga with antique-gold work",
        "pieces": ["a short cap-sleeved blouse",
                   "a flared floor-length net skirt",
                   "a matching net dupatta"],
        "detail": "The ground is dark charcoal grey net. Antique-gold "
                  "embroidery climbs it in thin branching sprays of leaves "
                  "and tiny flowers, tipped with bronze pearl beads. The "
                  "blouse is more solidly worked in gold across the bodice.",
        "drape": "the charcoal-grey net dupatta, edged in a gold border and "
                 "scattered with small gold flowers, is draped over her left "
                 "shoulder with its end falling behind her.",
        "colours": "dark charcoal grey, with antique gold and bronze pearls",
    },

    "fg04": {
        "crop": (0.00, 0.02, 0.84, 1.00),
        "summary": "an ivory lehenga worn with a contrasting olive-green "
                   "tissue dupatta",
        "pieces": ["a sleeveless ivory blouse",
                   "a flared floor-length ivory skirt",
                   "an olive-green tissue dupatta"],
        "detail": "The blouse and skirt are ivory silk embroidered in "
                  "silver-green thread with small ferns, sprigs and "
                  "flowers, growing denser towards a heavily worked hem.",
        "waist": "The sleeveless ivory blouse ends just below the bust, and the flared ivory skirt starts at her natural waist below it.",
        "drape": "the dupatta is a different colour from the rest of the "
                 "outfit — shimmering olive-green tissue with a scalloped "
                 "silver-embroidered border and a green tassel at the "
                 "corner. It is draped over her left shoulder and falls "
                 "open down her left side.",
        "colours": "ivory and cream with silver-green embroidery, and an "
                   "olive-green metallic dupatta",
    },

    "fg05": {
        "crop": (0.00, 0.00, 1.00, 1.00),
        "summary": "a pale silver-grey net lehenga with gold work",
        "pieces": ["a cap-sleeved blouse",
                   "a flared floor-length net skirt",
                   "a matching net dupatta"],
        "detail": "Pale blue-grey net, embroidered all over in warm gold "
                  "thread in slender branching sprays with pearl and gold "
                  "beads at the tips, thickening into a gold border at the "
                  "hem. The blouse is densely gold-worked.",
        "drape": "the pale grey net dupatta has a gold-edged border and is "
                 "draped over her left shoulder, its end falling behind her.",
        "colours": "pale silver-grey, with warm gold and pearl",
    },

    "fg06": {
        "crop": (0.00, 0.11, 1.00, 1.00),
        "summary": "a champagne-ivory bridal lehenga worn with a deep teal "
                   "dupatta",
        "pieces": ["a half-sleeved blouse",
                   "a very wide flared floor-length skirt",
                   "a deep teal-green dupatta"],
        "detail": "The blouse and skirt are pale champagne-ivory under dense "
                  "gold embroidery — scattered flowers and buds over the "
                  "whole skirt, gathering into a broad gold lattice band "
                  "just above the hem.",
        "waist": "The half-sleeved blouse ends just below the bust, and the wide flared skirt starts from a gold waistband at her waist.",
        "drape": "the dupatta is deep teal green, a strong contrast to the "
                 "ivory of the rest, worked with ivory and gold flowers and "
                 "a wide embroidered border. It is draped over her left "
                 "shoulder and falls long and open down her left side.",
        "colours": "champagne ivory with gold embroidery, and a deep teal "
                   "green dupatta",
    },

    "fg07": {
        "crop": (0.08, 0.09, 1.00, 1.00),
        "summary": "a lilac and ivory panelled lehenga",
        "pieces": ["a short puff-sleeved lilac blouse",
                   "a flared floor-length panelled skirt",
                   "a lilac net dupatta"],
        "detail": "The skirt is built from alternating lilac and ivory "
                  "panels, every panel embroidered in ivory pearl and "
                  "cream thread with small flowers, finished with a "
                  "scalloped pearl-drop hem. The blouse is lilac, pearl "
                  "embroidered, with tasselled sleeves.",
        "drape": "the lilac net dupatta, bordered in scalloped pearl work "
                 "and hung with long ivory tassels, crosses her body from "
                 "the left shoulder diagonally down to her right hip and "
                 "falls open to the floor.",
        "colours": "soft lilac-mauve and ivory, with pearl and cream "
                   "embroidery",
    },

    "fg08": {
        "crop": (0.00, 0.00, 1.00, 0.92),
        "summary": "a peach-gold tissue saree with a magenta embroidered "
                   "border",
        "pieces": ["a saree draped around the body",
                   "a matching embroidered blouse"],
        "detail": "A saree is one long unstitched cloth: wrapped around the "
                  "waist, pleated at the front, and the loose end carried up "
                  "across the torso and over one shoulder. The cloth is "
                  "liquid peach-gold tissue silk. Its wide border and its "
                  "end-panel are deep magenta, encrusted with gold sequin "
                  "and pearl flowers and a scalloped gold edge. The blouse "
                  "is magenta, heavily gold-embroidered, fringed with pearls.",
        "drape": "the pleats fall at the front from the waist, and the "
                 "embroidered magenta end is carried over her left shoulder "
                 "and falls down her back.",
        "colours": "shimmering peach-gold, with a deep magenta border in "
                   "gold and pearl",
    },

    "fg09": {
        "crop": (0.00, 0.00, 1.00, 1.00),
        "summary": "a purple Banarasi silk saree",
        "pieces": ["a saree draped around the body",
                   "a matching blouse"],
        "detail": "A saree is one long unstitched cloth: wrapped around the "
                  "waist, pleated at the front, and the loose end carried up "
                  "across the torso and over one shoulder. The silk is deep "
                  "violet-purple with small gold zari flowers woven across "
                  "it, a wide gold zari brocade border of lotus and paisley, "
                  "and a gold fringe. The end-panel is densely woven in gold "
                  "brocade.",
        "drape": "the pleats fall at the front, and the gold brocade end is "
                 "carried over her left shoulder and falls down her back.",
        "colours": "deep violet purple with antique gold zari",
    },

    "fg10": {
        "crop": (0.02, 0.02, 1.00, 1.00),
        "summary": "a maroon Banarasi silk saree",
        "pieces": ["a saree draped around the body",
                   "a matching blouse"],
        "detail": "A saree is one long unstitched cloth: wrapped around the "
                  "waist, pleated at the front, and the loose end carried up "
                  "across the torso and over one shoulder. The silk is rich "
                  "wine-maroon, woven all over with fine gold zari flowering "
                  "vines, and finished with a broad gold zari brocade border "
                  "of lotus and paisley.",
        "drape": "the pleats fall at the front, and the wide gold-brocade "
                 "end is carried over her left shoulder and falls down "
                 "her back.",
        "colours": "deep wine maroon with warm gold zari",
    },

    "fg11": {
        "crop": (0.00, 0.00, 1.00, 1.00),
        "summary": "a bottle-green Banarasi silk saree",
        "pieces": ["a saree draped around the body",
                   "a matching blouse"],
        "detail": "A saree is one long unstitched cloth: wrapped around the "
                  "waist, pleated at the front, and the loose end carried up "
                  "across the torso and over one shoulder. The silk is deep "
                  "bottle green, scattered with small gold zari leaf motifs, "
                  "with a broad gold zari brocade border of scrolling "
                  "flowers and paisleys.",
        "drape": "the pleats fall at the front, and the brocade end is "
                 "carried over her left shoulder and falls down her back.",
        "colours": "dark bottle green with antique gold zari",
    },

    "fg12": {
        "crop": (0.26, 0.16, 0.90, 1.00),
        "summary": "a royal-blue and ivory bridal lehenga worn with two "
                   "dupattas",
        "pieces": ["an ivory blouse",
                   "a royal-blue flared lehenga skirt",
                   "a royal-blue embroidered dupatta",
                   "an ivory sheer dupatta"],
        "detail": "The skirt is royal blue, covered in a gold trellis of "
                  "geometric and floral panels with a heavy gold border. "
                  "The blouse is ivory with fine gold work.",
        "drape": "this outfit has TWO dupattas and both are worn. The royal-"
                 "blue gold-embroidered one is draped over her right "
                 "shoulder and falls open down her right side. The ivory "
                 "sheer one, edged in gold, is draped over her left shoulder "
                 "and falls down her left side.",
        "colours": "royal blue and ivory, with antique gold embroidery",
    },

    "fg13": {
        "crop": (0.20, 0.20, 0.80, 1.00),
        "summary": "a coral-pink evening set with an embroidered bodice",
        "pieces": ["a sleeveless embroidered peplum bodice",
                   "a full flared floor-length skirt",
                   "a sheer pink dupatta"],
        "detail": "The bodice is coral pink, worked in silver thread, pearls "
                  "and beaded flowers in swirling panels, finished with a "
                  "pearl-beaded hem. The skirt is soft rose-pink georgette, "
                  "plain and finely pleated, falling full to the floor.",
        "drape": "the sheer rose-pink dupatta is draped from her shoulders "
                 "and falls down her back on the left side.",
        "colours": "coral pink shading to soft rose, with silver and pearl "
                   "embroidery",
    },

    "fg14": {
        "crop": (0.05, 0.14, 0.97, 0.98),
        "summary": "a midnight-blue gown embroidered with cranes in a marsh",
        "pieces": ["a fitted strapless beaded corset bodice",
                   "a floor-length A-line skirt with sheer side drapes"],
        "detail": "The bodice is midnight navy, boned and closely beaded in "
                  "dark crystal. The skirt is the same midnight navy, "
                  "sprinkled with fine crystal, and around the hem is an "
                  "embroidered marsh scene — two white cranes standing among "
                  "reeds, ferns and pastel pink, lilac and green flowers, "
                  "worked in sequins and silk thread. Sheer navy drape "
                  "panels fall from the waist at both sides and pool on "
                  "the floor.",
        "drape": "",
        "colours": "midnight navy blue, with white cranes and pastel pink, "
                   "lilac, green and iridescent sequin work",
    },

    "fg15": {
        "crop": (0.00, 0.05, 1.00, 0.95),
        "summary": "a midnight-blue gown with beaded cap sleeves, embroidered "
                   "with cranes in a marsh",
        "pieces": ["a beaded bodice with cap sleeves",
                   "a floor-length A-line skirt with a sheer side train"],
        "detail": "The bodice is midnight navy, densely beaded in dark "
                  "crystal, with short beaded cap sleeves at the shoulders "
                  "and a square neckline. The skirt is the same navy, and "
                  "around the hem is an embroidered marsh scene — two white "
                  "cranes among reeds, ferns and pastel lilac and green "
                  "flowers in sequins and silk thread. A sheer sequinned "
                  "navy train sweeps out from the left hip.",
        "drape": "",
        "colours": "midnight navy blue, with white cranes and pastel lilac, "
                   "green and iridescent sequins",
    },

    "fg16": {
        "crop": (0.05, 0.15, 0.98, 1.00),
        "summary": "a champagne bridal lehenga worn with a maroon velvet "
                   "dupatta",
        "pieces": ["a V-necked half-sleeved champagne blouse",
                   "a very full flared floor-length skirt",
                   "a maroon velvet dupatta"],
        "detail": "The blouse and skirt are pale champagne-gold under dense "
                  "tonal embroidery in the same shade — scrolling flowers "
                  "and vines over the whole skirt, which flares out very "
                  "wide and sweeps the floor. A gold embroidered waistband "
                  "with a paisley motif sits at the hip.",
        "waist": "The V-necked blouse ends just below the bust, and the very full skirt starts from the gold embroidered waistband at her hip.",
        "drape": "the dupatta is deep maroon velvet with a gold embroidered "
                 "border, draped over her left shoulder and falling long "
                 "down her left side to the floor.",
        "colours": "pale champagne gold, with a deep maroon velvet dupatta",
    },

    "fg17": {
        "crop": (0.05, 0.15, 0.98, 1.00),
        "summary": "a champagne bridal lehenga worn with a maroon velvet "
                   "dupatta",
        "pieces": ["a V-necked half-sleeved champagne blouse",
                   "a very full flared floor-length skirt",
                   "a maroon velvet dupatta"],
        "detail": "The blouse and skirt are pale champagne-gold under dense "
                  "tonal embroidery in the same shade — scrolling flowers "
                  "and vines over the whole skirt, which flares out very "
                  "wide and sweeps the floor. A gold embroidered waistband "
                  "with a paisley motif sits at the hip.",
        "waist": "The V-necked blouse ends just below the bust, and the very full skirt starts from the gold embroidered waistband at her hip.",
        "drape": "the dupatta is deep maroon velvet with a gold embroidered "
                 "border, draped over her left shoulder and falling long "
                 "down her left side to the floor.",
        "colours": "pale champagne gold, with a deep maroon velvet dupatta",
    },

    "fg18": {
        "crop": (0.12, 0.00, 0.88, 1.00),
        "summary": "an ivory lehenga worn with a rust-red velvet dupatta",
        "pieces": ["a rust-red velvet blouse",
                   "an ivory flared floor-length skirt",
                   "a rust-red velvet dupatta"],
        "detail": "The blouse is rust-red velvet with a silver-embroidered "
                  "band and a beaded fringe. The skirt is ivory, lightly "
                  "worked, with a deep silver-embroidered border at the hem.",
        "drape": "the dupatta is large rust-red velvet, embroidered all over "
                 "in silver-white flowers and paisleys with a wide silver "
                 "border. It is draped over her head and both shoulders, "
                 "framing her face, and falls open down both sides to below "
                 "the knee.",
        "colours": "rust red velvet and ivory, with silver-white embroidery",
    },

    "fg19": {
        "crop": (0.10, 0.03, 1.00, 1.00),
        "summary": "a red and indigo block-printed saree",
        "pieces": ["a saree draped around the body",
                   "a matching printed blouse"],
        "detail": "A saree is one long unstitched cloth: wrapped around the "
                  "waist, pleated at the front, and the loose end carried up "
                  "across the torso and over one shoulder. The cloth is "
                  "brick red, block-printed in indigo blue and cream with "
                  "small paisley and floral panels, and edged with sequin "
                  "and mirror trim that glitters along the border.",
        "drape": "the pleats fall at the front, and the printed end is "
                 "carried over her left shoulder and falls down her back.",
        "colours": "brick red with indigo blue and cream printing, and "
                   "silver sequin trim",
    },

    "fg20": {
        "crop": (0.18, 0.30, 0.90, 0.86),
        "summary": "a dark green saree with a heavily embroidered border",
        "pieces": ["a saree draped around the body",
                   "a matching embroidered blouse"],
        "detail": "A saree is one long unstitched cloth: wrapped around the "
                  "waist, pleated at the front, and the loose end carried up "
                  "across the torso and over one shoulder. The body of the "
                  "cloth is plain deep forest green silk. Its wide border "
                  "and its whole end-panel are worked solid in antique gold "
                  "sequins and thread in medallions and geometric bands.",
        "drape": "the pleats fall at the front, and the gold-encrusted end "
                 "is carried over her left shoulder and falls long down "
                 "her back.",
        "colours": "deep forest green with antique gold sequin embroidery",
    },

    "fg21": {
        "crop": (0.10, 0.26, 0.95, 0.88),
        "summary": "a red-and-white striped saree with indigo block-printed "
                   "panels",
        "pieces": ["a saree draped around the body",
                   "a matching printed blouse"],
        "detail": "A saree is one long unstitched cloth: wrapped around the "
                  "waist, pleated at the front, and the loose end carried up "
                  "across the torso and over one shoulder. The body is "
                  "printed in narrow red and white vertical stripes, broken "
                  "by broad panels of indigo-blue and red Ajrakh block print "
                  "with cream borders, and scattered with small mirror "
                  "sequins.",
        "drape": "the pleats fall at the front, and the block-printed end is "
                 "carried over her left shoulder and falls long down her "
                 "back.",
        "colours": "red and white stripes with indigo blue and cream block "
                   "print",
    },

    "fg22": {
        "crop": (0.20, 0.08, 0.80, 0.97),
        "summary": "a champagne beaded one-shoulder evening gown",
        "pieces": ["a floor-length fitted one-shoulder gown"],
        "detail": "One continuous fitted gown in champagne-nude, covered in "
                  "silver crystal beadwork laid in long chevrons and "
                  "sweeping curves that follow the body, ending in a beaded "
                  "fringe at the floor. A trail of three-dimensional silver "
                  "leaves and lilies runs from the left shoulder diagonally "
                  "across the bodice to the right hip. There is a high slit "
                  "on the right leg.",
        "drape": "",
        "colours": "champagne nude with silver crystal beading",
    },

    "fg23": {
        "crop": (0.10, 0.03, 0.90, 1.00),
        "summary": "a black tailored trouser suit",
        "pieces": ["a black single-breasted blazer with gold buttons",
                   "an ivory silk shirt",
                   "black wide-leg tailored trousers",
                   "a black leather belt with a gold buckle"],
        "detail": "The blazer is black wool with notched lapels and gold "
                  "buttons, worn open. The shirt beneath is ivory silk, "
                  "collared and softly draped. The trousers are black, high "
                  "waisted, sharply creased and wide through the leg, held "
                  "by a black leather belt with a gold buckle.",
        "drape": "",
        "colours": "black, with an ivory silk shirt and gold hardware",
    },
}
