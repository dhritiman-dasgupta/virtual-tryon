"""Hand-written specs for one-off pairs outside the 23-garment catalogue.

Same shape as garments.GARMENTS so build() works unchanged, but the
PRESERVE block differs per person — the catalogue's block describes model 6 in
her night garden, which would be actively wrong here.
"""
from __future__ import annotations

# The new model: photographed at an event under violet uplighting, ice
# sculptures behind her, holding a phone in her right hand with a flower band
# and a red rose on that wrist. All of it named because the generator alters
# whatever the prompt leaves unsaid.
ROLE_EVENT = (
    "The first image is a photograph of one woman. The second image shows a "
    "garment on its own. Edit the first photograph so that the same woman is "
    "wearing that garment.\n"
    "There is exactly ONE person in the result: the woman from the first "
    "photograph, standing exactly where she already stands."
)

PRESERVE_EVENT = (
    "KEEP EXACTLY AS IN THE FIRST PHOTOGRAPH:\n"
    "FACE: the same woman — the same face, the same closed-lip smile, the same "
    "round cheeks, the same age, the same skin tone, the same small dark bindi "
    "and the red sindoor in her hair parting. She reads as the same person at "
    "the same party.\n"
    "HAIR: dark hair drawn back from her face, a few strands loose at the "
    "temple.\n"
    "BODY: the same build, the same height, the same proportions. The garment "
    "is fitted to the body already in the photograph.\n"
    "POSE: standing turned slightly to her right, shoulders angled to the "
    "camera, head tilted a little, smiling just past the lens. Her right "
    "forearm is raised across her waist and she is holding a dark navy phone "
    "in her right hand, still held in exactly that way. Her left arm stays "
    "down at her side.\n"
    "HANDS: this woman has TWO hands in total and this is the complete list of "
    "them — one left hand and one right hand, each with five separated "
    "fingers. The right hand is closed around the phone; the left hand rests "
    "at her side.\n"
    "JEWELLERY AND WRIST: a gold kundan choker set with red stones at her "
    "throat, long gold jhumka earrings hung with red beads, and on her right "
    "wrist a band of small white flowers with a single red rose. All of it "
    "stays exactly as it is.\n"
    "BACKGROUND: the same event hall, unchanged — deep violet and purple "
    "uplighting across the back wall, a pale carved ice sculpture behind her "
    "on the right, a second ice piece and a low lit water feature further "
    "back, dark steps, a dark polished floor, and the shoulder of a man in a "
    "dark suit at the very left edge of the frame. The same violet light on "
    "her and the same darkness at the edges.\n"
    "FRAMING: the same tall narrow crop, from the top of her head down past "
    "her knees, at the same camera distance and angle."
)

SPECS: dict[str, dict] = {

    "maroon_velvet_saree": {
        # The mannequin sits centre-frame in a wide empty gallery. Cropping to
        # the garment keeps the room's arch, floor and lamps out of the input,
        # which is the only reliable way to keep them out of the result.
        "crop": (0.30, 0.18, 0.83, 1.00),
        "summary": "a deep maroon velvet bridal saree with heavy gold zari work",
        "pieces": ["a saree draped around the body",
                   "a matching high-necked velvet blouse"],
        "detail": "A saree is one long unstitched cloth: wrapped around the "
                  "waist, pleated at the front, and the loose end carried up "
                  "across the torso and over one shoulder. This one is deep "
                  "wine-maroon velvet, patterned all over with a gold zari "
                  "trellis of linked medallions and small flowers. Its border "
                  "is very wide and worked solid in gold — scrolling florals, "
                  "paisleys and a scalloped gold edge — and the same heavy "
                  "gold work fills the end-panel. The blouse is the same "
                  "maroon velvet, high at the neck with long sleeves, "
                  "embroidered in gold and pearl across the yoke.",
        "drape": "the pleats fall at the front from the waist, and the "
                 "gold-bordered end is carried up across her torso and over "
                 "her left shoulder, falling open down her front-left with the "
                 "wide gold border showing along its edge.",
        "colours": "deep wine maroon velvet with antique gold zari",
        "role": ROLE_EVENT,
        "preserve": PRESERVE_EVENT,
    },
}
