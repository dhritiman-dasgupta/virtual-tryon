#!/usr/bin/env python
"""Assemble a labelled evaluation set for choosing the guardrail's vision model.

The hard part of a QA benchmark is trustworthy labels. Adjudicating "is this
subtly the same woman" by eye produces labels an evaluator can argue with, so
most items here are built so the correct answer cannot be argued:

    same-person / same-background FAIL   pair a person photo with an output
                                         generated from a different person
                                         entirely - the answer is not a
                                         judgement call
    garment-type / match FAIL            pair a garment photo with the output
                                         for a different garment
    garment-pieces FAIL                  the round-4 images where a two-piece
                                         set rendered as one continuous panel,
                                         labelled by eye and later confirmed by
                                         the fact that a waist clause or a
                                         reseed changed them
    all-PASS                             round-4 outputs reviewed image by image

A model that only ever answers PASS scores 0 on the negatives; one that only
answers FAIL scores 0 on the positives. Both halves are needed, which is the
point of the set.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil

# Round 4 outputs reviewed by eye: person, background, hands and face correct on
# all 23; these five rendered a two-piece set as one continuous panel.
FUSED = ["fg01", "fg04", "fg06", "fg16", "fg17"]
CLEAN = ["fg02", "fg03", "fg05", "fg07", "fg08", "fg09", "fg10", "fg11",
         "fg12", "fg13", "fg18", "fg19", "fg20", "fg21", "fg22", "fg23"]

ALL_PASS = {"ONE_PERSON": "PASS", "SAME_PERSON": "PASS", "SAME_BACKGROUND": "PASS",
            "HANDS": "PASS", "GARMENT_TYPE": "PASS", "GARMENT_MATCH": "PASS",
            "GARMENT_PIECES": "PASS"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs4090", default="./runs/4090")
    ap.add_argument("--noprompt", default="./runs/noprompt")
    ap.add_argument("--source", default="./samples/inputs")
    ap.add_argument("--out", default="./runs/vlm-eval")
    a = ap.parse_args()

    ex = lambda p: pathlib.Path(p).expanduser()
    r4, npr, src, out = ex(a.runs4090), ex(a.noprompt), ex(a.source), ex(a.out)
    for sub in ("person", "garment", "output"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    model6 = src / "female models" / "model  (6).jpeg"
    crops = r4 / "cache/crops/round4"
    outputs = r4 / "outputs_round4/f6"
    items: list[dict] = []

    def copy(kind: str, srcp: pathlib.Path, name: str) -> str:
        dst = out / kind / name
        if not dst.exists():
            shutil.copy(srcp, dst)
        return f"{kind}/{name}"

    person6 = copy("person", model6, "model6.jpeg")

    # --- clean positives -----------------------------------------------------
    for g in CLEAN[:8]:
        items.append({
            "id": f"pass_{g}",
            "person": person6,
            "garment": copy("garment", crops / f"{g}.jpg", f"{g}.jpg"),
            "output": copy("output", outputs / f"{g}.png", f"r4_{g}.png"),
            "labels": dict(ALL_PASS),
            "why": "reviewed by eye: one person, face, hands and background intact",
        })

    # --- fused two-piece: everything else correct ---------------------------
    for g in FUSED:
        labels = dict(ALL_PASS)
        labels["GARMENT_PIECES"] = "FAIL"
        items.append({
            "id": f"fused_{g}",
            "person": person6,
            "garment": copy("garment", crops / f"{g}.jpg", f"{g}.jpg"),
            "output": copy("output", outputs / f"{g}.png", f"r4_{g}.png"),
            "labels": labels,
            "why": "blouse and skirt rendered as one continuous panel",
        })

    # --- wrong person: an output made from a different woman -----------------
    # The person image is model 6; the output is a different model's photo, so
    # SAME_PERSON and SAME_BACKGROUND are unambiguously wrong.
    wrong_person = [("f2__fg04", "fg04"), ("f2__fg12", "fg12"),
                    ("f1__fg18", "fg18"), ("f1__fg22", "fg22")]
    for stem, g in wrong_person:
        p = npr / f"{stem}.png"
        if not p.exists():
            continue
        labels = dict(ALL_PASS)
        labels.update(SAME_PERSON="FAIL", SAME_BACKGROUND="FAIL")
        items.append({
            "id": f"wrongperson_{stem}",
            "person": person6,
            "garment": copy("garment", crops / f"{g}.jpg", f"{g}.jpg"),
            "output": copy("output", p, f"np_{stem}.png"),
            "labels": labels,
            "why": "output is a different woman in a different place",
        })

    # --- wrong garment: right person, output for another garment -------------
    # Rotating the garment by a fixed offset guarantees a mismatch without
    # hand-picking, and the pairs chosen below are different garment TYPES so
    # GARMENT_TYPE is wrong too, not merely the colour.
    wrong_garment = [("fg09", "fg02"),   # purple saree output vs fuchsia lehenga
                     ("fg23", "fg13"),   # black suit output vs pink gown set
                     ("fg20", "fg07"),   # green saree output vs lilac lehenga
                     ("fg22", "fg10")]   # beaded gown output vs maroon saree
    for out_g, gar_g in wrong_garment:
        labels = dict(ALL_PASS)
        labels.update(GARMENT_TYPE="FAIL", GARMENT_MATCH="FAIL")
        items.append({
            "id": f"wronggarment_{out_g}_vs_{gar_g}",
            "person": person6,
            "garment": copy("garment", crops / f"{gar_g}.jpg", f"{gar_g}.jpg"),
            "output": copy("output", outputs / f"{out_g}.png", f"r4_{out_g}.png"),
            "labels": labels,
            "why": f"output wears {out_g}, garment photo shows {gar_g}",
        })

    (out / "labels.json").write_text(json.dumps(items, indent=2))

    n_fail = sum(1 for i in items for v in i["labels"].values() if v == "FAIL")
    n_pass = sum(1 for i in items for v in i["labels"].values() if v == "PASS")
    print(f"{len(items)} items -> {out}")
    print(f"{n_pass} PASS labels, {n_fail} FAIL labels")
    for kind in ("pass", "fused", "wrongperson", "wronggarment"):
        print(f"  {kind:14s} {sum(1 for i in items if i['id'].startswith(kind))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
