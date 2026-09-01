#!/usr/bin/env bash
# Build the visual showcase end to end: subjects -> catalogue -> try-on matrix -> gallery.
#
# This is the ONE step in this repository that needs a GPU. Everything else --
# the API, the tests, the diagrams, the benchmark tables -- runs without one.
# It is deliberately a thin wrapper over the pipeline entry points so there is
# no second implementation to drift.
#
#   SUBJECTS=./runs/subjects GARMENTS=./runs/garments bash scripts/build_showcase.sh
#
# SUBJECTS must hold the person photographs and GARMENTS the flat garment shots.
# Supply your own, or generate synthetic subjects -- this repository ships no
# imagery of people and will not invent results.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export TRYON_ROOT="${TRYON_ROOT:-$REPO/runs}"
SUBJECTS="${SUBJECTS:-$TRYON_ROOT/subjects}"
GARMENTS="${GARMENTS:-$TRYON_ROOT/garments}"
API="${TRYON_API:-http://127.0.0.1:8000}"
OUT="${OUT:-$REPO/docs/gallery}"

die() { printf '\n  %s\n\n' "$*" >&2; exit 1; }

# ---- preflight: fail loudly and specifically, never half-produce a showcase ----
command -v nvidia-smi >/dev/null 2>&1 || die \
  "no nvidia-smi: this step needs an NVIDIA GPU (>=12 GB). See docs/SETUP.md."

[ -d "$SUBJECTS" ] && [ -n "$(ls -A "$SUBJECTS" 2>/dev/null)" ] || die \
  "no subject photographs in $SUBJECTS
   Set SUBJECTS=/path/to/person/photos. This repository ships none by design --
   the original run used photographs of real people and those are not published."

[ -d "$GARMENTS" ] && [ -n "$(ls -A "$GARMENTS" 2>/dev/null)" ] || die \
  "no garment photographs in $GARMENTS
   Set GARMENTS=/path/to/flat/garment/photos."

curl -fsS "$API/readyz" >/dev/null 2>&1 || die \
  "backend not ready at $API/readyz
   Start it first ('make serve' or 'make docker-up') and wait for /readyz to
   return 200 -- it stays 503 until all four model files are loadable."

echo "==> subjects : $SUBJECTS"
echo "==> garments : $GARMENTS"
echo "==> run root : $TRYON_ROOT"

# ---- 1. catalogue every garment and subject with the vision model ----
echo "==> 1/3 cataloguing (vision model reads every photograph)"
python3 pipeline/run_catalogue.py --root "$TRYON_ROOT"

# ---- 2. generate every subject x garment pair, with the anatomy QA loop ----
echo "==> 2/3 generating the matrix (this is the long step)"
TRYON_SET="$SUBJECTS" python3 pipeline/run_matrix.py --steps 5 --seed 42 --mp 0.75

# ---- 3. build the browsable gallery from what came out ----
echo "==> 3/3 building the gallery"
mkdir -p "$OUT"
python3 reporting/build_gallery.py \
  --images "$TRYON_ROOT/final_out" \
  --report "$TRYON_ROOT/final_out/report.json" \
  --crops  "$TRYON_ROOT/cache/crops" \
  --out    "$OUT"

echo
echo "  done -> $OUT/gallery.html"
echo "  every image in it is one you generated; nothing is shipped in this repo."
