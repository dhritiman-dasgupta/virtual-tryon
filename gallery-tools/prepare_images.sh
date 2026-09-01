#!/bin/zsh
# Re-derive the web assets from the raw run output.
#
#   SRC  = the original run folders (raw JPEG portraits/garments + PNG results)
#   DEST = this repository
#
# Every asset in assets/ is a WebP re-encode of a source file — full size for the
# lightbox, a small one for the grids. Nothing here is generated; it is only
# resized. Requires cwebp (brew install webp).
set -e
SRC="${1:?usage: prepare_images.sh <source-dir> [dest-dir]}"
DEST="${2:-$(cd "$(dirname "$0")/.." && pwd)}"

mkdir -p "$DEST"/assets/{models,garments,results,thumbs/results,thumbs/models,thumbs/garments}

enc() { cwebp -quiet -q "$3" -resize "$2" 0 -m 4 "$1" -o "$4"; }

# portraits — note the double space in "model  (N).jpeg"
for i in 1 2 3 4 5 6 7; do
  s="$SRC/sample images/female models/model  ($i).jpeg"
  enc "$s" 760 82 "$DEST/assets/models/f$i.webp"
  enc "$s" 260 74 "$DEST/assets/thumbs/models/f$i.webp"
done
s="$SRC/sample images/male model/model.jpeg"
enc "$s" 760 82 "$DEST/assets/models/m1.webp"
enc "$s" 260 74 "$DEST/assets/thumbs/models/m1.webp"

# garments
for i in $(seq 1 23); do
  n=$(printf "%02d" $i); s="$SRC/sample images/female garment/garment ($i).jpeg"
  enc "$s" 760 82 "$DEST/assets/garments/fg$n.webp"
  enc "$s" 260 74 "$DEST/assets/thumbs/garments/fg$n.webp"
done
for i in 1 7 8 9 10 16 17 18 19 20; do
  n=$(printf "%02d" $i); s="$SRC/sample images/male garment/garment ($i).jpeg"
  enc "$s" 760 82 "$DEST/assets/garments/mg$n.webp"
  enc "$s" 260 74 "$DEST/assets/thumbs/garments/mg$n.webp"
done
s="$SRC/sample images/male garment/WhatsApp Image 2026-08-14 at 1.52.42 PM (1).jpeg"
enc "$s" 760 82 "$DEST/assets/garments/mg99.webp"
enc "$s" 260 74 "$DEST/assets/thumbs/garments/mg99.webp"

# results
for i in 1 2 3 4 5 6 7; do
  for g in $(seq 1 23); do
    n=$(printf "%02d" $g); s="$SRC/run-172/model ($i)/model_($i)__garment_$n.png"
    enc "$s" 768 80 "$DEST/assets/results/f${i}__fg$n.webp"
    enc "$s" 300 72 "$DEST/assets/thumbs/results/f${i}__fg$n.webp"
  done
done
for g in 01 07 08 09 10 16 17 18 19 20; do
  s="$SRC/run-172/male model/male_model__garment_$g.png"
  enc "$s" 768 80 "$DEST/assets/results/m1__mg$g.webp"
  enc "$s" 300 72 "$DEST/assets/thumbs/results/m1__mg$g.webp"
done
s="$SRC/run-172/male model/male_model__garment_20260814152421.png"
enc "$s" 768 80 "$DEST/assets/results/m1__mg99.webp"
enc "$s" 300 72 "$DEST/assets/thumbs/results/m1__mg99.webp"

echo "assets rebuilt into $DEST/assets"
