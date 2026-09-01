# data/

Symlinks to the run outputs, which live in ~/Downloads and are deliberately not
in git — the 4090 round alone is 145 MB of PNGs.

    runs-4090/      round 4 (23 garments), waist-fix, seed and LoRA sweeps, BEST/
    runs-5090/      the guarded rounds, retry round, resolution probes
    pair-maroon/    the one-off event-photo pair
    source-images/  the original garment and model photographs

The reporting scripts read from these paths:

    python3 reporting/build_benchmark.py --r5 data/runs-5090 --r4 data/runs-4090 \
        --out docs
    python3 reporting/build_gallery.py --images data/runs-4090/BEST \
        --report data/runs-4090/outputs_round4/f6_report.json \
        --crops  data/runs-4090/cache/crops/round4 \
        --model  "data/source-images/female models/model  (6).jpeg" --out docs

If a symlink is broken the run directory was moved; repoint it rather than
copying, so the outputs stay in one place.
