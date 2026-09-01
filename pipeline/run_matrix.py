#!/usr/bin/env python3
"""Full try-on pipeline with an anatomical QA loop.

    1  VLM reads every model photo (scene, pose, per-hand, build, face) and
       every garment photo (colour, metal, fit, description)
    2  generator produces every model x garment pair
    3  VLM inspects each output for extra hands, closed eyes, distorted faces
    4  failures are regenerated with a new seed, then re-inspected

The VLM and the generator cannot both hold VRAM on a 24 GB card, so the loop is
batched: one model swap per round, not per image.

usage: pipeline2.py [--models "model  (3)"] [--rounds 2] [--steps 5] [--mp 0.75]
       no --models  ->  every model in the set
"""
import argparse, gc, json, pathlib, re, subprocess, time

import os, sys

# Every path is environment-driven so this runs outside the box it was written on.
REPO  = pathlib.Path(__file__).resolve().parent.parent
ROOT  = pathlib.Path(os.environ.get("TRYON_ROOT", REPO / "runs"))
SET   = pathlib.Path(os.environ.get("TRYON_SET", ROOT / "set"))
OUT   = pathlib.Path(os.environ.get("TRYON_OUT", ROOT / "final_out")); OUT.mkdir(parents=True, exist_ok=True)
REJ   = OUT / "rejected"; REJ.mkdir(parents=True, exist_ok=True)
CAT   = pathlib.Path(os.environ.get("TRYON_CATALOGUE", ROOT / "catalogue_final.json"))
HFC   = os.environ.get("HF_HOME") or str(ROOT / ".hfcache")
VLM   = os.environ.get("TRYON_VLM", "Qwen/Qwen2.5-VL-7B-Instruct")
API   = os.environ.get("TRYON_API", "http://127.0.0.1:8000")

sys.path.insert(0, str(REPO))
from app.person_prompt import MODEL_ASK, GARMENT_ASK, MODEL_FIELDS, GARMENT_FIELDS, build
from app.guardrail import VERIFY_ASK, VERIFY_FIELDS, verdict

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="*", default=None)
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--steps", default="5")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--mp", default="0.75")
A = ap.parse_args()

CROP = {"model  (3)": (0, .06, 0, .03), "model  (5)": (0, .09, 0, .16),
        "model  (2)": (0, 0, 0, .05)}


def parse(t, fields):
    out, cur = {}, None
    for line in (t or "").splitlines():
        m = re.match(r"^\s*\**\s*(" + "|".join(fields) + r")\s*\**\s*:\s*(.*)$", line, re.I)
        if m: cur = m.group(1).upper(); out[cur] = m.group(2).strip()
        elif cur and line.strip(): out[cur] += " " + line.strip()
    return {k: re.sub(r"\s+", " ", v).strip(" *") for k, v in out.items()}


def api(up: bool):
    """ComfyUI keeps its weights resident, so it must be down for VLM work."""
    if up:
        subprocess.run(["tmux", "new-session", "-d", "-s", "tryon",
            f"cd {REPO} && PYTHONPATH=. {sys.executable} -m "
            f"uvicorn app.main:app --host 0.0.0.0 --port 8000 2>&1 | tee {ROOT}/api.log"],
            check=False)
        import urllib.request
        for _ in range(180):
            try:
                if urllib.request.urlopen(f"{API}/healthz", timeout=3).status == 200: return
            except Exception: pass
            time.sleep(1)
        raise SystemExit("API did not come up")
    # Killing the tmux session leaves uvicorn and its ComfyUI child orphaned,
    # still holding ~17 GB. Kill the processes, then wait for the card to clear.
    subprocess.run(["tmux", "kill-session", "-t", "tryon"], check=False,
                   capture_output=True)
    for pat in ("[u]vicorn", "[C]omfyUI/main.py"):
        subprocess.run(["pkill", "-f", pat], check=False)
    for _ in range(30):
        time.sleep(2)
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        if int(q.stdout.strip().splitlines()[0]) < 1500:
            return
    print("  warning: GPU still occupied after teardown", flush=True)


class Vision:
    """Loads on enter, frees the card on exit."""
    def __enter__(self):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(VLM, cache_dir=HFC)
        self.m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            VLM, dtype=torch.bfloat16, device_map="cuda:0", cache_dir=HFC)
        return self
    def ask(self, img, prompt, mx=900):
        from qwen_vl_utils import process_vision_info
        msgs = [{"role": "user", "content": [{"type": "image", "image": f"file://{img}"},
                                             {"type": "text", "text": prompt}]}]
        t = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        im, vi = process_vision_info(msgs)
        inp = self.proc(text=[t], images=im, videos=vi, padding=True,
                        return_tensors="pt").to("cuda:0")
        with self.torch.inference_mode():
            ids = self.m.generate(**inp, max_new_tokens=mx, do_sample=False)
        return self.proc.batch_decode([o[len(i):] for i, o in zip(inp.input_ids, ids)],
                                      skip_special_tokens=True)[0]
    def __exit__(self, *a):
        del self.m, self.proc
        gc.collect(); self.torch.cuda.empty_cache()


def gnum(p):
    d = "".join(c for c in pathlib.Path(p).stem if c.isdigit())
    return int(d) if d else 999


# ------------------------------------------------------------------ inputs

fem_models = sorted((SET / "female models").glob("*.jpeg"), key=gnum)
mal_models = sorted((SET / "male model").glob("*.jpeg"), key=gnum)
models = [(p, "woman") for p in fem_models] + [(p, "man") for p in mal_models]
if A.models:
    want = {m.strip() for m in A.models}
    models = [(p, s) for p, s in models if p.stem.strip() in want]
fem_gar = sorted((SET / "female garment").glob("*.jpeg"), key=gnum)
mal_gar = sorted((SET / "male garment").glob("*.jpeg"), key=gnum)

pairs = [(mp, sx, g) for mp, sx in models
         for g in (fem_gar if sx == "woman" else mal_gar)]
print(f"{len(models)} model(s) x garments = {len(pairs)} pairs\n", flush=True)

# ------------------------------------------------------------- 1 catalogue

if CAT.exists():
    cat = json.loads(CAT.read_text())
    print("catalogue: cached\n", flush=True)
else:
    api(False)
    cat = {"models": {}, "garments": {}}
    t0 = time.time()
    with Vision() as V:
        for mp, sx in models:
            cat["models"][str(mp)] = parse(V.ask(mp, MODEL_ASK), MODEL_FIELDS)
            d = cat["models"][str(mp)]
            print(f"  model {mp.stem:14} L={d.get('LEFT_HAND','?')[:34]:36} "
                  f"R={d.get('RIGHT_HAND','?')[:34]}", flush=True)
        for g in {p for _, _, p in pairs}:
            cat["garments"][str(g)] = parse(V.ask(g, GARMENT_ASK), GARMENT_FIELDS)
    CAT.write_text(json.dumps(cat, indent=2))
    print(f"\ncatalogue: {len(cat['models'])} models + {len(cat['garments'])} garments "
          f"in {time.time()-t0:.0f}s\n", flush=True)

# --------------------------------------------------------- prepped person

from PIL import Image
prepped = {}
for mp, sx in models:
    if mp.stem in CROP:
        l, t, r, b = CROP[mp.stem]
        im = Image.open(mp).convert("RGB"); w, h = im.size
        q = OUT / f"person_{mp.stem.replace(' ','')}.jpg"
        im.crop((int(w*l), int(h*t), int(w*(1-r)), int(h*(1-b)))).save(q, quality=95)
        prepped[str(mp)] = q
    else:
        prepped[str(mp)] = mp

# ------------------------------------------------------ 2/3/4 generate+QA

import httpx
c = httpx.Client(base_url=API, timeout=1800.0)

def generate(mp, sx, g, seed):
    prompt = build(sx, cat["models"][str(mp)], cat["garments"][str(g)])
    p = prepped[str(mp)]
    with open(p, "rb") as pf, g.open("rb") as gf:
        r = c.post("/v1/tryon", files={"person": (p.name, pf), "garment": (g.name, gf)},
                   data={"prompt": prompt, "steps": A.steps, "seed": str(seed),
                         "megapixels": A.mp})
    r.raise_for_status(); job = r.json()["job_id"]
    while True:
        info = c.get(f"/v1/jobs/{job}").json()
        if info["status"] in ("succeeded", "failed"): return info, job
        time.sleep(0.5)

todo = [(mp, sx, g, A.seed) for mp, sx, g in pairs]
records, t_all = {}, time.time()

for rnd in range(1, A.rounds + 1):
    if not todo: break
    print(f"=== round {rnd}: generating {len(todo)} ===", flush=True)
    api(True)
    made = []
    for i, (mp, sx, g, seed) in enumerate(todo, 1):
        tag = f"{mp.stem.replace(' ','')}__g{gnum(g):02d}"
        info, job = generate(mp, sx, g, seed)
        if info["status"] == "failed":
            print(f"  [{i}/{len(todo)}] {tag:22} GEN-FAIL", flush=True); continue
        img = c.get(f"/v1/jobs/{job}/image").content
        dest = OUT / f"{tag}.png"; dest.write_bytes(img)
        print(f"  [{i}/{len(todo)}] {tag:22} {info['duration_seconds']:>5.1f}s", flush=True)
        made.append((mp, sx, g, seed, tag, dest, info["duration_seconds"]))

    print(f"\n=== round {rnd}: inspecting {len(made)} ===", flush=True)
    api(False)
    retry = []
    with Vision() as V:
        for mp, sx, g, seed, tag, dest, secs in made:
            chk = parse(V.ask(dest, VERIFY_ASK, mx=700), VERIFY_FIELDS)
            ok, why = verdict(chk)
            records[tag] = {"tag": tag, "ok": ok, "seconds": secs, "seed": seed,
                            "model": mp.name, "garment": g.name, "round": rnd,
                            "result": str(dest), "model_file": str(prepped[str(mp)]),
                            "garment_file": str(g), "qa": chk, "reason": why,
                            "steps": int(A.steps), "megapixels": float(A.mp)}
            if ok:
                print(f"  PASS {tag}", flush=True)
            else:
                print(f"  FAIL {tag:22} {why}", flush=True)
                dest.replace(REJ / f"{tag}_r{rnd}.png")
                retry.append((mp, sx, g, seed + 1000 * rnd))
    todo = retry
    print(flush=True)

good = [r for r in records.values() if r["ok"]]
(OUT / "summary.json").write_text(json.dumps(list(records.values()), indent=2))
print(f"=== {len(good)}/{len(pairs)} passed QA | {len(pairs)-len(good)} still failing "
      f"| {(time.time()-t_all)/60:.1f} min ===")
for r in records.values():
    if not r["ok"]:
        print(f"  unresolved: {r['tag']:22} {r['reason']}")
