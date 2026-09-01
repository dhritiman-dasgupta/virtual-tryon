#!/usr/bin/env python3
"""Generate the Colab notebook from the real project sources.

Embedding the actual files (base64, so the Chinese trigger prompt and nested
quotes survive) means the notebook and the deployable project cannot drift.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "TryOn_API_Colab.ipynb"

EMBED = [
    "app/__init__.py",
    "app/config.py",
    "app/schemas.py",
    "app/workflow.py",
    "app/prompts.py",
    "app/comfy_client.py",
    "app/jobs.py",
    "app/main.py",
    "workflows/tryon_api.json",
    "requirements.txt",
]

payload = {
    rel: base64.b64encode((ROOT / rel).read_bytes()).decode()
    for rel in EMBED
}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip().splitlines(True)}


def code(src: str, title: str | None = None) -> dict:
    body = (f"#@title {title}\n" if title else "") + src.strip() + "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"cellView": "form"} if title else {},
        "outputs": [],
        "source": body.splitlines(True),
    }


cells = [
    md("""
# Virtual Try-On API — notebook harness

FLUX.2 **klein 9B** + fal's virtual try-on LoRA, served as a real HTTP API
(not a Gradio demo), with a public tunnel so you can call it from anywhere.

Both klein 9B and the LoRA are **Apache-2.0** — this stack is commercially
deployable, unlike FLUX.2 `[dev]`.

Runs unchanged in **Google Colab** or on any **JupyterLab GPU server** (RACE,
RunPod, your own VM). Cell 1 detects which and picks its paths accordingly:
`/content` on Colab, `~/tryon` on a server.

- **Colab:** Runtime → Change runtime type → GPU, then Runtime → Run all.
- **JupyterLab:** upload this file, open it, then Run → Run All Cells.

| Cell | Does | Time |
|---|---|---|
| 1 | GPU + environment check | instant |
| 2 | Install ComfyUI + ComfyUI-GGUF | ~3 min |
| 3 | Write the API source | instant |
| 4 | Download ~19 GB of weights | ~10-15 min |
| 5 | Launch API + public tunnel | ~2 min |
| 6 | Health check | instant |
| 7 | Run a try-on | ~30-90 s |
| 8 | Optional quick UI | instant |

Every step checks its own result. If something fails you get the actual error,
not a green progress bar followed by a mystery crash three cells later.
"""),

    code(r'''
import subprocess, sys, shutil, os

print("=" * 62)
r = subprocess.run(["nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader"],
                   capture_output=True, text=True)
if r.returncode != 0:
    raise SystemExit("No GPU. Runtime -> Change runtime type -> GPU, then rerun.")

name, mem, driver = [x.strip() for x in r.stdout.strip().split(",")]
vram_gb = int(mem.split()[0]) / 1024
print(f"GPU      : {name}")
print(f"VRAM     : {vram_gb:.1f} GB")
print(f"Driver   : {driver}")
print(f"Python   : {sys.version.split()[0]}")

import torch
print(f"Torch    : {torch.__version__}  (CUDA {torch.version.cuda})")
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ required.")

# Peak VRAM is ~10-12 GB because ComfyUI streams modules; the 19 GB is on disk.
LOWVRAM = vram_gb < 16
print(f"\nlowvram  : {LOWVRAM}" + ("  (T4-class card, expect slower runs)" if LOWVRAM else ""))

# Works in Colab and in a plain JupyterLab server (RACE, RunPod, your own VM).
IN_COLAB = "google.colab" in sys.modules or os.path.isdir("/content")
BASE = "/content" if IN_COLAB else os.path.expanduser("~/tryon")
os.makedirs(BASE, exist_ok=True)

free_gb = shutil.disk_usage(BASE).free / 1024**3
print(f"Disk free: {free_gb:.1f} GB  (at {BASE})")
if free_gb < 25:
    print("  WARNING: under 25 GB free — the weights are ~19 GB and may not fit.")

print(f"Runtime  : {'Google Colab' if IN_COLAB else 'JupyterLab / server'}")
print("=" * 62)

os.environ["LOWVRAM"] = "1" if LOWVRAM else ""
os.environ["IN_COLAB"] = "1" if IN_COLAB else ""
os.environ["TRYON_BASE"] = BASE
os.environ["COMFY_ROOT"] = f"{BASE}/ComfyUI"
os.environ["TRYON_PROJECT"] = f"{BASE}/virtual-tryon"
''', "1. GPU & environment check"),

    code(r'''
import subprocess, sys, os, textwrap

COMFY_ROOT = os.environ["COMFY_ROOT"]   # set by cell 1

def run(desc, cmd, fatal=True):
    """Run a step and SHOW the failure. The notebook this replaces used
    capture_output=True with no return-code check, so a 404 looked like success."""
    print(f"--> {desc}")
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if p.returncode != 0:
        tail = (p.stderr or p.stdout or "").strip().splitlines()[-15:]
        print(textwrap.indent("\n".join(tail), "    | "))
        if fatal:
            raise SystemExit(f"FAILED: {desc}")
        print(f"    (non-fatal: {desc})")
    return p.returncode == 0

# Colab already ships a working CUDA torch build. Reinstalling it is slow and
# frequently breaks the runtime, so we deliberately do NOT touch torch here.
run("Installing API dependencies",
    f"{sys.executable} -m pip install -q "
    "'fastapi>=0.115' 'uvicorn[standard]>=0.32' python-multipart "
    "'pydantic>=2.9' 'pydantic-settings>=2.6' 'httpx>=0.27' 'websockets>=13'")

if not os.path.isdir(f"{COMFY_ROOT}/.git"):
    run("Cloning ComfyUI",
        f"git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git {COMFY_ROOT}")
else:
    print("--> ComfyUI already present")

run("Installing ComfyUI requirements",
    f"{sys.executable} -m pip install -q -r {COMFY_ROOT}/requirements.txt")

# Only ONE custom pack is needed. The upstream workflow pulled in rgthree,
# LayerStyle and Comfyroll purely for debug/preview nodes; those were stripped
# from the graph, so UnetLoaderGGUF is the last remaining dependency.
if not os.path.isdir(f"{COMFY_ROOT}/custom_nodes/ComfyUI-GGUF/.git"):
    run("Cloning ComfyUI-GGUF",
        f"git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git "
        f"{COMFY_ROOT}/custom_nodes/ComfyUI-GGUF")
run("Installing gguf", f"{sys.executable} -m pip install -q gguf")

for d in ("unet", "clip", "vae", "loras"):
    os.makedirs(f"{COMFY_ROOT}/models/{d}", exist_ok=True)

CF_BIN = f"{os.environ['TRYON_BASE']}/cloudflared"
if not os.path.exists(CF_BIN):
    run("Installing cloudflared",
        "wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/"
        f"cloudflared-linux-amd64 -O {CF_BIN} && chmod +x {CF_BIN}", fatal=False)
os.environ["CLOUDFLARED_BIN"] = CF_BIN

print("\nInstall complete.")
''', "2. Install ComfyUI + dependencies"),

    code(r'''
import base64, json, os, pathlib

PROJECT = os.environ["TRYON_PROJECT"]   # set by cell 1

# Files are embedded base64 so the Chinese trigger prompt and nested quotes in
# prompts.py survive verbatim. Generated from the deployable project — the two
# cannot drift.
FILES = __FILES__

for rel, b64 in FILES.items():
    dest = pathlib.Path(PROJECT) / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(base64.b64decode(b64))
    print(f"  {rel:32} {dest.stat().st_size:>7,} bytes")

# Verify the graph survived the round trip and has no dangling links.
g = json.loads((pathlib.Path(PROJECT) / "workflows/tryon_api.json").read_text())
dangling = [(n, k, v[0]) for n, node in g.items()
            for k, v in node.get("inputs", {}).items()
            if isinstance(v, list) and v and isinstance(v[0], str) and v[0] not in g]
print(f"\nworkflow: {len(g)} nodes, dangling links: {dangling or 'none'}")
assert not dangling

os.environ["PYTHONPATH"] = PROJECT
print(f"\nSource written to {PROJECT}")
''', "3. Write API source"),

    code(r'''
import subprocess, os, sys

COMFY_ROOT = os.environ["COMFY_ROOT"]   # set by cell 1

ASSETS = [
    ("https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/resolve/main/flux-2-klein-9b-Q8_0.gguf",
     "unet",  "flux-2-klein-9b-Q8_0.gguf",        6000),
    ("https://huggingface.co/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors",
     "clip",  "qwen_3_8b_fp8mixed.safetensors",   4000),
    ("https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors",
     "vae",   "flux2-vae.safetensors",              50),
    ("https://huggingface.co/fal/flux-klein-9b-virtual-tryon-lora/resolve/main/flux-klein-tryon-comfy.safetensors",
     "loras", "flux-klein-tryon-comfy.safetensors",  10),
]

subprocess.run("apt-get install -y -qq aria2 >/dev/null 2>&1 || true", shell=True)

for url, subdir, fname, min_mb in ASSETS:
    dest_dir = f"{COMFY_ROOT}/models/{subdir}"
    path = f"{dest_dir}/{fname}"

    if os.path.exists(path) and os.path.getsize(path) // 1024**2 >= min_mb:
        print(f"[skip] {fname} ({os.path.getsize(path)//1024**2} MB)")
        continue

    print(f"[get ] {fname} ...")
    subprocess.run(
        f'aria2c -c -x 16 -s 16 -k 1M --console-log-level=warn '
        f'--auto-file-renaming=false --allow-overwrite=true '
        f'"{url}" -d "{dest_dir}" -o "{fname}"', shell=True)

    # Verify. A 404 from HuggingFace writes a small HTML body that aria2c
    # reports as a successful download.
    if not os.path.exists(path):
        raise SystemExit(f"FAILED: {fname} missing after download")
    got = os.path.getsize(path) // 1024**2
    if got < min_mb:
        raise SystemExit(f"FAILED: {fname} is {got} MB, expected >= {min_mb} MB "
                         f"(bad URL, gated repo, or partial download)")
    print(f"[ok  ] {fname} ({got} MB)")

print("\nAll four assets verified.")
''', "4. Download model weights (~19 GB)"),

    code(r'''
#  TUNNEL choice:
#    cloudflared - no account, unmetered, no interstitial. Random URL each run.
#    ngrok       - needs a free authtoken. Since Feb 2026 the free plan caps
#                  1 GB/month bandwidth (~250 try-ons), 20k requests/month and
#                  2-hour sessions, and shows an interstitial page. A paid plan
#                  removes all of that and gives you a stable domain.
TUNNEL          = "cloudflared"  #@param ["cloudflared", "ngrok"]
NGROK_AUTHTOKEN = ""             #@param {type:"string"}
NGROK_DOMAIN    = ""             #@param {type:"string"}
API_KEY         = ""             #@param {type:"string"}

import os, re, subprocess, sys, time, threading, urllib.request

PROJECT = os.environ["TRYON_PROJECT"]   # set by cell 1

env = dict(os.environ)
env.update({
    "PYTHONPATH": PROJECT,
    "COMFY_ROOT": os.environ["COMFY_ROOT"],
    "COMFY_MANAGE": "true",
    "COMFY_EXTRA_ARGS": "--lowvram" if os.environ.get("LOWVRAM") else "",
    "COMFY_BOOT_TIMEOUT": "900",
    "API_KEYS": API_KEY,     # empty = no auth; anyone with the URL can use the GPU
    "DEFAULT_STEPS": "8",
    "WORKERS": "1",
})

api = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app.main:app",
     "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"],
    cwd=PROJECT, env=env,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

def pump():
    for line in api.stdout:
        print("[api]", line.rstrip())

threading.Thread(target=pump, daemon=True).start()

print("Waiting for the API to become ready (ComfyUI loads ~19 GB lazily)...")
ready = False
for _ in range(900):
    if api.poll() is not None:
        raise SystemExit(f"API exited with code {api.returncode} — see [api] lines above")
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/healthz", timeout=3) as r:
            if r.status == 200:
                ready = True
                break
    except Exception:
        pass
    time.sleep(1)

if not ready:
    raise SystemExit("API did not come up in 900 s")
print("\nAPI is up on :8000")

PUBLIC_URL = None

if TUNNEL == "ngrok":
    if not NGROK_AUTHTOKEN.strip():
        raise SystemExit(
            "ngrok needs an authtoken. Get one free at\n"
            "  https://dashboard.ngrok.com/get-started/your-authtoken\n"
            "paste it into NGROK_AUTHTOKEN above, and rerun this cell."
        )
    subprocess.run(f"{sys.executable} -m pip install -q pyngrok", shell=True, check=True)
    from pyngrok import conf, ngrok

    conf.get_default().auth_token = NGROK_AUTHTOKEN.strip()
    ngrok.kill()  # clear any tunnel left over from a previous run of this cell

    opts = {"addr": 8000, "proto": "http"}
    if NGROK_DOMAIN.strip():
        opts["domain"] = NGROK_DOMAIN.strip()
    try:
        PUBLIC_URL = ngrok.connect(**opts).public_url
    except Exception as e:
        raise SystemExit(
            f"ngrok failed to connect: {e}\n"
            "Common causes: bad authtoken, another agent already online "
            "(free plan allows limited concurrent endpoints), or a reserved "
            "domain your plan does not own."
        )
else:
    # cloudflared quick tunnel: no account, unmetered, no interstitial.
    cf = os.environ.get("CLOUDFLARED_BIN", "cloudflared")
    if not (os.path.exists(cf) or __import__("shutil").which(cf)):
        raise SystemExit(f"cloudflared not found at {cf} — rerun cell 2, or switch TUNNEL to ngrok")
    tunnel = subprocess.Popen(
        [cf, "tunnel", "--url", "http://localhost:8000", "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in tunnel.stdout:
        if m := re.search(r"https://[-\w]+\.trycloudflare\.com", line):
            PUBLIC_URL = m.group(0)
            break

if not PUBLIC_URL:
    raise SystemExit("tunnel did not produce a URL — see the output above")

os.environ["PUBLIC_URL"] = PUBLIC_URL
print("\n" + "=" * 62)
print(f"  TUNNEL     : {TUNNEL}")
print(f"  PUBLIC URL : {PUBLIC_URL}")
print(f"  API DOCS   : {PUBLIC_URL}/docs")
print(f"  AUTH       : {'X-API-Key required' if API_KEY else 'NONE (open)'}")
print("=" * 62)

if not API_KEY:
    print("\nNo API key set — anyone with this URL can use your GPU. Fine for a")
    print("test session; put a random string in API_KEY above for anything else.")

if TUNNEL == "ngrok":
    print("\nngrok free plan serves an interstitial page to browser traffic.")
    print("Send this header on requests from your frontend to skip it:")
    print('    "ngrok-skip-browser-warning": "true"')

# Verify the tunnel actually reaches the API rather than just resolving.
try:
    req = urllib.request.Request(f"{PUBLIC_URL}/healthz",
                                 headers={"ngrok-skip-browser-warning": "true"})
    with urllib.request.urlopen(req, timeout=30) as r:
        print(f"\ntunnel round-trip OK (HTTP {r.status})")
except Exception as e:
    print(f"\nWARNING: could not reach the API through the tunnel: {e}")
''', "5. Launch API + public tunnel"),

    code(r'''
import json, urllib.request

with urllib.request.urlopen("http://127.0.0.1:8000/healthz", timeout=15) as r:
    h = json.load(r)

print(f"status         : {h['status']}")
print(f"comfy reachable: {h['comfy_reachable']}")
print(f"queue depth    : {h['queue_depth']}")
print("\nmodels visible to ComfyUI:")
for name, ok in (h.get("models_present") or {}).items():
    print(f"  {'OK     ' if ok else 'MISSING'}  {name}")
if h.get("detail"):
    print(f"\ndetail: {h['detail']}")

if h["status"] != "ok":
    print("\n^ fix this before generating. A MISSING model usually means the")
    print("  download cell was interrupted, or the filename in config.py does")
    print("  not match what landed on disk.")
else:
    print("\nReady to generate.")

with urllib.request.urlopen("http://127.0.0.1:8000/v1/prompts", timeout=15) as r:
    presets = json.load(r)
print(f"\n{len(presets)} presets available:")
for p in presets:
    print(f"  {p['id']:18} {p['label']}")
''', "6. Health check"),

    code(r'''
#  Run one try-on.
#  In Colab you get upload buttons. On a JupyterLab server, upload your images
#  with the file browser (left sidebar) and put their paths here instead.
#  Change PRESET to any id printed by cell 6, or set PROMPT to free text.

PERSON_PATH   = ""                    #@param {type:"string"}
GARMENT_PATH  = ""                    #@param {type:"string"}
PRESET        = "default_female_zh"   #@param {type:"string"}
PROMPT        = ""                    #@param {type:"string"}
STEPS         = 8                     #@param {type:"slider", min:1, max:32, step:1}
CFG           = 1.0                   #@param {type:"number"}
SEED          = 42                    #@param {type:"integer"}
LORA_STRENGTH = 0.4                   #@param {type:"slider", min:0, max:1.5, step:0.05}
MEGAPIXELS    = 1.0                   #@param {type:"number"}
SWAP_SLOTS    = False                 #@param {type:"boolean"}

import io, os, pathlib, time, httpx
from IPython.display import display
from PIL import Image

if PERSON_PATH.strip() and GARMENT_PATH.strip():
    # Explicit paths win in any environment.
    def _load(p, what):
        fp = pathlib.Path(p.strip()).expanduser()
        if not fp.exists():
            raise SystemExit(f"{what} not found: {fp}")
        return fp.name, fp.read_bytes()
    p_name, p_bytes = _load(PERSON_PATH, "PERSON_PATH")
    g_name, g_bytes = _load(GARMENT_PATH, "GARMENT_PATH")
    print(f"person : {p_name} ({len(p_bytes)//1024} KB)")
    print(f"garment: {g_name} ({len(g_bytes)//1024} KB)")
elif os.environ.get("IN_COLAB"):
    from google.colab import files
    print("Upload the PERSON photo:")
    p_name, p_bytes = next(iter(files.upload().items()))
    print("Upload the GARMENT photo:")
    g_name, g_bytes = next(iter(files.upload().items()))
else:
    raise SystemExit(
        "Set PERSON_PATH and GARMENT_PATH above.\n"
        "On a JupyterLab server, drag your images into the file browser on the\n"
        "left, then right-click a file -> Copy Path and paste it in."
    )

data = {"steps": str(STEPS), "cfg": str(CFG), "seed": str(SEED),
        "lora_strength": str(LORA_STRENGTH), "megapixels": str(MEGAPIXELS),
        "swap_slots": str(SWAP_SLOTS).lower()}
if PROMPT.strip():
    data["prompt"] = PROMPT.strip()
else:
    data["preset"] = PRESET

c = httpx.Client(base_url="http://127.0.0.1:8000", timeout=60.0)
t0 = time.time()
r = c.post("/v1/tryon",
           files={"person": (p_name, p_bytes), "garment": (g_name, g_bytes)},
           data=data)
r.raise_for_status()
job_id = r.json()["job_id"]
print(f"\njob {job_id}")

while True:
    info = c.get(f"/v1/jobs/{job_id}").json()
    if info["status"] == "running" and info.get("total_steps"):
        print(f"\r  step {info['step']}/{info['total_steps']}", end="", flush=True)
    if info["status"] in ("succeeded", "failed"):
        print()
        break
    time.sleep(1.0)

if info["status"] == "failed":
    raise SystemExit(f"FAILED: {info['error']}")

print(f"done in {info['duration_seconds']}s  (seed {info['seed']})")
img = c.get(f"/v1/jobs/{job_id}/image").content
out_path = pathlib.Path(os.environ["TRYON_BASE"]) / f"result_{job_id[:8]}.png"
out_path.write_bytes(img)

display(Image.open(io.BytesIO(p_bytes)).resize((256, 341)))
display(Image.open(io.BytesIO(g_bytes)).resize((256, 341)))
display(Image.open(io.BytesIO(img)))
print(f"\nSaved to {out_path}  ({time.time()-t0:.1f}s wall clock)")
''', "7. Run a try-on"),

    md("""
## Choosing a tunnel

| | cloudflared | ngrok (free) |
|---|---|---|
| Account | none | authtoken required |
| Bandwidth | unmetered | **1 GB/month** |
| Requests | unmetered | 20k/month |
| Session length | until runtime dies | **2 hours** |
| Interstitial page | none | yes, on browser traffic |
| Stable URL | no | paid plans only |

At roughly 4 MB per try-on (two images up, one down), ngrok's free 1 GB cap is
about **250 generations per month**. For testing model quality, cloudflared is
the better default. ngrok earns its place on a paid plan, where you get a stable
domain and no interstitial — but at that point the VM deployment is a better
answer anyway.

If you do use ngrok free, send this header on every request from a browser or
the interstitial HTML will come back instead of your JSON:

```
ngrok-skip-browser-warning: true
```

## Calling it from outside Colab

Use the tunnel URL printed by cell 5:

```bash
curl -X POST "$PUBLIC_URL/v1/tryon?wait=true" \\
  -H "X-API-Key: $KEY" \\
  -H "ngrok-skip-browser-warning: true" \\
  -F "person=@model.jpg" \\
  -F "garment=@saree.jpg" \\
  -F "preset=f4_saree" \\
  -F "steps=12" -F "seed=42"
```

Drop the `X-API-Key` header if you left `API_KEY` empty, and the ngrok header if
you are on cloudflared. Interactive docs live at `<tunnel-url>/docs`.

## Two experiments worth running first

**1. Which slot is the person?** The upstream graph is genuinely ambiguous —
node `157 GetImageSize` implies one assignment, the `ReferenceLatent` order and
the Chinese prompt imply the other. Run the same seed both ways and keep
whichever is right:

```python
for swap in (False, True):
    ...  # cell 7 with SWAP_SLOTS = swap, same SEED
```

**2. Chinese vs English prompt.** The LoRA shipped with
`将图1的女性模特服装换成图2`, which suggests it was trained on Chinese
instruction pairs. Compare `preset="default_female_zh"` against one of the
detailed English presets (`f4_saree`, `m2_bandhgala`, …) at a fixed seed before
committing to either.

## Tuning notes

- **Steps.** Upstream shipped 4 — a turbo setting. 8 is the default here; push
  to 12–16 when dense embroidery matters more than latency.
- **`lora_strength`.** 0.4 is the upstream value. Raise it if the garment is
  being ignored, lower it if the person's identity is drifting.
- **`cfg`.** Stays at 1.0. The graph builds a negative-conditioning branch that
  has *no effect* at cfg 1.0 — raising it makes that branch live, at the cost of
  a second model pass per step.

## Colab caveats

- Free-tier T4s get `--lowvram` automatically and will be noticeably slower.
- The tunnel dies when the runtime disconnects. Re-run cell 5 to get a new URL.
- Weights land in `/content` and are lost on disconnect. Mount Drive and point
  `COMFY_ROOT` at it if re-downloading 19 GB each session gets old.
"""),
]

# Splice the embedded payload into cell 3.
raw = json.dumps(cells)
nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True, "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": json.loads(raw.replace("__FILES__", json.dumps(json.dumps(payload))[1:-1])),
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"wrote {OUT}")
print(f"cells: {len(nb['cells'])}  embedded files: {len(payload)}")
print(f"size : {OUT.stat().st_size:,} bytes")
