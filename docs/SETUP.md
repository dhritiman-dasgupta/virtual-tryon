# Setup

Two supported paths: **Docker** (recommended — the CUDA toolchain is the fiddly
part and the image pins it) and **bare metal**. Both end with the API on
`:8000` and interactive docs at `/docs`.

---

## Before you start

| | |
|---|---|
| GPU | NVIDIA, **12 GB VRAM or more**. Weights total ~19 GB on disk but ComfyUI streams modules, so peak VRAM is ~10–12 GB. |
| Disk | **~40 GB** free — ~19 GB generator weights, ~16 GB vision model, plus the image. |
| Driver | CUDA 12.8-capable. Blackwell (RTX 5090, `sm_120`) *requires* it. |
| OS | Linux. macOS has no CUDA; use a remote box. |

Check the GPU is visible before anything else:

```bash
nvidia-smi
```

---

## Path A — Docker

Needs Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
so `--gpus` works.

```bash
git clone https://github.com/dhritiman-dasgupta/virtual-tryon.git
cd virtual-tryon

cp .env.example .env
# Open .env and set API_KEYS to a value of your own.
# An EMPTY API_KEYS disables auth entirely — never ship that.

make docker-build      # builds the CUDA 12.8 image
make docker-models     # one-time: populates the weights volume (~35 GB, slow)
make docker-up         # starts the service
make docker-logs       # follow it
```

The weights live in named volumes (`comfy-models`, `hf-cache`), **not** in the
image — rebuilding the app does not refetch 35 GB.

Wait for readiness before sending work. The first request after boot is slow
because ComfyUI loads the weights lazily; the healthcheck allows a 300 s start
period.

```bash
curl -fsS http://localhost:8000/readyz && echo READY
```

`/readyz` returns **503** until ComfyUI is reachable *and* all four model files
are actually loadable — it queries ComfyUI's `/object_info` rather than just
checking that the port is open. Use it for your load balancer, not `/healthz`.

### Compose directly

```bash
docker compose up -d          # pulls ghcr.io/dhritiman-dasgupta/virtual-tryon
docker compose logs -f tryon
```

Uncomment `build: .` in `docker-compose.yml` to build locally instead of
pulling. Note `shm_size: "8gb"` — diffusion needs far more than Docker's 64 MB
default, and the failure mode without it is an obscure crash.

---

## Path B — Bare metal

```bash
git clone https://github.com/dhritiman-dasgupta/virtual-tryon.git
cd virtual-tryon

make env               # writes .env from the example — then set API_KEYS
make install           # ComfyUI + ComfyUI-GGUF + python deps   (~5 min)
make models            # download and verify ~19 GB of weights  (~15 min)
make serve             # http://0.0.0.0:8000
```

`make install` and `make models` both fail loudly on error. That is deliberate:
the notebook this replaced ran with `capture_output=True` and no return-code
check, so a 404 on a weight download looked exactly like success.

Run with autoreload while editing:

```bash
make dev
```

---

## Verify it works

```bash
make smoke             # end-to-end: posts a person + garment, waits, saves the result
```

Or by hand:

```bash
KEY=$(grep '^API_KEYS=' .env | cut -d= -f2 | cut -d, -f1)

curl -X POST "http://localhost:8000/v1/tryon?wait=true" \
  -H "X-API-Key: $KEY" \
  -F "person=@person.jpg" \
  -F "garment=@garment.jpg" \
  -F "steps=8" -F "seed=42" \
  -o result.json

jq -r .job_id result.json
curl -H "X-API-Key: $KEY" \
  "http://localhost:8000/v1/jobs/$(jq -r .job_id result.json)/image" -o result.png
```

There is also a minimal browser client at `web/index.html` — open it and point
it at your host.

---

## Configuration

Everything is environment-driven; see `.env.example` for the annotated list and
`app/config.py` for the defaults. The ones that matter most:

| Variable | Default | Why you would change it |
|---|---|---|
| `API_KEYS` | *(empty)* | Comma-separated. **Empty disables auth.** Always set it. |
| `CORS_ORIGINS` | `*` | Lock to your frontend's origin in production. |
| `WORKERS` | `1` | ComfyUI serialises on the GPU. Raise only on a 48 GB card where two model copies fit. |
| `COMFY_EXTRA_ARGS` | *(empty)* | Set `--lowvram` on cards under ~16 GB. |
| `COMFY_MANAGE` | `true` | `true` = this API spawns and supervises ComfyUI. `false` = attach to one you started. |
| `JOB_TTL_SECONDS` | `3600` | Jobs are in-memory and expire. Results are not persisted. |

---

## Choosing hardware

| GPU | Works | Notes |
|---|---|---|
| T4 16 GB | yes | needs `COMFY_EXTRA_ARGS=--lowvram`; slow |
| **L4 24 GB** | **yes** | best value; no `--lowvram` needed |
| L40S 48 GB | yes | fastest per image; you can raise `WORKERS` |
| RTX 4090 / 5090 | yes | what the 172-generation benchmark ran on (4090, 9.4 s mean) |
| A100 / H100 | yes | overkill for a 9B model |

---

## Troubleshooting

**`no CUDA device visible — did you pass --gpus all?`**
The entrypoint checks this deliberately, because the image can be correct while
the host driver or the `--gpus` flag is not. Confirm the NVIDIA Container
Toolkit is installed and `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi` works.

**`/readyz` never turns 200.**
Weights are missing or partly downloaded. Re-run `make docker-models` (or
`make models`); `scripts/download_models.sh` verifies sizes and will tell you
which file is wrong.

**First request takes minutes.**
Expected — ComfyUI lazily loads ~19 GB. Subsequent requests are seconds. Warm it
with one throwaway request after boot.

**Results look right but the garment is paraphrased.**
Raise `steps` (8 → 12–16) and use a detailed per-garment prompt rather than a
generic one. Dense embroidery and metallic thread are exactly what few steps and
vague prompts lose. See *Prompting* in the main README.

**The person and garment appear swapped.**
A genuine ambiguity in the upstream graph, documented under *Known ambiguity* in
the main README. Send `swap_slots=true` with the same seed, compare, and pin the
default in `.env`.

**Out of memory.**
Lower `megapixels`, set `COMFY_EXTRA_ARGS=--lowvram`, and keep `WORKERS=1`.
