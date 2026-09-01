"""Virtual try-on API.

FastAPI front end over a ComfyUI backend running FLUX.2 klein 9B with fal's
virtual-try-on LoRA. Everything is Apache-2.0 / commercially deployable:
klein 9B is Apache-2.0, unlike FLUX.2 [dev].
"""
from __future__ import annotations

import asyncio
import logging
import shlex
import time
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from . import auth as auth_lib
from . import garment_bg
from . import prompts as prompt_lib
from . import workflow
from .batch import BatchStore, GarmentEntry
from .brochure import BrochureRunner
from .library import GarmentLibrary
from .comfy_client import ComfyClient
from .config import settings
from .jobs import JobManager
from .lifecycle import GPUManager, Phase, PhaseConflict
from .schemas import HealthInfo, JobCreated, JobInfo, JobStatus, PromptEntry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)-14s %(message)s",
)
log = logging.getLogger("api")

client = ComfyClient()
jobs = JobManager(client)
gpu = GPUManager(settings, comfy_free=lambda: client.free_models())
brochures = BrochureRunner(settings.garment_cache_dir)
batches = BatchStore()
library = GarmentLibrary(Path(settings.output_dir) / 'library')
# Generated when unset so a dev box works out of the box; tokens are then
# invalidated by a restart, which is the right default for a test platform.
_auth_secret = settings.auth_secret or __import__("secrets").token_hex(32)
_comfy_proc: asyncio.subprocess.Process | None = None


# --------------------------------------------------------------- supervisor


async def _pump_logs(stream: asyncio.StreamReader, level: int) -> None:
    """Forward ComfyUI output into our logger.

    The Colab notebook sent ComfyUI's stdout and stderr to DEVNULL, so a
    backend that failed to boot produced no diagnostics at all. Never do that.
    """
    backend = logging.getLogger("comfyui")
    while True:
        line = await stream.readline()
        if not line:
            return
        backend.log(level, line.decode(errors="replace").rstrip())


async def _spawn_comfy() -> asyncio.subprocess.Process:
    root = Path(settings.comfy_root)
    if not (root / "main.py").exists():
        raise RuntimeError(
            f"ComfyUI not found at {root}. Run scripts/install.sh, or set "
            f"COMFY_MANAGE=false to attach to an external instance."
        )

    args = [
        sys.executable,
        "main.py",
        "--listen",
        settings.comfy_host,
        "--port",
        str(settings.comfy_port),
        "--disable-auto-launch",
        *shlex.split(settings.comfy_extra_args),
    ]
    log.info("starting ComfyUI: %s", " ".join(args))

    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    asyncio.create_task(_pump_logs(proc.stdout, logging.INFO))
    asyncio.create_task(_pump_logs(proc.stderr, logging.WARNING))
    return proc


async def _wait_for_comfy(proc: asyncio.subprocess.Process | None) -> None:
    for _ in range(settings.comfy_boot_timeout):
        if proc is not None and proc.returncode is not None:
            raise RuntimeError(
                f"ComfyUI exited during startup with code {proc.returncode} — "
                f"see the comfyui logger above for the reason"
            )
        if await client.reachable():
            log.info("ComfyUI is up at %s", settings.comfy_base)
            return
        await asyncio.sleep(1)
    raise RuntimeError(f"ComfyUI not reachable after {settings.comfy_boot_timeout}s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _comfy_proc
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    if settings.comfy_manage:
        _comfy_proc = await _spawn_comfy()
    await _wait_for_comfy(_comfy_proc)
    await jobs.start()

    yield

    await jobs.stop()
    await client.aclose()
    if _comfy_proc and _comfy_proc.returncode is None:
        _comfy_proc.terminate()
        try:
            await asyncio.wait_for(_comfy_proc.wait(), timeout=20)
        except asyncio.TimeoutError:
            _comfy_proc.kill()


app = FastAPI(
    title="Virtual Try-On API",
    version="1.0.0",
    description="FLUX.2 klein 9B + fal try-on LoRA, served over HTTP.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------- auth


async def require_key(x_api_key: str | None = Header(None)) -> None:
    keys = settings.key_set
    if not keys:
        return  # auth disabled — development only
    if x_api_key not in keys:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


async def require_user(authorization: str | None = Header(None)) -> str:
    """Bearer token from /v1/auth/token, used by the test platform.

    Separate from require_key on purpose: machine clients hold a long-lived
    API key, reviewers hold a short-lived token issued against a password. One
    being disabled must not disable the other.
    """
    if not settings.auth_users:
        return "anonymous"  # login disabled — development only
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    user = auth_lib.verify(authorization.split(None, 1)[1].strip(), _auth_secret)
    if user is None:
        raise HTTPException(status_code=401, detail="invalid or expired token")
    return user


@app.post("/v1/auth/token", tags=["auth"])
async def issue_token(body: dict) -> dict:
    """Exchange a username and password for a short-lived bearer token."""
    users = auth_lib.parse_users(settings.auth_users)
    if not users:
        raise HTTPException(status_code=503, detail="login is not configured")
    username = str(body.get("username", ""))
    stored = users.get(username)
    # Verify even when the user is unknown, so a wrong username and a wrong
    # password take the same time and cannot be told apart.
    ok = auth_lib.verify_password(str(body.get("password", "")),
                                  stored or "sha256$x$y")
    if not stored or not ok:
        log.warning("failed login for %r", username[:40])
        raise HTTPException(status_code=401, detail="invalid credentials")
    token, exp = auth_lib.issue(username, _auth_secret, settings.auth_token_ttl)
    return {"token": token, "expires_at": exp, "username": username}


def _stash(data: bytes, kind: str, job_id: str) -> str:
    """Write an upload where the guardrail can read it back."""
    d = Path(settings.output_dir) / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{job_id}_{kind}.png"
    path.write_bytes(data)
    return str(path)


# -------------------------------------------------------------------- phase


@app.get("/v1/phase", tags=["phase"])
async def get_phase() -> dict:
    """Which workload is holding the GPU, and how much VRAM is in use."""
    return gpu.state()


@app.post("/v1/phase", dependencies=[Depends(require_key)], tags=["phase"])
async def set_phase(body: dict) -> dict:
    """Switch phase. Unloads before it loads; one transition at a time."""
    try:
        target = Phase(str(body.get("phase", "")).lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"phase must be one of {[p.value for p in Phase]}")
    return await gpu.switch(target)


# ----------------------------------------------------------------- brochure


@app.post("/v1/brochure", dependencies=[Depends(require_key)], tags=["brochure"])
async def create_brochure(body: dict) -> dict:
    """Analyse a set of garment photos with the brochure model.

    Cached by image + question + model, so re-running is nearly free and only
    genuinely new garments cost anything.
    """
    try:
        gpu.require(Phase.brochure)
    except PhaseConflict as e:
        raise HTTPException(status_code=409, detail=str(e))

    directory = Path(body.get("directory", settings.garment_dir)).expanduser()
    if not directory.is_dir():
        raise HTTPException(status_code=400, detail=f"no such directory: {directory}")
    names = body.get("garments") or []
    paths = ([directory / n for n in names] if names
             else sorted(p for p in directory.iterdir()
                         if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                         # macOS tar ships AppleDouble sidecars next to every
                         # file; they carry image extensions but no image.
                         and not p.name.startswith("._")))
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise HTTPException(status_code=400, detail=f"not found: {missing[:5]}")
    if not paths:
        raise HTTPException(status_code=400, detail=f"no images in {directory}")

    job = await brochures.run(gpu.vision, settings.brochure_model, paths,
                              force=bool(body.get("force")))
    return job.info()


@app.post("/v1/brochure/upload", dependencies=[Depends(require_key)],
          tags=["brochure"])
async def brochure_upload(files: list[UploadFile] = File(...)) -> dict:
    """Read a batch of uploaded garment photos and return a prompt for each.

    This is the first half of the two-phase flow: every garment is described by
    the large vision model while it has the card to itself, and the prompts are
    kept. Generation then runs afterwards with the small guardrail resident
    instead. The two models are never loaded at the same time, which is what
    makes both fit in 24 GB.

    Descriptions are cached by image content, so re-uploading the same garment
    costs nothing and only genuinely new ones are analysed.
    """
    try:
        gpu.require(Phase.brochure)
    except PhaseConflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    if not files:
        raise HTTPException(status_code=400, detail="no files")

    from .brochure import spec_to_prompt, to_spec

    staging = Path(settings.output_dir) / "brochure_uploads"
    staging.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for f in files:
        raw = await _read_upload(f, "garment")
        name = (f.filename or "garment").replace("/", "_")
        dst = staging / name
        dst.write_bytes(raw)
        paths.append(dst)

    job = await brochures.run(gpu.vision, settings.brochure_model, paths)

    out = []
    for p in paths:
        fields = job.specs.get(p.stem)
        if not fields:
            out.append({"garment": p.stem, "ok": False,
                        "error": "analysis failed"})
            continue
        spec = to_spec(fields)
        out.append({
            "garment": p.stem, "ok": True,
            "type": fields.get("TYPE"),
            "pieces": spec["pieces"],
            "piece_details": spec.get("piece_details", []),
            "colours": fields.get("COLOURS"),
            "dupatta": fields.get("DUPATTA"),
            "prompt": spec_to_prompt(spec),
        })
    return {"job_id": job.job_id, "analysed": job.analysed,
            "cached": job.cached, "seconds": job.seconds, "garments": out}


async def _run_batch(b, *, steps: int, seed: int, megapixels: float,
                     lora_strength: float, use_guardrail: bool,
                     t0: float) -> dict:
    """Generate every described garment in a batch onto its person.

    Shared by the two entry points - upload-and-go, and pick-from-library - so
    the guardrail wiring and the failure handling cannot drift apart between
    them. One garment failing never abandons the rest.
    """
    b.stage = "generating"
    person_bytes = b.person_path.read_bytes()
    queue = [e for e in b.garments.values() if e.prompt]
    for position, entry in enumerate(queue, 1):
        b.current = entry.name
        b.position = position
        b.queued = len(queue)
        entry.status = "running"
        t1 = time.time()
        try:
            p_name = await client.upload_image(person_bytes,
                                               f"person_{b.batch_id}.png")
            g_name = await client.upload_image(entry.path.read_bytes(),
                                               f"garment_{entry.name}.png")
            graph, resolved = workflow.build(
                person_filename=p_name, garment_filename=g_name,
                prompt=entry.prompt, steps=steps, cfg=settings.default_cfg,
                seed=seed, lora_strength=lora_strength, megapixels=megapixels)
            job = jobs.submit(graph, resolved)
            job.guardrail = use_guardrail
            if use_guardrail:
                job.person_path = str(b.person_path)
                job.garment_path = str(entry.path)
                job.inspector = gpu.guardrail
            finished = await jobs.wait(job.job_id, timeout=1800.0)
            if finished is None or finished.status is not JobStatus.succeeded:
                raise RuntimeError(getattr(finished, "error", None) or "failed")
            entry.image = finished.image
            # Write it out as well as keeping it in memory. Holding results only
            # in the process meant a paused box lost a completed run - 23 images
            # generated, none recoverable.
            try:
                out_dir = Path(settings.output_dir) / "batches" / b.batch_id
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"{entry.name}.png").write_bytes(finished.image)
            except OSError as exc:
                log.warning("could not persist %s: %s", entry.name, exc)
            entry.guardrail_ok = finished.guardrail_ok
            entry.guardrail_reason = finished.guardrail_reason
            entry.attempts = finished.attempts
            entry.status = "done"
        except Exception as exc:  # noqa: BLE001
            entry.status, entry.error = "failed", f"{type(exc).__name__}: {exc}"
            log.warning("batch %s garment %s failed: %s",
                        b.batch_id, entry.name, exc)
        finally:
            entry.seconds = round(time.time() - t1, 2)

    b.stage = "complete"
    b.current = None
    b.generate_seconds = round(time.time() - t0, 1)
    done = [e for e in b.garments.values() if e.status == "done"]
    secs = [e.seconds for e in done if e.seconds]
    log.info("batch %s generated %d/%d in %.1fs (mean %.1fs)",
             b.batch_id, len(done), len(b.garments), b.generate_seconds,
             sum(secs) / len(secs) if secs else 0.0)
    return b.info()


# ------------------------------------------------------------------- library
# Garments are uploaded and described once, with the big model resident.
# Inference later picks from this list and supplies only the person.


@app.post("/v1/garments", dependencies=[Depends(require_key)], tags=["library"])
async def add_garments(files: list[UploadFile] = File(...)) -> dict:
    """Upload garments and describe them. Switches phase by itself."""
    from .brochure import spec_to_prompt, to_spec

    if not files:
        raise HTTPException(status_code=400, detail="no files")

    added = []
    for f in files:
        raw = await _read_upload(f, "garment")
        stem = Path(f.filename or "garment").stem
        suffix = Path(f.filename or ".jpg").suffix or ".jpg"
        added.append(library.add(stem, raw, suffix))

    t0 = time.time()
    await gpu.switch(Phase.brochure)
    job = await brochures.run(gpu.vision, settings.brochure_model,
                              [i.image for i in added])
    described = 0
    for item in added:
        fields = job.specs.get(item.image.stem)
        if not fields:
            continue
        spec = to_spec(fields)
        library.describe(item, fields, spec["pieces"],
                         spec.get("piece_details", []))
        described += 1

    seconds = round(time.time() - t0, 1)
    log.info("library: %d uploaded, %d described in %.1fs (%d cached)",
             len(added), described, seconds, job.cached)
    return {"added": len(added), "described": described,
            "cached": job.cached, "seconds": seconds,
            "garments": [i.summary() for i in added]}


@app.get("/v1/garments", tags=["library"])
async def list_garments() -> dict:
    return {"count": len(library.items), "garments": library.list()}


@app.get("/v1/garments/{name}/image", tags=["library"], response_class=Response)
async def garment_image(name: str) -> Response:
    item = library.get(name)
    if item is None or not item.image.exists():
        raise HTTPException(status_code=404, detail="unknown garment")
    return Response(item.image.read_bytes(), media_type="image/jpeg")


@app.delete("/v1/garments/{name}", dependencies=[Depends(require_key)],
            tags=["library"])
async def delete_garment(name: str) -> dict:
    if not library.remove(name):
        raise HTTPException(status_code=404, detail="unknown garment")
    return {"removed": name, "remaining": len(library.items)}


@app.post("/v1/generate", dependencies=[Depends(require_key)], tags=["library"])
async def generate_from_library(
    person: UploadFile = File(...),
    garments: str = Form(..., description="comma-separated names from /v1/garments"),
    steps: int = Form(settings.default_steps),
    seed: int = Form(42),
    megapixels: float = Form(settings.default_megapixels),
    lora_strength: float = Form(settings.default_lora_strength),
    guardrail: bool | None = Form(None),
) -> dict:
    """Generate the chosen library garments onto an uploaded person.

    The person is read by whichever vision model is resident, so the preserve
    block describes this photograph - its actual setting, pose and accessories
    - rather than asserting someone else's. Nothing about the subject is
    assumed.
    """
    from .brochure import person_preserve, read_person, spec_to_prompt
    from pipeline.garments import PRESERVE_GENERIC

    names = [n.strip() for n in garments.split(",") if n.strip()]
    chosen = [library.get(n) for n in names]
    missing = [n for n, c in zip(names, chosen) if c is None]
    if missing:
        raise HTTPException(status_code=400, detail=f"unknown garments: {missing}")
    undescribed = [c.name for c in chosen if not c.pieces]
    if undescribed:
        raise HTTPException(
            status_code=409,
            detail=f"not described yet: {undescribed}. Upload them again to "
                   f"describe, or wait for the current description to finish.")

    b = batches.create()
    staging = Path(settings.output_dir) / "batches" / b.batch_id
    staging.mkdir(parents=True, exist_ok=True)
    person_bytes = await _read_upload(person, "person")
    b.person_path = staging / f"person{Path(person.filename or 'p.jpg').suffix or '.jpg'}"
    b.person_path.write_bytes(person_bytes)

    t0 = time.time()
    await gpu.switch(Phase.generate)
    use_guardrail = settings.guardrail_enabled if guardrail is None else guardrail
    if use_guardrail:
        if not await gpu.ensure_guardrail():
            raise HTTPException(
                status_code=409,
                detail="guardrail requested but GUARDRAIL_ENABLED is false")
    else:
        # Give the memory back. With the guardrail off it is not just unused,
        # it is not loaded - which is worth ~4-7 s per image.
        await gpu.release_guardrail()

    # Read the person with whatever vision model this phase has. The guardrail
    # model is smaller than the one that read the garments, which is the right
    # way round: garment construction is the harder reading, and it is done
    # once, while the person is read once per session either way.
    reader = gpu.guardrail
    if reader is not None:
        try:
            b.person_fields = await asyncio.to_thread(read_person, reader,
                                                      b.person_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("person reading failed (%s); using generic preserve", exc)
    preserve = person_preserve(b.person_fields) if b.person_fields else PRESERVE_GENERIC

    for item in chosen:
        entry = GarmentEntry(name=item.name, path=item.image)
        entry.fields = item.fields
        entry.pieces = item.pieces
        entry.piece_details = item.piece_details
        entry.prompt = spec_to_prompt({
            "summary": f"a {(item.fields.get('TYPE') or 'garment').lower()}"
                       + (f" in {item.fields['COLOURS']}"
                          if item.fields.get("COLOURS") else ""),
            "pieces": item.pieces, "piece_details": item.piece_details,
            "detail": item.fields.get("HOW_WORN") or "",
            "drape": (f"this garment includes a dupatta, worn like this: "
                      f"{item.fields['DUPATTA']}")
                     if (item.fields.get("DUPATTA") or "").lower() not in
                        ("", "none", "n/a", "-") else "",
            "colours": item.fields.get("COLOURS") or "as shown in the second image",
            "preserve": preserve,
        })
        b.garments[item.name] = entry

    b.stage = "queued"
    # Return at once and generate in the background. A batch of twenty-three is
    # minutes long; holding the request open gives the caller no progress and
    # no way to tell a slow run from a stalled one.
    asyncio.create_task(_run_batch(
        b, steps=steps, seed=seed, megapixels=megapixels,
        lora_strength=lora_strength, use_guardrail=use_guardrail, t0=t0))
    return b.info()


# --------------------------------------------------------------------- batch
# Two operations, each owning its phase switch, because the vision model and
# the generator cannot share 24 GB. Callers do not have to know that: uploading
# garments loads the reader, pressing generate loads the generator.


@app.post("/v1/batch", dependencies=[Depends(require_key)], tags=["batch"])
async def batch_create(
    person: UploadFile = File(...),
    garments: list[UploadFile] = File(...),
) -> dict:
    """Step one. Upload the person and every garment, and get a prompt for each.

    Switches to the brochure phase by itself, so the large vision model has the
    card to itself while it reads. Readings are cached by image content, so
    re-uploading a garment analysed before costs nothing.
    """
    from .brochure import person_preserve, read_person, spec_to_prompt, to_spec

    if not garments:
        raise HTTPException(status_code=400, detail="no garments")

    b = batches.create()
    staging = Path(settings.output_dir) / "batches" / b.batch_id
    staging.mkdir(parents=True, exist_ok=True)

    person_bytes = await _read_upload(person, "person")
    b.person_path = staging / f"person{Path(person.filename or 'p.jpg').suffix or '.jpg'}"
    b.person_path.write_bytes(person_bytes)

    paths: list[Path] = []
    for f in garments:
        raw = await _read_upload(f, "garment")
        name = Path(f.filename or "garment").stem.replace("/", "_") or "garment"
        dst = staging / f"{name}{Path(f.filename or '.jpg').suffix or '.jpg'}"
        dst.write_bytes(raw)
        paths.append(dst)
        b.garments[name] = GarmentEntry(name=name, path=dst)

    b.stage = "describing"
    t0 = time.time()
    try:
        await gpu.switch(Phase.brochure)
        # Read the person once, while the large model is loaded anyway. This is
        # what lets the prompt name their actual surroundings instead of
        # asserting a scene that is not there - the thing that made the
        # catalogue's backgrounds survive, generalised to an unseen subject.
        person_fields = await asyncio.to_thread(
            read_person, gpu.vision, b.person_path)
        preserve = person_preserve(person_fields)
        b.person_fields = person_fields

        job = await brochures.run(gpu.vision, settings.brochure_model, paths)
        for p_ in paths:
            entry = b.garments[p_.stem]
            fields = job.specs.get(p_.stem)
            if not fields:
                entry.status, entry.error = "failed", "analysis failed"
                continue
            spec = to_spec(fields)
            spec["preserve"] = preserve
            entry.fields = fields
            entry.pieces = spec["pieces"]
            entry.piece_details = spec.get("piece_details", [])
            entry.prompt = spec_to_prompt(spec)
        b.stage = "described"
    except Exception as exc:  # noqa: BLE001
        b.stage, b.error = "new", f"{type(exc).__name__}: {exc}"
        log.exception("batch %s describe failed", b.batch_id)
        raise HTTPException(status_code=500, detail=b.error)
    finally:
        b.describe_seconds = round(time.time() - t0, 1)

    log.info("batch %s described %d garments in %.1fs (%d cached)",
             b.batch_id, len(paths), b.describe_seconds or 0, job.cached)
    return b.info()


@app.post("/v1/batch/{batch_id}/generate", dependencies=[Depends(require_key)],
          tags=["batch"])
async def batch_generate(
    batch_id: str,
    steps: int = Form(settings.default_steps),
    seed: int = Form(42),
    megapixels: float = Form(settings.default_megapixels),
    lora_strength: float = Form(settings.default_lora_strength),
    guardrail: bool | None = Form(None),
) -> dict:
    """Step two. Generate every described garment onto the person.

    Switches to the generate phase by itself, which unloads the vision model
    that wrote the prompts and loads the generator and guardrail in its place.
    """
    b = batches.get(batch_id)
    if b is None:
        raise HTTPException(status_code=404, detail="unknown batch")
    if not any(g.prompt for g in b.garments.values()):
        raise HTTPException(status_code=409,
                            detail="nothing described yet; upload garments first")
    if b.stage == "generating":
        raise HTTPException(status_code=409, detail="already generating")

    t0 = time.time()
    await gpu.switch(Phase.generate)
    use_guardrail = settings.guardrail_enabled if guardrail is None else guardrail
    if use_guardrail:
        if not await gpu.ensure_guardrail():
            b.stage = "described"
            raise HTTPException(
                status_code=409,
                detail="guardrail requested but GUARDRAIL_ENABLED is false")
    else:
        await gpu.release_guardrail()
    return await _run_batch(b, steps=steps, seed=seed, megapixels=megapixels,
                            lora_strength=lora_strength,
                            use_guardrail=use_guardrail, t0=t0)


@app.get("/v1/batch/{batch_id}", tags=["batch"])
async def batch_status(batch_id: str) -> dict:
    b = batches.get(batch_id)
    if b is None:
        raise HTTPException(status_code=404, detail="unknown batch")
    return b.info()


@app.get("/v1/batch/{batch_id}/image/{garment}", tags=["batch"],
         response_class=Response)
async def batch_image(batch_id: str, garment: str) -> Response:
    b = batches.get(batch_id)
    if b is None:
        raise HTTPException(status_code=404, detail="unknown batch")
    entry = b.garments.get(garment)
    if entry is None or entry.image is None:
        raise HTTPException(status_code=404, detail="no image for that garment")
    return Response(entry.image, media_type="image/png")


@app.get("/v1/brochure/{job_id}", tags=["brochure"])
async def brochure_status(job_id: str) -> dict:
    job = brochures.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown brochure job")
    return job.info()


@app.get("/v1/brochure/{job_id}/specs", tags=["brochure"])
async def brochure_specs(job_id: str) -> dict:
    job = brochures.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown brochure job")
    return {"job_id": job_id, "specs": job.specs}


# ------------------------------------------------------------------- health


@app.get("/healthz", response_model=HealthInfo, tags=["ops"])
async def healthz() -> HealthInfo:
    """Liveness plus a real check that the models ComfyUI needs are on disk."""
    reachable = await client.reachable()
    present: dict[str, bool] = {}
    detail: str | None = None

    if reachable:
        try:
            info = await client.object_info()

            def options(node: str, field: str) -> list:
                try:
                    return info[node]["input"]["required"][field][0]
                except (KeyError, IndexError, TypeError):
                    return []

            present = {
                settings.unet_name: settings.unet_name in options("UnetLoaderGGUF", "unet_name"),
                settings.clip_name: settings.clip_name in options("CLIPLoader", "clip_name"),
                settings.vae_name: settings.vae_name in options("VAELoader", "vae_name"),
                settings.lora_name: settings.lora_name
                in options("LoraLoaderModelOnly", "lora_name"),
            }
        except Exception as exc:  # noqa: BLE001
            detail = f"could not read /object_info: {exc}"
    else:
        detail = f"ComfyUI unreachable at {settings.comfy_base}"

    ok = reachable and present and all(present.values())
    return HealthInfo(
        status="ok" if ok else "degraded",
        comfy_reachable=reachable,
        models_present=present,
        queue_depth=jobs.depth() + (await client.queue_depth() if reachable else 0),
        detail=detail,
    )


@app.get("/readyz", tags=["ops"])
async def readyz() -> Response:
    h = await healthz()
    return JSONResponse(
        h.model_dump(), status_code=200 if h.status == "ok" else 503
    )


# ------------------------------------------------------------------ prompts


@app.get("/v1/prompts", response_model=list[PromptEntry], tags=["prompts"])
async def list_prompts() -> list[PromptEntry]:
    """Preset prompts, including one per preset outfit."""
    return [PromptEntry(**p) for p in prompt_lib.all_presets()]


# ------------------------------------------------------------------- try-on


@app.post(
    "/v1/tryon",
    response_model=JobInfo | JobCreated,
    dependencies=[Depends(require_key)],
    tags=["tryon"],
)
async def create_tryon(
    person: UploadFile = File(..., description="Photo of the person"),
    garment: UploadFile = File(..., description="Photo of the garment"),
    prompt: str | None = Form(None),
    preset: str | None = Form(None, description="Preset id from GET /v1/prompts"),
    steps: int = Form(settings.default_steps, ge=1, le=50),
    cfg: float = Form(settings.default_cfg, ge=1.0, le=10.0),
    seed: int | None = Form(None),
    lora_strength: float = Form(settings.default_lora_strength, ge=0.0, le=2.0),
    megapixels: float = Form(settings.default_megapixels, ge=0.25, le=4.0),
    swap_slots: bool = Form(False),
    remove_background: bool = Form(
        False,
        description="Optional. Strip the room out of the garment photo before "
        "generating. Off by default: the garment is sent exactly as supplied. "
        "Worth knowing when a result takes on the garment photo's room - in a "
        "controlled test that happened with the room present and stopped when "
        "it was removed, with the prompt unchanged.",
    ),
    guardrail: bool | None = Form(
        None,
        description="Inspect the result and reseed on a critical failure. "
        "Defaults to GUARDRAIL_ENABLED. Requires the generate phase.",
    ),
    wait: bool = Query(False, description="Block until the job finishes"),
    wait_timeout: float = Query(300.0, ge=1.0, le=900.0),
):
    person_bytes = await _read_upload(person, "person")
    garment_bytes = await _read_upload(garment, "garment")

    try:
        text = prompt_lib.resolve(prompt, preset)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"unknown preset '{preset}'")

    bg_info: dict = {"applied": False, "reason": "not requested"}
    if remove_background:
        garment_bytes, bg_info = await asyncio.to_thread(
            garment_bg.strip, garment_bytes)

    p_name = await client.upload_image(person_bytes, f"person_{person.filename or 'p.png'}")
    g_name = await client.upload_image(garment_bytes, f"garment_{garment.filename or 'g.png'}")

    graph, resolved_seed = workflow.build(
        person_filename=p_name,
        garment_filename=g_name,
        prompt=text,
        steps=steps,
        cfg=cfg,
        seed=seed,
        lora_strength=lora_strength,
        megapixels=megapixels,
        swap_slots=swap_slots,
    )

    use_guardrail = settings.guardrail_enabled if guardrail is None else guardrail
    if use_guardrail:
        # Load it if it is not up. Never quietly proceed without it: a silent
        # downgrade is exactly how a broken guardrail once reported 23/23 while
        # reading zero checks.
        if not await gpu.ensure_guardrail():
            raise HTTPException(
                status_code=409,
                detail="guardrail requested but GUARDRAIL_ENABLED is false; "
                       "pass guardrail=false to generate without it",
            )
    else:
        await gpu.release_guardrail()

    job = jobs.submit(graph, resolved_seed)
    job.guardrail = use_guardrail
    if use_guardrail:
        # The inspector compares the result against the two inputs, so it needs
        # them on disk for the life of the job.
        job.person_path = _stash(person_bytes, "person", job.job_id)
        job.garment_path = _stash(garment_bytes, "garment", job.job_id)
        job.inspector = gpu.guardrail
    log.info("queued job %s (seed=%d steps=%d guardrail=%s)",
             job.job_id, resolved_seed, steps, use_guardrail)

    if wait:
        finished = await jobs.wait(job.job_id, timeout=wait_timeout)
        return _to_info(finished or job)

    return JobCreated(
        job_id=job.job_id,
        status=job.status,
        poll_url=f"/v1/jobs/{job.job_id}",
    )


@app.get(
    "/v1/jobs/{job_id}",
    response_model=JobInfo,
    dependencies=[Depends(require_key)],
    tags=["tryon"],
)
async def get_job(job_id: str) -> JobInfo:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown or expired job")
    return _to_info(job)


@app.get(
    "/v1/jobs/{job_id}/image",
    dependencies=[Depends(require_key)],
    tags=["tryon"],
    response_class=Response,
)
async def get_job_image(job_id: str) -> Response:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown or expired job")
    if job.status is JobStatus.failed:
        raise HTTPException(status_code=422, detail=job.error or "job failed")
    if job.image is None:
        raise HTTPException(status_code=409, detail=f"job is {job.status.value}")
    return Response(
        content=job.image,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


# ------------------------------------------------------------------ helpers


async def _read_upload(upload: UploadFile, field: str) -> bytes:
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"'{field}' is empty")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"'{field}' exceeds {settings.max_upload_bytes // (1024 * 1024)} MB",
        )
    return data


def _to_info(job) -> JobInfo:
    return JobInfo(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        step=job.step,
        total_steps=job.total_steps,
        seed=job.seed,
        duration_seconds=round(job.duration, 2) if job.duration else None,
        image_url=f"/v1/jobs/{job.job_id}/image"
        if job.status is JobStatus.succeeded
        else None,
        error=job.error,
        guardrail=getattr(job, "guardrail", False),
        guardrail_ok=getattr(job, "guardrail_ok", None),
        guardrail_reason=getattr(job, "guardrail_reason", None),
        guardrail_seconds=getattr(job, "guardrail_seconds", None),
        attempts=getattr(job, "attempts", 1),
    )
