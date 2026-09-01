"""Vision backends for the QA guardrail.

Two interchangeable backends, because the two machines have different budgets:

    LocalVision   Qwen2.5-VL-7B on the box. Used on the pod, where the card has
                  room once the generator is unloaded. No per-image cost, no
                  network, and native multi-image input.
    NIMVision     NVIDIA NIM over HTTP. Used on Colab, where a T4 cannot hold
                  the generator and a 7B VLM at once and the 16 GB download is
                  lost on every reset. Needs no GPU at all.

Most NIM vision models take a single image per request, so that backend
composites the three inputs into one labelled triptych. The local model gets
them as separate images, which preserves resolution — worth having, since face
and embroidery detail are what the fidelity checks turn on.

`inspect()` holds the QA logic and takes whichever backend, so the two paths
cannot drift apart.
"""
from __future__ import annotations

import base64
import gc
import io
import os
import re
import time

from .guardrail import (ANATOMY_ASK, ANATOMY_FIELDS, FIDELITY_ASK,
                        FIDELITY_FIELDS, numeric_gate, verdict)

# Qwen3-VL-8B replaces the Qwen2.5-VL-7B used on the pod. The two mistakes
# that pushed the upgrade were both perception, not reasoning: a false
# background rejection and a pink gown read as a saree.
LOCAL_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
# nemotron-nano-12b answers in a few seconds and counts people correctly, but
# its hosted engine goes down for long stretches — it returned
# "EngineCore encountered an issue" 500s on 13 of 23 images in one batch, for
# both single images and triptychs, so neither payload size nor pacing was the
# cause. llama-3.2-90b read-times-out at 90s and llama-3.2-11b was also 500ing;
# the 8b nemotron stayed up throughout. Fall through the list rather than fail
# the batch: a QA gate that dies when one hosted model is degraded is no gate.
NIM_MODEL = "nvidia/nemotron-nano-12b-v2-vl"
NIM_FALLBACKS = ("nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
                 "meta/llama-3.2-11b-vision-instruct")
NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def parse(text: str, fields: list[str]) -> dict:
    """Pull `LABEL: value` pairs out of the model's reply.

    Splits on label boundaries wherever they occur, not only at line starts:
    models run two answers together on one line, and a line-anchored parser
    folds the second into the first and silently loses a check.

    Falls back to positional mapping when the labels are missing altogether.
    Asked for twelve labelled lines, Qwen3-VL answered

        PASS: 1 person in IMAGE 3, same as IMAGE 1.
        ...
        FAIL: Garment type mismatch — IMAGE 2 is LEHENGA, IMAGE 3 is ANARKALI.

    using the verdict as the label. Every field came back empty, nothing looked
    failed, and an image with a real defect was certified clean. The answers
    were in order and complete — only the names were missing — so map them by
    position rather than discarding a correct inspection.
    """
    if not text:
        return {}
    label = re.compile(r"\b(" + "|".join(map(re.escape, fields)) + r")\s*\**\s*:",
                       re.I)
    parts = label.split(text)
    out: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        key = parts[i].upper()
        val = re.sub(r"\s+", " ", parts[i + 1]).strip(" *\n\t-")
        out.setdefault(key, val)
    if out:
        return out

    # No labels at all — try positional. Only accept it when the number of
    # verdict lines matches the number of fields exactly, so a partial or
    # rambling answer still fails the coverage check rather than being
    # mapped onto the wrong questions.
    verdicts = [ln.strip() for ln in (text or "").splitlines()
                if re.match(r"^\s*\**\s*(PASS|FAIL)\b", ln, re.I)]
    if len(verdicts) == len(fields):
        return {f: re.sub(r"\s+", " ", v).strip(" *")
                for f, v in zip(fields, verdicts)}
    return {}


def triptych(paths: list[str], label: tuple[str, ...] = ("IMAGE 1", "IMAGE 2", "IMAGE 3"),
             panel: int = 512) -> bytes:
    """Compose images side by side with captions, as PNG bytes.

    For single-image backends. The captions matter: the prompt refers to
    "IMAGE 1", "IMAGE 2" and "IMAGE 3", and without them on the pixels the
    model has no way to tell which panel is which.
    """
    from PIL import Image, ImageDraw

    tiles = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        im.thumbnail((panel, panel), Image.LANCZOS)
        tiles.append(im)
    band = 28
    w = panel * len(tiles)
    h = max(t.height for t in tiles) + band
    sheet = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, t in enumerate(tiles):
        x = i * panel + (panel - t.width) // 2
        sheet.paste(t, (x, band))
        d.rectangle([i * panel, 0, (i + 1) * panel - 1, band - 1], fill=(20, 20, 20))
        d.text((i * panel + 8, 7), label[i] if i < len(label) else f"IMAGE {i+1}",
               fill=(255, 255, 255))
    buf = io.BytesIO()
    sheet.save(buf, "PNG")
    return buf.getvalue()


def _fit_inline(paths: list[str], limit: int = 170_000) -> tuple[bytes, str]:
    """Encode one or more images small enough to inline in a NIM request.

    NIM rejects an inline base64 image above roughly 180 KB — anything larger
    has to be uploaded through the NVCF assets API, which is not worth the round
    trip for a QA check. A PNG triptych of three portrait photos lands near
    700 KB, so step JPEG quality down until the encoded payload fits. Quality 35
    still resolves a second person and a changed background; it is the fine
    embroidery comparison that suffers first.
    """
    from PIL import Image

    if len(paths) == 1:
        img = Image.open(paths[0]).convert("RGB")
        img.thumbnail((1024, 1024), Image.LANCZOS)
        sheet = img
    else:
        sheet = Image.open(io.BytesIO(triptych(paths, panel=460)))

    for quality in (85, 75, 65, 55, 45, 35):
        buf = io.BytesIO()
        sheet.save(buf, "JPEG", quality=quality, optimize=True)
        blob = buf.getvalue()
        if len(base64.b64encode(blob)) < limit:
            return blob, "image/jpeg"
    return blob, "image/jpeg"


class LocalVision:
    """A vision model resident on the same card as the generator.

    Stays loaded for the life of the process. That is the point: with both
    models resident an image can be inspected the moment it is generated and
    retried immediately, instead of the batch-generate / swap / batch-inspect
    dance a 24 GB card forces.

    Quantised by default. The generator needs ~18.2 GB resident, so on a 32 GB
    card the vision model has roughly 11 GB to live in — an 8B in bf16 wants 16.
    Quantising is the right side of that trade because this model classifies
    and counts; it does not generate anything a user sees.
    """

    multi_image = True

    def __init__(self, model: str = LOCAL_MODEL, cache_dir: str | None = None,
                 quantise: str = "4bit", max_memory_gb: float | None = None):
        """quantise: "4bit" | "8bit" | "none".

        4-bit by default, measured rather than assumed. The generator needs
        18.3 GB resident (encoder 8.3 + UNet 9.7 + VAE 0.3). An 8-bit
        Qwen3-VL-8B measured 10.2 GB, which leaves ComfyUI too little to hold
        the UNet — it offloaded all 9.7 GB to CPU and a generation took 215s
        instead of seconds. At 4-bit the vision model is ~5.5 GB and everything
        stays resident.
        """
        self.model_id = model
        self.cache_dir = cache_dir
        self.quantise = quantise
        self.max_memory_gb = max_memory_gb

    def __enter__(self) -> "LocalVision":
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(self.model_id,
                                                  cache_dir=self.cache_dir)
        # {"": 0} rather than "cuda:0": the whole model onto GPU 0, placed
        # through accelerate's dispatch. The plain string skips that dispatch,
        # and bitsandbytes layers then have no quantisation state - every
        # forward pass dies with "FP4 quantization state not initialized" and
        # an otherwise empty exception message.
        kwargs: dict = {"cache_dir": self.cache_dir, "device_map": {"": 0}}
        if self.quantise == "prequantized":
            # The repo already carries a quantization_config (unsloth's
            # bnb-4bit builds), so pass none of our own and let the checkpoint
            # describe itself. This also cuts the download to 20 GB from the
            # 67 GB bf16 original.
            #
            # device_map has to be {"": 0} - the whole model onto GPU 0, placed
            # through accelerate's dispatch. Both alternatives fail, in
            # different ways worth recording:
            #   "cuda:0"  skips the dispatch that initialises bitsandbytes' FP4
            #             state, so every forward pass dies with "FP4
            #             quantization state not initialized" and an otherwise
            #             empty exception message.
            #   "auto"    reserves headroom, decides 20.7 GB does not fit in
            #             24 GB, and tries to offload part of the model to CPU,
            #             which bnb 4-bit refuses outright.
            kwargs["device_map"] = {"": 0}
        elif self.quantise == "4bit":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True)
        elif self.quantise == "8bit":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["dtype"] = torch.bfloat16
        if self.max_memory_gb:
            # A hard ceiling so the vision model cannot creep into the memory
            # ComfyUI needs; without it a long run OOMs the generator instead.
            kwargs["max_memory"] = {0: f"{self.max_memory_gb}GiB", "cpu": "48GiB"}

        # AutoModelForImageTextToText resolves Qwen3-VL, Qwen2.5-VL and
        # InternVL alike, so swapping models for the bake-off is a config
        # change rather than a code change.
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, **kwargs)
        self.model.eval()
        return self

    def ask(self, images: str | list[str], prompt: str, max_new_tokens: int = 900) -> str:
        """Ask about one image or several, at full resolution.

        No triptych and no JPEG compression here — that was a workaround for
        NIM's 180 KB inline cap, and it cost exactly the detail these checks
        turn on. Locally each image goes in whole and separate.
        """
        from PIL import Image

        if isinstance(images, str):
            images = [images]
        pil = [Image.open(p).convert("RGB") for p in images]
        content = [{"type": "image"} for _ in pil]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self.proc.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = self.proc(text=[text], images=pil, padding=True,
                           return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, ids)]
        return self.proc.batch_decode(trimmed, skip_special_tokens=True)[0]

    def __exit__(self, *exc):
        del self.model, self.proc
        gc.collect()
        self.torch.cuda.empty_cache()


class NIMVision:
    """NVIDIA NIM over HTTP. No GPU, no download, per-request cost."""

    multi_image = False

    def __init__(self, api_key: str | None = None, model: str = NIM_MODEL,
                 timeout: float = 300.0, retries: int = 4, backoff: float = 2.0,
                 fallbacks: tuple[str, ...] = NIM_FALLBACKS,
                 qa_tokens: int = 420):
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY") \
            or os.environ.get("NIM_API_KEY")
        if not self.api_key:
            raise RuntimeError("set NVIDIA_API_KEY (or pass api_key=)")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        # Tried in order after the primary exhausts its retries. Once one works
        # it is promoted, so a degraded engine costs the batch one detour rather
        # than a wasted retry cycle on every image.
        self.fallbacks = tuple(fallbacks)
        # Decode budget for a full guardrail pass. Reasoning models need
        # thousands: muse-glimmer-30b emits its answer only after a long
        # reasoning_content block and returns nothing at all if truncated.
        self.qa_tokens = qa_tokens

    def __enter__(self) -> "NIMVision":
        return self

    def __exit__(self, *exc):
        return None

    def ask(self, images: str | list[str], prompt: str, max_new_tokens: int = 900) -> str:
        import httpx

        if isinstance(images, str):
            images = [images]
        blob, mime = _fit_inline(images)
        b64 = base64.b64encode(blob).decode()

        body = {
            "model": None,          # filled per candidate below
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]}],
            "max_tokens": max_new_tokens,
            # Deterministic: the same image must not pass one run and fail the
            # next, or a retry loop never converges.
            "temperature": 0.0,
            "top_p": 1.0,
        }
        # NIM returns intermittent 500s and 429s under load, and whole models
        # go down for long stretches. Retry with backoff, then move to the next
        # candidate model rather than failing the image.
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Accept": "application/json"}
        last = None
        r = None
        for candidate in (self.model, *self.fallbacks):
            body["model"] = candidate
            for attempt in range(self.retries):
                try:
                    r = httpx.post(NIM_URL, json=body, timeout=self.timeout,
                                   headers=headers)
                    if r.status_code < 400:
                        # Promote whatever answered, so the rest of the batch
                        # skips the dead engine entirely.
                        self.model = candidate
                        break
                    if r.status_code not in (408, 429, 500, 502, 503, 504):
                        r.raise_for_status()
                    last = httpx.HTTPStatusError(
                        f"HTTP {r.status_code} on {candidate}: {r.text[:160]}",
                        request=r.request, response=r)
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last = exc
                if attempt < self.retries - 1:
                    time.sleep(self.backoff * (2 ** attempt))
            else:
                continue        # this model is down; try the next
            break
        else:
            raise last or RuntimeError("all NIM models failed")

        msg = r.json()["choices"][0]["message"]
        # Reasoning models spend their budget thinking and return content=None;
        # the text we want is then under reasoning_content.
        return msg.get("content") or msg.get("reasoning_content") or ""


def inspect(backend, person: str, garment: str, result: str) -> dict:
    """Full QA on one result. Returns the verdict plus every check.

    Order is chosen for cost. The deterministic face and background checks run
    first — milliseconds, no GPU, no API call — and anything they reject never
    reaches the model. Anatomy runs next because it needs one image. Fidelity
    runs last because it needs three.
    """
    ok, reason, measured = numeric_gate(person, result)
    base = {"measured": measured, "anatomy": {}, "fidelity": {}}
    if not ok:
        return {**base, "ok": False, "reason": reason, "stage": "numeric"}

    anatomy = parse(backend.ask(result, ANATOMY_ASK, max_new_tokens=700),
                    ANATOMY_FIELDS)
    ok, reason = verdict(anatomy)
    if not ok:
        return {**base, "ok": False, "reason": reason, "stage": "anatomy",
                "anatomy": anatomy}

    fidelity = parse(
        backend.ask([person, garment, result], FIDELITY_ASK, max_new_tokens=900),
        FIDELITY_FIELDS)
    ok, reason = verdict(anatomy, fidelity)
    return {**base, "ok": ok, "reason": reason,
            "stage": "passed" if ok else "fidelity",
            "anatomy": anatomy, "fidelity": fidelity}
