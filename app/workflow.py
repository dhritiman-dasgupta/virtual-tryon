"""Builds a ComfyUI prompt graph from the vendored template.

Node map for workflows/tryon_api.json (derived from the upstream graph):

    270  LoadImage                 person image      -> 271 scale -> 164 VAEEncode
    278  LoadImage                 garment image     -> 291 scale -> 162 VAEEncode
    271  ImageScaleToTotalPixels   megapixels
    291  ImageScaleToTotalPixels   megapixels
    157  GetImageSize              drives output canvas (reads 271 = person)
    168  CLIPTextEncode            instruction text (inlined; upstream used a
                                   Comfyroll "CR Text" node, now removed)
    354  CLIPLoader                text encoder
    356  UnetLoaderGGUF            diffusion weights
    174  VAELoader                 vae
    309  LoraLoaderModelOnly       try-on LoRA + strength
    152  Flux2Scheduler            steps
    153  CFGGuider                 cfg
    158  RandomNoise               seed
    900  SaveImage                 result
"""
from __future__ import annotations

import copy
import json
import secrets
from pathlib import Path

from .config import settings

PERSON_NODE = "270"
GARMENT_NODE = "278"
TEXT_NODE = "168"
UNET_NODE = "356"
CLIP_NODE = "354"
VAE_NODE = "174"
LORA_NODE = "309"
SCHEDULER_NODE = "152"
GUIDER_NODE = "153"
NOISE_NODE = "158"
SAVE_NODE = "900"
SCALE_NODES = ("271", "291")

MAX_SEED = 2**63 - 1

_TEMPLATE: dict | None = None


def _template() -> dict:
    global _TEMPLATE
    if _TEMPLATE is None:
        _TEMPLATE = json.loads(Path(settings.workflow_path).read_text(encoding="utf-8"))
    return _TEMPLATE


def build(
    *,
    person_filename: str,
    garment_filename: str,
    prompt: str,
    steps: int,
    cfg: float,
    seed: int | None,
    lora_strength: float,
    megapixels: float,
    swap_slots: bool = False,
    filename_prefix: str = "tryon",
) -> tuple[dict, int]:
    """Return (graph, resolved_seed)."""
    g = copy.deepcopy(_template())

    if seed is None:
        seed = secrets.randbelow(MAX_SEED)

    person_slot, garment_slot = PERSON_NODE, GARMENT_NODE
    if swap_slots:
        person_slot, garment_slot = garment_slot, person_slot

    g[person_slot]["inputs"]["image"] = person_filename
    g[garment_slot]["inputs"]["image"] = garment_filename

    g[TEXT_NODE]["inputs"]["text"] = prompt
    g[UNET_NODE]["inputs"]["unet_name"] = settings.unet_name
    g[CLIP_NODE]["inputs"]["clip_name"] = settings.clip_name
    g[VAE_NODE]["inputs"]["vae_name"] = settings.vae_name

    g[LORA_NODE]["inputs"]["lora_name"] = settings.lora_name
    g[LORA_NODE]["inputs"]["strength_model"] = lora_strength

    g[SCHEDULER_NODE]["inputs"]["steps"] = steps
    g[GUIDER_NODE]["inputs"]["cfg"] = cfg
    g[NOISE_NODE]["inputs"]["noise_seed"] = seed

    for node in SCALE_NODES:
        g[node]["inputs"]["megapixels"] = megapixels

    g[SAVE_NODE]["inputs"]["filename_prefix"] = filename_prefix

    return g, seed
