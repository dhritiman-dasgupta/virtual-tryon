"""Request/response models."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class TryOnParams(BaseModel):
    """Generation knobs. The Colab notebook hid all of these in the workflow
    JSON, which made A/B testing impossible — they are all exposed here."""

    prompt: str | None = Field(
        None,
        description="Instruction text. Leave null to use the LoRA's native "
        "Chinese trigger phrase (see /v1/prompts).",
    )
    steps: int = Field(8, ge=1, le=50)
    cfg: float = Field(1.0, ge=1.0, le=10.0)
    seed: int | None = Field(None, description="Null picks a random seed.")
    lora_strength: float = Field(0.4, ge=0.0, le=2.0)
    megapixels: float = Field(1.0, ge=0.25, le=4.0)
    swap_slots: bool = Field(
        False,
        description="Swap which LoadImage node receives person vs garment. "
        "See README — the upstream graph is ambiguous about this and it is "
        "worth testing both ways on your own images.",
    )


class JobCreated(BaseModel):
    job_id: str
    status: JobStatus
    poll_url: str


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = Field(0.0, ge=0.0, le=1.0)
    step: int | None = None
    total_steps: int | None = None
    # null means the image was never inspected - never read it as a pass.
    guardrail: bool = False
    guardrail_ok: bool | None = None
    guardrail_reason: str | None = None
    guardrail_seconds: float | None = None
    attempts: int = 1
    seed: int | None = None
    duration_seconds: float | None = None
    image_url: str | None = None
    error: str | None = None


class HealthInfo(BaseModel):
    status: Literal["ok", "degraded"]
    comfy_reachable: bool
    models_present: dict[str, bool]
    queue_depth: int
    detail: str | None = None


class PromptEntry(BaseModel):
    id: str
    label: str
    prompt: str
