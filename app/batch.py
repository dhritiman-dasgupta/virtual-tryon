"""A batch: one person, many garments, described first and generated second.

This is the shape the GPU forces. The vision model that writes the prompts and
the generator that uses them cannot both be resident in 24 GB, so a batch has
two operations that each own a phase switch:

    describe()   loads the vision model alone, reads every garment, keeps the
                 prompts, and is charged once per garment ever because the
                 readings are cached
    generate()   unloads it, loads the generator and the guardrail, and runs
                 through the stored prompts

Keeping them apart is not a limitation to work around - it is why the large
model can be used at all. Sharing the card between them was measured at four
times slower than swapping.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("batch")


@dataclass
class GarmentEntry:
    name: str
    path: Path
    prompt: str = ""
    fields: dict = field(default_factory=dict)
    pieces: list[str] = field(default_factory=list)
    piece_details: list[str] = field(default_factory=list)
    status: str = "described"        # described | running | done | failed
    seconds: float | None = None
    guardrail_ok: bool | None = None
    guardrail_reason: str | None = None
    attempts: int = 1
    image: bytes | None = None
    error: str | None = None

    def info(self) -> dict:
        return {"garment": self.name, "status": self.status,
                "type": self.fields.get("TYPE"),
                "colours": self.fields.get("COLOURS"),
                "pieces": self.pieces, "piece_details": self.piece_details,
                "prompt": self.prompt, "seconds": self.seconds,
                "guardrail_ok": self.guardrail_ok,
                "guardrail_reason": self.guardrail_reason,
                "attempts": self.attempts, "error": self.error,
                "image_url": (f"/v1/batch/{{batch}}/image/{self.name}"
                              if self.image else None)}


@dataclass
class Batch:
    batch_id: str
    created: float = field(default_factory=time.time)
    person_path: Path | None = None
    person_fields: dict = field(default_factory=dict)
    garments: dict[str, GarmentEntry] = field(default_factory=dict)
    stage: str = "new"               # new | describing | described |
                                     # generating | complete
    current: str | None = None      # the garment being generated right now
    position: int = 0                # its place in the queue, 1-based
    queued: int = 0                  # how many are queued in total
    describe_seconds: float | None = None
    generate_seconds: float | None = None
    error: str | None = None

    def info(self) -> dict:
        done = [g for g in self.garments.values() if g.status == "done"]
        checked = [g for g in done if g.guardrail_ok is not None]
        return {
            "batch_id": self.batch_id, "stage": self.stage,
            "current": self.current, "position": self.position,
            "queued": self.queued,
            "eta_seconds": (round((self.queued - self.position + 1) * (
                sum(g.seconds for g in self.garments.values()
                    if g.seconds and g.status == "done") /
                max(1, sum(1 for g in self.garments.values()
                           if g.seconds and g.status == "done"))), 1)
                if self.stage == "generating" and any(
                    g.status == "done" for g in self.garments.values()) else None),
            "person": self.person_fields,
            "has_person": self.person_path is not None,
            "garments": len(self.garments),
            "described": sum(1 for g in self.garments.values() if g.prompt),
            "generated": len(done),
            "failed": [g.name for g in self.garments.values()
                       if g.status == "failed"],
            "inspected": len(checked),
            "passed": sum(1 for g in checked if g.guardrail_ok),
            # Unverified is reported separately and never folded into "passed".
            "unverified": len(done) - len(checked),
            "describe_seconds": self.describe_seconds,
            "generate_seconds": self.generate_seconds,
            "error": self.error,
            "items": [dict(g.info(), image_url=(
                f"/v1/batch/{self.batch_id}/image/{g.name}" if g.image else None))
                for g in self.garments.values()],
        }


class BatchStore:
    def __init__(self, ttl_seconds: int = 86400):
        self._batches: dict[str, Batch] = {}
        self.ttl = ttl_seconds
        self.lock = asyncio.Lock()

    def create(self) -> Batch:
        self._reap()
        b = Batch(batch_id=uuid.uuid4().hex[:12])
        self._batches[b.batch_id] = b
        return b

    def get(self, batch_id: str) -> Batch | None:
        return self._batches.get(batch_id)

    def _reap(self) -> None:
        cutoff = time.time() - self.ttl
        for bid in [k for k, v in self._batches.items() if v.created < cutoff]:
            self._batches.pop(bid, None)
