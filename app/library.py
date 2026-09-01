"""A persistent library of analysed garments.

Garments are uploaded and described once, with the large vision model resident
and the card to itself. The description is the expensive part and it never
changes, so it is written to disk and reused for every person afterwards.

Inference then needs only the person: pick garments from the library, upload a
photograph, generate. That ordering is what the hardware wants anyway - the
model that writes the descriptions and the model that generates cannot both be
resident, so doing all the describing first means one phase switch per session
rather than one per garment.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("library")


@dataclass
class LibraryItem:
    name: str
    image: Path
    fields: dict = field(default_factory=dict)
    pieces: list[str] = field(default_factory=list)
    piece_details: list[str] = field(default_factory=list)
    added: float = field(default_factory=time.time)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "type": self.fields.get("TYPE"),
            "colours": self.fields.get("COLOURS"),
            "fabric": self.fields.get("FABRIC"),
            "pieces": self.pieces,
            "piece_details": self.piece_details,
            "dupatta": self.fields.get("DUPATTA"),
            "added": self.added,
            "image_url": f"/v1/garments/{self.name}/image",
        }


class GarmentLibrary:
    """Disk-backed, so it survives an API restart but not a wiped volume."""

    def __init__(self, root: Path):
        self.root = Path(root)
        (self.root / "images").mkdir(parents=True, exist_ok=True)
        self.meta = self.root / "library.json"
        self.items: dict[str, LibraryItem] = {}
        self._load()

    def _load(self) -> None:
        if not self.meta.exists():
            return
        try:
            raw = json.loads(self.meta.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("library index unreadable; starting empty")
            return
        for name, d in raw.items():
            img = self.root / "images" / d["image"]
            if not img.exists():
                # The index must never advertise a garment whose picture is
                # gone: generation would fail later with a confusing error.
                continue
            self.items[name] = LibraryItem(
                name=name, image=img, fields=d.get("fields", {}),
                pieces=d.get("pieces", []),
                piece_details=d.get("piece_details", []),
                added=d.get("added", time.time()))
        log.info("library loaded: %d garments", len(self.items))

    def _save(self) -> None:
        raw = {n: {"image": i.image.name, "fields": i.fields,
                   "pieces": i.pieces, "piece_details": i.piece_details,
                   "added": i.added}
               for n, i in self.items.items()}
        tmp = self.meta.with_suffix(".tmp")
        tmp.write_text(json.dumps(raw, indent=2))
        tmp.replace(self.meta)

    def add(self, name: str, data: bytes, suffix: str = ".jpg") -> LibraryItem:
        name = "".join(c for c in name if c.isalnum() or c in "-_") or "garment"
        dst = self.root / "images" / f"{name}{suffix}"
        dst.write_bytes(data)
        item = LibraryItem(name=name, image=dst)
        self.items[name] = item
        self._save()
        return item

    def describe(self, item: LibraryItem, fields: dict, pieces: list[str],
                 piece_details: list[str]) -> None:
        item.fields, item.pieces, item.piece_details = fields, pieces, piece_details
        self._save()

    def get(self, name: str) -> LibraryItem | None:
        return self.items.get(name)

    def list(self) -> list[dict]:
        return [i.summary() for i in
                sorted(self.items.values(), key=lambda x: -x.added)]

    def remove(self, name: str) -> bool:
        item = self.items.pop(name, None)
        if item is None:
            return False
        item.image.unlink(missing_ok=True)
        self._save()
        return True

    def clear(self) -> int:
        n = len(self.items)
        self.items.clear()
        shutil.rmtree(self.root / "images", ignore_errors=True)
        (self.root / "images").mkdir(parents=True, exist_ok=True)
        self._save()
        return n
