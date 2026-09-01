"""Single owner of the GPU, because two workloads cannot share 24 GB.

    BROCHURE   Qwen3-VL-32B 4-bit, ~20 GB, alone
    GENERATE   ComfyUI 16.6 GB + Qwen3-VL-4B guardrail 5.5 GB = 22.1 GB
    IDLE       nothing resident

Transitions are serialised behind one lock and always unload before they load.
That ordering is the whole point: an earlier build let a vision model and
ComfyUI compete for memory, ComfyUI offloaded its 9.7 GB UNet to CPU, and an
8 s generation became 215 s. Refusing work that belongs to the other phase is
cheaper than discovering that at request time.
"""
from __future__ import annotations

import asyncio
import gc
import logging
import time
from enum import Enum

log = logging.getLogger("lifecycle")


class Phase(str, Enum):
    idle = "idle"
    brochure = "brochure"
    generate = "generate"


class PhaseConflict(RuntimeError):
    """Raised when a request needs a phase that is not loaded."""

    def __init__(self, needed: Phase, current: Phase):
        self.needed, self.current = needed, current
        super().__init__(
            f"this endpoint needs phase '{needed.value}' but the GPU is holding "
            f"'{current.value}'. POST /v1/phase {{\"phase\": \"{needed.value}\"}} first."
        )


def vram_gb() -> dict[str, float]:
    """Allocated and total VRAM, or zeros when torch has no device."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"allocated_gb": 0.0, "total_gb": 0.0}
        free, total = torch.cuda.mem_get_info()
        return {"allocated_gb": round((total - free) / 1e9, 2),
                "total_gb": round(total / 1e9, 2)}
    except Exception:                                    # noqa: BLE001
        return {"allocated_gb": 0.0, "total_gb": 0.0}


class GPUManager:
    """Loads and unloads the models for a phase. One transition at a time."""

    def __init__(self, settings, comfy_free=None):
        self.settings = settings
        # Injected rather than imported so the manager stays testable without
        # a running ComfyUI.
        self._comfy_free = comfy_free
        self.phase = Phase.idle
        self._lock = asyncio.Lock()
        self._vision = None          # the brochure model, when resident
        self._guardrail = None       # the guardrail model, when resident
        self.last_switch: float | None = None
        self.last_switch_seconds: float | None = None

    # ----------------------------------------------------------------- state
    def state(self) -> dict:
        return {
            "phase": self.phase.value,
            "brochure_model": self.settings.brochure_model
                              if self._vision else None,
            "guardrail_model": self.settings.guardrail_model
                               if self._guardrail else None,
            "last_switch_seconds": self.last_switch_seconds,
            **vram_gb(),
        }

    def require(self, phase: Phase) -> None:
        if self.phase is not phase:
            raise PhaseConflict(phase, self.phase)

    @property
    def vision(self):
        """The brochure model. Only valid in the brochure phase."""
        self.require(Phase.brochure)
        return self._vision

    @property
    def guardrail(self):
        """The guardrail model, or None when it is switched off."""
        return self._guardrail

    # ------------------------------------------------------------ transition
    async def switch(self, target: Phase) -> dict:
        async with self._lock:
            if target is self.phase:
                return self.state()
            t0 = time.time()
            log.info("phase %s -> %s", self.phase.value, target.value)
            # ComfyUI is a separate process holding its own ~18 GB. Ask it to
            # let go before loading anything large, or the brochure model has
            # nowhere to go.
            if target is not Phase.generate and self._comfy_free is not None:
                await self._comfy_free()
            # Unload first, always. Loading into a full card is what produced
            # the 215 s generation.
            await asyncio.to_thread(self._unload_all)
            if target is Phase.brochure:
                await asyncio.to_thread(self._load_brochure)
            # The generate phase deliberately loads nothing but ComfyUI. The
            # guardrail is fetched on demand by ensure_guardrail(), because
            # loading it unconditionally cost ~5.5 GB and 4-7 s per image even
            # when every request had it switched off - VRAM pressure paid for a
            # model that was doing nothing.
            self.phase = target
            self.last_switch = time.time()
            self.last_switch_seconds = round(time.time() - t0, 1)
            log.info("phase %s ready in %.1fs (%s)", target.value,
                     self.last_switch_seconds, vram_gb())
            return self.state()

    async def ensure_guardrail(self) -> bool:
        """Load the guardrail if a request needs it. Returns whether it is up."""
        if self._guardrail is not None:
            return True
        if not self.settings.guardrail_enabled:
            return False
        async with self._lock:
            if self._guardrail is None:
                t0 = time.time()
                await asyncio.to_thread(self._load_guardrail)
                log.info("guardrail loaded on demand in %.1fs (%s)",
                         time.time() - t0, vram_gb())
        return self._guardrail is not None

    async def release_guardrail(self) -> bool:
        """Unload the guardrail so generation gets the memory back.

        Called when a request explicitly asks for no guardrail. Freeing it is
        worth the reload cost: an idle guardrail measured 4-7 s per image in
        contention, and a reload is a one-off ~17 s.
        """
        if self._guardrail is None:
            return False
        async with self._lock:
            if self._guardrail is not None:
                try:
                    self._guardrail.__exit__(None, None, None)
                except Exception:                        # noqa: BLE001
                    log.warning("guardrail did not close cleanly", exc_info=True)
                self._guardrail = None
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:                        # noqa: BLE001
                    pass
                log.info("guardrail released (%s)", vram_gb())
        return True

    # --------------------------------------------------------------- loading
    def _load_brochure(self) -> None:
        from .vision import LocalVision
        s = self.settings
        log.info("loading brochure model %s (%s)", s.brochure_model,
                 s.brochure_quantise)
        self._vision = LocalVision(model=s.brochure_model,
                                   cache_dir=s.hf_cache_dir,
                                   quantise=s.brochure_quantise).__enter__()

    def _load_guardrail(self) -> None:
        if not self.settings.guardrail_enabled:
            log.info("guardrail disabled by configuration; not loading it")
            return
        from .vision import LocalVision
        s = self.settings
        log.info("loading guardrail model %s (%s)", s.guardrail_model,
                 s.guardrail_quantise)
        self._guardrail = LocalVision(model=s.guardrail_model,
                                      cache_dir=s.hf_cache_dir,
                                      quantise=s.guardrail_quantise,
                                      max_memory_gb=s.guardrail_max_memory_gb
                                      ).__enter__()

    def _unload_all(self) -> None:
        for name in ("_vision", "_guardrail"):
            obj = getattr(self, name)
            if obj is None:
                continue
            try:
                obj.__exit__(None, None, None)
            except Exception:                            # noqa: BLE001
                log.warning("%s did not close cleanly", name, exc_info=True)
            setattr(self, name, None)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        except Exception:                                # noqa: BLE001
            pass
