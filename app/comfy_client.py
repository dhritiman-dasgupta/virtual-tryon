"""Async client for a ComfyUI backend.

Differs from the Colab notebook in three ways that matter:
  * images are POSTed to ComfyUI's /upload/image instead of being copied into
    a local directory, so the backend can live on another host;
  * the websocket loop has a timeout and surfaces execution_error frames
    instead of blocking forever on a silent failure;
  * every failure carries the backend's actual message.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncIterator, Callable

import httpx
import websockets

from .config import settings

log = logging.getLogger("comfy")


class ComfyError(RuntimeError):
    """ComfyUI rejected the graph or failed mid-execution."""


class ComfyClient:
    def __init__(self, base: str | None = None, ws: str | None = None) -> None:
        self.base = base or settings.comfy_base
        self.ws_url = ws or settings.comfy_ws
        self._http = httpx.AsyncClient(base_url=self.base, timeout=60.0)

    async def aclose(self) -> None:
        await self._http.aclose()

    # ---------------------------------------------------------------- health

    async def reachable(self) -> bool:
        try:
            r = await self._http.get("/system_stats", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    async def object_info(self) -> dict[str, Any]:
        r = await self._http.get("/object_info", timeout=30.0)
        r.raise_for_status()
        return r.json()

    async def free_models(self) -> bool:
        """Ask ComfyUI to unload its models and release VRAM.

        ComfyUI runs as a separate process, so unloading the API's own vision
        models does nothing to the ~18 GB it holds. Without this, switching to
        the brochure phase after any generation tried to load a 16 GB model
        into a card ComfyUI had already filled, and died with CUDA OOM.
        """
        try:
            r = await self._http.post(
                "/free", json={"unload_models": True, "free_memory": True})
            return r.status_code < 400
        except Exception:                                # noqa: BLE001
            log.warning("could not free ComfyUI models", exc_info=True)
            return False

    async def queue_depth(self) -> int:
        try:
            r = await self._http.get("/queue", timeout=5.0)
            q = r.json()
            return len(q.get("queue_running", [])) + len(q.get("queue_pending", []))
        except Exception:
            return 0

    # ---------------------------------------------------------------- upload

    async def upload_image(self, data: bytes, filename: str) -> str:
        """Push bytes into ComfyUI's input store; returns the name to reference."""
        files = {"image": (filename, data, "application/octet-stream")}
        r = await self._http.post(
            "/upload/image", files=files, data={"overwrite": "true"}
        )
        if r.status_code != 200:
            raise ComfyError(f"upload failed ({r.status_code}): {r.text[:400]}")
        body = r.json()
        name = body.get("name", filename)
        subfolder = body.get("subfolder") or ""
        return f"{subfolder}/{name}" if subfolder else name

    # ------------------------------------------------------------- execution

    async def run(
        self,
        graph: dict,
        *,
        on_progress: Callable[[int, int], None] | None = None,
        timeout: float = 900.0,
    ) -> list[bytes]:
        """Submit a graph, follow it to completion, return decoded images."""
        client_id = str(uuid.uuid4())

        # Connect BEFORE queueing, otherwise fast jobs can finish before we
        # are listening and we would wait for frames that never arrive.
        async with websockets.connect(
            f"{self.ws_url}?clientId={client_id}",
            max_size=None,
            ping_interval=20,
        ) as ws:
            r = await self._http.post(
                "/prompt", json={"prompt": graph, "client_id": client_id}
            )
            if r.status_code != 200:
                raise ComfyError(_explain_prompt_error(r))

            prompt_id = r.json()["prompt_id"]
            await self._await_completion(ws, prompt_id, on_progress, timeout)

        return await self._collect_outputs(prompt_id)

    async def _await_completion(
        self,
        ws: Any,
        prompt_id: str,
        on_progress: Callable[[int, int], None] | None,
        timeout: float,
    ) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise ComfyError(f"timed out after {timeout:.0f}s waiting on ComfyUI")

            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                raise ComfyError(f"timed out after {timeout:.0f}s waiting on ComfyUI")

            if not isinstance(raw, str):
                continue  # binary preview frame

            msg = json.loads(raw)
            mtype, data = msg.get("type"), msg.get("data", {})

            if mtype == "progress" and on_progress:
                on_progress(int(data.get("value", 0)), int(data.get("max", 1)))

            elif mtype == "execution_error" and data.get("prompt_id") == prompt_id:
                raise ComfyError(
                    f"{data.get('node_type', '?')} (node {data.get('node_id', '?')}): "
                    f"{data.get('exception_message', 'unknown error')}"
                )

            elif mtype in ("execution_interrupted", "execution_cached_error"):
                if data.get("prompt_id") == prompt_id:
                    raise ComfyError(f"execution interrupted: {data}")

            elif mtype == "executing":
                if data.get("node") is None and data.get("prompt_id") == prompt_id:
                    return

    async def _collect_outputs(self, prompt_id: str) -> list[bytes]:
        r = await self._http.get(f"/history/{prompt_id}")
        r.raise_for_status()
        history = r.json().get(prompt_id)
        if not history:
            raise ComfyError("job finished but no history entry was recorded")

        images: list[bytes] = []
        for node_output in history.get("outputs", {}).values():
            for img in node_output.get("images", []):
                if img.get("type") == "temp":
                    continue  # previews, not results
                got = await self._http.get(
                    "/view",
                    params={
                        "filename": img["filename"],
                        "subfolder": img.get("subfolder", ""),
                        "type": img.get("type", "output"),
                    },
                )
                got.raise_for_status()
                images.append(got.content)

        if not images:
            raise ComfyError("job finished but produced no output images")
        return images


def _explain_prompt_error(r: httpx.Response) -> str:
    """ComfyUI returns structured validation errors — surface them properly."""
    try:
        body = r.json()
    except Exception:
        return f"/prompt rejected ({r.status_code}): {r.text[:400]}"

    parts: list[str] = []
    if err := body.get("error"):
        parts.append(f"{err.get('type', 'error')}: {err.get('message', '')}")
    for node_id, info in (body.get("node_errors") or {}).items():
        for e in info.get("errors", []):
            parts.append(
                f"node {node_id} ({info.get('class_type', '?')}): "
                f"{e.get('message')} {e.get('details', '')}".strip()
            )
    return " | ".join(parts) or f"/prompt rejected ({r.status_code})"
