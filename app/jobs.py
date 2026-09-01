"""In-process job queue.

ComfyUI serialises on the GPU, so the default worker count is 1. The queue
exists so HTTP callers get an immediate job id instead of holding a
connection open for the length of a diffusion run — most reverse proxies
kill requests well before a 4-minute generation finishes.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field

from .comfy_client import ComfyClient, ComfyError
from .config import settings
from .schemas import JobStatus

log = logging.getLogger("jobs")


@dataclass
class Job:
    job_id: str
    graph: dict
    seed: int
    status: JobStatus = JobStatus.queued
    progress: float = 0.0
    step: int | None = None
    total_steps: int | None = None
    image: bytes | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    # --- guardrail ---
    guardrail: bool = False
    # None means the image was never inspected. It must never be confused with
    # a pass: an unchecked result that reports itself as checked is the single
    # most expensive bug this pipeline has had.
    guardrail_ok: bool | None = None
    guardrail_reason: str | None = None
    guardrail_seconds: float | None = None
    attempts: int = 1
    person_path: str | None = None
    garment_path: str | None = None
    inspector: object | None = None

    @property
    def duration(self) -> float | None:
        if self.started_at is None:
            return None
        return (self.finished_at or time.time()) - self.started_at


class JobManager:
    def __init__(self, client: ComfyClient) -> None:
        self.client = client
        self._jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._events: dict[str, asyncio.Event] = {}

    # ------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        for i in range(max(1, settings.workers)):
            self._workers.append(asyncio.create_task(self._worker(i), name=f"worker-{i}"))
        self._workers.append(asyncio.create_task(self._reaper(), name="reaper"))

    async def stop(self) -> None:
        for t in self._workers:
            t.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    # ---------------------------------------------------------------- public

    def submit(self, graph: dict, seed: int) -> Job:
        job = Job(job_id=uuid.uuid4().hex, graph=graph, seed=seed)
        self._jobs[job.job_id] = job
        self._events[job.job_id] = asyncio.Event()
        self._queue.put_nowait(job.job_id)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    async def wait(self, job_id: str, timeout: float) -> Job | None:
        ev = self._events.get(job_id)
        if ev is None:
            return None
        try:
            await asyncio.wait_for(ev.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        return self._jobs.get(job_id)

    def depth(self) -> int:
        return self._queue.qsize()

    # --------------------------------------------------------------- workers

    async def _worker(self, index: int) -> None:
        while True:
            job_id = await self._queue.get()
            job = self._jobs.get(job_id)
            if job is None:
                continue

            job.status = JobStatus.running
            job.started_at = time.time()

            def on_progress(value: int, total: int, _j: Job = job) -> None:
                _j.step, _j.total_steps = value, total
                _j.progress = min(value / total, 1.0) if total else 0.0

            try:
                images = await self.client.run(job.graph, on_progress=on_progress)
                job.image = images[0]
                job.progress = 1.0
                # Inspect BEFORE the status flips. Callers poll for
                # "succeeded" and read the verdict in the same breath, so
                # publishing success first hands them an image whose
                # guardrail_ok is still null - indistinguishable, from the
                # outside, from an unchecked run.
                if job.guardrail and job.inspector is not None:
                    await self._inspect(job)
                job.status = JobStatus.succeeded
            except ComfyError as exc:
                job.error = str(exc)
                job.status = JobStatus.failed
                log.error("job %s failed: %s", job.job_id, exc)
            except Exception as exc:  # noqa: BLE001 - worker must never die
                job.error = f"{type(exc).__name__}: {exc}"
                job.status = JobStatus.failed
                log.exception("job %s crashed", job.job_id)
            finally:
                job.finished_at = time.time()
                job.graph = {}  # free the graph, keep the metadata
                if ev := self._events.get(job.job_id):
                    ev.set()
                self._queue.task_done()

    async def _inspect(self, job: Job) -> None:
        """Check the finished image, reseeding while a critical check fails.

        Retries re-run the graph with a fresh seed rather than re-wording the
        prompt: the defects that survive a good prompt - a blouse and skirt
        fusing into one panel, most notably - were measured to be decided by
        the seed, and unchanged by LoRA strength or step count.
        """
        import tempfile, os
        from .qa import inspect as qa_inspect, worth_retrying
        from .workflow import NOISE_NODE

        t0 = time.time()
        try:
            for attempt in range(1, settings.max_retries + 1):
                job.attempts = attempt
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
                    fh.write(job.image)
                    result_path = fh.name
                try:
                    report = await asyncio.to_thread(
                        qa_inspect, job.inspector, job.person_path,
                        job.garment_path, result_path)
                finally:
                    os.unlink(result_path)

                job.guardrail_ok = bool(report.get("ok"))
                job.guardrail_reason = report.get("reason")
                if job.guardrail_ok or not worth_retrying(report):
                    break
                if attempt >= settings.max_retries:
                    break

                # Reseed and regenerate.
                new_seed = job.seed + 1000 * attempt
                log.info("job %s attempt %d failed (%s); reseeding to %d",
                         job.job_id, attempt, (job.guardrail_reason or "")[:60],
                         new_seed)
                job.graph[NOISE_NODE]["inputs"]["noise_seed"] = new_seed
                job.seed = new_seed
                images = await self.client.run(job.graph)
                job.image = images[0]
        except Exception as exc:  # noqa: BLE001
            # Fail closed: an inspection that did not complete is not a pass.
            job.guardrail_ok = False
            job.guardrail_reason = f"guardrail error: {type(exc).__name__}: {exc}"
            log.exception("guardrail failed for job %s", job.job_id)
        finally:
            job.guardrail_seconds = round(time.time() - t0, 2)

    async def _reaper(self) -> None:
        """Drop finished jobs (and their image bytes) after the TTL."""
        while True:
            await asyncio.sleep(60)
            cutoff = time.time() - settings.job_ttl_seconds
            stale = [
                jid
                for jid, j in self._jobs.items()
                if j.finished_at and j.finished_at < cutoff
            ]
            for jid in stale:
                self._jobs.pop(jid, None)
                self._events.pop(jid, None)
            if stale:
                log.info("reaped %d expired job(s)", len(stale))
