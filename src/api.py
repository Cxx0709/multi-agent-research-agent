"""FastAPI service: async research jobs with durable state.

v0.2 is async-by-design (breaking change from v0.1's blocking call):

    POST   /v1/research        -> 202 {job_id, status: "queued"}
    GET    /v1/jobs/{job_id}   -> job record (poll until status=succeeded)
    GET    /v1/jobs            -> list, filterable by ?status=&limit=
    DELETE /v1/jobs/{job_id}   -> cooperative cancel
    GET    /health             -> liveness + db check + queue depth

Production notes
----------------
* Jobs persist in SQLite (``JOB_DB_PATH``); a restart requeues orphaned
  jobs instead of losing them.
* Graph state is checkpointed per job (``CHECKPOINT_DB_PATH``) so a crashed
  run can resume from the last node.
* ``_INVOKE_LOCK`` serializes graph execution because the SQLite
  checkpointer isn't safe for concurrent writes. To scale past one worker,
  point the checkpointer at Postgres (AsyncPostgresSaver) and drop the lock.
* Job timeout is enforced around the worker thread; the thread itself is
  left to finish (documented limitation, same tradeoff as Celery's
  soft time limits).
"""
from __future__ import annotations

import contextvars
import logging
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import settings
from .graph import get_checkpointer, run_research
from .jobs import JobStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("research-agent.api")

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

# SQLite checkpointer isn't thread-safe for concurrent writes -> serialize runs.
_INVOKE_LOCK = threading.Lock()

_store: JobStore | None = None
_store_lock = threading.Lock()
_pool: ThreadPoolExecutor | None = None


def get_store() -> JobStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                import os

                os.makedirs(os.path.dirname(settings.job_db_path) or ".", exist_ok=True)
                _store = JobStore(settings.job_db_path)
    return _store


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool
    get_store()  # init db + requeue orphans
    _pool = ThreadPoolExecutor(
        max_workers=settings.max_concurrent_jobs, thread_name_prefix="research-worker"
    )
    logger.info(
        "startup: db=%s max_workers=%d",
        settings.job_db_path,
        settings.max_concurrent_jobs,
    )
    yield
    if _pool:
        _pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="research-agent", version="0.2.0", lifespan=lifespan)


# -- middleware: request ids + rate limiting ------------------------------


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    _request_id.set(rid)
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


_rate_buckets: dict[str, deque[float]] = {}
_rate_lock = threading.Lock()


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _rate_lock:
        bucket = _rate_buckets.setdefault(ip, deque())
        while bucket and bucket[0] <= now - 60:
            bucket.popleft()
        if len(bucket) >= settings.rate_limit_per_min:
            return JSONResponse({"detail": "rate limit exceeded, slow down"}, status_code=429)
        bucket.append(now)
    return await call_next(request)


# -- worker ---------------------------------------------------------------


def _run_job(job_id: str) -> None:
    store = get_store()
    if not store.mark_running(job_id):
        logger.info("job=%s skipped (cancelled while queued)", job_id)
        return
    job = store.get(job_id)
    logger.info("job=%s start topic=%r", job_id, (job.topic if job else "")[:80])

    inner = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"job-{job_id}")
    try:
        fut = inner.submit(
            _invoke_guarded, job.topic if job else "", job_id
        )
        result = fut.result(timeout=settings.job_timeout_s)
    except FuturesTimeout:
        logger.warning("job=%s timed out after %ds", job_id, settings.job_timeout_s)
        store.mark_failed(job_id, f"job timed out after {settings.job_timeout_s}s")
        return
    except Exception as exc:  # noqa: BLE001 - surface any worker crash as failed
        logger.exception("job=%s crashed", job_id)
        store.mark_failed(job_id, f"{type(exc).__name__}: {exc}")
        return
    finally:
        inner.shutdown(wait=False, cancel_futures=True)

    if result.get("cancelled"):
        store.mark_cancelled(job_id)
        logger.info("job=%s cancelled", job_id)
    else:
        store.mark_succeeded(job_id, result)
        logger.info(
            "job=%s succeeded rounds=%s in=%s out=%s",
            job_id,
            result.get("rounds"),
            result.get("input_tokens"),
            result.get("output_tokens"),
        )


def _invoke_guarded(topic: str, job_id: str) -> dict:
    with _INVOKE_LOCK:
        return run_research(
            topic, job_id=job_id, checkpointer=get_checkpointer(settings.checkpoint_db_path)
        )


# -- schemas --------------------------------------------------------------


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=500)


class JobAccepted(BaseModel):
    job_id: str
    status: str = "queued"


# -- routes ---------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    store = get_store()
    try:
        queued = len(store.list(status="queued", limit=1000))
        running = len(store.list(status="running", limit=1000))
        db = "ok"
    except Exception as exc:  # noqa: BLE001
        db, queued, running = f"error: {exc}", -1, -1
    return {"status": "ok", "db": db, "queued": queued, "running": running}


@app.post("/v1/research", response_model=JobAccepted, status_code=202)
def submit_research(req: ResearchRequest, request: Request) -> dict:
    store = get_store()
    rid = _request_id.get()
    job = store.create(req.topic, request_id=rid)
    logger.info("rid=%s job=%s queued topic=%r", rid, job.id, req.topic[:80])
    assert _pool is not None
    _pool.submit(_run_job, job.id)
    return {"job_id": job.id, "status": "queued"}


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = get_store().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_dict()


@app.get("/v1/jobs")
def list_jobs(
    status: str | None = Query(default=None, description="queued|running|succeeded|failed|cancelled"),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict:
    if status and status not in ("queued", "running", "succeeded", "failed", "cancelled"):
        raise HTTPException(status_code=400, detail="invalid status filter")
    jobs = get_store().list(status=status, limit=limit)
    return {"jobs": [j.to_dict() for j in jobs]}


@app.delete("/v1/jobs/{job_id}")
def cancel_job(job_id: str) -> dict:
    job = get_store().cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    logger.info("job=%s cancel requested -> %s", job_id, job.status)
    return job.to_dict()
