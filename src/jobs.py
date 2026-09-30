"""Persistent job store for the research agent.

Each research request becomes a *job* with a durable lifecycle:

    queued -> running -> succeeded
                     |-> failed
                     |-> cancelled
              queued -> cancelled   (cancel before pickup)

Jobs live in SQLite so they survive process restarts. The table is the
source of truth for the API; worker threads update it as they go.

Cancellation is cooperative: nodes in the LangGraph pipeline check
``jobs.is_cancelled(job_id)`` at entry and abort cleanly.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    topic         TEXT NOT NULL,
    status        TEXT NOT NULL,
    request_id    TEXT,
    created_at    REAL NOT NULL,
    started_at    REAL,
    finished_at   REAL,
    report        TEXT,
    findings      TEXT,          -- JSON list
    rounds        INTEGER,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
"""

VALID_STATUSES = ("queued", "running", "succeeded", "failed", "cancelled")


@dataclass
class Job:
    id: str
    topic: str
    status: str
    request_id: str = ""
    created_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    report: str = ""
    findings: list = field(default_factory=list)
    rounds: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


_CANCEL_EVENTS: dict[str, threading.Event] = {}
_CANCEL_LOCK = threading.Lock()


def is_cancelled(job_id: str) -> bool:
    """Module-level cancel flag check — usable from graph nodes without a store handle."""
    with _CANCEL_LOCK:
        ev = _CANCEL_EVENTS.get(job_id or "")
        return ev.is_set() if ev else False


def _set_cancelled(job_id: str) -> threading.Event:
    with _CANCEL_LOCK:
        return _CANCEL_EVENTS.setdefault(job_id, threading.Event())


def _clear_cancelled(job_id: str) -> None:
    with _CANCEL_LOCK:
        _CANCEL_EVENTS.pop(job_id, None)


class JobStore:
    """Thread-safe SQLite job store. One instance per process."""

    def __init__(self, path: str):
        self._path = path
        self._lock = threading.RLock()
        # check_same_thread=False + explicit lock: safe for our worker threads.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        # Jobs left "running" by a crashed process can never finish.
        # Requeue them so they are picked up again instead of stuck forever.
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET status='queued', started_at=NULL "
                "WHERE status IN ('running','queued')"
            )
            self._conn.commit()
            if cur.rowcount:
                print(f"[jobs] requeued {cur.rowcount} orphaned job(s) from previous run")

    # -- lifecycle ------------------------------------------------------

    def create(self, topic: str, request_id: str = "") -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            topic=topic,
            status="queued",
            request_id=request_id,
            created_at=time.time(),
        )
        with self._lock:
            _set_cancelled(job.id)
            self._conn.execute(
                "INSERT INTO jobs (id, topic, status, request_id, created_at)"
                " VALUES (?,?,?,?,?)",
                (job.id, job.topic, job.status, job.request_id, job.created_at),
            )
            self._conn.commit()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
        return self._row_to_job(row) if row else None

    def list(self, status: str | None = None, limit: int = 50) -> list[Job]:
        with self._lock:
            if status:
                rows = self._conn.execute(
                    "SELECT * FROM jobs WHERE status=? ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
        return [self._row_to_job(r) for r in rows]

    def mark_running(self, job_id: str) -> bool:
        """Move queued -> running. Returns False if the job was cancelled meanwhile."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET status='running', started_at=? "
                "WHERE id=? AND status='queued'",
                (time.time(), job_id),
            )
            self._conn.commit()
            return cur.rowcount == 1

    def mark_succeeded(self, job_id: str, result: dict) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE jobs SET status='succeeded', finished_at=?,
                   report=?, findings=?, rounds=?, input_tokens=?, output_tokens=?
                   WHERE id=?""",
                (
                    time.time(),
                    result.get("report", ""),
                    json.dumps(result.get("findings", []), ensure_ascii=False),
                    result.get("rounds", 0),
                    result.get("input_tokens", 0),
                    result.get("output_tokens", 0),
                    job_id,
                ),
            )
            self._conn.commit()
            _clear_cancelled(job_id)

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status='failed', finished_at=?, error=? WHERE id=?",
                (time.time(), error[:2000], job_id),
            )
            self._conn.commit()
            _clear_cancelled(job_id)

    def mark_cancelled(self, job_id: str) -> None:
        """Flip a running job to cancelled (called by the worker thread itself)."""
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status='cancelled', finished_at=? "
                "WHERE id=? AND status='running'",
                (time.time(), job_id),
            )
            self._conn.commit()
            _clear_cancelled(job_id)

    def cancel(self, job_id: str) -> Job | None:
        """Request cancellation. Queued jobs stop immediately; running jobs
        abort at the next node boundary (cooperative)."""
        with self._lock:
            job = self.get(job_id)
            if job is None or job.status in ("succeeded", "failed", "cancelled"):
                return job
            _set_cancelled(job_id).set()
            if job.status == "queued":
                self._conn.execute(
                    "UPDATE jobs SET status='cancelled', finished_at=? WHERE id=?",
                    (time.time(), job_id),
                )
                self._conn.commit()
                _clear_cancelled(job_id)
                return self.get(job_id)
            return job  # running: node wrapper will flip it to cancelled

    def is_cancelled(self, job_id: str) -> bool:
        return is_cancelled(job_id)

    # -- internals ------------------------------------------------------

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"],
            topic=row["topic"],
            status=row["status"],
            request_id=row["request_id"] or "",
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            report=row["report"] or "",
            findings=json.loads(row["findings"]) if row["findings"] else [],
            rounds=row["rounds"] or 0,
            input_tokens=row["input_tokens"] or 0,
            output_tokens=row["output_tokens"] or 0,
            error=row["error"] or "",
        )
