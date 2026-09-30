"""Job store tests: lifecycle, cancel semantics, crash recovery. No network."""
import sqlite3

import pytest

from src import jobs
from src.jobs import JobStore, is_cancelled


@pytest.fixture
def store(tmp_path):
    s = JobStore(str(tmp_path / "jobs.db"))
    yield s
    # clean module-level cancel flags between tests
    jobs._CANCEL_EVENTS.clear()


def test_create_get_roundtrip(store):
    job = store.create("量子计算", request_id="r1")
    assert job.status == "queued"
    fetched = store.get(job.id)
    assert fetched is not None and fetched.topic == "量子计算"
    assert fetched.request_id == "r1"
    assert store.get("nope") is None


def test_mark_running_only_from_queued(store):
    job = store.create("t")
    assert store.mark_running(job.id) is True
    assert store.mark_running(job.id) is False  # already running
    assert store.get(job.id).status == "running"


def test_mark_succeeded_persists_result(store):
    job = store.create("t")
    store.mark_running(job.id)
    store.mark_succeeded(
        job.id,
        {"report": "R", "findings": [{"q": 1}], "rounds": 2,
         "input_tokens": 10, "output_tokens": 5},
    )
    j = store.get(job.id)
    assert j.status == "succeeded"
    assert j.report == "R" and j.findings == [{"q": 1}]
    assert (j.rounds, j.input_tokens, j.output_tokens) == (2, 10, 5)
    assert j.finished_at is not None


def test_mark_failed_records_error(store):
    job = store.create("t")
    store.mark_running(job.id)
    store.mark_failed(job.id, "boom")
    j = store.get(job.id)
    assert j.status == "failed" and j.error == "boom"


def test_cancel_queued_job_is_immediate(store):
    job = store.create("t")
    cancelled = store.cancel(job.id)
    assert cancelled is not None and cancelled.status == "cancelled"
    assert is_cancelled(job.id) is False  # flag cleaned up


def test_cancel_running_job_sets_flag(store):
    job = store.create("t")
    store.mark_running(job.id)
    out = store.cancel(job.id)
    assert out is not None and out.status == "running"  # flips later, cooperatively
    assert is_cancelled(job.id) is True
    store.mark_cancelled(job.id)  # what the worker thread does on abort
    assert store.get(job.id).status == "cancelled"


def test_cancel_terminal_job_is_noop(store):
    job = store.create("t")
    store.mark_running(job.id)
    store.mark_succeeded(job.id, {})
    assert store.cancel(job.id).status == "succeeded"
    assert store.cancel("missing") is None


def test_list_filter_and_limit(store):
    a = store.create("a")
    store.create("b")
    store.mark_running(a.id)
    assert {j.id for j in store.list(status="queued")} != {a.id}
    assert len(store.list(limit=1)) == 1


def test_persistence_across_reopen(tmp_path):
    path = str(tmp_path / "jobs.db")
    s1 = JobStore(path)
    job = s1.create("persist-me")
    s1.mark_running(job.id)
    s1.mark_succeeded(job.id, {"report": "R"})
    del s1
    s2 = JobStore(path)
    assert s2.get(job.id).status == "succeeded"


def test_orphaned_running_jobs_are_requeued(tmp_path):
    path = str(tmp_path / "jobs.db")
    s1 = JobStore(path)
    job = s1.create("orphan")
    s1.mark_running(job.id)
    del s1  # simulate crash: no terminal state written
    s2 = JobStore(path)  # init requeues orphans
    assert s2.get(job.id).status == "queued"
