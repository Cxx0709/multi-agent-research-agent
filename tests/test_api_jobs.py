"""API tests: async submit -> poll -> result, cancel, errors. No network, no LLM."""
import time

import pytest
from fastapi.testclient import TestClient

from src import api
from src import jobs as jobs_mod
from src.config import settings


def _wait_for(client, job_id, want=("succeeded", "failed", "cancelled"), timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/v1/jobs/{job_id}")
        assert r.status_code == 200
        if r.json()["status"] in want:
            return r.json()
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} did not reach {want} in {timeout}s")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "job_db_path", str(tmp_path / "jobs.db"))
    monkeypatch.setattr(settings, "checkpoint_db_path", str(tmp_path / "ckpt.db"))
    monkeypatch.setattr(settings, "max_concurrent_jobs", 2)
    monkeypatch.setattr(settings, "rate_limit_per_min", 1000)
    api._store = None
    api._pool = None

    def fast_fake(topic, job_id):
        return {
            "topic": topic, "cancelled": False, "report": "# 报告",
            "findings": [], "rounds": 1, "input_tokens": 3, "output_tokens": 2,
        }

    monkeypatch.setattr(api, "_invoke_guarded", fast_fake)
    with TestClient(api.app) as c:
        yield c
    jobs_mod._CANCEL_EVENTS.clear()


def test_submit_and_poll_to_succeeded(client):
    r = client.post("/v1/research", json={"topic": "量子计算"})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    assert r.json()["status"] == "queued"
    assert "X-Request-ID" in r.headers

    job = _wait_for(client, job_id, want=("succeeded",))
    assert job["report"] == "# 报告"
    assert job["rounds"] == 1 and job["input_tokens"] == 3


def test_get_missing_job_404(client):
    assert client.get("/v1/jobs/nope").status_code == 404
    assert client.delete("/v1/jobs/nope").status_code == 404


def test_validation_rejects_short_topic(client):
    assert client.post("/v1/research", json={"topic": "x"}).status_code == 422


def test_list_jobs_filter(client):
    r1 = client.post("/v1/research", json={"topic": "主题A"}).json()
    _wait_for(client, r1["job_id"], want=("succeeded",))
    r = client.get("/v1/jobs", params={"status": "succeeded"})
    assert r.status_code == 200
    assert r1["job_id"] in {j["id"] for j in r.json()["jobs"]}
    assert client.get("/v1/jobs", params={"status": "bogus"}).status_code == 400


def test_cancel_running_job(client, monkeypatch):
    def slow_fake(topic, job_id):
        for _ in range(100):
            if jobs_mod.is_cancelled(job_id):
                return {"cancelled": True}
            time.sleep(0.05)
        return {"cancelled": False, "report": "R", "findings": [],
                "rounds": 0, "input_tokens": 0, "output_tokens": 0}

    monkeypatch.setattr(api, "_invoke_guarded", slow_fake)
    job_id = client.post("/v1/research", json={"topic": "慢任务"}).json()["job_id"]
    time.sleep(0.5)  # let the worker pick it up
    r = client.delete(f"/v1/jobs/{job_id}")
    assert r.status_code == 200
    job = _wait_for(client, job_id, want=("cancelled",))
    assert job["status"] == "cancelled"


def test_job_timeout_marks_failed(client, monkeypatch):
    monkeypatch.setattr(settings, "job_timeout_s", 1)
    monkeypatch.setattr(api, "_invoke_guarded", lambda t, j: (time.sleep(3), {"cancelled": False})[1])
    job_id = client.post("/v1/research", json={"topic": "超时任务"}).json()["job_id"]
    job = _wait_for(client, job_id, want=("failed",), timeout=10)
    assert "timed out" in job["error"]


def test_health_reports_queue_depth(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["db"] == "ok"
    assert body["queued"] >= 0 and body["running"] >= 0
