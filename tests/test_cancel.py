"""Graph-level cancellation: a cancelled job aborts at the next node boundary."""
import pytest

from src import graph as gmod
from src.graph import run_research
from src import jobs


class FakeMsg:
    def __init__(self, content: str):
        self.content = content
        self.usage_metadata = {"input_tokens": 1, "output_tokens": 1}


class FakeLLM:
    def invoke(self, prompt: str):
        if "拆成" in prompt:
            return FakeMsg("1. 子问题A")
        return FakeMsg("摘要")


@pytest.fixture
def fake_env(monkeypatch):
    monkeypatch.setattr(gmod, "get_llm", lambda *a, **k: FakeLLM())
    monkeypatch.setattr(
        gmod.tools, "web_search",
        lambda q, max_results=None: [{"title": "t", "url": "", "snippet": "s"}],
    )
    yield
    jobs._CANCEL_EVENTS.clear()


def test_cancelled_job_aborts(fake_env):
    jobs._set_cancelled("job-1").set()  # cancel before the run starts
    result = run_research("主题", job_id="job-1")
    assert result["cancelled"] is True
    assert result["report"] == ""


def test_uncancelled_job_runs_through(fake_env):
    result = run_research("主题", job_id="job-2")
    assert result["cancelled"] is False
    assert result["report"]  # worker summary flows through


def test_checkpointer_is_real_saver(tmp_path):
    from langgraph.checkpoint.sqlite import SqliteSaver

    from src.graph import build_graph, get_checkpointer

    cp = get_checkpointer(str(tmp_path / "ckpt.db"))
    assert isinstance(cp, SqliteSaver)
    assert get_checkpointer(str(tmp_path / "ckpt.db")) is cp  # cached
    assert build_graph(checkpointer=cp) is not None
