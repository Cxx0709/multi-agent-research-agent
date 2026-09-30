"""Dry-run of the full multi-agent graph with a fake LLM and fake web tools.

No API keys, no network. Validates the orchestration itself:
fan-out via Send, reducer merging, critic revision loop, token accounting.
"""
import pytest

from src import graph as gmod
from src.graph import run_research


class FakeMsg:
    def __init__(self, content: str):
        self.content = content
        self.usage_metadata = {"input_tokens": 10, "output_tokens": 5}


class FakeLLM:
    """Dispatches canned responses by prompt content; critic rejects once."""

    def __init__(self):
        self.critic_calls = 0

    def invoke(self, prompt: str) -> FakeMsg:
        if "评审" in prompt:  # critic
            self.critic_calls += 1
            if self.critic_calls == 1:
                return FakeMsg('{"approved": false, "issues": ["缺少引用来源"]}')
            return FakeMsg('{"approved": true, "issues": []}')
        if "补充调研问题" in prompt:  # supervisor 回合 2+
            return FakeMsg("1. 补充问题A")
        if "拆成" in prompt:  # supervisor 首轮
            return FakeMsg("1. 子问题A\n2. 子问题B")
        if "写作者" in prompt:  # writer
            return FakeMsg("# 报告\n\n正文")
        return FakeMsg("摘要内容")  # worker


@pytest.fixture
def fake_env(monkeypatch):
    monkeypatch.setattr(gmod, "get_llm", lambda *a, **k: FakeLLM())
    monkeypatch.setattr(
        gmod.tools,
        "web_search",
        lambda q, max_results=None: [
            {"title": "t1", "url": "http://x/1", "snippet": "s1"},
            {"title": "t2", "url": "http://x/2", "snippet": "s2"},
        ],
    )
    monkeypatch.setattr(
        gmod.tools, "fetch_page", lambda url, max_chars=6000: "页面正文"
    )


def test_full_run_with_critic_revision(fake_env):
    result = run_research("测试主题")
    assert result["report"].startswith("# 报告")
    # 首轮 2 个 worker + critic 打回后 1 个补充 worker
    assert len(result["findings"]) == 3
    assert result["rounds"] == 2  # critic 跑了两轮
    assert result["input_tokens"] > 0 and result["output_tokens"] > 0
    # token 账本：8 次 LLM 调用 × (10 in / 5 out)
    assert result["input_tokens"] == 80
    assert result["output_tokens"] == 40
