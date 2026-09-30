"""Smoke tests that need no API keys: config loads, graph compiles."""
from src.config import settings
from src.graph import build_graph


def test_settings_defaults():
    assert settings.max_plan_steps > 0
    assert settings.max_rounds >= 1
    assert settings.llm_model


def test_graph_compiles():
    assert build_graph() is not None


def test_spawn_workers_fanout():
    from langgraph.types import Send

    from src.graph import spawn_workers

    sends = spawn_workers({"plan": ["q1", "q2", "q3"]})  # 只读 plan，无需完整 state
    assert len(sends) == 3
    assert all(isinstance(s, Send) and s.node == "worker" for s in sends)
    assert [s.arg["question"] for s in sends] == ["q1", "q2", "q3"]


def test_route_after_critic():
    from src.graph import route_after_critic

    base = {"critic_issues": [], "rounds": 1}
    assert route_after_critic(base) == "writer"  # 通过
    assert route_after_critic({"critic_issues": ["x"], "rounds": 99}) == "writer"  # 超轮数
    assert (
        route_after_critic({"critic_issues": ["缺引用"], "rounds": 0}) == "supervisor"
    )  # 打回


def test_judge_prompt_renders():
    from src.eval.judge import JUDGE_PROMPT

    text = JUDGE_PROMPT.format(key_points="- a", report="b")
    assert "- a" in text and "b" in text
