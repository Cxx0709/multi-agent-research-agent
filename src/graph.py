"""Multi-agent research pipeline: supervisor -> parallel workers -> critic -> writer.

    ┌────────────┐
    │ supervisor │── 拆分子任务；被 critic 打回时根据 issues 补充派单
    └─────┬──────┘
          │ fan-out via Send
    ┌─────┴───────────────────┐
    ▼             ▼           ▼
 worker       worker       worker   (并行：搜索→抓取→摘要)
    └─────┬───────────────────┘
          │ fan-in
          ▼
    ┌──────────┐    issues     ┌────────────┐
    │  critic  │── 不通过 ───▶│ supervisor │ (rounds < max_rounds)
    └─────┬────┘              └────────────┘
          │ 通过 / 达到最大轮数
          ▼
    ┌──────────┐
    │  writer  │── 中文报告 + 参考来源
    └──────────┘
"""
from __future__ import annotations

import functools
import json
import logging
import operator
import re
import uuid
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Send
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from . import tools
from .config import get_llm, settings
from .jobs import is_cancelled

logger = logging.getLogger("research-agent.graph")


class JobCancelled(Exception):
    """Raised inside a node when the owning job was cancelled. Cooperative abort."""


def _guarded(fn):
    """Node wrapper: abort at node entry if the job was cancelled."""

    @functools.wraps(fn)
    def wrapper(state: ResearchState) -> dict:
        job_id = state.get("job_id") or ""
        if job_id and is_cancelled(job_id):
            raise JobCancelled(f"job {job_id} cancelled at node {fn.__name__}")
        return fn(state)

    return wrapper


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _llm_call(llm, prompt: str):
    """LLM call with exponential-backoff retry for transient failures."""
    return llm.invoke(prompt)


class Finding(TypedDict):
    question: str
    summary: str
    sources: list[str]


class ResearchState(TypedDict):
    topic: str
    job_id: str  # owning job; "" for ad-hoc runs. Used for cancel checks + checkpoint thread.
    question: str  # 单个 worker 的输入，由 Send 注入
    plan: list[str]
    findings: Annotated[list[Finding], operator.add]
    critic_issues: list[str]
    rounds: int
    report: str
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]


SUPERVISOR_PROMPT = """你是调研任务调度器。把下面的调研主题拆成 {n} 个以内具体、可独立检索的子问题。
每行一个，只输出编号列表，不要多余解释。

主题：{topic}
"""

REPLAN_PROMPT = """你是调研任务调度器。上一轮调研被评审打回，问题如下：
{issues}

请提出 {n} 个以内补充调研问题，专门弥补上述缺口。每行一个，只输出编号列表，不要多余解释。

主题：{topic}
"""

CRITIC_PROMPT = """你是挑剔的调研评审。检查下面的分节调研结果，找出：事实错误、缺少关键信息、引用缺失或不可靠的问题。
只输出 JSON，不要解释：
{{"approved": true/false, "issues": ["问题1", "问题2"]}}
approved 为 true 时 issues 为空数组。要求严格，有实质问题就不要放过。

调研主题：{topic}

分节结果：
{sections}
"""


def _parse_steps(text: str) -> list[str]:
    return [
        line.lstrip("0123456789.、) ").strip()
        for line in text.splitlines()
        if line.strip() and line.strip()[0].isdigit()
    ]


def _usage_meta(resp) -> tuple[int, int]:
    meta = getattr(resp, "usage_metadata", None) or {}
    return meta.get("input_tokens", 0), meta.get("output_tokens", 0)


def supervisor(state: ResearchState) -> dict:
    """拆分子任务；critic 打回后根据 issues 生成补充问题。"""
    llm = get_llm()
    n = settings.max_plan_steps
    if state["rounds"] == 0:
        prompt = SUPERVISOR_PROMPT.format(n=n, topic=state["topic"])
    else:
        issues = "\n".join(f"- {i}" for i in state["critic_issues"])
        prompt = REPLAN_PROMPT.format(n=n, topic=state["topic"], issues=issues)
    resp = _llm_call(llm, prompt)
    plan = _parse_steps(resp.content)[:n] or [state["topic"]]
    in_t, out_t = _usage_meta(resp)
    return {
        "plan": plan,
        "critic_issues": [],  # 新一轮开始，清空上一轮 issues
        "input_tokens": in_t,
        "output_tokens": out_t,
    }


def spawn_workers(state: ResearchState) -> list[Send]:
    """Fan-out: 每个子问题派给一个独立 worker 并行执行。"""
    return [Send("worker", {"question": q}) for q in state["plan"]]


def worker(state: ResearchState) -> dict:
    """单个调研员：搜索 → 抓取 → 摘要，产出一份 finding。"""
    question = state["question"]
    llm = get_llm()

    hits = tools.web_search(question)
    sources: list[str] = []
    snippets: list[str] = []
    for hit in hits[:3]:
        sources.append(hit["url"])
        body = tools.fetch_page(hit["url"]) if hit["url"] else hit["snippet"]
        snippets.append(f"来源：{hit['title']}（{hit['url']}）\n{body[:2000]}")

    prompt = (
        "根据以下网页资料回答问题，只陈述资料支持的事实，"
        "不确定的地方明确标注「未证实」。\n\n"
        f"问题：{question}\n\n资料：\n" + "\n\n---\n\n".join(snippets)
    )
    resp = _llm_call(llm, prompt)
    in_t, out_t = _usage_meta(resp)
    finding: Finding = {
        "question": question,
        "summary": resp.content,
        "sources": sources,
    }
    return {"findings": [finding], "input_tokens": in_t, "output_tokens": out_t}


def critic(state: ResearchState) -> dict:
    """评审所有 findings；不通过则给出 issues 打回补充调研。"""
    llm = get_llm(temperature=0.0)
    sections = "\n\n".join(
        f"### {f['question']}\n{f['summary']}\n来源：{', '.join(f['sources'])}"
        for f in state["findings"]
    )
    prompt = CRITIC_PROMPT.format(topic=state["topic"], sections=sections[:15000])
    resp = _llm_call(llm, prompt)
    in_t, out_t = _usage_meta(resp)

    issues: list[str] = []
    try:
        match = re.search(r"\{.*\}", resp.content, re.S)
        data = json.loads(match.group(0) if match else resp.content)
        if not data.get("approved", False):
            issues = [str(i) for i in data.get("issues", [])]
    except (json.JSONDecodeError, AttributeError):
        issues = [f"critic 输出无法解析，视为不通过：{resp.content[:200]}"]

    return {
        "critic_issues": issues,
        "rounds": state["rounds"] + 1,
        "input_tokens": in_t,
        "output_tokens": out_t,
    }


def route_after_critic(state: ResearchState) -> str:
    if not state["critic_issues"]:
        return "writer"
    if state["rounds"] >= settings.max_rounds:
        return "writer"  # 达到最大轮数，带着已知问题交付
    return "supervisor"


def writer(state: ResearchState) -> dict:
    llm = get_llm()
    sections = "\n\n".join(
        f"## {f['question']}\n\n{f['summary']}\n\n参考：\n"
        + "\n".join(f"- {u}" for u in f["sources"])
        for f in state["findings"]
    )
    prompt = (
        "你是一名中文科技写作者。根据以下分节调研结果，写一份结构清晰的调研报告：\n"
        "要求：开头 3 句话摘要；正文保留分节；结尾列出「参考来源」URL 列表；"
        "不要编造来源中没有的信息。\n\n"
        f"主题：{state['topic']}\n\n{sections}"
    )
    resp = _llm_call(llm, prompt)
    in_t, out_t = _usage_meta(resp)
    return {"report": resp.content, "input_tokens": in_t, "output_tokens": out_t}


_checkpointer_cache: dict[str, object] = {}
_checkpointer_cms: dict[str, object] = {}  # hold context managers for process lifetime


def get_checkpointer(path: str):
    """Cached SqliteSaver so graph state survives restarts (interrupt/resume).

    Note: ``from_conn_string`` is a context manager in checkpoint-sqlite v3+;
    we enter it once and hold it for the process lifetime because the graph
    needs the saver on every invoke.
    """
    if path not in _checkpointer_cache:
        from langgraph.checkpoint.sqlite import SqliteSaver

        cm = SqliteSaver.from_conn_string(path)
        _checkpointer_cms[path] = cm  # keep a ref so it is never GC-closed
        _checkpointer_cache[path] = cm.__enter__()
    return _checkpointer_cache[path]


def build_graph(checkpointer=None):
    g = StateGraph(ResearchState)
    g.add_node("supervisor", _guarded(supervisor))
    g.add_node("worker", _guarded(worker))
    g.add_node("critic", _guarded(critic))
    g.add_node("writer", _guarded(writer))
    g.set_entry_point("supervisor")
    g.add_conditional_edges("supervisor", spawn_workers, ["worker"])
    g.add_edge("worker", "critic")
    g.add_conditional_edges("critic", route_after_critic, ["writer", "supervisor"])
    g.add_edge("writer", END)
    return g.compile(checkpointer=checkpointer)


def run_research(topic: str, job_id: str = "", checkpointer=None) -> dict:
    """Run the full pipeline. Returns report + rounds + token accounting.

    When ``job_id`` is given, the run is checkpointed under that thread id
    (crash-resume) and nodes abort cooperatively if the job is cancelled.
    """
    graph = build_graph(checkpointer=checkpointer)
    thread_id = job_id or f"adhoc-{uuid.uuid4().hex[:8]}"
    try:
        final = graph.invoke(
            {
                "topic": topic,
                "job_id": job_id,
                "question": "",
                "plan": [],
                "findings": [],
                "critic_issues": [],
                "rounds": 0,
                "report": "",
                "input_tokens": 0,
                "output_tokens": 0,
            },
            config={"configurable": {"thread_id": thread_id}},
        )
    except JobCancelled:
        logger.info("job %s cancelled mid-run", job_id)
        return {
            "topic": topic,
            "cancelled": True,
            "report": "",
            "findings": [],
            "rounds": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
    return {
        "topic": topic,
        "cancelled": False,
        "report": final["report"],
        "findings": final["findings"],
        "rounds": final["rounds"],
        "input_tokens": final["input_tokens"],
        "output_tokens": final["output_tokens"],
    }
