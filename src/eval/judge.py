"""LLM-as-judge: scores a report against the case's key points."""
from __future__ import annotations

import json
import re

from ..config import get_llm, settings

JUDGE_PROMPT = """你是严格的评测员。根据下面的期望要点给调研报告打分（1-5 分）。
只输出 JSON，不要解释：
{{"coverage": <要点覆盖度>, "accuracy": <事实准确性>, "citation": <引用规范性>, "comment": "<一句话点评>"}}

期望要点：
{key_points}

调研报告：
{report}
"""


def judge(case_id: str, key_points: list[str], report: str) -> dict:
    llm = get_llm(model=settings.judge_model or settings.llm_model, temperature=0.0)
    prompt = JUDGE_PROMPT.format(
        key_points="\n".join(f"- {p}" for p in key_points),
        report=report[:12000],
    )
    resp = llm.invoke(prompt)
    text = resp.content.strip()
    match = re.search(r"\{.*\}", text, re.S)
    try:
        score = json.loads(match.group(0) if match else text)
    except json.JSONDecodeError:
        score = {
            "coverage": 0,
            "accuracy": 0,
            "citation": 0,
            "comment": f"judge 输出无法解析: {text[:200]}",
        }
    score["case_id"] = case_id
    meta = getattr(resp, "usage_metadata", None) or {}
    score["judge_input_tokens"] = meta.get("input_tokens", 0)
    score["judge_output_tokens"] = meta.get("output_tokens", 0)
    return score
