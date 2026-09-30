"""Eval cases: stable, checkable research questions.

Replace/extend with questions from your own domain — the judge only needs
`key_points` to be facts a good report should contain.
"""
from __future__ import annotations

from typing import TypedDict


class EvalCase(TypedDict):
    id: str
    topic: str
    key_points: list[str]


CASES: list[EvalCase] = [
    {
        "id": "langgraph-basics",
        "topic": "LangGraph 的核心概念与典型用法",
        "key_points": [
            "基于 StateGraph / 图结构编排",
            "节点（nodes）与边（edges）",
            "支持循环、分支与条件边",
            "checkpoint 实现断点续跑与人机回环",
            "常用于多智能体系统",
        ],
    },
    {
        "id": "mcp-basics",
        "topic": "Model Context Protocol（MCP）是什么，能解决什么问题",
        "key_points": [
            "Anthropic 提出的开放协议",
            "统一模型与外部工具/数据源的交互方式",
            "Client-Server 架构",
            "替代各家私有的 function calling 封装",
        ],
    },
    {
        "id": "rag-eval",
        "topic": "RAG 系统常用的评测方法",
        "key_points": [
            "检索召回率 / 命中率",
            "答案忠实度（faithfulness）",
            "引用准确性",
            "常用框架如 RAGAS",
            "需要构造带标准答案的评测集",
        ],
    },
]
