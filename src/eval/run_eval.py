"""Run the eval set: research -> judge -> markdown report.

Usage (from project root):
    python -m src.eval.run_eval
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ..graph import run_research
from .dataset import CASES
from .judge import judge

OUT_DIR = Path(__file__).resolve().parent / "reports"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    rows: list[dict] = []
    for case in CASES:
        print(f"[{case['id']}] researching: {case['topic']}", flush=True)
        result = run_research(case["topic"])
        print(f"[{case['id']}] judging...", flush=True)
        score = judge(case["id"], case["key_points"], result["report"])
        rows.append(
            {
                **score,
                "topic": case["topic"],
                "rounds": result["rounds"],
                "agent_input_tokens": result["input_tokens"],
                "agent_output_tokens": result["output_tokens"],
            }
        )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (OUT_DIR / f"results-{stamp}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Eval 报告",
        "",
        f"时间：{stamp}，用例数：{len(rows)}",
        "",
        "| 用例 | coverage | accuracy | citation | rounds | agent tokens | 点评 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        toks = r["agent_input_tokens"] + r["agent_output_tokens"]
        lines.append(
            f"| {r['case_id']} | {r['coverage']} | {r['accuracy']} | "
            f"{r['citation']} | {r['rounds']} | {toks} | {r['comment']} |"
        )

    def avg(k: str) -> float:
        vals = [r[k] for r in rows if isinstance(r[k], (int, float))]
        return sum(vals) / len(vals) if vals else 0.0

    lines += [
        "",
        f"平均分：coverage {avg('coverage'):.2f} / "
        f"accuracy {avg('accuracy'):.2f} / citation {avg('citation'):.2f}",
    ]
    report = "\n".join(lines)
    (OUT_DIR / f"report-{stamp}.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
