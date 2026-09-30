# Research Agent · 多智能体协作调研助手 + 评测体系

用 LangGraph 实现的多智能体调研系统：supervisor 动态拆解任务并派单，多个 worker 并行检索撰写，critic 评审打回补充调研，最后 writer 成文。配套 **LLM-as-judge 评测流水线**，每次改 prompt 或换模型都能自动打分——这是面试官最爱问的"你的 agent 怎么评测、怎么兜底"的现成答案。

```
                ┌────────────┐
  topic ───────▶│ supervisor │── 拆分子任务；critic 打回时根据 issues 补充派单
                └─────┬──────┘
                      │ fan-out (Send)
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   ┌────────┐   ┌────────┐   ┌────────┐
   │ worker │   │ worker │   │ worker │  并行：搜索→抓取→摘要
   └───┬────┘   └───┬────┘   └───┬────┘
       └────────────┼────────────┘  fan-in
                    ▼
              ┌──────────┐    issues    ┌────────────┐
              │  critic  │── 不通过 ──▶│ supervisor │ (rounds < max_rounds)
              └─────┬────┘             └────────────┘
                    │ 通过 / 达到最大轮数
                    ▼
              ┌──────────┐
              │  writer  │── 中文报告 + 参考来源
              └──────────┘
  全链路 token 账本：每个节点只上报增量，用 operator.add 自动汇总
```

## 快速开始

```bash
cd project1-research-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # 填入 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
# 国内可直接用 DeepSeek / Qwen / Moonshot 的 OpenAI 兼容接口

# 跑一次调研
python -c "from src.graph import run_research; print(run_research('MCP 协议介绍')['report'])"

# 启动 API 服务（v0.2 起为异步任务 API）
uvicorn src.api:app --reload

# 提交任务 -> 202 返回 job_id；轮询直到 status=succeeded
curl -X POST localhost:8000/v1/research -H 'Content-Type: application/json' \
  -d '{"topic": "MCP 协议介绍"}'
# {"job_id": "a1b2c3d4e5f6", "status": "queued"}

curl localhost:8000/v1/jobs/a1b2c3d4e5f6
curl 'localhost:8000/v1/jobs?status=succeeded&limit=20'
curl -X DELETE localhost:8000/v1/jobs/a1b2c3d4e5f6   # 取消任务
curl localhost:8000/health                            # 存活 + 队列深度

# 跑评测（research → judge → 生成 reports/eval 报告）
python -m src.eval.run_eval

# 跑测试（含多智能体全链路干跑，无需 API key）
pip install pytest && python -m pytest tests/ -q

# Docker 部署
docker compose up --build
```

## 目录结构

```
src/
  config.py        # 全部配置走环境变量（pydantic-settings）
  tools.py         # 联网检索工具：Tavily（有 key 时）/ DuckDuckGo（免费兜底）
  graph.py         # LangGraph 多智能体图：supervisor→并行worker→critic→writer + token 记账
  jobs.py          # SQLite 任务持久化：queued/running/succeeded/failed/cancelled 状态机
  api.py           # FastAPI 异步任务 API：提交/查询/列表/取消 + 限流 + request-id
  eval/
    dataset.py     # 评测用例：topic + 期望要点
    judge.py       # LLM-as-judge：coverage / accuracy / citation 三维打分
    run_eval.py    # 一键评测，输出 JSON + Markdown 报告
tests/
  test_smoke.py    # 无需 API key 的冒烟测试
  test_graph_dryrun.py  # 多智能体全链路干跑（FakeLLM）
  test_jobs.py     # 任务状态机、取消语义、崩溃恢复
  test_cancel.py   # 运行中取消的协作式中断
  test_api_jobs.py # 异步 API：提交→轮询→结果、取消、超时、限流
```

## 生产级设计（v0.2）

一次调研可能跑好几分钟，所以 v0.2 把 API 改成了**异步任务模型**（breaking change，v0.1 的同步 `POST /v1/research` 已移除）：

```
POST /v1/research ──▶ 202 {job_id} ──▶ worker 线程池 ──▶ SQLite 落盘
                                              │
GET /v1/jobs/{id} ◀── 轮询 ◀──────────────────┘
```

生产化手段：

| 手段 | 实现 | 位置 |
|---|---|---|
| 任务持久化 | SQLite 状态机；重启时把"孤儿" running 任务重回 queued | `jobs.py` |
| 断点续跑 | LangGraph `SqliteSaver` checkpointer，按 job_id 做 thread | `graph.py` |
| 协作式取消 | `DELETE /v1/jobs/{id}` 置 flag，各节点入口检查后抛 `JobCancelled` | `jobs.py` + `graph.py` |
| 重试 | LLM 调用 tenacity 指数退避重试 3 次 | `graph.py::_llm_call` |
| 超时 | 单任务 `JOB_TIMEOUT_S`（默认 30min）超时标 failed；LLM 单次调用 `REQUEST_TIMEOUT_S` | `api.py` / `config.py` |
| 限流 | 按 IP 滑动窗口，默认 30 req/min | `api.py` |
| 可观测 | request-id 中间件（`X-Request-ID`）、结构化日志、`/health` 暴露队列深度 | `api.py` |
| 部署 | docker-compose 数据卷持久化 + healthcheck + restart 策略 | `docker-compose.yml` |

诚实说明的局限（面试被问到先自己说）：

1. **SQLite checkpointer 不是线程安全的**，所以 graph 执行被一把全局锁串行化了——`api.py` 里有注释。要真并发得换 Postgres checkpointer。
2. **超时是"软超时"**：标记 failed 后工作线程还会把当前节点跑完（和 Celery soft time limit 同一个 trade-off）。
3. 取消是**协作式**的：卡在单次 LLM/HTTP 调用里时，要等那次调用返回才能中断。

## 评测说明

`python -m src.eval.run_eval` 会对每个用例跑完整调研，再用 judge 模型按三个维度打分（1–5），同时记录 agent 的 token 消耗。报告落在 `src/eval/reports/`。换 prompt、换模型、改检索策略后重跑，对比分数即是迭代依据。

## 下一步 TODO（面试加分项）

- [x] 检索失败/网页抓取失败的降级策略 → 已加 tenacity 指数退避重试（v0.2）
- [x] 异步任务 API + 任务持久化 + 取消/超时（v0.2）
- [ ] 接入 Langfuse 做全链路 trace 可视化
- [ ] 给 writer 加"引用必须来自 sources"的后校验
- [ ] 把 checkpointer 换成 Postgres，拿掉全局执行锁，实现真并发
- [ ] 把评测用例换成你目标行业的真实问题

## 面试时可以讲的点

1. 为什么是 supervisor + workers 而不是流水线：子任务可独立检索，并行天然加速；supervisor 只做调度，职责分离后每个 prompt 都更短更稳。
2. critic 闭环怎么收敛：issues 驱动补充调研 + max_rounds 兜底，避免无限打回；评测报告里 rounds 列直接体现"自我修正"能力。
3. 并行 fan-out/fan-in 的状态合并：findings 和 token 用 `operator.add` 做 reducer 增量汇总，避免多分支互相覆盖。
4. 评测体系怎么设计的：自建数据集 + LLM-as-judge 三维打分 + token 成本账本，改 prompt 有数据支撑。
5. 生产化取舍：为什么用 SQLite 而不是 Redis/Celery——单机作品集场景下零运维、重启可恢复；同时诚实说明它的天花板（checkpointer 线程安全问题），以及怎么演进（换 Postgres）。
