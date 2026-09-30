"""Central configuration — everything comes from environment variables."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM (OpenAI-compatible endpoint; point base_url at DeepSeek/Qwen/etc. if you like)
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = "sk-your-key-here"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2

    # Optional: higher-quality search via Tavily. Falls back to free DuckDuckGo.
    tavily_api_key: str = ""

    # Agent behaviour
    max_plan_steps: int = 5
    max_search_results: int = 5
    max_rounds: int = 2  # critic 打回补充调研的最大轮数
    request_timeout_s: int = 30

    # Eval judge model (defaults to the main LLM when empty)
    judge_model: str = ""

    # --- production: async jobs -------------------------------------
    job_db_path: str = "data/jobs.db"          # SQLite job store (durable)
    checkpoint_db_path: str = "data/checkpoints.db"  # LangGraph checkpointer
    max_concurrent_jobs: int = 4               # worker thread pool size
    job_timeout_s: int = 1800                  # 30 min per job, then marked failed
    rate_limit_per_min: int = 30               # simple per-IP sliding window


settings = Settings()


def get_llm(model: str = "", temperature: float | None = None):
    """Build a chat model. Imported lazily so `import src.config` needs no keys."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=model or settings.llm_model,
        temperature=settings.llm_temperature if temperature is None else temperature,
        request_timeout=settings.request_timeout_s,
    )
