"""
app/core/config.py
==================
全局配置模块：读取 .env 环境变量并通过 Pydantic Settings 进行类型校验。
所有模块均应从此处导入 `settings` 单例，禁止直接 os.getenv 散落各处。
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    应用全局配置，字段与 .env.example 一一对应。
    使用 lru_cache 保证全局只实例化一次（单例模式）。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # 忽略 .env 中多余的键，避免报错
    )

    # ── LLM / Embedding API ──────────────────────────────────
    my_api_key: str = "your_api_key_here"
    my_api_base: str = "https://api.chat.csu.edu.cn/v1"
    my_model: str = "deepseek-v3-thinking"
    my_emb_model: str = "bge-m3"
    my_embedding_model: str = "bge-m3"

    # ── Milvus ───────────────────────────────────────────────
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "rag_memory"
    milvus_dim: int = 1024  # 向量维度，需与 Embedding 模型匹配

    # ── Reranker ─────────────────────────────────────────────
    reranker_url: str = "http://localhost:8080/rerank"

    # ── 记忆管理 ─────────────────────────────────────────────
    memory_window_size: int = 10          # 滑动窗口保留的对话轮数
    memory_compress_threshold: int = 20   # 触发压缩的历史轮数阈值

    # ── 检索 ─────────────────────────────────────────────────
    retrieval_top_k: int = 10   # 向量召回数量
    rerank_top_n: int = 5       # 重排后保留数量

    # ── 路径 ─────────────────────────────────────────────────
    data_dir: str = "./data"
    log_dir: str = "./logs"

    # ── 服务 ─────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    获取全局配置单例（通过 lru_cache 保证只初始化一次）。

    Returns:
        Settings: 已从 .env 加载的全局配置对象。
    """
    return Settings()


# 模块级单例，供其他模块直接 `from app.core.config import settings` 使用
settings = get_settings()
