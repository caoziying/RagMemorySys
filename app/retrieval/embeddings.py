"""
app/retrieval/embeddings.py
===========================
向量化服务模块：将文本转换为向量表示，供 Milvus 存储与检索使用。

⚠️ 严格遵循规范：
  - 使用自定义 API_BASE_URL（读取 .env），不使用 OpenAI 默认地址。
  - 支持批量向量化以提升效率。
"""

import os
from typing import List

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.exceptions import EmbeddingError
from app.core.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────
# 客户端实例化（严格按规范）
# ──────────────────────────────────────────────────────────────

MY_API_KEY: str = settings.my_api_key
API_BASE_URL: str = settings.my_api_base
CHAT_MODEL_NAME: str = settings.my_emb_model
EMBEDDING_MODEL_NAME: str = settings.my_embedding_model

logger.info(
    "Embedding 服务初始化 | base_url={} | model={}",
    API_BASE_URL, EMBEDDING_MODEL_NAME,
)

# 使用自定义 base_url，绝不使用 OpenAI 默认地址
client: OpenAI = OpenAI(
    api_key=MY_API_KEY,
    base_url=API_BASE_URL,
)

# 批量向量化的最大批次大小（避免单次请求过大）
BATCH_SIZE = 32


# ──────────────────────────────────────────────────────────────
# 核心向量化函数
# ──────────────────────────────────────────────────────────────

def get_embeddings(test_texts: List[str]) -> list:
    """
    【规范实现】对文本列表进行向量化，返回原始 API 响应对象。

    Args:
        test_texts: 待向量化的文本列表。

    Returns:
        OpenAI Embeddings API 响应对象。
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL_NAME,
        input=test_texts,
    )
    return response


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    将文本列表向量化，返回向量浮点数列表。
    支持批量处理，内置重试机制（最多 3 次，指数退避）。

    Args:
        texts: 待向量化的文本列表（不得为空）。

    Returns:
        每个文本对应的向量列表，维度与配置的 MILVUS_DIM 一致。

    Raises:
        EmbeddingError: 在所有重试失败后抛出。
    """
    if not texts:
        return []

    all_embeddings: List[List[float]] = []

    # 按批次处理，避免单次请求过大
    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch = texts[batch_start : batch_start + BATCH_SIZE]
        logger.debug(
            "向量化批次 | batch [{}/{}] | texts={}",
            batch_start // BATCH_SIZE + 1,
            (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE,
            len(batch),
        )

        try:
            response = get_embeddings(batch)
            # 解析响应：按 index 排序确保顺序与输入一致
            sorted_data = sorted(response.data, key=lambda x: x.index)
            batch_embeddings = [item.embedding for item in sorted_data]
            all_embeddings.extend(batch_embeddings)

        except Exception as e:
            logger.error("Embedding 调用失败 | batch_start={} | error={}", batch_start, e)
            raise EmbeddingError(detail=f"Embedding API 调用失败: {e}")

    logger.debug("向量化完成 | texts={} | dim={}", len(texts), len(all_embeddings[0]) if all_embeddings else 0)
    return all_embeddings


def embed_single(text: str) -> List[float]:
    """
    对单条文本进行向量化，是 embed_texts 的便捷封装。

    Args:
        text: 待向量化的单条文本。

    Returns:
        向量浮点数列表。

    Raises:
        EmbeddingError: 向量化失败时抛出。
    """
    results = embed_texts([text])
    if not results:
        raise EmbeddingError(detail="向量化结果为空")
    return results[0]
