"""
app/retrieval/reranker.py
=========================
重排与降级策略模块。

重排策略（按优先级降级）：
  Level 1（主路）：调用 RERANKER_URL 指定的本地 HTTP 重排接口（高精度）。
  Level 2（备用）：调用内部 Embedding API 计算余弦相似度重排（中等精度）。
  Level 3（兜底）：直接返回原始召回顺序，不进行重排。

⚠️ 任何一级失败均自动切换到下一级，不会中断主线程。
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI

from app.core.config import settings
from app.core.exceptions import RerankerError
from app.core.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────
# 内部 Embedding 客户端（用于降级重排）
# ──────────────────────────────────────────────────────────────

MY_API_KEY: str = settings.my_api_key
API_BASE_URL: str = settings.my_api_base
CHAT_MODEL_NAME: str = settings.my_emb_model
EMBEDDING_MODEL_NAME: str = settings.my_embedding_model

# 降级用 Embedding 客户端
_embed_client: OpenAI = OpenAI(
    api_key=MY_API_KEY,
    base_url=API_BASE_URL,
)

# 本地重排服务地址（通过 .env 配置，默认 localhost）
RERANKER_URL: str = os.getenv("RERANKER_URL", "http://host.docker.internal:8877/rerank")

logger.info("Reranker 初始化 | reranker_url={} | embed_model={}", RERANKER_URL, EMBEDDING_MODEL_NAME)


# ──────────────────────────────────────────────────────────────
# 主入口：级联重排
# ──────────────────────────────────────────────────────────────

def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    对候选结果进行重排，返回 Top-N 个最相关条目。

    级联策略：本地 Reranker HTTP → Embedding 余弦相似度 → 原始顺序。

    Args:
        query:      用户查询文本。
        candidates: 候选结果列表，每项必须包含 "content" 字段。
        top_n:      最终返回的结果数量。

    Returns:
        重排后的 Top-N 候选列表，按相关性降序排列。
    """
    if not candidates:
        return []

    texts = [c.get("content", "") for c in candidates]

    # Level 1: 本地 Reranker HTTP 服务（主路，首选）
    ranked_indices = _try_local_rerank(query=query, texts=texts)
    if ranked_indices is not None:
        logger.debug("使用本地 Reranker HTTP 重排 | candidates={}", len(candidates))
        return _apply_ranking(candidates, ranked_indices, top_n)

    # Level 2: 内部 Embedding 余弦相似度重排（备用）
    ranked_indices = _try_embedding_rerank(query=query, texts=texts)
    if ranked_indices is not None:
        logger.debug("使用 Embedding 余弦重排（降级）| candidates={}", len(candidates))
        return _apply_ranking(candidates, ranked_indices, top_n)

    # Level 3: 直接返回原始顺序（兜底）
    logger.warning("所有重排策略失败，返回原始召回顺序（兜底降级）")
    return candidates[:top_n]


# ──────────────────────────────────────────────────────────────
# Level 1: 本地 Reranker HTTP 接口
# ──────────────────────────────────────────────────────────────

# def _try_local_rerank(query: str, texts: List[str]) -> Optional[List[int]]:
#     """
#     调用本地/自部署的 Reranker HTTP 服务进行重排。

#     请求格式：{"query": "...", "texts": ["...", "..."]}
#     响应格式：[{"index": 2, "score": 0.99}, {"index": 0, "score": 0.39}]

#     Args:
#         query: 查询文本。
#         texts: 候选文本列表。

#     Returns:
#         按相关性降序排列的原始索引列表；失败时返回 None 触发降级。
#     """
#     url = RERANKER_URL
#     payload = {"query": query, "texts": texts}
#     headers = {"Content-Type": "application/json"}

#     try:
#         response = requests.post(
#             url,
#             headers=headers,
#             data=json.dumps(payload, ensure_ascii=False),
#             timeout=10,
#         )
#         response.raise_for_status()
#         result = response.json()
#         # 响应格式: [{"index": 2, "score": 0.99}, {"index": 0, "score": 0.39}]
#         sorted_result = sorted(result, key=lambda x: x.get("score", 0), reverse=True)
#         indices = [item["index"] for item in sorted_result]
#         logger.debug(
#             "本地 Reranker 重排成功 | top_score={:.4f}",
#             sorted_result[0].get("score", 0) if sorted_result else 0,
#         )
#         return indices

#     except requests.exceptions.ConnectionError:
#         logger.warning("本地 Reranker 连接失败，触发降级 | url={}", url)
#     except requests.exceptions.Timeout:
#         logger.warning("本地 Reranker 调用超时（>10s），触发降级 | url={}", url)
#     except Exception as e:
#         logger.warning("本地 Reranker 异常，触发降级 | error={}", str(e))

#     return None

def _try_local_rerank(query: str, texts: List[str]) -> Optional[List[int]]:
    """
    调用本地/自部署的 Reranker HTTP 服务进行重排。
    """
    url = RERANKER_URL
    
    # 过滤掉可能的空字符串，有些 Reranker 遇到空字符串会报 400 错误
    valid_texts = [t for t in texts if t.strip()]
    if not valid_texts:
        return None

    payload = {"query": query, "texts": valid_texts}

    try:
        # 直接使用 json=payload，不再需要手动 json.dumps 和声明 Header
        response = requests.post(
            url,
            json=payload,
            timeout=10,
        )
        
        # 如果不是 200，这行会抛出 HTTPError
        response.raise_for_status() 
        
        result = response.json()
        
        # 兼容不同 Reranker 的返回格式
        # 假设响应格式为: [{"index": 2, "score": 0.99}, {"index": 0, "score": 0.39}]
        sorted_result = sorted(result, key=lambda x: x.get("score", 0), reverse=True)
        indices = [item["index"] for item in sorted_result]
        
        logger.debug(
            "本地 Reranker 重排成功 | top_score={:.4f}",
            sorted_result[0].get("score", 0) if sorted_result else 0,
        )
        logger.warning(
            "本地 Reranker 重排成功 | top_score={:.4f}",
            sorted_result[0].get("score", 0) if sorted_result else 0,
        )
        return indices

    except requests.exceptions.HTTPError as e:
        # 捕获 HTTP 错误，并把服务端真正的报错信息（response.text）打印出来
        logger.warning("本地 Reranker HTTP 错误，触发降级 | url={} | status={} | detail={}", 
                       url, e.response.status_code, e.response.text)
    except requests.exceptions.ConnectionError:
        logger.warning("本地 Reranker 连接失败，触发降级 | url={}", url)
    except requests.exceptions.Timeout:
        logger.warning("本地 Reranker 调用超时（>10s），触发降级 | url={}", url)
    except Exception as e:
        logger.warning("本地 Reranker 异常，触发降级 | error={}", str(e))

    return None

# ──────────────────────────────────────────────────────────────
# Level 2: 内部 Embedding 余弦相似度重排
# ──────────────────────────────────────────────────────────────

def _try_embedding_rerank(query: str, texts: List[str]) -> Optional[List[int]]:
    """
    使用 Embedding API 计算 query 与各候选文本的余弦相似度，并按相似度降序排列。

    Args:
        query: 查询文本。
        texts: 候选文本列表。

    Returns:
        按余弦相似度降序排列的原始索引列表；失败时返回 None。
    """
    try:
        all_texts = [query] + texts
        response = _embed_client.embeddings.create(
            model=EMBEDDING_MODEL_NAME,
            input=all_texts,
        )
        # 解析向量，按 index 排序确保顺序正确
        sorted_data = sorted(response.data, key=lambda x: x.index)
        embeddings = [item.embedding for item in sorted_data]

        query_vec = embeddings[0]
        candidate_vecs = embeddings[1:]

        # 计算余弦相似度
        scores: List[Tuple[int, float]] = []
        for idx, vec in enumerate(candidate_vecs):
            score = _cosine_similarity(query_vec, vec)
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        indices = [idx for idx, _ in scores]
        logger.debug("Embedding 余弦重排成功 | top_score={:.4f}", scores[0][1] if scores else 0)
        return indices

    except Exception as e:
        logger.warning("Embedding 降级重排失败 | error={}", str(e))
        return None


# ──────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────

def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    计算两个向量的余弦相似度。

    Args:
        vec_a: 向量 A。
        vec_b: 向量 B。

    Returns:
        余弦相似度值，范围 [-1, 1]。
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a ** 2 for a in vec_a) ** 0.5
    norm_b = sum(b ** 2 for b in vec_b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _apply_ranking(
    candidates: List[Dict[str, Any]],
    ranked_indices: List[int],
    top_n: int,
) -> List[Dict[str, Any]]:
    """
    根据重排索引列表对候选结果重新排序并截取 Top-N。

    Args:
        candidates:     原始候选列表。
        ranked_indices: 按相关性降序的原始索引列表。
        top_n:          截取的最终数量。

    Returns:
        重排后的 Top-N 候选列表。
    """
    reranked = []
    for idx in ranked_indices[:top_n]:
        if 0 <= idx < len(candidates):
            reranked.append(candidates[idx])
    return reranked