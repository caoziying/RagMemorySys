"""
app/retrieval/retriever.py
==========================
核心检索调度器：串联向量召回、重排与降级策略的完整 RAG 检索流程。

工作流：
  retrieve()：
    1. 将 query 向量化（embeddings.py）
    2. 从 Milvus 按 user_id 过滤进行 ANN 检索（milvus_client.py）
    3. 对召回结果进行重排（reranker.py，带三级降级）
    4. 返回 RetrievedChunk 列表（schemas.py）

  store()：
    1. 对输入文本进行分块（chunking.py）
    2. 批量向量化（embeddings.py）
    3. 写入 Milvus（milvus_client.py）
"""

import asyncio
from datetime import datetime, timezone
from typing import List

from app.api.schemas import RetrievedChunk
from app.core.config import settings
from app.core.exceptions import EmbeddingError, MilvusUnavailableError
from app.core.logger import get_logger
from app.retrieval import embeddings as emb_module
from app.retrieval import reranker as reranker_module
from app.retrieval.chunking import TextChunker
from app.retrieval.milvus_client import MilvusClient

logger = get_logger(__name__)


class Retriever:
    """
    RAG 核心调度器。

    将 Embedding、Milvus 检索、Reranker 三个子模块串联为完整的
    向量检索流水线，并暴露统一的 retrieve() 和 store() 接口。

    所有异常均被捕获并降级，确保主请求链路的稳定性。
    """

    def __init__(self) -> None:
        self._milvus = MilvusClient()
        self._chunker = TextChunker(
            chunk_size=512,
            chunk_overlap=64,
            min_chunk_size=10,  # 对话消息通常较短，降低过滤阈值避免丢弃有效内容
        )
        self._top_k = settings.retrieval_top_k
        self._rerank_top_n = settings.rerank_top_n

        # 尝试在初始化时连接 Milvus
        self._milvus.connect()
        logger.info(
            "Retriever 初始化完成 | top_k={} | rerank_top_n={}",
            self._top_k, self._rerank_top_n,
        )

    # ──────────────────────────────────────────────────────────
    # 公共接口：检索
    # ──────────────────────────────────────────────────────────

    async def retrieve(
        self,
        user_id: str,
        query: str,
    ) -> List[RetrievedChunk]:
        """
        执行完整的 RAG 检索流程，返回重排后的相关记忆切片。

        流程：query → 向量化 → Milvus ANN 检索 → Rerank → TopN

        Args:
            user_id: 用户 ID，用于多租户数据隔离。
            query:   用户当前的查询文本。

        Returns:
            RetrievedChunk 列表，按相关性降序排列。
            若任何步骤失败，返回降级结果（可能为空列表）。
        """
        logger.info("开始检索 | user_id={} | query[:50]={}", user_id, query[:50])

        # Step 1: 向量化 query
        query_vector = await self._embed_query(query)
        if query_vector is None:
            logger.warning("query 向量化失败，返回空检索结果 | user_id={}", user_id)
            return []

        # Step 2: Milvus ANN 检索（多租户隔离）
        raw_hits = await self._milvus_search(
            user_id=user_id,
            query_vector=query_vector,
        )
        if not raw_hits:
            logger.info("Milvus 无相关历史记忆 | user_id={}", user_id)
            return []

        # Step 3: 重排（带三级降级）
        reranked = await self._rerank(query=query, candidates=raw_hits)

        # Step 4: 转换为 RetrievedChunk 格式
        chunks = [
            RetrievedChunk(
                content=hit.get("content", ""),
                score=hit.get("score", 0.0),
                source="milvus",
                metadata={
                    "timestamp": hit.get("timestamp", ""),
                    "user_id": hit.get("user_id", ""),
                    "id": hit.get("id", ""),
                },
            )
            for hit in reranked
        ]

        logger.info("检索完成 | user_id={} | chunks={}", user_id, len(chunks))
        return chunks

    # ──────────────────────────────────────────────────────────
    # 公共接口：存储
    # ──────────────────────────────────────────────────────────

    async def store(
        self,
        user_id: str,
        texts: List[str],
        timestamp: datetime,
    ) -> int:
        """
        将文本列表分块、向量化并存入 Milvus。

        流程：texts → 分块 → 向量化 → Milvus 插入

        Args:
            user_id:   用户 ID。
            texts:     待存储的文本列表。
            timestamp: 存储时间戳。

        Returns:
            成功存入 Milvus 的切片数量；失败时返回 0（降级）。
        """
        logger.info("开始存储记忆 | user_id={} | texts={}", user_id, len(texts))

        # Step 1: 文本分块
        chunks = self._chunker.chunk_texts(texts)
        if not chunks:
            logger.warning("分块结果为空，跳过存储 | user_id={}", user_id)
            return 0

        chunk_contents = [c.content for c in chunks]
        ts_str = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Step 2: 批量向量化
        embeddings = await self._embed_texts(chunk_contents)
        if embeddings is None or len(embeddings) != len(chunk_contents):
            logger.warning("向量化失败或数量不匹配，跳过 Milvus 存储 | user_id={}", user_id)
            return 0

        # Step 3: 写入 Milvus
        stored = await self._milvus_insert(
            user_id=user_id,
            contents=chunk_contents,
            embeddings=embeddings,
            timestamp=ts_str,
        )

        logger.info("记忆存储完成 | user_id={} | chunks_stored={}", user_id, stored)
        return stored

    # ──────────────────────────────────────────────────────────
    # 内部方法：各步骤的异步封装与异常隔离
    # ──────────────────────────────────────────────────────────

    async def _embed_query(self, query: str) -> list | None:
        """异步向量化单条 query，失败返回 None。"""
        loop = asyncio.get_event_loop()
        try:
            vec = await loop.run_in_executor(None, emb_module.embed_single, query)
            return vec
        except EmbeddingError as e:
            logger.error("query 向量化失败 | error={}", e)
            return None
        except Exception as e:
            logger.error("query 向量化未预期异常 | error={}", e)
            return None

    async def _embed_texts(self, texts: List[str]) -> list | None:
        """异步批量向量化文本列表，失败返回 None。"""
        loop = asyncio.get_event_loop()
        try:
            vecs = await loop.run_in_executor(None, emb_module.embed_texts, texts)
            return vecs
        except EmbeddingError as e:
            logger.error("批量向量化失败 | error={}", e)
            return None
        except Exception as e:
            logger.error("批量向量化未预期异常 | error={}", e)
            return None

    async def _milvus_search(
        self,
        user_id: str,
        query_vector: list,
    ) -> list:
        """异步执行 Milvus 检索，Milvus 不可用时返回空列表（降级）。"""
        loop = asyncio.get_event_loop()
        try:
            hits = await loop.run_in_executor(
                None,
                lambda: self._milvus.search(
                    user_id=user_id,
                    query_vector=query_vector,
                    top_k=self._top_k,
                ),
            )
            return hits
        except MilvusUnavailableError:
            logger.warning("Milvus 检索失败，降级为空结果 | user_id={}", user_id)
            return []
        except Exception as e:
            logger.error("Milvus 检索未预期异常 | user_id={} | error={}", user_id, e)
            return []

    async def _milvus_insert(
        self,
        user_id: str,
        contents: List[str],
        embeddings: list,
        timestamp: str,
    ) -> int:
        """异步写入 Milvus，失败返回 0（降级）。"""
        loop = asyncio.get_event_loop()
        try:
            count = await loop.run_in_executor(
                None,
                lambda: self._milvus.insert(
                    user_id=user_id,
                    contents=contents,
                    embeddings=embeddings,
                    timestamp=timestamp,
                ),
            )
            return count
        except Exception as e:
            logger.error("Milvus 写入失败 | user_id={} | error={}", user_id, e)
            return 0

    async def _rerank(
        self,
        query: str,
        candidates: list,
    ) -> list:
        """异步执行重排，任何级别失败均降级。"""
        loop = asyncio.get_event_loop()
        try:
            reranked = await loop.run_in_executor(
                None,
                lambda: reranker_module.rerank(
                    query=query,
                    candidates=candidates,
                    top_n=self._rerank_top_n,
                ),
            )
            return reranked
        except Exception as e:
            logger.error("重排失败，返回原始 Top-N 结果 | error={}", e)
            return candidates[: self._rerank_top_n]