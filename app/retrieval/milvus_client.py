"""
app/retrieval/milvus_client.py
==============================
Milvus 数据库连接与操作模块。

修复历史：
  v3: 加入懒加载重连（_ensure_connected）
  v4: 修复 disconnect 导致的索引冲突
      - connect() 不再预先 disconnect，避免 release collection 后重建索引冲突
      - _create_index() 加幂等检查，已有索引直接跳过
"""

import asyncio
from typing import Any, Dict, List, Optional

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusException,
    connections,
    utility,
)

from app.core.config import settings
from app.core.exceptions import MilvusUnavailableError
from app.core.logger import get_logger

logger = get_logger(__name__)

FIELD_ID        = "id"
FIELD_USER_ID   = "user_id"
FIELD_CONTENT   = "content"
FIELD_TIMESTAMP = "timestamp"
FIELD_EMBEDDING = "embedding"

MAX_VARCHAR_LEN = 65535
USER_ID_MAX_LEN = 128


def _build_schema() -> CollectionSchema:
    fields = [
        FieldSchema(name=FIELD_ID,        dtype=DataType.INT64,         is_primary=True, auto_id=True),
        FieldSchema(name=FIELD_USER_ID,   dtype=DataType.VARCHAR,       max_length=USER_ID_MAX_LEN),
        FieldSchema(name=FIELD_CONTENT,   dtype=DataType.VARCHAR,       max_length=MAX_VARCHAR_LEN),
        FieldSchema(name=FIELD_TIMESTAMP, dtype=DataType.VARCHAR,       max_length=64),
        FieldSchema(name=FIELD_EMBEDDING, dtype=DataType.FLOAT_VECTOR,  dim=settings.milvus_dim),
    ]
    return CollectionSchema(
        fields=fields,
        description="RAG_Memory 多用户对话记忆存储",
        enable_dynamic_field=True,
    )


class MilvusClient:
    """
    Milvus 客户端封装，insert/search 均内置懒加载重连。
    """

    def __init__(self) -> None:
        self._collection_name = settings.milvus_collection
        self._host = settings.milvus_host
        self._port = settings.milvus_port
        self._collection: Optional[Collection] = None
        self._connected = False
        logger.info(
            "MilvusClient 初始化 | host={}:{} | collection={}",
            self._host, self._port, self._collection_name,
        )

    # ── 连接管理 ─────────────────────────────────────────────

    def connect(self) -> bool:
        """
        建立与 Milvus 的连接并初始化 Collection。

        ⚠️ 不再预先调用 disconnect()：
           disconnect 会导致服务端 release collection，
           再次 connect 时 load() 触发索引冲突报错。
           改为直接 connect，pymilvus 内部会复用已有连接。

        Returns:
            True 表示连接成功，False 表示失败。
        """
        try:
            connections.connect(
                alias="default",
                host=self._host,
                port=str(self._port),
                timeout=10,
            )
            self._collection = self._get_or_create_collection()
            self._connected = True
            logger.info("Milvus 连接成功 | {}:{}", self._host, self._port)
            return True
        except Exception as e:
            self._connected = False
            self._collection = None
            logger.error("Milvus 连接失败 | host={}:{} | error={}", self._host, self._port, e)
            return False

    def _ensure_connected(self) -> bool:
        """
        确保处于已连接状态，若未连接则自动重连一次。

        Returns:
            True 表示连接可用，False 表示重连失败。
        """
        if self._connected and self._collection is not None:
            return True
        logger.warning("Milvus 未连接，尝试自动重连... | host={}:{}", self._host, self._port)
        return self.connect()

    async def ping(self) -> bool:
        """异步检查 Milvus 是否可达（供健康检查接口调用）。"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._ping_sync)

    def _ping_sync(self) -> bool:
        try:
            if not self._connected:
                return self.connect()
            utility.get_server_version()
            return True
        except Exception:
            return False

    def _get_or_create_collection(self) -> Collection:
        """
        获取已有 Collection 或创建新 Collection（含索引）。

        load 策略：
          - 已有 Collection：检查 load_state，已加载则跳过 load()，
            避免触发 can't change the index for loaded collection 冲突。
          - 新建 Collection：创建索引后再 load()。
        """
        if utility.has_collection(self._collection_name):
            collection = Collection(self._collection_name)
            logger.info("找到已有 Collection: {}", self._collection_name)
            # 已加载则直接使用，避免重复 load() 引发索引冲突
            load_state = utility.load_state(self._collection_name)
            logger.info("Collection load_state={} | {}", str(load_state), self._collection_name)
            if str(load_state) != "Loaded":
                collection.load()
                logger.info("Collection load 完成: {}", self._collection_name)
            else:
                logger.info("Collection 已 Loaded，跳过 load(): {}", self._collection_name)
        else:
            schema = _build_schema()
            collection = Collection(name=self._collection_name, schema=schema)
            logger.info("创建新 Collection: {}", self._collection_name)
            self._create_index(collection)
            collection.load()
            logger.info("新 Collection load 完成: {}", self._collection_name)

        return collection

    @staticmethod
    def _create_index(collection: Collection) -> None:
        """
        创建向量索引与标量索引（仅在新建 Collection 时调用）。
        加幂等保护：已有索引则跳过，避免重复创建报错。
        """
        # 检查向量索引是否已存在
        existing = [idx.field_name for idx in collection.indexes]

        if FIELD_EMBEDDING not in existing:
            collection.create_index(
                field_name=FIELD_EMBEDDING,
                index_params={
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128},
                },
            )
            logger.info("向量索引创建完成: {}", collection.name)
        else:
            logger.debug("向量索引已存在，跳过创建: {}", collection.name)

        if FIELD_USER_ID not in existing:
            collection.create_index(field_name=FIELD_USER_ID)
            logger.info("标量索引创建完成: {}", collection.name)
        else:
            logger.debug("标量索引已存在，跳过创建: {}", collection.name)

    # ── 写操作 ───────────────────────────────────────────────

    def insert(
        self,
        user_id: str,
        contents: List[str],
        embeddings: List[List[float]],
        timestamp: str,
    ) -> int:
        """
        批量插入文本切片及其向量至 Milvus。
        内置懒加载重连：未连接时自动重连后再执行插入。

        Returns:
            成功插入的条目数量；失败返回 0（降级）。
        """
        if not self._ensure_connected():
            logger.error("Milvus 重连失败，跳过插入 | user_id={}", user_id)
            return 0

        if len(contents) != len(embeddings):
            logger.error("内容与向量数量不匹配 | contents={} vs embeddings={}",
                         len(contents), len(embeddings))
            return 0

        try:
            data = [
                [user_id] * len(contents),
                contents,
                [timestamp] * len(contents),
                embeddings,
            ]
            result = self._collection.insert(data)
            self._collection.flush()
            count = len(result.primary_keys)
            logger.info("Milvus 插入成功 | user_id={} | count={}", user_id, count)
            return count
        except MilvusException as e:
            logger.error("Milvus 插入失败 | user_id={} | error={}", user_id, e)
            self._connected = False
            return 0

    # ── 检索操作 ─────────────────────────────────────────────

    def search(
        self,
        user_id: str,
        query_vector: List[float],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        对指定用户执行向量相似度检索（带 user_id 过滤）。
        内置懒加载重连：未连接时自动重连后再执行检索。

        Returns:
            检索结果列表；Milvus 不可用时返回空列表（降级）。
        """
        if not self._ensure_connected():
            logger.error("Milvus 重连失败，返回空检索结果 | user_id={}", user_id)
            return []

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
        expr = f'{FIELD_USER_ID} == "{user_id}"'

        try:
            results = self._collection.search(
                data=[query_vector],
                anns_field=FIELD_EMBEDDING,
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=[FIELD_CONTENT, FIELD_TIMESTAMP, FIELD_USER_ID],
            )

            hits: List[Dict[str, Any]] = []
            for hit in results[0]:
                # pymilvus 新版 hit.entity.get() 只接受一个参数（字段名），
                # 不支持第二个默认值参数，需手动处理 None
                entity = hit.entity
                hits.append({
                    "content":   entity.get(FIELD_CONTENT)   or "",
                    "score":     float(hit.score),
                    "timestamp": entity.get(FIELD_TIMESTAMP) or "",
                    "user_id":   entity.get(FIELD_USER_ID)   or "",
                    "id":        hit.id,
                })

            logger.debug("Milvus 检索完成 | user_id={} | hits={}", user_id, len(hits))
            return hits

        except MilvusException as e:
            logger.error("Milvus 检索失败 | user_id={} | error={}", user_id, e)
            self._connected = False
            raise MilvusUnavailableError(detail=str(e))
