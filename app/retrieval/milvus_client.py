"""
app/retrieval/milvus_client.py
==============================
Milvus 原生全文索引版（Milvus >= 2.5）

核心变化：
  - Schema 中为 content 字段启用 enable_analyzer=True
  - 新增 BM25 Function，Milvus 自动将 content → sparse_embedding
  - 插入时无需本地计算 BM25，直接写入原始文本
  - 检索时 sparse_req 直接传 query 原始文本字符串
  - 删除对 bm25.py 的依赖
"""

import asyncio
from typing import Any, Dict, List, Optional

from pymilvus import (
    AnnSearchRequest,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    MilvusException,
    WeightedRanker,
    connections,
    utility,
)

from app.core.config import settings
from app.core.exceptions import MilvusUnavailableError
from app.core.logger import get_logger

logger = get_logger(__name__)

FIELD_ID               = "id"
FIELD_USER_ID          = "user_id"
FIELD_CONTENT          = "content"
FIELD_TIMESTAMP        = "timestamp"
FIELD_DENSE_EMBEDDING  = "embedding"
FIELD_SPARSE_EMBEDDING = "sparse_embedding"

MAX_VARCHAR_LEN = 65535
USER_ID_MAX_LEN = 128

SPARSE_WEIGHT = 0.3
DENSE_WEIGHT  = 0.7


def _build_schema() -> CollectionSchema:
    """
    Schema 关键变化：
      1. content 字段添加 enable_analyzer=True，允许 Milvus 对其进行分词
      2. 新增 BM25 Function，声明 content → sparse_embedding 的自动转换关系
    """
    fields = [
        FieldSchema(name=FIELD_ID,               dtype=DataType.INT64,         is_primary=True, auto_id=True),
        FieldSchema(name=FIELD_USER_ID,          dtype=DataType.VARCHAR,       max_length=USER_ID_MAX_LEN),
        FieldSchema(
            name=FIELD_CONTENT,
            dtype=DataType.VARCHAR,
            max_length=MAX_VARCHAR_LEN,
            enable_analyzer=True,       # ← 新增：允许 Milvus 对该字段分词
        ),
        FieldSchema(name=FIELD_TIMESTAMP,        dtype=DataType.VARCHAR,       max_length=64),
        FieldSchema(name=FIELD_DENSE_EMBEDDING,  dtype=DataType.FLOAT_VECTOR,  dim=settings.milvus_dim),
        FieldSchema(name=FIELD_SPARSE_EMBEDDING, dtype=DataType.SPARSE_FLOAT_VECTOR),
    ]

    schema = CollectionSchema(
        fields=fields,
        description="RAG_Memory 多用户对话记忆存储（Milvus 原生 BM25 版）",
        enable_dynamic_field=True,
    )

    # 注册 BM25 Function：声明 content 字段自动生成 sparse_embedding
    # Milvus 在插入和查询时均自动调用此 Function，无需客户端手动编码
    bm25_fn = Function(
        name="content_bm25",
        function_type=FunctionType.BM25,
        input_field_names=[FIELD_CONTENT],
        output_field_names=[FIELD_SPARSE_EMBEDDING],
    )
    schema.add_function(bm25_fn)

    return schema


class MilvusClient:

    def __init__(self) -> None:
        self._collection_name = settings.milvus_collection
        self._host = settings.milvus_host
        self._port = settings.milvus_port
        self._collection: Optional[Collection] = None
        self._connected = False
        self._sparse_weight = settings.hybrid_sparse_weight
        self._dense_weight  = settings.hybrid_dense_weight
        logger.info(
            "MilvusClient 初始化（原生BM25模式）| host={}:{} | 权重=BM25:{}/Dense:{}",
            self._host, self._port, self._sparse_weight, self._dense_weight,
        )

    def connect(self) -> bool:
        try:
            connections.connect(alias="default", host=self._host, port=str(self._port), timeout=10)
            self._collection = self._get_or_create_collection()
            self._connected = True
            logger.info("Milvus 连接成功 | {}:{}", self._host, self._port)
            return True
        except Exception as e:
            self._connected = False
            self._collection = None
            logger.error("Milvus 连接失败 | error={}", e)
            return False

    def _ensure_connected(self) -> bool:
        if self._connected and self._collection is not None:
            return True
        logger.warning("Milvus 未连接，尝试自动重连...")
        return self.connect()

    async def ping(self) -> bool:
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
        if utility.has_collection(self._collection_name):
            collection = Collection(self._collection_name)
            logger.info("找到已有 Collection: {}", self._collection_name)
            load_state = utility.load_state(self._collection_name)
            if str(load_state) != "Loaded":
                collection.load()
            else:
                logger.info("Collection 已 Loaded，跳过 load()")
        else:
            schema = _build_schema()
            collection = Collection(name=self._collection_name, schema=schema)
            logger.info("创建新 Collection（原生BM25）: {}", self._collection_name)
            self._create_index(collection)
            collection.load()

        return collection

    @staticmethod
    def _create_index(collection: Collection) -> None:
        """
        索引变化：
          - 稠密向量：IVF_FLAT + COSINE（不变）
          - 稀疏向量：SPARSE_INVERTED_INDEX + BM25（metric_type 改为 "BM25"，不再是 "IP"）
        """
        existing = [idx.field_name for idx in collection.indexes]

        if FIELD_DENSE_EMBEDDING not in existing:
            collection.create_index(
                field_name=FIELD_DENSE_EMBEDDING,
                index_params={"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}},
            )
            logger.info("稠密向量索引创建完成")

        if FIELD_SPARSE_EMBEDDING not in existing:
            collection.create_index(
                field_name=FIELD_SPARSE_EMBEDDING,
                index_params={
                    "metric_type": "BM25",               # ← 关键：原生 BM25 使用专用度量
                    "index_type": "SPARSE_INVERTED_INDEX",
                    "params": {"bm25_k1": 1.5, "bm25_b": 0.75},  # BM25 超参数（可选）
                },
            )
            logger.info("稀疏向量索引（原生BM25）创建完成")

        if FIELD_USER_ID not in existing:
            collection.create_index(field_name=FIELD_USER_ID)
            logger.info("标量索引创建完成")

    # ── 写操作 ───────────────────────────────────────────────

    def insert(
        self,
        user_id: str,
        contents: List[str],
        embeddings: List[List[float]],
        timestamp: str,
    ) -> int:
        """
        插入变化：
          - 不再需要本地计算 BM25 稀疏向量
          - 直接写入 content 原始文本，Milvus 自动通过 BM25 Function 生成 sparse_embedding
          - 数据字典中不包含 sparse_embedding 字段
        """
        if not self._ensure_connected():
            logger.error("Milvus 重连失败，跳过插入 | user_id={}", user_id)
            return 0

        if len(contents) != len(embeddings):
            logger.error("内容与向量数量不匹配")
            return 0

        try:
            # 只写 content（原始文本）和 embedding（稠密向量）
            # sparse_embedding 由 Milvus BM25 Function 自动生成，无需客户端传入
            rows = [
                {
                    FIELD_USER_ID:         user_id,
                    FIELD_CONTENT:         contents[i],
                    FIELD_TIMESTAMP:       timestamp,
                    FIELD_DENSE_EMBEDDING: list(embeddings[i]),
                    # 注意：不传 FIELD_SPARSE_EMBEDDING，Milvus 自动填充
                }
                for i in range(len(contents))
            ]
            result = self._collection.insert(rows)
            self._collection.flush()
            count = len(result.primary_keys)
            logger.info("Milvus 插入成功（原生BM25自动编码）| user_id={} | count={}", user_id, count)
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
        query_text: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        if not self._ensure_connected():
            return []

        expr = f'{FIELD_USER_ID} == "{user_id}"'

        try:
            return self._hybrid_search(
                query_vector=query_vector,
                query_text=query_text,
                expr=expr,
                top_k=top_k,
            )
        except Exception as e:
            logger.warning("混合检索失败，降级为纯稠密检索 | error={}", str(e))

        return self._dense_search(query_vector=query_vector, expr=expr, top_k=top_k)

    def _hybrid_search(
        self,
        query_vector: List[float],
        query_text: str,
        expr: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        检索变化（核心）：
          - sparse_req 的 data 直接传原始文本字符串 [query_text]
          - metric_type 改为 "BM25"（与索引一致）
          - 无需本地调用 bm25.py 编码 query
          - Milvus 服务端自动完成 query 的 BM25 编码和打分
        """
        # 稠密检索请求（不变）
        dense_req = AnnSearchRequest(
            data=[query_vector],
            anns_field=FIELD_DENSE_EMBEDDING,
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k,
            expr=expr,
        )

        # 稀疏检索请求：直接传原始文本，Milvus 自动 BM25 编码
        sparse_req = AnnSearchRequest(
            data=[query_text],                          # ← 原始文本字符串，不是向量
            anns_field=FIELD_SPARSE_EMBEDDING,
            param={"metric_type": "BM25"},              # ← 使用 BM25 度量
            limit=top_k,
            expr=expr,
        )

        ranker = WeightedRanker(self._sparse_weight, self._dense_weight)

        results = self._collection.hybrid_search(
            [sparse_req, dense_req],
            ranker,
            limit=top_k,
            output_fields=[FIELD_CONTENT, FIELD_TIMESTAMP, FIELD_USER_ID],
        )

        hits = []
        for hit in results[0]:
            entity = hit.entity
            hits.append({
                "content":     entity.get(FIELD_CONTENT)   or "",
                "score":       float(hit.score),
                "timestamp":   entity.get(FIELD_TIMESTAMP) or "",
                "user_id":     entity.get(FIELD_USER_ID)   or "",
                "id":          hit.id,
                "search_type": "hybrid_native_bm25",
            })

        logger.debug("原生BM25混合检索完成 | hits={}", len(hits))
        return hits

    def _dense_search(
        self,
        query_vector: List[float],
        expr: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """降级：纯稠密向量检索（不变）"""
        try:
            results = self._collection.search(
                data=[query_vector],
                anns_field=FIELD_DENSE_EMBEDDING,
                param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                limit=top_k,
                expr=expr,
                output_fields=[FIELD_CONTENT, FIELD_TIMESTAMP, FIELD_USER_ID],
            )
            hits = []
            for hit in results[0]:
                entity = hit.entity
                hits.append({
                    "content":     entity.get(FIELD_CONTENT)   or "",
                    "score":       float(hit.score),
                    "timestamp":   entity.get(FIELD_TIMESTAMP) or "",
                    "user_id":     entity.get(FIELD_USER_ID)   or "",
                    "id":          hit.id,
                    "search_type": "dense_fallback",
                })
            return hits
        except MilvusException as e:
            logger.error("稠密检索失败 | error={}", e)
            self._connected = False
            raise MilvusUnavailableError(detail=str(e))