"""
app/retrieval/milvus_client.py
==============================
Milvus 数据库连接与操作模块。

核心升级：混合检索（Hybrid Search）
  - Schema 新增 sparse_embedding 字段（SPARSE_FLOAT_VECTOR），存储 BM25 稀疏向量。
  - insert()  同时写入稠密向量（embedding）与稀疏向量（BM25）。
  - search()  使用 hybrid_search + WeightedRanker(0.3, 0.7) 实现：
              最终得分 = 0.3 * BM25稀疏得分 + 0.7 * 余弦相似度得分
  - 降级策略：hybrid_search 失败时自动退回纯稠密向量检索。

修复历史：
  v3: 懒加载重连
  v4: 移除 disconnect 预调用，修复索引冲突
  v5: load_state 检查，跳过重复 load()
  v6: hit.entity.get() 兼容性修复
  v7: 混合检索（本次）
"""

import asyncio
from typing import Any, Dict, List, Optional

from pymilvus import (
    AnnSearchRequest,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusException,
    RRFRanker,
    WeightedRanker,
    connections,
    utility,
)

from app.core.config import settings
from app.core.exceptions import MilvusUnavailableError
from app.core.logger import get_logger
from app.retrieval.bm25 import text_to_sparse_vector

# pymilvus 稀疏向量辅助：不同版本对稀疏向量格式要求不同
# 统一转为 [{index: value, ...}] 的行级别字典列表，兼容性最好
def _sparse_dict_to_pymilvus(sparse: dict) -> dict:
    """
    将 {token_id: weight} 字典转为 pymilvus SPARSE_FLOAT_VECTOR 所需格式。
    pymilvus >= 2.4 接受原生 Python {int: float} dict，
    但 key 必须是 Python int（不能是 numpy int），value 必须是 float。
    """
    return {int(k): float(v) for k, v in sparse.items()}

logger = get_logger(__name__)

# ── 字段名常量 ────────────────────────────────────────────────
FIELD_ID              = "id"
FIELD_USER_ID         = "user_id"
FIELD_CONTENT         = "content"
FIELD_TIMESTAMP       = "timestamp"
FIELD_DENSE_EMBEDDING = "embedding"          # 稠密向量（余弦相似度）
FIELD_SPARSE_EMBEDDING = "sparse_embedding"  # 稀疏向量（BM25）

MAX_VARCHAR_LEN = 65535
USER_ID_MAX_LEN = 128

# 混合检索权重（默认值，实际运行时从 settings 读取）
SPARSE_WEIGHT = 0.3
DENSE_WEIGHT  = 0.7


def _build_schema() -> CollectionSchema:
    """
    构建包含稠密向量 + 稀疏向量双字段的 Collection Schema。

    字段：
      id               - INT64 主键，自增
      user_id          - VARCHAR，多租户过滤键
      content          - VARCHAR，原始文本
      timestamp        - VARCHAR，存储时间
      embedding        - FLOAT_VECTOR(dim)，稠密向量（用于余弦相似度）
      sparse_embedding - SPARSE_FLOAT_VECTOR，BM25 稀疏向量（用于词汇匹配）
    """
    fields = [
        FieldSchema(name=FIELD_ID,              dtype=DataType.INT64,          is_primary=True, auto_id=True),
        FieldSchema(name=FIELD_USER_ID,         dtype=DataType.VARCHAR,        max_length=USER_ID_MAX_LEN),
        FieldSchema(name=FIELD_CONTENT,         dtype=DataType.VARCHAR,        max_length=MAX_VARCHAR_LEN),
        FieldSchema(name=FIELD_TIMESTAMP,       dtype=DataType.VARCHAR,        max_length=64),
        FieldSchema(name=FIELD_DENSE_EMBEDDING, dtype=DataType.FLOAT_VECTOR,   dim=settings.milvus_dim),
        FieldSchema(name=FIELD_SPARSE_EMBEDDING,dtype=DataType.SPARSE_FLOAT_VECTOR),
    ]
    return CollectionSchema(
        fields=fields,
        description="RAG_Memory 多用户对话记忆存储（混合检索版）",
        enable_dynamic_field=True,
    )


class MilvusClient:
    """
    Milvus 客户端封装。

    核心特性：
      - insert/search 内置懒加载重连。
      - search 优先使用混合检索（BM25 + 余弦），失败时降级为纯稠密检索。
    """

    def __init__(self) -> None:
        self._collection_name = settings.milvus_collection
        self._host = settings.milvus_host
        self._port = settings.milvus_port
        self._collection: Optional[Collection] = None
        self._connected = False
        self._sparse_weight = settings.hybrid_sparse_weight
        self._dense_weight  = settings.hybrid_dense_weight
        logger.info(
            "MilvusClient 初始化 | host={}:{} | collection={} | 混合检索权重=BM25:{}/Dense:{}",
            self._host, self._port, self._collection_name,
            self._sparse_weight, self._dense_weight,
        )

    # ── 连接管理 ─────────────────────────────────────────────

    def connect(self) -> bool:
        """建立与 Milvus 的连接并初始化 Collection（幂等）。"""
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
        """确保处于已连接状态，未连接则自动重连一次。"""
        if self._connected and self._collection is not None:
            return True
        logger.warning("Milvus 未连接，尝试自动重连... | host={}:{}", self._host, self._port)
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
        """
        获取或创建 Collection，检查 load_state 避免重复 load() 触发索引冲突。
        """
        if utility.has_collection(self._collection_name):
            collection = Collection(self._collection_name)
            logger.info("找到已有 Collection: {}", self._collection_name)
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
            logger.info("创建新 Collection（含稀疏向量字段）: {}", self._collection_name)
            self._create_index(collection)
            collection.load()
            logger.info("新 Collection load 完成: {}", self._collection_name)

        return collection

    def _has_sparse_field(self) -> bool:
        """检查当前 Collection 是否包含稀疏向量字段（用于兼容旧版 Collection）。"""
        if self._collection is None:
            return False
        field_names = [f.name for f in self._collection.schema.fields]
        return FIELD_SPARSE_EMBEDDING in field_names

    @staticmethod
    def _create_index(collection: Collection) -> None:
        """
        创建三类索引（幂等）：
          1. 稠密向量索引（IVF_FLAT + COSINE）
          2. 稀疏向量索引（SPARSE_INVERTED_INDEX + IP，BM25 使用内积）
          3. user_id 标量索引（加速多租户过滤）
        """
        existing = [idx.field_name for idx in collection.indexes]

        if FIELD_DENSE_EMBEDDING not in existing:
            collection.create_index(
                field_name=FIELD_DENSE_EMBEDDING,
                index_params={
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128},
                },
            )
            logger.info("稠密向量索引创建完成")
        else:
            logger.debug("稠密向量索引已存在，跳过")

        if FIELD_SPARSE_EMBEDDING not in existing:
            collection.create_index(
                field_name=FIELD_SPARSE_EMBEDDING,
                index_params={
                    "metric_type": "IP",           # 稀疏向量使用内积（等价于 BM25 得分累加）
                    "index_type": "SPARSE_INVERTED_INDEX",
                    "params": {"drop_ratio_build": 0.2},  # 构建时丢弃权重最低的 20% token
                },
            )
            logger.info("稀疏向量索引（BM25）创建完成")
        else:
            logger.debug("稀疏向量索引已存在，跳过")

        if FIELD_USER_ID not in existing:
            collection.create_index(field_name=FIELD_USER_ID)
            logger.info("标量索引创建完成")
        else:
            logger.debug("标量索引已存在，跳过")

    # ── 写操作 ───────────────────────────────────────────────

    def insert(
        self,
        user_id: str,
        contents: List[str],
        embeddings: List[List[float]],
        timestamp: str,
    ) -> int:
        """
        批量插入文本切片，同时写入稠密向量（来自 Embedding API）和稀疏向量（BM25 本地计算）。

        Args:
            user_id:    用户 ID。
            contents:   文本内容列表。
            embeddings: 与 contents 对应的稠密向量列表。
            timestamp:  存储时间字符串。

        Returns:
            成功插入的条目数；失败返回 0（降级）。
        """
        if not self._ensure_connected():
            logger.error("Milvus 重连失败，跳过插入 | user_id={}", user_id)
            return 0

        if len(contents) != len(embeddings):
            logger.error("内容与向量数量不匹配 | contents={} vs embeddings={}",
                         len(contents), len(embeddings))
            return 0

        # 检查 Collection 是否有稀疏向量字段（兼容旧版 Collection）
        use_sparse = self._has_sparse_field()

        if use_sparse:
            # 本地计算 BM25 稀疏向量，确保 key/value 均为原生 Python int/float
            sparse_vectors = [
                _sparse_dict_to_pymilvus(text_to_sparse_vector(c) or {0: 1e-9})
                for c in contents
            ]
        else:
            logger.warning(
                "Collection 无稀疏向量字段（旧版 Schema），仅插入稠密向量 | user_id={}",
                user_id,
            )

        try:
            if use_sparse:
                # 使用行级别列表格式插入，每行是一个字典，兼容性最强
                rows = []
                for i in range(len(contents)):
                    rows.append({
                        FIELD_USER_ID:          user_id,
                        FIELD_CONTENT:          contents[i],
                        FIELD_TIMESTAMP:        timestamp,
                        FIELD_DENSE_EMBEDDING:  list(embeddings[i]),
                        FIELD_SPARSE_EMBEDDING: sparse_vectors[i],
                    })
                result = self._collection.insert(rows)
            else:
                # 旧版 Collection：列表格式，不含稀疏字段
                data = [
                    [user_id] * len(contents),
                    list(contents),
                    [timestamp] * len(contents),
                    list(embeddings),
                ]
                result = self._collection.insert(data)
            self._collection.flush()
            count = len(result.primary_keys)
            logger.info(
                "Milvus 混合插入成功 | user_id={} | count={} | 含BM25稀疏向量",
                user_id, count,
            )
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
        """
        混合检索：0.3 * BM25稀疏得分 + 0.7 * 余弦相似度得分。

        使用 Milvus hybrid_search + WeightedRanker 实现加权融合。
        若混合检索失败（如旧版 Collection 无稀疏字段），自动降级为纯稠密检索。

        Args:
            user_id:      目标用户 ID（多租户过滤）。
            query_vector: 查询稠密向量。
            query_text:   查询原始文本（用于生成 BM25 稀疏查询向量）。
            top_k:        召回数量。

        Returns:
            检索结果列表，每项含 content/score/timestamp/user_id/id。
        """
        if not self._ensure_connected():
            logger.error("Milvus 重连失败，返回空检索结果 | user_id={}", user_id)
            return []

        expr = f'{FIELD_USER_ID} == "{user_id}"'

        # 若 Collection 无稀疏字段（旧版 Schema），直接走稠密检索
        if not self._has_sparse_field():
            logger.warning("Collection 无稀疏向量字段，使用纯稠密检索 | user_id={}", user_id)
            return self._dense_search(query_vector=query_vector, expr=expr, top_k=top_k)

        # 优先尝试混合检索
        try:
            return self._hybrid_search(
                query_vector=query_vector,
                query_text=query_text,
                expr=expr,
                top_k=top_k,
            )
        except Exception as e:
            logger.warning(
                "混合检索失败，降级为纯稠密向量检索 | error={}", str(e)
            )

        # 降级：纯稠密向量检索
        return self._dense_search(
            query_vector=query_vector,
            expr=expr,
            top_k=top_k,
        )

    def _hybrid_search(
        self,
        query_vector: List[float],
        query_text: str,
        expr: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        执行 Milvus hybrid_search（稠密 + 稀疏双路召回 + WeightedRanker 融合）。

        融合公式：score = 0.3 * sparse_score + 0.7 * dense_score
        """
        # 生成查询的 BM25 稀疏向量，确保 key/value 为原生 Python int/float
        raw_sparse = text_to_sparse_vector(query_text)
        if not raw_sparse:
            raise ValueError("查询文本无有效 BM25 token，退化为稠密检索")
        query_sparse = _sparse_dict_to_pymilvus(raw_sparse)

        # 构建稠密检索请求
        dense_req = AnnSearchRequest(
            data=[query_vector],
            anns_field=FIELD_DENSE_EMBEDDING,
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k,
            expr=expr,
        )

        # 构建稀疏检索请求（BM25）
        sparse_req = AnnSearchRequest(
            data=[query_sparse],
            anns_field=FIELD_SPARSE_EMBEDDING,
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=top_k,
            expr=expr,
        )

        # WeightedRanker：[稀疏权重, 稠密权重] 对应 [sparse_req, dense_req] 的顺序
        # 注意：reqs 列表的顺序决定权重的对应关系
        ranker = WeightedRanker(self._sparse_weight, self._dense_weight)

        # 兼容不同版本 pymilvus：ranker 既可能是关键字参数也可能是位置参数
        # 统一用位置参数传递，避免版本差异导致的 missing argument 报错
        results = self._collection.hybrid_search(
            [sparse_req, dense_req],       # reqs，sparse first → weight[0]=0.3
            ranker,                        # rerank/ranker（位置参数，兼容两种版本）
            limit=top_k,
            output_fields=[FIELD_CONTENT, FIELD_TIMESTAMP, FIELD_USER_ID],
        )

        hits = []
        for hit in results[0]:
            entity = hit.entity
            hits.append({
                "content":   entity.get(FIELD_CONTENT)   or "",
                "score":     float(hit.score),
                "timestamp": entity.get(FIELD_TIMESTAMP) or "",
                "user_id":   entity.get(FIELD_USER_ID)   or "",
                "id":        hit.id,
                "search_type": "hybrid",
            })

        logger.debug(
            "混合检索完成 | hits={} | 权重=BM25:{}/Dense:{}",
            len(hits), self._sparse_weight, self._dense_weight,
        )
        logger.warning(
            "混合检索完成 | hits={} | 权重=BM25:{}/Dense:{}",
            len(hits), self._sparse_weight, self._dense_weight,
        )
        return hits

    def _dense_search(
        self,
        query_vector: List[float],
        expr: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        纯稠密向量检索（降级用）。
        """
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
        try:
            results = self._collection.search(
                data=[query_vector],
                anns_field=FIELD_DENSE_EMBEDDING,
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=[FIELD_CONTENT, FIELD_TIMESTAMP, FIELD_USER_ID],
            )
            hits = []
            for hit in results[0]:
                entity = hit.entity
                hits.append({
                    "content":   entity.get(FIELD_CONTENT)   or "",
                    "score":     float(hit.score),
                    "timestamp": entity.get(FIELD_TIMESTAMP) or "",
                    "user_id":   entity.get(FIELD_USER_ID)   or "",
                    "id":        hit.id,
                    "search_type": "dense_fallback",
                })
            logger.debug("稠密检索（降级）完成 | hits={}", len(hits))
            return hits
        except MilvusException as e:
            logger.error("稠密检索失败 | error={}", e)
            self._connected = False
            raise MilvusUnavailableError(detail=str(e))
