"""
app/api/schemas.py
==================
Pydantic 数据验证模型，定义所有 API 接口的请求体与响应体结构。

设计原则：
  - 入参严格校验，出参结构统一，便于前端/调用方解析。
  - 响应统一使用 BaseResponse 包裹，包含 success 标志。
"""

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
# 通用响应基类
# ──────────────────────────────────────────────────────────────

class BaseResponse(BaseModel):
    """所有 API 响应的基础结构。"""

    success: bool = Field(True, description="请求是否成功处理")
    message: str = Field("OK", description="人类可读的状态描述")


# ──────────────────────────────────────────────────────────────
# 接口 1：/api/v1/chat/memory/query
# ──────────────────────────────────────────────────────────────

class MemoryQueryRequest(BaseModel):
    """
    记忆查询请求体。
    外部 AI 代理在每次对话前调用此接口，获取相关记忆上下文。
    """

    user_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="用户唯一标识符，用于隔离多租户数据",
        examples=["user_12345"],
    )
    query: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="当前用户输入的消息文本",
        examples=["我上周说的那个项目怎么样了？"],
    )
    time: datetime = Field(
        ...,
        description="请求发生时间（ISO 8601 格式，UTC）",
        examples=["2023-10-27T10:00:00Z"],
    )


class RetrievedChunk(BaseModel):
    """单条向量检索结果切片。"""

    content: str = Field(..., description="切片原文内容")
    score: float = Field(..., description="相关性得分（越高越相关）")
    source: str = Field("vector_db", description="来源标识")
    metadata: dict = Field(default_factory=dict, description="附加元数据（时间、会话 ID 等）")


class MemoryQueryResponse(BaseResponse):
    """
    记忆查询响应体。
    返回向量检索结果、用户画像及合并后的增强上下文，
    供外部 AI 代理直接拼入系统 Prompt。
    """

    user_id: str = Field(..., description="对应的用户 ID")
    user_profile: str = Field("", description="用户画像（user.md 文件内容）")
    retrieved_chunks: List[RetrievedChunk] = Field(
        default_factory=list, description="向量检索到的相关历史片段列表"
    )
    augmented_context: str = Field(
        "", description="合并用户画像与检索结果后的最终增强上下文字符串，可直接注入 Prompt"
    )
    query_time_ms: Optional[float] = Field(None, description="本次查询耗时（毫秒）")


# ──────────────────────────────────────────────────────────────
# 接口 2：/api/v1/chat/memory/upload
# ──────────────────────────────────────────────────────────────

class MemoryUploadRequest(BaseModel):
    """
    对话历史/文件上传请求体。
    外部代理在每轮对话结束后调用，将本轮内容存入记忆系统。
    """

    user_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="用户唯一标识符",
        examples=["user_12345"],
    )
    messages: Optional[List[dict]] = Field(
        None,
        description=(
            "本轮对话消息列表，格式为 [{role: 'user'|'assistant', content: '...'}]。"
            "与 multifiles 二选一，或同时提供。"
        ),
    )
    multifiles: Optional[List[str]] = Field(
        None,
        description="Base64 编码的文件字符串列表，支持文本文件批量导入",
    )
    time: datetime = Field(
        ...,
        description="上传发生时间（ISO 8601 格式，UTC）",
        examples=["2023-10-27T10:05:00Z"],
    )


class MemoryUploadResponse(BaseResponse):
    """对话历史上传响应体。"""

    user_id: str = Field(..., description="对应的用户 ID")
    chunks_stored: int = Field(0, description="本次成功存入 Milvus 的切片数量")
    profile_updated: bool = Field(False, description="用户画像是否被本次上传触发更新")
    process_time_ms: Optional[float] = Field(None, description="处理耗时（毫秒）")


# ──────────────────────────────────────────────────────────────
# 健康检查
# ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """健康检查响应体。"""

    status: str = Field("ok", description="服务状态")
    version: str = Field("1.0.0", description="服务版本号")
    milvus_connected: bool = Field(False, description="Milvus 连接状态")
