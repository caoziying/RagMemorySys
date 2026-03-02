"""
app/api/endpoints.py
====================
路由层：定义所有 HTTP 端点，负责请求接收、参数校验、
调用业务服务层，并返回标准化 JSON 响应。

端点清单：
  POST /api/v1/chat/memory/query  - 查询记忆上下文
  POST /api/v1/chat/memory/upload - 上传对话历史/文件
  GET  /health                    - 服务健康检查
"""

import asyncio
import time
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks

from app.api.schemas import (
    HealthResponse,
    MemoryQueryRequest,
    MemoryQueryResponse,
    MemoryUploadRequest,
    MemoryUploadResponse,
    RetrievedChunk,
)
from app.core.logger import get_logger, log_conversation
from app.memory.manager import MemoryManager
from app.memory.profile import ProfileManager
from app.retrieval.retriever import Retriever
from app.retrieval.milvus_client import MilvusClient

logger = get_logger(__name__)

# APIRouter 支持后续在 main.py 中按前缀挂载
router = APIRouter()

# 模块级服务单例（实际项目可用 FastAPI Depends 注入）
_retriever = Retriever()
_memory_manager = MemoryManager()
_profile_manager = ProfileManager()
_milvus_client = MilvusClient()


# ──────────────────────────────────────────────────────────────
# POST /api/v1/chat/memory/query
# ──────────────────────────────────────────────────────────────

@router.post(
    "/chat/memory/query",
    response_model=MemoryQueryResponse,
    summary="查询记忆上下文",
    description=(
        "接收用户当前输入，从 Milvus 检索相关历史记忆，"
        "结合用户画像(user.md)，返回合并后的增强上下文，供 AI 代理注入 Prompt 使用。"
    ),
)
async def query_memory(
    request: MemoryQueryRequest,
    background_tasks: BackgroundTasks,
) -> MemoryQueryResponse:
    """
    记忆查询主流程（参考 README 工作流要求）：
      1. 提取 user_id
      2. 向量检索相关历史切片（语义召回）
      3. 读取 user.md 用户画像
      4. 读取最近 N 轮对话历史（时序上下文）
      5. 合并检索结果、用户画像、近期对话，构造增强上下文
      6. （后台异步）将本次 query 记录至对话日志
    """
    start_ts = time.monotonic()
    user_id = request.user_id
    logger.info("收到记忆查询请求 | user_id={} | query[:50]={}", user_id, request.query[:50])

    # Step 2: 向量检索（语义相关历史）
    raw_chunks = await _retriever.retrieve(user_id=user_id, query=request.query)

    # Step 3: 读取用户画像
    user_profile = await _profile_manager.read_profile(user_id=user_id)

    # Step 4: 读取最近 N 轮对话（时序上下文，补充向量检索的近期感知盲区）
    # 兼容新旧两版 MemoryManager：
    #   新版 get_recent_messages 返回 List[dict]
    #   旧版 get_recent_history  返回 List[str]，在此统一转为 List[dict]
    if hasattr(_memory_manager, 'get_recent_messages'):
        recent_messages = await _memory_manager.get_recent_messages(user_id=user_id)
    else:
        raw_texts = await _memory_manager.get_recent_history(user_id=user_id)
        recent_messages = _parse_history_texts(raw_texts)

    # Step 5: 合并检索结果、用户画像、最近对话，构造增强上下文字符串
    augmented_context = _build_augmented_context(user_profile, raw_chunks, recent_messages)

    # Step 6（后台）：记录本次 query 到对话日志，不阻塞响应
    background_tasks.add_task(
        log_conversation, user_id, "user", request.query
    )

    elapsed_ms = (time.monotonic() - start_ts) * 1000
    logger.info(
        "记忆查询完成 | user_id={} | chunks={} | 耗时={:.1f}ms",
        user_id, len(raw_chunks), elapsed_ms,
    )

    return MemoryQueryResponse(
        success=True,
        message="查询成功",
        user_id=user_id,
        user_profile=user_profile,
        retrieved_chunks=raw_chunks,
        augmented_context=augmented_context,
        query_time_ms=round(elapsed_ms, 2),
    )


# ──────────────────────────────────────────────────────────────
# POST /api/v1/chat/memory/upload
# ──────────────────────────────────────────────────────────────

@router.post(
    "/chat/memory/upload",
    response_model=MemoryUploadResponse,
    summary="上传对话历史或文件至记忆系统",
    description=(
        "将本轮对话消息或 Base64 文件存入 Milvus 向量数据库及基础记忆系统，"
        "并异步触发用户画像(user.md)更新。"
    ),
)
async def upload_memory(
    request: MemoryUploadRequest,
    background_tasks: BackgroundTasks,
) -> MemoryUploadResponse:
    """
    对话历史上传主流程：
      1. 解析消息列表或 Base64 文件为文本
      2. 分块并向量化，写入 Milvus
      3. 更新基础记忆（滑动窗口 / 压缩）
      4. （后台异步）调用 LLM 提取用户信息，更新 user.md
    """
    start_ts = time.monotonic()
    user_id = request.user_id
    logger.info(
        "收到记忆上传请求 | user_id={} | messages={} | files={}",
        user_id,
        len(request.messages or []),
        len(request.multifiles or []),
    )

    # Step 1: 收集所有文本内容
    texts: list[str] = []

    if request.messages:
        for msg in request.messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            texts.append(f"[{role}]: {content}")
            # 同步写入对话日志
            background_tasks.add_task(log_conversation, user_id, role, content)

    if request.multifiles:
        decoded_texts = _decode_base64_files(request.multifiles)
        texts.extend(decoded_texts)

    if not texts:
        return MemoryUploadResponse(
            success=False,
            message="请求中不包含任何有效内容（messages 或 multifiles 均为空）",
            user_id=user_id,
        )

    # Step 2: 分块存入 Milvus（在 retriever 中完成向量化）
    chunks_stored = await _retriever.store(
        user_id=user_id,
        texts=texts,
        timestamp=request.time,
    )

    # Step 3: 更新基础记忆（滑动窗口）
    await _memory_manager.update(user_id=user_id, new_texts=texts)

    # Step 4（后台）：LLM 提取用户信息并更新 user.md
    background_tasks.add_task(
        _profile_manager.extract_and_update_profile,
        user_id,
        "\n".join(texts),
    )

    elapsed_ms = (time.monotonic() - start_ts) * 1000
    logger.info(
        "记忆上传完成 | user_id={} | chunks_stored={} | 耗时={:.1f}ms",
        user_id, chunks_stored, elapsed_ms,
    )

    return MemoryUploadResponse(
        success=True,
        message="上传成功，画像更新已在后台异步进行",
        user_id=user_id,
        chunks_stored=chunks_stored,
        profile_updated=False,  # 画像更新为后台异步，此处标记 False
        process_time_ms=round(elapsed_ms, 2),
    )


# ──────────────────────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="服务健康检查",
    tags=["运维"],
)
async def health_check() -> HealthResponse:
    """检查 API 服务及依赖（Milvus）的连通状态。"""
    milvus_ok = await _milvus_client.ping()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        milvus_connected=milvus_ok,
    )


# ──────────────────────────────────────────────────────────────
# 内部辅助函数
# ──────────────────────────────────────────────────────────────

def _parse_history_texts(texts: list[str]) -> list[dict]:
    """
    将旧版 get_recent_history() 返回的 List[str] 解析为 List[dict]。
    输入格式：["[user]: 内容", "[assistant]: 内容", ...]
    输出格式：[{"role": "user", "content": "内容", "timestamp": ""}, ...]
    """
    messages = []
    for text in texts:
        if text.startswith("[user]: "):
            messages.append({"role": "user",      "content": text[len("[user]: "):],      "timestamp": ""})
        elif text.startswith("[assistant]: "):
            messages.append({"role": "assistant", "content": text[len("[assistant]: "):], "timestamp": ""})
        else:
            messages.append({"role": "unknown",   "content": text, "timestamp": ""})
    return messages


def _build_augmented_context(
    user_profile: str,
    chunks: list[RetrievedChunk],
    recent_messages: list[dict] | None = None,
) -> str:
    """
    将用户画像、最近 N 轮对话、语义检索结果合并为结构化增强上下文。
    可直接注入外部 AI 代理的系统 Prompt。

    三个信息源的分工：
      - 用户画像：LLM 提炼的稳定个人信息（姓名/职业/偏好等）
      - 最近对话：时序上最近的 N 轮消息（确保近期上下文不丢失）
      - 语义检索：与当前 query 最相关的历史片段（可能跨越很久以前）

    Args:
        user_profile:    user.md 的文本内容。
        chunks:          重排后的向量检索结果列表。
        recent_messages: 最近 N 轮对话，格式 [{role, content, timestamp}, ...]

    Returns:
        格式化后的增强上下文字符串。
    """
    sections: list[str] = []

    # 1. 用户画像
    if user_profile.strip():
        sections.append(f"## 用户画像\n{user_profile.strip()}")

    # 2. 最近 N 轮对话（时序上下文，补充向量检索的近期感知盲区）
    if recent_messages:
        lines = []
        for msg in recent_messages:
            role    = msg.get("role", "unknown")
            content = msg.get("content", "").strip()
            ts      = msg.get("timestamp", "")
            prefix  = f"[{ts[:16]}] " if ts else ""
            lines.append(f"{prefix}[{role}]: {content}")
        sections.append("## 最近对话记录\n" + "\n".join(lines))

    # 3. 语义检索结果（与当前 query 相关的历史片段）
    if chunks:
        history_lines = [
            f"- [相关度 {c.score:.3f}] {c.content.strip()}"
            for c in chunks
        ]
        sections.append("## 相关历史记忆（语义检索）\n" + "\n".join(history_lines))

    if not sections:
        return "（暂无历史记忆或用户画像）"

    return "\n\n".join(sections)


def _decode_base64_files(encoded_files: list[str]) -> list[str]:
    """
    将 Base64 编码的文件字符串列表解码为文本列表。
    对解码失败的条目记录警告并跳过，不中断整体流程。

    Args:
        encoded_files: Base64 字符串列表。

    Returns:
        解码后的文本字符串列表。
    """
    import base64

    results: list[str] = []
    for idx, encoded in enumerate(encoded_files):
        try:
            decoded_bytes = base64.b64decode(encoded)
            text = decoded_bytes.decode("utf-8", errors="replace")
            results.append(text)
        except Exception as e:
            logger.warning("Base64 文件解码失败 | index={} | error={}", idx, e)
    return results
