"""
rag_client.py - RAG_Memory 接口客户端
仅通过 HTTP 调用 RAG_Memory 的两个标准接口，与其完全解耦。
"""
import os
from datetime import datetime, timezone
from typing import Optional

import httpx

RAG_MEMORY_BASE = os.getenv("RAG_MEMORY_URL", "http://rag-api:8000")
TIMEOUT = 30.0


async def query_memory(user_id: str, query: str) -> dict:
    """
    调用 RAG_Memory 查询接口，获取相关记忆上下文与用户画像。

    Returns:
        包含 augmented_context、user_profile、retrieved_chunks 的字典；
        失败时返回空结构，不抛出异常（降级处理）。
    """
    url = f"{RAG_MEMORY_BASE}/api/v1/chat/memory/query"
    payload = {
        "user_id": user_id,
        "query": query,
        "time": datetime.now(timezone.utc).isoformat(),
    }
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        print(f"[RAG query 失败] {e}")
        return {"augmented_context": "", "user_profile": "", "retrieved_chunks": []}


async def upload_memory(user_id: str, messages: list[dict]) -> bool:
    """
    调用 RAG_Memory 上传接口，将本轮对话存入记忆系统。

    Args:
        user_id:  用户 ID。
        messages: [{"role": "user"|"assistant", "content": "..."}]

    Returns:
        True 表示上传成功，False 表示失败（不影响主流程）。
    """
    url = f"{RAG_MEMORY_BASE}/api/v1/chat/memory/upload"
    payload = {
        "user_id": user_id,
        "messages": messages,
        "time": datetime.now(timezone.utc).isoformat(),
    }
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return True
    except Exception as e:
        print(f"[RAG upload 失败] {e}")
        return False
