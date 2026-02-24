"""
chat.py - 带 RAG 增强的 LLM 对话逻辑（流式输出）

流程：
  1. 调用 RAG_Memory query 接口获取增强上下文
  2. 构造带记忆上下文的系统 Prompt
  3. 调用 DeepSeek（OpenAI 兼容接口）流式生成回答
  4. 流式 yield token
  5. 完整回答生成后，异步触发 RAG_Memory upload
"""
import asyncio
import json
import os
from typing import AsyncGenerator

from openai import AsyncOpenAI

from app.rag_client import query_memory, upload_memory

API_KEY = os.getenv("MY_API_KEY", "")
API_BASE = os.getenv("MY_API_BASE", "https://api.chat.csu.edu.cn/v1")
MODEL = os.getenv("MY_MODEL", "deepseek-v3-thinking")

_llm = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE)

SYSTEM_PROMPT_TEMPLATE = """你是一个智能个人助手，拥有持久记忆能力。你能记住用户过去的对话内容和个人信息，并在回答时加以利用，提供个性化的回答。

{memory_section}

请根据以上记忆信息，结合当前对话，给出准确、个性化的回答。若记忆中没有相关信息，则直接回答即可。"""


def _build_system_prompt(augmented_context: str) -> str:
    if augmented_context and augmented_context.strip() and "暂无" not in augmented_context:
        memory_section = f"## 你关于该用户的记忆\n\n{augmented_context}"
    else:
        memory_section = "## 记忆状态\n\n（该用户暂无历史记忆，这是全新对话）"
    return SYSTEM_PROMPT_TEMPLATE.format(memory_section=memory_section)


async def chat_stream(
    user_id: str,
    query: str,
    history: list[dict],
) -> AsyncGenerator[str, None]:
    """
    流式对话生成器：RAG 增强 → LLM 流式输出 → 后台 upload 记忆。

    Args:
        user_id: 用户 ID（同时作为 RAG_Memory 的 user_id）。
        query:   用户当前输入。
        history: 本次对话的历史消息列表（不含当前 query）。

    Yields:
        SSE 格式的字符串，每帧为 "data: {json}\n\n"
        结束帧为 "data: [DONE]\n\n"
    """
    # Step 1: 并发获取 RAG 上下文（不阻塞，超时则降级）
    try:
        rag_result = await asyncio.wait_for(query_memory(user_id, query), timeout=10.0)
        augmented_context = rag_result.get("augmented_context", "")
    except asyncio.TimeoutError:
        augmented_context = ""

    # Step 2: 构造消息列表
    system_prompt = _build_system_prompt(augmented_context)
    messages = [{"role": "system", "content": system_prompt}]

    # 加入历史消息（最多保留最近 20 条，避免 context 过长）
    recent_history = history[-20:] if len(history) > 20 else history
    messages.extend(recent_history)
    messages.append({"role": "user", "content": query})

    # Step 3: 流式调用 LLM
    # Step 3: 流式调用 LLM
    full_response = ""
    try:
        stream = await _llm.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=4096,
        )

        async for chunk in stream:
            # 【关键修复 1】防御性判断：如果 choices 为空，直接跳过当前 chunk
            if not chunk.choices:
                continue
                
            delta = chunk.choices[0].delta
            
            # 【关键修复 2】确保 content 不是 None 再拼接
            if delta.content:
                full_response += delta.content
                # SSE 格式输出
                yield f"data: {json.dumps({'content': delta.content}, ensure_ascii=False)}\n\n"

    except Exception as e:
        error_msg = f"LLM 调用失败：{str(e)}"
        yield f"data: {json.dumps({'content': error_msg, 'error': True}, ensure_ascii=False)}\n\n"
        full_response = error_msg
    # full_response = ""
    # try:
    #     stream = await _llm.chat.completions.create(
    #         model=MODEL,
    #         messages=messages,
    #         stream=True,
    #         temperature=0.7,
    #         max_tokens=4096,
    #     )

    #     async for chunk in stream:
    #         delta = chunk.choices[0].delta
    #         if delta.content:
    #             full_response += delta.content
    #             # SSE 格式输出
    #             yield f"data: {json.dumps({'content': delta.content}, ensure_ascii=False)}\n\n"

    # except Exception as e:
    #     error_msg = f"LLM 调用失败：{str(e)}"
    #     yield f"data: {json.dumps({'content': error_msg, 'error': True}, ensure_ascii=False)}\n\n"
    #     full_response = error_msg

    yield "data: [DONE]\n\n"

    # Step 4: 后台异步上传本轮对话到 RAG_Memory（不阻塞流式响应）
    if full_response and "LLM 调用失败" not in full_response:
        asyncio.create_task(
            upload_memory(
                user_id=user_id,
                messages=[
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": full_response},
                ],
            )
        )
