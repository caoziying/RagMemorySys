"""
app/llm/client.py
=================
模型网关层：大语言模型客户端统一初始化与调用入口。

提供两种客户端：
  - client  (openai.OpenAI)      : 底层 OpenAI SDK，适合精细控制与流式输出。
  - llm     (langchain ChatOpenAI): LangChain 封装，适合链式调用与结构化输出。

⚠️ 严格遵循规范：必须读取 .env 中的自定义配置，不使用 OpenAI 默认地址。
"""

import os
from typing import List, Optional

from langchain_openai import ChatOpenAI
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.exceptions import LLMClientError
from app.core.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────
# 客户端实例化（严格按规范：读取 .env 配置）
# ──────────────────────────────────────────────────────────────

MY_API_KEY: str = settings.my_api_key
API_BASE_URL: str = settings.my_api_base
MODEL_NAME: str = settings.my_model

logger.info("LLM 网关初始化 | base_url={} | model={}", API_BASE_URL, MODEL_NAME)

# 底层 OpenAI 客户端：适用于需要精细控制的场景（如自定义流式处理）
client: OpenAI = OpenAI(
    api_key=MY_API_KEY,
    base_url=API_BASE_URL,
)

# LangChain 封装客户端：适用于链式调用、结构化输出、LCEL 管道
llm: ChatOpenAI = ChatOpenAI(
    model=MODEL_NAME,
    api_key=MY_API_KEY,
    base_url=API_BASE_URL,
    temperature=0.0,  # 信息提取场景需要确定性输出，温度设为 0
)


# ──────────────────────────────────────────────────────────────
# 通用调用接口（带重试与异常包装）
# ──────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=False,
)
async def call_llm_async(
    user_message: str,
    system_message: str = "",
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    """
    异步调用 LLM 并返回文本响应（封装重试与错误处理）。

    使用底层 OpenAI client 发起调用，适合大多数文本生成场景。

    Args:
        user_message:   用户消息内容。
        system_message: 系统 Prompt，若为空则不添加 system 角色。
        model:          指定模型名称，默认使用 MODEL_NAME。
        temperature:    采样温度，0.0 为确定性输出。
        max_tokens:     最大输出 token 数。

    Returns:
        LLM 生成的文本字符串。

    Raises:
        LLMClientError: 在所有重试失败后抛出。
    """
    target_model = model or MODEL_NAME
    messages: List[dict] = []

    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})

    try:
        logger.debug("调用 LLM | model={} | msg_len={}", target_model, len(user_message))

        # 使用同步 client 在线程池中运行（FastAPI 异步环境下的最佳实践）
        import asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )

        result = response.choices[0].message.content or ""
        logger.debug("LLM 调用成功 | output_len={}", len(result))
        return result

    except Exception as e:
        logger.error("LLM 调用失败 | model={} | error={}", target_model, str(e))
        raise LLMClientError(detail=str(e))


def call_llm_sync(
    user_message: str,
    system_message: str = "",
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    """
    同步调用 LLM 并返回文本响应（适用于非异步上下文）。

    Args:
        user_message:   用户消息内容。
        system_message: 系统 Prompt。
        model:          指定模型名称，默认使用 MODEL_NAME。
        temperature:    采样温度。
        max_tokens:     最大输出 token 数。

    Returns:
        LLM 生成的文本字符串。若调用失败则返回空字符串（降级）。
    """
    target_model = model or MODEL_NAME
    messages: List[dict] = []

    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model=target_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error("同步 LLM 调用失败 | model={} | error={}", target_model, str(e))
        return ""  # 降级：返回空字符串，不中断调用方
