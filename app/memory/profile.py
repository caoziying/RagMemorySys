"""
app/memory/profile.py
=====================
用户画像管理模块：负责 user.md 文件的读取、写入与 LLM 驱动的自动更新。

文件路径规则：data/users/{user_id}/user.md
职责：
  - read_profile()                 : 异步读取用户画像文件内容。
  - write_profile()                : 异步将新画像内容写入文件。
  - extract_and_update_profile()   : 调用 LLM 从对话中提取信息并合并至画像（后台任务）。
"""

import asyncio
from pathlib import Path
from typing import Optional

import aiofiles

from app.core.config import settings
from app.core.exceptions import UserProfileError
from app.core.logger import get_logger
from app.prompts.extraction import (
    get_extraction_system_prompt,
    get_extraction_user_prompt,
    get_merge_system_prompt,
    get_merge_user_prompt,
)

logger = get_logger(__name__)


class ProfileManager:
    """
    用户画像管理器。

    负责 user.md 的持久化读写及基于 LLM 的自动信息提取与合并。
    每个用户的画像独立存储于 data/users/{user_id}/user.md。
    """

    def __init__(self) -> None:
        self._base_dir = Path(settings.data_dir) / "users"
        self._base_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ProfileManager 初始化 | base_dir={}", self._base_dir)

    def _get_profile_path(self, user_id: str) -> Path:
        """
        获取用户画像文件的绝对路径，并确保父目录存在。

        Args:
            user_id: 用户唯一标识符。

        Returns:
            Path 对象，指向 data/users/{user_id}/user.md。
        """
        user_dir = self._base_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir / "user.md"

    async def read_profile(self, user_id: str) -> str:
        """
        异步读取用户画像文件内容。

        Args:
            user_id: 用户唯一标识符。

        Returns:
            画像文件的文本内容；若文件不存在则返回空字符串。
        """
        profile_path = self._get_profile_path(user_id)

        if not profile_path.exists():
            logger.debug("用户画像不存在，返回空内容 | user_id={}", user_id)
            return ""

        try:
            async with aiofiles.open(profile_path, "r", encoding="utf-8") as f:
                content = await f.read()
            logger.debug("读取用户画像成功 | user_id={} | len={}", user_id, len(content))
            return content
        except Exception as e:
            logger.error("读取用户画像失败 | user_id={} | error={}", user_id, e)
            raise UserProfileError(detail=f"读取 user.md 失败: {e}")

    async def write_profile(self, user_id: str, content: str) -> None:
        """
        异步将新内容写入用户画像文件（全量覆盖）。

        Args:
            user_id: 用户唯一标识符。
            content: 新的画像 Markdown 文本内容。
        """
        profile_path = self._get_profile_path(user_id)

        try:
            async with aiofiles.open(profile_path, "w", encoding="utf-8") as f:
                await f.write(content)
            logger.info("用户画像已更新 | user_id={} | len={}", user_id, len(content))
        except Exception as e:
            logger.error("写入用户画像失败 | user_id={} | error={}", user_id, e)
            raise UserProfileError(detail=f"写入 user.md 失败: {e}")

    async def extract_and_update_profile(
        self, user_id: str, conversation_text: str
    ) -> None:
        """
        【后台异步任务】
        调用 LLM 从对话内容中提取用户信息，并与现有画像合并后写回文件。

        此方法设计为 FastAPI BackgroundTask，执行失败不会影响主请求响应。

        Args:
            user_id:           用户唯一标识符。
            conversation_text: 本轮对话的完整文本（多条消息拼接）。
        """
        # 避免循环导入，在函数内部导入
        from app.llm.client import call_llm_async

        logger.info("开始后台提取用户信息 | user_id={}", user_id)

        try:
            # Step 1: 读取现有画像
            existing_profile = await self.read_profile(user_id)

            # Step 2: 调用 LLM 提取新信息
            extraction_result = await call_llm_async(
                user_message=get_extraction_user_prompt(
                    conversation_text=conversation_text,
                    existing_profile=existing_profile,
                ),
                system_message=get_extraction_system_prompt(),
                max_tokens=1024,
            )

            # 若 LLM 返回无新信息的提示，跳过合并步骤
            if "无新增用户信息" in extraction_result or not extraction_result.strip():
                logger.info("本次对话无新增用户信息，跳过画像更新 | user_id={}", user_id)
                return

            logger.debug("LLM 提取结果 | user_id={} | extracted_len={}", user_id, len(extraction_result))

            # Step 3: 若已有画像，调用 LLM 合并；若无则直接使用提取结果
            if existing_profile.strip():
                merged_profile = await call_llm_async(
                    user_message=get_merge_user_prompt(
                        existing_profile=existing_profile,
                        new_info=extraction_result,
                    ),
                    system_message=get_merge_system_prompt(),
                    max_tokens=2048,
                )
            else:
                # 首次建立画像，无需合并
                merged_profile = f"# 用户画像\n\n{extraction_result}"

            # Step 4: 写回文件
            await self.write_profile(user_id=user_id, content=merged_profile)
            logger.info("用户画像后台更新完成 | user_id={}", user_id)

        except Exception as e:
            # 后台任务失败不上报，仅记录日志
            logger.error("后台画像更新失败 | user_id={} | error={}", user_id, str(e))
