"""
app/memory/manager.py
=====================
基础记忆管理模块：维护每个用户的本地对话历史文件，
实现滑动窗口保留与超阈值自动压缩（摘要）机制。

文件结构：
  data/users/{user_id}/history.jsonl    - 完整对话历史（JSONL 格式，每行一条消息）
  data/users/{user_id}/compressed.md    - 历史摘要文件（压缩后的旧对话记忆）
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import aiofiles

from app.core.config import settings
from app.core.logger import get_logger
from app.prompts.summarization import (
    get_incremental_summarization_system_prompt,
    get_incremental_summarization_user_prompt,
    get_summarization_system_prompt,
    get_summarization_user_prompt,
)

logger = get_logger(__name__)


class MemoryManager:
    """
    基础记忆管理器：滑动窗口 + 自动压缩。

    核心策略：
      - 保留最近 MEMORY_WINDOW_SIZE 条对话消息（滑动窗口）。
      - 当历史消息超过 MEMORY_COMPRESS_THRESHOLD 条时，触发 LLM 压缩，
        将旧消息摘要追加至 compressed.md，并清空旧历史。
    """

    def __init__(self) -> None:
        self._base_dir = Path(settings.data_dir) / "users"
        self._window_size = settings.memory_window_size
        self._compress_threshold = settings.memory_compress_threshold
        self._base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "MemoryManager 初始化 | window_size={} | compress_threshold={}",
            self._window_size, self._compress_threshold,
        )

    # ── 路径辅助 ────────────────────────────────────────────

    def _history_path(self, user_id: str) -> Path:
        """返回用户对话历史文件路径（JSONL 格式）。"""
        user_dir = self._base_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir / "history.jsonl"

    def _compressed_path(self, user_id: str) -> Path:
        """返回用户历史摘要文件路径。"""
        user_dir = self._base_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir / "compressed.md"

    # ── 核心公共方法 ─────────────────────────────────────────

    async def update(self, user_id: str, new_texts: List[str]) -> None:
        """
        将新消息追加至历史文件，并在必要时触发压缩。

        Args:
            user_id:   用户唯一标识符。
            new_texts: 本次新增的消息文本列表。
        """
        # 构造带时间戳的消息条目
        timestamp = datetime.now(timezone.utc).isoformat()
        entries = [
            {"text": text, "timestamp": timestamp}
            for text in new_texts
        ]

        # 追加写入 history.jsonl
        await self._append_history(user_id, entries)

        # 检查是否需要压缩
        all_history = await self._read_all_history(user_id)
        if len(all_history) >= self._compress_threshold:
            logger.info(
                "历史消息数 {} >= 阈值 {}，触发压缩 | user_id={}",
                len(all_history), self._compress_threshold, user_id,
            )
            await self._compress(user_id, all_history)

    async def get_recent_history(self, user_id: str) -> List[str]:
        """
        获取最近 N 条对话历史文本（滑动窗口）。

        Args:
            user_id: 用户唯一标识符。

        Returns:
            最近 MEMORY_WINDOW_SIZE 条消息的文本列表，按时间升序排列。
        """
        all_history = await self._read_all_history(user_id)
        recent = all_history[-self._window_size:]
        return [entry["text"] for entry in recent]

    async def get_compressed_summary(self, user_id: str) -> str:
        """
        读取用户的历史压缩摘要。

        Args:
            user_id: 用户唯一标识符。

        Returns:
            compressed.md 的文本内容；若不存在则返回空字符串。
        """
        compressed_path = self._compressed_path(user_id)
        if not compressed_path.exists():
            return ""
        try:
            async with aiofiles.open(compressed_path, "r", encoding="utf-8") as f:
                return await f.read()
        except Exception as e:
            logger.error("读取 compressed.md 失败 | user_id={} | error={}", user_id, e)
            return ""

    # ── 内部方法 ─────────────────────────────────────────────

    async def _append_history(self, user_id: str, entries: List[dict]) -> None:
        """以追加模式将消息条目写入 history.jsonl。"""
        history_path = self._history_path(user_id)
        try:
            async with aiofiles.open(history_path, "a", encoding="utf-8") as f:
                for entry in entries:
                    await f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error("追加历史记录失败 | user_id={} | error={}", user_id, e)

    async def _read_all_history(self, user_id: str) -> List[dict]:
        """读取 history.jsonl 中的所有条目。"""
        history_path = self._history_path(user_id)
        if not history_path.exists():
            return []

        entries: List[dict] = []
        try:
            async with aiofiles.open(history_path, "r", encoding="utf-8") as f:
                async for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            logger.warning("JSONL 行解析失败，跳过 | user_id={}", user_id)
        except Exception as e:
            logger.error("读取历史记录失败 | user_id={} | error={}", user_id, e)

        return entries

    async def _compress(self, user_id: str, all_history: List[dict]) -> None:
        """
        触发 LLM 压缩：
          1. 将旧消息（超出窗口的部分）与现有摘要合并为新摘要。
          2. 将新摘要写入 compressed.md。
          3. 清除历史文件，只保留最近 WINDOW_SIZE 条消息。
        """
        from app.llm.client import call_llm_async

        # 分割：旧消息（将被压缩）和最近消息（将被保留）
        old_entries = all_history[: -self._window_size]
        recent_entries = all_history[-self._window_size :]

        old_text = "\n".join(
            f"[{e.get('timestamp', '')}] {e.get('text', '')}"
            for e in old_entries
        )

        # 读取现有摘要（增量合并）
        existing_summary = await self.get_compressed_summary(user_id)

        try:
            if existing_summary.strip():
                # 增量合并：将旧摘要与新旧消息合并
                new_summary = await call_llm_async(
                    user_message=get_incremental_summarization_user_prompt(
                        existing_summary=existing_summary,
                        new_conversation=old_text,
                        target_length=400,
                    ),
                    system_message=get_incremental_summarization_system_prompt(),
                    max_tokens=1024,
                )
            else:
                # 首次压缩
                new_summary = await call_llm_async(
                    user_message=get_summarization_user_prompt(
                        conversation_history=old_text,
                        target_length=300,
                    ),
                    system_message=get_summarization_system_prompt(),
                    max_tokens=1024,
                )
        except Exception as e:
            logger.error("LLM 压缩调用失败，跳过压缩 | user_id={} | error={}", user_id, e)
            return

        # 写入新摘要
        compressed_path = self._compressed_path(user_id)
        try:
            async with aiofiles.open(compressed_path, "w", encoding="utf-8") as f:
                await f.write(f"# 历史对话摘要\n\n{new_summary}\n")
            logger.info("压缩摘要写入成功 | user_id={}", user_id)
        except Exception as e:
            logger.error("写入 compressed.md 失败 | user_id={} | error={}", user_id, e)
            return

        # 重置 history.jsonl，仅保留最近 WINDOW_SIZE 条
        history_path = self._history_path(user_id)
        try:
            async with aiofiles.open(history_path, "w", encoding="utf-8") as f:
                for entry in recent_entries:
                    await f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.info(
                "历史记录压缩完成，保留最近 {} 条 | user_id={}",
                len(recent_entries), user_id,
            )
        except Exception as e:
            logger.error("重写 history.jsonl 失败 | user_id={} | error={}", user_id, e)
