"""
app/core/logger.py
==================
双层日志配置模块：
  - 层 1（系统日志）：写入 logs/system/system.log，记录运行时事件与错误。
  - 层 2（对话日志）：写入 logs/conversations/{user_id}_YYYY-MM-DD.md，
                      记录每用户每天的对话明细，便于审计与追溯。

基于 loguru 实现，简洁且支持结构化。
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from loguru import logger as _logger

from app.core.config import settings


def _ensure_dirs() -> None:
    """确保日志目录存在，不存在则创建。"""
    Path(settings.log_dir, "system").mkdir(parents=True, exist_ok=True)
    Path(settings.log_dir, "conversations").mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    """
    初始化日志配置，挂载控制台 Sink 与系统文件 Sink。
    应在 FastAPI 应用启动时调用一次。
    """
    _ensure_dirs()

    # 清除 loguru 默认配置，避免重复输出
    _logger.remove()

    # ── Sink 1：控制台输出（开发友好的彩色格式）──────────────
    _logger.add(
        sys.stdout,
        level="DEBUG" if settings.debug else "INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
        enqueue=True,  # 异步写入，避免阻塞主线程
    )

    # ── Sink 2：系统日志文件（按天轮转，保留 30 天）───────────
    system_log_path = Path(settings.log_dir, "system", "system.log")
    _logger.add(
        str(system_log_path),
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
        rotation="00:00",    # 每天午夜轮转
        retention="30 days", # 保留最近 30 天
        compression="gz",    # 旧日志压缩存储
        encoding="utf-8",
        enqueue=True,
    )

    _logger.info("日志系统初始化完成。系统日志路径: {}", system_log_path)


def get_logger(name: str):
    """
    获取携带模块名称标签的 logger 实例。

    Args:
        name: 模块名，通常传入 __name__。

    Returns:
        loguru logger 绑定实例。
    """
    return _logger.bind(module=name)


def log_conversation(user_id: str, role: str, content: str) -> None:
    """
    将单条对话消息追加写入对应用户的每日对话日志文件。

    日志文件路径格式：logs/conversations/{user_id}_YYYY-MM-DD.md

    Args:
        user_id: 用户唯一标识符。
        role:    消息角色，如 "user" 或 "assistant"。
        content: 消息文本内容。
    """
    today = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_file = Path(settings.log_dir, "conversations", f"{user_id}_{today}.md")

    # 首次写入当天文件时，添加 Markdown 标题
    is_new = not log_file.exists()
    with open(log_file, "a", encoding="utf-8") as f:
        if is_new:
            f.write(f"# 对话日志 | 用户: {user_id} | 日期: {today}\n\n")
        f.write(f"**[{timestamp}] {role.upper()}**\n\n{content}\n\n---\n\n")


# 模块初始化时对外暴露的 logger 实例
logger = get_logger(__name__)
