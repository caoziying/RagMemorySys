"""
app/prompts/summarization.py
============================
对话历史压缩/摘要 Prompt 模板。

【解耦原则】
  - 本文件只存放 Prompt 字符串模板，不包含任何业务逻辑。
  - 业务代码（memory/manager.py）通过调用本模块的函数获取格式化后的 Prompt。
  - 修改摘要策略时只需修改本文件，不影响业务代码。
"""

from string import Template


# ──────────────────────────────────────────────────────────────
# 对话历史压缩摘要 Prompt
# ──────────────────────────────────────────────────────────────

SUMMARIZATION_SYSTEM_PROMPT = """你是一个专业的对话历史整理助手。
你的任务是将一段较长的多轮对话历史压缩为一份简洁、信息密集的摘要，
供后续对话中作为背景记忆使用。

【压缩规则】
1. 保留所有关键信息：用户的问题、决策、重要结论、提到的项目/任务状态。
2. 省略礼貌用语、重复内容、无实质信息的对话轮次。
3. 以第三人称描述，格式为连贯的段落文字（非列表），保持可读性。
4. 摘要长度控制在原文的 20%~30%，避免过度压缩导致信息丢失。
5. 若对话中存在多个主题，用自然语言过渡，不要强制分节。
6. 直接输出摘要内容，不要包含任何前言、解释或标注。
"""

_SUMMARIZATION_USER_TEMPLATE = Template(
    """请将以下多轮对话历史压缩为一份简洁的记忆摘要：

===== 待压缩的对话历史 =====
$conversation_history

===== 输出要求 =====
- 以流畅的段落形式输出摘要
- 重点保留：用户意图、关键决策、项目进展、重要事实
- 摘要长度目标：约 $target_length 字
- 直接输出摘要，无需任何前言
"""
)

# ──────────────────────────────────────────────────────────────
# 增量摘要合并 Prompt（将旧摘要与新对话合并为新摘要）
# ──────────────────────────────────────────────────────────────

_INCREMENTAL_SUMMARIZATION_SYSTEM_PROMPT = """你是一个对话记忆管理助手。
你的任务是将一份【已有的历史摘要】与【新增的对话内容】整合为一份更新后的摘要，
确保旧记忆与新内容无缝融合，不丢失重要信息。

【整合规则】
1. 已有摘要中的信息默认有效，除非新对话明确推翻。
2. 新对话中的重要信息需融入摘要，无需逐字保留原文。
3. 整合后的摘要应比两部分之和更简洁，去除过时或重复内容。
4. 保持第三人称、段落形式输出。
5. 直接输出整合后的摘要，无需前言。
"""

_INCREMENTAL_SUMMARIZATION_USER_TEMPLATE = Template(
    """请将以下已有摘要与新对话内容整合为一份更新后的摘要：

===== 已有历史摘要 =====
$existing_summary

===== 新增对话内容 =====
$new_conversation

===== 输出要求 =====
- 输出整合后的完整摘要（段落格式）
- 目标长度：约 $target_length 字
- 直接输出，无需解释
"""
)


# ──────────────────────────────────────────────────────────────
# 公共接口函数
# ──────────────────────────────────────────────────────────────

def get_summarization_system_prompt() -> str:
    """
    返回对话历史全量压缩任务的系统 Prompt。

    Returns:
        系统 Prompt 字符串。
    """
    return SUMMARIZATION_SYSTEM_PROMPT


def get_summarization_user_prompt(
    conversation_history: str,
    target_length: int = 300,
) -> str:
    """
    生成对话历史全量压缩任务的用户 Prompt。

    Args:
        conversation_history: 待压缩的多轮对话文本。
        target_length:        目标摘要字数，默认 300 字。

    Returns:
        格式化后的用户 Prompt 字符串。
    """
    return _SUMMARIZATION_USER_TEMPLATE.substitute(
        conversation_history=conversation_history.strip(),
        target_length=str(target_length),
    )


def get_incremental_summarization_system_prompt() -> str:
    """
    返回增量摘要合并任务的系统 Prompt。

    Returns:
        系统 Prompt 字符串。
    """
    return _INCREMENTAL_SUMMARIZATION_SYSTEM_PROMPT


def get_incremental_summarization_user_prompt(
    existing_summary: str,
    new_conversation: str,
    target_length: int = 400,
) -> str:
    """
    生成增量摘要合并任务的用户 Prompt。

    Args:
        existing_summary:  已有的历史摘要文本（compressed.md 内容）。
        new_conversation:  本次新增的对话文本。
        target_length:     整合后的目标摘要字数，默认 400 字。

    Returns:
        格式化后的用户 Prompt 字符串。
    """
    return _INCREMENTAL_SUMMARIZATION_USER_TEMPLATE.substitute(
        existing_summary=existing_summary.strip() or "（暂无历史摘要）",
        new_conversation=new_conversation.strip(),
        target_length=str(target_length),
    )
