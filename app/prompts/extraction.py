"""
app/prompts/extraction.py
=========================
用户个人信息提取 Prompt 模板。

【解耦原则】
  - 本文件只存放 Prompt 字符串模板，不包含任何业务逻辑。
  - 业务代码（profile.py）通过调用本模块的函数获取格式化后的 Prompt。
  - 修改提示词策略时只需修改本文件，不影响业务代码。
"""

from string import Template


# ──────────────────────────────────────────────────────────────
# 系统 Prompt：设定 LLM 角色与行为约束
# ──────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """你是一个专业的用户信息分析助手。
你的唯一任务是从用户的对话内容中，精准提取与该用户相关的个人信息、偏好、背景和重要事实。

【提取规则】
1. 只提取对话中明确出现或可被高置信度推断的信息，不要捏造或过度推断。
2. 以 Markdown 格式输出，使用二级标题（##）组织类别，条目用 `-` 列举。
3. 若某类别无有效信息，则完全省略该类别，不输出空条目。
4. 信息要简洁、客观，避免冗余描述。
5. 如果对话中完全没有可提取的个人信息，仅输出：`（本次对话无新增用户信息）`

【可提取的信息类别示例（不限于此）】
- 姓名 / 称谓
- 职业 / 工作单位
- 技能 / 专业领域
- 个人偏好（技术栈、工具、习惯等）
- 正在进行的项目或任务
- 明确表达的目标或计划
- 重要的个人背景（教育经历、地区等）
"""

# ──────────────────────────────────────────────────────────────
# 用户 Prompt 模板（使用 Python 标准库 string.Template）
# 占位符：$conversation_text, $existing_profile
# ──────────────────────────────────────────────────────────────

_EXTRACTION_USER_TEMPLATE = Template(
    """请从以下对话内容中提取用户信息。

===== 当前已有用户画像 =====
$existing_profile

===== 本次对话内容 =====
$conversation_text

===== 提取要求 =====
请仅提取【新增或需要更新】的信息条目，避免与已有画像重复。
以 Markdown 格式直接输出提取结果，无需任何前言或解释。
"""
)

# ──────────────────────────────────────────────────────────────
# 合并已有画像与新提取信息的 Prompt
# ──────────────────────────────────────────────────────────────

_MERGE_SYSTEM_PROMPT = """你是一个用户画像整合助手。
你的任务是将【已有用户画像】与【新提取的用户信息】合并为一份更完整、无重复的用户画像。

【合并规则】
1. 以 Markdown 格式输出最终画像，使用二级标题（##）组织类别。
2. 若新信息与已有信息冲突，优先采用新信息并注明更新。
3. 去除重复条目，保持画像简洁。
4. 直接输出合并后的完整画像内容，不需要解释或前言。
"""

_MERGE_USER_TEMPLATE = Template(
    """请将以下两部分信息合并为一份完整的用户画像：

===== 已有用户画像 =====
$existing_profile

===== 新提取的用户信息 =====
$new_info

请直接输出合并后的完整 Markdown 格式用户画像：
"""
)


# ──────────────────────────────────────────────────────────────
# 公共接口函数
# ──────────────────────────────────────────────────────────────

def get_extraction_system_prompt() -> str:
    """
    返回信息提取任务的系统 Prompt。

    Returns:
        系统 Prompt 字符串。
    """
    return EXTRACTION_SYSTEM_PROMPT


def get_extraction_user_prompt(
    conversation_text: str,
    existing_profile: str = "（暂无历史画像）",
) -> str:
    """
    根据对话内容与现有画像，生成信息提取任务的用户 Prompt。

    Args:
        conversation_text: 本轮对话的完整文本。
        existing_profile:  当前 user.md 的内容，默认提示暂无。

    Returns:
        格式化后的用户 Prompt 字符串。
    """
    return _EXTRACTION_USER_TEMPLATE.substitute(
        conversation_text=conversation_text.strip(),
        existing_profile=existing_profile.strip() or "（暂无历史画像）",
    )


def get_merge_system_prompt() -> str:
    """
    返回画像合并任务的系统 Prompt。

    Returns:
        系统 Prompt 字符串。
    """
    return _MERGE_SYSTEM_PROMPT


def get_merge_user_prompt(existing_profile: str, new_info: str) -> str:
    """
    生成画像合并任务的用户 Prompt。

    Args:
        existing_profile: 当前 user.md 内容。
        new_info:         本次提取到的新用户信息。

    Returns:
        格式化后的合并 Prompt 字符串。
    """
    return _MERGE_USER_TEMPLATE.substitute(
        existing_profile=existing_profile.strip() or "（暂无历史画像）",
        new_info=new_info.strip() or "（无新信息）",
    )
