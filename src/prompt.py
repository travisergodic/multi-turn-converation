SYSTEM_PROMPT_TEMPLATE = """你是一个有记忆能力的智能助手。

{profile_section}{memory_section}请根据以上背景信息，自然地回应用户。"""

PROFILE_SECTION_TEMPLATE = """## 用户档案
{profile_text}

"""

MEMORY_SECTION_TEMPLATE = """## 相关记忆
{memory_text}

"""

MEMORY_UPDATE_PROMPT = """你是一个记忆管理助手。根据以下对话，判断是否需要更新长期记忆。

## 现有相关记忆
{existing_memories}

## 最新一轮对话
用户：{user_message}
助手：{assistant_message}

请输出一个 JSON 数组，每项包含以下字段：
- action: "add" | "append" | "delete" | "noop"
- memory_id: 目标记忆的 ID（add 时为 null）
- content: 新增或追加的内容（delete/noop 时为 null）

只输出 JSON 数组，不要其他内容。"""

PROFILE_UPDATE_PROMPT = """你是一个用户档案管理助手。根据最新对话，判断是否需要更新用户档案。

## 当前用户档案
{current_profile}

## 最新一轮对话
用户：{user_message}
助手：{assistant_message}

如果有字段需要更新，输出更新后的完整 JSON 档案；如无需更新，输出 null。
只输出 JSON 或 null，不要其他内容。"""
