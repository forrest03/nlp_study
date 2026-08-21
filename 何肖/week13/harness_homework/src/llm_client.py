import os
import json
import sys
from pathlib import Path
from typing import Optional

# Allow module-level imports
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from openai import OpenAI


def _get_api_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    if key:
        return key
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            val, _ = winreg.QueryValueEx(k, "DASHSCOPE_API_KEY")
            if val:
                os.environ["DASHSCOPE_API_KEY"] = val
                return val
    except Exception:
        pass
    return ""


DASHSCOPE_API_KEY = _get_api_key()
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
AGENT_MODEL = os.getenv("AGENT_MODEL", "qwen-max")

client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL,
)


def chat(
    messages: list[dict],
    model: Optional[str] = None,
    temperature: float = 0.7,
    tools: Optional[list[dict]] = None,
    tool_choice: Optional[str] = None,
) -> dict:
    if not DASHSCOPE_API_KEY:
        raise ValueError(
            "DASHSCOPE_API_KEY 环境变量未设置。\n"
            "  PowerShell: $env:DASHSCOPE_API_KEY='your_key'"
        )

    kwargs = {
        "model": model or AGENT_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        kwargs["tool_choice"] = tool_choice

    try:
        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        result = {
            "role": msg.role,
            "content": msg.content,
            "tool_calls": None,
        }
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return result
    except Exception as e:
        raise RuntimeError(f"DashScope 调用失败: {e}")


def build_system_prompt(skills_info: list[dict]) -> str:
    """构建系统提示词，告知 LLM 所有可用的 Skills 和工具，由 LLM 自动识别用户意图。"""
    skills_text = ""
    for s in skills_info:
        steps_line = ""
        if s.get("steps"):
            steps_line = "；".join(f"Step{st['step_num']}:{st['title']}" for st in s["steps"])
        skills_text += f"- **{s['name']}**: {s.get('description', '无描述')}"
        if steps_line:
            skills_text += f"（步骤: {steps_line}）"
        skills_text += "\n"

    return f"""你是一个智能 Skill 执行助手。你的任务是：

1. **识别用户意图**：分析用户输入，从下方可用的 Skills 中匹配最适合的 Skill
2. **调用工具执行**：根据匹配的 Skill 调用对应的工具
3. **总结执行结果**：工具执行完成后，用简洁的中文总结执行耗时和完成效果

## 可用 Skills

{skills_text}

## 工作流程

1. 接收用户输入，判断属于哪个 Skill 的场景
2. 调用对应 Skill 的工具（可能多次调用）
3. 所有工具执行完毕后，**只总结**：
   - 调用了哪些工具
   - 执行耗时
   - 完成状态（成功/失败/部分完成）
   - 关键结果数据
   - 输出文件路径（如有）

## 各 Skill 执行要点

### flash-card（单词闪卡）
当用户给出一个英语单词（如"给我做 crazy 的闪卡"）时：
1. 你必须利用自身语言知识，为该单词填写**完整**的学习数据：音标、词性、中文释义、恰好 3 条地道的中英对照例句、4-6 个近义词。
2. 先调用 `save_flashcard_data` 保存 JSON 数据到 `json_data/<word>.json`（所有字段都要填好，不要留空）。
3. 再调用 `generate_flashcard` 生成 HTML 闪卡到 `html_data/<word>.html`。
4. 两个工具依次调用，缺一不可。总结时只使用工具返回的 `generated_urls`，不要自己编造路径。

### stock-dashboard（股票看板）
当用户查询某只股票某日的行情时，直接调用 `fetch_stock`（company + date）即可，工具会自动完成数据获取与 HTML 生成。

## 回复要求

- 全程使用中文
- 不要展示渐进式步骤引导，直接执行
- 总结简洁明了，用列表呈现
- 工具执行中如遇参数不足，主动询问用户补充
- 如果用户输入无法匹配任何 Skill，礼貌告知并列出可用能力
- 当工具返回了 generated_urls 字段时，必须在总结中以"📄 [文件名](URL)"的格式列出每个文件链接，方便用户点击访问
- 文件名应根据 URL 中的最后一部分来命名，例如 URL "/files/skills/stock-dashboard/html_data/贵州茅台_2026-07-24.html" 应显示为 "贵州茅台_2026-07-24.html"
- 同时也要在总结中列出 JSON 数据文件的链接（如果有）"""


def build_skill_prompt(skill_name: str, skill_md: str, skill_info: dict) -> str:
    """为特定 Skill 构建系统提示词（兼容旧流程）。"""
    steps_text = ""
    for s in skill_info.get("steps", []):
        steps_text += f"- Step {s['step_num']}: {s['title']} — {s['description']}\n"

    return f"""你是一个 Skill 执行助手，通过工具调用来帮助用户完成任务。

## 当前 Skill: {skill_name}

### 描述:
{skill_info.get('description', '无')}

### 渐进式执行步骤:
{steps_text}

### 工作方式:
1. 分析用户意图，决定是否需要调用工具
2. 需要执行 Skill 脚本时，调用对应的工具
3. 获取工具执行结果后，用中文总结回复用户
4. 按步骤引导，缺失参数时主动询问
5. 最终给出简明的执行摘要

### 回复要求:
- 用中文回复
- 调用工具时先告知用户「正在执行...」
- 执行完成后总结结果
- 如果参数不足，询问用户补充"""
