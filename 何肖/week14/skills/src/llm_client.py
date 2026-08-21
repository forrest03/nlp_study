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
            "usage": {
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                "total_tokens": resp.usage.total_tokens if resp.usage else 0,
            },
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

    return f"""你是一个智能 Skill 执行助手。

你的任务是：

1. **识别用户意图**：分析用户输入，从下方可用的 Skills 中匹配最适合的 Skill
2. **调用工具执行**：根据匹配的 Skill 调用对应的工具
3. **总结执行结果**：工具执行完成后，用简洁的中文总结执行耗时和完成效果

## 严格规则

- **禁止直接回复文本**：在调用工具之前，你**绝对不能**直接回复文字内容。必须先调用工具，获取工具结果后再回复。
- 当用户请求股票查询时，**必须使用 compare_stock_tools 工具**，它会同时执行原版和优化版两个工具。
- 如果用户请求不明确，也要先调用 list_all_skills 或其他工具来获取信息，而不是直接回复。

## 可用 Skills

{skills_text}

## 工具选择策略

当用户请求股票查询时，**必须使用 compare_stock_tools 工具**，它会同时执行原版和优化版两个工具，并返回详细的对比数据，包括：
- 原版执行时间、状态、K线数量、判定结果
- 优化版执行时间、状态、K线数量、判定结果
- 加速比（优化版比原版快多少倍）

## 工作流程

1. 接收用户输入，判断属于哪个 Skill 的场景
2. **立即调用对应工具**（不要先回复文字）
3. 所有工具执行完毕后，**总结**：
   - 调用了哪些工具
   - 执行耗时
   - 完成状态（成功/失败/部分完成）
   - 关键结果数据
   - 输出文件路径（如有）

## 回复要求

- 全程使用中文
- 不要展示渐进式步骤引导
- 总结简洁明了，用列表呈现
- **必须包含对比分析**：当使用 compare_stock_tools 时，需明确列出：
  - 原版 vs 优化版 的执行时间对比
  - 原版 vs 优化版 的执行状态
  - 加速比
  - 哪个版本生成了可用的 HTML 看板链接
- 当工具返回了 generated_urls 字段时，必须在总结中以"📄 [文件名](URL)"的格式列出每个文件链接，方便用户点击访问
- 文件名应根据 URL 中的最后一部分来命名
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
