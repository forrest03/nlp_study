"""
------------------------------------------------
预加载所有 SKILL.md 的 frontmatter 作为系统提示，
在对话中根据用户输入自动触发匹配的技能并执行。
使用 DeepSeek V4 Flash 作为对话模型。
"""

import json
import re
import subprocess
import sys
import webbrowser
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from openai import OpenAI
import yaml

# ========== 配置 ==========
SKILLS_ROOT = Path(__file__).parent / "skills"

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
MODEL_NAME = os.getenv("AGENT_MODEL", "deepseek-v4-flash")

# ========== 技能加载 ==========

def load_frontmatter(skill_dir: Path) -> Optional[Dict[str, str]]:
    """从 SKILL.md 中读取 YAML frontmatter（name, description）"""
    md_path = skill_dir / "SKILL.md"
    if not md_path.exists():
        return None
    content = md_path.read_text(encoding="utf-8")
    pattern = r"^---\n(.*?)\n---\n"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return None
    frontmatter = yaml.safe_load(match.group(1))
    return frontmatter


def load_full_skill(skill_dir: Path) -> str:
    """
    读取完整的 SKILL.md 内容（用于触发后加载）
    只返回 frontmatter 之后的部分（即正文内容），不包含 YAML 头部。
    """
    md_path = skill_dir / "SKILL.md"
    if not md_path.exists():
        return ""
    
    content = md_path.read_text(encoding="utf-8")
    # 匹配 frontmatter 块（--- 开头和结尾）
    pattern = r"^---\n.*?\n---\n(.*)$"
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        # 只返回 frontmatter 之后的正文
        return match.group(1).strip()
    else:
        # 如果没有 frontmatter，返回整个文件内容
        return content.strip()


def discover_skills() -> Dict[str, Dict]:
    """
    扫描 SKILLS_ROOT 下的所有技能目录，返回 {skill_name: {dir, frontmatter, full_md}}
    当前工作目录指运行脚本时所在的目录。
    每个技能目录必须包含 SKILL.md 文件。
    """
    skills = {}

    for item in SKILLS_ROOT.iterdir():
        if not item.is_dir():
            continue
        # 检查是否是技能目录（包含 SKILL.md）
        fm = load_frontmatter(item)
        if fm and "name" in fm and "description" in fm:
            skills[fm["name"]] = {
                "dir": item,
                "frontmatter": fm,
                "full_md": load_full_skill(item),
            }
    return skills


# ========== 技能执行函数（针对 flash-card） ==========
def generate_word_json(word: str, skill_md: str) -> Optional[Dict]:
    """
    使用完整的 SKILL.md 作为系统提示，让 LLM 生成符合规范的数据。
    skill_md 内容来自 flash-card 技能的 SKILL.md，已包含格式要求。
    """
    # 构建用户提示，简洁地指定单词，并明确要求只输出 JSON
    user_prompt = f'请为单词 "{word}" 生成学习卡片数据，严格按照 SKILL.md 中定义的 JSON 格式输出，仅输出 JSON，不要额外文字。'

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": skill_md},   # 全量注入
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content.strip()
        # 清理可能的 markdown 标记
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        data = json.loads(content)
        # 基本校验（SKILL.md 已保证格式，但保险起见）
        required = ["word", "phonetic", "pos", "definition", "examples", "synonyms"]
        if all(k in data for k in required) and len(data["examples"]) == 3:
            return data
        else:
            print("   ⚠️  LLM 返回的数据格式不完整，尝试重试...")
            return None
    except Exception as e:
        print(f"   ❌ LLM 调用失败: {e}")
        return None


# ========== 工具函数（原子操作） ==========

def tool_generate_word_json(word: str, skill_md: str) -> Dict:
    """调用 LLM 生成单词 JSON 数据（内部使用 SKILL.md 作为系统提示）"""
    print(f"   🔧 工具调用: generate_word_json(word='{word}')")
    user_prompt = f'请为单词 "{word}" 生成学习卡片数据，严格按照 SKILL.md 中定义的 JSON 格式输出，仅输出 JSON，不要额外文字。'
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": skill_md},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        data = json.loads(content)
        required = ["word", "phonetic", "pos", "definition", "examples", "synonyms"]
        if all(k in data for k in required) and len(data["examples"]) == 3:
            print(f"   ✅ 生成 JSON 成功: word={data['word']}")
            return data
        else:
            print("   ⚠️  格式不完整，返回空数据")
            return {}
    except Exception as e:
        print(f"   ❌ 生成 JSON 失败: {e}")
        return {}


def tool_save_json_data(data: Dict, word: str) -> str:
    """保存 JSON 文件到 data 目录，返回文件路径"""
    print(f"   🔧 工具调用: save_json_data(word='{word}')")
    data_dir = SKILLS_ROOT / "flash-card" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / f"{word}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"   ✅ JSON 已保存: {json_path}")
    return str(json_path)


def tool_run_make_script(json_path: str, word: str, output_dir: str = ".") -> str:
    """运行 make_flashcard.py 生成 HTML，返回 HTML 文件路径"""
    print(f"   🔧 工具调用: run_make_script(word='{word}')")
    script_path = SKILLS_ROOT / "flash-card" / "scripts" / "make_flashcard.py"
    if not script_path.exists():
        raise FileNotFoundError(f"脚本不存在: {script_path}")
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_html = out_dir / f"{word}.html"
    cmd = [sys.executable, str(script_path), json_path, "-o", str(output_html)]
    subprocess.run(cmd, check=True)
    print(f"   ✅ HTML 已生成: {output_html}")
    return str(output_html)


def tool_open_browser(html_path: str) -> str:
    """在默认浏览器中打开 HTML 文件"""
    print(f"   🔧 工具调用: open_browser(html_path='{html_path}')")
    webbrowser.open(html_path)
    print("   🌐 已打开浏览器预览")
    return f"已打开 {html_path}"


# ========== 工具定义（供 LLM 使用） ==========

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_word_json",
            "description": "为指定单词生成符合 SKILL.md 格式的 JSON 数据（含音标、词性、释义、3条例句、近义词）",
            "parameters": {
                "type": "object",
                "properties": {
                    "word": {"type": "string", "description": "英语单词"}
                },
                "required": ["word"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_json_data",
            "description": "将单词数据保存为 JSON 文件到 data 目录",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "object", "description": "单词数据对象"},
                    "word": {"type": "string", "description": "单词"}
                },
                "required": ["data", "word"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_make_script",
            "description": "执行 make_flashcard.py 脚本生成 HTML 闪卡",
            "parameters": {
                "type": "object",
                "properties": {
                    "json_path": {"type": "string", "description": "JSON 文件路径"},
                    "word": {"type": "string", "description": "单词"},
                    "output_dir": {"type": "string", "description": "输出目录，默认为当前目录"}
                },
                "required": ["json_path", "word"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_browser",
            "description": "在浏览器中打开生成的 HTML 闪卡",
            "parameters": {
                "type": "object",
                "properties": {
                    "html_path": {"type": "string", "description": "HTML 文件路径"}
                },
                "required": ["html_path"]
            }
        }
    }
]

# ========== 重构后的 execute_flash_card（Agent 循环） ==========

def execute_flash_card(word: str, skill_md: str, output_dir: str = "."):
    """
    使用 ReAct 循环完成闪卡制作。
    流程依据 SKILL.md，但由 LLM 自主调用工具一步步执行。
    """
    print(f"\n⚡ 执行技能: flash-card (单词: {word})")

    # 构建 Agent 系统提示：包含 SKILL.md 内容 + 工具使用说明
    agent_system = f"""{skill_md}

你是一个负责生成英语单词闪卡的 Agent。请按照以下流程，依次调用提供的工具完成卡片制作：
1. 调用 generate_word_json 生成单词数据。
2. 调用 save_json_data 保存数据。
3. 调用 run_make_script 生成 HTML。
4. 调用 open_browser 预览卡片。

每一步完成后，根据返回结果决定下一步。最终回复用户“卡片已生成”。
如果工具调用失败，请尝试重试或报告错误。
"""
    messages.append = [
        {"role": "system", "content": agent_system},
        {"role": "user", "content": f"请为单词 '{word}' 制作闪卡，输出目录为 '{output_dir}'。"}
    ]

    max_iterations = 10
    for _ in range(max_iterations):
        # 调用 LLM，允许使用工具
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.1,
        )
        assistant_message = response.choices[0].message
        messages.append(assistant_message.model_dump())  # 保存助手回复

        # 如果 LLM 没有要求调用工具，则认为任务完成
        if not assistant_message.tool_calls:
            print("🤖 Agent 回复:", assistant_message.content)
            break

        # 处理每个工具调用
        for tool_call in assistant_message.tool_calls:
            func_name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}

            # 根据函数名调用对应的 Python 函数
            result = None
            if func_name == "generate_word_json":
                result = tool_generate_word_json(args.get("word"), skill_md)
            elif func_name == "save_json_data":
                data = args.get("data")
                word_arg = args.get("word")
                if data and word_arg:
                    result = tool_save_json_data(data, word_arg)
                else:
                    result = "错误：缺少 data 或 word 参数"
            elif func_name == "run_make_script":
                json_path = args.get("json_path")
                word_arg = args.get("word")
                out_dir = args.get("output_dir", output_dir)
                if json_path and word_arg:
                    try:
                        result = tool_run_make_script(json_path, word_arg, out_dir)
                    except Exception as e:
                        result = f"运行脚本失败: {e}"
                else:
                    result = "错误：缺少 json_path 或 word 参数"
            elif func_name == "open_browser":
                html_path = args.get("html_path")
                if html_path:
                    result = tool_open_browser(html_path)
                else:
                    result = "错误：缺少 html_path 参数"
            else:
                result = f"未知工具: {func_name}"

            # 将工具调用结果作为 tool 消息返回给 LLM
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result) if result is not None else "执行完成"
            })

    else:
        print("⚠️  Agent 循环达到最大迭代次数，可能未完成任务。")

    print("🎉 闪卡制作流程结束。")

# ========== Harness 主循环 ==========

def build_system_prompt(skills: Dict[str, Dict]) -> str:
    """根据所有技能的 frontmatter 构建系统提示"""
    lines = [
        "你是一个智能助手，可以调用以下技能来帮助用户。",
        "请根据用户输入判断是否需要触发某个技能，如果需要，请以 JSON 格式返回你的操作：",
        "",
        "可用技能列表："
    ]
    for name, info in skills.items():
        fm = info["frontmatter"]
        lines.append(f"- 技能名: {name}")
        lines.append(f"  描述: {fm['description']}")
        lines.append("")
    lines.append("返回格式（仅当需要触发技能时）：")
    lines.append('{"action": "trigger_skill", "skill": "<技能名>", "params": {"word": "<提取的单词>"}}')
    lines.append('若无需触发，则返回 {"action": "chat", "reply": "你的回复内容"}')
    lines.append("只返回 JSON，不要其他文字。")
    return "\n".join(lines)


def parse_llm_response(content: str) -> Dict[str, Any]:
    """解析 LLM 返回的 JSON 动作"""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"action": "chat", "reply": "抱歉，我无法理解您的请求，请重试。"}

messages = []
def main():
    # 1. 发现所有技能
    skills = discover_skills()
    if not skills:
        print("未找到任何技能，请检查 .cursor/skills 目录结构。")
        sys.exit(1)

    system_prompt = build_system_prompt(skills)
    print("🤖 Harness 已启动 (使用 DeepSeek V4 Flash)，输入 'exit' 退出。\n")

    messages = [{"role": "system", "content": system_prompt}]

    while True:
        user_input = input("用户: ")
        if user_input.lower() in ("exit", "quit"):
            break

        messages.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2,
            )
            assistant_reply = response.choices[0].message.content
            print(f"🤖 (中间判断): {assistant_reply}")  # 调试时可保留

            action = parse_llm_response(assistant_reply)
            if action.get("action") == "trigger_skill":
                skill_name = action.get("skill")
                params = action.get("params", {})
                word = params.get("word")
                if skill_name == "flash-card" and word:
                    skill_info = skills.get(skill_name)
                    if skill_info:
                        full_md = skill_info["full_md"]
                        execute_flash_card(word, full_md)
                    else:
                        print(f"❌ 未知技能: {skill_name}")
                else:
                    print(f"❌ 技能 '{skill_name}' 参数缺失或不支持")
            else:
                reply = action.get("reply", "好的，我明白了。")
                print(f"🤖 {reply}")
                messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            print(f"❌ 错误: {e}")


if __name__ == "__main__":
    main()