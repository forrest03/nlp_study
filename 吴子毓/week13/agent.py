"""
agent.py — Harness 编排层（CLI 主入口）
========================================
职责：主循环、两阶段匹配、渐进式加载编排、skill 执行分发、进度打印。

对标原项目（仅借鉴设计，不 import 任何外部模块，homework 完全自包含）：
  - agent_memory_system/src/agent.py            → main() 主循环 + print_layer_info 进度展示
  - agent_memory_system/src/heartbeat_parser.py → SCHEDULE_PATTERNS 正则初筛 + analyze_and_write LLM 二次确认 + _parse_json_safe
  - agent_memory_system/src/scheduler.py        → _execute_task 的 action 分发 + broadcast 事件
  - agent_memory_system/src/llm_config.py       → get_chat_client 的 PROVIDERS 模式（内联精简版，不 import）

设计原则：
  homework 仅依赖同级 skills/ 目录 + pip 包（openai），不关联 agent_memory_system/src。
  把能内联的都内联进本文件——主循环、匹配、执行、进度、LLM 客户端，全在此。
"""
from __future__ import annotations  # 兼容 Python 3.8 的 str | None 等类型注解

# ════════════════════════════════════════════════════════════════════════════
# 模块 1：导入与路径设置（homework 自包含，不 import agent_memory_system/src）
# ════════════════════════════════════════════════════════════════════════════
# - import os / sys / re / json / subprocess / logging
# - from pathlib import Path
# - Windows OpenMP 兼容：os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# - sys.path.insert(0, str(Path(__file__).parent))
#   → 以便能 `from skill_loader import build_registry, ...`（同目录 import）
#
# LLM 客户端（内联，对标原 llm_config.py 的 PROVIDERS 模式但精简）：
#   - 仅依赖 pip 包 openai，不 import 任何外部 src 模块；
#   - 支持 DeepSeek / Qwen 两家，环境变量切换：
#       LLM_PROVIDER=deepseek  →  DEEPSEEK_API_KEY=sk-xxx  （默认）
#       LLM_PROVIDER=qwen      →  DASHSCOPE_API_KEY=sk-xxx
#   - _LLM_OK 在导入时检查 openai 包 + API Key 是否就绪；
#   - get_chat_client() 运行时读环境变量构造 OpenAI client。

import os
import sys
import re
import json
import shutil
import subprocess
import logging
from pathlib import Path

# Windows OpenMP 兼容（照搬原 agent.py）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 让 skill_loader 可被 import（同目录，homework 自包含）
sys.path.insert(0, str(Path(__file__).parent))

# ── LLM 客户端（内联精简版，对标原 llm_config.py 的 PROVIDERS 模式）──────────
_PROVIDERS = {
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url":    "https://api.deepseek.com",
        "chat_model":  "deepseek-chat",
    },
    "qwen": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url":    "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "chat_model":  "qwen3.7-max",
    },
}

_LLM_OK = False
try:
    from openai import OpenAI
    # 检查任一 provider 的 API Key 是否已设置（不依赖 LLM_PROVIDER 默认值）
    if os.getenv("DEEPSEEK_API_KEY") or os.getenv("DASHSCOPE_API_KEY"):
        _LLM_OK = True
except ImportError:
    pass


def get_chat_client():
    """
    返回 (OpenAI client, model_name)，API Key 缺失时 raise EnvironmentError。
    provider 选择优先级：已设置 API Key 的 provider > 报错（DASHSCOPE 优先于 DEEPSEEK）。
    """
    if os.getenv("DASHSCOPE_API_KEY"):
        provider = "qwen"
    elif os.getenv("DEEPSEEK_API_KEY"):
        provider = "deepseek"
    else:
        raise EnvironmentError("需要设置 DEEPSEEK_API_KEY 或 DASHSCOPE_API_KEY")
    cfg = _PROVIDERS[provider]
    api_key = os.getenv(cfg["api_key_env"])
    if not api_key:
        raise EnvironmentError(f"需要设置环境变量 {cfg['api_key_env']}")
    client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
    return client, cfg["chat_model"]


from skill_loader import build_registry, SkillRegistry, SkillMeta, LoadedSkill

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


# ════════════════════════════════════════════════════════════════════════════
# 模块 2：Stage 1 正则初筛模式表（对标 SCHEDULE_PATTERNS / CANCEL_PATTERNS）
# ════════════════════════════════════════════════════════════════════════════
#
# 这是"两阶段匹配"的廉价第一关——零 LLM 成本筛掉明显无关的输入。
# 与原项目 heartbeat_parser 用固定正则表不同，本 harness 的 patterns 动态生成：
# 从已注册 skill 的 description 里抽关键词建正则，让 patterns 随 skill 增删自动更新。
#
# def build_stage1_patterns(metas: list[SkillMeta]) -> list[tuple[str, re.Pattern]]:
#     """
#     根据当前注册表动态生成 Stage 1 正则表。
#     返回 [(skill_name, compiled_pattern), ...]。
#     """
#     关键细节：
#       - 对每个 skill 的 description，按以下规则抽词建正则：
#         * 中文：直接取 2~4 字关键词（如"闪卡""图表""架构图"）；
#         * 英文：取词根（如 "flash card" / "diagram"），大小写不敏感；
#         * 触发短语：从 description 里"当用户..."之后抽取示例短语（参考 flash-card SKILL.md
#           的"触发场景"段写了"给我做张 crazy 词的闪卡"这类自然语言）；
#       - 每词编译一个 re.compile(r'闪卡|flash\s*card', re.IGNORECASE)；
#       - 也保留几个通用的"动作词"正则：画图/生成/做一张/做一张...卡；
#       - 对标 heartbeat_parser.SCHEDULE_PATTERNS 的"多正则任一命中即过初筛"语义；
#       - 动态生成的好处：新增 skill 不用改代码，reload_if_changed 后 patterns 自动重建。
#
# def stage1_filter(user_input: str, patterns: list) -> list[str]:
#     """返回 Stage 1 命中的 skill_name 候选列表（去重）。"""
#     关键细节：
#       - any(p.search(user_input) for _, p in patterns) 逐条扫；
#       - 命中 0 个 → 返回 []，主循环走"普通对话"分支（不调 LLM，省 token）；
#       - 命中多个 → 全部作为候选传给 Stage 2，由 LLM 定夺；
#       - 对标 heartbeat_parser.may_contain_schedule_intent() 的布尔初筛，但返回候选列表而非布尔。


def build_stage1_patterns(metas: list) -> list:
    """
    根据当前注册表动态生成 Stage 1 正则表。
    对每个 skill 的 description 抽中英文关键词 + skill name，合并成一条正则。
    """
    patterns = []
    for meta in metas:
        keywords = set()
        # skill name 本身就是强信号（如 "flash-card" / "baoyu-diagram"）
        keywords.add(meta.name)
        # 中文关键词：连续 2~4 个汉字（如"闪卡""图表""架构图"）
        for m in re.finditer(r"[\u4e00-\u9fff]{2,4}", meta.description):
            keywords.add(m.group())
        # 英文关键词：3 字符以上的单词（如 "flash" "card" "diagram" "svg"）
        for m in re.finditer(r"[a-zA-Z]{3,}", meta.description):
            keywords.add(m.group())

        # 去重 + 转义，合并成一条"任一命中"的正则
        escaped = [re.escape(k) for k in keywords if k]
        if escaped:
            pat = re.compile("|".join(escaped), re.IGNORECASE)
            patterns.append((meta.name, pat))
    return patterns


def stage1_filter(user_input: str, patterns: list) -> list:
    """返回 Stage 1 命中的 skill_name 候选列表（去重，保序）。"""
    candidates = []
    seen = set()
    for skill_name, pat in patterns:
        if skill_name in seen:
            continue
        if pat.search(user_input):
            candidates.append(skill_name)
            seen.add(skill_name)
    return candidates


# ════════════════════════════════════════════════════════════════════════════
# 模块 3：ANSI 颜色常量 + 进度打印（对标 agent.print_layer_info）
# ════════════════════════════════════════════════════════════════════════════
#
# RESET/BOLD/CYAN/GREEN/YELLOW/MAGENTA/DIM = "\033[xxm" 系列
#   —— 从原 agent.py 顶部照搬，保证 CLI 输出风格一致。
#
# def print_registry(metas: list[SkillMeta]):
#     """启动时打印已注册 skill 清单，让"渐进式加载第 1 层"可见。"""
#     关键细节：
#       - 仿原 agent.print_layer_info() 的分隔线 + 逐行 icon + name + description + 字符数格式；
#       - 例如：  🧩 baoyu-diagram  创建专业的暗色主题 SVG 图表...  [meta 86 字符]
#       - 让用户一眼看到 harness 加载了哪些 skill 的元数据。
#
# def print_load_progress(stage: str, name: str, char_count: int = 0):
#     """
#     渐进式加载每完成一层调用一次，打印进度。
#     stage 取值："body" / "references" / "execute"。
#     """
#     关键细节：
#       - 这是 CLI 版的 broadcast——对标 scheduler.broadcast("heartbeat_start"/"heartbeat_message")，
#         只是把 SSE 推流换成 print；
#       - 输出形如：  ✓ [body] 已加载 baoyu-diagram 正文  [1240 字符]
#       - 用 DIM 灰色弱化，避免抢主输出焦点。
#
# def print_match_result(candidates: list[str], chosen: str | None):
#     """打印两阶段匹配结果：候选 → 选中。"""
#     关键细节：
#       - 候选用 YELLOW 列出，选中用 GREEN 高亮；
#       - 未选中（chosen=None）时打印"未匹配到 skill，走普通对话"。


# ANSI 颜色常量（照搬原 agent.py）
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
DIM = "\033[2m"


def print_registry(metas: list):
    """启动时打印已注册 skill 清单，让"渐进式加载第 1 层"可见。"""
    print(f"\n{CYAN}{'─'*60}{RESET}")
    print(f"{CYAN}  已注册 skill 清单（渐进式第 1 层：front-matter）{RESET}")
    print(f"{CYAN}{'─'*60}{RESET}")
    if not metas:
        print(f"  {DIM}（空）{RESET}")
    for m in metas:
        desc = m.description[:50] + "..." if len(m.description) > 50 else m.description
        desc = desc.replace("\n", " ")
        print(f"  🧩 {BOLD}{m.name}{RESET}  v{m.version}  {DIM}{desc}{RESET}")
    print(f"{CYAN}{'─'*60}{RESET}\n")


def print_load_progress(stage: str, name: str, char_count: int = 0):
    """渐进式加载每完成一层调用一次，打印进度（CLI 版 broadcast）。"""
    icons = {"body": "📖", "references": "📚", "execute": "⚡"}
    icon = icons.get(stage, "·")
    print(f"  {DIM}{icon} [{stage}] 已加载 {name}  [{char_count} 字符]{RESET}")


def print_match_result(candidates: list, chosen):
    """打印两阶段匹配结果：候选 → 选中。"""
    if candidates:
        print(f"  {YELLOW}[Stage 1] 候选：{', '.join(candidates)}{RESET}")
    if chosen:
        print(f"  {GREEN}[Stage 2] 选中：{chosen}{RESET}")
    else:
        print(f"  {DIM}[匹配] 未命中 skill，走普通对话{RESET}")


# ════════════════════════════════════════════════════════════════════════════
# 模块 4：LLM 输出安全解析（对标 heartbeat_parser._parse_json_safe）
# ════════════════════════════════════════════════════════════════════════════
#
# def _parse_json_safe(text: str) -> dict | None:
#     """
#     从 LLM 输出里稳健提取 JSON：剥代码围栏 + 正则取首个 {...} + json.loads 兜底。
#     """
#     关键细节：
#       - 直接照搬 heartbeat_parser._parse_json_safe() 三步法：
#         1) re.sub(r"^```[a-zA-Z]*\n?", "", text) 去开头围栏
#         2) re.sub(r"\n?```$", "", text) 去结尾围栏
#         3) re.search(r"\{[\s\S]*\}", text) 取首个 JSON 对象 → json.loads
#       - 任一步失败返回 None，调用方按"未匹配"处理；
#       - LLM 输出常带解释文字，必须用正则抠 JSON，不能直接 json.loads 整段。


def _parse_json_safe(text: str):
    """从 LLM 输出里稳健提取 JSON：剥代码围栏 + 正则取首个 {...} + json.loads 兜底。"""
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


# ════════════════════════════════════════════════════════════════════════════
# 模块 5：Stage 2 LLM 二次确认（对标 heartbeat_parser.analyze_and_write）
# ════════════════════════════════════════════════════════════════════════════
#
# _MATCH_PROMPT 模板字符串：
#   用户说："{message}"
#   候选 skills：
#   - baoyu-diagram: 创建专业的暗色主题 SVG 图表...
#   - flash-card: 为一个英语单词生成静态 HTML 学习闪卡...
#   判断用户想触发哪个 skill，并从消息中抽取该 skill 需要的参数。
#   返回 JSON：{"skill": "skill名或null", "args": {"word": "crazy"}, "reason": "一句话"}
#   只返回 JSON。
#
# def stage2_llm_match(user_input: str, candidates: list[str], registry: SkillRegistry) -> tuple[str | None, dict]:
#     """
#     Stage 1 命中后才调：让 LLM 在候选 skills 里选一个 + 抽参数。
#     返回 (skill_name_or_None, args_dict)。
#     """
#     关键细节：
#       - 仅当 candidates 非空才调用，避免每条消息都过 LLM（对标 heartbeat_parser
#         "正则命中才 analyze_and_write"的两段式设计）；
#       - 用 get_chat_client() 拿客户端，temperature=0 保证确定性；
#       - 候选 skills 的描述从 registry.get_meta(name).description 取，注入 prompt；
#       - 调 _parse_json_safe 解析输出；解析失败或 skill=null → 返回 (None, {})；
#       - LLM 返回的 skill 名不在 candidates 里 → 视为幻觉，返回 (None, {})；
#       - args 用于后续 execute_skill：如 flash-card 需要 {"word": "crazy"}，
#         baoyu-diagram 需要 {"topic": "...", "type": "architecture"}；
#       - 用 run_in_executor？本 harness 是 CLI 同步循环，直接同步调即可，
#        （若后续接 Web 再改 async，对标 scheduler 用 asyncio.run_in_executor 的做法）。


_MATCH_PROMPT = """\
用户说："{message}"

候选 skills：
{candidates}

判断用户想触发哪个 skill，并从消息中抽取该 skill 需要的参数。
返回 JSON：
{{"skill": "skill名或null", "args": {{}}, "reason": "一句话"}}

只返回 JSON，不要其他文字。"""


def stage2_llm_match(user_input: str, candidates: list, registry: SkillRegistry):
    """
    Stage 1 命中后才调：让 LLM 在候选 skills 里选一个 + 抽参数。
    返回 (skill_name_or_None, args_dict)。
    """
    if not candidates:
        return (None, {})

    # 无 LLM 时降级：直接取第一个候选（保证 harness 可用）
    if not _LLM_OK:
        logger.warning("LLM 未配置，Stage 2 降级为取首个候选")
        return (candidates[0], {})

    # 组装候选 skills 描述
    lines = []
    for name in candidates:
        meta = registry.get_meta(name)
        if meta:
            desc = meta.description[:80].replace("\n", " ")
            lines.append(f"- {name}: {desc}")
    candidate_block = "\n".join(lines)

    prompt = _MATCH_PROMPT.format(message=user_input, candidates=candidate_block)

    try:
        client, model = get_chat_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        data = _parse_json_safe(raw)
        if not data:
            return (None, {})
        skill = data.get("skill")
        if not skill or skill == "null":
            return (None, {})
        # 幻觉防护：LLM 返回的 skill 必须在候选列表里
        if skill not in candidates:
            logger.warning(f"LLM 返回了不在候选里的 skill：{skill}")
            return (None, {})
        args = data.get("args") or {}
        return (skill, args)
    except Exception as e:
        logger.error(f"Stage 2 LLM 匹配失败：{e}")
        return (None, {})


# ════════════════════════════════════════════════════════════════════════════
# 模块 6：Skill 执行分发（对标 scheduler._execute_task 的 action 分发）
# ════════════════════════════════════════════════════════════════════════════
#
# def execute_skill(name: str, args: dict, registry: SkillRegistry) -> str:
#     """
#     执行指定 skill：先 load_references（渐进第 3 层），再按 skill 类型分发执行。
#     返回执行结果文本（供主循环回显给用户）。
#     """
#     关键细节（执行前置——完成渐进式加载最后一层）：
#       - registry.load_body(name) 确保正文已加载（若已缓存则 no-op）；
#       - registry.load_references(name) 加载参考文件（最重一层，仅此处触发）；
#       - print_load_progress("references", name, char_count) 让加载可见；
#       - 至此三层渐进加载全部完成：front-matter（启动）→ body（匹配命中）→ references（执行前）。
#
#     关键细节（分发执行——按 script 类型走不同子流程）：
#       - loaded = registry.get_loaded(name)；
#       - 若 loaded.script_path 为 None → 纯文本 skill，把 body 当指令交 LLM 执行
#         （参考 baoyu-diagram SKILL.md 正文本身就是给 LLM 的指令）；
#       - 若 script_path 后缀 == ".py" → subprocess.run([sys.executable, script, ...args])；
#       - 若 script_path 后缀 == ".ts" → 解析 ${BUN_X}：
#           * 优先 bun，次选 npx -y bun（参考 baoyu-diagram SKILL.md 的运行时探测约定）；
#           * subprocess.run([bun_or_npx, script, ...args])；
#       - 参数传递：
#         * flash-card：先按 args 生成 data/<word>.json（LLM 写或模板填），
#           再调 make_flashcard.py <json路径>，输出 <word>.html 到 cwd；
#         * baoyu-diagram：LLM 直接按 body 指令生成 SVG，再调 main.ts <svg> 转 @2x PNG；
#       - 捕获 stdout/stderr，text=True，encoding="utf-8"；
#       - subprocess 超时设 120s，超时返回错误文本（不卡死主循环）；
#       - 对标 scheduler._execute_task() 按 action 分发到 _action_xxx() 的模式，
#         这里按 script 类型 + skill name 分发到具体执行逻辑。
#
# def _exec_python_script(script: Path, argv: list[str]) -> str:
#     """subprocess 跑 .py 脚本，返回 stdout。"""
#     关键细节：[sys.executable, str(script)] + argv；cwd 设为用户当前工作目录
#     （参考 flash-card SKILL.md："HTML 输出到当前工作目录"）。
#
# def _exec_bun_script(script: Path, argv: list[str]) -> str:
#     """探测 bun/npx 跑 .ts 脚本，返回 stdout。"""
#     关键细节：shutil.which("bun") 优先，否则 ["npx", "-y", "bun"]；
#     失败时返回提示"请安装 bun"。"""


def execute_skill(name: str, args: dict, registry: SkillRegistry, user_input: str = "") -> str:
    """
    执行指定 skill：先完成渐进式加载第 2、3 层，再按 script 类型分发执行。
    返回执行结果文本。
    """
    # ── 渐进式加载第 2 层：body ──
    registry.load_body(name)
    loaded = registry.get_loaded(name)
    if not loaded:
        return f"{YELLOW}加载 skill {name} 失败{RESET}"
    print_load_progress("body", name, len(loaded.body))

    # ── 渐进式加载第 3 层：references（最重，仅此处触发）──
    refs = registry.load_references(name)
    if refs:
        print_load_progress("references", name, sum(len(v) for v in refs.values()))

    print_load_progress("execute", name)

    # ── 分发执行 ──
    sp = loaded.script_path
    if sp is None:
        # 纯文本 skill：body 当 LLM 指令
        return _exec_text_skill(loaded, args)

    if sp.suffix == ".py":
        # flash-card：LLM 生成 JSON 数据 → 调 make_flashcard.py
        if name == "flash-card":
            return _run_flashcard(loaded, args)
        # 通用 .py：把 args 写临时 JSON 传给脚本
        return _exec_python_skill_generic(loaded, args)

    if sp.suffix == ".ts":
        # baoyu-diagram：LLM 按 body 指令生成 SVG → 调 main.ts 转 PNG
        if name == "baoyu-diagram":
            return _run_baoyu_diagram(loaded, args, user_input)
        # 通用 .ts：args 值作为 CLI 参数传入
        argv = [str(v) for v in args.values()]
        return _exec_bun_script(sp, argv)

    return f"{YELLOW}不支持的脚本类型：{sp.suffix}{RESET}"


def _exec_python_script(script: Path, argv: list) -> str:
    """subprocess 跑 .py 脚本，返回 stdout。"""
    try:
        result = subprocess.run(
            [sys.executable, str(script)] + argv,
            capture_output=True, text=True, encoding="utf-8",
            timeout=120, cwd=str(Path.cwd()),
        )
        if result.returncode != 0:
            return f"{YELLOW}脚本失败（退出码 {result.returncode}）：\n{result.stderr}{RESET}"
        return result.stdout.strip() or f"{GREEN}脚本执行完成{RESET}"
    except subprocess.TimeoutExpired:
        return f"{YELLOW}脚本执行超时（120s）{RESET}"
    except Exception as e:
        return f"{YELLOW}脚本执行异常：{e}{RESET}"


def _exec_bun_script(script: Path, argv: list) -> str:
    """探测 bun/npx 跑 .ts 脚本，返回 stdout。"""
    # 解析 ${BUN_X}：优先 bun，次选 npx -y bun
    bun = shutil.which("bun")
    if bun:
        cmd = [bun, str(script)] + argv
    elif shutil.which("npx"):
        cmd = ["npx", "-y", "bun", str(script)] + argv
    else:
        return f"{YELLOW}未找到 bun 或 npx，请安装 bun 后重试{RESET}"

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            timeout=120, cwd=str(Path.cwd()),
        )
        if result.returncode != 0:
            return f"{YELLOW}脚本失败（退出码 {result.returncode}）：\n{result.stderr}{RESET}"
        return result.stdout.strip() or f"{GREEN}脚本执行完成{RESET}"
    except subprocess.TimeoutExpired:
        return f"{YELLOW}脚本执行超时（120s）{RESET}"
    except Exception as e:
        return f"{YELLOW}脚本执行异常：{e}{RESET}"


# ── 以下为 execute_skill 的内部辅助（未在注释清单里单列，但执行必需）──────────


def _exec_text_skill(loaded: LoadedSkill, args: dict) -> str:
    """纯文本 skill（无脚本）：把 body 当 LLM 指令执行。"""
    if not _LLM_OK:
        return f"{YELLOW}LLM 未配置，无法执行纯文本 skill。正文摘要：\n{loaded.body[:300]}...{RESET}"
    system = "你是一个 skill 执行器，严格按照以下 skill 指令完成任务。"
    user_msg = (
        f"Skill 指令：\n{loaded.body}\n\n"
        f"用户参数：{json.dumps(args, ensure_ascii=False)}\n\n"
        f"请执行上述指令。"
    )
    try:
        client, model = get_chat_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"{YELLOW}LLM 执行失败：{e}{RESET}"


def _exec_python_skill_generic(loaded: LoadedSkill, args: dict) -> str:
    """通用 .py skill 执行：把 args 写临时 JSON，作为第一个参数传给脚本。"""
    import tempfile
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(args, f, ensure_ascii=False)
        tmp_path = f.name
    return _exec_python_script(loaded.script_path, [tmp_path])


def _run_flashcard(loaded: LoadedSkill, args: dict) -> str:
    """
    flash-card 执行流程（参考其 SKILL.md）：
      1. LLM 生成单词学习数据（word/phonetic/pos/definition/examples/synonyms）
      2. 写入 skill 的 data/<word>.json
      3. 调 make_flashcard.py <json路径>，输出 <word>.html 到当前工作目录
    """
    word = (args.get("word") or "unknown").lower().strip()

    # Step 1: 生成数据
    if _LLM_OK:
        data = _llm_generate_flashcard_data(word)
    else:
        data = {
            "word": word, "phonetic": "", "pos": "", "definition": "",
            "examples": [{"en": "", "zh": ""} for _ in range(3)],
            "synonyms": [],
        }

    # Step 2: 写入 data/ 目录（与 SKILL.md 约定一致）
    data_path = loaded.meta.base_dir / "data" / f"{word}.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # Step 3: 调脚本生成 HTML
    output = _exec_python_script(loaded.script_path, [str(data_path)])
    return f"{GREEN}Flash Card 已生成：{word}.html{RESET}\n{output}"


def _llm_generate_flashcard_data(word: str) -> dict:
    """LLM 生成闪卡数据 JSON（word/phonetic/pos/definition/examples[3]/synonyms[4-6]）。"""
    try:
        client, model = get_chat_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": (
                    f'为英语单词 "{word}" 生成学习数据，返回 JSON：'
                    f'word/phonetic/pos/definition/examples(3条中英对照)/synonyms(4-6个)。'
                    f'只返回 JSON。'
                ),
            }],
            temperature=0.3,
        )
        raw = resp.choices[0].message.content.strip()
        data = _parse_json_safe(raw)
        if data and "word" in data:
            return data
    except Exception as e:
        logger.error(f"生成闪卡数据失败：{e}")
    return {
        "word": word, "phonetic": "", "pos": "", "definition": "",
        "examples": [{"en": "", "zh": ""} for _ in range(3)],
        "synonyms": [],
    }


def _run_baoyu_diagram(loaded: LoadedSkill, args: dict, user_input: str = "") -> str:
    """
    baoyu-diagram 执行流程（参考其 SKILL.md）：
      1. LLM 按 body 指令 + references 布局指引 生成 SVG
      2. 保存 SVG 到 cwd/diagram/<slug>.svg
      3. 调 main.ts <svg路径> 转 @2x PNG
    """
    if not _LLM_OK:
        return f"{YELLOW}LLM 未配置，无法生成 SVG{RESET}"

    # 优先用 args 里的 topic，其次用用户原始输入
    topic = args.get("topic") or args.get("subject") or user_input or "未指定主题"
    diagram_type = args.get("type") or "architecture"

    # 从 references 取对应类型的布局指引
    ref_content = loaded.references.get(diagram_type, "")

    try:
        client, model = get_chat_client()
        system = "你是一个 SVG 图表生成器，严格按照 skill 指令生成 SVG。只输出 SVG 代码。"
        user_msg = (
            f"Skill 指令：\n{loaded.body}\n\n"
            f"参考布局指引（{diagram_type}）：\n{ref_content[:2000]}\n\n"
            f"用户请求：{user_input}\n\n"
            f"请直接输出 SVG 代码（从 <svg> 到 </svg>），不要其他说明。"
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
        )
        raw = resp.choices[0].message.content.strip()

        # 从 LLM 输出里抠出纯 SVG（去掉 markdown 围栏、解释文字等）
        svg_match = re.search(r"<svg[\s\S]*</svg>", raw)
        if not svg_match:
            return f"{YELLOW}LLM 未返回有效 SVG。原始输出前 300 字符：\n{raw[:300]}{RESET}"
        svg = svg_match.group()

        # 保存 SVG
        out_dir = Path.cwd() / "diagram"
        out_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^\w-]", "_", topic)[:30] or "diagram"
        svg_path = out_dir / f"{slug}.svg"
        svg_path.write_text(svg, encoding="utf-8")

        # 调 main.ts 转 @2x PNG
        png_output = _exec_bun_script(loaded.script_path, [str(svg_path)])
        return f"{GREEN}SVG 已生成：{svg_path}{RESET}\n{png_output}"
    except Exception as e:
        return f"{YELLOW}生成图表失败：{e}{RESET}"


# ════════════════════════════════════════════════════════════════════════════
# 模块 7：普通对话回退（未命中 skill 时）
# ════════════════════════════════════════════════════════════════════════════
#
# def chat_fallback(user_input: str, history: list[dict]) -> str:
#     """
#     Stage 1/2 都未命中 skill 时，走普通 LLM 对话。
#     history 是本会话的 [{role, content}, ...]，注入为多轮上下文。
#     """
#     关键细节：
#       - system prompt 简短声明"你是一个 skill 路由助手，未触发 skill 时正常对话"；
#       - temperature=0.7，stream=True，逐 chunk 打印（照搬原 agent.py 的流式输出）；
#       - 这保证 harness 既是 skill 路由器，又是普通 chatbot，体验不割裂。


def chat_fallback(user_input: str, history: list) -> str:
    """未命中 skill 时走普通 LLM 对话，流式输出。"""
    if not _LLM_OK:
        msg = f"{YELLOW}LLM 未配置，无法对话。可用的命令：/skills /reload /layers <name> /exit{RESET}"
        print(msg)
        return msg

    system = "你是一个 skill 路由助手。未触发任何 skill 时，正常与用户对话。"
    messages = [{"role": "system", "content": system}] + history + [
        {"role": "user", "content": user_input}
    ]

    try:
        client, model = get_chat_client()
        print(f"{GREEN}助手：{RESET}", end="", flush=True)
        stream = client.chat.completions.create(
            model=model, messages=messages, temperature=0.7, stream=True
        )
        response = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            response += delta
        print()
        return response
    except Exception as e:
        msg = f"{YELLOW}对话失败：{e}{RESET}"
        print(msg)
        return msg


# ════════════════════════════════════════════════════════════════════════════
# 模块 8：主循环 main()（对标 agent_memory_system/src/agent.py 的 main()）
# ════════════════════════════════════════════════════════════════════════════
#
# def main():
#     """
#     CLI 主入口：初始化 → 加载注册表 → 循环接收输入 → 匹配 → 渐进加载 → 执行。
#     """
#     关键细节（按执行顺序）：
#
#     【初始化】
#       - get_chat_client() 提前验证 API Key，缺失则提示并 sys.exit(1)
#         （照搬原 agent.py 的 try/except EnvironmentError）；
#       - registry = build_registry()  → 启动即 scan_skills()，完成渐进式第 1 层加载；
#       - print_registry(registry.list_skills()) 让用户看到已加载 skill；
#       - history: list[dict] = []  本会话多轮历史。
#
#     【主循环 while True】
#       - user_input = input("你：").strip()；Ctrl+C/EOF → 当作 /exit。
#
#       ── 命令分发（对标原 agent.py 的 /flush /memory /layers /new）──
#       - /exit  → break
#       - /skills → print_registry(...)（重显注册表）
#       - /reload → registry.scan_skills() 强制重扫 + 重建 patterns
#       - /layers <name> → 打印某 skill 的渐进加载状态（meta / body / refs 各层字符数）
#
#       ── 热重载检查 ──
#       - if registry.reload_if_changed(): patterns = build_stage1_patterns(...)
#         （每轮都做，O(1) mtime 比对，变了才重扫——对标 scheduler._check_reload）
#
#       ── Stage 1 正则初筛 ──
#       - candidates = stage1_filter(user_input, patterns)
#       - 空候选 → chat_fallback(user_input, history) → 追加 history → continue
#
#       ── Stage 2 LLM 二次确认 ──
#       - chosen, args = stage2_llm_match(user_input, candidates, registry)
#       - print_match_result(candidates, chosen)
#       - chosen is None → chat_fallback → continue
#
#       ── 渐进式加载第 2 层：body ──
#       - registry.load_body(chosen)
#       - print_load_progress("body", chosen, char_count)
#
#       ── 执行（内部触发第 3 层 references 加载）──
#       - result = execute_skill(chosen, args, registry)
#       - print_load_progress("execute", chosen)
#       - print(result)
#       - history.append({"role":"user","content":user_input})
#       - history.append({"role":"assistant","content":result})
#
# 【if __name__ == "__main__": main()】


def main():
    """CLI 主入口：初始化 → 加载注册表 → 循环接收输入 → 匹配 → 渐进加载 → 执行。"""
    print(f"\n{BOLD}Skill Harness — 渐进式加载执行器{RESET}")
    if _LLM_OK:
        _p = "qwen" if os.getenv("DASHSCOPE_API_KEY") else "deepseek"
        print(f"{DIM}LLM 已就绪（provider={_p}）{RESET}")
    else:
        print(f"{YELLOW}LLM 未配置（需 pip install openai 并设置 DEEPSEEK_API_KEY 或 DASHSCOPE_API_KEY）{RESET}")
        print(f"{DIM}  仍可使用 /skills /reload /layers <name> 命令，Stage 1 正则匹配也可用{RESET}")
    print(f"命令：/skills  /reload  /layers <name>  /exit\n")

    # 【初始化】
    registry = build_registry()
    patterns = build_stage1_patterns(registry.list_skills())
    print_registry(registry.list_skills())

    history: list = []

    # 【主循环】
    while True:
        try:
            user_input = input(f"{BOLD}你：{RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            user_input = "/exit"

        if not user_input:
            continue

        # ── 命令分发 ──
        if user_input == "/exit":
            print("再见！")
            break

        if user_input == "/skills":
            print_registry(registry.list_skills())
            continue

        if user_input == "/reload":
            registry.scan_skills()
            patterns = build_stage1_patterns(registry.list_skills())
            print(f"{GREEN}已强制重扫，当前 {len(registry.list_skills())} 个 skill{RESET}")
            continue

        if user_input.startswith("/layers"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print(f"{YELLOW}用法：/layers <skill_name>{RESET}")
                continue
            name = parts[1].strip()
            meta = registry.get_meta(name)
            if not meta:
                print(f"{YELLOW}未找到 skill：{name}{RESET}")
                continue
            loaded = registry.get_loaded(name)
            print(f"\n{CYAN}── {name} 渐进加载状态 ──{RESET}")
            print(f"  第 1 层 front-matter：{len(meta.description)} 字符  v{meta.version}")
            if loaded:
                print(f"  第 2 层 body：{len(loaded.body)} 字符  段落：{list(loaded.body_sections.keys())}")
                refs_chars = sum(len(v) for v in loaded.references.values())
                print(f"  第 3 层 references：{refs_chars} 字符  {list(loaded.references.keys()) or '（无）'}")
                print(f"  script：{loaded.script_path or '（无）'}")
                print(f"  总字符数：{loaded.char_count}")
            else:
                print(f"  {DIM}第 2、3 层未加载（未命中过）{RESET}")
            print()
            continue

        # ── 热重载检查（每轮做，O(1) mtime 比对）──
        if registry.reload_if_changed():
            patterns = build_stage1_patterns(registry.list_skills())
            print(f"{MAGENTA}[热重载] skills 目录已更新，patterns 已重建{RESET}")

        # ── Stage 1 正则初筛 ──
        candidates = stage1_filter(user_input, patterns)
        if not candidates:
            # 空候选 → 普通对话
            resp = chat_fallback(user_input, history)
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": resp})
            continue

        # ── Stage 2 LLM 二次确认 ──
        chosen, args = stage2_llm_match(user_input, candidates, registry)
        print_match_result(candidates, chosen)

        if not chosen:
            resp = chat_fallback(user_input, history)
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": resp})
            continue

        # ── 执行（内部完成渐进第 2、3 层加载 + 分发执行）──
        result = execute_skill(chosen, args, registry, user_input)
        print(result)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": result})


if __name__ == "__main__":
    main()


# ════════════════════════════════════════════════════════════════════════════
# 设计说明（不写代码，仅备忘）
# ════════════════════════════════════════════════════════════════════════════
# 一次完整调用的时序（与 skill_loader 三层加载对应）：
#
#   启动：build_registry() → scan_skills()
#         ↓ 渐进第 1 层：front-matter 全量加载（廉价）
#         print_registry()
#
#   用户输入 → reload_if_changed() 热重载检查
#         ↓
#   Stage 1: stage1_filter()  正则初筛（零 LLM 成本）
#         ↓ 命中候选
#   Stage 2: stage2_llm_match()  LLM 在候选里选 + 抽参数
#         ↓ 选中 chosen
#   registry.load_body(chosen)
#         ↓ 渐进第 2 层：body 按需加载（中等）
#         print_load_progress("body")
#   execute_skill(chosen, args)
#         ├─ registry.load_references(chosen)  渐进第 3 层：references 执行前加载（最重）
#         ├─ print_load_progress("references")
#         ├─ print_load_progress("execute")
#         └─ subprocess / LLM 执行
#         ↓
#   输出结果
#
# 与原项目 agent.py 的对应：
#   原项目：四层记忆加载 → 语义检索 → 组装 context → LLM
#   本 harness：三层 skill 加载 → 两阶段匹配 → 执行分发
#   核心思想一致：按需渐进加载 + 分层反馈，避免一次性全量载入。
