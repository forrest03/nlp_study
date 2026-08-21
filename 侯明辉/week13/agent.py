"""
agent.py — 交互式多轮对话 Agent（CLI 入口，week13 Skills Harness 版）

使用：
  # 启动新会话
  python agent.py

  # 恢复历史会话
  python agent.py --session s_20260722_203000

  # 切换模型 provider
  python agent.py --provider dashscope

  # 切换 skills 目录
  python agent.py --skills-dir /path/to/skills

环境变量：
  DEEPSEEK_API_KEY / DASHSCOPE_API_KEY / SILICONFLOW_API_KEY 至少设一个
"""

import argparse
import os
import sys
from pathlib import Path

# Windows GBK 控制台兜底
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# 让同级目录（core/tools/storage）能 import
sys.path.insert(0, str(Path(__file__).parent))

from core.llm import PROVIDERS, build_client, preflight_check     # noqa: E402
from core.memory import FullHistoryMemory                       # noqa: E402
from core.runner import run_one                                 # noqa: E402
from core.session import Session                                # noqa: E402
from core.skill_registry import SkillRegistry                   # noqa: E402
from storage.session_store import SessionStore                   # noqa: E402

# ── 路径与默认 system prompt ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR   = PROJECT_ROOT / "output"
SESSIONS_DIR = OUTPUT_DIR / "sessions"
DEFAULT_SKILLS_DIR = PROJECT_ROOT / "skills"
DEFAULT_SYSTEM_PROMPT = (
    "你是一名出行与日常助手。回答用户问题时："
    "1) 天气/时间问题必须调用工具（get_weather / get_time）获取实时数据；"
    "2) 比较/排序/推荐等问题应对每个对象各调用一次工具；"
    "3) 能直接答的简短问题直接答，不要硬调工具；"
    "4) 回答要简洁，引用工具返回的具体数据。"
)

# ── ANSI 颜色 ──────────────────────────────────────────────────────────────
_COLOR_CODES = {
    "cyan":    "\033[36m",
    "yellow":  "\033[33m",
    "green":   "\033[32m",
    "magenta": "\033[35m",
    "red":     "\033[31m",
    "dim":     "\033[2m",
    "reset":   "\033[0m",
}


def _c(code: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{_COLOR_CODES.get(code, '')}{text}{_COLOR_CODES['reset']}"


# ── 命令帮助文本 ────────────────────────────────────────────────────────────
HELP_TEXT = """\
可用命令：
  <任何文本>      作为 user 消息发给 Agent
  /help          显示本帮助
  /history       打印当前 session 的消息历史
  /sessions      列出所有已存档的 session
  /clear         清空历史（保留 system prompt）
  /new           关闭当前 session（不自动存档），开新 session
  /save [path]   手动存档（默认 output/sessions/<id>.json）
  /load <id>     加载指定 id 的 session
  /skills        列出所有已注册 Skill（MEMORY.md 索引）
  /skill <name>  手动激活 Skill（强制加载完整 SKILL.md）
  /reload        重新扫描 skills/ 目录（热重载）
  /inspect       打印当前 session 已激活的 Skill
  /exit / /quit  退出 CLI，自动存档当前 session
"""


def cmd_history(session: Session) -> None:
    print(_c("cyan", f"\n─── 当前 session: {session.id} ({len(session.messages)} 条) ───"))
    for i, m in enumerate(session.messages, 1):
        role = m.get("role", "?")
        if role == "system":
            content_preview = (m.get("content") or "")[:80]
            print(_c("dim", f"  [{i}] system: {content_preview}..."))
        elif role == "user":
            print(_c("green", f"  [{i}] user: {m.get('content')}"))
        elif role == "assistant":
            if m.get("tool_calls"):
                tcs = m["tool_calls"]
                tc_names = [tc["function"]["name"] for tc in tcs]
                print(_c("yellow", f"  [{i}] assistant → 调用工具: {tc_names}"))
            else:
                content = m.get("content") or ""
                preview = content[:120] + ("..." if len(content) > 120 else "")
                print(_c("magenta", f"  [{i}] assistant: {preview}"))
        elif role == "tool":
            content = m.get("content") or ""
            preview = content[:80].replace("\n", " ") + ("..." if len(content) > 80 else "")
            print(_c("dim", f"  [{i}] tool ↩ {preview}"))
    print()


def cmd_sessions(store: SessionStore) -> None:
    ids = store.list_sessions()
    if not ids:
        print(_c("dim", "\n  （暂无存档）\n"))
        return
    print(_c("cyan", f"\n  共 {len(ids)} 个存档："))
    for sid in ids:
        print(f"    {sid}")
    print()


def cmd_clear(session: Session) -> None:
    session.clear()
    print(_c("dim", "\n  ✓ 历史已清空（保留 system prompt）\n"))


def cmd_new(model: str, provider: str) -> Session:
    s = Session(system_prompt=DEFAULT_SYSTEM_PROMPT, model=model, provider=provider)
    print(_c("dim", f"\n  ✓ 新 session 已开：{s.id}（旧 session 未自动存档）\n"))
    return s


def cmd_save(session: Session, store: SessionStore) -> None:
    store.save(session.id, session.to_dict())
    print(_c("green", f"\n  💾 已保存到 {SESSIONS_DIR / (session.id + '.json')}\n"))


def cmd_load(session_id: str, store: SessionStore):
    data = store.load(session_id)
    if data is None:
        print(_c("red", f"\n  ✗ 未找到 session：{session_id}\n"))
        return None
    s = Session.from_dict(data, default_system_prompt=DEFAULT_SYSTEM_PROMPT)
    print(_c("green", f"\n  🔄 已恢复 session {session_id}（{len(s.messages)} 条历史，{len(s.active_skills)} 个激活 Skill）\n"))
    return s


def cmd_skills(registry: SkillRegistry) -> None:
    """列出所有已注册的 Skill（MEMORY.md 索引）"""
    metas = registry.list_all()
    if not metas:
        print(_c("dim", "\n  （暂无已注册 Skill）\n"))
        return
    print(_c("cyan", f"\n─── 已注册 Skills（MEMORY.md 索引，常驻 < 200 tokens）───"))
    for m in metas:
        tools_hint = f"  [工具: {', '.join(m.tools)}]" if m.tools else ""
        print(f"  • {_c('green', m.name)} — {m.description}{_c('dim', tools_hint)}")
    print(_c("dim", "\n  提示：用 /skill <name> 手动激活，或直接提问触发自动加载"))
    print()


def cmd_skill_activate(name: str, session: Session, registry: SkillRegistry) -> None:
    """手动激活 Skill（强制加载完整 SKILL.md）"""
    try:
        skill = registry.load_skill(name)
    except KeyError:
        print(_c("red", f"\n  ✗ Skill 不存在: {name}\n"))
        print(_c("dim", f"  可用: {[m.name for m in registry.list_all()]}\n"))
        return
    session.mark_skill_active(name)
    print(_c("green", f"\n  ✓ 已激活 Skill: {name}\n"))
    preview = skill.body[:300].replace("\n", "\n    ")
    print(_c("dim", f"    {preview}{'...' if len(skill.body) > 300 else ''}\n"))


def cmd_reload(session: Session, registry: SkillRegistry) -> None:
    """热重载：重新扫描 skills/ 目录"""
    new_reg = SkillRegistry(SKILLS_DIR)
    registry._index = new_reg._index
    registry._cache.clear()
    print(_c("green", f"\n  ✓ 已重新扫描 skills/ 目录（{len(registry._index)} 个 Skill）\n"))


def cmd_inspect(session: Session, registry: SkillRegistry) -> None:
    """打印当前 session 已激活的 Skill"""
    print(_c("cyan", f"\n─── 当前 session 已激活 Skill ({len(session.active_skills)}) ───"))
    if not session.active_skills:
        print(_c("dim", "  （暂无）"))
    else:
        for name in session.active_skills:
            meta = registry.find_by_name(name)
            desc = meta.description if meta else "?"
            print(f"  • {_c('green', name)} — {desc}")
    print()


def handle_user_input(text: str, session: Session, client, model: str, registry: SkillRegistry) -> None:
    """用户消息处理：追加 user → 调 Runner → 打印 answer"""
    session.append_user(text)
    result = run_one(client, model, session, registry=registry, verbose=True)
    answer = result["answer"]
    if result["truncated"]:
        print(_c("red", f"\n  ⚠ 达到熔断（{result['rounds']} 轮）\n"))
    print(_c("magenta", "\n  Agent ➜ ") + answer + "\n")


# ── 主循环 ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="多轮对话 Agent + Skills Harness（CLI）")
    parser.add_argument("--session", help="启动时恢复指定 session id")
    parser.add_argument("--provider", default="siliconflow", choices=list(PROVIDERS.keys()))
    parser.add_argument("--skills-dir", type=Path, default=DEFAULT_SKILLS_DIR,
                        help="Skills 目录路径（默认 ./skills）")
    args = parser.parse_args()

    # 构造 LLM 客户端
    client, model = build_client(args.provider)
    preflight_check(client, model, args.provider)

    # 构造 store（自动建 output/sessions/）
    store = SessionStore(SESSIONS_DIR)

    # 构造 SkillRegistry
    skills_dir = args.skills_dir
    if not skills_dir.exists():
        print(_c("yellow", f"  ⚠ Skills 目录不存在: {skills_dir}（用 /skills 看会显示空）"))
    registry = SkillRegistry(skills_dir)

    # 构造或恢复 session
    if args.session:
        loaded = cmd_load(args.session, store)
        if loaded is None:
            sys.exit(1)
        session = loaded
    else:
        session = Session(system_prompt=DEFAULT_SYSTEM_PROMPT, model=model, provider=args.provider)

    # 启动 banner
    print(_c("cyan", "=" * 64))
    print(_c("cyan", f"  Skills Harness Agent  |  provider={args.provider}  model={model}"))
    print(_c("cyan", f"  session: {session.id}  |  Skills: {len(registry.list_all())} 个（输入 /skills 查看）"))
    print(_c("cyan", f"  输入 /help 查看命令（{len(HELP_TEXT.splitlines()) - 2} 个）"))
    print(_c("cyan", "=" * 64))
    print()

    # CLI 主循环
    while True:
        try:
            line = input(_c("green", ">>> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            line = "/exit"
        if not line:
            continue
        if line in ("/exit", "/quit"):
            store.save(session.id, session.to_dict())
            print(_c("green", f"\n  💾 已自动存档 {session.id}（含 {len(session.active_skills)} 个 Skill），再见 👋\n"))
            break
        elif line == "/help":
            print(HELP_TEXT)
        elif line == "/history":
            cmd_history(session)
        elif line == "/sessions":
            cmd_sessions(store)
        elif line == "/clear":
            cmd_clear(session)
        elif line == "/new":
            session = cmd_new(model, args.provider)
        elif line.startswith("/save"):
            cmd_save(session, store)
        elif line.startswith("/load "):
            sid = line.split(maxsplit=1)[1].strip()
            loaded = cmd_load(sid, store)
            if loaded is not None:
                session = loaded
        elif line == "/skills":
            cmd_skills(registry)
        elif line.startswith("/skill "):
            name = line.split(maxsplit=1)[1].strip()
            cmd_skill_activate(name, session, registry)
        elif line == "/reload":
            cmd_reload(session, registry)
        elif line == "/inspect":
            cmd_inspect(session, registry)
        else:
            # 普通 user 消息
            try:
                handle_user_input(line, session, client, model, registry)
            except Exception as e:
                print(_c("red", f"\n  ✗ 错误：{e}\n"))


if __name__ == "__main__":
    main()
