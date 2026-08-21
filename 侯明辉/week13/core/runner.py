"""
runner.py — 单轮"多轮工具调用"执行器

本质是 week11 loop_function_call.run_loop 的精简版，差异：
  1. 不再每次新建 messages 列表 — 直接接收 Session 对象
  2. 所有"追加消息"动作都通过 Session.append_* — 状态集中
  3. 暴露为 run_one(session) -> str，便于 CLI 嵌入

week13 升级：
  - 可选传入 SkillRegistry，渐进式加载 Skill 到 system prompt
  - 触发匹配用 SkillLoader（关键词 + LLM 二次确认）
  - 已激活 Skill 记录在 Session.active_skills，跳过二次确认

核心循环（与 week11 完全一致）：
  while rounds < MAX_ITERATIONS:
    resp = client.chat(messages, tools=schema, tool_choice='auto')
    msg  = resp.choices[0].message
    if not msg.tool_calls (且无 text-fallback):
        return msg.content
    for tc in msg.tool_calls:
        result = TOOL_DISPATCH[tc.name](**args)
        session.append_tool_result(tc.id, result)
"""

import json
import re
import sys
import time
import uuid
from types import SimpleNamespace
from typing import Any

from openai import OpenAI

from core.session import Session
from core.skill_registry import SkillRegistry
from core.skill_loader import SkillLoader
from core.skill_executor import build_skill_context
from tools.registry import TOOL_DISPATCH, TOOLS_SCHEMA

MAX_ITERATIONS = 8  # 循环熔断：单轮最多工具调用次数

# Windows GBK 兜底（与 week11 一致）
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass


# ── text-fallback：兜底"小模型把工具调用塞进 content"的情况 ────────────────

def parse_text_tool_calls(content: str) -> list[SimpleNamespace]:
    """
    从 content 文本里抽 JSON 工具调用，转成跟原生 tool_calls 形状一致的对象。
    返回 [] 表示没抽到。
    """
    if not content:
        return []
    text = re.sub(r"```(?:json)?", "", content).strip()
    decoder = json.JSONDecoder()
    found: list[SimpleNamespace] = []
    pos = 0
    seen_keys: set = set()
    while pos < len(text):
        while pos < len(text) and text[pos] in " \r\n\t,":
            pos += 1
        if pos >= len(text) or text[pos] != "{":
            pos += 1
            continue
        try:
            obj, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            pos += 1
            continue
        pos = end
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("function")
        if not isinstance(name, str) or name not in TOOL_DISPATCH:
            continue
        key = (name, json.dumps(obj.get("arguments"), sort_keys=True, ensure_ascii=False) if obj.get("arguments") else "")
        if key in seen_keys:
            continue
        seen_keys.add(key)
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        found.append(SimpleNamespace(
            id=f"text_{uuid.uuid4().hex[:10]}",
            function=SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False)),
        ))
    return found


# ── 主入口 ────────────────────────────────────────────────────────────────

def execute_tool(name: str, args: dict) -> str:
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return f"未知工具：{name}"
    try:
        result = fn(**args)
    except TypeError as e:
        return f"参数错误：{e}"
    except Exception as e:
        return f"工具执行失败：{e}"
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


def _activate_skills_for_input(
    session: Session,
    registry: SkillRegistry,
    client: OpenAI,
    model: str,
    verbose: bool,
) -> list[str]:
    """根据最后一次 user 消息触发 Skill 加载，返回新增激活的 Skill 名字"""
    last_user = next(
        (m for m in reversed(session.messages) if m.get("role") == "user"),
        None,
    )
    if not last_user:
        return []
    user_text = last_user.get("content", "") or ""
    if not user_text:
        return []

    # 临时 view：只对未激活的 Skill 做二次确认
    all_metas = registry.list_all()
    already = set(session.active_skill_names())
    unactivated = [m for m in all_metas if m.name not in already]
    if not unactivated:
        return []

    # 构造临时 registry view（仅未激活的）做关键词匹配
    from core.skill_registry import SkillRegistry as _Reg
    temp = _Reg.__new__(_Reg)
    temp.skills_dir = registry.skills_dir
    temp._memory_path = registry._memory_path
    temp._index = unactivated
    temp._cache = {}

    loader = SkillLoader(temp)
    selected = loader.select(user_text, client, model)
    for name in selected:
        session.mark_skill_active(name)
    if verbose and selected:
        print(f"  [Skill Loader] 激活: {selected}（已激活累计: {session.active_skills}）")
    return selected


def run_one(
    client: OpenAI,
    model: str,
    session: Session,
    registry: SkillRegistry | None = None,
    verbose: bool = True,
) -> dict:
    """
    在已有 session 上跑一次"多轮工具调用"循环。
    返回 {"answer": str, "rounds": int, "elapsed": float, "truncated": bool}

    会修改 session.messages：追加 assistant + tool 消息。
    会修改 session.active_skills：新增触发的 Skill。

    week13 新增：
      - 如果传入 registry，先触发 Skill 加载（关键词 + LLM 二次确认）
      - 把 Skill 索引 + 已激活 Skill 完整内容拼到 system prompt
    """
    # 1. 触发 Skill 加载
    if registry is not None:
        _activate_skills_for_input(session, registry, client, model, verbose)

    # 2. 构造 work_messages（注入 Skill 上下文）
    work_messages = list(session.messages)
    if registry is not None and session.active_skills:
        # 取出原始 system prompt
        base_system = ""
        if work_messages and work_messages[0].get("role") == "system":
            base_system = work_messages[0].get("content", "") or ""
        # 加载 Skill 完整内容
        active_skills = registry.load_multiple(session.active_skills)
        skill_ctx = build_skill_context(registry, active_skills)
        # 拼接：原始 system + Skill 段
        new_system = f"{base_system}\n\n{skill_ctx}" if base_system else skill_ctx
        work_messages[0] = {"role": "system", "content": new_system}

    t0 = time.time()
    rounds = 0
    final_answer = ""
    truncated = False
    last_msg = None

    while rounds < MAX_ITERATIONS:
        rounds += 1
        resp = client.chat.completions.create(
            model=model,
            messages=work_messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        last_msg = resp.choices[0].message

        # text-fallback 检测
        tool_calls_to_exec = last_msg.tool_calls
        if not tool_calls_to_exec and last_msg.content:
            text_calls = parse_text_tool_calls(last_msg.content)
            if text_calls:
                tool_calls_to_exec = text_calls

        # 终止：模型直接回答
        if not tool_calls_to_exec:
            final_answer = last_msg.content or ""
            session.append_assistant_text(final_answer)
            if verbose:
                print(f"  → [llm round {rounds}] 模型直接回答 → 循环结束")
            break

        # 记录 assistant 工具调用请求
        native_calls = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in tool_calls_to_exec
        ]
        session.append_assistant_tool_calls(native_calls)

        for tc in tool_calls_to_exec:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if verbose:
                print(f"  → [round {rounds}] {tc.function.name}({args})")
            result = execute_tool(tc.function.name, args)
            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}")
            session.append_tool_result(tc.id, result)

        if verbose:
            print(f"  ↻ round {rounds} 工具已执行完毕，继续下一轮...\n")
    else:
        # while 走完没 break → 熔断
        truncated = True
        final_answer = (last_msg.content if last_msg and last_msg.content else
                        "（循环被熔断，未获得文本回答）")
        session.append_assistant_text(final_answer)
        if verbose:
            print(f"  ⚠ 达到 max_iterations={MAX_ITERATIONS}，强制终止循环")

    return {
        "answer":    final_answer,
        "rounds":    rounds,
        "elapsed":   time.time() - t0,
        "truncated": truncated,
    }
