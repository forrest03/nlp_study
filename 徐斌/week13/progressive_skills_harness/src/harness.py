"""
Progressive Skills Harness — 渐进式加载执行 Skills 的 Agent 运行时。

核心流水线（对齐课件）：
  用户消息 → 触发初筛 → 仅注入 L0 索引 → LLM 决策 activate(L1)
  → 按需 read_skill_file(L2) → 写文件/跑脚本 → 可选 release → 最终回答

对比全量加载：system prompt 不塞入全部 SKILL.md，只常驻索引。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Generator

from llm_config import current_model_info, get_chat_client
from progressive_loader import ProgressiveLoader, estimate_tokens
from skill_registry import SkillRegistry
from tools import build_tools, dispatch


ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = ROOT / "skills"
WORKSPACE = ROOT / "workspace"


HARNESS_SYSTEM = """你是运行在 Progressive Skills Harness 中的 Agent。

## Harness 规则（必须遵守）
1. 你一开始**只有 Skill 索引（L0）**，没有完整 Skill 正文。
2. 需要某能力时，先 `activate_skill(name)` 加载完整 SKILL.md（L1）。
3. Skill 内部的 references / 大数据文件，用 `read_skill_file` 按需加载（L2），不要一次读完。
4. 按 Skill 流程使用 `write_file` / `run_skill_script` / `read_file` 完成任务。
5. 任务成功结束后调用 `release_skill` 释放上下文（教学演示）。
6. 不要编造未加载的 Skill 细节；不确定就 activate 或 list_skills。
7. 产物默认写到 `workspace/` 目录。
8. 用中文回复用户；工具调用之间可以简短说明意图。

## 当前可能相关的 Skill（触发初筛，仅供参考）
{trigger_hints}

## Skill 索引（L0 常驻）
{skill_index}
"""


class SkillsHarness:
    def __init__(
        self,
        skills_dir: Path | None = None,
        workspace: Path | None = None,
        max_steps: int = 12,
    ):
        self.root = ROOT
        self.skills_dir = Path(skills_dir or SKILLS_DIR)
        self.workspace = Path(workspace or WORKSPACE)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps

        self.registry = SkillRegistry(self.skills_dir)
        self.loader = ProgressiveLoader(
            registry=self.registry,
            project_root=self.root,
        )
        self.client, self.model = get_chat_client()
        self.schema, self.tools_map = build_tools(
            self.registry, self.loader, self.workspace, self.root
        )

    def reset_loader(self) -> None:
        self.loader = ProgressiveLoader(
            registry=self.registry,
            project_root=self.root,
        )
        self.schema, self.tools_map = build_tools(
            self.registry, self.loader, self.workspace, self.root
        )

    def _system_prompt(self, user_query: str) -> str:
        index = self.loader.index_text()
        hints = self.registry.match_by_triggers(user_query)
        if hints:
            hint_text = "、".join(hints[:5])
        else:
            hint_text = "（无明显关键词命中，由你根据索引语义判断是否 activate）"
        return HARNESS_SYSTEM.format(trigger_hints=hint_text, skill_index=index)

    def run(self, question: str) -> Generator[dict[str, Any], None, None]:
        """执行一轮任务，yield 教学事件。"""
        self.reset_loader()
        system = self._system_prompt(question)
        messages: list[Any] = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]

        snap0 = self.loader.snapshot()
        yield {
            "type": "session_start",
            "model": current_model_info(),
            "question": question,
            "trigger_hints": self.registry.match_by_triggers(question),
            "token_snapshot": snap0,
            "system_tokens": estimate_tokens(system),
        }

        for step in range(1, self.max_steps + 1):
            # L1/L2 正文通过 activate_skill / read_skill_file 的 tool result 进入对话历史，
            # 避免再往 system 重复注入同一份内容。
            yield {
                "type": "llm_call",
                "step": step,
                "active_skills": list(self.loader.active_skills.keys()),
                "token_snapshot": self.loader.snapshot(),
            }

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.schema,
                tool_choice="auto",
                temperature=0.2,
            )
            msg = resp.choices[0].message
            finish = resp.choices[0].finish_reason

            if not msg.tool_calls or finish == "stop":
                answer = msg.content or "（空回复）"
                yield {
                    "type": "final",
                    "step": step,
                    "answer": answer,
                    "token_snapshot": self.loader.snapshot(),
                }
                return

            # 记录 assistant tool call
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )

            for tc in msg.tool_calls:
                name = tc.function.name
                raw_args = tc.function.arguments or "{}"
                try:
                    args_obj = json.loads(raw_args)
                except json.JSONDecodeError:
                    args_obj = {}

                yield {
                    "type": "tool_call",
                    "step": step,
                    "tool": name,
                    "arguments": args_obj,
                }

                result_str = dispatch(self.tools_map, name, args_obj)
                try:
                    result_obj = json.loads(result_str)
                except json.JSONDecodeError:
                    result_obj = {"raw": result_str}

                # 对 activate / read 的 content 做截断展示，但完整结果仍回传模型
                preview = result_obj
                if isinstance(result_obj, dict) and "content" in result_obj:
                    content = result_obj.get("content") or ""
                    preview = {
                        **{k: v for k, v in result_obj.items() if k != "content"},
                        "content_preview": (content[:400] + "...")
                        if len(content) > 400
                        else content,
                        "content_tokens": estimate_tokens(content),
                    }

                yield {
                    "type": "tool_result",
                    "step": step,
                    "tool": name,
                    "result": preview,
                    "token_snapshot": self.loader.snapshot(),
                    "load_events": self.loader.snapshot()["events"][-3:],
                }

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    }
                )

        yield {
            "type": "final",
            "step": self.max_steps,
            "answer": "已达到最大步数限制，请提高 max_steps 或缩小任务范围。",
            "token_snapshot": self.loader.snapshot(),
        }


def run_and_print(question: str, max_steps: int = 12) -> str:
    harness = SkillsHarness(max_steps=max_steps)
    answer = ""
    print("=" * 60)
    print("Progressive Skills Harness")
    info = current_model_info()
    print(f"模型: {info['display']} ({info['model']})")
    print("=" * 60)

    for ev in harness.run(question):
        et = ev["type"]
        if et == "session_start":
            snap = ev["token_snapshot"]
            print(f"\n用户: {ev['question']}")
            print(f"触发初筛: {ev['trigger_hints'] or '无'}")
            print(
                f"L0 索引 ≈ {snap['l0_tokens']} tokens | "
                f"若全量加载 ≈ {snap['full_load_tokens']} tokens | "
                f"system ≈ {ev['system_tokens']} tokens"
            )
        elif et == "llm_call":
            snap = ev["token_snapshot"]
            print(
                f"\n── Step {ev['step']} LLM ── "
                f"active={ev['active_skills'] or []} "
                f"ctx≈{snap['current_tokens']} "
                f"(省 {snap['saved_tokens']})"
            )
        elif et == "tool_call":
            print(f"  → {ev['tool']}({json.dumps(ev['arguments'], ensure_ascii=False)[:200]})")
        elif et == "tool_result":
            snap = ev["token_snapshot"]
            level_hint = ""
            if ev["tool"] in ("activate_skill", "read_skill_file", "release_skill"):
                level_hint = f" | load={snap['l0_tokens']}+{snap['l1_tokens']}+{snap['l2_tokens']}"
            r = ev["result"]
            if isinstance(r, dict) and r.get("ok") is False:
                print(f"  ← ERROR {r.get('error')}{level_hint}")
            else:
                brief = {k: v for k, v in r.items() if k not in ("content", "content_preview")}
                if "content_preview" in r:
                    brief["content_preview"] = r["content_preview"][:120]
                print(f"  ← {json.dumps(brief, ensure_ascii=False)[:300]}{level_hint}")
        elif et == "final":
            answer = ev["answer"]
            snap = ev["token_snapshot"]
            print("\n" + "-" * 60)
            print("Final Answer:\n")
            print(answer)
            print("\n" + "-" * 60)
            print(
                f"Token 对照: 当前≈{snap['current_tokens']} / 全量≈{snap['full_load_tokens']} "
                f"/ 节省≈{snap['saved_tokens']}"
            )
            print(f"加载事件数: {len(snap['events'])}")
            for e in snap["events"]:
                print(f"  [{e['level']}] {e['skill']} {e['path']} (+{e['tokens']}t) {e['detail']}")
    return answer
