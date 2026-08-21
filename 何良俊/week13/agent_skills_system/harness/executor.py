"""
executor — 通用 skill 执行器
============================

根据 SKILL.md frontmatter 声明的执行契约分派执行模式，**不包含任何
skill 专用代码**。新增 skill 只需在 SKILL.md 声明契约，无需改本文件。

执行模式（按 frontmatter 字段组合自动判定）：

  模式 A — script + data_file:
    有 entry 且 entry_input=data_file + data_instructions
    → LLM 按 data_instructions 生成 JSON → 写 data/<name>.json
    → 执行 entry <path>

  模式 B — script + args:
    有 entry 且 entry_input=args
    → 参数直接作 argv → 执行 entry <args...>

  模式 C — script + stdin:
    有 entry 且 entry_input=stdin + data_instructions
    → LLM 生成 JSON → 通过 stdin 传给 entry

  模式 D — 生成模式 (generate):
    无 entry 但有 output_ext
    → LLM 按 SKILL.md body 生成产物 → 写 output_subdir/<name>.<ext>
    → 可选 post_process 脚本做后处理（如 SVG→PNG）

记忆接入：执行器从 memory 读取该 skill 的 items_generated 列表，拼到
LLM prompt 中以避免重复 / 参考历史风格。必需参数缺失时，调通用 LLM
函数结合历史补全。
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .loader import LoadedSkill
from .memory import MemorySystem
from .llm import DeepSeekClient


@dataclass
class Invocation:
    """一条可执行的 skill 调用。"""

    cmd: list[str]
    cwd: Path
    description: str
    stdin_data: Optional[str] = None
    artifacts: dict = field(default_factory=dict)


@dataclass
class ExecutionResult:
    ok: bool
    phase: str  # "executed" | "not-runnable" | "llm-error"
    stdout: str = ""
    stderr: str = ""
    exit_code: Optional[int] = None
    invocation: Optional[Invocation] = None
    note: str = ""


# ---------------------------------------------------------------------------
# 通用执行器
# ---------------------------------------------------------------------------

class GenericExecutor:
    """按 frontmatter 契约通用执行 skill。无任何 skill 专用逻辑。"""

    def __init__(
        self,
        memory: MemorySystem,
        *,
        llm: Optional[DeepSeekClient] = None,
        timeout: int = 120,
        on_progress: Optional[Callable[[str], None]] = None,
    ):
        self.memory = memory
        self.llm = llm
        self.timeout = timeout
        self._on_progress = on_progress

    def _progress(self, msg: str) -> None:
        """通知调用方当前执行进度（如「正在生成数据...」），用于实时反馈。"""
        if self._on_progress:
            self._on_progress(msg)

    def execute(
        self,
        skill: LoadedSkill,
        user_input: str,
        args: Optional[dict] = None,
    ) -> ExecutionResult:
        # Phase 3: 发现脚本/数据/参考
        _discover_scripts_inline(skill)

        args = args or {}
        try:
            invocation = self._dispatch(skill, user_input, args)
        except Exception as e:
            self.memory.record(
                user_input=user_input,
                skill=skill.name,
                phase="llm-error",
                result={"ok": False},
                error=str(e),
            )
            return ExecutionResult(
                ok=False, phase="llm-error",
                note=f"执行失败: {e}", stderr=str(e),
            )

        if invocation is None:
            return self._not_runnable(skill, user_input, args)

        # 空 cmd：执行器已直接产出文件，无需子进程
        if not invocation.cmd:
            self._update_state(skill, invocation, None)
            self.memory.record(
                user_input=user_input,
                skill=skill.name,
                phase="executed",
                result={
                    "ok": True,
                    "exit_code": None,
                    "description": invocation.description,
                    "artifacts": invocation.artifacts,
                    "no_subprocess": True,
                },
            )
            return ExecutionResult(
                ok=True, phase="executed",
                note=f"已直接产出: {invocation.artifacts}",
                invocation=invocation,
            )

        return self._run_invocation(skill, user_input, invocation)

    # ---- 分派 ----------------------------------------------------------
    def _dispatch(
        self,
        skill: LoadedSkill,
        user_input: str,
        args: dict,
    ) -> Optional[Invocation]:
        if self.llm is None:
            return None

        # 必需参数补全：缺失时调 LLM 结合历史填值
        args = self._fill_missing_params(skill, user_input, args)
        if args is None:
            return None  # 无法补全

        if skill.entry:
            entry_path = skill.skill_dir / skill.entry
            if not entry_path.exists():
                raise RuntimeError(f"entry script not found: {entry_path}")

            mode = (skill.entry_input or "args").strip()
            if mode == "data_file":
                return self._mode_script_data_file(skill, args)
            elif mode == "stdin":
                return self._mode_script_stdin(skill, args)
            elif mode == "args":
                return self._mode_script_args(skill, args, entry_path)
            else:
                raise RuntimeError(f"unknown entry_input: {mode}")

        # 无 entry → 生成模式
        if skill.output_ext:
            return self._mode_generate(skill, args)

        # 既无 entry 又无 output_ext → 该 skill 没有声明可执行契约
        return None

    # ---- 模式 A: script + data_file -----------------------------------
    def _mode_script_data_file(
        self, skill: LoadedSkill, args: dict,
    ) -> Optional[Invocation]:
        entry_path = skill.skill_dir / skill.entry
        # 用首个 param 的值作为文件名 key（如 word）
        key = self._item_name(skill, args)
        existing = self._existing_items(skill)

        # 已生成过 → 复用
        if key in [it["name"] for it in existing]:
            data_file = skill.data_files.get(key) or (skill.skill_dir / "data" / f"{key}.json")
            if data_file.exists():
                self._progress(f"  ✓ 已有数据，复用 {key}")
                return Invocation(
                    cmd=[sys.executable, str(entry_path), str(data_file)],
                    cwd=Path.cwd(),
                    description=f"复用已有产物: {skill.name}/{key}",
                    artifacts={"item_name": key, "data_file": str(data_file), "reused": True},
                )

        # 调 LLM 生成数据
        if not skill.data_instructions:
            raise RuntimeError(f"skill {skill.name} 声明 entry_input=data_file 但未声明 data_instructions")
        self._progress(f"  ⏳ 正在用 LLM 生成 {key} 的数据...")
        prompt = self._render_data_instructions(skill, args, existing)
        resp = self.llm.chat_json(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        if not resp.json_obj:
            raise RuntimeError(f"LLM 未返回 JSON: {resp.text[:200]}")
        self._progress(f"  ✓ 数据已生成，写入 {key}.json")

        data_file = skill.skill_dir / "data" / f"{key}.json"
        data_file.parent.mkdir(parents=True, exist_ok=True)
        data_file.write_text(
            json.dumps(resp.json_obj, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        skill.data_files[key] = data_file

        self._progress(f"  ⏳ 正在执行脚本 {skill.entry}...")
        return Invocation(
            cmd=[sys.executable, str(entry_path), str(data_file)],
            cwd=Path.cwd(),
            description=f"生成 {skill.name}: {key}",
            artifacts={"item_name": key, "data_file": str(data_file)},
        )

    # ---- 模式 B: script + args ----------------------------------------
    def _mode_script_args(
        self, skill: LoadedSkill, args: dict, entry_path: Path,
    ) -> Invocation:
        argv = [str(v) for p in skill.params for v in [args.get(p["name"])] if v]
        return Invocation(
            cmd=[sys.executable, str(entry_path)] + argv,
            cwd=Path.cwd(),
            description=f"执行 {skill.name}: {argv}",
            artifacts={"args": argv},
        )

    # ---- 模式 C: script + stdin ---------------------------------------
    def _mode_script_stdin(
        self, skill: LoadedSkill, args: dict,
    ) -> Optional[Invocation]:
        entry_path = skill.skill_dir / skill.entry
        existing = self._existing_items(skill)
        if not skill.data_instructions:
            raise RuntimeError(f"skill {skill.name} 声明 entry_input=stdin 但未声明 data_instructions")
        self._progress(f"  ⏳ 正在用 LLM 生成数据...")
        prompt = self._render_data_instructions(skill, args, existing)
        resp = self.llm.chat_json(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        if not resp.json_obj:
            raise RuntimeError(f"LLM 未返回 JSON: {resp.text[:200]}")
        self._progress(f"  ✓ 数据已生成，准备通过 stdin 传给脚本")
        stdin_data = json.dumps(resp.json_obj, ensure_ascii=False)
        self._progress(f"  ⏳ 正在执行脚本 {skill.entry}...")
        return Invocation(
            cmd=[sys.executable, str(entry_path)],
            cwd=Path.cwd(),
            stdin_data=stdin_data,
            description=f"执行 {skill.name} (stdin)",
            artifacts={"item_name": self._item_name(skill, args)},
        )

    # ---- 模式 D: 生成模式 ---------------------------------------------
    def _mode_generate(
        self, skill: LoadedSkill, args: dict,
    ) -> Optional[Invocation]:
        existing = self._existing_items(skill)
        history_hint = ""
        if existing:
            items_desc = ", ".join(
                f"{it['name']}({it.get('params', {})})" for it in existing[-5:]
            )
            history_hint = (
                f"# 历史参考\n用户已生成过: {items_desc}\n"
                f"请保持视觉风格一致，但不要重复同一主题。\n\n"
            )

        params_block = "\n".join(
            f"- {p['name']}: {args.get(p['name'], '(未提供)')}"
            for p in skill.params
        ) or "(无参数)"

        system_prompt = (
            f"你是 {skill.name} 产物生成器。请严格按照下列 SKILL 指令生成单个"
            f"自包含 {skill.output_ext} 文件。\n"
            f"输出必须是合法内容，不要加任何解释文字、不要包裹在代码块中、"
            f"不要加 ```。直接输出文件内容。\n"
        )
        user_prompt = (
            f"# SKILL 指令\n{skill.body}\n\n"
            f"{history_hint}"
            f"# 参数\n{params_block}\n\n"
            f"请生成 {skill.output_ext} 内容。"
        )

        name = self._item_name(skill, args)
        self._progress(f"  ⏳ 正在用 LLM 生成 {name}{skill.output_ext} 内容（可能需要数十秒，请稍候）...")

        # 生成模式较慢（SVG 可能要数十秒），启动后台线程每隔 5 秒打印一次
        # 「仍在生成...」让用户知道没卡死。流式 token 增量不打印（SVG 内容
        # 对用户无意义，打印出来全是点号或乱码）。
        import threading
        stop_hint = threading.Event()

        def _hint_loop():
            while not stop_hint.wait(10):
                self._progress("  ⏳ 仍在生成中...")

        hint_thread = threading.Thread(target=_hint_loop, daemon=True)
        hint_thread.start()

        try:
            resp = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=16000,
                stream=True,
                on_text_delta=lambda _: None,  # 不打印 SVG token 增量
            )
        finally:
            stop_hint.set()
            hint_thread.join(timeout=1.0)
        self._progress(f"  ✓ 内容生成完毕")

        content = _strip_code_fence(resp.text)
        slug = re.sub(r"[^\w-]+", "_", name)[:40].strip("_") or skill.name
        subdir = skill.output_subdir or skill.name
        out_dir = Path.cwd() / subdir / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{slug}{skill.output_ext}"
        out_path.write_text(content, encoding="utf-8")
        self._progress(f"  ✓ 已写入 {out_path}")

        artifacts = {
            "item_name": name,
            "output_path": str(out_path),
            "params": args,
            "llm_tokens": resp.usage.get("total_tokens"),
        }

        # 可选后处理（如 SVG→PNG）
        post_cmd: list[str] = []
        if skill.post_process:
            pp_path = skill.skill_dir / skill.post_process
            if pp_path.exists():
                bun = _find_bun_command()
                if bun and pp_path.suffix == ".ts":
                    post_cmd = bun + [str(pp_path), str(out_path)]
                elif pp_path.suffix == ".py":
                    post_cmd = [sys.executable, str(pp_path), str(out_path)]

        if post_cmd:
            self._progress(f"  ⏳ 正在执行后处理 {skill.post_process}...")
            return Invocation(
                cmd=post_cmd,
                cwd=Path.cwd(),
                description=f"后处理 {skill.name}: {slug}",
                artifacts=artifacts,
            )
        return Invocation(
            cmd=[],
            cwd=Path.cwd(),
            description=f"生成 {skill.name}: {slug}",
            artifacts=artifacts,
        )

    # ---- 通用辅助 ------------------------------------------------------
    def _existing_items(self, skill: LoadedSkill) -> list[dict]:
        if self.memory is None:
            return []
        state = self.memory.get_state(skill.name)
        return list(state.get("items_generated", []))

    def _item_name(self, skill: LoadedSkill, args: dict) -> str:
        """用 output_name 模板或首个参数值作为产物名。"""
        if skill.output_name:
            try:
                return skill.output_name.format(**args) or skill.name
            except (KeyError, IndexError):
                pass
        for p in skill.params:
            v = args.get(p["name"])
            if v:
                return str(v)
        return skill.name

    def _render_data_instructions(
        self, skill: LoadedSkill, args: dict, existing: list[dict],
    ) -> str:
        """渲染 data_instructions 模板，填入参数与 avoid_hint。"""
        tmpl = skill.data_instructions
        # 填 {param} 占位
        try:
            prompt = tmpl.format(
                **args,
                avoid_hint=self._avoid_hint(existing),
            )
        except KeyError:
            prompt = tmpl
        return prompt

    def _avoid_hint(self, existing: list[dict]) -> str:
        if not existing:
            return ""
        names = [it["name"] for it in existing]
        return (
            f"参考信息：用户已生成过: {names}\n"
            f"请确保本次生成的内容不与上述重复。"
        )

    def _fill_missing_params(
        self, skill: LoadedSkill, user_input: str, args: dict,
    ) -> Optional[dict]:
        """必需参数缺失时，调 LLM 结合历史补全。"""
        if not skill.params:
            return args
        missing = [
            p for p in skill.params
            if p.get("required") and not args.get(p["name"])
        ]
        if not missing:
            return args

        existing = self._existing_items(skill)
        prompt = (
            f"用户请求: {user_input}\n\n"
            f"skill: {skill.name}\n"
            f"需要补全的参数:\n"
            + "\n".join(
                f"- {p['name']} ({p.get('type','string')}): {p.get('description','')}"
                for p in missing
            )
            + f"\n\n已生成过的产物: {[it['name'] for it in existing] or '(无)'}\n"
            f"请为缺失参数推荐不与历史重复的值。"
            f"严格输出 JSON: {{\"<param_name>\": \"<value>\", ...}}"
        )
        resp = self.llm.chat_json(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        if not resp.json_obj:
            return args
        for p in missing:
            v = resp.json_obj.get(p["name"])
            if v:
                args[p["name"]] = v
        return args

    # ---- 子进程执行 ----------------------------------------------------
    def _run_invocation(
        self, skill: LoadedSkill, user_input: str, inv: Invocation,
    ) -> ExecutionResult:
        try:
            proc = subprocess.run(
                inv.cmd,
                cwd=str(inv.cwd),
                input=inv.stdin_data,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding="utf-8",
                errors="replace",
            )
            ok = proc.returncode == 0
            result = ExecutionResult(
                ok=ok, phase="executed",
                stdout=proc.stdout, stderr=proc.stderr,
                exit_code=proc.returncode, invocation=inv,
            )
            self._update_state(skill, inv, result)
            self.memory.record(
                user_input=user_input,
                skill=skill.name,
                phase="executed",
                result={
                    "ok": ok,
                    "exit_code": proc.returncode,
                    "cmd": inv.cmd,
                    "description": inv.description,
                    "artifacts": inv.artifacts,
                    "stdout_tail": proc.stdout[-500:],
                    "stderr_tail": proc.stderr[-500:],
                },
                error=None if ok else (proc.stderr[-500:] or f"exit {proc.returncode}"),
            )
            return result
        except subprocess.TimeoutExpired:
            err = f"timeout after {self.timeout}s"
            self.memory.record(user_input, skill.name, "executed", {"ok": False}, err)
            return ExecutionResult(ok=False, phase="executed", stderr=err, invocation=inv)
        except FileNotFoundError as e:
            err = f"command not found: {e}"
            self.memory.record(user_input, skill.name, "executed", {"ok": False}, err)
            return ExecutionResult(ok=False, phase="executed", stderr=err, invocation=inv)

    def _update_state(
        self, skill: LoadedSkill, inv: Invocation, result: Optional[ExecutionResult],
    ) -> None:
        ok = result.ok if result else True
        if not ok:
            return
        name = inv.artifacts.get("item_name") or skill.name
        params = inv.artifacts.get("params") or {}
        # 从 artifacts 反推 params（模式 A/B 没有 params 字段时）
        if not params and skill.params:
            params = {p["name"]: inv.artifacts.get(p["name"]) for p in skill.params}
        state = self.memory.get_state(skill.name)
        items = state.setdefault("items_generated", [])
        if name not in [it["name"] for it in items]:
            items.append({"name": name, "path": inv.artifacts.get("output_path") or inv.artifacts.get("data_file", ""), "params": params})
        state["last_item"] = {"name": name, "params": params}
        self.memory.set_state(skill.name, state)

    def _not_runnable(self, skill: LoadedSkill, user_input: str, args: dict) -> ExecutionResult:
        scripts_info = ", ".join(skill.scripts.keys()) or "(none)"
        note = (
            f"skill '{skill.name}' 被命中但当前不可执行。\n"
            f"  LLM 抽取参数: {args or '(空)'}\n"
            f"  契约: entry={skill.entry or '(无)'}, entry_input={skill.entry_input or '(无)'}, "
            f"output_ext={skill.output_ext or '(无)'}\n"
            f"  scripts: {scripts_info}\n"
            f"  → 可能原因：该 skill 未声明 entry/output_ext，或必需参数无法补全。"
        )
        self.memory.record(
            user_input=user_input,
            skill=skill.name,
            phase="not-runnable",
            result={"args": args, "note_preview": note[:500]},
        )
        return ExecutionResult(ok=False, phase="not-runnable", note=note)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def _find_bun_command() -> Optional[list[str]]:
    from shutil import which
    if which("bun"):
        return ["bun"]
    if sys.platform != "win32" and which("npx"):
        return ["npx", "-y", "bun"]
    return None


def _discover_scripts_inline(skill: LoadedSkill) -> None:
    """枚举 skill 目录下的 scripts / data / references 子目录文件。"""
    if skill._scripts_loaded:
        return
    for sub, target in (("scripts", skill.scripts), ("data", skill.data_files), ("references", skill.references)):
        d = skill.skill_dir / sub
        if d.is_dir():
            for p in sorted(d.iterdir()):
                if p.is_file() and not p.name.startswith("."):
                    if sub == "scripts" and p.suffix not in (".py", ".ts", ".js", ".sh"):
                        continue
                    key = p.stem if sub in ("data", "references") else p.name
                    target[key] = p
    skill._scripts_loaded = True
