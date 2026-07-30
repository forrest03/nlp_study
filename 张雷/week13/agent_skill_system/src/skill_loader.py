"""skill_loader.py - Skill 索引 / 主体加载 / LLM 工具提取 / 临时扩展构建。

skill.md 格式见 contracts/skill-format.md：YAML-like front-matter(name/description) + markdown 主体。
front-matter 用极简解析器（仅 key: value），不引入 pyyaml（简化，免依赖）。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .logging import log_event, log_error
from .whitelist import normalize_program

_BLOCK_SCALARS = {">", ">-", ">+", "|", "|-", "|+"}


def _parse_frontmatter(text: str) -> dict[str, str]:
    """极简 YAML front-matter 解析：支持 inline `key: value` 与块标量 `>`/`|`（含 -/+）。

    不引入 pyyaml（离线不可装）。仅覆盖 skill.md 用到的子集。
    """
    meta: dict[str, str] = {}
    lines = text.splitlines()
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            i += 1
            continue
        key, val = line.split(":", 1)
        key, val = key.strip(), val.strip()
        if val in _BLOCK_SCALARS:
            block: list[str] = []
            i += 1
            while i < n and (lines[i].startswith(" ") or lines[i] == ""):
                block.append("" if lines[i] == "" else lines[i].strip())
                i += 1
            if val.startswith(">"):  # folded: 折叠为空格分隔
                meta[key] = " ".join(" ".join(b.split()) for b in block if b)
            else:  # literal: 保留换行
                meta[key] = "\n".join(b for b in block if b).strip()
        else:
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            meta[key] = val
            i += 1
    return meta


@dataclass
class ToolSpec:
    """LLM 从 Skill 主体提取的工具规格（research D4）。"""
    name: str
    description: str
    parameters: dict
    command_template: str
    program: str = ""
    json_input: bool = False  # True: dispatch 把参数写为临时 JSON 文件，用 {__json__} 占位

    def to_function_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ToolSpec":
        ct = d.get("command_template") or ""
        return cls(
            name=d.get("name", ""),
            description=d.get("description", ""),
            parameters=d.get("parameters", {"type": "object", "properties": {}}),
            command_template=ct,
            program=normalize_program(ct.split()[0]) if ct else "",
            json_input=bool(d.get("json_input", False)),
        )


def programs_in_template(template: str) -> set[str]:
    """提取 command_template 中各子命令的程序名（用于临时白名单扩展）。

    覆盖组合命令（如 `mkdir ... && python ...`），返回 {mkdir, python}。
    """
    if not template:
        return set()
    programs: set[str] = set()
    for part in re.split(r";|&&|\|\||\||&", template):
        toks = part.strip().split()
        if toks:
            programs.add(normalize_program(toks[0]))
    return programs


@dataclass
class SkillIndexEntry:
    name: str
    description: str
    path: str


class SkillLoader:
    def __init__(self, skills_dir: str | Path = "skills"):
        self.skills_dir = Path(skills_dir)

    def load_index(self) -> list[SkillIndexEntry]:
        """扫描 skills/，仅读 front-matter（FR-009）。

        兼容 `skill.md` 与 `SKILL.md`（Claude 技能惯例）两种命名。
        """
        entries: list[SkillIndexEntry] = []
        if not self.skills_dir.exists():
            return entries
        for d in sorted(self.skills_dir.iterdir()):
            if not d.is_dir():
                continue
            md = d / "skill.md"
            if not md.exists():
                md = d / "SKILL.md"
            if not md.exists():
                continue
            name, desc, _ = self._parse(md)
            entries.append(SkillIndexEntry(name, desc, str(md)))
        return entries

    def load_body(self, path: str | Path) -> str:
        _, _, body = self._parse(Path(path))
        return body

    def _parse(self, path: Path) -> tuple[str, str, str]:
        text = path.read_text(encoding="utf-8")
        meta: dict[str, str] = {}
        body = text
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                meta = _parse_frontmatter(parts[1])
                body = parts[2].strip()
        return meta.get("name", path.parent.name), meta.get("description", ""), body

    def _extract_tools_from_body(self, body: str) -> list[ToolSpec] | None:
        """尝试从 ```tools 代码块直接解析 ToolSpec（确定性，不依赖 LLM）。

        返回 None 表示未找到结构化定义，调用方应 fallback 到 LLM 提取。
        """
        m = re.search(r"```tools\s*\n(.*?)```", body, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(1))
            arr = data.get("tools", data) if isinstance(data, dict) else data
            if not isinstance(arr, list):
                return None
            return [ToolSpec.from_dict(x) for x in arr if isinstance(x, dict)]
        except (ValueError, TypeError) as e:
            log_error("tools_block_parse_failed", error=repr(e))
            return None

    def extract_tools(self, client, model: str, body: str, skill_name: str) -> list[ToolSpec]:
        """从 Skill 主体提取 ToolSpec：优先 ```tools 结构化块，fallback 到 LLM 提取。

        提取失败 -> 返回 []，由 Agent 降级为默认工具（Assumptions）。
        """
        # 优先确定性解析（FR-011 改进：避免 LLM 提取的不确定性）
        direct = self._extract_tools_from_body(body)
        if direct is not None:
            return direct

        prompt = (
            "从以下技能主体中提取所需工具，返回 JSON 对象 {\"tools\":[...]}。"
            "每个元素含 name/description/parameters(JSON Schema)/command_template(用 {param} 占位标量参数)。"
            "若工具需把结构化参数作为 JSON 文件传给脚本（如 `python xxx.py <jsonfile>`），"
            "设 json_input=true 并在 command_template 中用 {__json__} 占位该文件路径"
            "（系统会自动把参数写为临时 JSON 文件并替换 {__json__}）；此时不要把数组/对象参数内联进命令。"
            "若无工具返回 {\"tools\":[]}。\n\n"
            f"技能：{skill_name}\n\n主体：\n{body}"
        )
        try:
            log_event("llm_request", purpose="extract_tools", skill=skill_name)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or "{}"
            data = json.loads(content)
            arr = data.get("tools", data) if isinstance(data, dict) else data
            if not isinstance(arr, list):
                return []
            tools = [ToolSpec.from_dict(x) for x in arr if isinstance(x, dict)]
            log_event("llm_response", purpose="extract_tools", skill=skill_name,
                      tools=len(tools))
            return tools
        except (ValueError, AttributeError, TypeError) as e:
            log_error("tool_extract_failed", skill=skill_name, error=repr(e))
            return []
