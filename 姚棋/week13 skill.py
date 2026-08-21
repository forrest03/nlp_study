"""
harness.py — 渐进式加载 Skills 的 Harness

演示：
  - 常驻技能索引（模拟 MEMORY.md）
  - 触发条件匹配
  - 按需加载完整 Skill 定义
  - 执行 Skill（含模拟工具调用）
  - Context Token 统计与释放

运行：
  python harness.py
"""

import re
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass, field


# ======================== 1. 数据结构 ========================

@dataclass
class SkillIndex:
    """常驻索引（占用极少 token）"""
    name: str
    description: str
    trigger_patterns: List[str]   # 关键词/正则，匹配用户输入


@dataclass
class SkillDefinition:
    """完整 Skill 定义（仅在触发后加载）"""
    name: str
    description: str
    trigger_patterns: List[str]
    steps: List[str]              # 执行步骤描述（自然语言）
    tool_calls: List[Dict]        # 工具调用列表（模拟）
    qa_checks: List[str]          # 自检清单


class SkillRegistry:
    """技能注册与检索"""
    def __init__(self):
        self._index: Dict[str, SkillIndex] = {}       # name -> SkillIndex
        self._definitions: Dict[str, SkillDefinition] = {}  # name -> 完整定义

    def register(self, definition: SkillDefinition):
        """注册一个技能（同时维护索引和完整定义）"""
        self._index[definition.name] = SkillIndex(
            name=definition.name,
            description=definition.description,
            trigger_patterns=definition.trigger_patterns,
        )
        self._definitions[definition.name] = definition

    def get_index(self) -> List[SkillIndex]:
        """获取所有索引（供常驻注入）"""
        return list(self._index.values())

    def match(self, user_input: str) -> Optional[SkillDefinition]:
        """
        根据用户输入匹配技能（基于 trigger_patterns）
        返回第一个匹配的完整定义，若未匹配返回 None
        """
        for idx in self._index.values():
            for pattern in idx.trigger_patterns:
                if re.search(pattern, user_input, re.IGNORECASE):
                    return self._definitions[idx.name]
        return None

    def load_definition(self, name: str) -> Optional[SkillDefinition]:
        """按名称加载完整定义（供显式调用）"""
        return self._definitions.get(name)


# ======================== 2. 模拟工具 ========================

def mock_tool_call(tool_name: str, args: Dict) -> str:
    """模拟工具执行，返回结果"""
    print(f"      🔧 [工具] {tool_name}({args})")
    if tool_name == "web_search":
        return f"搜索结果：关于 '{args['query']}' 的模拟结果..."
    elif tool_name == "calculator":
        return f"计算结果：{args['expr']} = {eval(args['expr'])}"
    elif tool_name == "file_read":
        return f"文件内容（模拟）：{args['path']} 读取成功。"
    else:
        return f"未知工具 {tool_name}，模拟返回空。"


# ======================== 3. Harness 引擎 ========================

class Harness:
    def __init__(self, registry: SkillRegistry):
        self.registry = registry
        self.loaded_skill: Optional[SkillDefinition] = None
        self.context_tokens = 0

    def _estimate_tokens(self, text: str) -> int:
        """粗略估算 token 数（按字符/3）"""
        return len(text) // 3

    def _format_index_prompt(self) -> str:
        """生成常驻索引的 prompt（模拟）"""
        indices = self.registry.get_index()
        lines = ["## 可用技能索引（常驻）"]
        for idx in indices:
            lines.append(f"- [{idx.name}] {idx.description} (触发词: {', '.join(idx.trigger_patterns)})")
        return "\n".join(lines)

    def _format_full_skill(self, skill: SkillDefinition) -> str:
        """生成完整技能定义（加载后注入）"""
        lines = [
            f"## 技能: {skill.name}",
            f"描述: {skill.description}",
            "执行步骤:"
        ]
        for i, step in enumerate(skill.steps, 1):
            lines.append(f"  {i}. {step}")
        if skill.tool_calls:
            lines.append("工具调用:")
            for tc in skill.tool_calls:
                lines.append(f"  - {tc['name']}({tc['args']})")
        if skill.qa_checks:
            lines.append("自检清单:")
            for q in skill.qa_checks:
                lines.append(f"  - [ ] {q}")
        return "\n".join(lines)

    def _execute_skill(self, skill: SkillDefinition, user_input: str) -> str:
        """执行技能：按步骤模拟执行，调用工具"""
        print(f"\n  📦 执行技能: {skill.name}")
        results = []
        for step_idx, step_desc in enumerate(skill.steps, 1):
            print(f"    Step {step_idx}: {step_desc}")
            # 模拟步骤中可能包含工具调用
            # 这里简单打印
            results.append(f"Step {step_idx}: 执行 {step_desc}")

        # 模拟工具调用（如果 skill 定义了 tool_calls）
        for tc in skill.tool_calls:
            result = mock_tool_call(tc['name'], tc['args'])
            results.append(f"工具结果: {result}")

        # 自检
        for q in skill.qa_checks:
            print(f"    ✓ 自检: {q} (通过)")

        return "\n".join(results)

    def process(self, user_input: str) -> str:
        """处理用户输入：匹配 → 加载 → 执行 → 释放"""
        print(f"\n[用户输入] {user_input}")

        # 1. 常驻索引（始终可用）
        index_prompt = self._format_index_prompt()
        index_tokens = self._estimate_tokens(index_prompt)
        print(f"  📋 常驻索引加载，占用 {index_tokens} tokens")

        # 2. 匹配技能
        matched = self.registry.match(user_input)
        if not matched:
            print("  ⚠️ 未匹配到任何技能，使用通用处理。")
            return "抱歉，我没有找到对应的技能来处理您的请求。"

        # 3. 加载完整技能（渐进式）
        print(f"  ✅ 匹配到技能: {matched.name}")
        full_prompt = self._format_full_skill(matched)
        full_tokens = self._estimate_tokens(full_prompt)
        print(f"  📥 加载完整技能定义，新增 {full_tokens} tokens (总 token: {index_tokens + full_tokens})")

        # 4. 执行
        result = self._execute_skill(matched, user_input)

        # 5. 释放技能（模拟）
        print(f"  🧹 技能执行完毕，释放 Context (释放 {full_tokens} tokens)")
        self.loaded_skill = None

        return f"技能执行结果:\n{result}"


# ======================== 4. 定义示例技能 ========================

def build_sample_skills() -> SkillRegistry:
    reg = SkillRegistry()

    # 技能1: 天气查询
    weather_skill = SkillDefinition(
        name="weather",
        description="查询城市天气",
        trigger_patterns=[r"天气", r"气温", r"预报"],
        steps=[
            "解析用户问题中的城市名",
            "调用天气 API 获取实时数据",
            "格式化输出温度、湿度、风速"
        ],
        tool_calls=[
            {"name": "web_search", "args": {"query": "城市天气"}}
        ],
        qa_checks=[
            "是否包含城市名？",
            "温度单位是否正确？"
        ]
    )
    reg.register(weather_skill)

    # 技能2: 计算器
    calc_skill = SkillDefinition(
        name="calculator",
        description="执行数学计算",
        trigger_patterns=[r"计算", r"等于", r"\\d+[+\\-*/]"],
        steps=[
            "提取表达式中的数字和运算符",
            "调用计算器工具计算",
            "返回结果"
        ],
        tool_calls=[
            {"name": "calculator", "args": {"expr": "2+3"}}
        ],
        qa_checks=[
            "表达式是否完整？",
            "结果是否在合理范围？"
        ]
    )
    reg.register(calc_skill)

    # 技能3: 代码审查
    review_skill = SkillDefinition(
        name="code-review",
        description="审查代码变更",
        trigger_patterns=[r"review", r"审查", r"代码", r"PR"],
        steps=[
            "获取变更文件列表",
            "检查语法与风格",
            "检查安全漏洞",
            "生成审查报告"
        ],
        tool_calls=[
            {"name": "file_read", "args": {"path": "src/main.py"}}
        ],
        qa_checks=[
            "是否覆盖所有变更？",
            "是否标注严重问题？"
        ]
    )
    reg.register(review_skill)

    return reg


# ======================== 5. 主程序 ========================

def main():
    print("=" * 60)
    print("渐进式 Skills Harness 演示")
    print("=" * 60)
    print("\n可用技能（常驻索引）:")
    registry = build_sample_skills()
    for idx in registry.get_index():
        print(f"  - {idx.name}: {idx.description} (触发: {', '.join(idx.trigger_patterns)})")
    print("\n输入您的问题（输入 'exit' 退出）:")

    harness = Harness(registry)

    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in ('exit', 'quit', 'q'):
            break
        if not user_input:
            continue
        response = harness.process(user_input)
        print(f"\n最终回复:\n{response}")

    print("\n再见！")


if __name__ == "__main__":
    main()
