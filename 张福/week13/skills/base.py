"""技能基类"""
from typing import Any, Dict, List


class Skill:
    name: str = ""
    description: str = ""
    tools: List[Dict] = []

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
