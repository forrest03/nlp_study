"""
工具管理类。

这里统一定义智能体可以调用的工具：
1. 执行本地脚本
2. 查询当前操作系统信息
3. 查询城市经纬度
4. 根据经纬度查询天气
5. 根据公司名称查询股票代码
6. 根据股票代码查询股票信息
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
WEEK11_DIR = PROJECT_DIR.parent / "week11"
WEEK12_TOOLS = PROJECT_DIR.parent / "week12" / "week12 agent" / "react_financial_agent" / "src"

if str(WEEK11_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK11_DIR))

if str(WEEK12_TOOLS) not in sys.path:
    sys.path.insert(0, str(WEEK12_TOOLS))

from query_geo import query_geo  # noqa: E402
from query_weater import query_weater  # noqa: E402
from memery_manager import MemoryManager  # noqa: E402


COMPANY_CODE_MAP = {
    "贵州茅台": "600519",
    "茅台": "600519",
    "五粮液": "000858",
    "宁德时代": "300750",
    "中国平安": "601318",
    "平安": "601318",
    "海康威视": "002415",
    "海康": "002415",
    "比亚迪": "002594",
    "招商银行": "600036",
    "隆基绿能": "601012",
}


class ToolManager:
    """统一管理工具 schema、工具函数映射和工具执行。"""

    def __init__(self, memory_manager: MemoryManager | None = None):
        self.memory_manager = memory_manager
        self.tools_schema = self._build_tools_schema()
        self.tools_map = {
            "run_local_script": self.run_local_script,
            "list_directory": self.list_directory,
            "read_text_file": self.read_text_file,
            "write_text_file": self.write_text_file,
            "open_file": self.open_file,
            "get_system_info": self.get_system_info,
            "query_geo": self.query_geo_tool,
            "query_weather": self.query_weather_tool,
            "company_lookup": self.company_lookup,
            "stock_info": self.stock_info,
            "update_soul_memory": self.update_soul_memory,
            "update_user_memory": self.update_user_memory,
            "update_long_term_memory": self.update_long_term_memory,
        }

    def _build_tools_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "run_local_script",
                    "description": "执行一个已经存在的本地脚本文件。支持 .py、.sh、.zsh、.bash、.command，也支持 .ts/.tsx（会优先用 bun，没有 bun 时尝试 npx -y bun）。script_path 必须是脚本文件真实路径，不是 shell 命令字符串，也不是 /bin/bash 或 /usr/bin/python3 这类解释器路径。不要用它修改 soul.md、user.md、memery.md 这类记忆文件。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "script_path": {
                                "type": "string",
                                "description": "脚本绝对路径或相对路径",
                            },
                            "args": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "传给脚本的参数列表，可选",
                            },
                        },
                        "required": ["script_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "列出目录内容，适合先确认 skill 目录下有哪些 scripts、data、references 文件。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "要查看的目录路径",
                            }
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_text_file",
                    "description": "读取本地文本文件内容，适合查看 skill 的参考文档、脚本源码、json、md、svg 等文本文件。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "要读取的文件路径",
                            }
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_text_file",
                    "description": "把文本内容写入指定文件。如果父目录不存在会自动创建。适合生成 json、md、txt、html、svg 等文本文件。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "目标文件路径",
                            },
                            "content": {
                                "type": "string",
                                "description": "要写入的完整文本内容",
                            },
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "open_file",
                    "description": "用系统默认程序打开一个本地文件，适合打开生成的 html 或图片进行预览。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "要打开的本地文件路径",
                            }
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_system_info",
                    "description": "查询当前操作系统、Python 版本、机器架构等基础环境信息。",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_geo",
                    "description": "根据城市名称查询经纬度。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city_name": {
                                "type": "string",
                                "description": "城市名称，比如北京、上海、广州",
                            }
                        },
                        "required": ["city_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_weather",
                    "description": "根据经纬度查询当前天气和未来3天预报。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "lat": {"type": "number", "description": "纬度"},
                            "lon": {"type": "number", "description": "经度"},
                            "location_name": {
                                "type": "string",
                                "description": "地点名称，可选",
                            },
                        },
                        "required": ["lat", "lon"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "company_lookup",
                    "description": "根据公司中文名称查询股票代码。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "公司中文名称，比如贵州茅台、宁德时代",
                            }
                        },
                        "required": ["name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "stock_info",
                    "description": "根据股票代码查询股票基础信息和最近行情概览。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "股票代码，比如600519、300750",
                            }
                        },
                        "required": ["symbol"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_soul_memory",
                    "description": "把智能体的名字、人格、回答风格、行为准则写入 soul.md。只有涉及智能体自身设定时才调用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "要写入 soul.md 的记忆内容",
                            }
                        },
                        "required": ["content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_user_memory",
                    "description": "把用户稳定偏好、身份、习惯写入 user.md。比如用户偏好中文回答、常用模型等。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "要写入 user.md 的记忆内容",
                            }
                        },
                        "required": ["content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_long_term_memory",
                    "description": "把项目上下文、长期任务、路径约束等写入 memery.md。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "要写入 memery.md 的记忆内容",
                            }
                        },
                        "required": ["content"],
                    },
                },
            },
        ]

    def get_tools_schema(self) -> list[dict[str, Any]]:
        return self.tools_schema

    def get_tools_map(self) -> dict[str, Any]:
        return self.tools_map

    def execute_tool(self, tool_name: str, tool_args: dict[str, Any] | None = None) -> str:
        tool_args = tool_args or {}
        tool_fn = self.tools_map.get(tool_name)
        if tool_fn is None:
            return f"未知工具：{tool_name}"

        try:
            return str(tool_fn(**tool_args))
        except TypeError as e:
            return f"工具参数错误：{e}"
        except Exception as e:
            return f"工具执行失败：{e}"

    def execute_tool_call(self, tool_call: Any) -> str:
        tool_name = tool_call.function.name
        raw_args = tool_call.function.arguments or "{}"
        tool_args = json.loads(raw_args)
        return self.execute_tool(tool_name, tool_args)

    def run_local_script(self, script_path: str, args: list[str] | None = None) -> str:
        args = args or []
        joined = " ".join([script_path, *args])
        if any(name in joined for name in ["soul.md", "user.md", "memery.md"]):
            return "禁止使用 run_local_script 修改记忆文件，请改用 update_soul_memory / update_user_memory / update_long_term_memory"

        if any(flag in script_path for flag in [" ", ";", "&&", "|", "2>/dev/null", ">"]):
            return "run_local_script 只接受脚本文件路径，不接受 shell 命令字符串。请改用真实脚本路径，或先用 write_text_file 生成脚本/数据文件。"

        path = Path(script_path).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()

        if not path.exists():
            return f"脚本不存在：{path}"

        allowed_exts = {".py", ".sh", ".zsh", ".bash", ".command", ".ts", ".tsx", ".js", ".mjs", ".cjs"}
        if path.suffix not in allowed_exts:
            return f"暂不支持这个脚本类型：{path.suffix}，当前只支持 {sorted(allowed_exts)}"

        if path.suffix == ".py":
            command = [sys.executable, str(path), *args]
        elif path.suffix in {".ts", ".tsx"}:
            bun_bin = shutil.which("bun")
            npx_bin = shutil.which("npx")
            if bun_bin:
                command = [bun_bin, str(path), *args]
            elif npx_bin:
                command = [npx_bin, "-y", "bun", str(path), *args]
            else:
                return "执行 TypeScript 脚本需要 bun，或者系统里至少要有 npx 以便自动调用 bun。"
        elif path.suffix in {".js", ".mjs", ".cjs"}:
            node_bin = shutil.which("node")
            if not node_bin:
                return "执行 JavaScript 脚本需要 node，但当前系统未找到 node。"
            command = [node_bin, str(path), *args]
        else:
            command = [str(path), *args]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
        )

        output = (result.stdout or "").strip()
        error = (result.stderr or "").strip()
        lines = [
            f"命令：{' '.join(command)}",
            f"工作目录：{Path.cwd()}",
            f"返回码：{result.returncode}",
        ]
        if output:
            lines.append(f"标准输出：\n{output}")
        if error:
            lines.append(f"标准错误：\n{error}")
        return "\n".join(lines)

    def list_directory(self, path: str) -> str:
        dir_path = Path(path).expanduser()
        if not dir_path.is_absolute():
            dir_path = (Path.cwd() / dir_path).resolve()
        if not dir_path.exists():
            return f"目录不存在：{dir_path}"
        if not dir_path.is_dir():
            return f"路径不是目录：{dir_path}"

        items = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        if not items:
            return f"目录为空：{dir_path}"

        lines = [f"目录内容：{dir_path}"]
        for item in items:
            prefix = "[DIR]" if item.is_dir() else "[FILE]"
            lines.append(f"{prefix} {item.name}")
        return "\n".join(lines)

    def read_text_file(self, path: str) -> str:
        file_path = Path(path).expanduser()
        if not file_path.is_absolute():
            file_path = (Path.cwd() / file_path).resolve()
        if not file_path.exists():
            return f"文件不存在：{file_path}"
        if not file_path.is_file():
            return f"路径不是文件：{file_path}"

        content = file_path.read_text(encoding="utf-8")
        max_chars = 12000
        if len(content) > max_chars:
            content = content[:max_chars] + "\n...(内容过长，已截断)"
        return f"文件内容：{file_path}\n{content}"

    def write_text_file(self, path: str, content: str) -> str:
        file_path = Path(path).expanduser()
        if not file_path.is_absolute():
            file_path = (Path.cwd() / file_path).resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"已写入文件：{file_path}"

    def open_file(self, path: str) -> str:
        file_path = Path(path).expanduser()
        if not file_path.is_absolute():
            file_path = (Path.cwd() / file_path).resolve()
        if not file_path.exists():
            return f"文件不存在：{file_path}"

        result = subprocess.run(
            ["open", str(file_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return f"已打开文件：{file_path}"
        return f"打开文件失败：{result.stderr.strip() or result.stdout.strip()}"

    def get_system_info(self) -> str:
        lines = [
            f"操作系统：{platform.system()}",
            f"系统版本：{platform.version()}",
            f"机器架构：{platform.machine()}",
            f"Python版本：{platform.python_version()}",
            f"平台信息：{platform.platform()}",
        ]
        return "\n".join(lines)

    def query_geo_tool(self, city_name: str) -> str:
        lat, lon = query_geo(city_name)
        return f"{city_name} 的经纬度为：{lat}, {lon}"

    def query_weather_tool(self, lat: float, lon: float, location_name: str = "") -> str:
        return query_weater(lat=lat, lon=lon, location_name=location_name)

    def company_lookup(self, name: str) -> str:
        name = name.strip()
        code = COMPANY_CODE_MAP.get(name)
        if code:
            return f"{name} 的股票代码为 {code}"

        candidates = [k for k in COMPANY_CODE_MAP if name in k]
        if candidates:
            candidate_text = "、".join(f"{k}({COMPANY_CODE_MAP[k]})" for k in candidates)
            return f"未精确匹配到 '{name}'，你可能想找：{candidate_text}"

        support_text = "、".join(COMPANY_CODE_MAP.keys())
        return f"没有找到 '{name}' 对应的股票代码。当前内置支持：{support_text}"

    def stock_info(self, symbol: str) -> str:
        try:
            import akshare as ak
        except ImportError:
            return "当前环境没有安装 akshare，先执行 pip install akshare 再试。"

        symbol = symbol.strip()

        try:
            info_df = ak.stock_individual_info_em(symbol=symbol)
        except Exception as e:
            return f"查询股票基础信息失败：{e}"

        info_lines = [f"股票代码：{symbol}"]
        if info_df is not None and not info_df.empty:
            for _, row in info_df.iterrows():
                item = row.get("item")
                value = row.get("value")
                if item is None:
                    item = row.get("项目")
                if value is None:
                    value = row.get("值")
                if item is not None:
                    info_lines.append(f"{item}：{value}")

        try:
            spot_df = ak.stock_zh_a_spot_em()
            target = spot_df[spot_df["代码"] == symbol]
            if not target.empty:
                row = target.iloc[0]
                info_lines.append("")
                info_lines.append("实时行情概览：")
                for field in ["名称", "最新价", "涨跌幅", "涨跌额", "成交量", "成交额", "今开", "最高", "最低", "昨收"]:
                    if field in row:
                        info_lines.append(f"{field}：{row[field]}")
        except Exception:
            pass

        return "\n".join(info_lines)

    def update_soul_memory(self, content: str) -> str:
        if self.memory_manager is None:
            return "MemoryManager 未注入，无法更新 soul.md"
        return self.memory_manager.update_soul_memory(content)

    def update_user_memory(self, content: str) -> str:
        if self.memory_manager is None:
            return "MemoryManager 未注入，无法更新 user.md"
        return self.memory_manager.update_user_memory(content)

    def update_long_term_memory(self, content: str) -> str:
        if self.memory_manager is None:
            return "MemoryManager 未注入，无法更新 memery.md"
        return self.memory_manager.update_long_term_memory(content)


if __name__ == "__main__":
    manager = ToolManager()
    print("当前工具：")
    for item in manager.get_tools_schema():
        print("-", item["function"]["name"])
