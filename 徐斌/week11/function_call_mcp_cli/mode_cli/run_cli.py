"""
run_cli.py — 方式三：CLI（命令行即工具），两种形态

教学重点：
  1. 形态 A（具名 run_cli）：LLM 调一个 run_cli(command, args) 工具，command 是白名单 enum，
     host 拼出子命令执行。安全可控，但每加一个命令要改代码
  2. 形态 B（通用 run_bash）：LLM 自己拼完整 shell 命令，host 在沙箱里执行。
     最灵活、最危险——教学重点是沙箱设计（白名单/黑名单/超时/工作目录锁定）
  3. 与前两方式对比：CLI 是"工具实现层"，Function Call 是"意图生成层"，MCP 是"协议接入层"

使用方式：
  pip install -e .
  python mode_cli/run_cli.py --mode named --question "宁德现在天气怎么样？"
  python mode_cli/run_cli.py --mode bash --question "北京今天天气如何？"
  python mode_cli/run_cli.py --mode named --demo

依赖：
  pip install openai
  环境变量：DEEPSEEK_API_KEY（默认 LLM）
"""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent
CLI_DIR = Path(__file__).parent / "cli"
PY = sys.executable

_FINCLI = shutil.which("fincli") or None
FINCLI_ARGV = ["fincli"] if _FINCLI else [PY, str(CLI_DIR / "main.py")]

PROVIDERS = {
    "deepseek": {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "dashscope": {
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
}


def build_client(provider: str):
    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        print(f"错误：未设置 {provider.upper()}_API_KEY", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


NAMED_COMMANDS = {
    "geocode": {
        "argv": FINCLI_ARGV + ["geocode"],
        "arg_map": {"city": "--city"},
    },
    "weather_by_coords": {
        "argv": FINCLI_ARGV + ["weather-by-coords"],
        "arg_map": {
            "latitude": "--latitude",
            "longitude": "--longitude",
            "location_name": "--location-name",
        },
    },
}


def run_named(command: str, args: dict) -> str:
    spec = NAMED_COMMANDS.get(command)
    if spec is None:
        return f"[run_cli] 未知命令：{command}（白名单：{list(NAMED_COMMANDS)})"

    argv = list(spec["argv"])
    for key, flag in spec["arg_map"].items():
        val = args.get(key)
        if val is not None:
            argv.extend([flag, str(val)])

    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=30,
            cwd=str(BASE_DIR), env={**os.environ},
        )
    except subprocess.TimeoutExpired:
        return "[run_cli] 命令执行超时（>30s）"
    if proc.returncode != 0:
        return f"[run_cli] 命令失败（code={proc.returncode}）：{proc.stderr[-500:]}"
    return proc.stdout


DANGEROUS_PATTERNS = [
    r"\brm\b", r"\bdel\b", r"\brmdir\b", r"\bdeltree\b",
    r"\bformat\b", r"\bmkfs\b", r"\bdd\b",
    r"\bshutdown\b", r"\breboot\b", r"\bpoweroff\b",
    r"[>;]\s*(?:rm|del|format)\b",
    r"\bcurl\b.*\|\s*sh",
    r"\bwget\b.*\|\s*sh",
    r"\bsudo\b", r"\bchmod\b.*-R", r"\bchown\b.*-R",
    r"\bnc\b", r"\bnetcat\b",
    r"/etc/passwd", r"/etc/shadow",
    r"\bTaskkill\b", r"\bStop-Process\b",
]

ALLOWED_HEADS = {"fincli", "python", "python3", "py", "git", "ls", "dir", "cat", "echo", "type"}


def sandbox_check(command: str) -> str | None:
    for pat in DANGEROUS_PATTERNS:
        if re.search(pat, command, re.IGNORECASE):
            return f"沙箱拦截：命中危险模式 {pat!r}"
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return "沙箱拦截：命令解析失败"
    if not tokens:
        return "沙箱拦截：空命令"
    head = Path(tokens[0]).name.lower()
    if head not in ALLOWED_HEADS:
        return f"沙箱拦截：{tokens[0]!r} 不在白名单 {sorted(ALLOWED_HEADS)} 中"
    return None


def run_bash(command: str) -> str:
    blocked = sandbox_check(command)
    if blocked:
        return f"[run_bash] {blocked}"

    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=15,
            cwd=str(BASE_DIR), env={**os.environ},
        )
    except subprocess.TimeoutExpired:
        return "[run_bash] 命令执行超时（>15s）"
    out = proc.stdout
    if proc.returncode != 0:
        out += f"\n[run_bash] 退出码 {proc.returncode}，stderr：{proc.stderr[-300:]}"
    return out


NAMED_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_cli",
            "description": (
                "执行预批准的命令行工具。command 只能取白名单内的值。"
                "查天气必须先 geocode 再 weather_by_coords。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": list(NAMED_COMMANDS.keys()),
                        "description": (
                            "geocode（城市→坐标，需 city）/"
                            " weather_by_coords（坐标→天气，需 latitude+longitude，可选 location_name）"
                        ),
                    },
                    "args": {
                        "type": "object",
                        "description": (
                            "命令参数。geocode: {city}；"
                            "weather_by_coords: {latitude, longitude, location_name?}"
                        ),
                    },
                },
                "required": ["command"],
            },
        },
    },
]

BASH_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": (
                "在沙箱里执行一条 shell 命令并返回 stdout。"
                "查天气分两步："
                "fincli geocode --city 宁德；"
                "再 fincli weather-by-coords --latitude 26.67 --longitude 119.52 --location-name 宁德。"
                "危险命令会被拦截；只允许白名单可执行文件。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "完整的 shell 命令字符串"},
                },
                "required": ["command"],
            },
        },
    },
]

MODE_DISPATCH = {
    "named": (NAMED_TOOLS_SCHEMA, lambda args: run_named(args["command"], args.get("args", {}))),
    "bash": (BASH_TOOLS_SCHEMA, lambda args: run_bash(args["command"])),
}

MAX_TOOL_ROUNDS = 5

SYSTEM_PROMPT_NAMED = (
    "你是一名天气助手。通过 run_cli 工具调用预批准命令查天气。"
    "查天气必须分两步：先 run_cli(command='geocode', args={city}) 拿坐标，"
    "再 run_cli(command='weather_by_coords', args={latitude, longitude, location_name?})；"
    "不要臆造坐标或天气数据，只依据命令返回结果作答。"
    "若某命令依赖另一步结果，请先调前者，等结果回填后再调后者。"
)

SYSTEM_PROMPT_BASH = (
    "你是一名天气助手。通过 run_bash 工具在沙箱里执行 fincli 命令查天气。"
    "查天气分两步：先 fincli geocode --city 宁德，"
    "再根据返回的 latitude/longitude 执行 "
    "fincli weather-by-coords --latitude ... --longitude ... --location-name ..."
    "不要臆造坐标或天气数据。若某命令依赖另一步结果，请先调前者，等结果回填后再调后者。"
)


def run(client, model: str, question: str, mode: str, verbose: bool = True) -> dict:
    tools_schema, executor = MODE_DISPATCH[mode]
    sys_prompt = SYSTEM_PROMPT_NAMED if mode == "named" else SYSTEM_PROMPT_BASH

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []

    for round_idx in range(1, MAX_TOOL_ROUNDS + 1):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=tools_schema, tool_choice="auto",
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            break

        if verbose:
            print(f"  —— 工具轮次 {round_idx} ——")
        messages.append(msg)

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": tc.function.name, "args": args, "round": round_idx})
            if verbose:
                print(f"  → [{mode}] {tc.function.name}({args})")
            try:
                result = executor(args)
            except Exception as e:
                result = f"[{mode}] 执行异常：{e}"
            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}\n")
            messages.append({
                "role": "tool", "tool_call_id": tc.id, "content": result,
            })
    else:
        if verbose:
            print(f"  ⚠ 已达工具轮次上限 {MAX_TOOL_ROUNDS}，强制生成最终回答")
        resp = client.chat.completions.create(model=model, messages=messages)
        msg = resp.choices[0].message

    answer = msg.content or ""
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 最终回答（{elapsed:.1f}s，工具轮次日志 {len(tool_call_log)} 次调用）")
    return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed}


DEMO_QUESTIONS = [
    "宁德现在天气怎么样？",
    "北京今天天气如何？另外未来三天呢？",
    "对比一下上海和深圳现在的天气。",
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="方式三：CLI")
    parser.add_argument("--mode", default="named", choices=["named", "bash"])
    parser.add_argument("--question", "-q")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    client, model = build_client(args.provider)
    if not args.json:
        print(f"[CLI/{args.mode}] provider={args.provider} model={model}\n", file=sys.stderr)

    questions = DEMO_QUESTIONS if args.demo else ([args.question] if args.question else [DEMO_QUESTIONS[0]])
    results = []
    for i, q in enumerate(questions, 1):
        if not args.json:
            print("=" * 60)
            print(f"Q{i}：{q}")
            print("=" * 60)
        result = run(client, model, q, args.mode, verbose=not (args.quiet or args.json))
        result["question"] = q
        result["mode"] = args.mode
        results.append(result)
        if not args.json:
            print("\n最终回答：")
            print(result["answer"])
            print()

    if args.json:
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


if __name__ == "__main__":
    main()
