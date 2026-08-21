"""
run_cli.py — 方式三：CLI（命令行即工具），支持链式调用

教学重点：
  1. 形态 A（具名 run_cli）：LLM 调一个 run_cli(command, args) 工具，command 是白名单 enum，
     host 拼出子命令执行。安全可控，但每加一个命令要改代码
  2. 形态 B（通用 run_bash）：LLM 自己拼完整 shell 命令，host 在沙箱里执行。
     最灵活、最危险——教学重点是沙箱设计（白名单/黑名单/超时/工作目录锁定）
  3. 链式调用：while 循环控制，模型看到工具结果后继续决策

使用方式：
  # 先把 fincli 装成 PATH 上的真实命令（一次即可）
  pip install -e .

  # 形态 A（具名，默认）
  python mode_cli/run_cli.py --mode named --question "北京天气怎么样"
  # 形态 B（通用 bash）
  python mode_cli/run_cli.py --mode bash --question "北京天气怎么样"
  # 内置示例
  python mode_cli/run_cli.py --mode named --demo

依赖：
  pip install openai
  环境变量：DEEPSEEK_API_KEY（默认 LLM）
            DASHSCOPE_API_KEY（Embedding，fincli 内部用）
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

# fincli 真实命令路径
_FINCLI = shutil.which("fincli") or None
FINCLI_ARGV = ["fincli"] if _FINCLI else [PY, str(CLI_DIR / "main.py")]
FINCLI_LABEL = "fincli" if _FINCLI else "python mode_cli/cli/main.py"

# ── LLM 配置 ───────────────────────────────────────────────────────────────

PROVIDERS = {
    "deepseek": {
        "api_key": "sk-bbad0654d3374b13903515bb74b04600",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "dashscope": {
        "api_key": "sk-ws-H.EMIILEM.re1g.MEUCIHySxOkNu8AKveLvomkwOC5Ri212gohBlrjbXYJzEu33AiEA-_Aw7BSJwouLSuRPj6mVH6--kdcQFzid9O1_hAH7m_w",
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


# ── 形态 A：具名 run_cli ───────────────────────────────────────────────────

NAMED_COMMANDS = {
    "region": {
        "argv": FINCLI_ARGV + ["region"],
        "arg_map": {"city": "--city"},
    },
    "weather": {
        "argv": FINCLI_ARGV + ["weather"],
        "arg_map": {"city": "--city"},
    },
}


def run_named(command: str, args: dict) -> str:
    """形态 A：按白名单拼出 argv，子进程执行，返回 stdout。"""
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
            #encoding='utf-8',  # 强制 UTF-8，避免 Windows GBK 解码错误
        )
    except subprocess.TimeoutExpired:
        return "[run_cli] 命令执行超时（>30s）"
    if proc.returncode != 0:
        return f"[run_cli] 命令失败（code={proc.returncode}）：{proc.stderr[-500:]}"
    return proc.stdout


# ── 形态 B：通用 run_bash（沙箱）──────────────────────────────────────────

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
    """返回 None 表示通过；返回字符串表示拒绝原因。"""
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
    """形态 B：模型生成的 shell 命令，经沙箱检查后在锁定工作目录执行。"""
    blocked = sandbox_check(command)
    if blocked:
        return f"[run_bash] {blocked}"

    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=15,
            cwd=str(BASE_DIR), env={**os.environ},
            #encoding='utf-8',  # 强制 UTF-8
        )
    except subprocess.TimeoutExpired:
        return "[run_bash] 命令执行超时（>15s）"
    out = proc.stdout
    if proc.returncode != 0:
        out += f"\n[run_bash] 退出码 {proc.returncode}，stderr：{proc.stderr[-300:]}"
    return out


# ── 两种形态各自的 tools schema ───────────────────────────────────────────

NAMED_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_cli",
            "description": (
                "执行预批准的命令行工具。command 只能取白名单内的值。"
                "可查区域经纬度（region）和天气（weather）。"
                "如果用户问天气，需要先调用 region 获取区域信息，再调用 weather。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": list(NAMED_COMMANDS.keys()),
                        "description": "region（查经纬度，需 city）/ weather（查天气，需 city）",
                    },
                    "args": {
                        "type": "object",
                        "description": "命令参数。region: {city: '北京'}; weather: {city: '北京'}",
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
                "可用工具 fincli（一条真实命令）："
                "fincli region --city 北京"
                "fincli weather --city 南京"
                "危险命令会被拦截。"
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

# ── 链式调用 ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT_NAMED = (
    "你是一个专业问答助手，通过 run_cli 工具调用预批准命令查地址与天气。"
    "涉及城市时调用 run_cli(command='region', args={city: '城市名'})。"
    "涉及天气时调用 run_cli(command='weather', args={city: '城市名'})。"
    "如果需要先查区域再查天气，请分两步调用工具。"
    "你可以一次调用多个工具。"
)

SYSTEM_PROMPT_BASH = (
    "你是一个专业问答助手，通过 run_bash 工具在沙箱里执行 fincli 命令。"
    "查区域：fincli region --city 北京"
    "查天气：fincli weather --city 南京"
    "回答必须依据命令返回的原文，不要编造。"
)


def run(client, model: str, question: str, mode: str, verbose: bool = True, max_iterations: int = 5) -> dict:
    """
    链式调用：while 循环控制，直到没有 tool_call
    伪代码：
        resp = model.chat(msgs, tools)
        while resp.has_tool_call():
            result = run(resp.tool_call)
            msgs = append(result)
            resp = model.chat(msgs, tools)
    """
    tools_schema, executor = MODE_DISPATCH[mode]
    sys_prompt = SYSTEM_PROMPT_NAMED if mode == "named" else SYSTEM_PROMPT_BASH

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []
    iteration = 0

    # ====== 第一次请求：resp = model.chat(msgs, tools) ======
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools_schema,
        tool_choice="auto",
    )
    msg = resp.choices[0].message

    # ====== while resp.has_tool_call(): ======
    while msg.tool_calls and iteration < max_iterations:
        iteration += 1

        # ====== msgs = append(msg) ======
        messages.append(msg)

        # ====== result = run(resp.tool_call) ======
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": tc.function.name, "args": args, "iteration": iteration})

            if verbose:
                print(f"  → [{mode}] {tc.function.name}({args})")

            try:
                result = executor(args)
            except Exception as e:
                result = f"[{mode}] 执行异常：{e}"

            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}\n")

            # ====== msgs = append(result) ======
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        # ====== resp = model.chat(msgs, tools) ======
        # 关键！重新调用模型，让模型看到工具结果后继续决策
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools_schema,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

    # ====== 最终回答 ======
    elapsed = time.time() - t0

    # 如果达到最大迭代且还有 tool_call，强制结束
    if iteration >= max_iterations and msg.tool_calls:
        if verbose:
            print(f"  ⚠️ 达到最大迭代次数 {max_iterations}，强制结束")
        messages.append(msg)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools_schema,
            tool_choice="none",
        )
        msg = resp.choices[0].message

    answer = msg.content or ""

    if verbose:
        print(f"  → [llm] 最终回答（{elapsed:.1f}s，{iteration}轮）")

    return {
        "answer": answer,
        "tool_calls": tool_call_log,
        "elapsed": elapsed,
        "iterations": iteration,
    }


# ── 入口 ───────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "北京的经纬度是多少？",
    "南京的经纬度是多少？",
    "北京今天的天气怎么样？",
    "南京明天的天气怎么样？",
]

# 形态 → (schema, executor)
MODE_DISPATCH = {
    "named": (NAMED_TOOLS_SCHEMA, lambda args: run_named(args["command"], args.get("args", {}))),
    "bash": (BASH_TOOLS_SCHEMA, lambda args: run_bash(args["command"])),
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="方式三：CLI（链式调用）")
    parser.add_argument("--mode", default="named", choices=["named", "bash"])
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置示例问题集")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true", help="少输出")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--max-iterations", type=int, default=5, help="最大迭代次数")
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
        result = run(
            client, model, q, args.mode,
            verbose=not (args.quiet or args.json),
            max_iterations=args.max_iterations,
        )
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