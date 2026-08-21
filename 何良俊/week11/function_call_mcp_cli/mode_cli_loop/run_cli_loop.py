"""
run_cli_loop.py — 方式三（循环版）：CLI 多轮 ReAct 循环（named + bash 两形态）

教学重点：
  1. 单轮闭环的局限：run_cli.py 只允许"模型调一次命令 → 看结果 → 出答案"，
     一旦模型想"先查年报、再根据结果决定要不要换个查询再查一次"，就做不到——
     ARCHITECTURE.md 第 99 行点名了这是 Agent 多轮循环要解决的问题。
  2. 循环调用的核心：把 create → tool_calls → 执行 CLI → 回填 → 再 create 包成 while 循环，
     每轮模型自己决定"继续调命令"还是"给出最终答案"（无 tool_calls 即退出循环）。
  3. CLI 特性全保留：形态 A 白名单 enum + 形态 B 沙箱（黑名单正则 + 命令头白名单 + 超时 + cwd 锁定），
     只把"协议层"从单轮改成多轮，命令实现层零改动。
  4. 安全护栏：MAX_ITER 防止死循环——Agent 循环必备兜底。

与原版 run_cli.run() 的唯一差异（仅 protocol 层）：
  · run_cli.run()         ：if msg.tool_calls → 执行 → 再 create 一次 → 结束
  · 本文件 run()          ：while msg.tool_calls → 执行 → 再 create → 直到无 tool_calls

使用方式：
  # 先把 fincli 装成 PATH 上的真实命令（一次即可）
  pip install -e .

  # 形态 A（具名，默认）
  python mode_cli_loop/run_cli_loop.py --mode named --question "宁德时代总部宁德的天气如何？如果下雨，推荐一个南方城市并查它的天气"
  # 形态 B（通用 bash）
  python mode_cli_loop/run_cli_loop.py --mode bash --demo

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
CLI_DIR = BASE_DIR / "mode_cli" / "cli"
PY = sys.executable

# fincli 真实命令路径：优先用 pip install -e . 注册到 PATH 的 fincli；
# 没装就退回 python mode_cli/cli/main.py（保证不安装也能跑）
_FINCLI = shutil.which("fincli") or None
FINCLI_ARGV = ["fincli"] if _FINCLI else [PY, str(CLI_DIR / "main.py")]
FINCLI_LABEL = "fincli" if _FINCLI else "python mode_cli/cli/main.py"

# ── LLM 配置（与 run_cli.py 一致）────────────────────────────────────────

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


# ── 形态 A：具名 run_cli（白名单 enum，与 run_cli.py 一致）─────────────────

NAMED_COMMANDS = {
    "rag_search": {
        "argv": FINCLI_ARGV + ["search"],
        "arg_map": {
            "query": "--query",
            "stock_code": "--stock-code",
            "year": "--year",
            "top_k": "--top-k",
        },
    },
    "rag_list_companies": {
        "argv": FINCLI_ARGV + ["list-companies"],
        "arg_map": {},
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
        )
    except subprocess.TimeoutExpired:
        return "[run_cli] 命令执行超时（>30s）"
    if proc.returncode != 0:
        return f"[run_cli] 命令失败（code={proc.returncode}）：{proc.stderr[-500:]}"
    return proc.stdout


# ── 形态 B：通用 run_bash（沙箱，与 run_cli.py 一致）──────────────────────

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
        )
    except subprocess.TimeoutExpired:
        return "[run_bash] 命令执行超时（>15s）"
    out = proc.stdout
    if proc.returncode != 0:
        out += f"\n[run_bash] 退出码 {proc.returncode}，stderr：{proc.stderr[-300:]}"
    return out


# ── 两种形态各自的 tools schema（与 run_cli.py 一致）─────────────────────

NAMED_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_cli",
            "description": (
                "执行预批准的命令行工具。command 只能取白名单内的值。"
                "可查 A 股年报（rag_search/list_companies）和天气（weather）。"
                "你可以分多轮调用：先查一个城市/公司，根据结果再决定是否继续查询。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": list(NAMED_COMMANDS.keys()),
                        "description": "rag_search（查年报，需 query+可选 stock_code/year/top_k）/"
                                       " rag_list_companies（列公司）/"
                                       " weather（查天气，需 city）",
                    },
                    "args": {
                        "type": "object",
                        "description": "命令参数。rag_search: {query, stock_code?, year?, top_k?}; weather: {city}",
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
                "fincli search --query '营收和净利润' --stock-code 300750 --year 2023 --top-k 3；"
                "fincli list-companies；"
                "fincli weather --city 宁德。"
                "危险命令（rm/del/format/sudo/curl|sh 等）会被拦截；只允许白名单可执行文件。"
                "你可以分多轮调用：先查一个城市/公司，根据结果再决定是否继续查询。"
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


# ── 多轮循环 ───────────────────────────────────────────────────────────────

MAX_ITER = 10

SYSTEM_PROMPT_NAMED = (
    "你是一名金融分析助手，具备多轮工具调用能力。通过 run_cli 工具调用预批准命令查 A 股年报与天气。"
    "回答年报问题前必须先 run_cli(command='rag_search', args={...}) 检索原文，只依据返回段落作答，不要编造。"
    "知识库仅含：贵州茅台(600519)/五粮液(000858)/宁德时代(300750)/海康威视(002415)/中国平安(601318)，年份 2021-2023。"
    "rag_search 的 query 不要含公司名/年份（已由 stock_code/year 过滤），用简短术语如 '营收和净利润'。"
    "不在库内的公司请明确告知，不要臆测。"
    "你可以分多轮调用工具：先查一个城市/公司，根据结果再决定是否继续查询其他城市/公司。"
    "一旦信息足够回答用户问题，请直接给出最终答案，不要再调用工具。"
)

SYSTEM_PROMPT_BASH = (
    "你是一名金融分析助手，具备多轮工具调用能力。通过 run_bash 工具在沙箱里执行 fincli 命令查 A 股年报与天气。"
    "查年报：fincli search --query '营收和净利润' --stock-code 300750 --year 2023 --top-k 3"
    "（query 不要含公司名/年份，用简短财务术语）。"
    "列公司：fincli list-companies。"
    "查天气：fincli weather --city 南京。"
    "回答必须依据命令返回的原文，不要编造。知识库仅含 5 家公司（茅台/五粮液/宁德时代/海康威视/中国平安），"
    "不在库内的明确告知。"
    "你可以分多轮调用工具：先查一个城市/公司，根据结果再决定是否继续查询其他城市/公司。"
    "一旦信息足够回答用户问题，请直接给出最终答案，不要再调用工具。"
)


def _execute_tool_call(tc, executor, mode: str, verbose: bool, iteration: int) -> tuple[str, dict]:
    """执行单个 tool_call，返回 (结果字符串, 日志条目)。"""
    args = json.loads(tc.function.arguments or "{}")
    log_entry = {"name": tc.function.name, "args": args, "round": iteration}
    if verbose:
        print(f"  → [round {iteration}] [{mode}] {tc.function.name}({args})")
    try:
        result = executor(args)
    except Exception as e:
        result = f"[{mode}] 执行异常：{e}"
    preview = (result or "")[:120].replace("\n", " ")
    if verbose:
        print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}\n")
    return result, log_entry


def run(client, model: str, question: str, mode: str, verbose: bool = True) -> dict:
    """
    多轮循环：提问 → 循环{ 模型输出 tool_call → 执行 CLI → 回填 → 再请求 } → 最终回答。
    与 run_cli.run() 的唯一差别：把"if msg.tool_calls"改成"while msg.tool_calls"。
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
    msg = None

    # 【教学时刻 1】：while 循环是本文件的核心——模型不再只能调一次命令
    while iteration < MAX_ITER:
        iteration += 1
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=tools_schema, tool_choice="auto",
        )
        msg = resp.choices[0].message

        # 模型本轮没有 tool_calls = 已生成最终答案，退出循环
        if not msg.tool_calls:
            if verbose:
                print(f"  → [round {iteration}] 模型未调用工具，直接作答\n")
            break

        # 有 tool_calls = 执行后回填，进入下一轮让模型决定是否继续
        messages.append(msg)
        for tc in msg.tool_calls:
            result, log_entry = _execute_tool_call(tc, executor, mode, verbose, iteration)
            tool_call_log.append(log_entry)
            messages.append({
                "role": "tool", "tool_call_id": tc.id, "content": result,
            })
        # 循环回到 while 顶部：再次请求模型，它可以继续调工具或给出答案
    else:
        # while 正常走完 MAX_ITER 次仍没退出——强制再请求一次拿最终答案
        if verbose:
            print(f"  → [!] 达到 MAX_ITER={MAX_ITER}，强制收尾\n")
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=tools_schema, tool_choice="none",
        )
        msg = resp.choices[0].message

    answer = msg.content or ""
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 最终回答（{elapsed:.1f}s，共 {iteration} 轮，{len(tool_call_log)} 次工具调用）")
    return {
        "answer": answer,
        "tool_calls": tool_call_log,
        "elapsed": elapsed,
        "iterations": iteration,
        "mode": mode,
    }


# ── 入口 ───────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "宁德时代总部宁德的天气如何？如果下雨，推荐一个南方城市并查它的天气。",
    "帮我依次查北京、上海、广州的天气，并告诉我哪座城市最适合明天出门。",
    "宁德时代的总部在哪？查一下总部所在地的天气。",
    "对比贵州茅台和五粮液2023年的营收，如果数据不够清楚可以再查一次。",
]


def main():
    global MAX_ITER
    import argparse
    parser = argparse.ArgumentParser(description="方式三（循环版）：CLI 多轮 ReAct")
    parser.add_argument("--mode", default="named", choices=["named", "bash"])
    parser.add_argument("--question", "-q")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--max-iter", type=int, default=MAX_ITER, help=f"循环上限（默认 {MAX_ITER}）")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    MAX_ITER = args.max_iter

    client, model = build_client(args.provider)
    if not args.json:
        print(f"[CLI Loop/{args.mode}] provider={args.provider} model={model} max_iter={MAX_ITER}\n", file=sys.stderr)

    questions = DEMO_QUESTIONS if args.demo else ([args.question] if args.question else [DEMO_QUESTIONS[0]])
    results = []
    for i, q in enumerate(questions, 1):
        if not args.json:
            print("=" * 60)
            print(f"Q{i}：{q}")
            print("=" * 60)
        result = run(client, model, q, args.mode, verbose=not (args.quiet or args.json))
        result["question"] = q
        results.append(result)
        if not args.json:
            print("\n最终回答：")
            print(result["answer"])
            print()

    if args.json:
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


if __name__ == "__main__":
    main()
