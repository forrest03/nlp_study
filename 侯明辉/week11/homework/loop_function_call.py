"""
loop_function_call.py — 循环工具调用版本（基于 week11 单轮实现改造）

教学重点（对比原 run_function_call.py）：
  原实现：单轮闭环 = 1 次模型调用 → 1 次工具执行 → 1 次模型生成回答（共 2 次 LLM 调用）。
  本实现：循环闭环 = 模型在「得出最终文本回答」之前可以连续多轮调用工具，
          直到模型输出 content 且不再输出 tool_calls 为止。
          这是 ReAct / Agent Loop 的基本形态 —— 模型自主规划“需要查几次工具”。

改造点：
  1. 把外层「请求 → 回填 → 再请求」的结构换成 while 循环
  2. 终止条件：msg.tool_calls 为空（模型给出了文本回答）或达到 max_iterations
  3. 新增 step_log：记录每轮模型决策、工具执行与最终回答，方便观察循环过程
  4. 安全护栏：MAX_ITERATIONS 默认 8，防止模型陷入死循环

使用方式：
  # 配置环境变量
  #   Windows:  set DEEPSEEK_API_KEY=sk-xxx
  #   Linux:    export DEEPSEEK_API_KEY=sk-xxx

  # 单个问题
  python loop_function_call.py --question "对比北京和上海今天的天气，哪个更适合出行？"

  # 内置示例问题集（演示多轮工具调用）
  python loop_function_call.py --demo

依赖：
  pip install openai httpx
"""

import argparse
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from openai import OpenAI

# 把项目根目录加入 sys.path，让 src 可 import
sys.path.insert(0, str(Path(__file__).parent))

from src.weather_backend import get_weather  # noqa: E402

# ── LLM 配置 ───────────────────────────────────────────────────────────────

PROVIDERS = {
    "deepseek": {
        # 环境变量名：DEEPSEEK_API_KEY
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "dashscope": {
        # 环境变量名：DASHSCOPE_API_KEY（阿里云百炼）
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
    "siliconflow": {
        # 环境变量名：SILICONFLOW_API_KEY（硅基流动 siliconflow.cn）
        # 兼容写法：SCY/SILICONFLOW_API_KEY 任意一个非空即可
        "api_key": (
            os.environ.get("SILICONFLOW_API_KEY", "")
            or os.environ.get("SCY_API_KEY", "")
        ),
        "base_url": "https://api.siliconflow.cn/v1",
        # 默认 14B 原版——7B 在硅基流动上对 tools schema 支持不稳，详见 README 「量化模型」一节
        # 也可手动改成 32B / 72B：更强的 function calling 与推理能力
        "model": "Qwen/Qwen2.5-14B-Instruct",
    },
}

MAX_ITERATIONS = 8  # 循环护栏：超过这个步数未出文本即熔断


def build_client(provider: str):
    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        hint = {
            "deepseek": "DEEPSEEK_API_KEY",
            "dashscope": "DASHSCOPE_API_KEY",
            "siliconflow": "SILICONFLOW_API_KEY（兼容 SCY_API_KEY）",
        }.get(provider, f"{provider.upper()}_API_KEY")
        print(f"错误：未设置 {hint}", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


# ── 工具 schema + dispatch ──────────────────────────────────────────────────

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "查询指定城市的当前天气及未来3天预报。城市用中文名，如 '宁德'、'北京'。"
                "若用户问「哪个城市更适合出行」之类的比较问题，请对涉及的每个城市各调一次。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市中文名，如 '宁德'、'北京'",
                    },
                },
                "required": ["city"],
            },
        },
    },
]

TOOL_DISPATCH = {
    "get_weather": get_weather,
}


def execute_tool(name: str, args: dict) -> str:
    """执行工具并以字符串形式返回结果；捕获异常转成可读字符串喂给 LLM。"""
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


# ── 【防御补丁】文本式工具调用解析 ──────────────────────────────────────────
# 当某些量化/小模型不认 OpenAI 兼容的 tools schema 时，会把 JSON 工具调用塞进 content
# （常见形态：{"name": "get_weather", "arguments": {...}}  或 {"function": "..."}）。
# 我们把这种「半成品」文本兜底解析成结构化 tool_calls，让后续循环照常运行。

def parse_text_tool_calls(content: str):
    """
    从模型返回的文本 content 里抽取工具调用 JSON。
    返回 [SimpleNamespace(id, function.name, function.arguments), ...]，无则返回 []。
    """
    if not content:
        return []

    # 去掉 markdown 代码围栏
    text = re.sub(r"```(?:json)?", "", content).strip()

    # 在文本里逐个位置尝试 JSON.decode，找到第一个含 name/function 的对象就收
    decoder = json.JSONDecoder()
    found = []
    pos = 0
    seen_keys = set()
    while pos < len(text):
        # 跳过空白
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
        # 跳过 pos（已跳过），把游标推进到 end
        pos = end
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("function")
        if not isinstance(name, str) or name not in TOOL_DISPATCH:
            continue
        # 防重：同一文本里同一个工具只取一次
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
            function=SimpleNamespace(
                name=name,
                arguments=json.dumps(args, ensure_ascii=False),
            ),
        ))
    return found


# ── 主循环：循环调用版本的核心 ────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一名出行建议助手。当用户询问天气时必须先调用 get_weather 工具获取实时数据，"
    "再依据工具返回内容作答。如果问题涉及多个城市（如对比、排序、推荐），"
    "请对每个城市都各调用一次工具——可以一次性并行发起多个 tool_calls。"
    "在拿到所有需要的天气数据并给出明确的文本回答之前，不要结束本轮。"
)


def run_loop(client, model: str, question: str, verbose: bool = True) -> dict:
    """
    循环工具调用：
        循环模型调用，直到给出文本回答（tool_calls 为空）或达到 max_iterations。
    返回 {answer, steps, rounds, elapsed}：
        - answer: 最终文本回答
        - steps:  每一步 {role, tool_calls?, content?, preview?} 的详细轨迹
        - rounds: 实际循环轮数
        - elapsed: 耗时
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    steps = []
    rounds = 0
    final_answer = ""

    # ── 关键差异点：这里是 while 循环，不是「调用两次就完」 ────────────────
    while rounds < MAX_ITERATIONS:
        rounds += 1
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        # ── [text-fallback] 模型没走原生 tool_calls，但 content 是 JSON 工具调用 ──
        # 部分量化模型会把 `{"name":"get_weather","arguments":{...}}` 塞进文本。
        # 我们解析为合成 tool_calls，让循环继续走工具分支，避免「单轮假结束」。
        synthetic_used = False
        tool_calls_to_exec = msg.tool_calls
        if not tool_calls_to_exec and msg.content:
            text_calls = parse_text_tool_calls(msg.content)
            if text_calls:
                tool_calls_to_exec = text_calls
                synthetic_used = True
                if verbose:
                    print(f"  ↪ [text-fallback] 从 content 解析出 {len(text_calls)} 个工具调用")

        # ── 终止条件：模型没有继续调用工具，意味着它给出了文本回答 ────────
        if not tool_calls_to_exec:
            final_answer = msg.content or ""
            steps.append({
                "round": rounds,
                "role": "assistant_text",
                "content": final_answer,
                "fallback": False,
            })
            if verbose:
                print(f"  → [llm round {rounds}] 模型在第 {rounds} 轮给出文本回答 → 循环结束")
            break

        # ── 接下来要执行工具：把这一轮 assistant 消息以正确协议形状回填 ────
        if not synthetic_used:
            # 原生 tool_calls：OpenAI 协议要求整条消息原样回填
            messages.append(msg)
        else:
            # 文本补丁：用合成 tool_calls 构造一条「修正后」的 assistant 消息
            # 这样下游协议（tool_call_id 对应）能继续工作
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments}}
                    for tc in tool_calls_to_exec
                ],
            })

        for tc in tool_calls_to_exec:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if verbose:
                tag = "text" if synthetic_used else "tool"
                print(f"  → [round {rounds} {tag}] {name}({args})")
            result = execute_tool(name, args)
            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}")
            steps.append({
                "round": rounds,
                "role": "tool",
                "name": name,
                "args": args,
                "result_preview": preview,
                "source": "text_fallback" if synthetic_used else "native",
            })
            # role=tool 回填，tool_call_id 必须对上
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        if verbose:
            print(f"  ↻ round {rounds} 工具已执行完毕，继续下一轮...\n")

    else:
        # while 循环「正常走完」（没被 break），意味着触发了熔断
        if verbose:
            print(f"  ⚠ 达到 max_iterations={MAX_ITERATIONS}，强制终止循环")
        # 取最后一轮 assistant 的 content（即使为空，作为兜底）
        final_answer = msg.content if msg and msg.content else "（循环被熔断，未获得文本回答）"

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  → [done] 共 {rounds} 轮，耗时 {elapsed:.1f}s")
    return {"answer": final_answer, "steps": steps, "rounds": rounds, "elapsed": elapsed}


# ── 入口 ───────────────────────────────────────────────────────────────────

# 故意挑选「必须并行 / 多轮调用工具」才能完整回答的问题，对比单轮时会被截断或被迫一次性说出
DEMO_QUESTIONS = [
    # 单工具即可
    "北京今天的天气怎么样？",
    # 并行多工具：模型一次发起两个 tool_calls
    "对比北京和上海今天的天气，哪个更适合户外活动？",
    # 串行多工具：模型需要先看天气，再综合回答
    "上海会下雨吗？如果会，未来3天哪天最适合出行？请说明原因。",
]


def main():
    parser = argparse.ArgumentParser(description="循环工具调用（多轮 Function Call）")
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置示例问题集")
    parser.add_argument(
        "--provider",
        default="deepseek",
        choices=list(PROVIDERS.keys()),
        help="deepseek | dashscope | siliconflow（硅基流动，env: SILICONFLOW_API_KEY）",
    )
    parser.add_argument("--quiet", action="store_true", help="少输出（被脚本调用时用）")
    parser.add_argument("--json", action="store_true", help="输出 JSON（供脚本解析）")
    args = parser.parse_args()

    client, model = build_client(args.provider)
    if not args.json:
        print(f"[Loop Function Call] provider={args.provider} model={model}\n")

    questions = DEMO_QUESTIONS if args.demo else (
        [args.question] if args.question else [DEMO_QUESTIONS[0]]
    )
    results = []
    for i, q in enumerate(questions, 1):
        if not args.json:
            print("=" * 60)
            print(f"Q{i}：{q}")
            print("=" * 60)
        result = run_loop(client, model, q, verbose=not (args.quiet or args.json))
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
