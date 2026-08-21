"""
主 agent + 并行 subagent 编排。

主 agent 是 ReAct 循环，有 2 个工具：
  - read_file: 直接读文件
  - dispatch_subagents: 派多个 subagent 并行读不同文件

Subagent 也是 ReAct 循环，有 2 个工具：
  - read_file: 读文件片段
  - list_files: 列目录

并行通过 ThreadPoolExecutor 实现，wall-clock ≈ max(subagent) 而非 sum。

依赖：concurrent.futures, react_loop, tools
"""
import uuid
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from react_loop import ReActLoop
from tools import read_file, list_files, DEFAULT_CORPUS_DIR

logger = logging.getLogger(__name__)

# 模块级常量（避免分散的 magic number）
MAX_SUBAGENTS = 6          # dispatch_subagents 一次性最多派几个子 agent
MAX_OBS_CHARS = 500        # 每个 subagent 答案在汇总里的截短上限
LOG_ACTION_INPUT_WIDTH = 60  # verbose 日志里 action_input 的截短宽度
DURATION_DECIMALS = 2      # 耗时/加速比四舍五入保留几位小数


MAIN_SYSTEM = """你是文档问答主 agent，能读本地课程笔记文件。

可用工具：
- read_file: 读取文件片段，参数格式 "path:start_line:end_line"
  例: "week15.md:1:50" 表示读 week15.md 第 1~50 行
  例: "week15.md" 不带行号读全文
- dispatch_subagents: 派多个子 agent 并行读不同文件，
  参数格式 "子课题1 | 子课题2 | ..."（用 | 分隔，≤6 个）
  每个子 agent 会独立 read_file 并给出该子课题的部分答案

【关键决策原则】
- 问题明确指向 1 个文件或 1 个具体概念 → 自己 read_file
- 问题涉及多周/多主题/跨文件对比 → 必须 dispatch_subagents
  示例：「对比 week10/11/12 的 RAG」→ dispatch_subagents
  示例：「week13 的 skills 是什么」→ 自己 read_file

【输出格式】每轮严格：
Thought: ...
Action: read_file 或 dispatch_subagents
Action Input: 参数
...（多轮）
Thought: 信息充分
Final Answer: 综合答案（中文，分点）"""

SUBAGENT_SYSTEM = f"""你是文档检索子 agent。用工具读取与子课题相关的文件片段，给出该子课题的部分答案。

可用工具：
- read_file: 参数 "path:start_line:end_line"
- list_files: 参数 "." 或 "week10"，列出目录

输出控制在 {MAX_OBS_CHARS} 字内，直接给要点。最后用 Final Answer: 开头。"""


def _dispatch_subagents(action_input: str, corpus_dir: Path,
                         max_subagents: int = MAX_SUBAGENTS) -> tuple[str, dict]:
    """
    dispatch_subagents 工具实现。
    返回 (汇总文本, parallel_stats)。
    """
    subtopics = [s.strip() for s in action_input.split("|") if s.strip()][:max_subagents]
    if not subtopics:
        return "未解析出子课题（需要 | 分隔）", {}

    defs = []
    for topic in subtopics:
        sid = f"sub_{uuid.uuid4().hex[:6]}"
        sub = ReActLoop(
            agent_name=sid,
            tools={
                "read_file": (lambda q: read_file(q, corpus_dir=corpus_dir),
                              "读取文件片段，参数 path:start:end"),
                "list_files": (lambda q: list_files(q, corpus_dir=corpus_dir),
                               "列目录，参数 . 或 weekN"),
            },
            system_prompt=SUBAGENT_SYSTEM,
            max_steps=4,
        )
        defs.append((sid, sub, topic))

    t0 = time.time()
    results = {}
    # O(1) 反查表：避免后续 O(n) 的 next(...) 扫描
    topic_by_sid = {sid: topic for sid, _, topic in defs}

    def _run_one(sid, sub, topic):
        return sid, sub.run(topic)

    with ThreadPoolExecutor(max_workers=len(defs)) as pool:
        futs = {pool.submit(_run_one, sid, sub, topic): sid
                for sid, sub, topic in defs}
        for fut in as_completed(futs):
            sid = futs[fut]
            try:
                sid_r, res = fut.result()
            except Exception as e:
                # 退化：单个 subagent 失败不拖垮整批
                logger.exception("subagent %s 异常", sid)
                results[sid] = (topic_by_sid[sid], {
                    "final_answer": f"[subagent 失败: {type(e).__name__}: {str(e)[:120]}]",
                    "duration": 0.0,
                    "trace": [],
                })
            else:
                results[sid] = (topic_by_sid[sid_r], res)

    wall = round(time.time() - t0, DURATION_DECIMALS)
    serial_sum = round(sum(r["duration"] for _, r in results.values()), DURATION_DECIMALS)
    # wall == 0 时显式 short-circuit，避免 ZeroDivisionError（0.0 是 falsy 但 wall > 0 更精确）
    speedup = round(serial_sum / wall, DURATION_DECIMALS) if wall > 0 else 0.0
    stats = {"n_subagents": len(defs), "wall_clock": wall,
             "serial_sum": serial_sum, "speedup": speedup}

    parts = []
    for sid, _, topic in defs:
        _, r = results[sid]
        parts.append(
            f"【子课题: {topic}】(用时{r['duration']}s)\n"
            f"{r['final_answer'][:MAX_OBS_CHARS]}"
        )
    summary = (f"并行调研完成：{len(defs)} 个子 agent，"
               f"wall-clock {wall}s (串行需 {serial_sum}s，加速 {speedup}×)\n\n"
               + "\n\n".join(parts))
    return summary, stats


def run_qa(question: str, corpus_dir: Path | None = None,
           verbose: bool = True) -> dict:
    """
    执行一次问答。返回 {final_answer, main_trace, duration}。
    """
    corpus_dir = corpus_dir or DEFAULT_CORPUS_DIR

    def dispatch_tool(action_input):
        summary, stats = _dispatch_subagents(action_input, corpus_dir)
        if verbose and stats:
            print(f"  [并行统计] {stats}")
        return summary

    main = ReActLoop(
        agent_name="main",
        tools={
            "read_file": (lambda q: read_file(q, corpus_dir=corpus_dir),
                          "读取文件片段，参数 path:start:end"),
            "dispatch_subagents": (dispatch_tool,
                                   "派多个子 agent 并行读不同文件，参数 课题1 | 课题2 | ..."),
        },
        system_prompt=MAIN_SYSTEM,
        max_steps=8,
    )

    if verbose:
        print(f"\n[主 agent 开始] 问题: {question}")

    def on_step(step):
        if verbose:
            tag = "→ Final" if step.get("final") else f"→ {step['action']}"
            inp = (step.get("action_input") or "")[:LOG_ACTION_INPUT_WIDTH]
            print(f"  [main.{step['idx']}] {tag} {inp!r}")

    result = main.run(question, on_step=on_step)
    return {"final_answer": result["final_answer"],
            "main_trace": result["trace"],
            "duration": result["duration"]}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).parent)
    # 自测：跑示例问题
    r = run_qa("对比 week10、week11、week12 学了什么主题")
    print(f"\n=== 最终答案 ===\n{r['final_answer']}")