"""
对话 Harness 主模块

流程：
  1. 新用户请求 → BM25+向量融合检索长期记忆 Top-3 → 整理为提示词
  2. 加载用户特征记忆、当日记忆、短期会话历史
  3. 调用 DashScope LLM 生成回复
  4. 保存短期/长期/按日记忆
  5. 定期检测并压缩长期记忆

工具调用模式（--allow-tools）：
  - LLM 可返回 function call 执行目录创建、文件读写、shell 命令
  - 结果自动回传给 LLM 继续推理，直至生成最终回复
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 将项目根目录加入 path，便于从 harness/ 子目录运行
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.llm_client import chat, chat_stream, chat_with_tools, DEFAULT_MODEL
from harness.tool_executor import get_all_tools, get_skill_descriptions, ToolExecutor
from src.memory_store import get_memory_store
from src.memory_retriever import get_memory_retriever, reload_retriever
from src.memory_compressor import run_compression_if_needed, compress_raw_memories, rebuild_vector_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_TEMPLATE = """你是一个有记忆能力的 AI 助手。请结合以下记忆信息回答用户问题。

{long_term_section}

{user_profile_section}

{daily_section}

规则：
1. 优先使用记忆中的事实，不要编造未提及的信息
2. 若记忆不足以回答，请诚实说明
3. 回答简洁清晰，使用中文
"""

TOOLS_SYSTEM_PROMPT = """

你还可以使用以下工具来完成用户请求：
- create_directory：创建目录
- write_file：写文件
- read_file：读文件
- execute_command：执行 shell 命令

{skill_prompts}

当用户要求创建项目、生成文件或执行命令时，请使用这些工具。
如果需要用户确认，请先说明你的计划，等待用户确认后再执行。
"""


def _build_system_prompt(query: str, session_id: str) -> str:
    store = get_memory_store()
    retriever = get_memory_retriever()

    long_term = retriever.build_memory_prompt(query, top_k=3)
    long_term_section = long_term if long_term else "【相关长期记忆】暂无匹配记录"

    profile = store.read_user_profile().strip()
    user_profile_section = f"【用户特征记忆】\n{profile}" if profile else "【用户特征记忆】暂无"

    daily = store.read_daily().strip()
    daily_section = f"【今日记忆 ({__import__('datetime').date.today().isoformat()})】\n{daily}" if daily else "【今日记忆】暂无"

    return SYSTEM_TEMPLATE.format(
        long_term_section=long_term_section,
        user_profile_section=user_profile_section,
        daily_section=daily_section,
    )


def run_chat(
    question: str,
    session_id: Optional[str] = None,
    stream: bool = False,
    model: str = DEFAULT_MODEL,
    allow_tools: bool = False,
    tool_executor: Optional[ToolExecutor] = None,
) -> tuple[str, str]:
    """
    执行一轮对话。

    allow_tools=True 时启用 function calling：
      - LLM 可返回工具调用（创建目录、写文件、读文件、执行命令）
      - 自动执行工具并将结果回传给 LLM
      - 循环直至 LLM 生成文本回复

    Returns:
        (answer, session_id)
    """
    from harness.tool_executor import get_tool_executor

    store = get_memory_store()
    if not session_id:
        session_id = store.create_session()
        logger.info(f"新建会话: {session_id}")

    system_prompt = _build_system_prompt(question, session_id)
    if allow_tools:
        skill_prompts = get_skill_descriptions()
        system_prompt += TOOLS_SYSTEM_PROMPT.format(skill_prompts=skill_prompts)
    history = store.short_term_to_messages(session_id)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    if allow_tools:
        executor = tool_executor or get_tool_executor()
        if not executor.allow_tools:
            executor.allow_tools = True
        answer = _run_tool_loop(messages, executor, model)
    else:
        if stream:
            parts = []
            for chunk in chat_stream(messages, model=model):
                print(chunk, end="", flush=True)
                parts.append(chunk)
            print()
            answer = "".join(parts)
        else:
            answer = chat(messages, model=model)
            print(answer)

    store.add_short_term_turn(session_id, question, answer)

    if run_compression_if_needed(session_id=session_id):
        reload_retriever()
        logger.info("长期记忆已自动压缩")

    return answer, session_id


def _run_tool_loop(
    messages: List[Dict[str, Any]],
    executor: ToolExecutor,
    model: str = DEFAULT_MODEL,
    max_turns: int = 10,
) -> str:
    """工具调用循环：LLM → 工具 → LLM → ... → 文本回复。"""
    tools = get_all_tools()
    for turn in range(max_turns):
        content, tool_calls = chat_with_tools(messages, tools, model=model)
        messages.append({"role": "assistant", "content": content, "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in tool_calls
        ]} if tool_calls else {"role": "assistant", "content": content})

        if not tool_calls:
            return content

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                args = {}
                logger.warning(f"工具 {name} 参数解析失败: {e}")

            logger.info(f"执行工具: {name}({args})")
            result = executor.execute_tool(name, args)
            result_str = json.dumps(result, ensure_ascii=False)
            if len(result_str) > 10000:
                result_str = result_str[:10000] + "\n...(截断)"
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

    return "工具调用次数超过上限，请简化你的请求。"


def interactive_loop(
    session_id: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    allow_tools: bool = False,
):
    """交互式对话循环。"""
    from harness.tool_executor import set_tool_executor

    store = get_memory_store()
    if not session_id:
        session_id = store.create_session()
    print(f"Harness 对话已启动 | 模型: {model} | 会话: {session_id}")
    print(f"工具调用: {'已启用' if allow_tools else '已禁用'}")

    executor = None
    if allow_tools:
        executor = ToolExecutor(
            project_root=str(ROOT),
            confirm_commands=True,
            allow_tools=True,
        )
        set_tool_executor(executor)

    print("命令: exit 退出 | compress 手动压缩长期记忆 | session 显示会话ID | tools 切换工具模式\n")

    while True:
        try:
            q = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not q:
            continue
        if q.lower() == "exit":
            print("再见。")
            break
        if q.lower() == "compress":
            compress_raw_memories()
            reload_retriever()
            print("长期记忆压缩完成。")
            continue
        if q.lower() == "session":
            print(f"当前会话 ID: {session_id}")
            continue
        if q.lower() == "tools":
            allow_tools = not allow_tools
            print(f"工具调用: {'已启用' if allow_tools else '已禁用'}")
            continue

        print("助手: ", end="", flush=True)
        _, session_id = run_chat(
            q,
            session_id=session_id,
            stream=not allow_tools,
            model=model,
            allow_tools=allow_tools,
            tool_executor=executor,
        )


def init_demo_vectorstore():
    """首次启动时，若 databases/ 尚无索引则从 memory_meta 重建。"""
    from src.paths import FAISS_INDEX_FILE, MEMORY_META_FILE
    if MEMORY_META_FILE.exists() and not FAISS_INDEX_FILE.exists():
        rebuild_vector_index()
        reload_retriever()
        logger.info("已根据 databases/memory_meta.json 初始化向量索引")
