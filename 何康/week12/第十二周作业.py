index.html:
.new-chat-btn {
    padding: 0 14px; border: 1px solid #3d4163; border-radius: 8px;
    background: transparent; color: #94a3b8; cursor: pointer; white-space: nowrap;
  }
  .new-chat-btn:hover { border-color: #7c3aed; color: #a78bfa; }
  .round-title { color: #94a3b8; font-size: 13px; margin: 8px 0 0; }

react_function_calling.py:
def run(question: str, max_steps: int = 10) -> Generator[dict, None, None]:
def run(
    question: str,
    max_steps: int = 10,
    chat_history: Sequence[dict[str, str]] | None = None,
) -> Generator[dict, None, None]:
# 历史只保留每轮的最终问答，工具调用和 Observation 不跨轮保存。
    messages = [{"role": "system", "content": FC_SYSTEM_PROMPT}]
    messages.extend(chat_history or [])
    messages.append({"role": "user", "content": question})


react_manual.py
def run(
    question: str,
    max_steps: int = 10,
    verbose: bool = True,
    chat_history: Sequence[dict[str, str]] | None = None,
) -> Generator[dict, None, None]:
  # 历史只包含已经完成的 user / assistant 对话轮次。工具过程仅属于当前轮。
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(chat_history or [])
    messages.append({"role": "user", "content": question})

serve.py:
# 同一对话中复用该 ID；不传时仍兼容原有单轮接口。
    session_id: str | None = None

MAX_HISTORY_MESSAGES = 12  # 最近 6 轮问答
_sessions: dict[str, list[dict[str, str]]] = {}
_session_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
# 同一对话中复用该 ID；不传时仍兼容原有单轮接口。
    session_id: str | None = None

# 同一 session 的请求串行执行，防止两个并发追问写乱历史。
    lock = _session_locks[session_id] if session_id else asyncio.Lock()
    async with lock:
        history = list(_sessions.get(session_id, [])) if session_id else []
        final_answer: str | None = None
        loop = asyncio.get_running_loop()

def _worker():
            try:
                for step_data in react_run(
                    question, max_steps=max_steps, chat_history=history
                ):
                    # Queue 不能从工作线程直接 put_nowait。
                    loop.call_soon_threadsafe(queue.put_nowait, step_data)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)


while True:
            step_data = await queue.get()
            if step_data is _SENTINEL:
                break
            if step_data.get("type") == "final":
                final_answer = step_data.get("answer", "")
            yield _sse(step_data)

# 只有成功得到最终回答才写入记忆，失败轮次不会污染后续追问。
        if session_id and final_answer:
            _sessions[session_id] = (
                history + [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": final_answer},
                ]
            )[-MAX_HISTORY_MESSAGES:]

        yield _sse({"type": "done", "session_id": session_id})
