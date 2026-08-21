# week10检索增强生成/qa_system/generator.py
"""构造 RAG Prompt、调用 LLM 生成答案、拼装来源列表。"""
import config

SYSTEM_PROMPT = """你是一个课程笔记问答助手，专门回答关于 AI/深度学习/NLP 课程笔记（week1~week10）的问题。

回答规则：
1. 只根据【参考资料】中的内容回答，不得引用或编造资料外的知识。
2. 若参考资料不足以支撑回答，直接说"根据提供的资料无法回答此问题"。
3. 引用具体结论或数据时，在句末标注来源编号，如：RRF 无需归一化[1]。
4. 回答简洁，重点突出，避免无关废话。"""


def build_context(chunks) -> str:
    """将检索到的块组装为带编号的上下文字符串。"""
    parts = []
    for i, c in enumerate(chunks, 1):
        label = f"[{i}] {c.get('section_path', '')}"
        parts.append(f"{label}\n{c.get('content', '')}")
    return "\n\n---\n\n".join(parts)


def format_sources(chunks) -> str:
    """拼装 `── 来源 ──` 列表。"""
    lines = ["── 来源 ──"]
    for i, c in enumerate(chunks, 1):
        week = c.get("week", "")
        section = c.get("section_path", "")
        lines.append(f"  [{i}] {week} · {section}")
    return "\n".join(lines)


def generate(query, chunks, client=None) -> str:
    """调用 LLM 生成答案文本（不含来源列表，由调用方拼接）。"""
    if client is None:
        import embedder
        client = embedder.get_client()
    context = build_context(chunks)
    user_msg = (
        f"【参考资料】\n{context}\n\n"
        f"【问题】\n{query}\n\n"
        "请根据参考资料回答，并在引用结论处标注来源编号（如[1]）。"
    )
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=config.LLM_TEMPERATURE,
    )
    return resp.choices[0].message.content
