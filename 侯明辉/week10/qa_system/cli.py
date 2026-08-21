# week10检索增强生成/qa_system/cli.py
"""终端问答主程序：编排 检索 → 阈值 → 生成。"""
import sys

import config
import retriever as retriever_mod
import generator
import embedder

# Windows 控制台常为 GBK，LLM 回答里的非 GBK 字符会导致 print 崩溃。
# 保留原生编码（中文正常），仅将无法编码的字符替换为占位符。
try:
    sys.stdout.reconfigure(errors="replace")
except (AttributeError, ValueError):
    pass

HELP = "命令：/help 显示帮助 · /rebuild 重建索引 · /exit 退出。直接输入问题即可提问。"
REFUSE = "根据课程知识库未能找到与该问题相关的内容。"


def _ensure_index():
    if not (config.INDEX_DIR / "embeddings.npy").exists():
        print("索引不存在，开始建库 ...")
        import build_index
        build_index.build()


def main():
    try:
        config.get_api_key()
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    _ensure_index()
    client = embedder.get_client()
    retriever = retriever_mod.Retriever(embed_fn=lambda ts: embedder.embed_texts(ts, client))

    print("课程知识问答系统 (week1~10 + 名词解释)")
    print(HELP)

    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not query:
            continue
        if query in ("/exit", "/quit"):
            print("再见。")
            break
        if query == "/help":
            print(HELP)
            continue
        if query == "/rebuild":
            import build_index
            build_index.build()
            retriever = retriever_mod.Retriever(embed_fn=lambda ts: embedder.embed_texts(ts, client))
            print("索引已重建。")
            continue

        print("[思考中...]")
        results, top_vec = retriever.retrieve(query)
        if not retriever_mod.passes_threshold(top_vec):
            print(REFUSE)
            continue
        answer = generator.generate(query, results, client=client)
        print("\n" + answer)
        print("\n" + generator.format_sources(results))


if __name__ == "__main__":
    main()
