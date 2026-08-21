"""
基于本地文件的问答系统（RAG 实现）
使用 sentence-transformers + faiss + Qwen2-0.5B-Instruct
"""

import os
import glob
import json
import argparse
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ---------- 1. 文档加载与分块 ----------
def load_documents(data_dir, extensions=(".txt", ".md", ".json")):
    """加载指定目录下所有扩展名的文件内容，按段落分块"""
    docs = []
    for ext in extensions:
        for filepath in glob.glob(os.path.join(data_dir, f"*{ext}")):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            # 按空行分割成段落
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            for para in paragraphs:
                docs.append({"text": para, "source": os.path.basename(filepath)})
    return docs

# ---------- 2. 向量索引 ----------
def build_index(docs, embed_model):
    """构建 FAISS 索引"""
    texts = [d["text"] for d in docs]
    embeddings = embed_model.encode(texts, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)   # 使用 L2 距离
    index.add(embeddings.astype(np.float32))
    return index, texts

# ---------- 3. 检索 ----------
def retrieve(query, embed_model, index, texts, top_k=3):
    """检索最相关的 top_k 文本块"""
    q_emb = embed_model.encode([query])
    distances, indices = index.search(q_emb.astype(np.float32), top_k)
    results = [texts[i] for i in indices[0]]
    return results

# ---------- 4. 生成回答 ----------
def generate_answer(query, contexts, tokenizer, model, max_new_tokens=256):
    """基于检索到的上下文生成回答"""
    # 构建提示词
    context_text = "\n\n".join(contexts)
    prompt = f"""你是一个基于给定资料回答问题的助手。请根据以下资料回答用户问题。如果资料中没有相关信息，请明确说明。

资料：
{context_text}

问题：{query}
回答："""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取回答部分（去除prompt）
    if "回答：" in answer:
        answer = answer.split("回答：")[-1].strip()
    return answer

# ---------- 5. 主程序 ----------
def main():
    parser = argparse.ArgumentParser(description="基于本地文件的问答系统")
    parser.add_argument("--data_dir", required=True, help="存放文档的文件夹路径")
    parser.add_argument("--model_name", default="Qwen/Qwen2-0.5B-Instruct", help="本地模型名称")
    parser.add_argument("--embed_model", default="all-MiniLM-L6-v2", help="嵌入模型")
    parser.add_argument("--top_k", type=int, default=3, help="检索返回的文档块数")
    args = parser.parse_args()

    # 加载文档
    print("加载文档...")
    docs = load_documents(args.data_dir)
    if not docs:
        print("错误：未找到任何文本文件（.txt, .md, .json）")
        return
    print(f"共加载 {len(docs)} 个文本块")

    # 加载嵌入模型
    print("加载嵌入模型...")
    embed_model = SentenceTransformer(args.embed_model)

    # 构建索引
    print("构建向量索引...")
    index, texts = build_index(docs, embed_model)
    print(f"索引构建完成，向量维度 {index.d}，样本数 {index.ntotal}")

    # 加载生成模型（本地）
    print("加载生成模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,   # CPU 用 float32，GPU 可改为 float16
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("系统已就绪！输入 'exit' 退出。")

    # 交互循环
    while True:
        query = input("\n请输入您的问题: ").strip()
        if query.lower() in ('exit', 'quit', 'q'):
            break
        if not query:
            continue

        # 检索
        contexts = retrieve(query, embed_model, index, texts, args.top_k)
        print("\n检索到的相关文档片段：")
        for i, ctx in enumerate(contexts, 1):
            print(f"{i}. {ctx[:80]}...")

        # 生成回答
        print("\n正在生成回答...")
        answer = generate_answer(query, contexts, tokenizer, model)
        print("\n回答：")
        print(answer)

if __name__ == "__main__":
    main()
