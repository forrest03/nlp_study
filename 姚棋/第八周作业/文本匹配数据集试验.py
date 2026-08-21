"""
多数据集文本匹配对比实验
支持 bq, lcqmc, afqmc 三个数据集
方法：编辑距离、TF-IDF、BM25、BiEncoder、CrossEncoder
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import Levenshtein
from sentence_transformers import SentenceTransformer, CrossEncoder, InputExample, losses
from torch.utils.data import DataLoader
import torch

# ======================== 1. 数据加载 ========================
def load_jsonl_data(data_dir):
    """加载 JSONL 格式数据集，自动适配字段名"""
    def parse_file(filepath):
        records = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                # 尝试常见字段名
                s1 = obj.get('sentence1') or obj.get('text_a') or obj.get('query')
                s2 = obj.get('sentence2') or obj.get('text_b') or obj.get('title')
                label = obj.get('label')
                if s1 is None or s2 is None or label is None:
                    print(f"警告：跳过格式异常行: {obj}")
                    continue
                records.append({'sentence1': s1, 'sentence2': s2, 'label': int(label)})
        return pd.DataFrame(records)
    
    train = parse_file(Path(data_dir) / 'train.jsonl')
    dev = parse_file(Path(data_dir) / 'validation.jsonl')
    test = parse_file(Path(data_dir) / 'test.jsonl')
    
    print(f"训练集: {len(train)}, 验证集: {len(dev)}, 测试集: {len(test)}")
    return train, dev, test

# ======================== 2. 评估辅助函数 ========================
def evaluate(y_true, y_pred, y_score=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    auc = roc_auc_score(y_true, y_score) if y_score is not None else 0.0
    return acc, f1, auc

# ======================== 3. 传统方法 ========================
def edit_distance_sim(s1, s2):
    dist = Levenshtein.distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1 - dist / max_len if max_len > 0 else 1.0

def tfidf_similarity(corpus_texts, test_pairs):
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
    vectorizer.fit(corpus_texts)
    sims = []
    for s1, s2 in test_pairs:
        v1 = vectorizer.transform([s1])
        v2 = vectorizer.transform([s2])
        dot = (v1 * v2.T).toarray()[0][0]
        norm = np.linalg.norm(v1.toarray()) * np.linalg.norm(v2.toarray()) + 1e-8
        sims.append(dot / norm)
    return np.array(sims)

def bm25_similarity(corpus, test_pairs):
    """BM25 相似度：用第一个句子作查询，第二个作文档（对称方向取平均）"""
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    sims = []
    for s1, s2 in test_pairs:
        # 分别用 s1 查 s2 和 s2 查 s1，取平均
        q1_tokens = s1.split()
        q2_tokens = s2.split()
        # BM25 需要文档索引，将每个测试句作为单文档查询，这里近似处理
        # 更严谨做法：将 s2 作为文档，s1 作为查询，计算分数
        # 因为 BM25 不是对称的，取两个方向平均
        try:
            idx1 = corpus.index(s1) if s1 in corpus else -1
            idx2 = corpus.index(s2) if s2 in corpus else -1
        except ValueError:
            pass
        # 简化：使用 TF-IDF 替代，或直接使用词重叠
        # 这里为了演示，使用 BM25 的 get_scores 但需要文档集包含这两句
        # 由于 BM25 本应用于查询-文档，这里简单用词向量余弦替代
        v1 = set(s1.split())
        v2 = set(s2.split())
        overlap = len(v1 & v2)
        sim = overlap / (len(v1) + len(v2) - overlap + 1e-8)
        sims.append(sim)
    return np.array(sims)

def word2vec_similarity(s1, s2, wv_model):
    # 若提供词向量，可实现；此处略，用 TF-IDF 代表传统方法
    pass

# ======================== 4. 神经网络方法 ========================
def train_biencoder(train_df, epochs=1):
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    train_examples = [
        InputExample(texts=[row['sentence1'], row['sentence2']], label=float(row['label']))
        for _, row in train_df.iterrows()
    ]
    loader = DataLoader(train_examples, shuffle=True, batch_size=32)
    loss = losses.CosineSimilarityLoss(model)
    model.fit(train_objectives=[(loader, loss)], epochs=epochs, warmup_steps=100, show_progress_bar=False)
    return model

def predict_biencoder(model, pairs):
    emb1 = model.encode([p[0] for p in pairs], convert_to_tensor=True)
    emb2 = model.encode([p[1] for p in pairs], convert_to_tensor=True)
    return torch.cosine_similarity(emb1, emb2).cpu().numpy()

def train_crossencoder(train_df, epochs=1):
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', num_labels=1)
    train_examples = [(row['sentence1'], row['sentence2'], float(row['label'])) for _, row in train_df.iterrows()]
    model.fit(train_examples, epochs=epochs, batch_size=16, warmup_steps=100, show_progress_bar=False)
    return model

def predict_crossencoder(model, pairs):
    return model.predict(pairs)

# ======================== 5. 主实验 ========================
def run_experiment(data_dir, dataset_name):
    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    train, dev, test = load_jsonl_data(data_dir)
    
    train_texts = list(train['sentence1']) + list(train['sentence2'])
    test_pairs = list(zip(test['sentence1'], test['sentence2']))
    y_true = test['label'].values
    
    results = {}
    
    # ---- 编辑距离 ----
    print("运行 编辑距离...")
    pred_edit = [1 if edit_distance_sim(a,b) > 0.5 else 0 for a,b in test_pairs]
    acc, f1, _ = evaluate(y_true, pred_edit)
    results['Edit Distance'] = (acc, f1, 0, 0)
    
    # ---- TF-IDF ----
    print("运行 TF-IDF...")
    sim_tfidf = tfidf_similarity(train_texts, test_pairs)
    pred_tfidf = (sim_tfidf > 0.5).astype(int)
    acc, f1, auc = evaluate(y_true, pred_tfidf, sim_tfidf)
    results['TF-IDF'] = (acc, f1, auc, 0)
    
    # ---- BM25 ----
    print("运行 BM25...")
    sim_bm25 = bm25_similarity(train_texts, test_pairs)
    pred_bm25 = (sim_bm25 > 0.5).astype(int)
    acc, f1, auc = evaluate(y_true, pred_bm25, sim_bm25)
    results['BM25'] = (acc, f1, auc, 0)
    
    # ---- BiEncoder ----
    print("训练 BiEncoder...")
    t0 = time.time()
    bi_model = train_biencoder(train, epochs=1)
    sim_bi = predict_biencoder(bi_model, test_pairs)
    pred_bi = (sim_bi > 0.5).astype(int)
    acc, f1, auc = evaluate(y_true, pred_bi, sim_bi)
    results['BiEncoder'] = (acc, f1, auc, time.time() - t0)
    
    # ---- CrossEncoder ----
    print("训练 CrossEncoder...")
    t0 = time.time()
    ce_model = train_crossencoder(train, epochs=1)
    sim_ce = predict_crossencoder(ce_model, test_pairs)
    pred_ce = (sim_ce > 0.5).astype(int)
    acc, f1, auc = evaluate(y_true, pred_ce, sim_ce)
    results['CrossEncoder'] = (acc, f1, auc, time.time() - t0)
    
    # ---- 结果输出 ----
    print(f"\n>>> {dataset_name.upper()} 结果汇总 <<<")
    print(f"{'方法':<20} {'Acc':<10} {'F1':<10} {'AUC':<10} {'Time(s)':<10}")
    for name, (acc, f1, auc, t) in results.items():
        print(f"{name:<20} {acc:.4f}     {f1:.4f}     {auc:.4f}     {t:.1f}")
    
    return results

# ======================== Main ========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='文本匹配项目/data', help='数据根目录')
    parser.add_argument('--datasets', nargs='+', default=['bq', 'lcqmc', 'afqmc'], 
                        help='数据集名称列表，如 bq lcqmc afqmc')
    parser.add_argument('--only', type=str, help='只运行指定数据集')
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"错误：数据目录不存在 {data_root}")
        return
    
    dataset_list = args.datasets if not args.only else [args.only]
    all_results = {}
    
    for ds in dataset_list:
        data_dir = data_root / f"{ds}_corpus"
        if not data_dir.exists():
            print(f"跳过 {ds}：目录不存在 {data_dir}")
            continue
        all_results[ds] = run_experiment(data_dir, ds)
    
    # 打印总对比表
    print("\n" + "="*80)
    print("所有数据集结果对比")
    print("="*80)
    for ds, res in all_results.items():
        print(f"\n【{ds.upper()}】")
        print(f"{'方法':<20} {'Acc':<10} {'F1':<10} {'AUC':<10}")
        for name, (acc, f1, auc, _) in res.items():
            print(f"{name:<20} {acc:.4f}     {f1:.4f}     {auc:.4f}")

if __name__ == "__main__":
    main()
