"""文本摘要脚本：基于统计方法提取文本关键句作为摘要"""
import sys
import json
import re
from collections import Counter


def split_sentences(text: str) -> list:
    """中文分句"""
    text = re.sub(r"\s+", " ", text)
    sentences = re.split(r"[。！？.!?\n]+", text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]


def tokenize(text: str) -> list:
    """简易分词：按非中文字符分割"""
    tokens = re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", text)
    return [t for t in tokens if len(t) > 1]


def summarize(text: str, max_words: int = 100) -> str:
    """基于词频的摘要提取"""
    sentences = split_sentences(text)
    if not sentences:
        return text[:max_words]

    if len(sentences) == 1:
        return sentences[0][:max_words]

    # 计算词频
    tokens = tokenize(text)
    if not tokens:
        return sentences[0][:max_words]

    word_freq = Counter(tokens)
    max_freq = max(word_freq.values())
    for w in word_freq:
        word_freq[w] /= max_freq

    # 为每个句子打分
    sentence_scores = []
    for i, sent in enumerate(sentences):
        sent_tokens = tokenize(sent)
        if not sent_tokens:
            sentence_scores.append((i, 0))
            continue
        score = sum(word_freq.get(t, 0) for t in sent_tokens) / len(sent_tokens)
        # 位置加权：前面的句子稍微加分
        position_weight = 1.0 if i < 2 else 0.8
        sentence_scores.append((i, score * position_weight))

    # 选取得分最高的句子，按原顺序排列
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    top_n = max(1, min(3, max_words // 30))
    selected_indices = sorted([idx for idx, _ in sentence_scores[:top_n]])

    summary = "。".join(sentences[i] for i in selected_indices)
    if len(summary) > max_words:
        summary = summary[:max_words] + "..."

    return summary + "。" if not summary.endswith("。") else summary


def main():
    raw = sys.stdin.read()
    try:
        params = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        print("错误：参数 JSON 格式无效")
        return

    text = params.get("text")
    if not text:
        print("错误：缺少必填参数 text")
        return

    max_words = params.get("max_words", 100)
    try:
        max_words = int(max_words)
    except (TypeError, ValueError):
        max_words = 100

    result = summarize(text, max_words)
    print(result)


if __name__ == "__main__":
    main()
