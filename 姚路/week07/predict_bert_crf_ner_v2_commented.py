#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chinese BERT + CRF 字符级命名实体识别预测脚本。

功能：
1. 加载训练阶段保存的模型；
2. 对输入中文文本进行字符级切分；
3. 使用 BERT 计算每个字符的表示；
4. 使用 CRF 解码得到最优 BIO 标签序列；
5. 按“字符 标签”的格式输出预测结果。

示例：
    python predict_bert_crf_ner_v2.py \
        --checkpoint bert_crf_ner_output_v2 \
        --text "张三毕业于北京大学，现在在腾讯工作。"
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class LinearChainCRF(nn.Module):
    """
    线性链条件随机场（Linear Chain CRF）。

    预测脚本中只需要使用 CRF 的解码功能，
    因此这里保留 BIO 约束构建和 Viterbi decode 逻辑。
    """

    def __init__(self, num_tags, labels=None):
        super().__init__()
        self.num_tags = num_tags
        self.labels = labels or []

        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

        # BIO 约束矩阵：合法转移为 0，非法转移为一个很小的负数。
        self.register_buffer(
            "constraint_mask",
            self._make_constraint_mask(num_tags, self.labels),
            persistent=False,
        )

    def _make_constraint_mask(self, n, labels):
        """
        构造 BIO 合法转移约束。

        规则：
            I-X 只能跟在 B-X 或 I-X 后面。
        """
        m = torch.zeros(n, n)

        for pi, p in enumerate(labels or []):
            for ci, c in enumerate(labels or []):
                if c.startswith("I-"):
                    t = c[2:]
                    if not (p == "B-" + t or p == "I-" + t):
                        m[pi, ci] = -10000.0

        return m

    def _trans(self):
        """
        返回加入 BIO 约束后的转移矩阵。
        """
        return self.transitions + self.constraint_mask

    @torch.no_grad()
    def decode(self, emissions, mask):
        """
        使用 Viterbi 算法解码最优标签序列。
        """
        score = self.start_transitions + emissions[:, 0]
        trans = self._trans()
        hist = []

        for t in range(1, emissions.size(1)):
            ns = score.unsqueeze(2) + trans.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            bs, bp = ns.max(dim=1)
            score = torch.where(mask[:, t].unsqueeze(1), bs, score)
            hist.append(bp)

        last = (score + self.end_transitions).argmax(dim=1)
        lengths = mask.long().sum(1).tolist()
        paths = []

        for b, L in enumerate(lengths):
            tag = last[b].item()
            path = [tag]
            for h in reversed(hist[: L - 1]):
                tag = h[b][tag].item()
                path.append(tag)
            paths.append(path[::-1])

        return paths


def compress_valid(emissions, label_mask):
    """
    根据 label_mask 压缩 BERT 输出。

    目的：
        让 CRF 只处理真实字符位置，跳过 [CLS] / [SEP] / PAD。
    """
    pieces = []
    lengths = []

    for i in range(emissions.size(0)):
        e = emissions[i][label_mask[i].bool()]
        pieces.append(e)
        lengths.append(e.size(0))

    B = emissions.size(0)
    C = emissions.size(-1)
    M = max(lengths)

    out = emissions.new_zeros(B, M, C)
    mask = torch.zeros(B, M, dtype=torch.bool, device=emissions.device)

    for i, e in enumerate(pieces):
        out[i, : e.size(0)] = e
        mask[i, : e.size(0)] = True

    return out, mask


class BertCRFForNER(nn.Module):
    """
    BERT + CRF 命名实体识别模型。

    预测流程：
        输入文本
            ↓
        BERT Encoder
            ↓
        Dropout + Linear
            ↓
        CRF Decode
            ↓
        BIO 标签序列
    """

    def __init__(self, model_name, num_labels, label_names, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = LinearChainCRF(num_labels, label_names)

    def forward(self, input_ids, attention_mask, token_type_ids=None, label_mask=None):
        """
        前向传播，返回压缩后的 emission 分数和对应 mask。
        """
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        em = self.classifier(self.dropout(out.last_hidden_state))
        return compress_valid(em, label_mask)


def main():
    """
    主函数：加载模型并对输入文本进行预测。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--text", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取训练阶段保存的 NER 配置。
    with open(os.path.join(args.checkpoint, "config_ner.json"), encoding="utf-8") as f:
        cfg = json.load(f)

    id2label = {int(k): v for k, v in cfg["id2label"].items()}
    label_names = [id2label[i] for i in range(len(id2label))]

    tok = AutoTokenizer.from_pretrained(args.checkpoint)

    # 注意：这里使用 cfg['model_name'] 初始化原始 BERT，
    # 再加载训练好的权重，避免把 checkpoint 目录误当成原始 BERT 加载。
    model = BertCRFForNER(
        cfg["model_name"],
        len(id2label),
        label_names,
        cfg.get("dropout", 0.1),
    ).to(device)

    model.load_state_dict(
        torch.load(os.path.join(args.checkpoint, "pytorch_model.bin"), map_location=device)
    )
    model.eval()

    # 字符级切分，保持与训练数据格式一致。
    chars = list(args.text.strip())

    enc = tok(
        chars,
        is_split_into_words=True,
        truncation=True,
        max_length=cfg.get("max_length", 128),
        padding="max_length",
        return_tensors="pt",
    )

    # 构造真实字符位置的 mask，跳过 [CLS] / [SEP] / PAD。
    word_ids = enc.word_ids(batch_index=0)
    lm = torch.tensor(
        [[1 if w is not None else 0 for w in word_ids]],
        dtype=torch.bool,
    ).to(device)

    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        em2, m2 = model(**enc, label_mask=lm)
        path = model.crf.decode(em2, m2)[0]

    tags = [id2label[x] for x in path][: len(chars)]

    # 按“字符 标签”的格式逐行输出预测结果。
    for ch, tag in zip(chars, tags):
        print(ch, tag)


if __name__ == "__main__":
    main()
