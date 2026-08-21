#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chinese BERT + CRF 字符级命名实体识别训练脚本。

功能：
1. 读取 train / validation / test 数据集；
2. 使用 bert-base-chinese 作为编码器；
3. 在线性分类层后接 CRF 进行序列标注；
4. 修复特殊 token 对齐问题：CRF 只作用在真实字符位置，不处理 [CLS] / [SEP] / PAD；
5. 训练过程中根据验证集 F1 保存最佳模型；
6. 训练结束后在测试集上评估模型效果。

数据格式：
    [
        {
            "tokens": ["张", "三", "在", "北", "京"],
            "ner_tags": ["B-PER", "I-PER", "O", "B-LOC", "I-LOC"]
        }
    ]

标签格式：
    O
    B-PER / I-PER
    B-ORG / I-ORG
    B-LOC / I-LOC
"""

import argparse
import json
import os
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


def set_seed(seed: int):
    """
    固定随机种子，尽量保证实验结果可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path):
    """
    读取 JSON 文件。
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_entities(tags: List[str]) -> set:
    """
    从 BIO 标签序列中抽取实体集合。

    返回格式：
        (实体类型, 起始位置, 结束位置)

    示例：
        ["B-PER", "I-PER", "O"] -> {("PER", 0, 1)}
    """
    ents = set()
    start = None
    typ = None

    # 末尾额外追加一个 O，方便统一处理最后一个实体的闭合。
    for i, tag in enumerate(tags + ["O"]):
        if tag.startswith("B-"):
            if typ is not None:
                ents.add((typ, start, i - 1))
            typ = tag[2:]
            start = i
        elif tag.startswith("I-") and typ == tag[2:]:
            continue
        else:
            if typ is not None:
                ents.add((typ, start, i - 1))
            typ = None
            start = None

    return ents


def compute_metrics(preds, golds):
    """
    计算实体级 Precision / Recall / F1，以及 token-level accuracy。
    """
    cp = tp = tg = tc = tt = 0

    for p, g in zip(preds, golds):
        pe = extract_entities(p)
        ge = extract_entities(g)

        # 实体级统计。
        cp += len(pe & ge)
        tp += len(pe)
        tg += len(ge)

        # token 级准确率统计。
        for pp, gg in zip(p, g):
            tc += int(pp == gg)
            tt += 1

    P = cp / tp if tp else 0.0
    R = cp / tg if tg else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0

    return {
        "precision": P,
        "recall": R,
        "f1": F,
        "token_acc": tc / tt if tt else 0.0,
    }


class LinearChainCRF(nn.Module):
    """
    线性链条件随机场（Linear Chain CRF）。

    作用：
    1. 学习标签之间的转移分数；
    2. 通过 BIO 约束避免非法路径，例如 O -> I-PER；
    3. 训练时计算负对数似然损失；
    4. 预测时使用 Viterbi 解码得到最优标签序列。
    """

    def __init__(self, num_tags: int, labels: List[str] = None):
        super().__init__()
        self.num_tags = num_tags
        self.labels = labels or []

        # 起始转移、结束转移、标签间转移矩阵。
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))  # prev -> curr

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

        if labels:
            for prev_i, prev in enumerate(labels):
                for cur_i, cur in enumerate(labels):
                    if cur.startswith("I-"):
                        cur_t = cur[2:]
                        ok = (prev == "B-" + cur_t) or (prev == "I-" + cur_t)
                        if not ok:
                            m[prev_i, cur_i] = -10000.0

        return m

    def _trans(self):
        """
        返回加入 BIO 约束后的转移矩阵。
        """
        return self.transitions + self.constraint_mask

    def forward(self, emissions, tags, mask):
        """
        计算 CRF 损失。

        emissions: [B, T, C]
        tags:      [B, T]
        mask:      [B, T]
        """
        return (self._normalizer(emissions, mask) - self._score(emissions, tags, mask)).mean()

    def _score(self, emissions, tags, mask):
        """
        计算真实标签路径的分数。
        """
        score = self.start_transitions[tags[:, 0]] + emissions[:, 0].gather(
            1,
            tags[:, :1],
        ).squeeze(1)
        trans = self._trans()

        for t in range(1, emissions.size(1)):
            emit = emissions[:, t].gather(1, tags[:, t : t + 1]).squeeze(1)
            tr = trans[tags[:, t - 1], tags[:, t]]
            score += (emit + tr) * mask[:, t]

        last_idx = mask.long().sum(1) - 1
        last_tags = tags.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]

        return score

    def _normalizer(self, emissions, mask):
        """
        使用动态规划计算所有可能路径的 log-sum-exp 分数。
        """
        score = self.start_transitions + emissions[:, 0]
        trans = self._trans()

        for t in range(1, emissions.size(1)):
            ns = score.unsqueeze(2) + trans.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            ns = torch.logsumexp(ns, dim=1)
            score = torch.where(mask[:, t].unsqueeze(1), ns, score)

        return torch.logsumexp(score + self.end_transitions, dim=1)

    @torch.no_grad()
    def decode(self, emissions, mask):
        """
        使用 Viterbi 算法解码最优标签路径。
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


class NERDataset(Dataset):
    """
    命名实体识别数据集类。

    将字符级 tokens 和 ner_tags 转换成 BERT 输入，并生成：
    - input_ids
    - attention_mask
    - token_type_ids
    - labels
    - label_mask

    其中 label_mask 用于标记真实字符位置，避免 CRF 处理 [CLS] / [SEP] / PAD。
    """

    def __init__(self, data, tokenizer, label2id, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"]
        tags = item["ner_tags"]

        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        word_ids = enc.word_ids()
        label_ids = []
        label_mask = []
        prev = None

        # 只给每个原始字符的第一个 sub-token 赋标签。
        for wid in word_ids:
            if wid is None:
                label_ids.append(0)
                label_mask.append(0)
            elif wid != prev:
                label_ids.append(self.label2id[tags[wid]])
                label_mask.append(1)
            else:
                label_ids.append(0)
                label_mask.append(0)
            prev = wid

        enc["labels"] = label_ids
        enc["label_mask"] = label_mask

        return {k: torch.tensor(v, dtype=torch.long) for k, v in enc.items()}


def collate_fn(batch):
    """
    DataLoader 的 batch 拼接函数。
    """
    return {k: torch.stack([x[k] for x in batch]) for k in batch[0].keys()}


def compress_valid(emissions, labels=None, label_mask=None):
    """
    根据 label_mask 压缩 BERT 输出。

    输入：
        emissions: [B, bert_len, C]
        labels:    [B, bert_len]
        label_mask:[B, bert_len]

    输出：
        out:     [B, max_chars, C]
        lab_out: [B, max_chars]
        mask:    [B, max_chars]

    目的：
        让 CRF 只看到真实字符位置，不处理 [CLS] / [SEP] / PAD。
    """
    pieces = []
    lab_pieces = []
    lengths = []

    for i in range(emissions.size(0)):
        m = label_mask[i].bool()
        e = emissions[i][m]
        pieces.append(e)
        lengths.append(e.size(0))

        if labels is not None:
            lab_pieces.append(labels[i][m])

    max_len = max(lengths)
    B = emissions.size(0)
    C = emissions.size(-1)

    out = emissions.new_zeros(B, max_len, C)
    mask = torch.zeros(B, max_len, dtype=torch.bool, device=emissions.device)
    lab_out = None

    if labels is not None:
        lab_out = torch.zeros(B, max_len, dtype=torch.long, device=emissions.device)

    for i, e in enumerate(pieces):
        L = e.size(0)
        out[i, :L] = e
        mask[i, :L] = True

        if labels is not None:
            lab_out[i, :L] = lab_pieces[i]

    return out, lab_out, mask


class BertCRFForNER(nn.Module):
    """
    BERT + CRF 命名实体识别模型。

    结构：
        输入字符序列
            ↓
        BERT Encoder
            ↓
        Dropout
            ↓
        Linear 分类层
            ↓
        CRF
            ↓
        BIO 标签序列
    """

    def __init__(self, model_name, num_labels, label_names, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = LinearChainCRF(num_labels, label_names)

    def emissions(self, input_ids, attention_mask, token_type_ids=None):
        """
        计算每个位置对应各标签的 emission 分数。
        """
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return self.classifier(self.dropout(out.last_hidden_state))

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, label_mask=None):
        """
        前向传播并计算 CRF 损失。
        """
        em = self.emissions(input_ids, attention_mask, token_type_ids)
        em2, lab2, m2 = compress_valid(em, labels, label_mask)
        return self.crf(em2, lab2, m2), em2, m2

    def decode_batch(self, input_ids, attention_mask, token_type_ids=None, label_mask=None):
        """
        对一个 batch 的样本进行 CRF 解码。
        """
        em = self.emissions(input_ids, attention_mask, token_type_ids)
        em2, _, m2 = compress_valid(em, None, label_mask)
        return self.crf.decode(em2, m2)


def evaluate(model, loader, id2label, device):
    """
    在验证集或测试集上评估模型。
    """
    model.eval()
    all_p = []
    all_g = []
    total = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            lm = batch.pop("label_mask")

            loss, em2, m2 = model(**batch, labels=labels, label_mask=lm)
            total += loss.item()
            paths = model.crf.decode(em2, m2)

            for i, p in enumerate(paths):
                gold_ids = labels[i][lm[i].bool()].cpu().tolist()
                all_p.append([id2label[x] for x in p[: len(gold_ids)]])
                all_g.append([id2label[x] for x in gold_ids])

    met = compute_metrics(all_p, all_g)
    met["loss"] = total / max(1, len(loader))

    return met


def save_checkpoint(path, model, tokenizer, label2id, id2label, args, metrics):
    """
    保存模型权重、tokenizer 和 NER 配置文件。
    """
    os.makedirs(path, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(path, "pytorch_model.bin"))
    tokenizer.save_pretrained(path)

    with open(os.path.join(path, "config_ner.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "label2id": label2id,
                "id2label": id2label,
                "max_length": args.max_length,
                "dropout": args.dropout,
                "metrics": metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():
    """
    主函数：解析参数、加载数据、训练模型、评估模型。
    """
    ap = argparse.ArgumentParser()

    # 数据路径参数。
    ap.add_argument("--train", default="train.json")
    ap.add_argument("--validation", default="validation.json")
    ap.add_argument("--test", default="test.json")
    ap.add_argument("--labels", default="label_names.json")

    # 模型与输出路径参数。
    ap.add_argument("--model-name", default="bert-base-chinese")
    ap.add_argument("--output-dir", default="bert_crf_ner_output_v2")

    # 训练超参数。
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ==========================
    # 加载标签与数据集
    # ==========================
    labels = load_json(args.labels)
    label2id = {x: i for i, x in enumerate(labels)}
    id2label = {i: x for x, i in label2id.items()}

    tr = load_json(args.train)
    va = load_json(args.validation)
    te = load_json(args.test)

    print(f"Dataset sizes: train={len(tr)}, valid={len(va)}, test={len(te)}")
    print("Labels:", labels)

    tok = AutoTokenizer.from_pretrained(args.model_name)

    # ==========================
    # 构建 DataLoader
    # ==========================
    train_loader = DataLoader(
        NERDataset(tr, tok, label2id, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        NERDataset(va, tok, label2id, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        NERDataset(te, tok, label2id, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # ==========================
    # 初始化模型
    # ==========================
    model = BertCRFForNER(args.model_name, len(labels), labels, args.dropout).to(device)

    # BERT 常规优化设置：bias 和 LayerNorm 不做 weight decay。
    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    opt = torch.optim.AdamW(params, lr=args.lr)
    steps = len(train_loader) * args.epochs
    sch = get_linear_schedule_with_warmup(
        opt,
        int(steps * args.warmup_ratio),
        steps,
    )

    best = -1

    # ==========================
    # 训练阶段
    # ==========================
    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{args.epochs}")

        for batch in pbar:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            labels_t = batch.pop("labels")
            lm = batch.pop("label_mask")

            loss, _, _ = model(**batch, labels=labels_t, label_mask=lm)
            loss.backward()

            # 梯度裁剪，防止梯度爆炸。
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            opt.step()
            sch.step()
            opt.zero_grad(set_to_none=True)

            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ==========================
        # 验证阶段
        # ==========================
        vm = evaluate(model, valid_loader, id2label, device)
        print(
            f"Epoch {ep}: "
            f"train_loss={total / len(train_loader):.4f} | "
            f"valid_loss={vm['loss']:.4f} | "
            f"P={vm['precision']:.4f} "
            f"R={vm['recall']:.4f} "
            f"F1={vm['f1']:.4f} "
            f"Acc={vm['token_acc']:.4f}"
        )

        # ==========================
        # 保存验证集 F1 最好的模型
        # ==========================
        if vm["f1"] > best:
            best = vm["f1"]
            save_checkpoint(args.output_dir, model, tok, label2id, id2label, args, vm)
            print("Saved best model to", args.output_dir)

    # ==========================
    # 测试阶段
    # ==========================
    model.load_state_dict(
        torch.load(os.path.join(args.output_dir, "pytorch_model.bin"), map_location=device)
    )
    tm = evaluate(model, test_loader, id2label, device)

    print(
        f"Test: "
        f"P={tm['precision']:.4f} "
        f"R={tm['recall']:.4f} "
        f"F1={tm['f1']:.4f} "
        f"Acc={tm['token_acc']:.4f} "
        f"Loss={tm['loss']:.4f}"
    )

    with open(os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(tm, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
