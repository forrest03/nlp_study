"""
人民日报 NER 数据集类：CoNLL BIO 格式 + BERT 子词对齐

与 cluener 版本的核心区别：
  1. 数据格式不同
     - cluener：span 标注 {"label": {"name": {"叶老桂": [[9, 11]]}}}
     - peoples_daily：已标注好 BIO {"tokens": [...], "ner_tags": ["O", "B-PER", ...]}
     → 无需 span_to_bio 转换，直接使用 ner_tags
  2. 实体类型不同
     - cluener：10 类 → 21 个 BIO 标签
     - peoples_daily：3 类（PER/ORG/LOC）→ 7 个 BIO 标签
  3. 输入形式不同
     - cluener：原始文本 text → 需要自己 list(text) 拆字
     - peoples_daily：已拆好 tokens 列表，直接使用

使用方式：
  from dataset import build_label_schema, build_dataloaders
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"

# 人民日报 NER 只有 3 类实体
ENTITY_TYPES = ["PER", "ORG", "LOC"]

# 完整标签体系（与 label_names.json 一致）
LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def build_label_schema() -> tuple[list[str], dict[str, int], dict[int, str]]:
    """构建 BIO 标签体系，返回 (labels, label2id, id2label)。

    人民日报 NER 标签固定为 7 个：O + 3类×2(B/I)。
    不像 cluener 需要动态构建，这里直接使用 LABELS 常量。
    """
    label2id = {lbl: i for i, lbl in enumerate(LABELS)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return LABELS[:], label2id, id2label


class PeoplesDailyDataset(Dataset):
    """人民日报 NER 的 PyTorch Dataset。

    数据格式：
      {"tokens": ["在", "这", "里", ...], "ner_tags": ["O", "O", "O", ...]}

    处理流程（与 cluener 版本对比）：
      cluener:  text → span_to_bio → char_labels → tokenizer → word_ids 对齐
      pd:       tokens + ner_tags → char_labels → tokenizer → word_ids 对齐
                                              ↑ 跳过了 span_to_bio 转换

    子词对齐逻辑完全相同：
      - word_ids() 获取每个 subword token 对应的原始字符索引
      - 非首子词、特殊 token 标记为 -100（cross_entropy 的 ignore_index）
    """

    def __init__(
        self,
        records: list,
        tokenizer: BertTokenizer,
        label2id: dict,
        max_length: int = 128,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        tokens: list[str] = row["tokens"]
        ner_tags: list[str] = row.get("ner_tags", [])

        # 1. 将 BIO 标签字符串转为 id
        #    peoples_daily 的 ner_tags 已经是 BIO 格式，直接映射即可
        #    不像 cluener 需要先做 span_to_bio 转换
        char_labels = [self.label2id.get(t, 0) for t in ner_tags]

        # 2. tokens 已经是字符列表，直接传入 tokenizer
        #    is_split_into_words=True：让 word_ids() 与 tokens 索引对齐
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # 3. 子词对齐：与 cluener 完全相同的逻辑
        #    - word_ids() 返回 [None, 0, 1, 2, ..., None]
        #    - None 对应 [CLS]/[SEP]/[PAD]
        #    - 非首子词标记为 -100
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)
            elif wid != prev_word_id:
                # 首次出现的 token：使用对应的 BIO 标签
                if wid < len(char_labels):
                    aligned_labels.append(char_labels[wid])
                else:
                    aligned_labels.append(-100)
                prev_word_id = wid
            else:
                # 同一 token 的后续子词：标记为 -100
                aligned_labels.append(-100)

        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": labels_tensor,
        }


def load_records(split: str, data_dir: Optional[Path] = None) -> list:
    """加载人民日报 NER 数据。"""
    d = data_dir or DATA_DIR
    with open(d / f"{split}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataloaders(
    tokenizer: BertTokenizer,
    label2id: dict,
    batch_size: int = 32,
    max_length: int = 128,
    data_dir: Optional[Path] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建训练/验证/测试 DataLoader，返回 (train_loader, val_loader, test_loader)。"""
    train_records = load_records("train", data_dir)
    val_records = load_records("validation", data_dir)
    test_records = load_records("test", data_dir)

    train_ds = PeoplesDailyDataset(train_records, tokenizer, label2id, max_length)
    val_ds = PeoplesDailyDataset(val_records, tokenizer, label2id, max_length)
    test_ds = PeoplesDailyDataset(test_records, tokenizer, label2id, max_length)

    print(f"数据集规模：训练={len(train_ds)}，验证={len(val_ds)}，测试={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader