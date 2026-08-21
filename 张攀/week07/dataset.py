"""
NER 数据集类：span 标注→BIO 转换 + BERT 子词对齐

教学重点：
  1. cluener2020 的 span 格式转为 BIO 格式
     - span: {"name": {"叶老桂": [[9, 11]]}}
     - BIO:  ['O','O',...,'B-name','I-name','I-name',...]
  2. BERT 子词对齐（word_ids 策略）
     - 中文字符通常一字一token，但 [UNK] 和特殊字符可能例外
     - 非首子词标记为 -100，在 loss 计算中被忽略
  3. DataLoader 工厂函数统一封装

使用方式：
  from dataset import build_label_schema, build_dataloaders
"""
import sys
import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"


def build_label_schema(data_dir: Path = DATA_DIR) -> tuple[list[str], dict, dict]:
    # 加载json文件读取内容
    with open(data_dir / "label_names.json", encoding="utf-8") as f:
        data = json.load(f)

    labels: list = data # 此处labels是BIO标签

    # 此处的id才是最后的label（数字）
    label2id = {lb: i for i, lb in enumerate(labels)}
    id2label = {i: lb for lb, i in label2id.items()}
    return labels, label2id, id2label

def span_to_bio(text: str, label_dict: dict, label2id: dict) -> list[int]:
    """将 cluener2020 的 span 格式标注转为逐字符 BIO 标签 id 列表。

    教学要点：先全部初始化为 O，再按 span 位置填入 B/I。
    若存在嵌套实体（本数据集极少），外层实体覆盖内层。
    """
    n = len(text)
    bio = ["O"] * n

    if not label_dict:
        return [label2id[t] for t in bio]

    for etype, spans in label_dict.items():
        b_tag = f"B-{etype}"
        i_tag = f"I-{etype}"
        for surface, positions in spans.items():
            for start, end in positions:
                if start >= n or end >= n:
                    continue
                bio[start] = b_tag
                for idx in range(start + 1, end + 1):
                    bio[idx] = i_tag
    # print(f"bio:\n{bio}")
    return [label2id.get(t, 0) for t in bio]


class DailyDataset(Dataset):
    """cluener2020 的 PyTorch Dataset。

    教学流程：
      text → span_to_bio → 字符级 BIO ids
           → BertTokenizer (is_split_into_words=True)
           → 用 word_ids() 对齐子词标签（非首子词设为 -100）
           → 返回 input_ids / attention_mask / token_type_ids / labels
    """

    def __init__(
        self,
        records: list,  # 记录列表，包含文本和标签信息
        tokenizer: BertTokenizer,  # BERT分词器，用于将文本转换为token
        label2id: dict,  # 标签到ID的映射字典
        max_length: int = 128,  # 最大序列长度，默认为128
    ):
        self.records = records  # 存储输入记录
        self.tokenizer = tokenizer  # 存储分词器
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]     # [{token:[], tag:[]}, {}, ...]
        text_list: list = row["tokens"]
        text_str = "".join(text_list)
        # print(f"text_str:\n{text_str}")

        encoding = self.tokenizer(
            text_list,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        # BIO 字符列表
        tag_list: list = row["ner_tags"]
        # BIO 转为 id 列表
        tag_ids = [self.label2id.get(t, 0) for t in tag_list]
        # max_length，token位置列表，非真实token为None
        word_ids = encoding.word_ids(batch_index=0)
        # print(f"tag_list:\n{tag_list}")
        # print(f"tag_ids:\n{tag_ids}")
        # print(f"word_ids:\n{word_ids}")

        # max_length长度的 BIO id 列表包含有效标签和-100
        aligned_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)
            elif wid != prev_word_id:
                # 首次出现这个字符索引：使用 BIO 标签
                if wid < len(tag_ids):
                    aligned_labels.append(tag_ids[wid])
                else:
                    aligned_labels.append(-100)
                prev_word_id = wid
            else:
                # 同一字符的后续子词（中文通常不会出现，但保留正确处理）
                aligned_labels.append(-100)

        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)

        # # 2D
        # print(f"input_ids:\n{encoding['input_ids']}")
        # print(f"attention_mask:\n{encoding['attention_mask']}")
        # print(f"token_type_ids:\n{encoding['token_type_ids']}")
        # # 1D
        # print(f"labels_tensor:\n{labels_tensor}")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": labels_tensor,
        }


def load_records(split: str, data_dir: Optional[Path] = None) -> list:
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

    train_ds = DailyDataset(train_records, tokenizer, label2id, max_length)
    val_ds = DailyDataset(val_records, tokenizer, label2id, max_length)
    test_ds = DailyDataset(test_records, tokenizer, label2id, max_length)

    print(f"数据集规模：训练={len(train_ds)}，验证={len(val_ds)}，测试={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


# def main():
#     """测试数据集构建函数。"""
#     from transformers import BertTokenizer, BertTokenizerFast

#     tokenizer = BertTokenizerFast.from_pretrained(r"C:\Users\Dell\Downloads\py文件\NLP\bert-base-chinese")
#     labels, label2id, id2label = build_label_schema(DATA_DIR)
#     print(f"labels:\n{labels}")
#     print(f"label2id:\n{label2id}")
#     print(f"id2label:\n{id2label}")
#     print("*" * 20)

#     train_loader, val_loader, test_loader = build_dataloaders(
#             tokenizer=tokenizer,
#             label2id=label2id,
#             batch_size=3,
#             max_length=128,
#             data_dir= DATA_DIR)

#     for step, batch in enumerate(train_loader):
#         input(f"batch{step} done")


# if __name__ == "__main__":
#     main()
