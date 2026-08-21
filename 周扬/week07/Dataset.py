"""
这段程序用来处理 peoples_daily 序列标注数据集，给后续 BERT 模型训练做准备。
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


# 当前文件所在目录：/Users/.../周扬/week07
current_path = Path(__file__).parent

# peoples_daily 数据目录就在当前目录下
data_dir = current_path / "peoples_daily"

# 数据文件路径
train_file = data_dir / "train.json"
test_file = data_dir / "test.json"
val_file = data_dir / "validation.json"
label_file = data_dir / "label_names.json"


def load_json(file_path: Path):
    """读取 json 文件。"""
    with open(file_path, "r", encoding="utf-8") as f:
        #使用json.load()读取json文件
        return json.load(f)


# 读取所有标签，并建立标签和 id 的映射
all_labels = load_json(label_file)
#根据标签获取id
label2id = {label: idx for idx, label in enumerate(all_labels)}
#根据id获取标签
id2label = {idx: label for label, idx in label2id.items()}


class PeopleDailyDataset(Dataset):
    """
    数据原始格式：
    {
        "tokens": ["海", "钓", "比", "赛", ...],
        "ner_tags": ["O", "O", "O", "O", ...]
    }
    """

    def __init__(
        self,
        file_path: Path,
        tokenizer: Optional[BertTokenizer] = None,
        max_length: int = 128,
    ):
        self.records = load_json(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        row = self.records[idx]
        tokens = row["tokens"]
        tag_names = row["ner_tags"]
        tag_ids = [label2id[tag] for tag in tag_names]

        # 1. 不传 tokenizer：返回原始样本，适合先观察数据和标签
        if self.tokenizer is None:
            return {
                "tokens": tokens,
                "tag_names": tag_names,
                "tag_ids": torch.tensor(tag_ids, dtype=torch.long),
                "text": "".join(tokens),
            }

        # 2. 传 tokenizer：把字符级标签对齐到 BERT token
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                # [CLS] / [SEP] / [PAD] 这些位置不参与 loss
                aligned_labels.append(-100)
            elif word_id != prev_word_id:
                # 首次遇到当前字符，使用真实标签
                if word_id < len(tag_ids):
                    aligned_labels.append(tag_ids[word_id])
                else:
                    aligned_labels.append(-100)
                prev_word_id = word_id
            else:
                # 如果一个字符被切成多个子词，后续子词位置不重复算 loss
                aligned_labels.append(-100)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }


def build_datasets(tokenizer=None, max_length: int = 128):
    """构建训练集、验证集、测试集。"""
    train_dataset = PeopleDailyDataset(train_file, tokenizer=tokenizer, max_length=max_length)
    val_dataset = PeopleDailyDataset(val_file, tokenizer=tokenizer, max_length=max_length)
    test_dataset = PeopleDailyDataset(test_file, tokenizer=tokenizer, max_length=max_length)
    return train_dataset, val_dataset, test_dataset


def build_dataloaders(
    tokenizer: BertTokenizer,
    batch_size: int = 32,
    max_length: int = 128,
):
    """
    构建训练/验证/测试
    """
    train_dataset, val_dataset, test_dataset = build_datasets(
        tokenizer=tokenizer,
        max_length=max_length,
    )

    print(
        f"数据集规模：训练={len(train_dataset)}，验证={len(val_dataset)}，测试={len(test_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = build_datasets()
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")

    sample = train_dataset[0]
    print("\n第一条样本:")
    print("text     :", sample["text"])
    print("tokens   :", sample["tokens"])
    print("tag_names:", sample["tag_names"])
    print("tag_ids  :", sample["tag_ids"].tolist())
