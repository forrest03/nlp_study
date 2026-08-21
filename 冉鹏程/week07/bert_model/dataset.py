from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

ROOT = Path(__file__).parent.parent


class MyBertDataset(Dataset):
    def __init__(self,
                 records: list,
                 tokenizer: BertTokenizer,
                 labels: list,
                 max_length: int):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = {name: id for id, name in enumerate(labels)}
        self.max_length = max_length
        self.labels = labels

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        tokens = row['tokens']
        ner_tags = row['ner_tags']
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)
            elif prev_word_id != wid:
                if wid < len(ner_tags):
                    aligned_labels.append(self.label2id[ner_tags[wid]])
                else:
                    aligned_labels.append(-100)
                prev_word_id = wid
            else:
                aligned_labels.append(-100)
        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": labels_tensor
        }