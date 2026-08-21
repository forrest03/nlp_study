import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class NERDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=64, label2id=None):
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        if label2id is None:
            self.label2id = self._build_label_vocab()
        else:
            self.label2id = label2id
        
        self.id2label = {v: k for k, v in self.label2id.items()}
    
    def _load_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _build_label_vocab(self):
        labels = set(['O'])
        for item in self.data:
            for tag in item['ner_tags']:
                if tag not in labels:
                    labels.add(tag)
        return {label: idx for idx, label in enumerate(sorted(labels))}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # print("进入__getitem__ func...")
        item = self.data[idx]
        tokens = item['tokens']
        tags = item['ner_tags']
        
        input_ids = []
        attention_mask = []
        label_ids = []
        
        for token, tag in zip(tokens, tags):
            sub_tokens = self.tokenizer.tokenize(token)
            if not sub_tokens:
                sub_tokens = [self.tokenizer.unk_token]
            
            input_ids.extend(self.tokenizer.convert_tokens_to_ids(sub_tokens))
            attention_mask.extend([1] * len(sub_tokens))
            
            label_ids.append(self.label2id.get(tag, self.label2id['O']))
            label_ids.extend([self.label2id['O']] * (len(sub_tokens) - 1))
        
        # 截断到 max_len - 2（预留 CLS 和 SEP 位置）
        input_ids = input_ids[:self.max_len - 2]
        attention_mask = attention_mask[:self.max_len - 2]
        label_ids = label_ids[:self.max_len - 2]
        
        # 添加特殊 token
        input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        attention_mask = [1] + attention_mask + [1]
        label_ids = [self.label2id['O']] + label_ids + [self.label2id['O']]
        
        # 计算需要填充的长度并进行 PAD 填充
        padding_len = self.max_len - len(input_ids)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        input_ids += [pad_token_id] * padding_len
        attention_mask += [0] * padding_len  # 填充位置 attention_mask 为 0
        label_ids += [self.label2id['O']] * padding_len  # 填充位置标签为 'O'
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long),
            'tokens': tokens,
            'original_tags': tags
        }