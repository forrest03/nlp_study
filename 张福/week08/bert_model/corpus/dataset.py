"""
BiEncoder文本匹配数据集处理模块

功能：
  1. PairDataset — 句对数据集，用于BiEncoder训练
  2. 数据加载与预处理
  3. 支持训练、验证、测试数据集

数据格式：
  {"sentence1": "文本1", "sentence2": "文本2", "label": 0或1}
  label=1表示相似/正确示例，label=0表示不相似/错误示例
"""

import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class BiEncoderDataset(Dataset):
    """BiEncoder训练数据集"""
    
    def __init__(self, data_path, tokenizer, max_length=64):
        """
        初始化数据集
        
        Args:
            data_path: JSONL数据文件路径
            tokenizer: BertTokenizer分词器
            max_length: 最大序列长度
        """
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def _load_data(self, path):
        """加载JSONL数据"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 对两个句子分别进行编码
        encoding1 = self.tokenizer(
            item['sentence1'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        encoding2 = self.tokenizer(
            item['sentence2'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids1': encoding1['input_ids'].squeeze(0),
            'attention_mask1': encoding1['attention_mask'].squeeze(0),
            'input_ids2': encoding2['input_ids'].squeeze(0),
            'attention_mask2': encoding2['attention_mask'].squeeze(0),
            'label': torch.tensor(item['label'], dtype=torch.float)
        }


def collate_fn(batch):
    """批处理整理函数"""
    return {
        'input_ids1': torch.stack([item['input_ids1'] for item in batch]),
        'attention_mask1': torch.stack([item['attention_mask1'] for item in batch]),
        'input_ids2': torch.stack([item['input_ids2'] for item in batch]),
        'attention_mask2': torch.stack([item['attention_mask2'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch])
    }


def load_jsonl(path):
    """加载JSONL文件"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data