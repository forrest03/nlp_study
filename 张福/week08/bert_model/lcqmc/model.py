"""
LCQMC BiEncoder模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig


class BiEncoder(nn.Module):
    """BiEncoder文本匹配模型"""
    
    def __init__(self, bert_model_path='bert-base-chinese', 
                 pool='mean', dropout=0.1, num_hidden_layers=None):
        super(BiEncoder, self).__init__()
        
        if num_hidden_layers is not None:
            config = BertConfig.from_pretrained(bert_model_path)
            config.num_hidden_layers = num_hidden_layers
            self.bert = BertModel.from_pretrained(bert_model_path, config=config)
        else:
            self.bert = BertModel.from_pretrained(bert_model_path)
        
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = self.bert.config.hidden_size
        
    def encode(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        sequence_output = outputs.last_hidden_state
        
        if self.pool == 'cls':
            pooled = sequence_output[:, 0, :]
        elif self.pool == 'mean':
            mask = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            masked_output = sequence_output * mask
            pooled = masked_output.sum(1) / mask.sum(1)
        elif self.pool == 'max':
            mask = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            masked_output = sequence_output.clone()
            masked_output[mask == 0] = float('-inf')
            pooled, _ = masked_output.max(1)
        else:
            raise ValueError(f"Unknown pool type: {self.pool}")
        
        return pooled
    
    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        encoded1 = self.encode(input_ids1, attention_mask1)
        encoded2 = self.encode(input_ids2, attention_mask2)
        
        encoded1 = F.normalize(encoded1, p=2, dim=1)
        encoded2 = F.normalize(encoded2, p=2, dim=1)
        
        similarity = (encoded1 * encoded2).sum(dim=1)
        
        return similarity