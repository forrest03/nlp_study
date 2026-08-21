"""
BiEncoder文本匹配模型定义

架构：Siamese Network
  - 两个共享参数的BERT编码器
  - 分别对sentence1和sentence2进行编码
  - 使用余弦相似度计算匹配度

特点：
  - 支持CLS、MEAN、MAX三种池化方式
  - L2归一化后余弦相似度等价于点积
  - 可控制BERT层数加速训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig


class BiEncoder(nn.Module):
    """BiEncoder表示型文本匹配模型"""
    
    def __init__(self, bert_model_path='bert-base-chinese', 
                 pool='mean', dropout=0.1, num_hidden_layers=None):
        """
        初始化BiEncoder模型
        
        Args:
            bert_model_path: 预训练BERT模型路径或名称
            pool: 池化方式，'cls', 'mean', 或 'max'
            dropout: Dropout比例
            num_hidden_layers: BERT层数，None表示使用全部层
        """
        super(BiEncoder, self).__init__()
        
        # 加载预训练BERT配置
        if num_hidden_layers is not None:
            config = BertConfig.from_pretrained(bert_model_path)
            config.num_hidden_layers = num_hidden_layers
            self.bert = BertModel.from_pretrained(bert_model_path, config=config)
        else:
            self.bert = BertModel.from_pretrained(bert_model_path)
        
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        
        # 输出维度
        self.hidden_size = self.bert.config.hidden_size
        
    def encode(self, input_ids, attention_mask):
        """
        编码输入序列
        
        Args:
            input_ids: 输入ID
            attention_mask: 注意力掩码
            
        Returns:
            编码后的句向量
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        sequence_output = outputs.last_hidden_state
        
        if self.pool == 'cls':
            # 使用CLS向量
            pooled = sequence_output[:, 0, :]
        elif self.pool == 'mean':
            # 使用平均池化
            mask = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            masked_output = sequence_output * mask
            pooled = masked_output.sum(1) / mask.sum(1)
        elif self.pool == 'max':
            # 使用最大池化
            mask = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            masked_output = sequence_output.clone()
            masked_output[mask == 0] = float('-inf')
            pooled, _ = masked_output.max(1)
        else:
            raise ValueError(f"Unknown pool type: {self.pool}")
        
        return pooled
    
    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        """
        前向传播
        
        Args:
            input_ids1: 句子1的输入ID
            attention_mask1: 句子1的注意力掩码
            input_ids2: 句子2的输入ID
            attention_mask2: 句子2的注意力掩码
            
        Returns:
            余弦相似度
        """
        # 分别编码两个句子
        encoded1 = self.encode(input_ids1, attention_mask1)
        encoded2 = self.encode(input_ids2, attention_mask2)
        
        # L2归一化
        encoded1 = F.normalize(encoded1, p=2, dim=1)
        encoded2 = F.normalize(encoded2, p=2, dim=1)
        
        # 计算余弦相似度
        similarity = (encoded1 * encoded2).sum(dim=1)
        
        return similarity
    
    def get_embeddings(self, input_ids, attention_mask):
        """获取单个句子的嵌入向量"""
        return self.encode(input_ids, attention_mask)