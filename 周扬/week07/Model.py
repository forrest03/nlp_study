'''
这里主要定义 bert 模型的序列标注任务模型，写 BertNer。
'''

import torch.nn.functional as F
from torch import nn
from transformers import BertModel

#定义bert模型的路径
bert_model_path = "/Users/zhouyang/myworkspace/badou-nlp/周扬/week07/bert-base-chinese"

class BertNer(nn.Module):
    """BERT 序列标注模型。"""

    def __init__(self, num_labels, dropout=0.1) -> None:
        super().__init__()
        # 定义模型结构
        # 1. 先把 bert 模型加载进来，路径是 bert_model_path
        self.bert = BertModel.from_pretrained(bert_model_path)
        hidden_size = self.bert.config.hidden_size
        # 2. dropout 层用于减少过拟合
        self.dropout = nn.Dropout(dropout)
        # 3. bert 后加一个输出线性层
        #    num_labels 是标签数量，hidden_size 是 bert 的隐藏层维度
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels
        # 先不加 CRF，先做最基础的 BertNER

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        """前向传播。"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        # last_hidden_state 的形状是 (batch_size, seq_len, hidden_size)
        seq_output = outputs.last_hidden_state
        logits = self.classifier(self.dropout(seq_output))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss
