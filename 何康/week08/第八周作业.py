在BQ_Corpus数据集上，采用BERT + Mean Pooling模型进行文本匹配训练。
在LCQMC数据集上，采用BERT + BiLSTM模型进行训练。
bq_corpus:
Sentence Pair
      ↓
BERT
      ↓
Mean Pooling
      ↓
Linear
      ↓
Softmax

import torch
import torch.nn as nn
from transformers import BertModel

class BertMeanPooling(nn.Module):

    def __init__(self):
        super().__init__()

        self.bert = BertModel.from_pretrained(
            "bert-base-chinese"
        )

        self.fc = nn.Linear(
            768,
            2
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1)

        pooled = (
            hidden * mask
        ).sum(dim=1) / mask.sum(dim=1)

        logits = self.fc(
            pooled
        )

        return logits

  训练：
model = BertMeanPooling().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5
)

lcqmc:
Sentence Pair
      ↓
BERT
      ↓
BiLSTM
      ↓
Linear
      ↓
Softmax

import torch
import torch.nn as nn
from transformers import BertModel

class BertBiLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.bert = BertModel.from_pretrained(
            "bert-base-chinese"
        )

        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=256,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(
            512,
            2
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden = outputs.last_hidden_state

        lstm_out, _ = self.lstm(
            hidden
        )

        feat = lstm_out[:, 0, :]

        logits = self.fc(
            feat
        )

        return logits
