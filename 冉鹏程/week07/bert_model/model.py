import transformers
from transformers import BertModel
import torch
from torch import nn


def _load_bert(bert_path: str):
    prev = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    try:
        bert = BertModel.from_pretrained(bert_path)
    finally:
        transformers.logging.set_verbosity(prev)
    return bert


class BertNER(nn.Module):
    def __init__(self, bert_path: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert = _load_bert(bert_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        seq_output = outputs.last_hidden_state  # (B, L, H)
        logits = self.classifier(self.dropout(seq_output))
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return logits, loss

        return logits, loss