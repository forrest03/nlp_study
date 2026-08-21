import torch
import torch.nn as nn
from transformers import BertModel

try:
    from torchcrf import CRF
    HAS_CRF = True
except ImportError:
    HAS_CRF = False

class BertNERModel(nn.Module):
    def __init__(self, num_labels, bert_path='pretrain_models/bert-base-chinese', use_crf=False):
        super(BertNERModel, self).__init__()
        # Try to load from local path first, then try HuggingFace hub
        try:
            self.bert = BertModel.from_pretrained(bert_path)
        except Exception as e:
            print(f"Failed to load model from {bert_path}, trying bert-base-chinese from HuggingFace...")
            self.bert = BertModel.from_pretrained('bert-base-chinese')
        
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.use_crf = use_crf
        
        if use_crf:
            if not HAS_CRF:
                raise ImportError("torchcrf is not installed. Please install it with: pip install torchcrf")
            self.crf = CRF(num_labels, batch_first=True)
        
    def forward(self, input_ids, attention_mask, labels=None):
        # Use return_dict=True to ensure consistent output format
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        
        # Handle both dict and tuple output formats
        if isinstance(outputs, tuple):
            sequence_output = outputs[0]
        else:
            sequence_output = outputs.last_hidden_state
        
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        
        if self.use_crf:
            if labels is not None:
                loss = -self.crf(logits, labels, mask=attention_mask.bool(), reduction='mean')
            return loss, logits
        else:
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
                loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return loss, logits
    
    def predict(self, input_ids, attention_mask):
        """Predict labels for input sequences"""
        outputs = self.bert(input_ids, attention_mask, return_dict=True)
        if isinstance(outputs, tuple):
            sequence_output = outputs[0]
        else:
            sequence_output = outputs.last_hidden_state
        
        logits = self.classifier(self.dropout(sequence_output))
        
        if self.use_crf:
            return self.crf.decode(logits, mask=attention_mask.bool())
        else:
            return torch.argmax(logits, dim=-1)
