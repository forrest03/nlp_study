import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel
from sklearn.metrics import f1_score, precision_score, recall_score

def collate_fn(batch):
    """Custom collate function to handle variable-length tokens and tags"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    tokens = [item['tokens'] for item in batch]
    original_tags = [item['original_tags'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'tokens': tokens,
        'original_tags': original_tags
    }

# Try to import CRF
try:
    from torchcrf import CRF
    HAS_CRF = True
except ImportError:
    HAS_CRF = False

print("1. Starting BERT NER Training...")

def parse_args():
    parser = argparse.ArgumentParser(description='BERT NER Training')
    parser.add_argument('--data_dir', type=str, default='D:\\workspace\\hub-TroE\\张福\\week07\\data\\peoples_daily')
    parser.add_argument('--output_dir', type=str, default='D:\\workspace\\hub-TroE\\张福\\week07\\outputs')
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--bert_path', type=str, default='pretrain_models/bert-base-chinese')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use_crf', action='store_true', default=False,
                        help='Whether to use CRF layer for sequence labeling')
    return parser.parse_args()

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

def decode_entities(tokens, pred_labels, id2label):
    entities = []
    current_entity = None
    
    for i, (token, label_id) in enumerate(zip(tokens, pred_labels)):
        label = id2label[label_id]
        
        if label.startswith('B-'):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                'text': token,
                'label': label[2:],
                'start': i,
                'end': i + 1
            }
        elif label.startswith('I-') and current_entity:
            current_entity['text'] += token
            current_entity['end'] = i + 1
        elif label.startswith('E-') and current_entity:
            current_entity['text'] += token
            current_entity['end'] = i + 1
            entities.append(current_entity)
            current_entity = None
        elif label == 'O':
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    if current_entity:
        entities.append(current_entity)
    
    return entities

def evaluate(model, dataloader, label2id, id2label, device, decode_samples=False):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    decoded_samples = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            loss, logits = model(input_ids, attention_mask, labels)
            total_loss += loss.item()
            
            # Use CRF decode if available, otherwise use argmax
            if model.use_crf:
                preds = model.crf.decode(logits, mask=attention_mask.bool())
                # Convert list of lists to padded tensor
                preds = pad_sequence([torch.tensor(p) for p in preds], batch_first=True, padding_value=label2id['O']).to(device)
                # Pad to max_len if needed
                if preds.size(1) < attention_mask.size(1):
                    preds = torch.nn.functional.pad(preds, (0, attention_mask.size(1) - preds.size(1)), value=label2id['O'])
            else:
                preds = torch.argmax(logits, dim=-1)
            
            if decode_samples and batch_idx < 3:
                for i in range(min(2, len(batch['tokens']))):
                    tokens = batch['tokens'][i]
                    pred_labels = preds[i][:len(tokens)].cpu().numpy()
                    entities = decode_entities(tokens, pred_labels, id2label)
                    decoded_samples.append({
                        'tokens': tokens,
                        'predicted_entities': entities,
                        'original_tags': batch['original_tags'][i]
                    })
            
            mask = (labels != label2id['O'])
            all_preds.extend(preds[mask].cpu().numpy())
            all_labels.extend(labels[mask].cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    if len(all_labels) == 0:
        return avg_loss, 0, 0, 0, decoded_samples
    
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    
    return avg_loss, f1, precision, recall, decoded_samples

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        loss, _ = model(input_ids, attention_mask, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("BERT NER Training Configuration")
    print("=" * 60)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\nLoading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    
    print("Loading datasets...")
    train_dataset = NERDataset(
        os.path.join(args.data_dir, 'train.json'),
        tokenizer,
        max_len=args.max_len
    )
    valid_dataset = NERDataset(
        os.path.join(args.data_dir, 'validation.json'),
        tokenizer,
        max_len=args.max_len,
        label2id=train_dataset.label2id
    )
    test_dataset = NERDataset(
        os.path.join(args.data_dir, 'test.json'),
        tokenizer,
        max_len=args.max_len,
        label2id=train_dataset.label2id
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of labels: {len(train_dataset.label2id)}")
    print(f"Labels: {train_dataset.label2id}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    print("\nInitializing model...")
    model = BertNERModel(len(train_dataset.label2id), args.bert_path, use_crf=args.use_crf).to(args.device)
    print(f"Using CRF layer: {args.use_crf}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    print("\nStarting training...")
    best_f1 = 0
    best_epoch = 0
    training_history = []
    
    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*50}")
        
        train_loss = train_epoch(model, train_loader, optimizer, args.device)
        print(f"Train Loss: {train_loss:.4f}")
        
        valid_loss, valid_f1, valid_precision, valid_recall, valid_decoded = evaluate(
            model, valid_loader, train_dataset.label2id, train_dataset.id2label, args.device, decode_samples=True
        )
        print(f"Valid Loss: {valid_loss:.4f}, F1: {valid_f1:.4f}, Precision: {valid_precision:.4f}, Recall: {valid_recall:.4f}")
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'valid_f1': valid_f1,
            'valid_precision': valid_precision,
            'valid_recall': valid_recall
        })
        
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_ner_model.pt'))
            print(f"New best model saved! F1: {best_f1:.4f}")
            
            with open(os.path.join(args.output_dir, 'validation_decoded_samples.json'), 'w', encoding='utf-8') as f:
                json.dump(valid_decoded, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("Loading best model for testing...")
    print("=" * 60)
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_ner_model.pt')))
    
    test_loss, test_f1, test_precision, test_recall, test_decoded = evaluate(
        model, test_loader, train_dataset.label2id, train_dataset.id2label, args.device, decode_samples=True
    )
    
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    
    with open(os.path.join(args.output_dir, 'test_decoded_samples.json'), 'w', encoding='utf-8') as f:
        json.dump(test_decoded, f, ensure_ascii=False, indent=2)
    
    results = {
        'configuration': vars(args),
        'label_mapping': {
            'label2id': train_dataset.label2id,
            'id2label': train_dataset.id2label
        },
        'training_summary': {
            'best_epoch': best_epoch,
            'best_validation_f1': best_f1
        },
        'test_results': {
            'loss': test_loss,
            'f1': test_f1,
            'precision': test_precision,
            'recall': test_recall
        },
        'training_history': training_history
    }
    
    with open(os.path.join(args.output_dir, 'training_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation F1: {best_f1:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"\nAll results saved to: {args.output_dir}")
    print(f"Best model saved to: {os.path.join(args.output_dir, 'best_ner_model.pt')}")
    print("=" * 60)

if __name__ == '__main__':
    main()
