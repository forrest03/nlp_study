import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score
import sys

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

sys.path.append('d:\\workspace\\hub-TroE\\张福\\week07\\bert_model')
from bert_ner_model import BertNERModel
from ner_dataset import NERDataset
from decoder import decode_entities, decode_batch
from pathlib import Path

# ─────────────────── 默认路径（相对于 src/ 目录）────────────────────────────
ROOT          = Path(__file__).parent.parent
DATA_DIR      = ROOT / "data"/ "peoples_daily"
BERT_PATH     = ROOT / "pretrain_models" / "bert-base-chinese"
OUTPUT_DIR    = ROOT / "outputs"
CKPT_DIR      = OUTPUT_DIR / "checkpoints"
def parse_args():
    parser = argparse.ArgumentParser(description='BERT NER Training with BIOES Labeling')
    
    parser.add_argument('--data_dir', type=str, default=str(DATA_DIR),
                        help='Directory containing data files')
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT_DIR),
                        help='Directory to save outputs')
    parser.add_argument('--max_len', type=int, default=64,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--bert_path', type=str, default=str(BERT_PATH),
                        help='Path or name of BERT model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--use_crf', action='store_true', default=False,
                        help='Whether to use CRF layer for sequence labeling')
    
    return parser.parse_args()

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
            
            # Decode predictions for first few samples
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
            
            # Only consider non-O labels for evaluation
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
    
    print("=" * 60)
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
            
            # Save decoded samples from best model
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
    
    # Save test decoded samples
    with open(os.path.join(args.output_dir, 'test_decoded_samples.json'), 'w', encoding='utf-8') as f:
        json.dump(test_decoded, f, ensure_ascii=False, indent=2)
    
    # Save comprehensive results
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