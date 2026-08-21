import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from pathlib import Path

print("1. Starting Online LLM API-based NER Training with LoRA...")

def parse_args():
    parser = argparse.ArgumentParser(description='Online LLM API NER Training with LoRA')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default=str(Path(__file__).parent.parent / "data" / "peoples_daily"))
    parser.add_argument('--output_dir', type=str, default=str(Path(__file__).parent.parent / "outputs" / "llm" / "online_api"))
    
    # 在线模型参数（从HuggingFace Hub下载）
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='Online model from HuggingFace Hub')
    parser.add_argument('--max_len', type=int, default=128)
    
    # LoRA 参数
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--lora_target_modules', type=str, default='q_proj,v_proj', 
                        help='Target modules for LoRA')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--eval_steps', type=int, default=50)
    
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--fp16', action='store_true', default=True, help='Use FP16 training')
    parser.add_argument('--bf16', action='store_true', default=False, help='Use BF16 training')
    
    return parser.parse_args()

class NERDataset(Dataset):
    """NER数据集类 - 适配在线模型"""
    def __init__(self, data_path, tokenizer, max_len=128, label2id=None):
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        if label2id is None:
            self.label2id = self._build_label_vocab()
        else:
            self.label2id = label2id
        
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.num_labels = len(self.label2id)
    
    def _load_data(self, data_path):
        """加载JSON数据"""
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _build_label_vocab(self):
        """构建标签词汇表"""
        labels = set(['O'])
        for item in self.data:
            for tag in item['ner_tags']:
                if tag not in labels:
                    labels.add(tag)
        return {label: idx for idx, label in enumerate(sorted(labels))}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        item = self.data[idx]
        tokens = item['tokens']
        tags = item['ner_tags']
        
        # 使用在线模型的tokenizer进行分词
        encoded = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True
        )
        
        # 对齐标签（处理子词情况）
        labels = []
        word_ids = encoded.word_ids()
        
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)  # 特殊token用-100忽略
            elif word_idx != previous_word_idx:
                if word_idx < len(tags):
                    labels.append(self.label2id.get(tags[word_idx], self.label2id['O']))
                else:
                    labels.append(self.label2id['O'])
                previous_word_idx = word_idx
            else:
                labels.append(-100)  # 子词用-100忽略
        
        return {
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'tokens': tokens,
            'original_tags': tags
        }

def compute_metrics(p):
    """计算评估指标"""
    predictions, labels = p
    
    # 获取预测标签（忽略-100）
    predictions = np.argmax(predictions, axis=2)
    
    # 过滤掉-100的标签
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]
    
    if len(labels) == 0:
        return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    f1 = f1_score(labels, predictions, average='macro')
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def decode_entities(tokens, pred_labels, id2label):
    """解码实体（支持BIO/BIOES标注）"""
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
        elif label.startswith('S-'):
            entities.append({
                'text': token,
                'label': label[2:],
                'start': i,
                'end': i + 1
            })
        elif label == 'O':
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    if current_entity:
        entities.append(current_entity)
    
    return entities

def evaluate_and_decode(model, dataloader, label2id, id2label, tokenizer, device):
    """评估并解码样本"""
    model.eval()
    decoded_samples = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            if batch_idx < 5:  # 只解码前5个batch的样本
                for i in range(min(3, len(batch['tokens']))):
                    tokens = batch['tokens'][i]
                    pred_labels = predictions[i].cpu().numpy()
                    
                    # 找到有效的预测标签（跳过特殊token和子词）
                    word_ids = tokenizer(
                        tokens,
                        is_split_into_words=True,
                        max_length=args.max_len,
                        padding='max_length',
                        truncation=True
                    ).word_ids()
                    
                    valid_preds = []
                    prev_word_idx = None
                    for word_idx, pred in zip(word_ids, pred_labels):
                        if word_idx is not None and word_idx != prev_word_idx:
                            valid_preds.append(pred)
                            prev_word_idx = word_idx
                    
                    # 截断到原始token长度
                    valid_preds = valid_preds[:len(tokens)]
                    
                    entities = decode_entities(tokens, valid_preds, id2label)
                    decoded_samples.append({
                        'tokens': tokens,
                        'predicted_entities': entities,
                        'original_tags': batch['original_tags'][i]
                    })
    
    return decoded_samples

def main():
    global args
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("Online LLM API NER Training Configuration")
    print("=" * 60)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\nLoading online tokenizer from HuggingFace Hub...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # 加载数据集
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
    print(f"Number of labels: {train_dataset.num_labels}")
    print(f"Labels: {train_dataset.label2id}")
    
    # 配置LoRA
    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[m.strip() for m in args.lora_target_modules.split(',')],
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type='TOKEN_CLS',
    )
    
    # 加载在线模型（从HuggingFace Hub下载）
    print(f"\nLoading online model: {args.model_name_or_path}...")
    print("This may take a few minutes to download...")
    
    torch_dtype = torch.float16 if args.fp16 else torch.bfloat16 if args.bf16 else torch.float32
    
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=train_dataset.num_labels,
        torch_dtype=torch_dtype,
        device_map='auto',
        trust_remote_code=True
    )
    
    # 应用LoRA
    print("Applying LoRA adapters...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 训练参数（兼容不同transformers版本）
    print("\nSetting up TrainingArguments...")
    training_args_dict = {
        'output_dir': args.output_dir,
        'per_device_train_batch_size': args.batch_size,
        'per_device_eval_batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'num_train_epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'logging_dir': os.path.join(args.output_dir, 'logs'),
        'logging_steps': args.logging_steps,
        'eval_steps': args.eval_steps,
        'eval_strategy': 'steps',
        'save_strategy': 'steps',
        'save_steps': args.eval_steps,
        'load_best_model_at_end': True,
        'fp16': args.fp16,
        'bf16': args.bf16,
        'report_to': 'none',
        'metric_for_best_model': 'f1',
        'greater_is_better': True,
    }
    
    # 动态过滤参数以兼容不同版本
    import inspect
    sig = inspect.signature(TrainingArguments.__init__)
    valid_args = {k: v for k, v in training_args_dict.items() if k in sig.parameters}
    
    training_args = TrainingArguments(**valid_args)
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    
    # 开始训练
    print("\nStarting training with online model...")
    trainer.train()
    
    # 保存LoRA权重
    print("\nSaving LoRA model...")
    model.save_pretrained(os.path.join(args.output_dir, 'lora_model'))
    
    # 在测试集上评估
    print("\nEvaluating on test set...")
    test_results = trainer.predict(test_dataset)
    print(f"Test Results: {test_results.metrics}")
    
    # 解码样本
    print("Decoding test samples...")
    test_decoded = evaluate_and_decode(
        model, 
        DataLoader(test_dataset, batch_size=args.batch_size),
        train_dataset.label2id,
        train_dataset.id2label,
        tokenizer,
        args.device
    )
    
    # 保存验证集解码结果
    print("Decoding validation samples...")
    valid_decoded = evaluate_and_decode(
        model,
        DataLoader(valid_dataset, batch_size=args.batch_size),
        train_dataset.label2id,
        train_dataset.id2label,
        tokenizer,
        args.device
    )
    
    # 保存所有结果
    results = {
        'configuration': vars(args),
        'label_mapping': {
            'label2id': train_dataset.label2id,
            'id2label': train_dataset.id2label
        },
        'training_results': trainer.state.log_history,
        'test_results': test_results.metrics,
        'validation_decoded_samples': valid_decoded,
        'test_decoded_samples': test_decoded
    }
    
    with open(os.path.join(args.output_dir, 'training_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n" + "=" * 60)
    print("Training Summary (Online Model API)")
    print("=" * 60)
    print(f"Model: {args.model_name_or_path}")
    print(f"Best Epoch: {trainer.state.best_epoch}")
    print(f"Best Validation F1: {trainer.state.best_metric}")
    print(f"Test F1: {test_results.metrics.get('test_f1', 'N/A')}")
    print(f"Test Precision: {test_results.metrics.get('test_precision', 'N/A')}")
    print(f"Test Recall: {test_results.metrics.get('test_recall', 'N/A')}")
    print(f"\nAll results saved to: {args.output_dir}")
    print(f"LoRA model saved to: {os.path.join(args.output_dir, 'lora_model')}")
    print("=" * 60)

if __name__ == '__main__':
    main()
