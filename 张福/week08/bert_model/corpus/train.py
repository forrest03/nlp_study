"""
BiEncoder文本匹配模型训练脚本

功能：
  1. 支持命令行参数配置所有训练超参数
  2. 训练、验证、测试完整流程
  3. 保存最优模型和训练结果
  4. 打印详细训练日志和预测结果

使用方式：
  python train.py --data_dir ../../data/bq_corpus --epochs 3 --batch_size 32
  
  自定义参数示例：
  python train.py --max_len 64 --learning_rate 2e-5 --epochs 5 --pool mean
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataset import BiEncoderDataset, collate_fn, load_jsonl
from model import BiEncoder


class CosineEmbeddingLoss(nn.Module):
    """余弦嵌入损失"""
    
    def __init__(self, margin=0.3):
        super(CosineEmbeddingLoss, self).__init__()
        self.margin = margin
        
    def forward(self, similarity, labels):
        # 正例对的损失：1 - similarity
        # 负例对的损失：max(0, similarity - margin)
        loss_pos = (1 - similarity) * labels
        loss_neg = torch.clamp(similarity - self.margin, min=0) * (1 - labels)
        loss = loss_pos + loss_neg
        return loss.mean()


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        optimizer.zero_grad()
        
        # 数据移动到设备
        input_ids1 = batch['input_ids1'].to(device)
        attention_mask1 = batch['attention_mask1'].to(device)
        input_ids2 = batch['input_ids2'].to(device)
        attention_mask2 = batch['attention_mask2'].to(device)
        labels = batch['label'].to(device)
        
        # 前向传播
        similarity = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
        
        # 计算损失
        loss = criterion(similarity, labels)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # 记录预测
        preds = (similarity > 0.5).float()
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    all_similarities = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids1 = batch['input_ids1'].to(device)
            attention_mask1 = batch['attention_mask1'].to(device)
            input_ids2 = batch['input_ids2'].to(device)
            attention_mask2 = batch['attention_mask2'].to(device)
            labels = batch['label'].to(device)
            
            similarity = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            loss = criterion(similarity, labels)
            
            total_loss += loss.item()
            
            preds = (similarity > 0.5).float()
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            all_similarities.extend(similarity.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'similarities': all_similarities
    }
    
    return metrics


def predict(model, dataloader, device):
    """在测试集上进行预测"""
    model.eval()
    predictions = []
    true_labels = []
    all_similarities = []
    decoded_results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Predicting'):
            input_ids1 = batch['input_ids1'].to(device)
            attention_mask1 = batch['attention_mask1'].to(device)
            input_ids2 = batch['input_ids2'].to(device)
            attention_mask2 = batch['attention_mask2'].to(device)
            labels = batch['label'].to(device)
            
            similarity = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            
            preds = (similarity > 0.5).float()
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            all_similarities.extend(similarity.cpu().numpy())
            
            # 解码并保存结果
            for i in range(len(preds)):
                decoded_results.append({
                    'similarity': float(similarity[i].cpu()),
                    'prediction': int(preds[i].cpu()),
                    'label': int(labels[i].cpu())
                })
    
    return predictions, true_labels, all_similarities, decoded_results


def main(args):
    """主训练函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载分词器
    print(f"加载分词器: {args.bert_model}")
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    
    # 创建数据集
    print(f"加载数据集...")
    train_dataset = BiEncoderDataset(
        os.path.join(args.data_dir, 'train.jsonl'),
        tokenizer,
        max_length=args.max_len
    )
    val_dataset = BiEncoderDataset(
        os.path.join(args.data_dir, 'validation.jsonl'),
        tokenizer,
        max_length=args.max_len
    )
    test_dataset = BiEncoderDataset(
        os.path.join(args.data_dir, 'test.jsonl'),
        tokenizer,
        max_length=args.max_len
    )
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # 创建模型
    print(f"创建BiEncoder模型...")
    model = BiEncoder(
        bert_model_path=args.bert_model,
        pool=args.pool,
        dropout=args.dropout,
        num_hidden_layers=args.num_hidden_layers
    )
    model.to(device)
    
    # 定义损失函数
    criterion = CosineEmbeddingLoss(margin=args.margin)
    
    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # 学习率调度器
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    # 训练循环
    best_val_f1 = 0.0
    train_history = []
    val_history = []
    
    print("\n开始训练...")
    print("=" * 80)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        
        # 验证
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # 打印训练结果
        print(f"\n训练结果:")
        print(f"  Loss: {train_loss:.4f}")
        print(f"  Accuracy: {train_acc:.4f}")
        
        print(f"\n验证结果:")
        print(f"  Loss: {val_metrics['loss']:.4f}")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall: {val_metrics['recall']:.4f}")
        print(f"  F1: {val_metrics['f1']:.4f}")
        
        # 记录历史
        train_history.append({
            'epoch': epoch + 1,
            'loss': train_loss,
            'accuracy': train_acc
        })
        val_history.append({
            'epoch': epoch + 1,
            'loss': val_metrics['loss'],
            'accuracy': val_metrics['accuracy'],
            'precision': val_metrics['precision'],
            'recall': val_metrics['recall'],
            'f1': val_metrics['f1']
        })
        
        # 保存最优模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            model_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'args': vars(args)
            }, model_path)
            print(f"✓ 保存最优模型到 {model_path}")
    
    # 加载最优模型进行测试
    print("\n" + "=" * 80)
    print("加载最优模型进行测试...")
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 测试集评估
    print("\n测试集评估:")
    print("-" * 40)
    test_predictions, test_labels, test_similarities, test_decoded = predict(
        model, test_loader, device
    )
    
    test_accuracy = accuracy_score(test_labels, test_predictions)
    test_precision = precision_score(test_labels, test_predictions, zero_division=0)
    test_recall = recall_score(test_labels, test_predictions, zero_division=0)
    test_f1 = f1_score(test_labels, test_predictions, zero_division=0)
    
    print(f"测试集结果:")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1: {test_f1:.4f}")
    
    # 保存预测结果
    results = {
        'test_metrics': {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1
        },
        'predictions': test_decoded,
        'similarity_distribution': {
            'mean': float(torch.tensor(test_similarities).mean()),
            'std': float(torch.tensor(test_similarities).std()),
            'min': float(torch.tensor(test_similarities).min()),
            'max': float(torch.tensor(test_similarities).max())
        }
    }
    
    # 保存结果
    results_path = output_dir / 'test_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 保存测试结果到 {results_path}")
    
    # 打印部分预测结果
    print("\n预测结果示例（前10条）:")
    print("-" * 40)
    for i, result in enumerate(test_decoded[:10]):
        pred_label = "正确" if result['prediction'] == 1 else "错误"
        true_label = "正确" if result['label'] == 1 else "错误"
        status = "✓" if result['prediction'] == result['label'] else "✗"
        print(f"{status} 相似度: {result['similarity']:.4f}, 预测: {pred_label}, 实际: {true_label}")
    
    # 保存训练历史
    history = {
        'train_history': train_history,
        'val_history': val_history,
        'best_val_f1': best_val_f1,
        'test_metrics': results['test_metrics']
    }
    
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 保存训练历史到 {history_path}")
    
    # 保存配置信息
    config_path = output_dir / 'config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    print(f"✓ 保存配置信息到 {config_path}")
    
    print("\n" + "=" * 80)
    print("训练完成！")
    print(f"最优验证F1: {best_val_f1:.4f}")
    print(f"测试集F1: {test_f1:.4f}")
    print(f"所有结果保存在: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BiEncoder文本匹配训练')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, 
                       default='../../data/bq_corpus',
                       help='数据目录路径')
    parser.add_argument('--max_len', type=int, default=64, 
                       help='最大序列长度')
    
    # 模型参数
    parser.add_argument('--bert_model', type=str, 
                       default='../../pretrain_models/bert-base-chinese',
                       help='预训练模型路径')
    parser.add_argument('--pool', type=str, default='mean', 
                       choices=['cls', 'mean', 'max'],
                       help='池化方式')
    parser.add_argument('--dropout', type=float, default=0.1, 
                       help='Dropout比例')
    parser.add_argument('--num_hidden_layers', type=int, default=None, 
                       help='BERT层数，None表示使用全部层')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='批处理大小')
    parser.add_argument('--epochs', type=int, default=3, 
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=2e-5, 
                       help='学习率')
    parser.add_argument('--margin', type=float, default=0.3, 
                       help='余弦损失margin')
    parser.add_argument('--weight_decay', type=float, default=0.01, 
                       help='权重衰减')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='../../outputs',
                       help='输出目录')
    
    args = parser.parse_args()
    
    main(args)