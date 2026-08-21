"""
LCQMC BiEncoder文本匹配训练脚本

完整功能：
  1. 训练、验证、测试完整流程
  2. 支持LoRA高效微调
  3. 支持命令行参数配置
  4. 保存最优模型和详细结果
  5. 解码并保存预测结果
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

from dataset import LCQMCDataset, collate_fn


class CosineEmbeddingLoss(nn.Module):
    """余弦嵌入损失"""
    
    def __init__(self, margin=0.3):
        super(CosineEmbeddingLoss, self).__init__()
        self.margin = margin
        
    def forward(self, similarity, labels):
        loss_pos = (1 - similarity) * labels
        loss_neg = torch.clamp(similarity - self.margin, min=0) * (1 - labels)
        loss = loss_pos + loss_neg
        return loss.mean()


class BiEncoder(nn.Module):
    """BiEncoder文本匹配模型"""
    
    def __init__(self, bert_model_path='bert-base-chinese', 
                 pool='mean', dropout=0.1, num_hidden_layers=None,
                 use_lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        super(BiEncoder, self).__init__()
        
        from transformers import BertModel, BertConfig
        
        if num_hidden_layers is not None:
            config = BertConfig.from_pretrained(bert_model_path)
            config.num_hidden_layers = num_hidden_layers
            self.bert = BertModel.from_pretrained(bert_model_path, config=config)
        else:
            self.bert = BertModel.from_pretrained(bert_model_path)
        
        self.pool = pool
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = self.bert.config.hidden_size
        self.use_lora = use_lora
        
        # LoRA配置
        if use_lora:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["query", "value"]  # 只在attention的query和value上应用LoRA
            )
            self.bert = get_peft_model(self.bert, lora_config)
            print(f"LoRA配置: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
            # 统计可训练参数
            trainable_params, all_params = self._count_parameters()
            print(f"可训练参数: {trainable_params:,} / {all_params:,} ({trainable_params/all_params*100:.2f}%)")
        
    def _count_parameters(self):
        """统计参数数量"""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.parameters())
        return trainable_params, all_params
        
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
        
        encoded1 = torch.nn.functional.normalize(encoded1, p=2, dim=1)
        encoded2 = torch.nn.functional.normalize(encoded2, p=2, dim=1)
        
        similarity = (encoded1 * encoded2).sum(dim=1)
        
        return similarity


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        optimizer.zero_grad()
        
        input_ids1 = batch['input_ids1'].to(device)
        attention_mask1 = batch['attention_mask1'].to(device)
        input_ids2 = batch['input_ids2'].to(device)
        attention_mask2 = batch['attention_mask2'].to(device)
        labels = batch['label'].to(device)
        
        similarity = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
        loss = criterion(similarity, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
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
            
            for i in range(len(preds)):
                decoded_results.append({
                    'similarity': float(similarity[i].cpu()),
                    'prediction': int(preds[i].cpu()),
                    'label': int(labels[i].cpu()),
                    'prediction_label': '相似' if int(preds[i].cpu()) == 1 else '不相似',
                    'true_label': '相似' if int(labels[i].cpu()) == 1 else '不相似',
                    'correct': bool(preds[i].cpu() == labels[i].cpu())
                })
    
    return predictions, true_labels, all_similarities, decoded_results


def main(args):
    """主训练函数"""
    print("=" * 80)
    print("LCQMC BiEncoder文本匹配训练")
    print("=" * 80)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置路径
    base_dir = Path(__file__).parent
    data_dir = base_dir / '..' / '..' / 'data' / 'lcqmc'
    output_dir = base_dir / 'outputs'
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 加载分词器
    print(f"\n加载分词器: {args.bert_model}")
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    
    # 创建数据集
    print("\n加载数据集...")
    train_dataset = LCQMCDataset(
        str(data_dir / 'train.jsonl'),
        tokenizer,
        max_length=args.max_len
    )
    val_dataset = LCQMCDataset(
        str(data_dir / 'validation.jsonl'),
        tokenizer,
        max_length=args.max_len
    )
    test_dataset = LCQMCDataset(
        str(data_dir / 'test.jsonl'),
        tokenizer,
        max_length=args.max_len
    )
    
    # 随机采样数据集
    if args.sample_ratio < 1.0:
        print(f"\n随机采样数据集 (采样比例: {args.sample_ratio})...")
        import random
        random.seed(args.seed)  # 设置随机种子保证可复现性
        
        # 保存原始数据大小
        original_train_size = len(train_dataset)
        original_val_size = len(val_dataset)
        original_test_size = len(test_dataset)
        
        # 训练集采样
        train_size = int(len(train_dataset) * args.sample_ratio)
        train_indices = random.sample(range(len(train_dataset)), train_size)
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        
        # 验证集采样
        val_size = int(len(val_dataset) * args.sample_ratio)
        val_indices = random.sample(range(len(val_dataset)), val_size)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        
        # 测试集采样
        test_size = int(len(test_dataset) * args.sample_ratio)
        test_indices = random.sample(range(len(test_dataset)), test_size)
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
        
        print(f"采样后训练集: {len(train_dataset)}条 (原始: {original_train_size}条)")
        print(f"采样后验证集: {len(val_dataset)}条 (原始: {original_val_size}条)")
        print(f"采样后测试集: {len(test_dataset)}条 (原始: {original_test_size}条)")
    else:
        print(f"训练集: {len(train_dataset)}条")
        print(f"验证集: {len(val_dataset)}条")
        print(f"测试集: {len(test_dataset)}条")
    
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
    print("\n创建BiEncoder模型...")
    model = BiEncoder(
        bert_model_path=args.bert_model,
        pool=args.pool,
        dropout=args.dropout,
        num_hidden_layers=args.num_hidden_layers,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    model.to(device)
    
    # 定义损失函数和优化器
    criterion = CosineEmbeddingLoss(margin=args.margin)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
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
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        
        # 验证
        val_metrics = evaluate(model, val_loader, criterion, device)
        
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
        train_history.append({'epoch': epoch + 1, 'loss': train_loss, 'accuracy': train_acc})
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
            model_path = output_dir / 'lcqmq_best_model.pt'
            
            if args.use_lora:
                # 保存LoRA适配器（兼容不同版本的PEFT）
                model.bert.save_pretrained(output_dir / 'lora_adapter')
                # 保存训练元数据
                torch.save({
                    'epoch': epoch + 1,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': best_val_f1,
                    'args': vars(args)
                }, output_dir / 'training_metadata.pt')
                print(f"OK 保存最优LoRA适配器到 {output_dir / 'lora_adapter'}")
            else:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': best_val_f1,
                    'args': vars(args)
                }, model_path)
                print(f"OK 保存最优模型到 {model_path}")
    
    # 加载最优模型进行测试
    print("\n" + "=" * 80)
    print("加载最优模型进行测试...")
    
    if args.use_lora:
        # 加载LoRA适配器
        from peft import PeftModel
        model.bert = PeftModel.from_pretrained(model.bert, str(output_dir / 'lora_adapter'))
        print("OK 加载LoRA适配器")
    else:
        checkpoint = torch.load(output_dir / 'best_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("OK 加载模型权重")
    
    # 测试集评估
    print("\n测试集评估:")
    print("-" * 40)
    test_predictions, test_labels, test_similarities, test_decoded = predict(model, test_loader, device)
    
    test_accuracy = accuracy_score(test_labels, test_predictions)
    test_precision = precision_score(test_labels, test_predictions, zero_division=0)
    test_recall = recall_score(test_labels, test_predictions, zero_division=0)
    test_f1 = f1_score(test_labels, test_predictions, zero_division=0)
    
    print(f"测试结果:")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1: {test_f1:.4f}")
    
    # 保存测试结果
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
    
    with open(output_dir / 'test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nOK 保存测试结果到 {output_dir / 'test_results.json'}")
    
    # 打印预测示例
    print("\n预测结果示例（前15条）:")
    print("-" * 80)
    print(f"{'序号':<4} {'相似度':<10} {'预测':<6} {'实际':<6} {'是否正确'}")
    print("-" * 80)
    for i, result in enumerate(test_decoded[:15]):
        status = "OK" if result['correct'] else "FAIL"
        print(f"{i+1:<4} {result['similarity']:<10.4f} {result['prediction_label']:<6} {result['true_label']:<6} {status}")
    
    # 保存训练历史
    history = {
        'train_history': train_history,
        'val_history': val_history,
        'best_val_f1': best_val_f1,
        'test_metrics': results['test_metrics']
    }
    
    with open(output_dir / 'training_history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"\nOK 保存训练历史到 {output_dir / 'training_history.json'}")
    
    # 保存配置信息
    with open(output_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    print(f"OK 保存配置信息到 {output_dir / 'config.json'}")
    
    # 生成详细报告
    generate_report(output_dir, history, results, args)
    
    print("\n" + "=" * 80)
    print("训练完成！")
    print(f"最优验证F1: {best_val_f1:.4f}")
    print(f"测试集F1: {test_f1:.4f}")
    print(f"所有结果保存在: {output_dir}")


def generate_report(output_dir, history, results, args):
    """生成详细报告"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("LCQMC数据集 - BiEncoder训练报告")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    report_lines.append("【配置参数】")
    report_lines.append("-" * 40)
    report_lines.append(f"最大序列长度: {args.max_len}")
    report_lines.append(f"批处理大小: {args.batch_size}")
    report_lines.append(f"训练轮数: {args.epochs}")
    report_lines.append(f"学习率: {args.learning_rate}")
    report_lines.append(f"池化方式: {args.pool}")
    report_lines.append(f"Dropout: {args.dropout}")
    report_lines.append(f"Margin: {args.margin}")
    report_lines.append("")
    
    report_lines.append("【训练历史】")
    report_lines.append("-" * 40)
    report_lines.append(f"{'Epoch':<6} {'训练Loss':<12} {'训练Acc':<10} {'验证Loss':<12} {'验证F1':<10}")
    report_lines.append("-" * 50)
    for train, val in zip(history['train_history'], history['val_history']):
        report_lines.append(f"{train['epoch']:<6} {train['loss']:<12.4f} {train['accuracy']:<10.4f} {val['loss']:<12.4f} {val['f1']:<10.4f}")
    report_lines.append("")
    
    report_lines.append("【测试结果】")
    report_lines.append("-" * 40)
    metrics = results['test_metrics']
    report_lines.append(f"准确率: {metrics['accuracy']:.4f}")
    report_lines.append(f"精确率: {metrics['precision']:.4f}")
    report_lines.append(f"召回率: {metrics['recall']:.4f}")
    report_lines.append(f"F1分数: {metrics['f1']:.4f}")
    report_lines.append("")
    
    report_lines.append("【相似度分布】")
    report_lines.append("-" * 40)
    dist = results['similarity_distribution']
    report_lines.append(f"平均值: {dist['mean']:.4f}")
    report_lines.append(f"标准差: {dist['std']:.4f}")
    report_lines.append(f"最小值: {dist['min']:.4f}")
    report_lines.append(f"最大值: {dist['max']:.4f}")
    report_lines.append("")
    
    report_lines.append("【预测示例】")
    report_lines.append("-" * 40)
    report_lines.append(f"{'相似度':<10} {'预测':<6} {'实际':<6} {'是否正确'}")
    report_lines.append("-" * 35)
    for result in results['predictions'][:10]:
        status = "OK" if result['correct'] else "FAIL"
        report_lines.append(f"{result['similarity']:<10.4f} {result['prediction_label']:<6} {result['true_label']:<6} {status}")
    
    with open(output_dir / 'training_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"OK 保存训练报告到 {output_dir / 'training_report.txt'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LCQMC BiEncoder文本匹配训练')
    
    # 数据参数
    parser.add_argument('--max_len', type=int, default=64, help='最大序列长度')
    parser.add_argument('--sample_ratio', type=float, default=0.03, help='数据集随机采样比例（0.0-1.0），默认0.3')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，用于保证实验可复现性')
    
    # 模型参数
    parser.add_argument('--bert_model', type=str, default='../../pretrain_models/bert-base-chinese', help='预训练模型')
    parser.add_argument('--pool', type=str, default='mean', choices=['cls', 'mean', 'max'], help='池化方式')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout比例')
    parser.add_argument('--num_hidden_layers', type=int, default=None, help='BERT层数')
    
    # LoRA参数
    parser.add_argument('--use_lora', action='store_true', default=True, help='使用LoRA微调')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA秩rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha参数')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout比例')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--epochs', type=int, default=1, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率')
    parser.add_argument('--margin', type=float, default=0.3, help='余弦损失margin')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    
    args = parser.parse_args()
    
    main(args)