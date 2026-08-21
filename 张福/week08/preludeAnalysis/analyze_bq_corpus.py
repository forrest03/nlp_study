import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

def read_jsonl(file_path):
    """读取JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def analyze_data(data, dataset_name):
    """分析数据集"""
    # 1. 统计类别分布
    labels = [item['label'] for item in data]
    label_counts = Counter(labels)
    
    # 2. 统计类别个数
    num_classes = len(label_counts)
    
    # 3. 找到最大长度
    sentence1_lengths = [len(item['sentence1']) for item in data]
    sentence2_lengths = [len(item['sentence2']) for item in data]
    max_len = max(max(sentence1_lengths), max(sentence2_lengths))
    
    # 4. 实体长度分布
    all_lengths = sentence1_lengths + sentence2_lengths
    
    return {
        'dataset_name': dataset_name,
        'label_distribution': label_counts,
        'num_classes': num_classes,
        'max_len': max_len,
        'length_distribution': all_lengths,
        'sentence1_lengths': sentence1_lengths,
        'sentence2_lengths': sentence2_lengths,
        'total_samples': len(data)
    }

def plot_class_distribution(results, output_dir):
    """绘制类别分布图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    datasets = ['train', 'validation', 'test']
    for idx, (result, dataset) in enumerate(zip(results, datasets)):
        ax = axes[idx]
        labels = list(result['label_distribution'].keys())
        counts = list(result['label_distribution'].values())
        
        bars = ax.bar(labels, counts, color=['#3498db', '#e74c3c'])
        ax.set_xlabel('类别')
        ax.set_ylabel('样本数量')
        ax.set_title(f'{dataset} 数据集类别分布')
        ax.set_xticks(labels)
        
        # 在柱子上显示数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height}',
                   ha='center', va='bottom')
        
        # 显示总数
        ax.text(0.5, 0.95, f'总数: {result["total_samples"]}', 
               transform=ax.transAxes, ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("类别分布图已保存到 output/class_distribution.png")

def plot_length_distribution(results, output_dir):
    """绘制长度分布图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    datasets = ['train', 'validation', 'test']
    for idx, (result, dataset) in enumerate(zip(results, datasets)):
        ax = axes[idx]
        lengths = result['length_distribution']
        
        # 创建直方图
        n, bins, patches = ax.hist(lengths, bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('句子长度（字符数）')
        ax.set_ylabel('频数')
        ax.set_title(f'{dataset} 数据集句子长度分布')
        
        # 显示统计信息
        mean_len = np.mean(lengths)
        median_len = np.median(lengths)
        max_len = result['max_len']
        
        stats_text = f'平均长度: {mean_len:.1f}\n中位数: {median_len:.1f}\n最大长度: {max_len}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
               ha='right', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'length_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("长度分布图已保存到 output/length_distribution.png")

def plot_comparison(results, output_dir):
    """绘制对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    datasets = ['train', 'validation', 'test']
    
    # 1. 类别分布对比
    ax1 = axes[0, 0]
    x = np.arange(len(datasets))
    width = 0.35
    
    class_0_counts = [result['label_distribution'].get(0, 0) for result in results]
    class_1_counts = [result['label_distribution'].get(1, 0) for result in results]
    
    bars1 = ax1.bar(x - width/2, class_0_counts, width, label='类别 0', color='#e74c3c')
    bars2 = ax1.bar(x + width/2, class_1_counts, width, label='类别 1', color='#3498db')
    
    ax1.set_xlabel('数据集')
    ax1.set_ylabel('样本数量')
    ax1.set_title('各数据集类别分布对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend()
    
    # 2. 样本总数对比
    ax2 = axes[0, 1]
    total_samples = [result['total_samples'] for result in results]
    bars = ax2.bar(datasets, total_samples, color=['#9b59b6', '#1abc9c', '#f39c12'])
    ax2.set_xlabel('数据集')
    ax2.set_ylabel('样本总数')
    ax2.set_title('各数据集样本总数对比')
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}',
                ha='center', va='bottom')
    
    # 3. 最大长度对比
    ax3 = axes[1, 0]
    max_lengths = [result['max_len'] for result in results]
    bars = ax3.bar(datasets, max_lengths, color=['#e67e22', '#16a085', '#c0392b'])
    ax3.set_xlabel('数据集')
    ax3.set_ylabel('最大长度（字符数）')
    ax3.set_title('各数据集最大句子长度对比')
    
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}',
                ha='center', va='bottom')
    
    # 4. 平均长度对比
    ax4 = axes[1, 1]
    mean_lengths = [np.mean(result['length_distribution']) for result in results]
    bars = ax4.bar(datasets, mean_lengths, color=['#8e44ad', '#27ae60', '#d35400'])
    ax4.set_xlabel('数据集')
    ax4.set_ylabel('平均长度（字符数）')
    ax4.set_title('各数据集平均句子长度对比')
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("对比图已保存到 output/comparison.png")

def generate_summary(results, output_dir):
    """生成统计摘要"""
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("BQ Corpus 数据集统计分析报告")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    datasets = ['train', 'validation', 'test']
    for result, dataset in zip(results, datasets):
        summary_lines.append(f"【{dataset.upper()} 数据集】")
        summary_lines.append("-" * 40)
        summary_lines.append(f"样本总数: {result['total_samples']}")
        summary_lines.append(f"类别数量: {result['num_classes']}")
        summary_lines.append(f"最大长度: {result['max_len']}")
        
        # 类别分布
        summary_lines.append("\n类别分布:")
        for label, count in sorted(result['label_distribution'].items()):
            percentage = (count / result['total_samples']) * 100
            summary_lines.append(f"  类别 {label}: {count} ({percentage:.2f}%)")
        
        # 长度统计
        all_lengths = result['length_distribution']
        summary_lines.append(f"\n长度统计:")
        summary_lines.append(f"  平均长度: {np.mean(all_lengths):.2f}")
        summary_lines.append(f"  中位数长度: {np.median(all_lengths):.2f}")
        summary_lines.append(f"  标准差: {np.std(all_lengths):.2f}")
        summary_lines.append(f"  最小长度: {np.min(all_lengths)}")
        summary_lines.append("")
    
    # 保存摘要
    summary_path = os.path.join(output_dir, 'analysis_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"统计摘要已保存到 {summary_path}")
    print('\n' + '\n'.join(summary_lines))

def main():
    # 设置路径
    base_dir = r'/张福/week08/data/bq_corpus'
    output_dir = r'/张福/week08/output'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    print("正在读取数据...")
    train_data = read_jsonl(os.path.join(base_dir, 'train.jsonl'))
    validation_data = read_jsonl(os.path.join(base_dir, 'validation.jsonl'))
    test_data = read_jsonl(os.path.join(base_dir, 'test.jsonl'))
    
    print(f"训练集样本数: {len(train_data)}")
    print(f"验证集样本数: {len(validation_data)}")
    print(f"测试集样本数: {len(test_data)}")
    
    # 分析数据
    print("\n正在分析数据...")
    train_result = analyze_data(train_data, 'train')
    validation_result = analyze_data(validation_data, 'validation')
    test_result = analyze_data(test_data, 'test')
    
    results = [train_result, validation_result, test_result]
    
    # 生成图表
    print("\n正在生成图表...")
    plot_class_distribution(results, output_dir)
    plot_length_distribution(results, output_dir)
    plot_comparison(results, output_dir)
    
    # 生成摘要
    print("\n正在生成统计摘要...")
    generate_summary(results, output_dir)
    
    print("\n分析完成！所有结果已保存到 output 目录")

if __name__ == "__main__":
    main()