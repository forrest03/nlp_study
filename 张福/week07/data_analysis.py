import json
import os
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
data_dir = 'd:\\workspace\\hub-TroE\\张福\\week07\\data\\peoples_daily'
output_dir = 'd:\\workspace\\hub-TroE\\张福\\week07\\output'

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_dataset(data, dataset_name):
    print(f"\n=== 分析 {dataset_name} ===")
    
    # 统计类别分布
    label_counts = {}
    total_entities = 0
    text_lengths = []
    entity_lengths = []
    
    for item in data:
        text = item['text']
        text_lengths.append(len(text))
        
        entities = item.get('entities', [])
        for entity in entities:
            label = entity['label']
            start = entity['start']
            end = entity['end']
            entity_len = end - start
            
            label_counts[label] = label_counts.get(label, 0) + 1
            total_entities += 1
            entity_lengths.append(entity_len)
    
    # 统计类别个数
    num_classes = len(label_counts)
    print(f"类别个数: {num_classes}")
    print(f"总实体数: {total_entities}")
    
    # 最大文本长度
    max_len = max(text_lengths) if text_lengths else 0
    print(f"最大文本长度: {max_len}")
    
    # 平均文本长度
    avg_len = sum(text_lengths) / len(text_lengths) if text_lengths else 0
    print(f"平均文本长度: {avg_len:.2f}")
    
    # 平均实体长度
    avg_entity_len = sum(entity_lengths) / len(entity_lengths) if entity_lengths else 0
    print(f"平均实体长度: {avg_entity_len:.2f}")
    
    # 打印类别分布
    print("\n类别分布:")
    for label, count in sorted(label_counts.items()):
        percentage = (count / total_entities) * 100 if total_entities > 0 else 0
        print(f"  {label}: {count} ({percentage:.2f}%)")
    
    return {
        'label_counts': label_counts,
        'num_classes': num_classes,
        'total_entities': total_entities,
        'max_len': max_len,
        'avg_len': avg_len,
        'avg_entity_len': avg_entity_len,
        'entity_lengths': entity_lengths,
        'text_lengths': text_lengths
    }

def plot_label_distribution(label_counts, dataset_name, output_dir):
    labels = list(label_counts.keys())
    counts = list(label_counts.values())
    
    plt.figure(figsize=(12, 6))
    plt.bar(labels, counts, color='skyblue')
    plt.title(f'{dataset_name} - 类别分布')
    plt.xlabel('类别')
    plt.ylabel('数量')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 标注数值
    for i, v in enumerate(counts):
        plt.text(i, v, str(v), ha='center', va='bottom')
    
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_label_dist.png'), dpi=100)
    plt.close()

def plot_entity_length_distribution(entity_lengths, dataset_name, output_dir):
    # 统计实体长度分布
    len_counts = {}
    for length in entity_lengths:
        len_counts[length] = len_counts.get(length, 0) + 1
    
    # 取前20个最常见的长度
    sorted_lens = sorted(len_counts.items(), key=lambda x: x[0])[:20]
    lengths = [str(item[0]) for item in sorted_lens]
    counts = [item[1] for item in sorted_lens]
    
    plt.figure(figsize=(12, 6))
    plt.bar(lengths, counts, color='orange')
    plt.title(f'{dataset_name} - 实体长度分布')
    plt.xlabel('实体长度')
    plt.ylabel('数量')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_entity_length_dist.png'), dpi=100)
    plt.close()

# 读取数据集
train_data = load_json(os.path.join(data_dir, 'train.json'))
valid_data = load_json(os.path.join(data_dir, 'validation.json'))
test_data = load_json(os.path.join(data_dir, 'test.json'))

# 分析数据集
train_stats = analyze_dataset(train_data, '训练集')
valid_stats = analyze_dataset(valid_data, '验证集')
test_stats = analyze_dataset(test_data, '测试集')

# 绘制柱状图
plot_label_distribution(train_stats['label_counts'], '训练集', output_dir)
plot_label_distribution(valid_stats['label_counts'], '验证集', output_dir)
plot_label_distribution(test_stats['label_counts'], '测试集', output_dir)

plot_entity_length_distribution(train_stats['entity_lengths'], '训练集', output_dir)
plot_entity_length_distribution(valid_stats['entity_lengths'], '验证集', output_dir)
plot_entity_length_distribution(test_stats['entity_lengths'], '测试集', output_dir)

# 汇总统计信息
print("\n=== 汇总统计 ===")
print(f"训练集样本数: {len(train_data)}, 实体数: {train_stats['total_entities']}, 类别数: {train_stats['num_classes']}, 最大长度: {train_stats['max_len']}")
print(f"验证集样本数: {len(valid_data)}, 实体数: {valid_stats['total_entities']}, 类别数: {valid_stats['num_classes']}, 最大长度: {valid_stats['max_len']}")
print(f"测试集样本数: {len(test_data)}, 实体数: {test_stats['total_entities']}, 类别数: {test_stats['num_classes']}, 最大长度: {test_stats['max_len']}")

# 检查类别分布是否均匀
print("\n=== 类别分布均匀性分析 ===")
for name, stats in [('训练集', train_stats), ('验证集', valid_stats), ('测试集', test_stats)]:
    if stats['total_entities'] > 0:
        counts = list(stats['label_counts'].values())
        min_count = min(counts)
        max_count = max(counts)
        ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"{name}: 最小实体数={min_count}, 最大实体数={max_count}, 最大/最小比率={ratio:.2f}")
        if ratio < 5:
            print(f"   {name}类别分布较为均匀")
        else:
            print(f"   {name}类别分布不均匀，部分类别样本较少")

print(f"\n分析完成！结果已保存到 {output_dir}")