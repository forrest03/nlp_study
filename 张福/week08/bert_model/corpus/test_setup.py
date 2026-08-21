"""
BiEncoder模型测试脚本

用于验证代码结构是否正确，不依赖实际训练
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """测试导入是否成功"""
    try:
        import torch
        print("OK PyTorch导入成功")
        print(f"  PyTorch版本: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  CUDA可用: {torch.cuda.get_device_name(0)}")
        else:
            print("  CUDA不可用，将使用CPU")
        return True
    except ImportError as e:
        print(f"FAIL PyTorch导入失败: {e}")
        return False

def test_bert_imports():
    """测试transformers导入"""
    try:
        from transformers import BertTokenizer, BertModel
        print("OK Transformers导入成功")
        return True
    except ImportError as e:
        print(f"FAIL Transformers导入失败: {e}")
        return False

def test_dataset():
    """测试数据集类"""
    try:
        from dataset import BiEncoderDataset, collate_fn, load_jsonl
        print("OK Dataset模块导入成功")
        return True
    except Exception as e:
        print(f"FAIL Dataset模块测试失败: {e}")
        return False

def test_model():
    """测试模型类"""
    try:
        from model import BiEncoder
        print("OK Model模块导入成功")
        return True
    except Exception as e:
        print(f"FAIL Model模块测试失败: {e}")
        return False

def check_data_files():
    """检查数据文件是否存在"""
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'bq_corpus')
    files = ['train.jsonl', 'validation.jsonl', 'test.jsonl']
    
    all_exist = True
    for file in files:
        path = os.path.join(data_dir, file)
        if os.path.exists(path):
            # 统计行数
            with open(path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            print(f"OK {file}: {line_count}条数据")
        else:
            print(f"FAIL {file} 不存在")
            all_exist = False
    
    return all_exist

def print_summary():
    """打印代码结构摘要"""
    print("\n" + "=" * 60)
    print("BiEncoder代码结构")
    print("=" * 60)
    
    base_dir = os.path.dirname(__file__)
    files = {
        'dataset.py': '数据处理模块',
        'model.py': 'BiEncoder模型定义',
        'train.py': '训练脚本',
        'README.md': '操作说明文档'
    }
    
    for filename, description in files.items():
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"OK {filename:15s} - {description:20s} ({size} bytes)")
        else:
            print(f"FAIL {filename:15s} - {description:20s} (未找到)")
    
    print("\n使用示例:")
    print("-" * 60)
    print("基本训练:")
    print("  cd bert_model/corpus")
    print("  python train.py --data_dir ../../data/bq_corpus")
    print("\n自定义参数:")
    print("  python train.py --data_dir ../../data/bq_corpus \\")
    print("    --max_len 64 --batch_size 32 --epochs 5")
    print("\n快速训练（4层BERT）:")
    print("  python train.py --data_dir ../../data/bq_corpus \\")
    print("    --num_hidden_layers 4 --epochs 2")
    print("=" * 60)

if __name__ == '__main__':
    print("=" * 60)
    print("BiEncoder模型代码测试")
    print("=" * 60)
    
    print("\n1. 检查数据文件:")
    print("-" * 40)
    check_data_files()
    
    print("\n2. 测试模块导入:")
    print("-" * 40)
    test_dataset()
    test_model()
    
    print("\n3. 外部依赖测试:")
    print("-" * 40)
    test_imports()
    test_bert_imports()
    
    print("\n4. 代码结构:")
    print("-" * 40)
    print_summary()
    
    print("\n提示: 安装依赖后运行 python train.py 开始训练")
    print("  pip install torch transformers scikit-learn tqdm")