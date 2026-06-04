# Qwen2.5-7B-Instruct NER 训练系统

基于 Qwen2.5-7B-Instruct 大模型，使用 LoRA 进行序列标注任务的微调训练。

## 训练脚本说明

本目录包含三个训练脚本，用于区分不同的训练方式：

| 脚本 | 用途 | 输出目录 |
|------|------|----------|
| `train_qwen_ner.py` | 本地/在线模型训练（默认使用在线模型） | `outputs/llm/` |
| `train_online_api.py` | 在线模型API专用训练（LoRA微调） | `outputs/llm/online_api/` |
| `train_sft_online.py` | 在线大模型API SFT训练（Zero/One/Few-shot）- 阿里云百炼 | `outputs/llm/online/` |

## 环境依赖

```bash
pip install torch transformers peft accelerate sklearn openai
```

## 使用方法

### 1. SFT训练（Zero-shot/One-shot/Few-shot）- 阿里云百炼API

```bash
# 设置阿里云百炼API密钥
$env:DASHSCOPE_API_KEY="your-dashscope-api-key"

# Zero-shot训练（无示例）
& "D:\Python\anaconda3\envs\ai_swap\python.exe" train_sft_online.py --prompt_type zero-shot

# One-shot训练（1个示例）
& "D:\Python\anaconda3\envs\ai_swap\python.exe" train_sft_online.py --prompt_type one-shot

# Few-shot训练（多个示例，默认3个）
& "D:\Python\anaconda3\envs\ai_swap\python.exe" train_sft_online.py --prompt_type few-shot --num_examples 5
```

### 2. 在线模型API训练（LoRA微调）

```bash
# 使用默认在线模型 Qwen2.5-7B-Instruct
& "D:\Python\anaconda3\envs\ai_swap\python.exe" train_online_api.py

# 指定其他在线模型
& "D:\Python\anaconda3\envs\ai_swap\python.exe" train_online_api.py --model_name_or_path Qwen/Qwen2.5-14B-Instruct
```

### 3. 通用训练脚本

```bash
# 使用默认参数训练
& "D:\Python\anaconda3\envs\ai_swap\python.exe" train_qwen_ner.py

# 指定参数训练
& "D:\Python\anaconda3\envs\ai_swap\python.exe" train_qwen_ner.py --epochs 5 --batch_size 16 --lr 2e-4
```

## SFT训练参数说明（阿里云百炼API）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --prompt_type | str | few-shot | 提示类型：zero-shot/one-shot/few-shot |
| --num_examples | int | 3 | Few-shot示例数量 |
| --api_key | str | None | API密钥（也可通过DASHSCOPE_API_KEY环境变量设置） |
| --base_url | str | https://dashscope.aliyuncs.com/compatible-mode/v1 | API基础URL |
| --model | str | qwen3.7-plus | 模型名称（阿里云百炼） |
| --enable_thinking | bool | False | 是否启用深度思考模式 |
| --max_retries | int | 3 | API调用最大重试次数 |
| --label_scheme | str | BIOES | 标注方案：BIO或BIOES |

## LoRA训练参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --data_dir | str | data/peoples_daily | 数据集目录 |
| --output_dir | str | outputs/llm | 输出目录 |
| --model_name_or_path | str | Qwen/Qwen2.5-7B-Instruct | 模型路径（HuggingFace Hub） |
| --max_len | int | 128 | 最大序列长度 |
| --lora_rank | int | 8 | LoRA rank |
| --lora_alpha | int | 32 | LoRA alpha |
| --lora_dropout | float | 0.05 | LoRA dropout |
| --batch_size | int | 8 | 批次大小 |
| --epochs | int | 3 | 训练轮数 |
| --lr | float | 1e-4 | 学习率 |
| --fp16 | bool | True | 是否使用FP16 |

## SFT训练完整示例（阿里云百炼）

```bash
# 设置阿里云百炼API密钥
$env:DASHSCOPE_API_KEY="sk-xxx"

# Zero-shot训练
& "D:\Python\anaconda3\envs\ai_swap\python.exe" train_sft_online.py ^
    --prompt_type zero-shot ^
    --api_key $env:DASHSCOPE_API_KEY ^
    --model qwen3.7-plus

# Few-shot训练（5个示例）
& "D:\Python\anaconda3\envs\ai_swap\python.exe" train_sft_online.py ^
    --prompt_type few-shot ^
    --num_examples 5 ^
    --api_key $env:DASHSCOPE_API_KEY ^
    --enable_thinking
```

## 输出文件

### SFT训练输出 (`outputs/llm/online/`)

- `sft_results_zero-shot.json` - Zero-shot结果
- `sft_results_one-shot.json` - One-shot结果
- `sft_results_few-shot.json` - Few-shot结果

每个结果文件包含：
- 配置信息
- 标签映射
- 验证集和测试集的F1/Precision/Recall指标
- 解码后的样本数据

### LoRA训练输出 (`outputs/llm/online_api/`)

- `lora_model/` - LoRA 适配器权重
- `training_results.json` - 训练结果和配置（包含解码样本）
- `logs/` - 训练日志

### 通用训练输出 (`outputs/llm/`)

- `lora_model/` - LoRA 适配器权重
- `training_results.json` - 训练结果和配置
- `logs/` - 训练日志

## 模型结构

```
Qwen2.5-7B-Instruct (在线模型)
├── Transformer layers
│   └── LoRA adapters (q_proj, v_proj)
└── Token classification head
```

## LoRA 可训练参数

默认仅训练约 0.1% 的模型参数，大大降低显存占用：

```
trainable params: 2,621,440 || all params: 7,611,736,576 || trainable%: 0.0344
```

## SFT提示方式对比

| 方式 | 示例数量 | 适用场景 | 优点 | 缺点 |
|------|----------|----------|------|------|
| Zero-shot | 0 | 快速测试 | 无需示例，快速 | 准确率较低 |
| One-shot | 1 | 小数据集 | 提供格式参考 | 示例单一 |
| Few-shot | 多个（默认3） | 标准场景 | 准确率高 | API调用成本高 |

## 注意事项

### SFT训练（阿里云百炼API）
- 需要提供有效的阿里云百炼API密钥（DASHSCOPE_API_KEY）
- API调用会产生费用，建议先用小数据集测试
- Few-shot方式会显著增加API调用次数
- 支持的模型包括：qwen3.7-plus, qwen-turbo, qwen-max等
- 启用`--enable_thinking`可以开启深度思考模式

### LoRA训练
- 首次运行会自动从 HuggingFace Hub 下载模型（约 15GB-30GB）
- 建议使用 GPU 进行训练，显存需求约 16GB（FP16）
- 如需使用其他在线模型，可通过 `--model_name_or_path` 参数指定
- `train_online_api.py` 专为在线模型设计，包含版本兼容处理

## API调用方式说明

`train_sft_online.py` 使用 OpenAI 兼容的客户端调用阿里云百炼API：

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

completion = client.chat.completions.create(
    model="qwen3.7-plus",
    messages=[{"role": "user", "content": "你的请求"}],
    extra_body={"enable_thinking": True}
)
```