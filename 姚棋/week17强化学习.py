import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
import re

# ============ 1. 配置 ============
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"  # 也可用 "Qwen/Qwen2.5-0.5B-Instruct" 以降低显存需求
OUTPUT_DIR = "./grpo_math_model"
NUM_GENERATIONS = 8  # 每个 prompt 采样数量 (G)
LEARNING_RATE = 1e-5
MAX_COMPLETION_LENGTH = 512

# ============ 2. 加载模型和 Tokenizer ============
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
# 设置 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============ 3. 加载数据集 ============
def format_prompt(example):
    return {
        "prompt": [
            {"role": "system", "content": "You are a helpful math assistant. Solve the problem step by step and put your final answer inside <answer> tags."},
            {"role": "user", "content": example["question"]}
        ],
        "answer": re.search(r'####\s*([\d,]+\.?\d*)', example["answer"]).group(1).replace(',', '')
    }

dataset = load_dataset("openai/gsm8k", "main")
train_dataset = dataset["train"].map(format_prompt)
train_dataset = train_dataset.filter(lambda x: x["answer"] is not None)

# ============ 4. 定义奖励函数 ============
def format_reward_func(completions, **kwargs):
    rewards = []
    for completion in completions:
        if re.search(r'<answer>.*?</answer>', completion, re.DOTALL):
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards

def accuracy_reward_func(completions, answer, **kwargs):
    rewards = []
    for completion, gt in zip(completions, answer):
        match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
        if not match:
            rewards.append(0.0)
            continue
        pred = match.group(1).strip()
        rewards.append(1.0 if pred == gt else 0.0)
    return rewards

# ============ 5. GRPO 训练 ============
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    num_train_epochs=1,
    per_device_train_batch_size=1,  # GRPO 中每个设备的批次大小
    gradient_accumulation_steps=4,
    num_generations=NUM_GENERATIONS,  # 每个 prompt 的采样数量[reference:20]
    max_completion_length=MAX_COMPLETION_LENGTH,
    logging_steps=10,
    save_steps=100,
    report_to="wandb",  # 可选
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=train_dataset.select(range(200)),  # 先用 200 条数据快速验证
    reward_funcs=[format_reward_func, accuracy_reward_func],
)

# 开始训练
trainer.train()

# 保存模型
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
