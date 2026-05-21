import torch
import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer


"""
训练基于transformer的单向语言模型, 并完成文本生成
"""


def load_data(file_path):
    """从 jsonl 文件加载数据集"""
    data_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data_list.append(json.loads(line))
    return data_list


class TextGenDataset(Dataset):
    """
    文本生成数据集.
    date_list 格式: [{"instruction": ..., "output": ...}]
    """
    def __init__(self, data_list, tokenizer, max_length=128):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        item = self.data_list[index]
        # 假设使用标准的对话格式
        prompt = f"User: {item['instruction']}\nAssistant: "
        response = item['output']
        full_text = prompt + response
        
        # 1. Tokenize 完整序列
        encodings = self.tokenizer(
            full_text, 
            truncation=True, 
            max_length=self.max_length, 
            padding="max_length", 
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"].squeeze(0)
        labels = input_ids.clone()
        
        # 2. 构造 Label Mask：计算 prompt 的长度，将其对应的 label 设为 -100
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        # 找到 prompt 结束的位置（最后一个 prompt token 的索引+1）
        # 通过比对 full_text tokenize 后的结果来确定
        full_ids = input_ids.tolist()
        prompt_end_idx = 0
        for i, tid in enumerate(full_ids):
            if tid in self.tokenizer.all_special_ids and tid != self.tokenizer.pad_token_id:
                # 跳过所有特殊token后开始计算
                continue
            # 检查后续 token 是否匹配 prompt_ids
            if full_ids[i:i+len(prompt_ids)] == prompt_ids:
                prompt_end_idx = i + len(prompt_ids)
                break
        # 如果没找到精确匹配，使用 prompt_ids 长度作为参考（考虑特殊token）
        if prompt_end_idx == 0:
            prompt_end_idx = len(prompt_ids)
        labels[:prompt_end_idx] = -100 
        
        # 3. 处理 padding 部分的 mask（避免对填充符计算 loss）
        labels[encodings["attention_mask"].squeeze(0) == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": labels
        }


# 加载单向语言模型 (CausalLM)
def load_model(model_name="Qwen/Qwen2.5-0.5B"):
    """
    加载模型: 加载预训练模型, 并设置 pad_token
    model_name 支持: "Qwen/Qwen2.5-0.5B", "gpt2", "meta-llama/Llama-2-7b-hf" (Qwen/Qwen2.5-0.5B 中文能力更强)
    """
    # 使用 Hugging Face 的 hf 命令下载模型后, 这里可以直接使用模型名称加载模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 确保设置 pad_token，否则生成时会报错, 比如Qwen/Qwen2.5-0.5B 默认没有 pad_token，需要手动指定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def train_model(model, train_dataset, eval_dataset=None):
    """
    训练模型
    """
    # 检测是否支持 fp16，不支持则降级
    fp16_setting = True
    if not torch.cuda.is_available():
        fp16_setting = False
    elif torch.cuda.get_device_capability(0)[0] < 7:
        fp16_setting = False  # 老 GPU 不支持 fp16

    training_args = TrainingArguments(
        output_dir="./单向语言模型_SFT",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        learning_rate=5e-5,
        warmup_steps=100,
        logging_steps=10,
        save_strategy="epoch",
        fp16=fp16_setting,  # 自动检测是否支持混合精度训练
        # report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # 可选，传入验证集用于监控过拟合
    )

    trainer.train()


def generate_text(model, tokenizer, input_text, max_new_tokens=100, temperature=0.7, top_p=0.9):
    """
    生成文本
    """
    model.eval()
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 使用 generate 方法进行自回归生成
    outputs = model.generate(
        inputs.input_ids, 
        max_new_tokens=max_new_tokens,      # 最多新生成 100 个 token
        do_sample=True,                    # 启用采样，增加生成的多样性
        temperature=temperature,           # 温度系数，越低越保守，越高越随机
        top_p=top_p,                       # 核采样参数
        pad_token_id=tokenizer.pad_token_id
    )

    return tokenizer.decode(outputs, skip_special_tokens=True)


# 文本生成推理
def main():
    model, tokenizer = load_model()

    # 加载数据集
    train_data_list = load_data("train.jsonl")
    eval_data_list = load_data("eval.jsonl")

    # 训练模型
    train_dataset = TextGenDataset(train_data_list, tokenizer)
    train_model(model, train_dataset, eval_dataset=TextGenDataset(eval_data_list, tokenizer))

    # 生成文本
    input_text = "User: 请帮我写一首关于春天的诗\nAssistant: "
    generated_text = generate_text(model, tokenizer, input_text)
    print(generated_text)   


if __name__ == "__main__":
    main()