MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# 1. 加载数学数据集
dataset = load_dataset("open-r1/Mixture-of-Thoughts", split="train")
dataset = dataset.select(range(5000))

def convert_example(example):
    return {
        "prompt": [
            {
                "role": "user",
                "content": example["problem"]
            }
        ],
        "answer": example["solution"]
    }

dataset = dataset.map(convert_example)

# 2. 提取最终答案
def extract_answer(text):
    """
    从模型输出中提取最后的数字答案。
    """
    numbers = re.findall(
        r"-?\d+(?:\.\d+)?",
        text.replace(",", "")
    )

    if not numbers:
        return None

    return numbers[-1]


def extract_target(solution):
    numbers = re.findall(
        r"-?\d+(?:\.\d+)?",
        solution.replace(",", "")
    )

    if not numbers:
        return None

    return numbers[-1]


# 3. GRPO Reward
def reward_func(completions, answer, **kwargs):
    rewards = []

    for completion, target in zip(completions, answer):

        if isinstance(completion, list):
            text = completion[-1]["content"]
        else:
            text = completion

        pred = extract_answer(text)
        target = extract_target(target)

        # 正确答案奖励
        if pred is not None and target is not None:
            if abs(float(pred) - float(target)) < 1e-6:
                reward = 1.0
            else:
                reward = 0.0
        else:
            reward = 0.0

        # 包含推理过程给予少量奖励
        if len(text) > 50:
            reward += 0.1

        rewards.append(reward)

    return rewards


# 4. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

# 5. GRPO 参数
training_args = GRPOConfig(
    output_dir="./grpo-math",
    learning_rate=1e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    num_generations=4,

    max_prompt_length=512,
    max_completion_length=512,

    num_train_epochs=1,

    logging_steps=10,
    save_steps=500,

    bf16=torch.cuda.is_available(),

    report_to="none"
)

# 6. GRPO Trainer
trainer = GRPOTrainer(
    model=MODEL_NAME,
    processing_class=tokenizer,
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=dataset
)

# 7. 开始训练
trainer.train()

trainer.save_model("./grpo-math-final")
tokenizer.save_pretrained("./grpo-math-final")
