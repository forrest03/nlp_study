"""题目生成、输出解析、Prompt 构造 —— 训练与测试共用。

难度设计（基于基座模型实测摸底，Qwen3.5-0.8B greedy 准确率）：
  L1_add_2digit       两位数加减   ~100%  —— 简单，训练集外，作 sanity check
  L2_mul_2x1digit     两位数×一位数 ~100%  —— 简单，训练集外
  L3_mul_2x2digit     两位数×两位数 ~90%   —— 偏易，训练集外
  L4_mul_3x2digit     三位数×两位数 ~70%   —— 训练集（甜区）
  L5_mul_3x3digit     三位数×三位数 ~20%   —— 训练集（甜区，提升空间最大）

设计动机：GRPO 只对"组内有对有错"的 prompt 产生梯度。太易（全对）或太难（全错）的
组 advantage 恒为 0，纯浪费算力 —— 训练集只保留基座 20%~70% 正确率的甜区难度。
"""
import random
import re

# 系统提示：要求模型只输出 <answer> 标签
SYSTEM_PROMPT = (
    "你是一个算术助手。用户会给你一道算术题，请计算出结果，"
    "并把最终答案放在 <answer> 标签中，例如 <answer>42</answer>。"
    "不要输出其他内容。"
)

# 解析规则：优先取 <answer> 标签内的整数，否则取输出中最后一个整数（宽松）
TAG_RE = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")
NUM_RE = re.compile(r"-?\d+")

LEVELS = [
    "L1_add_2digit",
    "L2_mul_2x1digit",
    "L3_mul_2x2digit",
    "L4_mul_3x2digit",
    "L5_mul_3x3digit",
]

# 训练集难度配比（只选甜区：基座正确率 20%~70%，有提升空间且有梯度）
LEVEL_MIX = [
    ("L4_mul_3x2digit", 0.5),
    ("L5_mul_3x3digit", 0.5),
]


def make_problem(level: str, rng: random.Random):
    """按难度生成一道算术题，返回 (表达式文本, 标准答案)。"""
    if level == "L1_add_2digit":
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        if rng.random() < 0.5:
            return f"{a} + {b}", a + b
        a, b = max(a, b), min(a, b)  # 保证减法结果非负
        return f"{a} - {b}", a - b
    if level == "L2_mul_2x1digit":
        a, b = rng.randint(10, 99), rng.randint(3, 9)
        return f"{a} × {b}", a * b
    if level == "L3_mul_2x2digit":
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        return f"{a} × {b}", a * b
    if level == "L4_mul_3x2digit":
        a, b = rng.randint(100, 999), rng.randint(10, 99)
        return f"{a} × {b}", a * b
    if level == "L5_mul_3x3digit":
        a, b = rng.randint(100, 999), rng.randint(100, 999)
        return f"{a} × {b}", a * b
    raise ValueError(level)


def parse_output(text: str, answer: int):
    """解析模型输出，返回 (是否符合 <answer> 格式, 严格正确, 宽松正确)。"""
    m = TAG_RE.search(text)
    fmt_ok = m is not None
    strict_ok = fmt_ok and int(m.group(1)) == answer
    nums = NUM_RE.findall(text)
    loose_ok = bool(nums) and int(nums[-1]) == answer  # 宽松：最后一个数字正确
    return fmt_ok, strict_ok, loose_ok


def build_messages(expr: str):
    """构造 chat 消息（系统 + 用户），供 tokenizer.apply_chat_template 使用。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"计算：{expr} = ?"},
    ]


def pick_level(rng: random.Random) -> str:
    """按 LEVEL_MIX 配比随机选一个训练难度。"""
    r = rng.random()
    acc = 0.0
    for lv, p in LEVEL_MIX:
        acc += p
        if r <= acc:
            return lv
    return LEVEL_MIX[-1][0]
