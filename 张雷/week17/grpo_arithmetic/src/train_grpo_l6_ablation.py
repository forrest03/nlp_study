"""
消融实验：把 L6（两位数×两位数）加入训练集，验证其 informative 组能否提供学习信号。

背景：本机基线摸底显示 L6 的 loose_informative_group_rate = 0.58（处于可学习甜区 0.5~0.8），
而原项目（CUDA + Qwen2）基线上 L6 仅 0.24（组内常全错，无信号）故不进训练集。
本脚本把 L6 以 20% 配比加入训练集（其余难度按原比例缩放），观察：
  1. L6 正确率是否比不加 L6 的原训练（0.50→0.54）提升更多；
  2. 训练集内其他难度（L2/L3/L5）是否因配比降低而回退；
  3. 训练动态上退化组比例是否因 L6 组的存在而保持更低（信号未枯竭）。

用法（与 train_grpo.py 相同的参数）：
  python src/train_grpo_l6_ablation.py --max_steps 50 --tag l6ablation

输出：outputs/grpo_ckpt_l6ablation/ + outputs/train_log_l6ablation.json
"""
import train_grpo  # noqa: F401  复用模块级配置与 main()（含 trl_compat 补丁）

# ── 消融改动点：L6 以 20% 加入训练集，L3/L5/L2 按原比例缩放到 80% ────────────
train_grpo.LEVEL_MIX = [
    ("L3_addsub_3digit", 0.40),   # 原 0.50 → 0.40
    ("L5_mul_2x1digit", 0.20),    # 原 0.25 → 0.20
    ("L2_addsub_2digit", 0.20),   # 原 0.25 → 0.20
    ("L6_mul_2x2digit", 0.20),    # 新增：本机 informative=0.58 的可学性验证
]

train_grpo.main()
