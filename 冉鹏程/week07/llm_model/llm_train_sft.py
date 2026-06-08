import os
import random
from pathlib import Path
import argparse
import torch
import json

import time

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from torch import nn

from llm_dataset import SFTDataset

# 设置环境变量：允许 libomp 库重复加载（避免某些环境的报错）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 尝试导入 PEFT 库（用于 LoRA 微调），如果失败则标记为不可用
try:
    from peft import LoraConfig, get_peft_model, TaskType

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODEL_PATH = '/root/.cache/modelscope/hub/models/Qwen/Qwen3-4B'
OUTPUT_DIR = ROOT / "outputs"


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LLM SFT NER 训练（LoRA / 全量微调）")
    parser.add_argument("--model_path", default=str(MODEL_PATH),
                        help="预训练模型的路径")
    parser.add_argument("--data_dir", default=str(DATA_DIR),
                        help="数据目录路径")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR),
                        help="输出目录路径")
    parser.add_argument("--num_train", default=-1, type=int,
                        help="训练样本数，-1 使用全部 10748 条（默认）")
    parser.add_argument("--epochs", default=3, type=int,
                        help="训练轮数")
    parser.add_argument("--batch_size", default=4, type=int,
                        help="每个设备的批量大小")
    parser.add_argument("--grad_accum", default=4, type=int,
                        help="梯度累积步数，有效批量大小 = batch_size * grad_accum")
    parser.add_argument("--lr", default=None, type=float,
                        help="学习率；默认 LoRA=2e-4，全量=2e-5（自动判断）")
    parser.add_argument("--max_length", default=256, type=int,
                        help="序列最大长度；NER 的 JSON 输出比分类长，建议 256")
    # 全量微调开关
    parser.add_argument("--full_ft", action="store_true",
                        help="全量微调：跳过 LoRA，更新所有 495M 参数（需显存 ≥ 16GB）")
    # LoRA 超参（full_ft 时忽略）
    parser.add_argument("--lora_r", default=8, type=int,
                        help="LoRA 的秩，控制可训练参数数量")
    parser.add_argument("--lora_alpha", default=16, type=int,
                        help="LoRA 的缩放系数")
    parser.add_argument("--seed", default=42, type=int,
                        help="随机种子，保证实验可复现性")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="设备: cuda / cuda:0 / cuda:1 / cpu")
    return parser.parse_args()


def collate_fn(batch, pad_token_id):
    max_len = max(len(item['input_ids']) for item in batch)
    input_ids_list, label_ids_list, mask_list = [], [], []

    for item in batch:
        pad_len = max_len - len(item['input_ids'])
        input_ids_list.append(
            torch.cat([
                item['input_ids'],
                torch.full((pad_len,), pad_token_id, dtype=torch.long)
            ])
        )
        label_ids_list.append(
            torch.cat([
                item['labels'],
                torch.full((pad_len,), -100, dtype=torch.long)
            ])
        )
        mask_list.append(
            torch.cat([
                torch.ones(len(item['input_ids']), dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long)
            ])
        )

    return {
        'input_ids': torch.stack(input_ids_list),
        'attention_mask': torch.stack(mask_list),
        'labels': torch.stack(label_ids_list)
    }


def main():
    args = parse_args()
    device = args.device

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.lr is None:
        args.lr = 2e-5 if args.full_ft else 2e-4

    # 设置路径
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / ("sft_full_ckpt" if args.full_ft else "sft_adapter")
    ckpt_dir.mkdir(parents=True, exist_ok=True)  # 创建检查点目录

    mode_str = "全量微调" if args.full_ft else "LoRA 微调"
    print(f"使用设备: {device}  |  微调模式: {mode_str}")

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    # 读取训练集和验证集
    with open(data_dir / "train.json", encoding="utf-8") as f:
        train_raw = json.load(f)
    with open(data_dir / "validation.json", encoding="utf-8") as f:
        val_raw = json.load(f)

    if args.num_train > 0:
        train_raw = train_raw[:args.num_train]
    # 取前300条做验证
    val_raw = val_raw[:300]

    # 加载模型的tokenizer，AutoTokenizer会自动根据模型路径自动识别，加载模型的tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(args.model_path).resolve()),
        trust_remote_code=True
    )
    # 如果模型没有pad token，则使用eos token代替
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 创建collate函数（绑定pad_token_id）
    _collate = lambda b: collate_fn(b, tokenizer.pad_token_id)

    # 创建数据加载器
    train_loader = DataLoader(
        SFTDataset(train_raw, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate
    )
    val_loader = DataLoader(
        SFTDataset(val_raw, tokenizer, args.max_length),
        batch_size=args.batch_size * 2,  # 验证集可以使用更大的batch size
        shuffle=False,
        collate_fn=_collate
    )

    # 加载模型
    # AutoModelForCausalLM 加载因果语言模型（用于生成任务）
    model = AutoModelForCausalLM.from_pretrained(
        str(Path(args.model_path).resolve()),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # LoRA微调 或 全量微调
    if args.full_ft:
        # 全量微调，更新所有参数，打印参数量
        total = sum([p.numel() for p in model.parameters()])
        print(f"全量微调，参数数量为 {total:,}")
    else:
        if not PEFT_AVAILABLE:
            raise ImportError("LoRA 模式需要 peft 库：pip install peft>=0.14.0")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # 任务类型：因果语言建模
            r=args.lora_r,  # LoRA的秩，控制可训练参数数量
            lora_alpha=args.lora_alpha,  # LoRA的缩放系数
            bias="none",  # lora训练的时候，默认不更新bias参数，除非数据量很小、模型欠拟合、rank很低等
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
        )
        model = get_peft_model(model, lora_config)  # 将lora模型应用到模型上
        model.print_trainable_parameters()  # 打印模型参数
    model = model.to(device)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # 计算总训练步数
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    print(f"总训练步数: {total_steps}（batch={args.batch_size}, "
          f"grad_accum={args.grad_accum}, epochs={args.epochs}, lr={args.lr}）\n")

    # 训练循环
    best_val_loss = float("inf")  # 最佳验证损失
    log_records = []  # 记录每个 epoch 的训练指标
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        total_loss, total_tokens = 0, 0
        optimizer.zero_grad()
        t0 = time.time()

        # 创建进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", leave=False)
        for step, batch in enumerate(pbar):
            # 从batch中提取数据并移动到设备
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # 向前传播，模型自动计算loss
            outputs = model(input_ids=inputs, labels=labels, attention_mask=attention_mask)
            raw_loss = outputs.loss  # 原始loss（未经梯度累积缩放）

            # 反向传播
            # 除以 grad_accum 实现梯度累计（模拟更大batch）
            loss = raw_loss / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                # 梯度裁剪，防止过大的梯度导致梯度爆炸
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            # 统计有效token数（排查-100 mask 的部分）
            n_tokens = (labels != -100).sum().item()
            # 使用原始loss累加，避免被 grad_accum 缩放影响日志统计
            total_loss += raw_loss.item() * n_tokens
            total_tokens += n_tokens

            pbar.set_postfix(loss=f'{raw_loss.item():.4f}')

        # epoch 末尾若仍有未 step 的累积梯度，补一次更新，避免被下个 epoch 的 zero_grad 清掉
        if (step + 1) % args.grad_accum != 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        # 计算平均训练损失
        avg_train_loss = total_loss / max(total_tokens, 1)

        # 验证阶段
        model.eval()
        val_loss, val_tokens = 0, 0
        with torch.no_grad():  # 评估阶段，禁用梯度
            # 验证时使用混合精度推理，降低显存占用，防止batch_size*2 OOM
            autocast_device = "cuda" if device.startswith("cuda") else "cpu"
            with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
                for batch in tqdm(val_loader, desc="Val", leave=False):
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    attention_mask = batch["attention_mask"].to(device)

                    outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                    loss = outputs.loss
                    n_tokens = (labels != -100).sum().item()
                    val_loss += loss.item() * n_tokens
                    val_tokens += n_tokens

        avg_val_loss = val_loss / max(val_tokens, 1)
        print(f"Epoch {epoch + 1}/{args.epochs} [Train] loss: {avg_train_loss:.4f}, [Val] loss: {avg_val_loss:.4f}")
        elapsed = time.time() - t0  # 计算耗时
        # 记录训练日志
        log_records.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'elapsed': elapsed
        })
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # 保存模型（LoRA 模式下只保存 adapter 权重）
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir) # tokenizer可能被修改（添加了pad_token_id），即使没有修改也需要保存，确保与模型保持一致

            ckpt_label = "完整模型" if args.full_ft else "LoRA adapter"
            print(f"  ✓ 最优{ckpt_label}已保存 → {ckpt_dir}  (val_loss={avg_val_loss:.4f})")
    # ── 保存训练日志 ──────────────────────────────────────────────────────────
    log_tag = "full_ft" if args.full_ft else "sft"
    log_path = output_dir / "logs" / f"train_{log_tag}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    ckpt_label = "完整模型" if args.full_ft else "LoRA adapter"
    print(f"\n训练完成。最优 val_loss={best_val_loss:.4f}")
    print(f"训练日志 → {log_path}")
    print(f"{ckpt_label} → {ckpt_dir}")
    print(f"\n下一步：python evaluate_sft.py 查看 entity F1 与多方对比")


if __name__ == "__main__":
    main()
