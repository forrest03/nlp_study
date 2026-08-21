"""
训练 BertNer。
"""

from argparse import ArgumentParser
from pathlib import Path
import time

import torch
from torch.optim import AdamW
from transformers import BertTokenizer

from Dataset import all_labels, build_dataloaders
from Model import BertNer, bert_model_path


CURRENT_DIR = Path(__file__).parent
OUTPUT_DIR = CURRENT_DIR / "outputs"
BEST_MODEL_PATH = OUTPUT_DIR / "best_bert_ner.pt"


def parse_args():
    parser = ArgumentParser(description="BERT 序列标注训练")
    #模型的相关参数

    #下载的bert模型目录，比较大 不上传了
    parser.add_argument("--bert_path", default=bert_model_path, type=str, help="BERT 模型目录")
    parser.add_argument("--epochs", default=1, type=int, help="训练轮数")
    parser.add_argument("--batch_size", default=32, type=int, help="batch大小")
    parser.add_argument("--max_length", default=128, type=int, help="序列最大长度")
    parser.add_argument("--lr", default=2e-5, type=float, help="学习率")
    parser.add_argument("--dropout", default=0.1, type=float, help="dropout率")

    #用来打印日志用的，友好显示
    parser.add_argument("--log_step", default=100, type=int, help="每隔多少个 step 打印一次训练日志")
    return parser.parse_args()


def get_device():
    """选设备，我的电脑是mac，用mps 不知道跟cpu比如何"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def calc_token_accuracy(logits, labels):
    """
    计算 token 级准确率。

    只统计 labels != -100 的位置，
    """
    preds = logits.argmax(dim=-1)
    valid_mask = labels != -100
    if valid_mask.sum().item() == 0:
        return 0, 0
    correct = ((preds == labels) & valid_mask).sum().item()
    total = valid_mask.sum().item()
    return correct, total


def evaluate(model, dataloader, device, split_name="验证集"):
    """评估 loss 和 token 准确率。"""
    model.eval()
    total_loss = 0.0
    total_steps = 0
    total_correct = 0
    total_tokens = 0
    start_time = time.time()

    #评估的时候不用计算梯度
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            logits, loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )

            total_loss += loss.item()
            total_steps += 1

            correct, total = calc_token_accuracy(logits, labels)
            total_correct += correct
            total_tokens += total

    #平均loss和准确率
    avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
    acc = total_correct / total_tokens if total_tokens > 0 else 0.0
    elapsed = time.time() - start_time
    print(
        f"[{split_name}] 评估完成 | "
        f"loss={avg_loss:.4f} | acc={acc:.4f} | "
        f"有效token数={total_tokens} | 用时={elapsed:.1f}s"
    )
    return avg_loss, acc


#一次训练的逻辑，多次训练的时候循环调用
def train_one_epoch(model, dataloader, optimizer, device, epoch_idx, total_epochs, log_step):
    """训练 1 个 epoch。"""
    model.train()
    total_loss = 0.0
    total_steps = 0
    total_correct = 0
    total_tokens = 0
    start_time = time.time()

    for step, batch in enumerate(dataloader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        #前向传播，
        logits, loss = model(
            input_ids=input_ids,
            #输入的序列掩码，用于忽略 padding token
            attention_mask=attention_mask,
            #输入的序列类型掩码，用于区分不同序列的 token
            token_type_ids=token_type_ids,
            labels=labels,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_steps += 1

        #计算准确率
        correct, total = calc_token_accuracy(logits, labels)
        total_correct += correct
        total_tokens += total
        
        #打印日志
        if step % log_step == 0 or step == len(dataloader):
            avg_loss = total_loss / total_steps
            avg_acc = total_correct / total_tokens if total_tokens > 0 else 0.0
            elapsed = time.time() - start_time
            print(
                f"[训练中] Epoch {epoch_idx}/{total_epochs} | "
                f"step {step}/{len(dataloader)} | "
                f"avg_loss={avg_loss:.4f} | avg_acc={avg_acc:.4f} | "
                f"已用时={elapsed:.1f}s"
            )

    #平均loss和准确率
    epoch_loss = total_loss / total_steps if total_steps > 0 else 0.0
    epoch_acc = total_correct / total_tokens if total_tokens > 0 else 0.0
    elapsed = time.time() - start_time
    return epoch_loss, epoch_acc, elapsed


def main():
    args = parse_args()
    device = get_device()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== BertNER 训练开始 ===")
    print(f"BERT 路径: {args.bert_path}")
    print(f"标签数: {len(all_labels)}")
    print(f"训练轮数: {args.epochs}")
    print(f"batch_size: {args.batch_size}")
    print(f"max_length: {args.max_length}")
    print(f"学习率: {args.lr}")
    print(f"log_step: {args.log_step}")
    print(f"设备: {device}")

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    train_loader, val_loader, test_loader = build_dataloaders(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    model = BertNer(num_labels=len(all_labels), dropout=args.dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== 开始第 {epoch}/{args.epochs} 轮训练 =====")
        train_loss, train_acc, train_elapsed = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch_idx=epoch,
            total_epochs=args.epochs,
            log_step=args.log_step,
        )

        print(f"----- 第 {epoch} 轮训练结束，开始跑验证集 -----")
        val_loss, val_acc = evaluate(model, val_loader, device, split_name="验证集")
        print(
            f"[Epoch 总结] {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"训练用时={train_elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            prev_best = best_val_acc
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "args": vars(args),
                    "label_list": all_labels,
                },
                BEST_MODEL_PATH,
            )
            if prev_best < 0:
                print(f"[最佳模型] 首次保存最佳模型到: {BEST_MODEL_PATH}")
            else:
                print(
                    f"[最佳模型] val_acc 从 {prev_best:.4f} 提升到 {best_val_acc:.4f}，"
                    f"已更新保存: {BEST_MODEL_PATH}"
                )
        else:
            print(f"[最佳模型] 当前最佳 val_acc 仍为 {best_val_acc:.4f}")

    print("\n===== 训练结束，开始测试集评估 =====")
    test_loss, test_acc = evaluate(model, test_loader, device, split_name="测试集")
    print(f"[最终结果] test_loss={test_loss:.4f} | test_acc={test_acc:.4f}")
    print(f"最佳验证集准确率: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
