import time
from pathlib import Path
import json
import argparse

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyBertDataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from model import BertNER

from seqeval.metrics import f1_score as seqeval_f1

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / 'data'
BERT_PATH = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
CKPT_DIR = ROOT / "outputs" / "checkpoints"  # 模型检查点保存目录
LOG_DIR = ROOT / "outputs" / "logs"  # 训练日志保存目录


def load_records(data_filename: str, data_dir: Path = DATA_DIR):
    path = f'{data_dir}/{data_filename}.json'
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_label_schema():
    labels = load_records('label_names', DATA_DIR)
    id2label = {id: name for id, name in enumerate(labels)}
    label2id = {name: id for id, name in enumerate(labels)}
    return labels, label2id, id2label


def parse_args():
    parser = argparse.ArgumentParser(description="训练 BERT NER 模型")
    parser.add_argument("--use_crf", action="store_true", help="使用 CRF 层（否则使用线性头）")
    parser.add_argument("--bert_path", type=Path, default=BERT_PATH)
    parser.add_argument("--device", type=str, default="cuda", help="设备: cuda / cuda:0 / cuda:1 / cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=150)
    parser.add_argument("--lr", type=float, default=2e-5, help="BERT 层学习率")
    parser.add_argument("--head_lr_mult", type=float, default=5.0, help="分类头学习率倍数")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)

    return parser.parse_args()


def build_data_loader(batch_size, tokenizer, max_length, labels):
    train_records = load_records('train', DATA_DIR)
    val_records = load_records('validation', DATA_DIR)
    test_records = load_records('test', DATA_DIR)

    train_dataloader = DataLoader(
        MyBertDataset(train_records, tokenizer, labels, max_length),
        batch_size,
        shuffle=True)
    val_dataloader = DataLoader(
        MyBertDataset(val_records, tokenizer, labels, max_length),
        batch_size,
        shuffle=False)
    test_dataloader = DataLoader(
        MyBertDataset(test_records, tokenizer, labels, max_length),
        batch_size,
        shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def train_one_epoch(
        model,
        train_dataloader,
        optimizer,
        scheduler,
        device,
        epoch,
        epochs,
        grad_accum):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{epochs}", leave=False)
    for steps, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)
        _, loss = model(input_ids, attention_mask, token_type_ids, labels)
        loss = loss / grad_accum
        loss.backward()
        if (steps + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，防止梯度爆炸
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        total_loss += loss.item()
        pbar.set_postfix(loss=f'{loss.item():.4f}')

    # 处理最后不足grad_accum的批次
    remainder = len(train_dataloader) % grad_accum
    if remainder != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(train_dataloader)  # 返回平均损失


def evaluate_epoch(model, val_dataloader, id2label, device):
    model.eval()
    total_loss = 0.0
    all_preds: list[list[str]] = []
    all_labels: list[list[str]] = []
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            logits, loss = model(input_ids, attention_mask, token_type_ids, labels)

            # Linear模式：logits是每个类别的未归一化分数
            # 直接取概率最大的类别作为预测
            pred_ids_list = logits.argmax(dim=-1).tolist()

            total_loss += loss.item()

            labels_np = labels.cpu().tolist()

            # 将标签从ID转换为字符串形式，用于seqeval评估
            for i in range(len(input_ids)):
                gold_seq = []
                pred_seq = []
                labels = labels_np[i]
                pred_ids = pred_ids_list[i]

                for j, gold_id in enumerate(labels):
                    if gold_id == -100:
                        continue

                    gold_seq.append(id2label[gold_id])
                    pred_seq.append(id2label.get(pred_ids[j], "O"))
                all_preds.append(pred_seq)
                all_labels.append(gold_seq)

    avg_loss = total_loss / len(val_dataloader)
    entity_f1 = seqeval_f1(all_labels, all_preds)  # 计算entity-level F1分数

    return avg_loss, entity_f1

def main():
    args = parse_args()
    # 使用指定设备，如果不可用则回退到 CPU
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"警告: {args.device} 不可用，将使用 CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    labels, label2id, id2label = build_label_schema()
    tokenizer = AutoTokenizer.from_pretrained(str(args.bert_path), use_fast=True)
    
    train_dataloader, val_dataloader, test_dataloader = build_data_loader(args.batch_size,
                                                                          tokenizer,
                                                                          args.max_length,
                                                                          labels)
    model = BertNER(
        bert_path=str(args.bert_path),
        num_labels=len(labels),
        dropout=args.dropout
    ).to(device)
    bert_params = list(model.bert.parameters())
    head_params = (
        list(model.classifier.parameters())
    )

    optimizer = AdamW(
        [
            {"params": bert_params, "lr": args.lr},
            {"params": head_params, "lr": args.lr * args.head_lr_mult},
        ],
        weight_decay=0.01,
    )

    total_steps = len(train_dataloader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )
    print("Total steps:", total_steps, "Warmup steps:", warmup_steps)

    # 准备保存路径
    run_tag = "linear"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"best_{run_tag}.pt"
    log_path = LOG_DIR / f"{run_tag}.log"

    best_f1 = 0.0
    log_records = []

    # 执行训练
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            device,
            epoch,
            args.epochs,
            args.grad_accum,
        )

        # 在验证集上评估
        val_loss, val_f1 = evaluate_epoch(model, val_dataloader, id2label, device)
        elapsed = time.time() - t0  # 计算耗时

        # 打印当前epoch的训练结果
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_entity_f1={val_f1:.4f} | "
            f"time={elapsed:.0f}s"
        )

        # 记录训练日志
        log_records.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_entity_f1": round(val_f1, 6),
            "elapsed_s": round(elapsed, 1),
        })

        # ===== 第九步：保存最佳模型 =====
        if val_f1 > best_f1:  # 如果当前F1超过历史最佳
            best_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "use_crf": args.use_crf,
                    "state_dict": model.state_dict(),  # 模型参数
                    "val_entity_f1": val_f1,
                    "label2id": label2id,  # 保存标签映射，推理时需要
                    "id2label": id2label,
                    "args": vars(args),  # 保存训练参数，便于复现
                },
                ckpt_path,
            )
            print(f"  ★ 新最优 F1={val_f1:.4f}，已保存 → {ckpt_path}")
    # ===== 第十步：保存训练日志 =====
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成！最优 val_entity_f1={best_f1:.4f}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  训练日志:   {log_path}")
    print(f"\n下一步：python evaluate.py {'--use_crf' if args.use_crf else ''}")


if __name__ == '__main__':
    main()
