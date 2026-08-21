import time
from pathlib import Path
import json
import argparse
from typing import Dict, List, Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import itertools

from dataset import MyBertDataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from model import BertNER

from seqeval.metrics import f1_score as seqeval_f1

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / 'data'
BERT_PATH = ROOT.parent / "model" / "bert-base-chinese"
OUTPUT_DIR = ROOT / "outputs" / "hyperparam_tuning"


def load_records(data_filename: str, data_dir: Path = DATA_DIR):
    path = f'{data_dir}/{data_filename}.json'
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_label_schema():
    labels = load_records('label_names', DATA_DIR)
    id2label = {id: name for id, name in enumerate(labels)}
    label2id = {name: id for id, name in enumerate(labels)}
    return labels, label2id, id2label


def build_data_loader(batch_size, tokenizer, max_length, labels):
    train_records = load_records('train', DATA_DIR)
    val_records = load_records('validation', DATA_DIR)

    train_dataloader = DataLoader(
        MyBertDataset(train_records, tokenizer, labels, max_length),
        batch_size,
        shuffle=True)
    val_dataloader = DataLoader(
        MyBertDataset(val_records, tokenizer, labels, max_length),
        batch_size,
        shuffle=False)

    return train_dataloader, val_dataloader


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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        total_loss += loss.item()
        pbar.set_postfix(loss=f'{loss.item():.4f}')

    remainder = len(train_dataloader) % grad_accum
    if remainder != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(train_dataloader)


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

            pred_ids_list = logits.argmax(dim=-1).tolist()
            total_loss += loss.item()
            labels_np = labels.cpu().tolist()

            for i in range(len(input_ids)):
                gold_seq = []
                pred_seq = []
                labels_i = labels_np[i]
                pred_ids = pred_ids_list[i]

                for j, gold_id in enumerate(labels_i):
                    if gold_id == -100:
                        continue

                    gold_seq.append(id2label[gold_id])
                    pred_seq.append(id2label.get(pred_ids[j], "O"))
                all_preds.append(pred_seq)
                all_labels.append(gold_seq)

    avg_loss = total_loss / len(val_dataloader)
    entity_f1 = seqeval_f1(all_labels, all_preds)

    return avg_loss, entity_f1


def train_single_run(config: Dict[str, Any], run_id: int, device: str, labels, id2label, tokenizer):
    """单次训练运行，返回训练历史记录"""
    print(f"\n{'='*60}")
    print(f"Run {run_id}: {config}")
    print(f"{'='*60}")

    train_dataloader, val_dataloader = build_data_loader(
        config['batch_size'],
        tokenizer,
        config['max_length'],
        labels
    )

    model = BertNER(
        bert_path=str(BERT_PATH),
        num_labels=len(labels),
        dropout=config['dropout']
    ).to(device)

    bert_params = list(model.bert.parameters())
    head_params = list(model.classifier.parameters())

    optimizer = AdamW(
        [
            {"params": bert_params, "lr": config['lr']},
            {"params": head_params, "lr": config['lr'] * config['head_lr_mult']},
        ],
        weight_decay=0.01,
    )

    total_steps = len(train_dataloader) * config['epochs'] // config['grad_accum']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    history = {
        'config': config,
        'run_id': run_id,
        'epochs': []
    }

    best_f1 = 0.0

    for epoch in range(1, config['epochs'] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            device,
            epoch,
            config['epochs'],
            config['grad_accum'],
        )

        val_loss, val_f1 = evaluate_epoch(model, val_dataloader, id2label, device)
        elapsed = time.time() - t0

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_entity_f1": round(val_f1, 6),
            "elapsed_s": round(elapsed, 1),
        }
        history['epochs'].append(epoch_record)

        print(
            f"Epoch {epoch}/{config['epochs']} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_f1={val_f1:.4f} | "
            f"time={elapsed:.0f}s"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1

    history['best_val_f1'] = best_f1
    print(f"Run {run_id} complete! Best val F1: {best_f1:.4f}")

    return history


def get_default_search_space():
    """定义默认的超参数搜索空间"""
    return {
        'lr': [1e-5, 2e-5, 3e-5, 5e-5],
        'batch_size': [16, 32, 64],
        'dropout': [0.1, 0.2, 0.3],
        'head_lr_mult': [2.0, 5.0],
        'warmup_ratio': [0.1],
        'grad_accum': [1],
        'max_length': [150],
        'epochs': [5]
    }


def generate_configs(search_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """生成所有超参数组合"""
    keys = list(search_space.keys())
    values = list(search_space.values())
    configs = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        configs.append(config)
    return configs


def parse_args():
    parser = argparse.ArgumentParser(description="超参数调优训练脚本")
    parser.add_argument("--device", type=str, default="cuda", help="设备: cuda / cuda:0 / cuda:1 / cpu")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR, help="结果保存目录")
    parser.add_argument("--config_file", type=Path, default=None, help="超参数搜索空间配置JSON文件")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"警告: {args.device} 不可用，将使用 CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    labels, label2id, id2label = build_label_schema()
    tokenizer = AutoTokenizer.from_pretrained(str(BERT_PATH), use_fast=True)

    if args.config_file and args.config_file.exists():
        with open(args.config_file, 'r', encoding='utf-8') as f:
            search_space = json.load(f)
        print(f"加载搜索空间配置: {args.config_file}")
    else:
        search_space = get_default_search_space()
        print("使用默认搜索空间")

    configs = generate_configs(search_space)
    print(f"共 {len(configs)} 组超参数组合需要训练")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    results_file = args.output_dir / "all_results.json"

    for run_id, config in enumerate(configs, 1):
        try:
            history = train_single_run(config, run_id, str(device), labels, id2label, tokenizer)
            all_results.append(history)

            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Run {run_id} 失败: {e}")
            continue

    print(f"\n{'='*60}")
    print("所有训练完成！结果总结:")
    print(f"{'='*60}")

    sorted_results = sorted(all_results, key=lambda x: x['best_val_f1'], reverse=True)
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"{i}. Best F1: {result['best_val_f1']:.4f} | Config: {result['config']}")

    print(f"\n完整结果已保存到: {results_file}")
    print("使用 plot_results.py 进行可视化分析")


if __name__ == '__main__':
    main()
