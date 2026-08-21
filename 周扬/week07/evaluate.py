"""
评估 BertNer。

"""

from argparse import ArgumentParser
from pathlib import Path

import torch
from transformers import BertTokenizer

from Dataset import all_labels, build_dataloaders
from Model import BertNer, bert_model_path


CURRENT_DIR = Path(__file__).parent
DEFAULT_CKPT_PATH = CURRENT_DIR / "outputs" / "best_bert_ner.pt"


def parse_args():
    parser = ArgumentParser(description="BertNER 评估脚本")
    # 默认直接评估 test，也可以手动切到 val 看验证集
    parser.add_argument("--bert_path", default=bert_model_path, type=str, help="BERT 模型目录")
    parser.add_argument("--ckpt_path", default=str(DEFAULT_CKPT_PATH), type=str, help="模型权重路径")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="评估数据集")
    parser.add_argument("--batch_size", default=32, type=int, help="batch 大小")
    parser.add_argument("--max_length", default=128, type=int, help="序列最大长度")
    parser.add_argument("--dropout", default=0.1, type=float, help="模型 dropout，加载 checkpoint 时会自动覆盖")
    return parser.parse_args()


def get_device():
    # 跟训练时保持一样，能上 GPU 就优先上
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_loader(args, device):
    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到模型文件：{ckpt_path}")

    # 这里主要是把训练时存下来的配置一并恢复出来
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    label_list = checkpoint.get("label_list", all_labels)
    dropout = ckpt_args.get("dropout", args.dropout)
    bert_path = ckpt_args.get("bert_path", args.bert_path)
    max_length = ckpt_args.get("max_length", args.max_length)
    batch_size = ckpt_args.get("batch_size", args.batch_size)

    # tokenizer 和模型结构尽量跟训练时一致，不然容易对不上
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    _, val_loader, test_loader = build_dataloaders(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
    )

    model = BertNer(num_labels=len(label_list), dropout=dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 按参数决定这次到底评估 val 还是 test
    loader = val_loader if args.split == "val" else test_loader
    return model, loader, label_list


def compute_label_metrics(true_ids, pred_ids, label_list):
    # 这里就老老实实自己算，不额外依赖 sklearn
    metrics = []

    total_support = 0
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    for label_id, label_name in enumerate(label_list):
        # 这几个量是分类指标里最基础的那套
        tp = 0
        fp = 0
        fn = 0
        support = 0

        for true_label, pred_label in zip(true_ids, pred_ids):
            if true_label == label_id:
                support += 1
            if true_label == label_id and pred_label == label_id:
                tp += 1
            elif true_label != label_id and pred_label == label_id:
                fp += 1
            elif true_label == label_id and pred_label != label_id:
                fn += 1

        # 分母为 0 的情况顺手兜一下
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics.append(
            {
                "label": label_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

        total_support += support
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

    label_count = len(label_list)
    macro_avg = {
        "precision": macro_precision / label_count if label_count > 0 else 0.0,
        "recall": macro_recall / label_count if label_count > 0 else 0.0,
        "f1": macro_f1 / label_count if label_count > 0 else 0.0,
        "support": total_support,
    }
    weighted_avg = {
        "precision": weighted_precision / total_support if total_support > 0 else 0.0,
        "recall": weighted_recall / total_support if total_support > 0 else 0.0,
        "f1": weighted_f1 / total_support if total_support > 0 else 0.0,
        "support": total_support,
    }
    return metrics, macro_avg, weighted_avg


@torch.no_grad()
def evaluate(model, dataloader, device, label_list):
    # 这里只评估，不回传梯度
    total_loss = 0.0
    total_steps = 0
    valid_true_ids = []
    valid_pred_ids = []

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

        preds = logits.argmax(dim=-1)
        valid_mask = labels != -100

        # -100 这些位置本来就不参与训练和评估，这里一起跳过
        valid_true_ids.extend(labels[valid_mask].detach().cpu().tolist())
        valid_pred_ids.extend(preds[valid_mask].detach().cpu().tolist())

    avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
    accuracy = (
        sum(int(t == p) for t, p in zip(valid_true_ids, valid_pred_ids)) / len(valid_true_ids)
        if valid_true_ids
        else 0.0
    )
    metrics, macro_avg, weighted_avg = compute_label_metrics(valid_true_ids, valid_pred_ids, label_list)
    return avg_loss, accuracy, metrics, macro_avg, weighted_avg, len(valid_true_ids)


def print_report(metrics, macro_avg, weighted_avg):
    # 打印得像个简易分类报告，终端里看着直观一点
    print("\n=== 分类报告 ===")
    print(f"{'label':<10} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
    for item in metrics:
        print(
            f"{item['label']:<10} "
            f"{item['precision']:>10.4f} "
            f"{item['recall']:>10.4f} "
            f"{item['f1']:>10.4f} "
            f"{item['support']:>10}"
        )

    print(
        f"{'macro avg':<10} "
        f"{macro_avg['precision']:>10.4f} "
        f"{macro_avg['recall']:>10.4f} "
        f"{macro_avg['f1']:>10.4f} "
        f"{macro_avg['support']:>10}"
    )
    print(
        f"{'weighted':<10} "
        f"{weighted_avg['precision']:>10.4f} "
        f"{weighted_avg['recall']:>10.4f} "
        f"{weighted_avg['f1']:>10.4f} "
        f"{weighted_avg['support']:>10}"
    )


def main():
    args = parse_args()
    device = get_device()
    model, dataloader, label_list = load_model_and_loader(args, device)

    # 这里先把评估环境打印一下，跑起来心里更有数
    print("=== BertNER 评估开始 ===")
    print(f"模型路径: {args.ckpt_path}")
    print(f"评估数据集: {args.split}")
    print(f"设备: {device}")

    avg_loss, accuracy, metrics, macro_avg, weighted_avg, valid_token_count = evaluate(
        model,
        dataloader,
        device,
        label_list,
    )

    print("\n=== 总体结果 ===")
    print(f"loss           : {avg_loss:.4f}")
    print(f"token_acc      : {accuracy:.4f}")
    print(f"有效token数    : {valid_token_count}")
    print_report(metrics, macro_avg, weighted_avg)


if __name__ == "__main__":
    main()
