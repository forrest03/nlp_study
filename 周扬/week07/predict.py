"""
试试训练好的模型，标准结果如何
"""

from argparse import ArgumentParser
from pathlib import Path

import torch
from transformers import BertTokenizer

from Dataset import all_labels
from Model import BertNer, bert_model_path


CURRENT_DIR = Path(__file__).parent
DEFAULT_CKPT_PATH = CURRENT_DIR / "outputs" / "best_bert_ner.pt"


def parse_args():
    parser = ArgumentParser(description="BertNER")
    # 这几个参数基本够用了，先保持简单
    parser.add_argument("--bert_path", default=bert_model_path, type=str, help="BERT 模型目录")
    parser.add_argument("--ckpt_path", default=str(DEFAULT_CKPT_PATH), type=str, help="模型权重路径")
    parser.add_argument("--max_length", default=128, type=int, help="预测时的最大长度")
    parser.add_argument("--dropout", default=0.1, type=float, help="模型 dropout，加载 checkpoint 时会自动覆盖")
    return parser.parse_args()


def get_device():
    # 能用 GPU 就优先用，苹果芯片走 mps
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(args, device):
    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到模型文件：{ckpt_path}")

    # 先把训练时存下来的 checkpoint 读出来
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    label_list = checkpoint.get("label_list", all_labels)
    dropout = ckpt_args.get("dropout", args.dropout)
    bert_path = ckpt_args.get("bert_path", args.bert_path)
    max_length = ckpt_args.get("max_length", args.max_length)

    # tokenizer 和模型最好跟训练时保持一致
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    model = BertNer(num_labels=len(label_list), dropout=dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    # 预测完拿到的是数字 id，这里转回标签名方便看
    id2label = {idx: label for idx, label in enumerate(label_list)}
    return model, tokenizer, id2label, max_length


@torch.no_grad()
def predict_text(model, tokenizer, id2label, text, max_length, device):
    # 这里还是按字来做，和训练时的数据形式对齐
    chars = list(text)
    encoding = tokenizer(
        chars,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    word_ids = encoding.word_ids(batch_index=0)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding["token_type_ids"].to(device)

    # 这里只做前向，不算 loss
    logits, _ = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        labels=None,
    )
    pred_ids = logits.argmax(dim=-1).squeeze(0).tolist()

    char_labels = []
    prev_word_id = None
    for token_idx, word_id in enumerate(word_ids):
        # special token 直接跳过
        if word_id is None:
            continue
        # 如果一个字被拆成多个子词，只保留第一次出现的位置
        if word_id == prev_word_id:
            continue
        if word_id >= len(chars):
            continue
        char_labels.append((chars[word_id], id2label[pred_ids[token_idx]]))
        prev_word_id = word_id

    return char_labels


def pretty_print_result(text, char_labels):
    print("\n=== 预测结果 ===")
    print(f"输入文本: {text}")
    print("逐字标签:")
    for idx, (char, label) in enumerate(char_labels):
        print(f"{idx:>2} | {char} | {label}")

    # 顺手把连续实体拼一下，不然只看逐字标签会有点累
    entities = []
    current_text = ""
    current_type = None
    for char, label in char_labels:
        if label == "O":
            if current_text:
                entities.append((current_text, current_type))
                current_text = ""
                current_type = None
            continue

        prefix, entity_type = label.split("-", 1)
        if prefix == "B":
            if current_text:
                entities.append((current_text, current_type))
            current_text = char
            current_type = entity_type
        elif prefix == "I" and current_type == entity_type:
            current_text += char
        else:
            if current_text:
                entities.append((current_text, current_type))
            current_text = char
            current_type = entity_type

    if current_text:
        entities.append((current_text, current_type))

    print("\n抽取出的实体:")
    if not entities:
        print("没有识别出明确实体")
    else:
        for entity_text, entity_type in entities:
            print(f"- {entity_text} -> {entity_type}")


def main():
    args = parse_args()
    device = get_device()
    model, tokenizer, id2label, max_length = load_model_and_tokenizer(args, device)

    print("=== BertNER 交互式预测 ===")
    print(f"模型路径: {args.ckpt_path}")
    print(f"设备: {device}")
    print("输入一句中文开始预测，输入 q 退出。")

    while True:
        text = input("\n请输入文本: ").strip()
        if text.lower() in {"q", "quit", "exit"}:
            print("已退出预测。")
            break
        if not text:
            print("输入不能为空，请重新输入。")
            continue

        # 每输一句就跑一次预测
        char_labels = predict_text(model, tokenizer, id2label, text, max_length, device)
        pretty_print_result(text, char_labels)


if __name__ == "__main__":
    main()
