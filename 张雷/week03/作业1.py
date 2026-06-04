"""
语序信息对比实验 —— "你"在第几位就属于第几类

任务：随机生成含"你"的 5 字句子，"你"在第 pos 位（1-indexed 类别 1~5）
      每个类别对"词袋模型"而言完全无法区分（相同字符集，只是顺序不同）

模型A（RNN + MaxPool）：Embedding → RNN → MaxPool → Linear
  RNN 逐字处理，每步隐藏状态携带"已读到哪"的上下文，池化后仍保留语序信息

模型B（BoW  MaxPool）：Embedding → MaxPool → Linear   （不使用 RNN）
  直接对字符 Embedding 做 MaxPool ≈ 词袋，输出与字符顺序无关，
  理论上完全无法判断"你"的位置，准确率应接近随机基准（20%）

运行结果对比展示语序信息对 NLP 任务的重要性。

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt

# ─── 超参数 ────────────────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 10000    # 总样本数；每类 2000 条
SEQ_LEN     = 5        # 句子固定 5 字
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8
NUM_CLASSES = 5        # "你"的位置：第 1~5 位对应类别 0~4

random.seed(SEED)
torch.manual_seed(SEED)

# -------------------- 1. 生成数据集 --------------------
# 不含"你"的字符池，保证每类样本的字符集完全相同，仅顺序不同
CHAR_POOL = list("我是他她它很好吗的了一不人大小中上下来去天工和机学会可要产民对能行方说这时那也里后前在有个到出过得子开着道看场面把样关点心然现想起经发理用家意成所事法没如还问话知信重体相东路已手都题自量明实物从当气本打做此进力内平实加回定总数正比老很名高文公战国水青红头问记组特表神教太眼长声府区快技济族早马夜嗯哦呵嘛嘿哎喂")

def generate_sample(pos=None):
    if pos is None:
        pos = random.randint(0, 4)  # "你"的位置，0~4 对应类别 0~4
    chars = random.choices(CHAR_POOL, k=5) # 随机生成 5 个字符
    chars[pos] = "你" # 将 "你" 放在 pos 位
    return "".join(chars), pos  # "你"在第 pos 位，类别为 pos

def generate_dataset(n:int = N_SAMPLES):
    samples = [generate_sample() for _ in range(n)]
    texts, labels = zip(*samples)
    return list(texts), list(labels)

# -------------------- 2. 构建词表 --------------------
def build_vocab(chars):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for ch in chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab
VOCAB = build_vocab(sorted(set("".join(CHAR_POOL))))

# print(f"词表: {VOCAB}")
# print(VOCAB.get("你"))
# VOCAB_SIZE = len(VOCAB)
# NUM_CLASSES = 5

# print(f"词表大小: {len(CHAR_POOL)}")
# print(f"词表: {VOCAB}")
# print(VOCAB.get("很"))

def text_to_tensor(text):
    return torch.tensor([VOCAB.get(ch, 1) for ch in text], dtype=torch.long)

# -------------------- 3. Dataset --------------------
class CharPositionDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return (
            text_to_tensor(self.texts[idx]), 
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

# -------------------- 4. 模型定义 --------------------
class RNNModel(nn.Module):
    """
    模型A：Embedding → RNN → MaxPool（沿时序）→ Linear

    RNN 按位置顺序逐字处理，第 t 步的隐藏状态 h_t 编码了
    "前 t 个字"的上下文；MaxPool 汇聚全局最显著特征，
    但特征本身已携带位置信息。
    """
    def __init__(self, vocab_size: int): # vocab_size 词表大小
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.rnn       = nn.RNN(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.fc        = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

    def forward(self, x):
        # x      : (B, SEQ_LEN)
        emb         = self.embedding(x)       # (B, SEQ_LEN, EMBED_DIM)
        rnn_out, _  = self.rnn(emb)           # (B, SEQ_LEN, HIDDEN_DIM)
        pooled      = rnn_out.max(dim=1)[0]   # (B, HIDDEN_DIM)  时序 max-pool
        return self.fc(pooled)                # (B, NUM_CLASSES)


class BagModel(nn.Module):
    """
    模型B：Embedding → MaxPool（沿时序）→ Linear   【不使用 RNN】

    直接对字符 Embedding 做 MaxPool，结果与字符顺序无关，
    相当于"词袋(Bag-of-Characters)"表示，无法区分"你"所在位置。
    """
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.fc        = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def forward(self, x):
        # x      : (B, SEQ_LEN)
        emb    = self.embedding(x)            # (B, SEQ_LEN, EMBED_DIM)
        pooled = emb.max(dim=1)[0]            # (B, EMBED_DIM)  顺序无关！
        return self.fc(pooled)                # (B, NUM_CLASSES)

# -------------------- 5. 训练与评估 --------------------
def evaluate_per_class(model, loader):
    model.eval()
    correct = [0] * NUM_CLASSES # 每类正确预测数
    total   = [0] * NUM_CLASSES # 每类样本总数
    with torch.no_grad():
        for X, y in loader:
            pred = model(X).argmax(dim=1)
            for cls in range(NUM_CLASSES):
                mask = (y == cls) # 选出真实标签为 cls 的样本
                correct[cls] += (pred[mask] == cls).sum().item() # 累积 cls 类的正确预测数
                total[cls]   += mask.sum().item() # 累积 cls 类的样本总数
    return [correct[c] / total[c] if total[c] else 0.0 for c in range(NUM_CLASSES)]


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x) # 前向传播得到预测结果
            correct += (logits.argmax(1) == y).sum().item() # 累积正确预测数
            total += len(y) # 累积样本总数
    return correct / total

def train_model(model, train_loader, val_loader, name):
    print(f"\n{'='*56}")
    print(f"  训练：{name}")
    print(f"  参数量：{sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*56}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1): # 训练 EPOCHS 轮
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            logits = model(X)
            loss   = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 4 == 0 or epoch == 1: # 每 4 轮评估一次
            avg_loss = total_loss / len(train_loader)
            val_acc  = evaluate(model, val_loader)
            print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    final_acc = evaluate(model, val_loader)
    print(f"  最终验证准确率：{final_acc:.4f}")
    return final_acc, model

# -------------------- 6. 主流程 --------------------
def main():
    random.seed(42)
    torch.manual_seed(42)

    train_texts, train_labels = generate_dataset(8000)
    test_texts, test_labels = generate_dataset(2000)

    train_dataset = CharPositionDataset(train_texts, train_labels)
    test_dataset = CharPositionDataset(test_texts, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("\n样例数据:")
    for i in range(6):
        print(f"  文本: '{train_texts[i]}' -> 类别: {train_labels[i]}")

    EMBED_DIM = 32
    HIDDEN_DIM = 64
    EPOCHS = 30
    LR = 0.001

    rnn_acc, rnn_model = train_model(
        RNNModel(len(VOCAB)), train_loader, test_loader,
        "模型A — RNN + MaxPool（保留语序）"
    )
    bow_acc, bow_model = train_model(
        BagModel(len(VOCAB)), train_loader, test_loader,
        "模型B — 直接 Embedding MaxPool（无 RNN，丢失语序）"
    )

    # ─── 总体对比 ──────────────────────────────────────────────────────
    print(f"\n{'='*56}")
    print(f"  结果对比（{NUM_CLASSES} 分类，随机基准 {1/NUM_CLASSES:.0%}）")
    print(f"{'='*56}")
    print(f"  模型A（RNN + MaxPool）          val_acc = {rnn_acc:.4f}")
    print(f"  模型B（直接 Embedding MaxPool）  val_acc = {bow_acc:.4f}")

    # ─── 逐类准确率 ────────────────────────────────────────────────────
    rnn_per_cls = evaluate_per_class(rnn_model, test_loader)
    bow_per_cls = evaluate_per_class(bow_model, test_loader)
    print('\n  各位置准确率（「你」在第 X 位）：')
    print(f"  {'位置':>6}  {'模型A(RNN)':>12}  {'模型B(BoW)':>12}")
    for i in range(NUM_CLASSES):
        print(f"  第 {i+1} 位   {rnn_per_cls[i]:>10.4f}    {bow_per_cls[i]:>10.4f}")

    # ─── 结论 ──────────────────────────────────────────────────────────
    print(f"""
【结论】
  模型A 使用 RNN 逐字处理，第 t 步的隐藏状态编码了"前 t 字的上下文"，
  因此即使最终对时序做 MaxPool，池化结果仍携带位置信息，能准确判断"你"在哪位。

  模型B 直接对 Embedding 做 MaxPool，等价于词袋(BoW)：
  把句子中所有字符的向量混在一起取最大值，输出与字符顺序完全无关。
  5 个类别对它而言"看起来一模一样"，准确率约等于随机基准 20%。

  → 语序信息对于位置感知类任务至关重要；RNN 通过顺序递推有效保留了这一信息。
""")

    # ─── 推理示例 ──────────────────────────────────────────────────────
    print('─── 推理示例（各模型预测「你」所在位置）──────────────────────')
    rnn_model.eval()
    bow_model.eval()
    random.seed(0)
    print(f"  {'句子':>6}  真实位置  模型A(RNN)  模型B(BoW)")
    with torch.no_grad():
        for pos in range(NUM_CLASSES):
            sent, label = generate_sample(pos)
            # ids         = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            ids         = text_to_tensor(sent).unsqueeze(0)  # 添加 batch 维度
            rnn_pred    = rnn_model(ids).argmax(dim=1).item()
            bow_pred    = bow_model(ids).argmax(dim=1).item()
            rnn_mark    = '✓' if rnn_pred == label else '✗'
            bow_mark    = '✓' if bow_pred == label else '✗'
            print(f"  「{sent}」  第{label+1}位     "
                  f"{rnn_mark} 第{rnn_pred+1}位      "
                  f"{bow_mark} 第{bow_pred+1}位")


if __name__ == "__main__":
    main()
