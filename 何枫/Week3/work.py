"""
train_position_cls_compare.py
语序信息对比实验 —— "你"在第几位就属于第几类

任务：随机生成含"你"的 5 字句子，"你"在第 pos 位（1-indexed 类别 1~5）
      每个类别对"词袋模型"而言完全无法区分（相同字符集，只是顺序不同）

模型A（RNN + MaxPool）：Embedding → RNN → MaxPool → Linear
  RNN 逐字处理，每步隐藏状态携带"已读到哪"的上下文，池化后仍保留语序信息

模型B（BoW  MaxPool）：Embedding → MaxPool → Linear   （不使用 RNN）
  直接对字符 Embedding 做 MaxPool ≈ 词袋，输出与字符顺序无关，
  理论上完全无法判断"你"的位置，准确率应接近随机基准（20%）

运行结果对比展示语序信息对 NLP 任务的重要性。

依赖：torch >= 2.0   (pip install torch)
"""
import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader

SEED = 42
N_SAMPLES = 4000
SEQ_LEN = 5
EMBED_DIM = 64
HIDDEN_DIM = 64
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 20
TRAIN_RATIO = 0.8

NUM_CLASSES = 5

random.seed(SEED)
torch.manual_seed(SEED)

CHAR_POOL = (
    'aabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    '就那和要她出也得里后自以会家可下而过天去能对小多然于心'
    '01234567890123456789012345678901234567890123456789'
    '无情己面最女但现前些所同日手又行意别信走定回爱进此'
)

# 生成数据集
def build_dataset(n=N_SAMPLES):
    per_class = n // NUM_CLASSES
    data = []
    for pos in range(NUM_CLASSES):
        for _ in range(per_class):
            chars = random.choices(CHAR_POOL, k=SEQ_LEN - 1)
            chars.insert(pos, '你')
            sentence = ''.join(chars)
            data.append((sentence, pos))
    random.shuffle(data)
    return data

# 词表与编码
def build_vocab(data):
    vocab = {'PAD': 0, 'UNK': 1}  # 用于填充的特殊标记
    for sentence, _ in data:
        for char in sentence:
            if char not in vocab:
                vocab[char] = len(vocab)
    return vocab

# 编码函数
def encode_sentence(sentence, vocab):
    ids = [vocab.get(char, 1) for char in sentence[:SEQ_LEN]]
    ids += [0] * (SEQ_LEN - len(ids))  # 填充到固定长度
    return ids

# 定义数据集
class PositionDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = [encode_sentence(sentence, vocab) for sentence, _ in data]
        self.vocab = [lb for _, lb in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return (
            torch.tensor(self.data[i], dtype=torch.long), 
            torch.tensor(self.vocab[i], dtype=torch.long),
        )

def evaluate(model, val_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch).argmax(dim=1)
            correct += (outputs == y_batch).sum().item()
            total += len(y_batch)
    return correct / total

def train_model(model, train_loader, val_loader, name):
    print()
    print(f"  训练：{name}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad() # 梯度清零
            outputs = model(X_batch) # 前向传播
            loss = criterion(outputs, y_batch) # 计算损失
            loss.backward() # 计算梯度
            optimizer.step() # 更新权重
            total_loss += loss.item()

        if epoch % 4 == 0 or epoch == 1:
            avg_loss = total_loss / len(train_loader)
            val_acc = evaluate(model, val_loader)
            print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}')
        
    final_acc = evaluate(model, val_loader)   
    print(f"  最终验证准确率：{final_acc:.4f}")
    return final_acc, model

class RNNModel(nn.Module):
    """
    模型A：Embedding → RNN → MaxPool（沿时序）→ Linear

    RNN 按位置顺序逐字处理，第 t 步的隐藏状态 h_t 编码了
    "前 t 个字"的上下文；MaxPool 汇聚全局最显著特征，
    但特征本身已携带位置信息。
    """
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.rnn = nn.RNN(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        # self.maxpool = nn.MaxPool1d(SEQ_LEN)
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

    def forward(self, x):
        # x      : (B, SEQ_LEN)
        emb = self.embedding(x)
        rnn_out, _ = self.rnn(emb)
        pooled = rnn_out.max(dim=1)[0]  # (B, HIDDEN_DIM)  顺序有关！
        return self.fc(pooled)

class BoWModel(nn.Module):
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

def evaluate_per_class(model, val_loader):
    model.eval()
    correct = [0] * NUM_CLASSES
    total = [0] * NUM_CLASSES
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            pred = model(X_batch).argmax(dim=1)
            for cls in range(NUM_CLASSES):
                cls_mask = (y_batch == cls)
                correct[cls] += (pred[cls_mask] == cls).sum().item()
                total[cls] += cls_mask.sum().item()
    return [correct[i] / total[i] if total[i] else 0 for i in range(NUM_CLASSES)]

def make_sample(pos):
    chars = random.choices(CHAR_POOL, k=SEQ_LEN - 1)
    chars.insert(pos, '你')
    return ''.join(chars), pos

def main():
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)

    print(f"  总样本：{len(data)}，每类 {len(data)//NUM_CLASSES} 条")
    print(f"  词表大小：{len(vocab)}")
    print(f'  任务：预测「你」在 5 字句子中的位置（第 1~5 位，{NUM_CLASSES} 分类）')
    print(f"  随机猜测基准准确率：{1/NUM_CLASSES:.0%}")

    # 后续代码：构建 Dataset、DataLoader，定义模型，训练并评估
    split = int(len(data) * TRAIN_RATIO)
    train_loader = DataLoader(PositionDataset(data[:split], vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(PositionDataset(data[split:], vocab), batch_size=BATCH_SIZE)

    # TODO: 定义模型、训练、评估
    rnn_acc, rnn_model = train_model(RNNModel(len(vocab)), train_loader, val_loader, "RNN + MaxPool")
    bow_acc, bow_model = train_model(BoWModel(len(vocab)), train_loader, val_loader, "BoW")

    # 打印最终对比结果
    print(f"\n{'='*56}")
    print(f"  结果对比（{NUM_CLASSES} 分类，随机基准 {1/NUM_CLASSES:.0%}）")
    print(f"{'='*56}")
    print(f"  模型A（RNN + MaxPool）          val_acc = {rnn_acc:.4f}")
    print(f"  模型B（直接 Embedding MaxPool）  val_acc = {bow_acc:.4f}")

    # 模型准确率
    rnn_per_cls = evaluate_per_class(rnn_model, val_loader)
    bow_per_cls = evaluate_per_class(bow_model, val_loader)

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
    
    rnn_model.eval()
    bow_model.eval()
    random.seed(0)
    print(f"  {'句子':>6}  真实位置  模型A(RNN)  模型B(BoW)")
    with torch.no_grad():
        for i in range(NUM_CLASSES):
            sent, label = make_sample(i)
            X = torch.tensor([encode_sentence(sent, vocab)], dtype=torch.long)
            rnn_pred = rnn_model(X).argmax(dim=1).item()
            bow_pred = bow_model(X).argmax(dim=1).item()
            rnn_mark    = '✓' if rnn_pred == label else '✗'
            bow_mark    = '✓' if bow_pred == label else '✗'
            print(f"  「{sent}」  第{label+1}位     "
                  f"{rnn_mark} 第{rnn_pred+1}位      "
                  f"{bow_mark} 第{bow_pred+1}位")

if __name__ == '__main__':
    main()
