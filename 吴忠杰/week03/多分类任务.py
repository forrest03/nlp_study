
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED = 42
N_SAMPLES = 5000  # 训练样本总数
MAXLEN = 5  # 固定文本长度为5
EMBED_DIM = 64  # 字向量维度
HIDDEN_DIM = 64  # RNN/LSTM 隐藏层维度
LR = 1e-3  # 学习率
BATCH_SIZE = 64
EPOCHS = 15
TRAIN_RATIO = 0.8
TARGET_CHAR = '你'  # 目标查找的字

# 供生成文本使用的字库（排除“你”字，以免干扰）
CHAR_POOL = 'abcdefghijkmnopqrstuvwxyz你我他她的好坏是与否有没有大小撒娇死哦达塞拉的而我认为浮点数的规划规范的色如体育面料辅料处理能力建设多少上下左右前后中'
CHAR_POOL = CHAR_POOL.replace(TARGET_CHAR, '')

random.seed(SEED)
torch.manual_seed(SEED)


# ─── 1. 数据生成 ────────────────────────────────────────────
def build_dataset(n=N_SAMPLES, length=MAXLEN, target=TARGET_CHAR):
    data = []
    for _ in range(n):
        # 随机生成一个不含“你”的文本
        text_list = [random.choice(CHAR_POOL) for _ in range(length)]

        # 随机决定“你”字的位置 (0 到 length-1)，或者 -1 代表不存在
        pos = random.randint(-1, length - 1)
        if pos != -1:
            text_list[pos] = target

        text = "".join(text_list)
        # 标签：如果 pos 是 -1（不存在），标签设为 length；否则标签就是位置索引 (0~4)
        label = pos if pos != -1 else length
        data.append((text, label))

    random.shuffle(data)
    return data


# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(sent, vocab, maxlen=MAXLEN):
    # 将文本转换为数字索引
    ids = [vocab.get(ch, 1) for ch in sent]
    ids = ids[:maxlen]
    # 填充到固定长度
    ids += [0] * (maxlen - len(ids))
    return ids


# ─── 3. Dataset ────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        # 注意：多分类任务的标签需要用 long 类型
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )


# ─── 4. 模型定义 (支持 RNN / LSTM / GRU 切换) ───────────────
class SequenceClassifyModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_classes=MAXLEN + 1,
                 model_type='rnn'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 根据传入的 model_type 动态选择循环神经网络层
        model_type = model_type.lower()
        if model_type == 'lstm':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        elif model_type == 'gru':
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)

        # 输出层：映射到 (MAXLEN + 1) 个类别
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)

        # rnn_out: (batch, seq_len, hidden_dim), hidden: (1, batch, hidden_dim)
        rnn_out, hidden = self.rnn(emb)

        # 提取最后一个时间步的隐藏状态作为整个句子的特征
        # 如果是 LSTM，hidden 是一个元组 (h_n, c_n)，取 h_n
        if isinstance(hidden, tuple):
            last_hidden = hidden[0].squeeze(0)
        else:
            last_hidden = hidden.squeeze(0)

        out = self.fc(last_hidden)  # (batch, num_classes)
        return out


# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            # 获取概率最大的类别的索引
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


def train_and_predict(model_type='rnn'):
    print(f"--- 开始使用 {model_type.upper()} 模型进行训练 ---")
    print("1. 生成数据集...")
    data = build_dataset(N_SAMPLES, MAXLEN, TARGET_CHAR)
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE)

    # 实例化模型，类别数为 MAXLEN + 1 (0~4位 + 1个不存在)
    model = SequenceClassifyModel(vocab_size=len(vocab), model_type=model_type)

    # 多分类任务必须使用交叉熵损失
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"  模型参数量：{sum(p.numel() for p in model.parameters()):,}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")
    return model, vocab


def predict(model, vocab, input_text):
    model.eval()
    print("\n--- 推理预测 ---")
    # 确保输入文本长度为 5
    if len(input_text) != MAXLEN:
        print(f"警告：输入文本长度不是 {MAXLEN}，将自动截断或填充！")

    with torch.no_grad():
        ids = torch.tensor([encode(input_text, vocab)], dtype=torch.long)
        logits = model(ids)
        # 获取预测的类别索引
        predicted_class = torch.argmax(logits, dim=1).item()

        if predicted_class < MAXLEN:
            print(f"文本：'{input_text}' -> 预测结果：'{TARGET_CHAR}' 字在第 {predicted_class + 1} 位")
        else:
            print(f"文本：'{input_text}' -> 预测结果：文本中不存在 '{TARGET_CHAR}' 字")


if __name__ == '__main__':
    # 在这里切换 'rnn', 'lstm', 或 'gru' 来对比效果
    selected_model_type = 'lstm'

    # 1. 训练模型
    trained_model, trained_vocab = train_and_predict(model_type=selected_model_type)

    # 2. 手动输入测试
    while True:
        user_input = input(f"\n请输入一个包含或不包含“{TARGET_CHAR}”的{MAXLEN}字文本（输入 q 退出）：")
        if user_input.lower() == 'q':
            break
        if len(user_input) == 0:
            continue
        predict(trained_model, trained_vocab, user_input)
