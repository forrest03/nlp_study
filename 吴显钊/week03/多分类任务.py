"""
设计一个以文本为输入的多分类任务
RNN
LSTM
跑通模型训练
对任意一个包含"你"字的文本，"你"在第几位，就属于第几类
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42    # 随机种子，保证结果可复现
N_SAMPLES   = 4000  # 总生成样本数量
MAXLEN      = 5    # 句子最大长度，不足补0，超出截断
EMBED_DIM   = 64    # 词向量维度
HIDDEN_DIM  = 64    # RNN隐藏层维度
LR          = 1e-3  # 学习率
BATCH_SIZE  = 64    # 每批次训练样本数
EPOCHS      = 20    # 训练轮数
TRAIN_RATIO = 0.8   # 训练集占总数据比例（剩余为验证集）
MOUDLE_TYPE = "LSTM" # 测试2种模型 RNN, LSTM

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────
TEMPLATES = list("阿萨的疯狂拉升的各类卡萨丁付款吗能力女开始了放假撒开绿灯飞机垃圾焚烧炉卡就发了就流口水阿斯利康节点发送大量")

def make_simples():
    """生成样本句子"""
    base = [random.choice(TEMPLATES) for _ in range(MAXLEN) ] #随机选取句子
    pos = random.randint(0, len(base)-1)
    base[pos] = '你'
    return (base, pos)

def build_dataset(n=N_SAMPLES):
    """构建完整数据集"""
    data = []
    for _ in range(n):
        data.append(make_simples())      
    random.shuffle(data)                       # 打乱数据顺序
    return data

# ─── 2. 词表构建与句子编码 ──────────────────────────────────────
def build_vocab(data):
    """
    从所有句子中构建字符词典
    给每个汉字分配唯一数字ID，方便模型处理
    """
    vocab = {'<PAD>': 0, '<UNK>': 1}  # 特殊符号：填充符0，未知字符1
    for sent, _ in data:
        for ch in sent:               # 遍历句子中每个字
            if ch not in vocab:
                vocab[ch] = len(vocab)# 新字分配ID
    return vocab

def encode(sent, vocab, maxlen=MAXLEN):
    """
    将句子转为数字序列
    汉字 → 对应ID；长度不足补0，超长截断
    """
    ids  = [vocab.get(ch, 1) for ch in sent]  # 字转ID，未知字用1
    ids  = ids[:maxlen]                       # 截断超长部分
    ids += [0] * (maxlen - len(ids))          # 填充PAD到固定长度
    return ids

# ─── 3. 自定义数据集类，供PyTorch加载 ────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        """初始化：把所有句子转成数字序列，保存标签"""
        self.X = [encode(s, vocab) for s, _ in data]  # 句子编码
        self.y = [lb for _, lb in data]               # 标签

    def __len__(self):
        """返回数据集总样本数"""
        return len(self.y)

    def __getitem__(self, i):
        """根据索引取一条样本（转成张量）"""
        return (
            torch.tensor(self.X[i], dtype=torch.long),   # 句子ID序列
            torch.tensor(self.y[i], dtype=torch.long),  # 标签
        )
    

# ─── 4. RNN模型定义 ────────────────────────────────────────────
class KeywordRNN(nn.Module):
    """
    中文关键词分类器（RNN）
    架构：Embedding → RNN  → 批归一化 → Dropout → 全连接
    """
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        # 词嵌入层：把数字ID转为低维向量
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # RNN层：提取序列特征
        self.rnn       = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        # 批归一化：加速训练、稳定梯度
        self.bn        = nn.BatchNorm1d(hidden_dim)
        # Dropout：防止过拟合
        self.dropout   = nn.Dropout(dropout)
        # 全连接层：输出分类概率
        self.fc        = nn.Linear(hidden_dim, MAXLEN)

    def forward(self, x):
        """前向传播：模型计算流程"""
        # x形状：(批次大小, 句子长度)
        e, _ = self.rnn(self.embedding(x))  # 嵌入 + RNN，输出：(B, L, hidden_dim)
        feat = e[:, -1, :]      # 取 RNN 最后时刻的隐藏状态
        pooled = self.dropout(self.bn(feat))  # 归一化 + 随机失活
        out = self.fc(pooled)  # 直接输出
        return out
    

# ─── 4. RNN模型定义 ────────────────────────────────────────────
class KeywordLSTM(nn.Module):
    """
    中文关键词分类器（LSTM）
    架构：Embedding → LSTM  → 批归一化 → Dropout → 全连接
    """
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        # 词嵌入层：把数字ID转为低维向量
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # LSTM层：提取序列特征
        self.lstm       = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        # 批归一化：加速训练、稳定梯度
        self.bn        = nn.BatchNorm1d(hidden_dim)
        # Dropout：防止过拟合
        self.dropout   = nn.Dropout(dropout)
        # 全连接层：输出分类概率
        self.fc        = nn.Linear(hidden_dim, MAXLEN)

    def forward(self, x):
        """前向传播：模型计算流程"""
        # x形状：(批次大小, 句子长度)
        e, _ = self.lstm(self.embedding(x))  # 嵌入 + LSTM，输出：(B, L, hidden_dim)
        feat = e[:, -1, :]      # 取 LSTM 最后时刻的隐藏状态
        pooled = self.dropout(self.bn(feat))  # 归一化 + 随机失活
        out = self.fc(pooled)  # 直接输出
        return out

# ─── 5. 模型评估函数 ──────────────────────────────────────────
def evaluate(model, loader):
    """在验证集上计算准确率"""
    model.eval()                # 切换到评估模式（关闭Dropout等）
    correct = total = 0         # 正确数、总数初始化
    with torch.no_grad():       # 不计算梯度，节省内存
        for X, y in loader:
            prob    = model(X)          # 模型预测概率
            pred    =  torch.argmax(prob, dim=1)# 取最大的分类
            correct += (pred == y.long()).sum().item()  # 统计正确数
            total   += len(y)           # 统计总数
    return correct / total     # 返回准确率

# ─── 6. 训练主函数 ──────────────────────────────────────────
def train():
    print("生成数据集...")
    data  = build_dataset(N_SAMPLES)     # 生成全部样本
    vocab = build_vocab(data)            # 构建字符词典
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")
    print(f" 运行{MOUDLE_TYPE}模型")
    split      = int(len(data) * TRAIN_RATIO)  # 训练/验证集分割点
    train_data = data[:split]                  # 训练集
    val_data   = data[split:]                  # 验证集

    # 构建数据加载器
    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_data,   vocab), batch_size=BATCH_SIZE)

    # 初始化模型、损失函数、优化器
    if MOUDLE_TYPE == "RNN":
        model = KeywordRNN(vocab_size=len(vocab))
    elif MOUDLE_TYPE == "LSTM":
        model = KeywordLSTM(vocab_size=len(vocab))
    else:
        model = KeywordRNN(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()                  # 交叉熵
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    # 开始训练循环
    for epoch in range(1, EPOCHS + 1):
        model.train()               # 切换为训练模式
        total_loss = 0.0            # 本轮总损失初始化
        
        # 遍历训练批次
        for X, y in train_loader:
            pred = model(X)                   # 前向传播，得到预测值
            loss = criterion(pred, y)        # 计算损失
            optimizer.zero_grad()            # 清空梯度
            loss.backward()                  # 反向传播，计算梯度
            optimizer.step()                 # 更新参数
            total_loss += loss.item()        # 累加损失

        # 计算本轮平均损失和验证集准确率
        avg_loss = total_loss / len(train_loader)
        val_acc  = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    # 训练结束，输出最终准确率
    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    # 测试示例：用真实句子推理
    print("\n--- 推理示例 ---")
    model.eval()
    test_sents = [
        '萨的疯狂你',
        '卡萨你丁付',
        '始你了放假',
        '你发了就流',
    ]
    with torch.no_grad():
        for sent in test_sents:
            true_pos = sent.index('你')  # 真实位置
            ids   = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            prob  = model(ids)
            print(f"{sent}:真实【{true_pos}】|模型预测【{torch.argmax(prob, dim=1).item()}】")


# 程序入口
if __name__ == '__main__':
    train()