'''
    作业内容：
    设计一个以文本为输入的多分类任务，实验一下用RNN，LSTM等模型的跑通训练。

    可以选择如下任务:对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。
'''
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader, Dataset

#超参数
SEED        = 42
N_SAMPLES   = 10000    # 总样本数；每类 2000 条
SEQ_LEN     = 5        # 句子固定 5 字
EMBED_DIM   = 64        #    字向量维度
HIDDEN_DIM  = 64  # RNN 隐藏层维度
LR          = 1e-3 # 学习率
BATCH_SIZE  = 64 # 每批训练样本数
EPOCHS      = 20 # 训练轮数
TRAIN_RATIO = 0.8 # 训练集占比
NUM_CLASSES = 5        # "你"的位置：第 1~5 位对应类别 0~4

random.seed(SEED) # 设置随机种子，保证结果可复现
torch.manual_seed(SEED) # 设置随机种子，保证结果可复现


# ─── 1. 字符池（不含"你"）──────────────────────────────────────────────
_RAW = (
    '的一是了我不人在他有这个上们来到时大地为子中说生国年着'
    '就那和要她出也得里后自以会家可下而过天去能对小多然于心'
    '学么之都好看起发当没成只如事把还用第样道想作种开美总从'
    '无情己面最女但现前些所同日手又行意别信走定回爱进此'
)
CHAR_POOL = list(set(ch for ch in _RAW if ch != '你'))

# ─── 2. 数据生成 ─
def make_sample(pos: int):
    """生成一条 5 字样本，"你"固定在 pos 位（0-indexed），标签 = pos"""
    chars = random.choices(CHAR_POOL, k=SEQ_LEN - 1) # 随机选取 SEQ_LEN-1 个字符
    chars.insert(pos, '你') # 将 "你" 插入 pos 位
    return ''.join(chars), pos # 返回文本和标签

def build_dataset(n: int = N_SAMPLES):
    per_class = n // NUM_CLASSES # 每类样本数
    data = []   # 数据列表，元素为 (文本, 标签) 二元组
    for cls in range(NUM_CLASSES): # 生成每类样本
        for _ in range(per_class): # 生成 per_class 条样本
            data.append(make_sample(cls)) # 生成一条样本，标签 = cls
    random.shuffle(data) # 打乱数据顺序
    return data # 返回数据列表，元素为 (文本, 标签) 二元组

# ─── 3. 词表与编码 ─────────────────────────────────────────────────────
def build_vocab(data):
    vocab = {'[PAD]': 0, '[UNK]': 1} # 词表，包含特殊符号 [PAD] 和 [UNK]
    for sent, _ in data: # 遍历数据中的每条文本
        for ch in sent: # 遍历文本中的每个字符
            if ch not in vocab: # 如果字符不在词表中，就添加到词表中，索引为当前词表长度
                vocab[ch] = len(vocab)
    return vocab # 返回词表，键为字符，值为索引


# 编码函数：将文本转换为索引列表，长度为 SEQ_LEN，不足部分用 0 填充，未知字符用 1 表示
def encode(sent, vocab):
    ids = [vocab.get(ch, 1) for ch in sent[:SEQ_LEN]]
    ids += [0] * (SEQ_LEN - len(ids))
    return ids


# ─── 4. Dataset ────────────────────────────────────────────────────────
# Dataset 是 PyTorch 中用于表示数据集的类，必须实现 __len__ 和 __getitem__ 方法。
class PositionDataset(Dataset):
    def __init__(self, data, vocab):
        # 初始化方法，接受数据列表和词表作为输入，构建文本索引列表和标签列表
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        # 返回数据集的样本数，即标签列表的长度
        return len(self.y)

    def __getitem__(self, i):
        # 返回第 i 条样本的文本索引列表和标签，文本索引列表转换为 LongTensor，标签转换为 LongTensor
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )


# ─── 6. 训练 & 评估工具 ────────────────────────────────────────────────
# 训练函数：接受模型、数据加载器和优化器作为输入，进行一个 epoch 的训练，返回平均 loss
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            pred    = model(X).argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += len(y)
    return correct / total


# 评估函数：接受模型和数据加载器作为输入，计算每类的准确率，返回一个列表，列表元素为每类的准确率
def evaluate_per_class(model, loader):
    model.eval() # 设置模型为评估模式，关闭 dropout 和 batchnorm 等训练时特有的层
    correct = [0] * NUM_CLASSES # 初始化一个列表，长度为 NUM_CLASSES，每个元素表示对应类别的正确预测数
    total   = [0] * NUM_CLASSES # 初始化一个列表，长度为 NUM_CLASSES，每个元素表示对应类别的总样本数
    with torch.no_grad(): # 关闭梯度计算，节省内存和计算资源，因为评估时不需要更新模型参数
        for X, y in loader: # 遍历数据加载器中的每个批次，X 是文本索引列表，y 是对应的标签
            pred = model(X).argmax(dim=1) # 模型预测，得到每个样本的预测类别索引，pred 的形状为 (batch_size,)
            for cls in range(NUM_CLASSES): # 遍历每个类别，统计该类别的正确预测数和总样本数
                mask = (y == cls) # 创建一个布尔掩码，表示标签为 cls 的样本位置，mask 的形状为 (batch_size,)
                correct[cls] += (pred[mask] == cls).sum().item() # 统计标签为 cls 的样本中，预测正确的数量，并累加到 correct[cls] 中
                total[cls]   += mask.sum().item()# 统计标签为 cls 的样本总数，并累加到 total[cls] 中
                # 最后，返回一个列表，列表元素为每类的准确率，即 correct[c] / total[c]，如果 total[c] 为 0，则准确率为 0.0
    return [correct[c] / total[c] if total[c] else 0.0 for c in range(NUM_CLASSES)]

# 训练函数：接受模型、训练数据加载器、验证数据加载器和模型名称作为输入，进行模型训练，并在每个 epoch 结束时评估验证集的准确率，返回最终的验证准确率和训练好的模型
def train_model(model, train_loader, val_loader, name):
    print(f"\n{'='*56}")
    print(f"  训练：{name}")
    print(f"  参数量：{sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*56}")
    # 定义损失函数和优化器，使用交叉熵损失函数和 Adam 优化器，学习率为 LR
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # 训练过程，进行 EPOCHS 轮训练，每轮训练结束后评估验证集的准确率，并打印当前 epoch 的平均 loss 和验证准确率
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            logits = model(X) # 模型前向传播，得到每个样本的预测 logits，logits 的形状为 (batch_size, NUM_CLASSES)
            loss   = criterion(logits, y) # 计算损失，使用交叉熵损失函数，输入为预测 logits 和真实标签 y，loss 是一个标量张量
            optimizer.zero_grad() # 梯度归零，清除之前的梯度信息，以免影响当前的梯度计算
            loss.backward() # 反向传播，计算当前批次的梯度信息，更新模型参数的梯度属性
            optimizer.step() # 更新模型参数，根据计算得到的梯度信息，调整模型参数的值，以最小化损失函数
            total_loss += loss.item() # 累加当前批次的损失值，loss.item() 返回一个 Python 数值，表示当前批次的平均损失

        if epoch % 4 == 0 or epoch == 1: # 每 4 轮或第一轮评估一次验证集的准确率，并打印当前 epoch 的平均 loss 和验证准确率
            avg_loss = total_loss / len(train_loader) # 计算当前 epoch 的平均 loss，total_loss 是所有批次的损失值之和，len(train_loader) 是训练数据加载器中的批次数
            # 评估验证集的准确率，调用 evaluate 函数，传入当前模型和验证数据加载器，返回验证集的准确率 val_acc
            val_acc  = evaluate(model, val_loader)

            print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    # 训练结束后，评估最终的验证准确率，并打印结果，返回最终的验证准确率和训练好的模型
    final_acc = evaluate(model, val_loader)
    print(f"  最终验证准确率：{final_acc:.4f}")
    return final_acc, model


#RNN模型
class RNNClassifier(nn.Module):
        """
    模型A：Embedding → RNN → MaxPool（沿时序）→ Linear

    RNN 按位置顺序逐字处理，第 t 步的隐藏状态 h_t 编码了
    "前 t 个字"的上下文；MaxPool 汇聚全局最显著特征，
    但特征本身已携带位置信息。
    """
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
        self.rnn       = nn.RNN(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.fc        = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

    def forward(self, x):
        # x      : (B, SEQ_LEN)
        emb         = self.embedding(x)       # (B, SEQ_LEN, EMBED_DIM)
        rnn_out, _  = self.rnn(emb)           # (B, SEQ_LEN, HIDDEN_DIM)
        pooled      = rnn_out.max(dim=1)[0]   # (B, HIDDEN_DIM)  时序 max-pool
        return self.fc(pooled)                # (B, NUM_CLASSES)

# ─── 7. 主流程 ─────────────────────────────────────────────────────────
def main():
    print("─── 数据集准备 ────────────────────────────────────────────────")
    data  = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  总样本：{len(data)}，每类 {len(data)//NUM_CLASSES} 条")
    print(f"  词表大小：{len(vocab)}")
    print(f'  任务：预测「你」在 5 字句子中的位置（第 1~5 位，{NUM_CLASSES} 分类）')
    print(f"  随机猜测基准准确率：{1/NUM_CLASSES:.0%}")

    split        = int(len(data) * TRAIN_RATIO)
    train_loader = DataLoader(
        PositionDataset(data[:split], vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(
        PositionDataset(data[split:], vocab),  batch_size=BATCH_SIZE)

    rnn_acc, rnn_model = train_model(
        RNNModel(len(vocab)), train_loader, val_loader,
        "模型A — RNN + MaxPool（保留语序）"
    )

    # ─── 总体对比 ──────────────────────────────────────────────────────
    print(f"\n{'='*56}")
    print(f"  结果对比（{NUM_CLASSES} 分类，随机基准 {1/NUM_CLASSES:.0%}）")
    print(f"{'='*56}")
    print(f"  模型（RNN + MaxPool）          val_acc = {rnn_acc:.4f}")

    # ─── 逐类准确率 ────────────────────────────────────────────────────
    rnn_per_cls = evaluate_per_class(rnn_model, val_loader)
    print('\n  各位置准确率（「你」在第 X 位）：')
    print(f"  {'位置':>6}  {'模型A(RNN)':>12}  ")
    for i in range(NUM_CLASSES):
        print(f"  第 {i+1} 位   {rnn_per_cls[i]:>10.4f}   ")

    # ─── 结论 ──────────────────────────────────────────────────────────
    print(f"""
【结论】
  模型使用 RNN 逐字处理，第 t 步的隐藏状态编码了"前 t 字的上下文"，
  因此即使最终对时序做 MaxPool，池化结果仍携带位置信息，能准确判断"你"在哪位。

  → 语序信息对于位置感知类任务至关重要；RNN 通过顺序递推有效保留了这一信息。
""")

    # ─── 推理示例 ──────────────────────────────────────────────────────
    print('─── 推理示例（各模型预测「你」所在位置）──────────────────────')
    rnn_model.eval()
    random.seed(0)
    print(f"  {'句子':>6}  真实位置    模型(RNN)")
    with torch.no_grad():
        for pos in range(NUM_CLASSES):
            sent, label = make_sample(pos)
            ids         = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            rnn_pred    = rnn_model(ids).argmax(dim=1).item()
            rnn_mark    = '✓' if rnn_pred == label else '✗'
            print(f"  「{sent}」  第{label+1}位     "
                  f"{rnn_mark} 第{rnn_pred+1}位      ")


if __name__ == '__main__':
    main()
