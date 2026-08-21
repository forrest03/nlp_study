#coding:utf8
"""
设计一个以文本为输入的多分类任务，对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#生成数据集
class MyDataset(Dataset):
    def __init__(self, n_samples=1000):
        self.samples = []
        for _ in range(n_samples):
            pos = random.randint(0, 4)
            text = ['你' if i == pos else 'X' for i in range(5)]
            label = pos
            self.samples.append((''.join(text), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

#词表构建与编码
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab
def encode(sent, vocab, maxlen=5):
    ids  = [vocab.get(ch, 1) for ch in sent]
    ids  = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids

#模型定义
class KeywordRNN(nn.Module):
    """
    中文关键词分类器（RNN + MaxPooling 版）
    架构：Embedding → RNN → MaxPool → BN → Dropout → Linear → Sigmoid → (MSELoss)
    """
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # (B, L) -> (B, L, D)
        x, _ = self.rnn(x)     # (B, L, D) -> (B, L, H)
        x = torch.max(x, dim=1)[0]  # MaxPool over time: (B, H)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)         # (B, H) -> (B, C)
        return x

#训练与评估
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for texts, labels in dataloader:
        texts = torch.stack([torch.tensor(encode(t, vocab)) for t in texts]).to(device)
        labels = labels.to(device)
        outputs = model(texts)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for texts, labels in dataloader:
            texts = torch.stack([torch.tensor(encode(t, vocab)) for t in texts]).to(device)
            labels = labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

if __name__ == '__main__':
    #生成数据
    dataset = MyDataset()
    vocab = build_vocab(dataset.samples)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    #模型、损失函数、优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KeywordRNN(vocab_size=len(vocab)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #训练
    for epoch in range(10):
        train_loss = train(model, dataloader, criterion, optimizer, device)
        print(f'Epoch {epoch+1}, Loss: {train_loss:.4f}')

    #评估
    test_loss = evaluate(model, dataloader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}')

print("\n--- 推理示例 ---")
with torch.no_grad():
    for i in range(5):
        text = ''.join(['你' if j == i else 'X' for j in range(5)])
        input_tensor = torch.tensor(encode(text, vocab)).unsqueeze(0).to(device)
        output = model(input_tensor)
        pred = output.argmax(dim=1).item()
        print(f'输入: {text}, 预测: {pred}')
