import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# ===================== 1. 固定随机种子（保证结果可复现）=====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"运行设备: {device}")

# ===================== 2. 构造模拟中文数据集（情感二分类）=====================
texts = [
    "这部电影很好看，剧情精彩", "演技太棒了，强烈推荐", "画面精美，观感极佳",
    "无聊至极，全程犯困", "剧情混乱，浪费时间", "演员演技尴尬，不推荐",
    "配乐好听，故事感人", "逻辑漏洞太多，看得难受", "整体一般，中规中矩",
    "良心佳作，值得二刷", "剪辑混乱，体验很差", "笑点密集，非常欢乐"
]
labels = [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1]  # 1:正面  0:负面

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.3, random_state=42
)

# 分词（简易按字分词，适配中文）
def char_tokenize(sent):
    return [c for c in sent if c.strip()]

train_tokens = [char_tokenize(t) for t in train_texts]
test_tokens = [char_tokenize(t) for t in test_texts]

# ===================== 方法1：TF-IDF + 逻辑回归 =====================
print("\n" + "="*60)
print("【方法1: TF-IDF + 逻辑回归】")
print("="*60)

tfidf = TfidfVectorizer(analyzer="char", max_features=2000)
train_tfidf = tfidf.fit_transform(train_texts)
test_tfidf = tfidf.transform(test_texts)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(train_tfidf, train_labels)
lr_pred = lr_model.predict(test_tfidf)
lr_acc = accuracy_score(test_labels, lr_pred)
print(f"准确率: {lr_acc:.4f}")

# ===================== 方法2：Word2Vec + LSTM =====================
print("\n" + "="*60)
print("【方法2: Word2Vec + LSTM】")
print("="*60)

# 训练Word2Vec词向量
w2v_model = Word2Vec(train_tokens, vector_size=64, window=3, min_count=1, epochs=10)
vocab = w2v_model.wv.key_to_index
vocab_size = len(vocab)
embed_dim = 64
max_len = 20

# 文本转索引序列
def text2seq(tokens_list, vocab, max_len):
    seqs = []
    for tokens in tokens_list:
        seq = [vocab.get(t, 0) for t in tokens[:max_len]]
        seq += [0] * (max_len - len(seq))
        seqs.append(seq)
    return np.array(seqs)

train_seq = text2seq(train_tokens, vocab, max_len)
test_seq = text2seq(test_tokens, vocab, max_len)
train_seq = torch.LongTensor(train_seq)
test_seq = torch.LongTensor(test_seq)
train_y = torch.LongTensor(train_labels)
test_y = torch.LongTensor(test_labels)

# 构造数据集
class SeqDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_loader = DataLoader(SeqDataset(train_seq, train_y), batch_size=4, shuffle=True)
test_loader = DataLoader(SeqDataset(test_seq, test_y), batch_size=4)

# LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 加载预训练Word2Vec权重
        emb_weight = torch.zeros(vocab_size, embed_dim)
        for word, idx in vocab.items():
            if word in w2v_model.wv:
                emb_weight[idx] = torch.from_numpy(w2v_model.wv[word])
        self.embedding.weight = nn.Parameter(emb_weight)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

lstm_model = LSTMClassifier(vocab_size, embed_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=1e-3)

# 训练
epochs = 20
for epoch in range(epochs):
    lstm_model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = lstm_model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

# 评估
lstm_model.eval()
all_pred = []
with torch.no_grad():
    for batch_x, _ in test_loader:
        batch_x = batch_x.to(device)
        logits = lstm_model(batch_x)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        all_pred.extend(pred)
lstm_acc = accuracy_score(test_labels, all_pred)
print(f"准确率: {lstm_acc:.4f}")

# ===================== 方法3：BERT 特征提取（冻结主干）=====================
print("\n" + "="*60)
print("【方法3: BERT 特征提取(冻结)】")
print("="*60)

bert_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(bert_name)

# BERT数据集
class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=32):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encode = self.tokenizer(
            text, padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )
        return {
            "input_ids": encode["input_ids"].flatten(),
            "attention_mask": encode["attention_mask"].flatten(),
            "label": torch.tensor(label)
        }

bert_train_ds = BertDataset(train_texts, train_labels, tokenizer)
bert_test_ds = BertDataset(test_texts, test_labels, tokenizer)
bert_train_loader = DataLoader(bert_train_ds, batch_size=4, shuffle=True)
bert_test_loader = DataLoader(bert_test_ds, batch_size=4)

# 冻结BERT模型
class BertFreezeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        # 冻结所有BERT参数
        for param in self.bert.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(768, 2)
    def forward(self, input_ids, attn_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        pool_out = out.pooler_output
        return self.classifier(pool_out)

bert_freeze_model = BertFreezeModel().to(device)
optimizer = optim.Adam(bert_freeze_model.parameters(), lr=1e-3)

# 训练
for epoch in range(10):
    bert_freeze_model.train()
    for batch in bert_train_loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        label = batch["label"].to(device)
        optimizer.zero_grad()
        logits = bert_freeze_model(input_ids, attn_mask)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

# 评估
bert_freeze_model.eval()
all_pred = []
with torch.no_grad():
    for batch in bert_test_loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        logits = bert_freeze_model(input_ids, attn_mask)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        all_pred.extend(pred)
bert_freeze_acc = accuracy_score(test_labels, all_pred)
print(f"准确率: {bert_freeze_acc:.4f}")

# ===================== 方法4：BERT 全量微调 =====================
print("\n" + "="*60)
print("【方法4: BERT 全量微调】")
print("="*60)

class BertFinetuneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.classifier = nn.Linear(768, 2)
    def forward(self, input_ids, attn_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        pool_out = out.pooler_output
        return self.classifier(pool_out)

bert_ft_model = BertFinetuneModel().to(device)
# 微调使用更小学习率
optimizer = optim.Adam(bert_ft_model.parameters(), lr=2e-5)

# 训练
for epoch in range(10):
    bert_ft_model.train()
    for batch in bert_train_loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        label = batch["label"].to(device)
        optimizer.zero_grad()
        logits = bert_ft_model(input_ids, attn_mask)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

# 评估
bert_ft_model.eval()
all_pred = []
with torch.no_grad():
    for batch in bert_test_loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        logits = bert_ft_model(input_ids, attn_mask)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        all_pred.extend(pred)
bert_ft_acc = accuracy_score(test_labels, all_pred)
print(f"准确率: {bert_ft_acc:.4f}")

# ===================== 汇总对比结果 =====================
print("\n" + "="*60)
print("【四种方法效果汇总】")
print("="*60)
print(f"1. TF-IDF + 逻辑回归      准确率: {lr_acc:.4f}")
print(f"2. Word2Vec + LSTM       准确率: {lstm_acc:.4f}")
print(f"3. BERT 冻结特征提取     准确率: {bert_freeze_acc:.4f}")
print(f"4. BERT 全量微调         准确率: {bert_ft_acc:.4f}")
