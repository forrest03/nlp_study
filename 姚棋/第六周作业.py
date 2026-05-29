# 1. 数据准备与预处理
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import time

# 加载数据集 (使用 HuggingFace 内置的 chnsenticorp)
dataset = load_dataset("seamew/ChnSentiCorp")   # 若无权限可用其他中文情感数据集
# 若加载失败，可手动从 https://github.com/SophonPlus/ChineseNlpCorpus 下载
# 此处假设已获得 train_texts, train_labels, test_texts, test_labels

# 为了可复现，这里用简化的结构示意，实际使用时请替换为真实数据加载代码
# 示例数据格式：
# train_texts = ["酒店环境很好", "服务太差", ...]
# train_labels = [1, 0, ...]  # 1:积极, 0:消极
# 我们模拟数据加载过程
def load_data():
    # 真实场景请取消注释下列代码
    # dataset = load_dataset("seamew/ChnSentiCorp")
    # train_texts = dataset["train"]["text"]
    # train_labels = dataset["train"]["label"]
    # test_texts = dataset["test"]["text"]
    # test_labels = dataset["test"]["label"]
    # return train_texts, train_labels, test_texts, test_labels
    
    # 为演示，生成示例数据（仅占位，实际运行时应使用真实数据）
    print("警告：当前为示例数据，请替换为真实 ChnSentiCorp 数据集")
    train_texts = ["这家酒店很干净", "房间太小了", "服务态度很好", "位置太偏"] * 250
    train_labels = [1, 0, 1, 0] * 250
    test_texts = ["性价比高", "热水不热"] * 250
    test_labels = [1, 0] * 250
    return train_texts, train_labels, test_texts, test_labels


# 2. 方法一：TF‑IDF + 逻辑回归
def train_tfidf_lr(train_texts, train_labels, test_texts, test_labels):
    start = time.time()
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, train_labels)
    preds = clf.predict(X_test)
    
    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average='binary')
    train_time = time.time() - start
    print(f"[TF-IDF+LR] 准确率: {acc:.4f}, F1: {f1:.4f}, 训练+推理时间: {train_time:.2f}s")
    return acc, f1, train_time


# 3. 方法二：TextCNN（深度学习）
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter

# 构建词表
def build_vocab(texts, max_size=5000):
    counter = Counter()
    for text in texts:
        counter.update(list(text))  # 字符级词表，也可用 jieba 分词
    vocab = {word: idx+2 for idx, (word, _) in enumerate(counter.most_common(max_size))}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

def encode(texts, vocab, max_len=128):
    ids = []
    for text in texts:
        seq = [vocab.get(ch, 1) for ch in text[:max_len]]
        seq += [0] * (max_len - len(seq))
        ids.append(seq)
    return torch.tensor(ids, dtype=torch.long)

class TextCNNDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=128):
        self.x = encode(texts, vocab, max_len)
        self.y = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, num_classes, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x).unsqueeze(1)  # (batch, 1, seq_len, embed_dim)
        conv_outs = []
        for conv in self.convs:
            c = torch.relu(conv(x)).squeeze(3)   # (batch, filters, seq_len - k + 1)
            pooled = torch.max(c, dim=2)[0]      # (batch, filters)
            conv_outs.append(pooled)
        out = torch.cat(conv_outs, dim=1)
        out = self.dropout(out)
        return self.fc(out)

def train_textcnn(train_texts, train_labels, test_texts, test_labels):
    # 构建词表
    vocab = build_vocab(train_texts, max_size=5000)
    vocab_size = len(vocab)
    max_len = 128
    batch_size = 64
    epochs = 10
    lr = 1e-3
    
    train_dataset = TextCNNDataset(train_texts, train_labels, vocab, max_len)
    test_dataset = TextCNNDataset(test_texts, test_labels, vocab, max_len)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TextCNN(vocab_size, embed_dim=100, num_filters=100, 
                    filter_sizes=[2,3,4], num_classes=2, dropout=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    start = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"TextCNN epoch {epoch+1}, loss: {total_loss/len(train_loader):.4f}")
    
    # 测试
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred)
            truths.extend(y.numpy())
    acc = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds, average='binary')
    train_time = time.time() - start
    print(f"[TextCNN] 准确率: {acc:.4f}, F1: {f1:.4f}, 训练+推理时间: {train_time:.2f}s")
    return acc, f1, train_time

# 4. 方法三：BERT 微调
def train_bert(train_texts, train_labels, test_texts, test_labels):
    model_name = "bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 自定义 dataset
    class BertDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=128):
            self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors='pt')
            self.labels = torch.tensor(labels, dtype=torch.long)
        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = self.labels[idx]
            return item
        def __len__(self):
            return len(self.labels)
    
    train_dataset = BertDataset(train_texts, train_labels, tokenizer)
    test_dataset = BertDataset(test_texts, test_labels, tokenizer)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir='./bert_results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=100,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
    )
    
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='binary')
        return {'accuracy': acc, 'f1': f1}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    start = time.time()
    trainer.train()
    train_time = time.time() - start
    
    # 最终测试
    pred_output = trainer.predict(test_dataset)
    acc = pred_output.metrics['test_accuracy']
    f1 = pred_output.metrics['test_f1']
    print(f"[BERT] 准确率: {acc:.4f}, F1: {f1:.4f}, 训练+推理时间: {train_time:.2f}s")
    return acc, f1, train_time


# 5. 运行主程序并对比结果
if __name__ == "__main__":
    # 加载真实数据
    train_texts, train_labels, test_texts, test_labels = load_data()
    print(f"训练集大小: {len(train_texts)}, 测试集大小: {len(test_texts)}")
    
    results = {}
    
    # 1. TF-IDF + LR
    acc, f1, t = train_tfidf_lr(train_texts, train_labels, test_texts, test_labels)
    results['TF-IDF+LR'] = (acc, f1, t)
    
    # 2. TextCNN
    acc, f1, t = train_textcnn(train_texts, train_labels, test_texts, test_labels)
    results['TextCNN'] = (acc, f1, t)
    
    # 3. BERT
    acc, f1, t = train_bert(train_texts, train_labels, test_texts, test_labels)
    results['BERT'] = (acc, f1, t)
    
    # 打印对比表格
    print("\n" + "="*60)
    print("模型对比结果")
    print("="*60)
    print(f"{'方法':<15} {'准确率':<10} {'F1分数':<10} {'时间(秒)':<10}")
    for name, (acc, f1, t) in results.items():
        print(f"{name:<15} {acc:.4f}       {f1:.4f}       {t:.2f}")

        
