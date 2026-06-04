import torch
import torch.nn as nn
import torch.optim as optim

# 中文训练文本
text = """
今天天气很好
我喜欢人工智能
人工智能改变世界
你好世界
你好人工智能
机器学习很强大
transformer算法非常厉害
gpt模型可以生成文本
"""

# 中文字符 tokenizer
chars = sorted(list(set(text)))

vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

# 超参数
block_size = 16
batch_size = 8

embed_size = 128

num_heads = 4
num_layers = 2

epochs = 1000

lr = 1e-3

# 获取 batch
def get_batch():

    ix = torch.randint(
        len(data) - block_size - 1,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i+block_size]
        for i in ix
    ])

    y = torch.stack([
        data[i+1:i+block_size+1]
        for i in ix
    ])

    return x, y

# GPT 模型
class GPTModel(nn.Module):

    def __init__(self):

        super().__init__()

        # token embedding
        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_size
        )

        # position embedding
        self.position_embedding = nn.Embedding(
            block_size,
            embed_size
        )

        # transformer
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.ln = nn.LayerNorm(embed_size)

        self.fc = nn.Linear(
            embed_size,
            vocab_size
        )

    def forward(self, x, targets=None):

        B, T = x.shape

        token_emb = self.token_embedding(x)

        pos = torch.arange(T)

        pos_emb = self.position_embedding(pos)

        x = token_emb + pos_emb

        # 单向 mask
        mask = torch.triu(
            torch.ones(T, T) * float('-inf'),
            diagonal=1
        )

        x = self.transformer(
            x,
            mask=mask
        )

        x = self.ln(x)

        logits = self.fc(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits = logits.view(B*T, C)

            targets = targets.view(B*T)

            loss = nn.functional.cross_entropy(
                logits,
                targets
            )

        return logits, loss

# 初始化模型
model = GPTModel()

optimizer = optim.AdamW(
    model.parameters(),
    lr=lr
)

# 训练
for epoch in range(epochs):

    x, y = get_batch()

    logits, loss = model(x, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch {epoch}, loss = {loss.item():.4f}")

# 文本生成
model.eval()

start_text = "人工"

context = torch.tensor(
    [encode(start_text)],
    dtype=torch.long
)

for _ in range(50):

    x = context[:, -block_size:]

    logits, _ = model(x)

    logits = logits[:, -1, :]

    probs = torch.softmax(logits, dim=-1)

    next_token = torch.multinomial(
        probs,
        num_samples=1
    )

    context = torch.cat(
        [context, next_token],
        dim=1
    )

result = decode(context[0].tolist())

print("\n生成结果：")
print(result)


运行结果：
epoch 0, loss = 3.9293
epoch 100, loss = 0.0901
epoch 200, loss = 0.0484
epoch 300, loss = 0.0666
epoch 400, loss = 0.1006
epoch 500, loss = 0.0566
epoch 600, loss = 0.0767
epoch 700, loss = 0.0229
epoch 800, loss = 0.0868
epoch 900, loss = 0.0513

生成结果：
人工智能改变世界
你好世界
你好人工智能
机器学习很强大
transformer算法非常厉害
gpt模型
