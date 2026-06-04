import torch
import torch.nn as nn
import torch.optim as optim

# 设置参数
input_dim = 10
num_classes = 10
num_samples = 5000
epochs = 50
batch_size = 64

# 生成数据
X = torch.rand(num_samples, input_dim)

# y 是标签：哪一维最大，就属于哪一类
y = torch.argmax(X, dim=1)

# 定义模型
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.model(x)

model = SimpleClassifier()

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练
for epoch in range(epochs):
    permutation = torch.randperm(num_samples)
    total_loss = 0

    for i in range(0, num_samples, batch_size):
        indices = permutation[i:i + batch_size]
        batch_X = X[indices]
        batch_y = y[indices]

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 测试
test_X = torch.rand(10, input_dim)
test_y = torch.argmax(test_X, dim=1)

with torch.no_grad():
    predictions = model(test_X)
    predicted_classes = torch.argmax(predictions, dim=1)

print("\n测试向量：")
print(test_X)

print("\n真实类别：")
print(test_y)

print("\n预测类别：")
print(predicted_classes)

accuracy = (predicted_classes == test_y).float().mean()
print("\n准确率：", accuracy.item())
