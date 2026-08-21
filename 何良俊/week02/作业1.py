import os
# 解决OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 设置随机种子，保证结果可重现
torch.manual_seed(42)
np.random.seed(42)

# 超参数设置
input_dim = 5  # 输入向量维度
output_dim = 5  # 输出类别数（与输入维度相同）
batch_size = 32
accept_loss = 0.01
learning_rate = 0.1
epoches = 1000

# 生成训练数据
def generate_data(num_samples, input_dim):
    """生成随机向量和对应的标签"""
    # 生成随机向量 (num_samples, input_dim)
    data = np.random.randn(num_samples, input_dim)
    # 标签是向量中最大值的索引 (num_samples,)
    labels = np.argmax(data, axis=1)
    return data, labels

# 生成训练和测试数据
train_data, train_labels = generate_data(10000, input_dim)
test_data, test_labels = generate_data(100, input_dim)

# 转换为PyTorch张量
train_data = torch.tensor(train_data, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_data = torch.tensor(test_data, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# 创建数据加载器
train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定义神经网络模型
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 初始化模型、损失函数和优化器
model = SimpleClassifier(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练模型
def train(model, train_loader, criterion, optimizer, accept_loss):
    model.train()
    epoch = 0
    for epoch in range(epoches):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_data, batch_labels in train_loader:
            # 前向传播
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 计算统计信息
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        # 打印每个epoch的信息
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch [{epoch+1}], Loss: {epoch_loss}, correct: {correct}, total: {total}, Accuracy: {epoch_acc:.2f}%')
        if epoch_loss < accept_loss:
            break
        
# 测试模型
def test(model, test_data, test_labels):
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        loss = criterion(outputs, test_labels)
        _, predicted = torch.max(outputs.data, 1)
        total = test_labels.size(0)
        correct = (predicted == test_labels).sum().item()
        
        test_loss = loss.item()
        test_acc = 100 * correct / total
        print(f'Test Loss: {test_loss}, test_correct: {correct}, test_total: {total}, Accuracy: {test_acc:.2f}%')

# 运行训练和测试
if __name__ == "__main__":
    print("开始训练...")
    train(model, train_loader, criterion, optimizer, accept_loss)
    print("\n开始测试...")
    test(model, test_data, test_labels)
    
    # 展示一些预测结果
    print("\n ---------------------------------")
    print("\n测试样本的预测结果：")
    with torch.no_grad():
        sample_outputs = model(test_data)
        _, sample_predicted = torch.max(sample_outputs, 1)
        print("样本输入:")
        print(test_data)
        print("真实标签:")
        print(test_labels)
        print("预测标签:")
        print(sample_predicted)
        print("sample_outputs:")
        print(sample_outputs)

        mask = sample_predicted != test_labels
        wrong_data = test_data[mask]
        wrong_labels = test_labels[mask]
        wrong_predicted = sample_predicted[mask]
        print(f"\n预测错误的样本数量：{len(wrong_data)}")
        for i in range(len(wrong_data)):
            print(f"样本 {i+1}:")
            print(f"  输入数据: {wrong_data[i].numpy()}")
            print(f"  真实标签: {wrong_labels[i].item()}")
            print(f"  预测标签: {wrong_predicted[i].item()}")
