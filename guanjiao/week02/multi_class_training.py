# coding: utf-8
"""
多分类训练任务：
给模型输入一个随机向量，向量中哪个维度的数字最大，就属于哪一类。
例如 [1, 8, 3, 2, 5] 的最大值在下标 1，因此类别为 1。
该任务的期望值(标签)是“类比编号”，不是向量
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 定义文件路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "week2"
MODEL_PATH = MODEL_DIR / "multi_class_model.bin"
PLOT_PATH = MODEL_DIR / "multi_class_training.png"

# 生成一个样本数据
# x为num_feature维的随机向量
# y为最大值所在的下标，作为类别
def build_sample(num_feature):
    x = np.random.randint(0, 1000, num_feature)

    # np.argmax(x)返回x中最大的值的索引
    y = int(np.argmax(x))
    return x, y

# 生成批量样本做训练或测试集
def build_dataset(total_sample_num, num_feature):
    X = []
    Y = []

    for _ in range(total_sample_num):
        x, y = build_sample(num_feature)
        X.append(x)
        Y.append(y)
    
    """
    输出X.shape= [total_sample_sum, num_feature]
    输出Y为类别值组成的一维数组(ex: [1, 2, 0, 0, 1, 2, 1, 3, 0, 2])
    Y.shape=[total_sample_sum]
    """
    return torch.LongTensor(np.array(X)), torch.LongTensor(np.array(Y))

# print(build_dataset(10, 4))

# 定义模型
# 多分类任务用 CrossEntropyLoss 时，标签 y 通常直接用类别编号，不需要 one-hot。
# 深度学习目的调整参数
# 深度学习链路：输入---> 线性层+Activation函数---> 计算loss---> 反向传播---> 更新参数
class TorchModel(nn.Module):
    def __init__(self, num_feature):
        super().__init__()
        # nn.Linear(input_size, output_size)
        self.linear = nn.Linear(num_feature, num_feature)

        # Softmax目的将线性层的输出，即每个类别的分数，归一化，变成可解释的类别概率。
        self.activation = nn.Softmax(dim=1)
        self.loss = F.cross_entropy

    def forward(self, x, y=None):
        # 将数据样本压缩到0~1之间，避免整数值过大导致训练不稳定。
        # 对每个样本(x的每一行)求最大值
        # keepdim=True 保持维度，因为x是二维数组，max(dim=1)返回的是一维数组，所以需要keepdim=True
        # clamp_min(1e-8) 防止除以0
        x = x / x.max(dim=1, keepdim=True).values.clamp_min(1e-8)
        logits = self.linear(x)

        if y is not None:
            return self.loss(logits, y)

        return self.activation(logits)

# 评估模型
def evaluate(model, test_sample_num=1000):
    """评估模型准确率。"""
    model.eval()
    num_feature = model.linear.in_features
    test_x, test_y = build_dataset(test_sample_num, num_feature)

    correct = 0
    with torch.no_grad():
        y_pred = model(test_x)
        for y_p, y_t in zip(y_pred, test_y):
            if int(y_p.argmax()) == int(y_t):
                correct += 1

    return correct / test_sample_num

# 训练模型
def train(num_feature, epoch_num=100, batch_size=20, train_sample=5000, learning_rate=0.01):
    """训练模型并保存权重。"""
    # torch.manual_seed(42)
    # np.random.seed(42)

    model = TorchModel(num_feature)

    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_x, train_y = build_dataset(train_sample, num_feature)

    # 记录训练过程中的准确率和损失
    log = []

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        permutation = torch.randperm(train_sample)
        train_x = train_x[permutation]
        train_y = train_y[permutation]

        for batch_index in range(train_sample // batch_size):
            start_index = batch_index * batch_size
            end_index = start_index + batch_size
            batch_x = train_x[start_index:end_index]
            batch_y = train_y[start_index:end_index]

            # 计算损失
            loss = model(batch_x, batch_y)

            # 反向传播计算梯度
            loss.backward()

            # 更新参数
            optimizer.step()

            # 清空梯度
            optimizer.zero_grad()

            # 记录损失
            watch_loss.append(loss.item())

        # 计算平均损失
        avg_loss = float(np.mean(watch_loss))

        # 评估模型
        acc = evaluate(model)
        log.append([acc, avg_loss])
        print("=========\n第%d轮平均loss:%f，测试准确率:%f" % (epoch + 1, avg_loss, acc))

    MODEL_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    draw_training_curve(log)
    return model

# 绘制训练曲线
def draw_training_curve(log):
    """保存训练曲线。"""
    plt.plot(range(1, len(log) + 1), [item[0] for item in log], label="acc")
    plt.plot(range(1, len(log) + 1), [item[1] for item in log], label="loss")
    # 显示label 说明是每条线代表什么
    plt.legend()
    plt.show()

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    """使用训练好的模型做预测。"""
    num_feature = len(input_vec[0])
    model = TorchModel(num_feature)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        result = model(torch.FloatTensor(input_vec))

    for vec, res in zip(input_vec, result):
        print(
            "输入:%s，预测类别:%d，概率值:%f"
            % (vec, int(res.argmax()) + 1 , float(res.max()))
        )


if __name__ == "__main__":
    train(5)
    predict(
        MODEL_PATH,
        [[1, 8, 3, 2, 5], [99, 100, 98, 97, 1000], [200, 32, 45, 67, 890]],
    )
