# coding:utf8
# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
多分类任务：
输入5维向量，哪一维数字最大，就属于第几类（0/1/2/3/4）
"""

# ===================== 1. 模型定义（多分类专用）=====================
class TorchModel(nn.Module):
    def __init__(self, input_size, class_num):
        super(TorchModel, self).__init__()
        # 线性层：输入5维 → 输出5个分类得分
        self.linear = nn.Linear(input_size, class_num)
        # 多分类损失函数：交叉熵（自带softmax，不用手动加）
        self.loss = nn.CrossEntropyLoss() 

    # 前向传播
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch, 5) → (batch, 5)
        if y is not None:
            return self.loss(x, y)  # 训练：返回loss
        else:
            return torch.softmax(x, dim=1)  # 预测：转成概率

# ===================== 2. 生成多分类样本 =====================
def build_sample():
    """生成1个样本：哪一维最大，标签就是几"""
    x = np.random.random(5)
    y = np.argmax(x)  # 取最大值所在的下标（0-4）
    return x, y

def build_dataset(total_sample_num):
    """生成一批样本"""
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 标签必须是LongTensor

# ===================== 3. 多分类评估函数 =====================
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0

    with torch.no_grad():
        y_pred_prob = model(x)  # 输出概率 (100,5)
        y_pred = torch.argmax(y_pred_prob, dim=1)  # 取概率最大的下标

        for y_p, y_t in zip(y_pred, y):
            if y_p == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

# ===================== 4. 训练主函数 =====================
def main():
    # 超参数
    epoch_num = 20        # 训练轮数
    batch_size = 20       # 批次大小
    train_sample = 5000   # 总样本
    input_size = 5        # 输入5维
    class_num = 5         # 5分类
    learning_rate = 0.01

    # 创建模型
    model = TorchModel(input_size, class_num)
    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 构建训练集
    train_x, train_y = build_dataset(train_sample)

    # 开始训练
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]

            loss = model(x, y)   # 计算loss
            loss.backward()      # 反向传播
            optim.step()         # 更新权重
            optim.zero_grad()    # 清空梯度
            watch_loss.append(loss.item())

        print("=========\n第%d轮 平均loss:%.4f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "multi_class_model.pth")

    # 画图
    plt.plot([l[0] for l in log], label="acc")
    plt.plot([l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

# ===================== 5. 多分类预测 =====================
def predict(model_path, input_vecs):
    input_size = 5
    class_num = 5
    model = TorchModel(input_size, class_num)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        pred_prob = model(torch.FloatTensor(input_vecs))
        pred_class = torch.argmax(pred_prob, dim=1)

    for vec, cls, prob in zip(input_vecs, pred_class, pred_prob):
        print(f"输入：{vec}\n预测类别：{cls}，各类别概率：{prob.numpy()}\n")

# ===================== 运行 =====================
if __name__ == "__main__":
    main()

    # 训练完后可以解开下面注释，直接预测
    # test_vecs = [
    #     [0.9,0.1,0.2,0.3,0.1],
    #     [0.1,0.9,0.2,0.2,0.1],
    #     [0.2,0.1,0.9,0.1,0.1],
    # ]
    # predict("multi_class_model.pth", test_vecs)
