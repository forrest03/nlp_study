# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，第几个数字最大，则对应为第几类样本。

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        # 增加一个隐藏层，把维度从 5 升到 10，再降到 5
        self.linear1 = nn.Linear(input_size, 10)
        self.relu = nn.ReLU()  # 引入非线性激活函数，增强找规律能力
        self.linear2 = nn.Linear(10, 5)
        # self.activation = torch.sigmoid  # nn.Sigmoid() sigmoid归一化函数
        # 多分类任务使用交叉熵损失函数
        self.loss = nn.CrossEntropyLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear1(x)
        x = self.relu(x)  # 隐藏层经过 ReLU 激活
        y_pred = self.linear2(x)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，第几个数字最大，则对应为第几类样本。
def build_sample():
    x = np.random.random(5)
    # np.argmax 返回数组中最大值的索引，正是我们的类别标签 (0, 1, 2, 3, 4)
    y = np.argmax(x)
    return x, y


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    print(X)
    print(Y)
    # 将 Y 转换为 LongTensor，因为交叉熵损失函数要求目标标签为整型
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)

    # 统计一下每个类别的样本分布
    from collections import Counter
    print("本次预测集中真实类别分布：", dict(Counter(y.tolist())))

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # y_pred的形状是 (100, 5)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            # y_p是一个5维向量，取出最大值的索引作为预测结果
            pred_class = torch.argmax(y_p)
            if int(pred_class) == int(y_t):
                correct += 1  # 预测正确
            else:
                wrong += 1  # 预测错误
        print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
        return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 100  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size): 
            #取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        pred_class = torch.argmax(res)  # 获取预测出的类别
        # 用softmax将得分转为概率（仅仅是为了打印好看）
        prob = torch.nn.functional.softmax(res, dim=0)
        print("输入：%s, 预测类别：%d, 该类概率值：%f" % (vec, pred_class, prob[pred_class]))


if __name__ == "__main__":
    main()
    test_vec = [[0,8,0,5,2],
                [5,2,0,1,3],
                [0,3,0,7,0],
                [6,2,7,1,3]]
    predict("model.bin", test_vec)
