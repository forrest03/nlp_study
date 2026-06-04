"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，第几维最大，就认为是第几类
"""
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

class TorchModel(nn.Module):
    def __init__(self):
        super(TorchModel, self).__init__()
        self.classify = nn.Linear(5, 5)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return torch.softmax(y_pred, axis=-1)

# 随机生成一个5维向量，一个随机向量，哪一维数字最大就属于第几类
def build_sample():
    x = np.random.random(5)
    return x, np.argmax(x)

# 随机生成样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(np.asarray(X)), torch.LongTensor(np.asarray(Y))

# 训练代码
def main():
    learning_rate = 0.01   # 学习率
    epoch_num = 50         # 最大轮数
    train_data_size = 900  # 训练数据量
    batch_size = 16        # 分片大小
    patience = 7           # 若没达到最大轮数，acc在patience轮没上升就停止
    # 构造模型
    model = TorchModel()
    # 构造优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    plots = []
    maxcount, count = 0, 0
    # 训练模型
    for epoch in range(epoch_num):
        watch_loss = []
        model.train()
        x_data, y_data = build_dataset(train_data_size)
        for batch_index in range(train_data_size // batch_size):
            x = x_data[batch_index * batch_size : (batch_index + 1) *batch_size]
            y = y_data[batch_index * batch_size : (batch_index + 1) *batch_size]
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())
        print("epoch: %d, loss: %f" % (epoch + 1, np.mean(watch_loss)))
        # 计算正确率
        acc = evaluate(model)
        # 记录画图数据
        plots.append([acc, np.mean(watch_loss)])
        # 早停
        if acc > maxcount:
            maxcount = acc
            count = 0
        else:
            count += 1
        if count > patience:
            break;
    # 保存模型
    torch.save(model, "model.bin")
    # 画图
    plt.plot(range(len(plots)), [plot[0] for plot in plots], label="acc")  # 画acc曲线
    plt.plot(range(len(plots)), [plot[1] for plot in plots], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

# 测试代码
def evaluate(model):
    test_data_size = 100  # 测试数据量
    x, y = build_dataset(test_data_size)
    model.eval()
    correct, wrong = 0, 0
    # 测试不更新梯度
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y in zip(y_pred, y):
            if np.argmax(y_p) == y:
                correct += 1
            else:
                wrong += 1
    print("样本总数：%d, 正确预测个数：%d, 正确率：%f" %(test_data_size, correct, correct / test_data_size))
    return correct / test_data_size

if __name__ == "__main__":
    main()
