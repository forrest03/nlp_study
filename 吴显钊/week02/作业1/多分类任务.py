
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


"""
一个多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类
"""

class TorchModel(nn.Module):
    def __init__(self, input_size = 2):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, input_size)  # 线性层
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵

    def forward(self, x, y=None):
        x = self.linear(x)  
        if y is not None:
            y = torch.argmax(y, dim=1)
            return self.loss(x, y)  # 预测值和真实值计算损失
        else:
            return torch.softmax(x, dim=1)  # 输出预测结果
        
# 随机生成一个n维向量 输出向量 和 最大值类别
def build_sample(n):
    x = np.random.random(n)
    index = np.argmax(x)
    y = np.zeros(n)
    y[index] = 1
    return x, y

# 随机生成一批样本
def build_dataset(total_sample_num, n):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample(n)
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, n):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num, n)
    y_pred = model(x)  # 模型预测 model.forward(x)
     # 每个样本取最大概率下标
    pred_classes = torch.argmax(y_pred, dim=1)
    true_classes = torch.argmax(y, dim=1)
    correct = (pred_classes == true_classes).sum().item()
    wrong = test_sample_num - correct
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 配置参数
    epoch_num = 200  # 训练轮数
    batch_size = 200  # 每次训练样本个数
    train_sample = 9000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample, input_size)
    print(train_x, train_y)
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
        acc = evaluate(model,input_size)  # 测试本轮模型结果
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
    input_size = len(input_vec[0])
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        index = torch.argmax(res)
        print("输入：%s, 预测类别：%s, 概率值：%s\n概率集合:%s\n" % (vec,  index, res[index],res))  # 打印结果


if __name__ == "__main__":
    main()
    # test_vec = [[0.88889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("model.bin", test_vec)
