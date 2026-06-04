import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

# 构造模型
class ModelDemo(nn.Module):
    def __init__(self, input_size, output_size):
        super(ModelDemo, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y = None):
        logits = self.linear(x) #(batch_size, input_size) -> (batch_size, output_size)
        if y is not None:
            loss = self.loss(logits, y) # 通过预测值和真实值计算损失
            return loss 
        else:
            return logits # 输出预测结果


# 生成一个样本
# 随机生成一个5维向量，哪一个维度大就属于哪一类
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x) #返回最大值下标
    return x, y

#随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)





#测试代码
# 每个epoch过后测试模型准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    with torch.no_grad():
        logits = model(x)
        y_pred = torch.argmax(logits, dim=1) # dim = 1 每一行找最大
        correct = (y_pred == y).sum().item()
        acc = correct / test_sample_num

    print(f"准确率: {acc:.2%}")
    return acc


def main():
    # 配置参数
    epoch_num = 20 # 训练轮数
    batch_size = 20 # 每次训练的样本个数
    train_sample = 800 # 每轮训练总共的样本数
    input_size = 5 # 输入向量维度
    output_size = 5 # 输出向量维度
    learning_rate = 0.01 # 学习率
    # 实例化模型
    model = ModelDemo(input_size, output_size)
    # 实例化参数更新方法(需要传入模型的参数以及学习率)
    optim = torch.optim.Adam(model.parameters(), lr = learning_rate)
    # 日志列表
    log = []
    # 创建训练数据集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = [] # 记录损失
        for batch_index in range(train_sample // batch_size): # 每个epoch总共有多少batch
            #取出一个batch的数据作为输入
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size] #(batch_size, input_size)            
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size] #(batch_size, )          
            loss = model(x, y)  # 计算loss
            optim.zero_grad()           # 梯度清零
            loss.backward()             # 计算梯度 
            optim.step()                # 更新权重 
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)   #测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label = "acc") # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label = "loss") # 画loss曲线
    plt.legend()
    plt.show()
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    output_size = 5
    model = ModelDemo(input_size, output_size)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(input_vec))
        pred_y = torch.argmax(logits, dim=1)
    for vec, res in zip(input_vec, pred_y):
        print("输入: %s, 预测类别： %d" % (vec, res.item()))    

if __name__ == "__main__":
    main()
    test_vec = [[0.88889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.bin", test_vec)

