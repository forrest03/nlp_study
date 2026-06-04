
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn     
import numpy as np
import random
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 5)
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        y_pred = self.model(x)
        if y is not None:
            return self.loss(y_pred, y.long())
        return y_pred
            
def build_sample():
    num = random.randint(2,5) #随机生成维度
    x = np.random.random(num) # 随机生成一个维度的向量
    x = np.pad(x, (0, 5-len(x))) #长度固定，短的补齐
    max_index = 0
    max_num = x[0]
    for i in range(0,num):
        if x[i] > max_num:
            max_num = x[i]
            max_index = i
    
    return x,max_index

def build_data_set(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x,y = build_sample()
        X.append(x)
        Y.append(y)
    # Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_new.cpp:256.)
    # return torch.FloatTensor(X),torch.FloatTensor(Y)
    # X 是一个 “numpy数组的列表”（list of numpy arrays）
    # PyTorch 在转换时：
    #     * 会一个一个处理 → 很慢
    #     * 所以给你警告 ⚠️
    # 在转 Tensor 之前，先用 numpy 统一成一个大数组：
    return torch.FloatTensor(np.array(X)),torch.FloatTensor(np.array(Y))

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_data_set(test_sample_num)

    correct = 0
    with torch.no_grad():
        y_pred = model(x)
        pred = torch.argmax(y_pred, dim=1)

        for p, t in zip(pred, y.squeeze()):
            if int(p) == int(t):
                correct += 1

    acc = correct / test_sample_num
    print("准确率：", acc)
    return acc

def main():
    # 配置参数
    epoch_num = 50 # 训练轮数
    batch_size = 20 # 每次训练样本数
    train_sample = 5000 # 总共样本总数
    learning_rate = 0.0008 # 学习率
    input_size = 5 #向量维度

    # 建立模型
    model = TorchModel(input_size=input_size)
    log = []
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(),lr=learning_rate)

    train_x,train_y = build_data_set(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]
            loss = model(x,y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    torch.save(model.state_dict(),"./hub-TroE/李强/week02/model.bin")


def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(input_vec)
        result = model(x)
        pred = torch.argmax(result, dim=1)
    for vec, p in zip(input_vec, pred):
        print("输入：%s, 预测类别：%d" % (vec, int(p)))

if __name__ == "__main__":
    main()
    test_vec = np.random.random((10, 5)).tolist()
    predict("./hub-TroE/李强/week02/model.bin", test_vec)