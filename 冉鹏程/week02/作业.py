import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self, n, input_size, hide_size, output_size):
        """
        :param n: 神经网络层级
        :param input_size: 输入维度
        :param hide_size: 隐藏层维度
        :param output_size: 输出层维度
        """
        super(TorchModel, self).__init__()
        if n <= 0:
            raise Exception('n 必须大于0')
        self.n = n
        if n == 1:
            self.linear1 = nn.Linear(input_size, output_size)
        else:
            for i in range(1, n + 1):
                att_name = f'linear{i}'
                if i == 1:
                    setattr(self, att_name, nn.Linear(input_size, hide_size))
                elif i == n:
                    setattr(self, att_name, nn.Linear(hide_size, output_size))
                else:
                    setattr(self, att_name, nn.Linear(hide_size, hide_size))
        ### 激活函数 ###
        # self.activate = torch.sigmoid
        self.activate = torch.relu

        ### 损失函数 ###
        self.loss = nn.functional.mse_loss #均方差
        # self.loss = nn.functional.cross_entropy  # 交叉熵

    def forward(self, x, y=None):
        y_hat = x
        for i in range(1, self.n + 1):
            att_name = f'linear{i}'
            y_hat = getattr(self, att_name)(y_hat)
            # y_hat = self.activate(y_hat)

        if y is not None:
            return self.loss(y_hat, y)
        return y_hat


def max_val_idx(x):
    idx = 0
    for i in range(1, len(x)):
        if x[i] > x[idx]:
            idx = i
    return idx


def get_dataset(size_, dimension):
    X = []
    Y = []
    for _ in range(size_):
        x = [random.random() for _ in range(dimension)]
        y = [0 for _ in range(dimension)]
        y[max_val_idx(x)] = 1
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)


train_data_size = 1000
test_data_size = 200
input_size = 5
hide_size = 8
output_size = 5
epochs = 50
batch_size = 10
lr = 0.01
model = TorchModel(1, input_size, hide_size, output_size)

optim = torch.optim.SGD(model.parameters(), lr=lr)  # 优化器

X, Y = get_dataset(train_data_size, input_size)

def main():
    model.train()
    log = []
    watch_loss = []
    for epoch in range(epochs):
        for idx in range(0, len(X), batch_size):
            train_x = X[idx:min(idx + batch_size, len(X))]
            train_y = Y[idx:min(idx + batch_size, len(X))]
            loss = model.forward(train_x, train_y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        acc = evaluate()

        print(f'第{epoch}轮，loss: {np.mean(watch_loss)}')
        log.append([np.mean(watch_loss), acc])

    plt.plot(range(len(log)), [i[0] for i in log], label='loss')
    plt.plot(range(len(log)), [i[1] for i in log], label='acc')
    plt.legend()
    plt.show()

def evaluate():
    test_x , test_y = get_dataset(test_data_size, input_size)
    model.eval()
    y_hat = model.forward(test_x)
    correct_cnt = 0
    for i in range(test_data_size):
        if max_val_idx(test_y[i]) == max_val_idx(y_hat[i]):
            correct_cnt += 1
    print(f'准确率：{correct_cnt / test_data_size}')
    return correct_cnt / test_data_size



if __name__ == '__main__':
    main()
    # evaluate()
