import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


input_size = 5

# 创建模型
class DemoModel(nn.Module):
    def __init__(self,size):
        super().__init__()
        self.linear = nn.Linear(size,size) # 输入尺寸，输出尺寸
        self.loss = nn.functional.cross_entropy
        self.activation = nn.functional.softmax
        
    def forward(self,x,y=None):
        y_pred = self.linear(x)
        y_pred = self.activation(y_pred,dim=1)
        return self.loss(y_pred,y)

# 生成测试和训练数据
'''
    @size 输入数组尺寸
    @num 多少组数据
'''
def build_data(num):
    X = []
    Y = []
    for i in range(num):
        x = np.random.random(input_size)
        y = np.argmax(x) 
        # y = np.argmax(x) + 1
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X),torch.LongTensor(Y) # 转化为torch读的懂得的数据格式
        
# 测试模型
# @torch.no_grad
def evaluate(model):
    model.eval() # 告诉模型开始测试
    test_sample_num = 100
    x,y = build_data(test_sample_num)
    with torch.no_grad():
        loss = model(x,y)
        print('损失率',loss.item())
        return loss

def main():
    epoch_num = 40
    batch_size = 20
    train_sample = 5000
    learning_rate = 0.01
    model = DemoModel(input_size)
    optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
    log = []
    train_x,train_y = build_data(train_sample)

    for epoch in range(epoch_num):
        model.train() # 告诉模型开始训练
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x,y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc,float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()

main()
