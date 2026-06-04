'''
主模型程序
本周作业是给出海量5维向量，推理出哪个向量最大，就把这个向量值所在的索引打印出来
实际是一个五分类的输出维度，即最后的输出层是5个
'''
import torch
import torch.nn as nn
import numpy as np
#老师，把创建样本的函数放在一起有点乱，我把创建样本的函数单独放到CreateSample.py文件中
import CreateSample as cs
#我把准确率检测也放在外面了，import TestModel as tm
import TestModel as tm
import os

#构建模型类,只做训练
class CnnWeek2(nn.Module):

    def __init__(self):
        super(CnnWeek2, self).__init__()
        # 增加隐藏层
        #这里折腾了挺久，刚开始还是以宋老师的一层神经网络为基础训练，纠结了很久，意识到可以增加多个隐藏层
        self.l1 = nn.Linear(5, 128)  # 第一层：输入5维，升维到20维
        self.l2 = nn.Linear(128, 128) # 第二层：隐藏层，保持20维
        self.l3 = nn.Linear(128, 128)  # 第三层：输出层，降回5维分类
        self.l4 = nn.Linear(128, 5)  # 第四层：输出层，降回5维分类
        # 激活函数，加入非线性因素
        self.activation = nn.ReLU()
        # 定义损失函数loss为交叉熵损失函数，这里也使用了很长时间， 终于知道这种回归任务用交叉熵损失函数，ppt里宋老师讲了
        self.loss = nn.CrossEntropyLoss()

    #再定义前向传播
    def forward(self, x, y=None):
        # 前向传播，经过多层网络和激活函数
        x = self.l1(x)
        x = self.activation(x)
        x = self.l2(x)
        x = self.activation(x)
        x = self.l3(x)
        x = self.activation(x)
        y_p = self.l4(x) # 最后一层不需要激活，直接给交叉熵
        
        #如果传入了真实标签y，说明是在训练，返回损失值
        if y is not None:
            return self.loss(y_p, y)
        #如果没传y，说明是在预测，直接返回预测得分
        else:
            return y_p


def main():
    '''
    开始训练模型
    '''
    #加个训练的期望损失值
    expect_loss = 0.01
    #第一步，实例化模型
    model = CnnWeek2()
    #第二部，选择好使用的优化器 lr是学习率
    lr = 0.001
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    #第三步加载训练集，用5000条训练集训练
    train_sample = 5000
    train_x, train_y = cs.create_sample(train_sample)
    # print(train_x)
    # print(train_y)
    #第四步，标记训练的轮数
    epoch_num = 200
    #第五步，记录每次训练的样本个数
    batch_size = 32
    #第七步，训练模型
    for epoch in range(epoch_num):
        print("开始第%d轮训练" % (epoch + 1))
        model.train()#固定要求，切换到训练模式
        #取出这一轮的训练样本以及标签，每个batch_size个样本为一个batch
        #共train_sample // batch_size个batch
        #每个batch_size个样本为一个batch
        for batch_index in range(train_sample // batch_size):
            #取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            #前向传播，得到预测值y_p
            loss = model(x,y)#得出损失值
            loss.backward()#反向传播
            #使用优化器更新权重
            optim.step()
            #梯度归零
            optim.zero_grad()
        #第八步，测试模型
        accuracy = tm.test_model_train(model, train_x, train_y)
        print("第%d轮训练模型在训练集上的准确率为: %d%%" % (epoch + 1, accuracy * 100))
        #打印损失值
        print("第%d轮训练损失值为%f" % (epoch + 1, loss.item()))
        if loss.item() < expect_loss:
            print(f"损失值小于期望值{expect_loss}，训练结束")
            break
    #要保存模型参数了
    #获取当前py的路径
    current_path = os.path.dirname(os.path.abspath(__file__))
    torch.save(model.state_dict(), current_path + "/CnnWeek2.m")



def test_model_final(input_vec):
    '''
    测试模型,用来再测试集上测试模型的准确率
    '''
    #加载模型参数，带上当前目录
    model = CnnWeek2()
    #加载模型参数
    current_py_path = os.path.dirname(__file__)
    model.load_state_dict(torch.load(os.path.join(current_py_path, "CnnWeek2.m")))
    #设置测试模型
    model.eval()
    #关闭梯度计算
    with torch.no_grad():
        #模型推算
        y_p = model.forward(input_vec)
        # 获取这5个维度中得分最高（最大值）的那个维度的索引就是预测的类别了
        pre_index = torch.argmax(y_p).item()
        pre_class = pre_index
        print("*********")
        print("输入向量：%s" % [round(v, 4) for v in input_vec.tolist()])
        print("模型预测输出：%s" % [round(v, 4) for v in y_p.tolist()])
        print("最终预测类别：属于第 %d 类" % pre_class)
        print("*********")

if __name__ == "__main__":
    main()
    # 测试模型，测试10个样本的预测结果
    test_model_final(torch.FloatTensor([0.1, 0.22, -0.3, 0.1, 0.5]))
    test_model_final(torch.FloatTensor([0.2, 0.1, 0.4, 0.3, 0.6]))
    test_model_final(torch.FloatTensor([0.3, 0.2, 0.5, 0.4, 0.7]))
    test_model_final(torch.FloatTensor([0.4, 0.3, 0.6, 0.5, 0.8]))
    test_model_final(torch.FloatTensor([0.5, 0.4, 0.7, 0.6, 0.9]))
    test_model_final(torch.FloatTensor([0.6, 0.5, 0.8, 0.7, 0.1]))
    test_model_final(torch.FloatTensor([0.7, 0.6, 0.9, 0.8, 0.2]))
    test_model_final(torch.FloatTensor([0.8, 0.7, 0.1, 0.9, 0.3]))
    test_model_final(torch.FloatTensor([0.9, 0.8, 0.2, 0.1, 0.4]))
