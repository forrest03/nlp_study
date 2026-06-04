'''
专门用于创建样本的程序
'''
import numpy as np
import torch
def build_sample():
    '''
    这个方法创建样本，返回5维随机向量 x
    '''
    # 生成5维随机向量 x
    x = np.random.random(5)
    return x

#生成一批样本
def build_dataset():
    '''
    生成一个数据集，包含样本与标签
    '''
    x.append(build_sample())
    y.append(np.argmax(build_sample()))
    #返回张量与标签张量
    return x, y

def create_sample(num):
    '''
    创建一个样本，返回5维随机向量 x
    '''
    x = []
    y = []
    for i in range(num):
        x_ = build_sample()
        x.append(x_)
        y.append(np.argmax(x_))
    return torch.FloatTensor(np.array(x)), torch.LongTensor(np.array(y))

