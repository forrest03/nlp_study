import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


'''
多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类.

遇到的问题和解决方法:
1. 分批训练: 使用 DataLoader 进行分批训练，而不是一次性传入全部数据
   - 使用 TensorDataset 封装数据: dataset = TensorDataset(x, y)
   - 使用 DataLoader 批量化加载: dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
   - 训练循环中遍历: for batch_x, batch_y in dataloader

2. 多分类任务的输出层设计:
   - 输出层神经元数量 = 类别数量（本例中 dim=5，输出5个神经元的向量）
   - 输出层不需要 Sigmoid 等激活函数，CrossEntropyLoss 内部会自动进行 softmax 归一化处理
   - 模型输出形状: (batch_size, num_classes)，即 (20, 5)

3. 多分类任务的损失函数选择:
   - 使用 nn.CrossEntropyLoss() 而不是 BCEWithLogitsLoss()
   - BCEWithLogitsLoss() 是二分类损失函数，不适合多分类
   - CrossEntropyLoss() 的 target 是类别索引 (batch_size,)，不需要 one-hot 编码

4. 评估模型时 logits 到类别索引的转换:
   - 模型输出是 logits 形状 (n, num_classes)，包含每个类别的原始得分
   - 需要使用 torch.argmax(y_pred, dim=1) 提取预测的类别索引
   - 不能直接用 y_pred == y_t 进行比较，会报错: "Boolean value of Tensor with more than one value is ambiguous"

5. torch.mean() 参数类型问题:
   - watch_loss.append(loss.item()) 存储的是 Python float 的 list
   - torch.mean() 需要接收 tensor，不能直接传 list
   - 解决方法: torch.tensor(watch_loss).mean() 或使用 sum(watch_loss) / len(watch_loss)

6. 广播机制:
   - y = W @ x + b 中，b 会自动广播到 W @ x 的形状
   - 需要注意矩阵乘法的维度匹配: (m, n) @ (n,) = (m,)
'''

# 生成模拟数据
def build_data(num: int, dim: int):
    # x = torch.randint(0, 10, (8, ))
    x = torch.rand(num, dim)
    _ , y = torch.max(x, dim=1)
    # print(x)
    # print(y)
    return x, y


def evaluate(model, test_x, test_y):
    model.eval()
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(test_x)  # 模型预测 model.forward(x)，形状 (n, 5)
        # 从 logits 中提取预测类别索引（得分最高的类别）
        predicted_classes = torch.argmax(y_pred, dim=1)  # 形状 (n,)
        for y_p, y_t in zip(predicted_classes, test_y):  # 与真实标签进行对比
            # print(f"预测类别: {y_p}, 真实类别: {y_t}")
            if y_p == y_t:
                correct += 1
            else:
                wrong += 1
    # print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

# 定义模型
class DiyModel(nn.Module):
    def __init__(self, input_size, hidden_size1, out_size):
        super(DiyModel, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size1)  # 线性层
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size1, out_size)  # 线性层
        
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

def main():
    epoch_num = 100  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 1000  # 每轮训练总共训练的样本总数
    dim = 5 # 生成样本的维度

    x, y = build_data(train_sample, dim)  # 生成样本数据, 每个样本5维向量, 一个标签(标签为向量中最大的数的索引)
    # print(x.shape)

    # 封装成数据集 dataloader, 可以进行批次训练, 并每次训练随机打乱样本
    dataset = TensorDataset(x, y)  # 封装成数据集
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 定义模型, 定义损失函数, 定义优化器
    model = DiyModel(input_size=dim, hidden_size1=32, out_size=dim)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数 内部自带激活函数，softmax, 适合多分类任务
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    log = []

    # 训练
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_x, batch_y in dataloader:
            outputs = model(batch_x)
            #print("outputs:", outputs)
            #print("batch_y:", batch_y)

            loss = criterion(outputs, batch_y) 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            watch_loss.append(loss.item())
        
        # 评估模型
        test_x, test_y = build_data(100, dim)
        acc = evaluate(model, test_x, test_y)
        log.append([acc, torch.tensor(watch_loss).mean()])

    # 可视化
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画 accuracy 曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画 loss 曲线
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()