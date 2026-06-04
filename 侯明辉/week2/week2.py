#完成一个多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类。
import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class ClassifyModel(nn.Module):
    def __init__(self,input_size):
        super(ClassifyModel,self).__init__()
        self.linear = nn.Linear(input_size,input_size)
        #self.relu = nn.ReLU()  #不使用relu，因为是分类任务，不需要非线性，relu会导致会把负数变成 0，这破坏了信息
        self.loss = nn.functional.cross_entropy #已经包含softmax了
    
    def forward(self,x,y):
        y_pred=self.linear(x)
        return self.loss(y_pred,y)
    
    def pred(self,x):
        y_pred=self.linear(x)
        return y_pred

def build_sample():
    x=np.random.random(5)
    return x,x.argmax()

def build_database(total_sample_num):
    X=[]
    Y=[]
    for i in range(total_sample_num):
        x,y=build_sample()
        X.append(x)
        Y.append(y) 
    return torch.FloatTensor(X), torch.LongTensor (Y)


def evaluate(model):
    model.eval()
    test_sample_num=100
    x,y=build_database(test_sample_num)
    input_size = 5
    correct,wrong = 0,0
    # 统计每个类别的正确和错误数量
    class_correct = [0] * input_size
    class_wrong = [0] * input_size
    with torch.no_grad():
        y_pred=model.pred(x)
        for y_p,y_t in zip(y_pred,y):
            pred_class = y_p.argmax().item()
            true_class = y_t.item()
            if pred_class == true_class:
                correct+=1
                class_correct[true_class] += 1
            else:
                wrong+=1
                class_wrong[true_class] += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong), class_correct, class_wrong

def main():
    #参数
    epoch_num = 20
    batch_size = 10
    train_sample = 5000
    input_size = 5
    learning_rate = 0.01

    #模型
    model = ClassifyModel(input_size)
    #优化器
    optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
    log = []
    #训练集
    train_x,train_y=build_database(train_sample)
    #训练
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample//batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x,y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())

        mean_loss = np.mean(watch_loss)
        acc, class_correct, class_wrong = evaluate(model)
        log.append([acc, mean_loss])
        print("epoch: %d, loss: %f, acc: %f" % (epoch + 1, mean_loss, acc))
    # 保存模型
    torch.save(model.state_dict(), "model.bin")

    # 测试预测：生成一些样本并打印输入和输出
    test_x, test_y = build_database(10)
    model.eval()
    with torch.no_grad():
        test_pred = model.pred(test_x)
    print("\n===== 预测结果 =====")
    for i in range(len(test_y)):
        pred_class = test_pred[i].argmax().item()
        true_class = test_y[i].item()
        prob = torch.softmax(test_pred[i], dim=0)
        status = "✓" if pred_class == true_class else "✗"
        print("输入: %s | 真实类别: %d | 预测类别: %d | 概率分布: %s %s" % (
            np.round(test_x[i].numpy(), 2), true_class, pred_class,
            np.round(prob.numpy(), 3), status))
        

   #画图
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("acc", color="tab:blue")
    ax1.plot([l[0] for l in log], color="tab:blue", label="acc")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("loss", color="tab:red")
    ax2.plot([l[1] for l in log], color="tab:red", label="loss")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    plt.title("Training Progress")
    plt.show()        
 


if __name__ == "__main__":
    main()