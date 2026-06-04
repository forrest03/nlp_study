'''
用于模型训练过程中验证样本正确率的程序
'''
import torch
import CreateSample as cs

def test_model_train(model, x=None, y=None):
    model.eval()
    #生成1000条测试数据
    test_sample_num = 1000
    x, y = cs.create_sample(test_sample_num)
    with torch.no_grad():#关闭梯度计算
        #输出模型结果的预测值
        y_pred = model.forward(x)
        # print("预测值y_pred:", y_pred)
        # print("真实答案y:", y)
        #对比推理结果与真实结果
        #预测值为y_pred，真实值为y，遍历y_pred和y，判断是否相等
        correct_num = 0
        #yp为预测值，yt为真实值 ，遍历y_pred和y，判断是否相等
        for yp,yt in zip(y_pred,y):
            # yp 是模型输出的 5 个类别的得分，并不是直接的类别索引
            # 用 torch.argmax(yp) 找出得分最高的索引，再和真实标签 yt 比对
            if torch.argmax(yp) == yt:
                correct_num += 1


    # 4. 计算准确率并打印
    accuracy = correct_num / test_sample_num
    return accuracy

