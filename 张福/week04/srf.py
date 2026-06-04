'''
    输入法模拟训练模型
'''
import argparse
import glob
import random

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import math

#--------------------加载数据----------------------------
'''
    加载所有txt文件,放入到数组当中   
'''
def load_txt(pattern="*.txt"):
    files = glob.glob(pattern)
    texts = []
    for file in files:
        with open(file, "r",encoding="UTF-8") as f:
            texts.append(f.read())
    return "\n".join(texts)

'''
    将读取到的文字信息去重排序,并建立枚举词表关联信息
'''
def build_vocab(text):
    chars = sorted(set(text))
    char2idx = {c: i for i,c in enumerate(chars)}
    idx2char = np.array(char2idx)
    return char2idx, idx2char

'''
    构建数据集类,将词表信息放入类对象当中,通过遍历获取x、y信息
    x为指定位置的字符到需要读取长度的字符串(eg: 我很爱中国,读取值:我很爱)
    y为指定位置的下一个字符到需要读取长度的字符串(eg: 我很爱中国,读取值:很爱中)
'''
class CharDataset(Dataset):
    def __init__(self, texts, char2idx, seq_len):
        self.seq_len = seq_len
        ids = [char2idx[c] for c in texts if c in char2idx]
        self.data = torch.tensor(ids,dtype=torch.long)

    def __len__(self):
        return max(0,len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y

#--------------------------模型---------------------
'''
    建立rnn/lstm模型
'''
class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim,hidden_dim,num_layers,model_type,dropout=0.3):
        '''
            初始化语言模型
        :param vocab_size:  词表大小
        :param embed_dim:   embed层大小
        :param hidden_dim:  隐藏层大小
        :param num_layers:  训练层数
        :param model_type:  模型类型 如果是 lstm 就选择lstm模型，否则选择rnn模型
        :param dropout:  随机屏蔽 · 防过拟合 · 训练/推理切换   随机屏蔽的概率 0~1之间
        '''
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        rnn_cls = nn.LSTM if model_type == "lstm" else nn.RNN
        self.rnn = rnn_cls(
            embed_dim, hidden_dim,
            num_layers = num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        '''
            1.根据传入的信息,建立embed随机的维度编码
            2.随机按比例隐藏embed编码
            3.通过rnn/lstm模型对embed编码进行向量运算
            4.按比例隐藏向量信息
            5.线性函数对象隐藏后的向量进行运算,得到结果向量
        :param x:
        :return:
        '''
        embed = self.embed(x)
        e = self.dropout(embed)
        out,_ = self.rnn(e)
        logits = self.fc(self.dropout(out))
        return logits
#------------------------ 训练 / 评估 ----------------------------
'''
    建立训练与评估方法，返回平均损失值与ppl值
'''
def run_epoch(model,loader,loss_func,optimizer,device,train=True):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        logits = model.forward(data)
        print("logits.reshape(-1, logits.size(-1)) 维度:", logits.reshape(-1, logits.size(-1)).shape)
        print("target.reshape(-1) 维度:", target.reshape(-1).shape)
        loss = loss_func(logits.reshape(-1 , logits.size(-1)),target.reshape(-1))
        if train :
            optimizer.zero_grad() #梯度归0
            loss.backward()  #求导
            optimizer.step()  #调整参数

        total_loss += loss.item() * target.numel()
        total_tokens += target.numel()
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl

def main():
    # 统一设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='rnn',choices=['lstm','rnn'])
    parser.add_argument('--epochs',     default=20,         type=int)#训练轮数
    parser.add_argument('--seq_len',    default=32,         type=int)#读取序列长度
    parser.add_argument('--batch_size', default=4,        type=int)# 每次训练样本个数 128卡死
    parser.add_argument('--embed_dim',  default=128,        type=int)# embed 向量维度
    parser.add_argument('--hidden_dim', default=256,        type=int)# 隐藏维度
    parser.add_argument('--num_layers', default=2,          type=int)# 训练层数
    parser.add_argument('--dropout',    default=0.3,        type=float)# 隐藏数据概率
    parser.add_argument('--lr',         default=1e-3,       type=float)# 学习率
    parser.add_argument('--val_ratio',  default=0.05,       type=float)# 验证集数据占比,将现有训练数据按比例分配 训练数据和验证数据
    parser.add_argument('--corpus',     default="*.txt")
    parser.add_argument('--save',       default="best_model.pt")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备 device:{device} 进行训练, 使用模型 model:{args.model_type.upper()}")

    #数据准备
    text = load_txt(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件 , 请确认路径是否正确.")
    print(f"语料字符数:{len(text):,}")
    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小 vocab_size:{vocab_size}")

    lines = text.splitlines() # 按\n 进行分组
    random.shuffle(lines) # 模块中用于原地打乱列表顺序的函数，直接修改原列表且不返回值。
    split = int(len(lines) * (1 - args.val_ratio))
    train_lines = "\n".join(lines[:split]) # 训练集数据字符串
    val_lines   = "\n".join(lines[split:]) # 验证集数据字符串
    print(f"训练集 数据数长度:{len(train_lines):,}")
    print(f"验证集 数据数长度:{len(val_lines):,}")

    train_ds    =       CharDataset(train_lines,    char2idx,       args.seq_len) # 生成训练数据集
    val_ds      =       CharDataset(val_lines,      char2idx,       args.seq_len) # 生成验证数据集

    train_loader    = DataLoader(train_ds,  batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader      = DataLoader(val_ds,    batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)


    #模型
    model = LM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        model_type=args.model_type,
        dropout=args.dropout).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量:{total_params:,}")

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf") # 用于表示‌正无穷大‌（positive infinity），这是一个特殊的浮点数值，遵循 IEEE 754 浮点数标准。
    print(f"\n{'Epoch':>6} {'Train Loss':>10} {'Train PPL:':>10} {'Val Loss':>10} {'Val PPL:':>10}")
    print("-" * 56)
    for epoch in range(1,args.epochs+1):
        # 进行训练,返回损失函数值、ppl
        tr_loss, train_ppl = run_epoch(model, train_loader,loss_func,optimizer,device,True)
        with torch.no_grad(): # 不计算梯度
            #进行验证,返回验证数据的损失函数值、ppl
            val_loss, val_ppl = run_epoch(model, val_loader,loss_func,optimizer,device,False)

        marker = "  *" if val_ppl < best_val_ppl else ""
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save({
                "model_state":model.state_dict(),
                "char2idx":char2idx,
                "idx2char":idx2char,
                "args":vars(args),# 返回指定对象的 __dict__.前提是该对象有 __dict__ 属性
            },args.save)
        print(f"{epoch:>6} {tr_loss:>10.4f} {train_ppl:>10.2f} {val_loss:>10.4f} {val_ppl:>10.2f}  {marker}")
    print(f"训练完毕. 最佳验证 PPL:{best_val_ppl:.2f} 已保存至{args.save}")
    pass

if __name__ == '__main__':
    main()


