'''
    手写transformer运行原理
    1.输入向量
    2.多头注意力机制
        2.1 embed向量
        2.2 线性3个向量出来 分别是 q、k、v
        2.3 分别将向量均分12份 然后循环每一份:
            softmax(q[0] * k[0].T / sqrt(向量/12)) * v[0]
        2.4 将12份向量组合在一起
    3.残差1  z = layernorm( x + MHA(x)) 归一化
    4.前馈函数 fwn = fnn(z) 前馈函数调用
    5.残差2  layernorm( z + fwn)
    6.输出向量
'''
import argparse
import glob
import math
import random

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from transformers import BertModel
from torch.utils.data import Dataset, DataLoader

#--------------------加载数据----------------------------
'''
    加载所有txt文件,放入到数组当中   
'''
def load_txt(pattern="*.txt"):
    # files = glob.glob("diy_transformer_resources/"+pattern)
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


'''
    根据路径下的txt文档加载词向量,生成idx2chars 和 chars2idx 
'''
def load_vocab(path):
    chars = []
    with open(path, 'r', encoding='utf-8') as f:
        chars = f.readlines()
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = np.array(char2idx)
    return char2idx, idx2char

def clone_linear(old_linear):
    """
        克隆线性函数,根据老线性函数的参数、输入向量维度、输出向量维度
    :param old_linear:  老线性函数
    :return: 新的线性函数
    """
    new_linear = nn.Linear(old_linear.in_features, old_linear.out_features)
    new_linear.load_state_dict(old_linear.state_dict())
    return new_linear

class LM(nn.Module):
    def __init__(self,vocab_size,old_model,avg_h=12):
        super(LM, self).__init__()
        #获取自注意力机制
        multi_head_attention_learner = old_model.encoder.layer[0].attention.self


        #三个目标线性层Q、K、V
        old_q_linear = multi_head_attention_learner.query
        old_k_linear = multi_head_attention_learner.key
        old_v_linear = multi_head_attention_learner.value

        print(f"q、k、v 线性函数的输入维度:{old_q_linear.in_features}，输出维度:{old_q_linear.out_features}")

        #ffn 升维、降维 两个线性层
        # ffn-第一层线性层
        old_ffn_intermediate = old_model.encoder.layer[0].intermediate.dense
        # ffn-第二层线性层
        old_ffn_output = old_model.encoder.layer[0].output.dense
        print(f"ffn 第一层线性函数的输入维度:{old_ffn_intermediate.in_features}，输出维度:{old_ffn_intermediate.out_features}")
        print(f"ffn 第二层线性函数的输入维度:{old_ffn_output.in_features}，输出维度:{old_ffn_output.out_features}")

        embed_dim = old_q_linear.in_features
        self.embedding = nn.Embedding(vocab_size, embed_dim) # embed 向量维度 21128 * 768
        self.q_linear = clone_linear(old_q_linear)
        self.k_linear = clone_linear(old_k_linear)
        self.v_linear = clone_linear(old_v_linear)

        self.ffn_intermediate   = clone_linear(old_ffn_intermediate)
        self.ffn_output         = clone_linear(old_ffn_output)

        self.avg_h = avg_h #设置均分份数
        self.end_linear = nn.Linear(embed_dim, vocab_size)


    def forward(self,x):
        # print("输入信息维度:",x.shape)
        x = self.embedding(x)
        vec_my_transformer = self.my_transformer(x)
        # print("自定义transformer维度:",vec_my_transformer.shape) # torch.Size([4, 32, 768])
        return self.end_linear(vec_my_transformer)

    def my_transformer(self,x):
        #多头注意力计算
        mha_x = self.mha(x)
        #3.合并之后的向量 + 原向量 ,进行残差1计算
        # z = LayerNorm( x + MHA(x) )
        z = F.layer_norm(x + mha_x,x.shape)
        # print("残差1 输出维度:",z.shape)
        #4.前馈函数
        ffn = self.my_ffn(z)
        # print("前馈函数输出维度:",ffn.shape)
        #5.残差2计算
        #output = LayerNorm( z + FFN(z) )
        output  = F.layer_norm(z + ffn,x.shape)
        # print("残差2输出维度：:",output.shape)
        return output
    def mha(self,x):
        # print("embedding输入信息维度:", x.shape)
        B = x.shape[0]
        N = x.shape[1]
        hidden_dim = x.shape[2]

        # 1.对输入的embed向量进行线性计算,得到线性后的向量
        vec_q = self.q_linear(x)  # q 线性,输出向量 768 * 768
        vec_k = self.k_linear(x)  # k 线性,输出向量 768 * 768
        vec_v = self.v_linear(x)  # v 线性,输出向量 768 * 768
        # print("q维度:", vec_q.shape)
        # print("k转置维度:", vec_k.T.shape)
        # print("v维度:", vec_v.shape)
        # 2.均分份数,按2维没问题，3维就存在问题了
        # vec_q_chunks = torch.chunk(vec_q,chunks=self.avg_h,dim=-1)
        # vec_k_chunks = torch.chunk(vec_k,chunks=self.avg_h,dim=-1)
        # vec_v_chunks = torch.chunk(vec_v,chunks=self.avg_h,dim=-1)
        num_heads = self.avg_h  # 12
        hidden_size = hidden_dim  # 768
        head_dim = hidden_dim // self.avg_h  # 768/12 = 64 双斜杠就是整数除法
        # 2.
        # 【正确切分多头】
        # shape → [B, 头数, 序列长度, 头维度]
        q = vec_q.view(B, N, num_heads, head_dim).transpose(1, 2)
        k = vec_k.view(B, N, num_heads, head_dim).transpose(1, 2)
        v = vec_v.view(B, N, num_heads, head_dim).transpose(1, 2)

        # 【正确转置】只转最后两维
        k_t = k.transpose(-2, -1)  # [B, heads, head_dim, N]
        # vec_out = torch.tensor([])
        # for index in range(len(vec_q_chunks)):
        #     scores = vec_q_chunks[index] * vec_k_chunks[index].T / np.sqrt(vec_q_chunks[index].shape[2])
        #     # print("scores维度:",scores.shape)
        #     softmax_scores = torch.softmax(scores, dim=-1)
        #     # print("softmax_scores维度:", softmax_scores.shape)
        #     out_index = softmax_scores * vec_v_chunks[index]
        #     vec_out = torch.cat((vec_out,out_index),dim=0)
        # 计算注意力分数
        attn_score = torch.matmul(q, k_t) / (head_dim ** 0.5)
        attn_weight = torch.softmax(attn_score, dim=-1)

        # 输出
        out = torch.matmul(attn_weight, v)
        vec_out = out.transpose(1, 2).reshape(B, N, hidden_size)
        # print("mha 向量输出维度:", vec_out.shape)
        return vec_out
    def my_ffn(self,add_layer_norm_one):
        '''
        前馈网络函数 ：2层线性函数 + GELU激活函数
        FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
        :param add_layer_norm_one:残差1向量
        :return:
        '''
        n = self.ffn_intermediate(add_layer_norm_one)
        output = self.ffn_output(n)
        return output

#------------------------ 训练 / 评估 ----------------------------
'''
    建立训练与评估方法，返回平均损失值与ppl值
'''
def run_epoch(model,loader,loss_func,optimizer,device,train=True,model_name="diy_model"):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        if model_name == "diy_model":
            logits = model.forward(data)
        else:
            outputs = model.forward(data)
            logits = outputs[0]  # 这才是 tensor，不是 tuple

        # 目标分类数比训练数大，需要调用线性层进行转化
        # 1. 自动获取当前 batch 的动态类别数
        # num_classes = target.max().item() + 1  # 自动变！
        # auto_linear = nn.Linear(input_dim, num_classes)
        # logits = auto_linear(logits)
        # print("logits.reshape(-1, logits.size(-1)) 维度:",logits.reshape(-1, logits.size(-1)).shape)
        # print("target.reshape(-1) 维度:",target.reshape(-1).shape)
        loss = loss_func(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

        #logits 维度: torch.Size([4, 32, 768]) 768
        #target 维度: torch.Size([4, 32])
        # print("logits 维度:",logits.reshape(-1 , logits.size(-1)).shape)
        # print("target 维度:", target.reshape(-1).shape)
        # 查看标签的 最小值、最大值、总类别数
        # print("标签最小值：", target.min().item())
        # print("标签最大值：", target.max().item())


        if train and model_name == "diy_model":
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
    parser.add_argument('--epochs', default=20, type=int)  # 训练轮数
    parser.add_argument('--seq_len', default=32, type=int)  # 读取序列长度
    parser.add_argument('--batch_size', default=4, type=int)  # 每次训练样本个数 128卡死
    parser.add_argument('--max_len', default=768, type=int)  # 语句样本最大限制,原bert模型就是768的长度
    parser.add_argument('--lr', default=1e-3, type=float)  # 学习率
    parser.add_argument('--val_ratio', default=0.05, type=float)  # 验证集数据占比,将现有训练数据按比例分配 训练数据和验证数据
    parser.add_argument('--corpus', default="*.txt")
    parser.add_argument('--save', default="best_diy_transformer_model.pt")

    args = parser.parse_args()
    # 数据准备
    text = load_txt(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件 , 请确认路径是否正确.")
    print(f"语料字符数:{len(text):,}")
    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小 vocab_size:{vocab_size}")

    lines = text.splitlines()  # 按\n 进行分组
    random.shuffle(lines)  # 模块中用于原地打乱列表顺序的函数，直接修改原列表且不返回值。
    split = int(len(lines) * (1 - args.val_ratio))
    train_lines = "\n".join(lines[:split])  # 训练集数据字符串
    val_lines = "\n".join(lines[split:])  # 验证集数据字符串
    print(f"训练集 数据数长度:{len(train_lines):,}")
    print(f"验证集 数据数长度:{len(val_lines):,}")

    train_ds = CharDataset(train_lines, char2idx, args.seq_len)  # 生成训练数据集
    val_ds = CharDataset(val_lines, char2idx, args.seq_len)  # 生成验证数据集

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备 device:{device} 进行训练")

    # 加载词向量
    chars2idx, idx2chars = load_vocab("./diy_transformer_resources/vocab.txt")
    # 加载已存在的模型,直接填文件夹，不带文件名
    old_model = BertModel.from_pretrained('./diy_transformer_resources')
    print(f"公开模型的词向量大小:{old_model.config.vocab_size},本次加载词向量大小:{len(chars2idx)}")
    new_model = LM(vocab_size=len(chars2idx), old_model=old_model)
    total_params = sum(p.numel() for p in new_model.parameters())
    print(f"模型参数量:{total_params:,}")

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(new_model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")  # 用于表示‌正无穷大‌（positive infinity），这是一个特殊的浮点数值，遵循 IEEE 754 浮点数标准。
    print(f"\n {'Epoch':>6} {'Train Loss':>10} {'Train PPL:':>10} {'Val Loss':>10} {'Val PPL:':>10}  {'Val old_Loss':>10}  {'Val old_PPL:':>10}")
    print("-" * 86)

    old_val_loss =0
    old_val_ppl =0
    for epoch in range(1, args.epochs + 1):
        # 进行训练,返回损失函数值、ppl
        tr_loss, train_ppl = run_epoch(new_model, train_loader, loss_func, optimizer, device, True,"diy_model")
        with torch.no_grad():  # 不计算梯度
            # 进行验证,返回验证数据的损失函数值、ppl
            val_loss, val_ppl = run_epoch(new_model, val_loader, loss_func, optimizer, device, False,"diy_model")
            # old_val_loss, old_val_ppl = run_epoch(old_model, val_loader, loss_func, optimizer, device, False, "common_model")

        marker = "  *" if val_ppl < best_val_ppl else ""
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save({
                "model_state": new_model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),  # 返回指定对象的 __dict__.前提是该对象有 __dict__ 属性
            }, args.save)
        print(f"{epoch:>6} {tr_loss:>10.4f} {train_ppl:>10.2f} {val_loss:>10.4f} {val_ppl:>10.2f} {old_val_loss:>10.2f}  {old_val_ppl:>10.2f}  {marker}  ")
    print(f"训练完毕. 最佳验证 PPL:{best_val_ppl:.2f} 已保存至{args.save}")
    pass


if __name__ == '__main__':
    main()


