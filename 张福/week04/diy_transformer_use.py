'''
    输入法的使用
'''
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

#将 numpy.core.multiarray._reconstruct 添加到安全全局变量白名单
torch.serialization.add_safe_globals([
    np.core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
    np.dtypes.ObjectDType])
#------------------ 拼音->候选汉字映射表--------------------
'''
    加载pinyin_map.json文件
    解析json格式，文件读取{音节：【候选字,...】}的映射表
'''
def load_pinyin_map():
    if not os.path.exists('pinyin_map.json'):
        raise FileNotFoundError(f"找不到拼音映射表文件pinyin_map.json")
    with open('pinyin_map.json', 'r',encoding="utf-8") as f:
        return json.load(f)
def clone_linear(old_linear):
    """
        克隆线性函数,根据老线性函数的参数、输入向量维度、输出向量维度
    :param old_linear:  老线性函数
    :return: 新的线性函数
    """
    new_linear = nn.Linear(old_linear.in_features, old_linear.out_features)
    new_linear.load_state_dict(old_linear.state_dict())
    return new_linear
#--------------模型定义-------------------------
PINYIN_MAP = {}
'''
    建立rnn/lstm模型
'''
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

#--------------------- 拼音分词 ------------------------------
'''
    拼音节长度降序排列,保证贪心匹配时优先命中较长的音节
    例如 "zhuang" 应整体匹配，而非"zh" + "uang"
    模块级先置空列表,main()加载 PINYIN_MAP 后重建
'''
_SYLLABLES = []
def segment(pinyin_str):
    """
        得拼音字符串切分为音节到表
        支持带空格 "huang jin" 和连续输入 "huangjin"两种方式
        对每个空格分隔的 token 使用贪心最长匹配,无法识别的字符直接跳过
    :param pinyin_str: 传入的字符串
    :return:
    """
    syllables = []
    for token in pinyin_str.strip().lower().split():
        i=0
        while i < len(token):
            #在已排序的音节表中找第一个能匹配当前位置的音节
            matched = next((s for s in _SYLLABLES if token[i:].startswith(s)), None)
            if matched:
                syllables.append(matched)
                i += len(matched)
            else:
                i +=1 # 当前字符无法匹配任何音节,跳过
    return syllables
#--------------------------- 束搜索 -----------------------------
def beam_search(syllables,prefix,model,char2idx,idx2char,beam_size,device):
    """
        对音节列表逐字做 束搜索 ，返回按得分降序排列的候选列表
        原理:
            每处理一个音节，就把当前所有beam与该音节的后端汉字做笛卡尔积展开
            用语言模型对 prefix + 已生成部分， 预测下一字的log_prob 作为得分增量
            展开后按总得分取 top-beam_size 条路径继续
    :param syllables:  本轮待转换的音节列表，如["huang","jin"]
    :param prefix: 已确认的历史文字->作为语言模型的上文
    :param model: 模型
    :param char2idx: 模型所用字符列表和索引
    :param idx2char: 模型所用字符列表和索引
    :param beam_size: 束宽 - 每步保留的最优路径数量
    :param device: 使用的设备
    :return:  [(累计log_prob,转换结果字符串),...]
    """
    beams = [(0.0,"")]#初始只有一条空路径,得分为0
    for syllable in syllables:
        #过滤掉不在训练词表中的候选字(模型无法为其打分)
        candidates = [c for c in PINYIN_MAP.get(syllable,[]) if c in char2idx]
        if not candidates:
            continue # 该音节无可用候选，跳过(不终止整个搜索)
        new_beams = []
        for score,partial in beams:
            #拼接历史上文与当前已生成部分，送入模型
            context = prefix + partial
            if context:
                ids = [char2idx[c] for c in context if c in char2idx]
                x = torch.tensor([ids],dtype=torch.long,device=device)
                with torch.no_grad():
                    logits = model(x)
                #只取最后一个时间步的输出，作为下一个字的概率分布
                log_probs = F.log_softmax(logits[0,-1,:],dim=-1)
            else:
                log_probs = None
            for char in candidates:
                lp = log_probs[char2idx[char]].item() if log_probs is not None else 0.0
                new_beams.append((score + lp, partial + char))
        #按累计得分降序,保留 beam_size 条最优路径
        new_beams.sort( reverse=True)
        beams = new_beams[:beam_size]
    return beams

#--------------------交互主循环-------------------
def run(model,char2idx,idx2char,topk,beam_size,device):
    """
    交互式输入法主循环
    用户输入一段拼音. 程序展示 topk 个候选转换结果
    用户选择编号后，结束追加到已确认文字，作为下一轮的上文
    :param model:
    :param char2idx:
    :param idx2char:
    :param topk: 显示 可选项个数
    :param beam_size:
    :param device:
    :return:
    """
    print("=" * 52)
    print("""
        拼音输入法(字符级语言模型)
        输入拼音回车 -> 选择候选编号追加到已输入文字
        r = 重置  q = 退出
    """)
    print("=" * 52)

    input_str = ""
    while True:
        print(f"\n 已输入:[{input_str}]" if input_str else "\n 已输入:(空)")
        raw_str = input("拼音> ").strip()
        if not raw_str:
            continue
        if raw_str == "q":
            print("退出.")
            break
        if raw_str == "r":
            input_str = ""
            continue
        syllables = segment(raw_str)
        if not syllables:
            print("无法识别任何音节,请检查拼音拼写.")
            continue
        print(f"音节 : {' '.join(syllables)}")
        results = beam_search(syllables,input_str,model,char2idx,idx2char,beam_size,device)

        if not results:
            print("无候选结果")
            continue
        print("候选:")
        for i,(score,text) in enumerate(results[:topk]):
            print(f"  [{i}] {text} ({score:.2f})")
        choice = input("选择编号 (回车跳过):").strip()
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(results):
                input_str = results[idx][1]
            else:
                print("编号超出范围.")
    pass

#---------------------入口-------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,   default='best_diy_transformer_model.pt',    help="选择模型")
    parser.add_argument('--topk',       type=int,   default=5,                  help="展示候选数")
    parser.add_argument("--beam",       type=int,   default=10,                 help="束搜索宽度")
    args = parser.parse_args()

    #加载拼音映射列表，并重建音节排序列表
    global PINYIN_MAP,_SYLLABLES
    PINYIN_MAP = load_pinyin_map()
    _SYLLABLES = sorted(PINYIN_MAP.keys(), key = len , reverse=True)
    print(f"拼音表: pinyin_map.json ({len(PINYIN_MAP)} 个音节)")

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 从 checkpoint 中恢复词表和模型超参
    ckpt= torch.load(args.model_path, map_location=device,weights_only=True)
    char2idx = ckpt['char2idx']
    idx2char = ckpt['idx2char']
    cfg = ckpt['args']

    # 加载已存在的模型,直接填文件夹，不带文件名
    old_model = BertModel.from_pretrained('./diy_transformer_resources')
    print(f"公开模型的词向量大小:{old_model.config.vocab_size},本次加载词向量大小:{len(char2idx)}")
    model = LM(vocab_size=old_model.config.vocab_size, old_model=old_model)

    model.load_state_dict(ckpt['model_state'])
    model.eval()

    print(f"模型: {args.model_path} , 词表 {len(char2idx)} 字)")
    run(model, char2idx, idx2char, args.topk, args.beam, device)

if __name__ == '__main__':
    main()

