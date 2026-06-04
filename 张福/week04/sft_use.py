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

#--------------模型定义-------------------------
PINYIN_MAP = {}
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
    parser.add_argument('--model_path', type=str,   default='best_model.pt',    help="选择模型")
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

    model = LM(
        vocab_size = len(char2idx),
        embed_dim = cfg["embed_dim"],
        hidden_dim = cfg["hidden_dim"],
        num_layers = cfg["num_layers"],
        model_type= cfg["model_type"],
        dropout = 0.0 #推理阶段关闭 dropout，保证输出确定性
    ).to(device)

    model.load_state_dict(ckpt['model_state'])
    model.eval()

    print(f"模型: {args.model_path}  ({cfg['model_type'].upper()}, 词表 {len(char2idx)} 字)")
    run(model, char2idx, idx2char, args.topk, args.beam, device)

if __name__ == '__main__':
    main()

