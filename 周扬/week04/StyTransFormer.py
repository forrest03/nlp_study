import torch 
from transformers import BertModel
import math
import numpy as np
'''
利用google bert的预训练模型权重自行实现transformer结果，比对手工结果跟bert结果验证手工编写的是不是对

先手工写一下模型结构
假设输入4个字  跟老师的一样，便于过程debug分析哪个步骤的结果有问题
embedding阶段
    第一步：先把数据的内容转embedding，转成768维的向量   形状 4*768       4个字代表4个token 每个token是768维
    第二步：全部一样的位置编码维度，也是768维度的向量     形状：4*768       与token的形状相同
    第三步：句子编码                                 形状：4*768       与token的形状相同
    上面三个步骤的编码会相加，再经过一次层归一化（LayerNorm），得到embedding层的编码
多头注意力实现阶段：
    第一步：把第一阶段算出来的向量进行三次线性映射，分别得到q k v               形状都是 4*768
    第二步：把q k v在特征维度上切分成12份，并交换维度，代表12个头独立计算          形状变成 12*4*64
    第三步：q * k转置 除以根号下dk（dk=768/12=64） 然后结果softmax    形状是  12*4*4 、
    第四步：把softmax后的结果乘以v，得到最终的注意力向量                形状变成 12*4*64
    第五步：把12个头拼接在一起，再经过一次全连接层（线性输出），形成最后的融合特征  形状恢复成 4*768
残差连接与归一化
    第六步：将输入向量与第五步的注意力输出残差相加，并进行层归一化
FFN前馈网络阶段：
    第七步：先升维（4*3072）并经过GELU激活函数，再降维（4*768）
再次残差连接与归一化
    第八步：将第六步的结果与FFN的结果残差相加，并进行最终的层归一化

'''

#加载bert模型的权重参数 代码学习
#/Users/zhouyang/myworkspace/badou-nlp/不同步/week04/bert-base-chinese

#把bert模型加载进来
bert = BertModel.from_pretrained("/Users/zhouyang/myworkspace/badou-nlp/不同步/week04/bert-base-chinese",return_dict = False)
#bert模型测试模式
x = np.array([2450, 15486, 102, 2110])   #假想成4个字的句子
#获取参数信息
state_dict = bert.state_dict()
#打印看下都是啥
# print(state_dict.keys())
#格式化打印模型权重信息
# for key, value in state_dict.items():
#     print(f"权重名称:{key},形状{value.shape}")

#手动实现softmax
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)
 
#手动实现gelu激活函数
def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))

class StyTransformer:
    
    #初始化
    def __init__(self, state_dict):
        '''
        传入模型参数state_dict
        把多头的头数定好  bert-base是12头
        把隐藏层的维度定好，bert-base是768维
        把训练层数定好，要跟预训练config.json文件中的模型层数一致
        '''

        #多头注意力的头数
        self.num_attention_heads = 12
        #隐藏层的维度
        self.hidden_dim = 768
        #训练层数
        self.num_layers = bert.config.num_hidden_layers
        #加载模型全中
        self.load_model_weight(state_dict)


    #加载模型权重，并把模型权重放在列表中
    def load_model_weight(self,state_dict):
        #加载模型权重信息
        #1 token层的权重
        self.word_embeddings = state_dict['embeddings.word_embeddings.weight'].numpy()
        # pos层的权重 位置编码
        self.position_embeddings = state_dict['embeddings.position_embeddings.weight'].numpy()
        #segment层的权重 句子编码
        self.token_type_embeddings = state_dict['embeddings.token_type_embeddings.weight'].numpy()
        #层归一化参数 weight bias 
        self.layer_norm_weights = state_dict['embeddings.LayerNorm.weight'].numpy()
        self.layer_norm_bias = state_dict['embeddings.LayerNorm.bias'].numpy()
        #模型有12层，每层的权重加载进来（通过前面打印也能看到layer.0. layer.1 。。。。。）

        #pooler层 句子二分类任务？网上查的
        self.pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()
        self.pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()

        #初始化一个列表，存放transformer层所有的模型权重
        self.sty_bert_weights = []
        #开始遍历       
        for layer in range(self.num_layers):
            #——--------------------------------qkv------------------
            #Q
            q_w = state_dict[f'encoder.layer.{layer}.attention.self.query.weight'].numpy()
            #bias就是线性变换里的  kx+b  的b
            q_b = state_dict[f'encoder.layer.{layer}.attention.self.query.bias'].numpy()

            #K
            k_w = state_dict[f'encoder.layer.{layer}.attention.self.key.weight'].numpy()
            #bias就是线性变换里的  kx+b  的b
            k_b = state_dict[f'encoder.layer.{layer}.attention.self.key.bias'].numpy()

            #V
            v_w = state_dict[f'encoder.layer.{layer}.attention.self.value.weight'].numpy()
            #bias就是线性变换里的  kx+b  的b
            v_b = state_dict[f'encoder.layer.{layer}.attention.self.value.bias'].numpy()
            #——--------------------------------qkv------------------

            #注意力输出时的线性变换权重参数
            attention_output_weight = state_dict[f'encoder.layer.{layer}.attention.output.dense.weight'].numpy()
            attention_output_bias = state_dict[f'encoder.layer.{layer}.attention.output.dense.bias'].numpy()
            
            #注意力输出的层归一化参数
            attention_layer_norm_w = state_dict[f'encoder.layer.{layer}.attention.output.LayerNorm.weight'].numpy()
            attention_layer_norm_b = state_dict[f'encoder.layer.{layer}.attention.output.LayerNorm.bias'].numpy()
        
            #前馈神经网络升维权重参数
            feed_forward_intermediate_weights = state_dict[f'encoder.layer.{layer}.intermediate.dense.weight'].numpy()
            feed_forward_intermediate_bias = state_dict[f'encoder.layer.{layer}.intermediate.dense.bias'].numpy()
            
            #降为
            feed_forward_output_weights = state_dict[f'encoder.layer.{layer}.output.dense.weight'].numpy()
            feed_forward_output_bias = state_dict[f'encoder.layer.{layer}.output.dense.bias'].numpy()
            
            #归一化层权重参数
            layer_norm_weights = state_dict[f'encoder.layer.{layer}.output.LayerNorm.weight'].numpy()
            layer_norm_bias = state_dict[f'encoder.layer.{layer}.output.LayerNorm.bias'].numpy()

            #把这些权重信息追加到列表中
            self.sty_bert_weights.append(
                [q_w,k_w,v_w,
                 q_b,k_b,v_b,
                 attention_output_weight,
                 attention_output_bias,
                 attention_layer_norm_w,
                 attention_layer_norm_b,
                 feed_forward_intermediate_weights,
                 feed_forward_intermediate_bias,
                 feed_forward_output_weights,
                 feed_forward_output_bias,
                 layer_norm_weights,
                 layer_norm_bias])


    def get_embedding(self,embedding_matrix,x):
        '''
        获取embedding层的输出
        embedding_matrix为权重矩阵
        x为数据的句子（转成文字索引了）
        '''
        embed_arr = []
        for index in x:
            embed_arr.append(embedding_matrix[index])
        return np.array(embed_arr)


    def embedding(self,x):
        '''
        embedding层，token emb + pos emb +seg emb
        然后归一化
        '''
        #文字嵌入
        word_emb = self.get_embedding(self.word_embeddings,x)
        #位置嵌入 0 1 2 3 的数组,根据输入长度x获取长度大小的数组，每个元素为位置索引
        pos_arr = np.arange(len(x))
        #位置嵌入
        pos_emb = self.get_embedding(self.position_embeddings,pos_arr)
        #句子嵌入，0000 或者0001
        seg_arr = np.zeros(len(x),dtype=int)
        token_type_emb = self.get_embedding(self.token_type_embeddings,seg_arr)
        #加和求embedding层
        embedding = word_emb + pos_emb + token_type_emb
        #归一化层
        embedding = self.layer_norm(embedding,self.layer_norm_weights,self.layer_norm_bias)
        return embedding

    #归一化层
    def layer_norm(self, x, w, b):
        x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
        x = x * w + b
        return x

    #多头注意力实现阶段，一共12个layer，要循环12次执行
    def multi_head_attention(self,x):
        #循环12次，执行12次注意力，输出
        for layer in range(self.num_layers):
            #以下就是方法里循环后填进去的各层权重的列表
            weights = self.sty_bert_weights[layer]
            #从列表里取出各个参数（根据存入顺序按照顺序去除）
            q_w,\
            k_w,\
            v_w,\
            q_b,\
            k_b,\
            v_b,\
            attention_output_weight,\
            attention_output_bias,\
            attention_layer_norm_w,\
            attention_layer_norm_b,\
            feed_forward_intermediate_weights,\
            feed_forward_intermediate_bias,\
            feed_forward_output_weights,\
            feed_forward_output_bias,\
            ff_layer_norm_weights,\
            ff_layer_norm_bias = weights
            
            #q k v各完成一次线性投影 形状都是4 *768
            q = np.dot(x,q_w.T) + q_b
            k = np.dot(x,k_w.T) + k_b
            v = np.dot(x,v_w.T) + v_b

            #切分头 768/12 = 64
            attention_head_size = int(self.hidden_dim / self.num_attention_heads)

            #当前x的形状
            max_len, hidden_size = x.shape  #4 * 768
            #变形
            #变成 4 * 12 *64 
            q = q.reshape(max_len,self.num_attention_heads,attention_head_size)
            #变成 4 * 12 *64 
            k = k.reshape(max_len,self.num_attention_heads,attention_head_size)
            #变成 4 * 12 *64 
            v = v.reshape(max_len,self.num_attention_heads,attention_head_size)
            # 位置变换？
            q = q.swapaxes(1,0) #形状 12 * 4 * 64
            k = k.swapaxes(1,0)
            v = v.swapaxes(1,0)
            #批量矩阵乘法
            qk = np.matmul(q, k.swapaxes(1, 2))  #手动转置 
            qk /= np.sqrt(attention_head_size) #qk除以根号下头大小
            #做softmax
            qk = softmax(qk)
            #接着  q乘以k的转置除以根号下头大小再乘以v
            qkv = np.matmul(qk, v)

            #变换形状
            qkv = qkv.swapaxes(0, 1)
            #拼接起来
            qkv = qkv.reshape(-1, self.hidden_dim)

            #线性输出
            attention = np.dot(qkv, attention_output_weight.T) + attention_output_bias
            
            #***************归一化*****
            x = self.layer_norm(x+attention, attention_layer_norm_w, attention_layer_norm_b)
            
            #************FNN***********
            #升维
            ffn_x = np.dot(x, feed_forward_intermediate_weights.T) + feed_forward_intermediate_bias
            #非线性激活
            ffn_x = gelu(ffn_x)
            #降维
            ffn_x = np.dot(ffn_x, feed_forward_output_weights.T) + feed_forward_output_bias
            #**********FNN结束**********
            
            #残差连接 & 归一化
            x = self.layer_norm(x + ffn_x, ff_layer_norm_weights, ff_layer_norm_bias)
            
        return x


    def pooler_output_layer(self, x):
        x = np.dot(x, self.pooler_dense_weight.T) + self.pooler_dense_bias
        x = np.tanh(x)
        return x

    #归一化层
    def layer_norm(self, x, w, b):
        x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
        x = x * w + b
        return x
    
    def forward(self,x):
        #第一步 embedding
        x = self.embedding(x)
        #第二步 多头注意力
        s = self.multi_head_attention(x)
        
        p = self.pooler_output_layer(s[0])
        return s,p
    
    
#测试执行
bt = StyTransformer(state_dict)
s , p = bt.forward(x)
print(s)
#print(p)

#bert模型的结果
torch_x = torch.LongTensor([x])
bert.eval()
s_ , p_ = bert(torch_x)
print(s_)
#print(p_)