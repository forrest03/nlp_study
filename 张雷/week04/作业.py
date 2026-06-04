import torch
import torch.nn as nn
import math
from transformers import BertModel

bert = BertModel.from_pretrained(r"/Users/zhanglei/projects/llm-bootcamp/week04/bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()
bert.eval()
x = [2450, 15486, 102, 2110] #假想成4个字的句子
torch_x = torch.LongTensor([x])          #pytorch形式输入
seqence_output, pooler_output = bert(torch_x)
print(seqence_output.shape, pooler_output.shape)
# print(seqence_output, pooler_output)

# print(bert.state_dict().keys())  #查看所有的权值矩阵名称

#gelu激活函数
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class DiyBert:
    def __init__(self, state_dict):
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.num_layers = bert.config.num_hidden_layers
        self.load_weights(state_dict)
    
    def load_weights(self, state_dict):
        #embedding部分
        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"]
        self.position_embeddings = state_dict["embeddings.position_embeddings.weight"]
        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"]
        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"]
        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"]
        self.transformer_weights = []
        #transformer部分，有多层
        for i in range(self.num_layers):
            q_w = state_dict["encoder.layer.%d.attention.self.query.weight" % i]
            q_b = state_dict["encoder.layer.%d.attention.self.query.bias" % i]
            k_w = state_dict["encoder.layer.%d.attention.self.key.weight" % i]
            k_b = state_dict["encoder.layer.%d.attention.self.key.bias" % i]
            v_w = state_dict["encoder.layer.%d.attention.self.value.weight" % i]
            v_b = state_dict["encoder.layer.%d.attention.self.value.bias" % i]
            attention_output_weight = state_dict["encoder.layer.%d.attention.output.dense.weight" % i]
            attention_output_bias = state_dict["encoder.layer.%d.attention.output.dense.bias" % i]
            attention_layer_norm_w = state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % i]
            attention_layer_norm_b = state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % i]
            intermediate_weight = state_dict["encoder.layer.%d.intermediate.dense.weight" % i]
            intermediate_bias = state_dict["encoder.layer.%d.intermediate.dense.bias" % i]
            output_weight = state_dict["encoder.layer.%d.output.dense.weight" % i]
            output_bias = state_dict["encoder.layer.%d.output.dense.bias" % i]
            ff_layer_norm_w = state_dict["encoder.layer.%d.output.LayerNorm.weight" % i]
            ff_layer_norm_b = state_dict["encoder.layer.%d.output.LayerNorm.bias" % i]
            self.transformer_weights.append([q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias,
                                             attention_layer_norm_w, attention_layer_norm_b, intermediate_weight, intermediate_bias,
                                             output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b])
        #pooler层
        self.pooler_dense_weight = state_dict["pooler.dense.weight"]
        self.pooler_dense_bias = state_dict["pooler.dense.bias"]
    
    #bert embedding，使用3层叠加，在经过一个Layer norm层
    def embedding_forward(self, x):
        # x.shape = [max_len]
        we = self.get_embedding(self.word_embeddings, x)  # shpae: [max_len, hidden_size]
        # position embeding的输入 [0, 1, 2, 3]
        pe = self.get_embedding(self.position_embeddings, list(range(len(x))))  # shpae: [max_len, hidden_size]
        # token type embedding,单输入的情况下为[0, 0, 0, 0]
        te = self.get_embedding(self.token_type_embeddings, [0] * len(x))  # shpae: [max_len, hidden_size]
        embedding = we + pe + te
        # 加和后有一个归一化层
        embedding = self.layer_norm(embedding, self.embeddings_layer_norm_weight, self.embeddings_layer_norm_bias)  # shpae: [max_len, hidden_size]
        return embedding
    
    #embedding层实际上相当于按index索引，或理解为onehot输入乘以embedding矩阵
    def get_embedding(self, embedding_matrix, x):
        return embedding_matrix[torch.tensor(x)]

    #执行全部的transformer层计算
    def all_transformer_layer_forward(self, x):
        for i in range(self.num_layers):
            x = self.single_transformer_layer_forward(x, i)
        return x
    
    #执行单层transformer层计算
    def single_transformer_layer_forward(self, x, layer_index):
        weights = self.transformer_weights[layer_index]
        #取出该层的参数，在实际中，这些参数都是随机初始化，之后进行预训练
        q_w, q_b, \
        k_w, k_b, \
        v_w, v_b, \
        attention_output_weight, attention_output_bias, \
        attention_layer_norm_w, attention_layer_norm_b, \
        intermediate_weight, intermediate_bias, \
        output_weight, output_bias, \
        ff_layer_norm_w, ff_layer_norm_b = weights
        #self attention层
        attention_output = self.self_attention(x,
                                q_w, q_b,
                                k_w, k_b,
                                v_w, v_b,
                                attention_output_weight, attention_output_bias,
                                self.num_attention_heads,
                                self.hidden_size)
        #bn层，并使用了残差机制
        x = self.layer_norm(x + attention_output, attention_layer_norm_w, attention_layer_norm_b)
        #feed forward层
        feed_forward_x = self.feed_forward(x,
                              intermediate_weight, intermediate_bias,
                              output_weight, output_bias)
        #bn层，并使用了残差机制
        x = self.layer_norm(x + feed_forward_x, ff_layer_norm_w, ff_layer_norm_b)
        return x
    
        # self attention的计算
    
    def self_attention(self,
                       x,
                       q_w,
                       q_b,
                       k_w,
                       k_b,
                       v_w,
                       v_b,
                       attention_output_weight,
                       attention_output_bias,
                       num_attention_heads,
                       hidden_size):
        # x.shape = max_len * hidden_size
        # q_w, k_w, v_w  shape = hidden_size * hidden_size
        # q_b, k_b, v_b  shape = hidden_size
        # q = np.dot(x, q_w.T) + q_b  # shape: [max_len, hidden_size]      W * X + B lINER
        # k = np.dot(x, k_w.T) + k_b  # shpae: [max_len, hidden_size]
        # v = np.dot(x, v_w.T) + v_b  # shpae: [max_len, hidden_size]
        q = torch.matmul(x, q_w.T) + q_b  # shape: [max_len, hidden_size]
        k = torch.matmul(x, k_w.T) + k_b  # shpae: [max_len, hidden_size]
        v = torch.matmul(x, v_w.T) + v_b  # shpae: [max_len, hidden_size]
        attention_head_size = int(hidden_size / num_attention_heads)
        # q.shape = num_attention_heads, max_len, attention_head_size
        q = self.transpose_for_scores(q, attention_head_size, num_attention_heads)
        # k.shape = num_attention_heads, max_len, attention_head_size
        k = self.transpose_for_scores(k, attention_head_size, num_attention_heads)
        # v.shape = num_attention_heads, max_len, attention_head_size
        v = self.transpose_for_scores(v, attention_head_size, num_attention_heads)
        # qk.shape = num_attention_heads, max_len, max_len
        qk = torch.matmul(q, k.transpose(1, 2))
        qk /= math.sqrt(attention_head_size)
        qk = torch.softmax(qk, dim=-1)
        # qkv.shape = num_attention_heads, max_len, attention_head_size
        qkv = torch.matmul(qk, v)
        # qkv.shape = max_len, hidden_size
        qkv = qkv.transpose(0, 1).reshape(-1, hidden_size)
        # attention.shape = max_len, hidden_size
        attention = torch.matmul(qkv, attention_output_weight.T) + attention_output_bias
        return attention
    
    #多头机制
    def transpose_for_scores(self, x, attention_head_size, num_attention_heads):
        # hidden_size = 768  num_attent_heads = 12 attention_head_size = 64
        max_len, hidden_size = x.shape
        x = x.reshape(max_len, num_attention_heads, attention_head_size)
        x = x.transpose(0, 1)  # output shape = [num_attention_heads, max_len, attention_head_size]
        return x
    
    #前馈网络的计算
    def feed_forward(self,
                     x,
                     intermediate_weight,  # intermediate_size, hidden_size
                     intermediate_bias,  # intermediate_size
                     output_weight,  # hidden_size, intermediate_size
                     output_bias,  # hidden_size
                     ):
        # output shpae: [max_len, intermediate_size]
        x = torch.matmul(x, intermediate_weight.T) + intermediate_bias
        x = gelu(x)
        # output shpae: [max_len, hidden_size]
        x = torch.matmul(x, output_weight.T) + output_bias
        return x

    #归一化层
    def layer_norm(self, x, w, b):
        x = (x - torch.mean(x, dim=1, keepdim=True)) / torch.std(x, dim=1, keepdim=True)
        x = x * w + b
        return x

    #链接[cls] token的输出层
    def pooler_output_layer(self, x):
        x = torch.matmul(x, self.pooler_dense_weight.T) + self.pooler_dense_bias
        x = torch.tanh(x)
        return x

    #最终输出
    def forward(self, x):
        x = self.embedding_forward(x)
        sequence_output = self.all_transformer_layer_forward(x)
        pooler_output = self.pooler_output_layer(sequence_output[0])
        return sequence_output, pooler_output

#自制
db = DiyBert(state_dict)
diy_sequence_output, diy_pooler_output = db.forward(x)
#torch
torch_sequence_output, torch_pooler_output = bert(torch_x)

print(diy_sequence_output)
print(torch_sequence_output)
