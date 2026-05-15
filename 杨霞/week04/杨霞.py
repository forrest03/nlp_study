import torch
import math
import numpy as np
from transformers import BertModel
import torch.nn.functional as nn

bert = BertModel.from_pretrained(r"D:\NLP-HOMEWORK\week04\bert下载\bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()
print(state_dict.keys())
bert.eval()
x = torch.tensor([[2450, 15486, 102, 2110]],dtype=torch.long)
print('22222',x,x.shape)
seqence_output, pooler_output = bert(x)
print(seqence_output.shape, pooler_output.shape)



#softmax归一化
def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))

nn.softmax(seqence_output, dim=-1)
nn.gelu(seqence_output)

class DiyBert:
    def __init__(self, state_dict):
        self.load_weight(state_dict)
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.num_layers = bert.config.num_hidden_layers
    def load_weight(self, state_dict):
        self.word_embedding = state_dict['embeddings.word_embeddings.weight'].numpy()
        self.position_embedding = state_dict['embeddings.position_embeddings.weight'].numpy()
        self.token_type_embeddings = state_dict['embeddings.token_type_embeddings.weight'].numpy()
        self.embeddings_layer_norm_weight = state_dict['embeddings.LayerNorm.weight'].numpy()
        self.embeddings_layer_norm_bias = state_dict['embeddings.LayerNorm.bias'].numpy()
        self.transformer_weights = []
        for i in range(self.num_layers):
            q_w = state_dict['encoder.layer.%d.attention.self.query.weight' %i].numpy()
            q_b = state_dict['encoder.layer.%d.attention.self.query.bias' %i].numpy()
            k_w = state_dict['encoder.layer.%d.attention.self.key.weight' %i].numpy()
            k_b = state_dict['encoder.layer.%d.attention.self.key.bias' %i].numpy()
            v_w = state_dict['encoder.layer.%d.attention.self.value.weight' %i].numpy()
            v_b = state_dict['encoder.layer.%d.attention.self.value.bias' %i].numpy()
            attention_output_weight = state_dict['encoder.layer.%d.attention.output.dense.weight' %i].numpy()
            attention_output_bias = state_dict['encoder.layer.%d.attention.output.dense.bias' %i].numpy()
            attention_layer_norm_weight = state_dict['encoder.layer.%d.attention.output.LayerNorm.weight' %i].numpy()
            attention_layer_norm_bias = state_dict['encoder.layer.%d.attention.output.LayerNorm.bias' %i].numpy()
            intermediate_weight = state_dict['encoder.layer.%d.intermediate.dense.weight' %i].numpy()
            intermediate_bias = state_dict['encoder.layer.%d.intermediate.dense.bias' %i].numpy()
            output_weight = state_dict['encoder.layer.%d.output.dense.weight' %i].numpy()
            output_bias = state_dict['encoder.layer.%d.output.dense.bias' %i].numpy()
            ff_layer_norm_w = state_dict['encoder.layer.%d.output.LayerNorm.weight' %i].numpy()
            ff_layer_norm_b = state_dict['encoder.layer.%d.output.LayerNorm.bias' %i].numpy()
            self.transformer_weights.append([
                q_w,q_b,
                k_w,k_b,
                v_w,v_b,
                attention_output_weight,
                attention_output_bias,
                attention_layer_norm_weight,
                attention_layer_norm_bias,
                intermediate_weight,
                intermediate_bias,
                output_weight,
                output_bias,
                ff_layer_norm_w,
                ff_layer_norm_b,
            ])
        # pooler层
        self.pooler_dense_weight = state_dict['pooler.dense.weight'].numpy()
        self.pooler_dense_bias = state_dict['pooler.dense.bias'].numpy()
    # bert embedding 三层 position，segment，token
    def embedding_forward(self,x):
        # x.shape = [max_len]
        we = self.get_embedding(self.word_embeddings, x)
        pe = self.get_embedding(self.position_embedding,np.array(list(range(len(x)))))
        te = self.get_embedding(self.token_type_embeddings,np.array([0]*len(x)))
        embedding = we + pe + te
        # embedding = self.layer_norm(embedding,self.embeddings_layer_norm_weight,self.embeddings_layer_norm_bias)
        embedding = torch.layer_norm(embedding, weight = self.embeddings_layer_norm_weight, bias = self.embeddings_layer_norm_bias)
        return embedding
    def get_embedding(self,embedding_matrix,x):
        return torch.tensor([embedding_matrix[index] for index in x])
        # return numpy.array([embedding_matrix[index] for index in x])

    def all_transformer_layer_forward(self,x):
        for i in range(self.num_layers):
            x = self.single_transformer_layer_forward(x,i)
        return x
    def single_transformer_layer_forward(self,x,layer_index):
        weights = self.transformer_weights[layer_index]
        [q_w,q_b,
        k_w,k_b,
        v_w,v_b,
        attention_output_weight,
        attention_output_bias,
        attention_layer_norm_weight,
        attention_layer_norm_bias,
        intermediate_weight,
        intermediate_bias,
        output_weight,
        output_bias,
        ff_layer_norm_w,
        ff_layer_norm_b]  = weights
        attention_output = self.self_attention(x,q_w,q_b,k_w,k_b,v_w,v_b,
                                               attention_output_weight,attention_output_bias)
    
    def self_attention(self,
                       x,
                       q_w,
                       q_b,
                       k_w,
                       k_b,
                       v_w,
                       v_b,
                       attention_output_weight,
                       attention_output_bias):
        q = np.dot(x,q_w.T) + q_b 
        k = np.dot(x,k_w.T) + q_b 
        v = np.dot(x,v_w.T) + q_b 
        attention_head_size = self.hidden_size //  self.num_attention_heads,
        q = self.transpose_for_scores(q,attention_head_size)
        k = self.transpose_for_scores(q,attention_head_size)
        v = self.transpose_for_scores(q,attention_head_size)
        qk = np.matmul(q,k.swapaxes([1,2]))
        qk = qk / np.sqrt(attention_head_size)
        qk = torch.softmax(qk,dim=-1)
        qkv = np.matmul(qk,v) 
        qkv = qkv.swapaxes(0, 1).reshape(-1, self.hidden_size)
        qkv = np.dot(qkv,attention_output_weight.T) + attention_output_bias
        return qkv

    def transpose_for_scores(self,x,attention_head_size):
        max_len,hidden_size = x.shape
        x = x.reshape(max_len,self.num_attention_heads,attention_head_size)
        x = x.swapaxes(0,1)
        return x
    def feed_forward(self,
                     x,
                     intermediate_weight,  # intermediate_size, hidden_size
                     intermediate_bias,  # intermediate_size
                     output_weight,  # hidden_size, intermediate_size
                     output_bias,  # hidden_size
                     ):
        # output shpae: [max_len, intermediate_size]
        x = np.dot(x, intermediate_weight.T) + intermediate_bias
        x = gelu(x)
        # output shpae: [max_len, hidden_size]
        x = np.dot(x, output_weight.T) + output_bias
        return x
    def pooler_output_layer(self, x):
        x = np.dot(x, self.pooler_dense_weight.T) + self.pooler_dense_bias
        x = np.tanh(x)
        return x
    def forward(self, x):
        x = self.embedding_forward(x)
        sequence_output = self.all_transformer_layer_forward(x)
        pooler_output = self.pooler_output_layer(sequence_output[0])
        return sequence_output, pooler_output
                                                                                                         
                                                                                                         


DiyBert(state_dict)
diy_sequence_output, diy_pooler_output = db.forward(x)
torch_sequence_output, torch_pooler_output = bert(torch_x)
print(diy_sequence_output)
print(torch_sequence_output)
