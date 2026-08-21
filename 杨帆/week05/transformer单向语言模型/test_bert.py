import numpy as np
import torch.nn as nn
import torch

class TestBert(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, max_position_size=512, num_hidden_layer=12):
        super(TestBert, self).__init__()
        self.hidden_size = hidden_size
        self.max_position_size = max_position_size
        self.num_hidden_layer = num_hidden_layer
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.segment_embedding = nn.Embedding(2, hidden_size)
        self.position_embedding = nn.Embedding(max_position_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.q_linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layer)])
        self.k_linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layer)])
        self.v_linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layer)])
        self.cc_layer_norm_embedding_attention_list = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(num_hidden_layer)])
        self.feed_linear_in_list = nn.ModuleList(
            [nn.Linear(hidden_size, 4 * hidden_size) for _ in range(num_hidden_layer)])
        self.feed_linear_out_list = nn.ModuleList(
            [nn.Linear(4 * hidden_size, hidden_size) for _ in range(num_hidden_layer)])
        self.cc_layer_norm_feed_list = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(num_hidden_layer)])
        self.softmax = nn.Softmax(dim=-1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)


    def embedding_forward(self, tensor_token):
        token = self.token_embedding(tensor_token)
        tensor_seg = torch.zeros_like(tensor_token)
        segment = self.segment_embedding(tensor_seg)
        tensor_pos = torch.arange(0, tensor_token.shape[-1], dtype=torch.long)
        position = self.position_embedding(tensor_pos)
        embedding = token + segment + position
        embedding = self.dropout(embedding)
        embedding = self.layer_norm(embedding)
        return embedding

    def muti_head(self, x, head_num):
        batch_size, max_len, hidden_size = x.shape
        x = x.reshape(batch_size, max_len, head_num, hidden_size // head_num).transpose(1, 2)  # batch_size, head_num, max_len, head_size
        return x

    def self_attention_forward(self, x, hidden_size, head_num, layer_num):
        q = self.q_linears[layer_num](x)
        k = self.k_linears[layer_num](x)
        v = self.v_linears[layer_num](x)
        q_one_head = self.muti_head(q, head_num)
        k_one_head = self.muti_head(k, head_num)
        v_one_head = self.muti_head(v, head_num)
        qk = torch.matmul(q_one_head, k_one_head.transpose(2, 3))
        dk = hidden_size / head_num
        qk_dk = qk / np.sqrt(dk)
        for i in range(qk.shape[0]):
            for j in range(i + 1, qk.shape[0]):
                qk_dk[:, :, i, j] = float("-inf")
        softmax = self.softmax(qk_dk)
        attention = torch.matmul(softmax, v_one_head).transpose(1, 2).reshape(x.shape)
        attention = self.dropout(attention)
        return attention

    def feed_forward(self, x, layer_num):
        x = self.feed_linear_in_list[layer_num](x)
        x = self.gelu(x)
        feed = self.feed_linear_out_list[layer_num](x)
        feed = self.dropout(feed)
        return feed

    def forward(self, tensor_token, head_num=12):
        x = self.embedding_forward(tensor_token)
        for i in range(self.num_hidden_layer):
            attention = self.self_attention_forward(x, self.hidden_size, head_num, i)
            x = self.cc_layer_norm_embedding_attention_list[i](x + attention)
            feed = self.feed_forward(x, i)
            x = self.cc_layer_norm_feed_list[i](x + feed)
        return x