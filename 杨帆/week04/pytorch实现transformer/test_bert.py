import numpy as np
import torch.nn as nn
import torch

class TestBert(nn.Module):
    def __init__(self, vocab, hidden_size=768, max_position_size=512, num_hidden_layer=12):
        super(TestBert, self).__init__()
        self.token_embedding = nn.Embedding(len(vocab), hidden_size)
        self.segment_embedding = nn.Embedding(2, hidden_size)
        self.position_embedding = nn.Embedding(max_position_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.cc_layer_norm_embedding_attention = nn.LayerNorm(hidden_size)
        self.feed_linear_in = nn.Linear(hidden_size, 4 * hidden_size)
        self.gelu = nn.GELU()
        self.feed_linear_out = nn.Linear(4 * hidden_size, hidden_size)
        self.cc_layer_norm_feed = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def embedding_forward(self, tensor_token, tensor_seg, tensor_pos):
        token = self.token_embedding(tensor_token)
        segment = self.segment_embedding(tensor_seg)
        position = self.position_embedding(tensor_pos)
        embedding = token + segment + position
        embedding = self.dropout(embedding)
        embedding = self.layer_norm(embedding)
        return embedding

    def muti_head(self, x, head_num):
        batch_size, max_len, hidden_size = x.shape
        x = x.reshape(batch_size, max_len, head_num, hidden_size // head_num).transpose(1, 2)  # batch_size, head_num, max_len, head_size
        return x

    def self_attention_forward(self, x, hidden_size, head_num):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        q_one_head = self.muti_head(q, head_num)
        k_one_head = self.muti_head(k, head_num)
        v_one_head = self.muti_head(v, head_num)
        qk = torch.matmul(q_one_head, k_one_head.transpose(2, 3))
        dk = hidden_size / head_num
        qk_dk = qk / np.sqrt(dk)
        softmax = self.softmax(qk_dk)
        attention = torch.matmul(softmax, v_one_head).transpose(1, 2).reshape(x.shape)
        attention = self.dropout(attention)
        return attention

    def feed_forward(self, x):
        x = self.feed_linear_in(x)
        x = self.gelu(x)
        feed = self.feed_linear_out(x)
        feed = self.dropout(feed)
        return feed

    def forward(self, tensor_token, tensor_seg, tensor_pos, hidden_size, head_num, num_hidden_layer=12):
        x = self.embedding_forward(tensor_token, tensor_seg, tensor_pos)
        for i in range(num_hidden_layer):
            attention = self.self_attention_forward(x, hidden_size, head_num)
            x = self.cc_layer_norm_embedding_attention(x + attention)
            feed = self.feed_forward(x)
            x = self.cc_layer_norm_feed(x + feed)
        return x

if __name__ == "__main__":
    hidden_size = 768
    max_position_size = 512
    head_num = 12
    vocab = {
        "[pad]": 0,
        "你": 1,
        "你好": 2,
        "中国": 3,
        "好": 4,
        "[cls]": 5,
        "[sep]": 6,
        "[unk]": 7
    }
    token = [5, 1, 2, 3, 6, 3, 4, 6]
    seg = [0, 0, 0, 0, 0, 1, 1, 1]
    pos = [0, 1, 2, 3, 4, 5, 6, 7]
    tensor_token = torch.LongTensor([token])
    tensor_seg = torch.LongTensor([seg])
    tensor_pos = torch.LongTensor([pos])
    model = TestBert(vocab, hidden_size, max_position_size)
    y = model.forward(tensor_token, tensor_seg, tensor_pos, hidden_size, head_num)
    print(y)
