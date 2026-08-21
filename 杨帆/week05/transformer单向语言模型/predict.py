import torch
from language_model import LM  # 直接引用你训练的模型！

# ===================== 配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pt"
MAX_SEQ_LEN = 4500
# ==================================================

# 加载保存的权重 + 词表 + 训练参数
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
char2idx = checkpoint["char2idx"]
idx2char = checkpoint["idx2char"]
train_args = checkpoint["args"]
vocab_size = len(char2idx)

# ===================== 直接加载你训练的 LM 模型 =====================
# 完全用你训练时的参数初始化
model = LM(
    vocab_size=vocab_size,
    hidden_dim=train_args["hidden_dim"],
    dropout=train_args["dropout"],
    layer_nums=train_args["layer_nums"]
).to(DEVICE)

# 直接加载权重（不需要修改任何 key！）
model.load_state_dict(checkpoint["model_state"])
model.eval()

# ===================== 预测下一个字 =====================
def predict_next_char(input_text, last_char=None):
    # 文本转索引
    input_indices = [char2idx[c] for c in input_text if c in char2idx]
    input_indices = input_indices[-MAX_SEQ_LEN:]
    x = torch.tensor([input_indices]).to(DEVICE)

    with torch.no_grad():
        logits = model(x)[:, -1, :]  #  shape: [1, 词表大小]

    # 1. 强制屏蔽上一个字，杜绝连续复读
    if last_char and last_char in char2idx:
        logits[0, char2idx[last_char]] = -1e10

    # ========== 正确 top-k 采样（无报错、不复读）==========
    top_k = 3
    temp = 0.7  # 温度越低越保守通顺，0.6~0.8最佳
    values, indices = torch.topk(logits / temp, top_k)

    # 强制转成 一维 数组，绝对安全
    values = values.view(-1)  # [5]
    indices = indices.view(-1)  # [5]

    probs = torch.softmax(values, dim=-1)
    idx = torch.multinomial(probs, 1).item()

    # 安全取值
    next_id = indices[idx].item()
    return idx2char[next_id]
# ===================== 运行 =====================
if __name__ == "__main__":
    print("✅ 模型加载成功！输入文字预测下一个字，quit 退出\n")
    inp = input("请输入前面的字：").strip()
    print(inp, end="", flush=True)
    i = 0
    last = ""
    while i < 1000:
        res = predict_next_char(inp, last)
        print(res, end="", flush=True)
        last = res
        inp = inp + res
        i += 1