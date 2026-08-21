"""
基于 Transformer 的单向字符级语言模型训练与文本生成脚本。

1. 从当前目录的 corpus.txt 读取中文文本语料。
2. 构建字符级词表，将每个中文字符、标点、数字等映射成 token id。
3. 使用带因果 mask 的 Transformer 训练自回归语言模型。
4. 训练时学习“根据前面的字符预测下一个字符”的条件概率。
5. 保存验证集 loss 最低的 checkpoint，并支持从 checkpoint 重新加载生成文本。

"""

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F



@dataclass
class ModelConfig:
    """
    模型结构配置。

    vocab_size:
        词表大小，corpus.txt 中不同字符的数量。
    max_seq_len:
        模型一次最多读取的上下文长度。生成时超过该长度会截断到最后 max_seq_len 个 token
    d_model:
        token embedding 和 Transformer 隐藏状态的维度
    n_heads:
        Multi-Head的head 数量
    n_layers:
        Transformer 层数。层数越多表达能力越强
    dim_feedforward:
        FFN的维度
    dropout:
        训练时的随机失活比例，用于缓解过拟合
    """

    vocab_size: int
    max_seq_len: int = 128
    d_model: int = 128
    n_heads: int = 12
    n_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.1


class CharTokenizer:
    """
    字符级 tokenizer
    一个字符就是一个 token
    """

    def __init__(self, chars):
        # stoi: string to index，字符到 id 的映射
        # itos: index to string，id 到字符的映射
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @classmethod
    def from_text(cls, text):
        """
        从训练语料构建词表。
        sorted(set(text)) 让词表顺序固定，保证相同语料下每次运行得到的字符到 id 映射一致。
        """
        return cls(sorted(set(text)))

    def encode(self, text):
        """把字符串转换成 token id 列表。"""
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        """把 token id 列表还原成字符串。"""
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path):
        """保存字符词表，方便单独查看或复现实验。"""
        data = {"chars": self.chars}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["chars"])


class CausalTransformerLM(nn.Module):
    """
    基于 TransformerEncoder的单向字符级语言模型。

    PyTorch 的 TransformerEncoder 默认并不是单向的。
    通过 forward 中传入的 causal_mask 保证“单向”特性。
    每个位置只能看到自己和之前的位置，不能看到未来位置。

    输入形状:
        idx: LongTensor，形状为 [batch_size, seq_len]

    输出形状:
        logits: FloatTensor，形状为 [batch_size, seq_len, vocab_size]
        loss: 如果传入 targets，则返回交叉熵；否则为 None
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False,
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        # lm_head 将每个位置的隐藏状态映射为词表上每个字符的 logits。
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # 小标准差正态初始化是 Transformer 语言模型里常见的稳定起点。
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """
        前向传播。

        idx 中的每个数字都是一个字符 id。
        先查 token embedding，再加上 position embedding，得到包含字符身份和位置信息的向量序列。
        然后通过带 causal_mask 的 Transformer，
        最后用 lm_head 把每个位置的隐藏状态映射成词表大小的 logits。

        提供 targets，计算每个位置预测“下一个字符”的交叉熵损失。
        """
        batch_size, seq_len = idx.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"序列长度 {seq_len} 超过 max_seq_len={self.config.max_seq_len}")

        pos = torch.arange(seq_len, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)[None, :, :]
        
        # 上三角 mask 屏蔽未来位置，保证第 t 个字符只能看见自己和之前的字符。
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=idx.device),
            diagonal=1,
        )
        x = self.blocks(x, mask=causal_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # logits/targets 展平成二维/一维后计算所有 batch、所有位置的下一个字符交叉熵。
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        自回归生成文本。

        每次循环只生成一个新 token：
        1. 把当前上下文输入模型。
        2. 取最后一个位置的 logits 作为下一个字符的概率分布。
        3. 用 temperature 和 top_k 调整采样分布。
        4. 采样得到下一个 token，并拼接回上下文。

        temperature 越低，输出越保守；temperature 越高，输出越随机。
        top_k 会限制候选字符数量，减少极低概率字符被采样的机会。
        """
        self.eval()
        for _ in range(max_new_tokens):
            # 只保留模型能处理的最后 max_seq_len 个 token 作为上下文窗口。
            idx_cond = idx[:, -self.config.max_seq_len :]
            logits, _ = self(idx_cond)
            
            # 取最后一个位置的分布来预测下一个字符。
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                # top-k 采样只允许概率最高的 k 个候选参与抽样，减少低质量随机字符。
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


def set_seed(seed):
    """固定随机种子，使训练和生成结果尽量可复现。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_corpus(path):
    """
    读取语料文件。

    utf-8-sig 可以兼容带 BOM 的 UTF-8 文件；统一换行符后，Windows 和其他系统下的训练结果
    会更加一致。
    """
    text = Path(path).read_text(encoding="utf-8-sig")
    text = text.replace("\r\n", "\n")
    if len(text) < 2:
        raise ValueError("corpus.txt 内容太短，无法训练语言模型")
    return text


def split_data(ids, train_ratio, device):
    """
    切分训练集和验证集。

    这里按原始 token 序列顺序切分，而不是随机打散。语言模型训练依赖连续文本片段，
    每个 batch 内部再随机抽取窗口即可。
    """
    split_idx = max(1, int(len(ids) * train_ratio))
    split_idx = min(split_idx, len(ids) - 1)
    data = torch.tensor(ids, dtype=torch.long)
    return data[:split_idx].to(device), data[split_idx:].to(device)


def get_batch(data, batch_size, seq_len, device):
    """
    从一整段 token 序列中随机抽取一个 batch。

    x 是长度为 seq_len 的上下文序列。
    y 是 x 右移一位后的目标序列。

    例如文本片段为 “黄金上涨”，如果 x 是 “黄金上”，则 y 是 “金上涨”。
    模型在每个位置都要预测下一个字符。
    """
    if len(data) <= seq_len + 1:
        raise ValueError("语料太短，请减小 --seq-len 或使用更大的 corpus.txt")
    # 随机抽取多个连续片段；x 是当前字符序列，y 是整体右移一位后的目标序列。
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,), device=device)
    x = torch.stack([data[i : i + seq_len] for i in ix])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, args, device):
    """
    估计训练集和验证集 loss。

    为了节省时间，不遍历完整数据集，而是随机采样 eval_iters 个 batch 后取平均值。
    这足够用来观察模型是否在学习，以及验证集 loss 是否持续下降。
    """
    model.eval()
    result = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(args.eval_iters):
            xb, yb = get_batch(data, args.batch_size, args.seq_len, device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        result[split] = sum(losses) / len(losses)
    model.train()
    return result


def steps_per_epoch(train_tokens, batch_size, seq_len):
    """根据 token 数估算一个 epoch 大约需要多少个参数更新 step。"""
    return max(1, math.ceil(train_tokens / (batch_size * seq_len)))


def save_checkpoint(path, model, optimizer, tokenizer, args, step, best_val_loss):
    """
    保存训练 checkpoint。

    checkpoint 中包含：
    - model_state_dict: 模型参数
    - optimizer_state_dict: 优化器状态，后续可继续训练
    - model_config: 模型结构参数
    - chars: tokenizer 词表
    - train_args: 本次训练命令行参数
    - step/best_val_loss: 当前训练进度和最佳验证集指标
    """
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(model.config),
        "chars": tokenizer.chars,
        "train_args": vars(args),
        "step": step,
        "best_val_loss": best_val_loss,
    }
    torch.save(payload, path)


def load_model_for_generation(checkpoint_path, device):
    """
    从 checkpoint 恢复模型和 tokenizer，用于文本生成。

    生成阶段不需要优化器，因此这里只恢复模型结构、权重和词表。
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    tokenizer = CharTokenizer(checkpoint["chars"])
    config = ModelConfig(**checkpoint["model_config"])
    model = CausalTransformerLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer


def train(args):
    """
    完整训练流程。

    流程概览：
    1. 设置随机种子和运行设备。
    2. 读取 corpus.txt，并构造字符词表。
    3. 将文本编码为 token id 序列。
    4. 切分训练集和验证集。
    5. 初始化 CausalTransformerLM 和 AdamW 优化器。
    6. 循环抽 batch、计算 loss、反向传播、更新参数。
    7. 定期评估验证集 loss，并保存当前最优模型。
    8. 训练结束后用 prompt 生成一段文本示例。
    """
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    text = read_corpus(args.corpus)
    # 字符级语言模型的训练数据就是整段文本转成的一串字符 id。
    tokenizer = CharTokenizer.from_text(text)
    ids = tokenizer.encode(text)
    train_data, val_data = split_data(ids, args.train_ratio, device)

    config = ModelConfig(
        vocab_size=len(tokenizer.chars),
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    model = CausalTransformerLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    total_steps = args.max_steps or args.epochs * steps_per_epoch(len(train_data), args.batch_size, args.seq_len)
    best_val_loss = float("inf")
    print(f"device={device}, tokens={len(ids)}, vocab={len(tokenizer.chars)}, total_steps={total_steps}")

    for step in range(1, total_steps + 1):
        xb, yb = get_batch(train_data, args.batch_size, args.seq_len, device)
        _, loss = model(xb, yb)
        # 标准训练四步：清梯度、反向传播、梯度裁剪、参数更新。
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        should_log = step == 1 or step % args.log_interval == 0 or step == total_steps
        if should_log:
            losses = estimate_loss(model, train_data, val_data, args, device)
            # 困惑度 perplexity = exp(loss)，这里裁到 20 防止 loss 很大时数值溢出。
            ppl = math.exp(min(losses["val"], 20))
            print(
                f"step {step:5d}/{total_steps}: "
                f"train_loss={losses['train']:.4f}, val_loss={losses['val']:.4f}, val_ppl={ppl:.2f}"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                # 只保存验证集 loss 目前最好的模型，避免最后一步过拟合后覆盖更好权重。
                save_checkpoint(args.checkpoint, model, optimizer, tokenizer, args, step, best_val_loss)
                tokenizer.save(args.vocab_out)
                print(f"saved checkpoint -> {args.checkpoint}")

    prompt = args.prompt or text[: min(32, len(text))]
    print("\n生成示例：")
    print(generate_text(model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_k, device))


def generate_text(model, tokenizer, prompt, max_new_tokens, temperature, top_k, device):
    """
    对外层 generate 命令使用的文本生成函数。

    prompt 必须只包含训练语料出现过的字符，因为本 tokenizer 没有 <unk> 未知字符。
    """
    unknown = sorted(set(prompt) - set(tokenizer.stoi))
    if unknown:
        raise ValueError(f"prompt 中包含训练词表外字符：{unknown[:10]}")
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(output_ids[0].tolist())


def generate(args):
    """加载 checkpoint，并根据用户提供的 prompt 生成文本。"""
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tokenizer = load_model_for_generation(args.checkpoint, device)
    text = generate_text(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, args.top_k, device)
    print(text)


def parse_args():
    """
    命令行参数定义。

    常用训练命令：
        python train_transformer_lm.py --max-steps 300 --batch-size 16 --seq-len 96

    常用生成命令：
        python train_transformer_lm.py --mode generate --checkpoint transformer_lm.pt --prompt 黄金
    """
    parser = argparse.ArgumentParser(description="基于 Transformer 的单向字符级语言模型")
    # mode=train 会读取语料并训练；mode=generate 会从 checkpoint 载入模型并续写 prompt。
    parser.add_argument("--mode", choices=["train", "generate"], default="train")
    parser.add_argument("--corpus", default="corpus.txt")
    parser.add_argument("--checkpoint", default="transformer_lm.pt")
    parser.add_argument("--vocab-out", default="vocab.json")
    parser.add_argument("--device", default="", help="默认自动选择 cuda/cpu，也可手动指定 cpu 或 cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=50)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--prompt", default="黄金")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_steps == 0:
        args.max_steps = None
    if args.mode == "train":
        train(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
