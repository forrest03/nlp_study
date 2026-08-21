from Corpus import Corpus
from Vocab import Vocab
import torch
import math
import torch.nn as nn
from CharDataset import CharDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
'''
定义模型结构
'''
class LargeLanguage(nn.Module):
    #模型初始化
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_heads=4, num_layers=2, dropout=0.1, max_seq_len=128):
        super().__init__()
        #以下是需要初始化传进来的参数
        # 1 embedding 层的维度数量
        self.embed_dim = embed_dim
        # 2 hidden 层的维度数量
        self.hidden_dim = hidden_dim
        #多头注意力的头数
        self.num_heads = num_heads
        #GPT block数量
        self.num_layers = num_layers
        # dropout层概率
        self.dropout_rate = dropout

        #embedding 
        self.embed = nn.Embedding(vocab_size, self.embed_dim)
        
        #位置编码
        self.max_seq_len = max_seq_len
        self.pos_embed = nn.Embedding(self.max_seq_len, self.embed_dim)

        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, # 输入维度
            nhead=self.num_heads, # 多头注意力的头数
            dim_feedforward=self.hidden_dim, # 前馈层的维度
            dropout=self.dropout_rate, # dropout层概率
            activation="gelu", # 激活函数
            batch_first=True # 输入张量形状使用 [batch_size, seq_len, hidden_size]
        )
        
        # 叠加多个 GPT block
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        # dropout层
        self.dropout = nn.Dropout(self.dropout_rate)
        # 最后的输出层：把 768 维的特征，转换成预测每一个字的得分 (vocab_size 维)
        self.fc_out = nn.Linear(self.embed_dim, vocab_size)

    # 下三角掩码
    def generate_square_subsequent_mask(self, sz):
        # 1. sz是窗口的长度，也就是铺一个窗口长宽的全1矩阵
        paper = torch.ones(sz, sz)
        
        # 2. 下半部分剪掉，只保留右上角（上三角）
        # torch.triu 函数要学习下 
        upper_triangle = torch.triu(paper)
        
        # 3. 把它变成布尔值 (True/False)，方便后面操作
        bool_matrix = (upper_triangle == 1)
        
        # 4. 翻转
        lower_triangle_bool = bool_matrix.transpose(0, 1)
        mask = lower_triangle_bool.float()
        
        # 变成负无穷大 (-inf)
        mask = mask.masked_fill(mask == 0.0, float('-inf'))
    
        mask = mask.masked_fill(mask == 1.0, float(0.0))
        
        return mask

    # 前向传播
    def forward(self, x):
        if x.size(1) > self.max_seq_len:
            x = x[:, -self.max_seq_len:]

        # 获取词向量
        embed_x = self.embed(x)
        
        # 加入位置信息
        seq_len = x.size(1)
        # 生成 0 到 seq_len-1 的位置索引 
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        # 加上位置向量
        embed_x = embed_x + self.pos_embed(positions)
        embed_x = self.dropout(embed_x)
        
        # 生成掩码
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        
        out = self.transformer(embed_x, mask=tgt_mask)
        
        # 4. 通过最后的线性层，概率分不
        logits = self.fc_out(out)
        return logits
    
    # 文本生成
    def generate(self, start_text, c2i, i2c, max_len, device, top_p=0.9):
        with torch.no_grad():
            self.eval() 
            
            # 把起始文本转成索引张量
            input_indices = [c2i.get(char, c2i.get("<UNK>", 0)) for char in start_text]
            input_tensor = torch.tensor([input_indices], dtype=torch.long).to(device)
            
            generated_chars = []
            
            # 循环生成后续字符
            for _ in range(max_len):
                if input_tensor.size(1) > self.max_seq_len:
                    input_tensor = input_tensor[:, -self.max_seq_len:]

                logits = self(input_tensor) 
                
                # 取最后一个位置的输出
                last_char_logits = logits[0, -1, :]
                
                # 将 logits 转换为概率
                probs = torch.softmax(last_char_logits, dim=-1)
                
                # 采样
                # 1. 把概率从大到小排序
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                
                # 2. 计算累积概率
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # 3. top-p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                probs[indices_to_remove] = 0.0
                
                probs = probs / probs.sum()
                
                # 6. 依概率随机抽一个字
                predicted_idx = torch.multinomial(probs, num_samples=1).item()
                
                # 4. 把预测出来的数字变回汉字
                predicted_char = i2c.get(predicted_idx, "")
                generated_chars.append(predicted_char)
                
                # 5. 把预测出来的数字拼接到原来的句子里，作为下一轮的输入
                # 形状变成 [1, seq_len + 1]
                predicted_tensor = torch.tensor([[predicted_idx]], dtype=torch.long).to(device)
                input_tensor = torch.cat([input_tensor, predicted_tensor], dim=1)
                
            # 把列表里的字拼成一个完整的字符串返回
            return start_text + "".join(generated_chars)




#主函数，过程控制
def main():
    print("================ 欢迎使用语言模型 ================")
    mode = input("请选择要进行的操作： (1: 训练新模型, 2: 直接测试已有模型) > ")
    
    # 判断电脑的训练设备
    if torch.cuda.is_available():
        #英伟达
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        #苹果m芯片
        device = torch.device("mps")
    else:
        #cpu
        device = torch.device("cpu")
    print(f"----->使用设备：{device}")

    if mode == '2':
        # ==================== 测试模式 ====================
        checkpoint_path = "gpt_model_checkpoint.pth"
        print(f"正在加载模型文件: {checkpoint_path} ...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except FileNotFoundError:
            print("错误：找不到模型文件！请先选择 1 进行训练。")
            return
            
        c2i = checkpoint["char2idx"]
        i2c = checkpoint["idx2char"]
        vocab_size = checkpoint["vocab_size"]
        
        # 加载模型
        model = LargeLanguage(
            vocab_size=vocab_size,
            embed_dim=checkpoint["embed_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_heads=checkpoint.get("num_heads", 12),
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint.get("dropout", 0.1),
            max_seq_len=checkpoint.get("max_seq_len", 512)
        )
        
        # 权重
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)#放到我的苹果m芯片上
        model.eval()
        print("----->模型加载成功！")
        
    else:
        # ==================== 训练模式 ====================
        #-------------以下是参数区域----------------
        #预料位置
        corpus_path = "/Users/zhouyang/myworkspace/badou-nlp/周扬/week05/corpus.txt"
        #训练集比例
        train_ratio = 0.9#90%的数据作为训练集
        #预料采集窗口大小，滑动窗口
        seq_len = 32
        #批次大小
        batch_size = 32
        #学习率
        lr = 3e-4
        #训练轮数
        num_epochs = 10
        #模型参数
        embed_dim = 128
        hidden_dim = 256
        num_heads = 4
        num_layers = 2
        dropout = 0.1
        max_seq_len = 128
    
        #预料处理好，实例化预料对象
        crp = Corpus(corpus_path)
        texts = crp.load_train_data()
        print(f"----->预料数据加载完毕，共有{len(texts)}个字符")
        #预料加载完毕后，构建词表
        vb = Vocab(texts)
        i2c , c2i = vb.build_vocab()
        print(f"----->词表构建完毕,词表中共有{len(i2c)}个字符（去重了）")
    
        #把预料根据训练集比例切分出来
        tsi = len(texts) * train_ratio#90%在什么长度位置
        train_texts = texts[0:int(tsi)]#训练集
        val_texts = texts[int(tsi):]#验证集
        print(f"----->训练集有{len(train_texts)}个字符")
        print(f"----->验证集有{len(val_texts)}个字符")
    
        #实例化数据集对象
        train_ds = CharDataset(train_texts, c2i, seq_len = seq_len)
        val_ds = CharDataset(val_texts, c2i, seq_len = seq_len)
        print(f"----->训练集有{len(train_ds)}个样本")
        print(f"----->验证集有{len(val_ds)}个样本")
    
        #shuffle=True, 每个epoch都打乱数据
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    
        #获取词表大小
        vocab_size = len(i2c)
        #实力化模型，并开始训练
        model = LargeLanguage(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_len=max_seq_len
        ).to(device)
        model.train()
        
        #交叉熵损失函数
        criterion = nn.CrossEntropyLoss()
    
        #优化器 
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    
        best_ppl = float("inf")
        for epoch in range(num_epochs):
            # ========== 1. 训练阶段 ==========
            model.train()
            total_train_loss = 0.0
            total_train_tokens = 0
            # 使用 tqdm 包装 train_loader，显示进度条 进度条是ai帮我写的
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for x , y in pbar:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                total_train_loss += loss.item() * y.numel()
                total_train_tokens += y.numel()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                # 在进度条尾部实时显示当前的 loss
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
            avg_train_loss = total_train_loss / total_train_tokens
            train_ppl = math.exp(avg_train_loss)
            
            # 验证
            model.eval()
            total_val_loss = 0.0
            total_val_tokens = 0
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
                for x, y in val_pbar:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                    total_val_loss += loss.item() * y.numel()
                    total_val_tokens += y.numel()
                    val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            avg_val_loss = total_val_loss / total_val_tokens
            val_ppl = math.exp(avg_val_loss)

            # 记录历史最佳模型 (应该用验证集的 ppl 来评判)
            if val_ppl < best_ppl:
                best_ppl = val_ppl
                torch.save(model.state_dict(), "best_model.pth")
                print(f"\n★ 第{epoch+1}轮 验证集PPL ({val_ppl:.4f}) 创历史新低，已保存最佳模型！")
                
            print(f"-----> 第{epoch+1}轮总结: 训练损失={avg_train_loss:.4f}, 训练PPL={train_ppl:.4f} | 验证损失={avg_val_loss:.4f}, 验证PPL={val_ppl:.4f}\n")
            
        print("\n================ 训练结束 ================")
        # 训练结束后，保存模型权重、词表和超参数，供预测脚本使用
        save_path = "gpt_model_checkpoint.pth"
        torch.save({
            "model_state": model.state_dict(),
            "char2idx": c2i,
            "idx2char": i2c,
            "vocab_size": vocab_size,
            "embed_dim": model.embed_dim,
            "hidden_dim": model.hidden_dim,
            "num_heads": model.num_heads,
            "num_layers": model.num_layers,
            "dropout": model.dropout_rate,
            "max_seq_len": model.max_seq_len
        }, save_path)
        print(f"模型和词表已保存到: {save_path}")
        
    print("\n================ 开始对话测试 ================")
    while True:
        user_input = input("请输入文字 ：")
            
        if not user_input.strip():
            continue
            
        # 调用模型的 generate 函数，让它往下续写 50 个字
        output_text = model.generate(
            start_text=user_input, 
            c2i=c2i, 
            i2c=i2c, 
            max_len=50, 
            device=device,
            top_p=0.9
        )
        print(f"\n[模型续写]: \n{output_text}\n")
        print("-" * 50)
            
if __name__ == "__main__":
    main()
