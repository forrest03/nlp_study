import torch
from torch.utils.data import Dataset



class CharDataset(Dataset):
    def __init__(self, text, c2i, seq_len):
        self.seq_len = seq_len
        ids = []
        for c in text:
            if c in c2i:
                # 查字典，把汉字换成对应的数字ID，存起来
                ids.append(c2i[c])
                
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        # 1. 共有多少个字
        total_length = len(self.data)
        
        # 2. 算一切出多少刀
        # (比如总长 100，每次切 64，那最多只能从第 36 个字开始切)
        possible_cuts = total_length - self.seq_len
        
        # 3. 万一原始文本太短了会是负数 (-54)。
        # 用 max(0, ...) 兜底，
        final_count = max(0, possible_cuts)
        
        return final_count

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y