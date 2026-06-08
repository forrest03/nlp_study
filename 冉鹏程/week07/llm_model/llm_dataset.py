import json

from torch.utils.data import Dataset
import torch

SYSTEM_PROMPT = (
    "你是一个命名实体识别助手，从文本中识别命名实体，以 JSON 格式输出。\n"
    "实体类型（英文标识）：person（人）、location（地点）、organization（组织机构）。\n"
    '输出格式（严格遵守，不输出其他内容）：{"entities": [{"text": "实体文本", "type": "实体类型"}]}\n'
    '无实体时输出：{"entities": []}'
)


def record_to_target(record):
    entities = []
    target = {"entities": entities}

    entity = ''
    entity_type = None

    tokens = record['tokens']
    ner_tags = record['ner_tags']
    for token, tag in zip(tokens, ner_tags):
        if tag != 'O':
            if tag.startswith('B-'):
                if entity:
                    entities.append({"text": entity, "type": entity_type})
                entity = token
                if tag == "B-PER":
                    entity_type = "person"
                elif tag == "B-ORG":
                    entity_type = "organization"
                elif tag == "B-LOC":
                    entity_type = "location"
                continue
            entity += token

        else:
            if entity:
                entities.append({"text": entity, "type": entity_type})
            entity = ''
            entity_type = None
    if entity:
        entities.append({"text": entity, "type": entity_type})
    return json.dumps(target, ensure_ascii=False)


class SFTDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=256):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        target = record_to_target(record)
        text = "".join(record["tokens"])

        # step 1 : 构建 Prompt（tokenize=False 兼容 transformers 5.x）
        # apply_chat_template 是 transformers 提供的对话格式化工具
        # 它会自动添加特殊的 token（如 <|im_start|>, <|im_end|> 等）
        prompt_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ],
            tokenize=False,  # 只返回文本，不进行 tokenization（兼容新版本）
            add_generation_prompt=True  # 添加 assistant 开头标记（如 "<|im_start|>assistant\n"）
        )
        # 将prompt转换为ids，不包含特殊token
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        # step 2 ： 构建response（目标输出）
        response_ids = (
                self.tokenizer.encode(target, add_special_tokens=False)  # json转id
                + [self.tokenizer.eos_token_id]  # 添加结束符 EOS
        )

        # step 3 ： 构建input_ids （拼接 prompt + response，并截断到 max_length）
        input_ids = (prompt_ids + response_ids)[:self.max_length]

        # step 4 ： 构建 labels （带mask的标签）
        # 因果模型不需要将labels与input_ids进行错位处理，是因为模型会自动将labels与input_ids进行错位处理
        labels = ([-100] * len(prompt_ids) + response_ids)[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
