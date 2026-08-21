1. 将数据集路径由 cluener 修改为 peoples_daily。

2. 数据读取模块由 CLUENER 的实体区间标注格式，
改为人民日报数据集提供的 tokens + ner_tags 格式。

3. 标签映射由手工构造改为根据 label_names.json 自动生成。

4. BERT编码器、Linear分类层、CRF解码层、
损失函数及训练流程保持不变。

tokens = sample["tokens"]
tags = sample["ner_tags"]


with open(
    "data/peoples_daily/label_names.json",
    "r",
    encoding="utf8"
) as f:
    labels = json.load(f)

tag2id = {
    tag: idx
    for idx, tag in enumerate(labels)
}

id2tag = {
    idx: tag
    for tag, idx in tag2id.items()
}

TRAIN_FILE = "data/peoples_daily/train.json"
DEV_FILE = "data/peoples_daily/validation.json"
TEST_FILE = "data/peoples_daily/test.json"

