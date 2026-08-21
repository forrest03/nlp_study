import sys
print("1. Starting test...")
sys.stdout.flush()

import json
import os
import torch
from transformers import BertTokenizer

print("2. Imports successful")
sys.stdout.flush()

# Test data loading
data_path = 'D:\\workspace\\hub-TroE\\张福\\week07\\data\\peoples_daily\\train.json'
print(f"3. Loading data from {data_path}")
sys.stdout.flush()

with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"4. Loaded {len(data)} samples")
sys.stdout.flush()

# Test label loading
label_path = 'D:\\workspace\\hub-TroE\\张福\\week07\\data\\peoples_daily\\label_names.json'
with open(label_path, 'r', encoding='utf-8') as f:
    labels = json.load(f)
print(f"5. Labels: {labels}")
sys.stdout.flush()

# Test tokenizer
print("6. Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
print("7. Tokenizer loaded")
sys.stdout.flush()

# Test model
print("8. Testing model import...")
from bert_ner_model import BertNERModel
print("9. Model imported")
sys.stdout.flush()

# Test dataset
print("10. Testing dataset import...")
from ner_dataset import NERDataset
print("11. Dataset imported")
sys.stdout.flush()

# Create small dataset
print("12. Creating dataset...")
dataset = NERDataset(data_path, tokenizer, max_len=64)
print(f"13. Dataset created with {len(dataset)} samples")
sys.stdout.flush()

# Test model creation
print("14. Creating model...")
model = BertNERModel(len(dataset.label2id))
print(f"15. Model created with {len(dataset.label2id)} labels")
sys.stdout.flush()

print("\n16. All tests passed!")
sys.stdout.flush()