@echo off
cd /d "D:\workspace\hub-TroE\张福\week07\bert_model"
"D:\Python\anaconda3\envs\ai_swap\python.exe" train_bert_ner.py --max_len 64 --batch_size 32 --epochs 10 --lr 2e-5