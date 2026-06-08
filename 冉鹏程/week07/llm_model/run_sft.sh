#!/usr/bin/env bash
# LLM SFT 训练启动脚本（覆盖核心参数）
set -e

cd "$(dirname "$0")"

# ── 核心参数 ──────────────────────────────────────────────────────────────
MODEL_PATH="/root/.cache/modelscope/hub/models/Qwen/Qwen3-4B"
DATA_DIR="../data"
OUTPUT_DIR="../outputs"

NUM_TRAIN=-1            # -1 使用全部样本
EPOCHS=3
BATCH_SIZE=4
GRAD_ACCUM=4            # 有效 batch = BATCH_SIZE * GRAD_ACCUM = 16
MAX_LENGTH=256
LR=2e-4                 # LoRA 默认 2e-4；全量微调建议 2e-5
LORA_R=8
LORA_ALPHA=16
SEED=42
DEVICE="cuda:6"

# ── 启动训练 ──────────────────────────────────────────────────────────────
python llm_train_sft.py \
    --model_path   "$MODEL_PATH" \
    --data_dir     "$DATA_DIR" \
    --output_dir   "$OUTPUT_DIR" \
    --num_train    "$NUM_TRAIN" \
    --epochs       "$EPOCHS" \
    --batch_size   "$BATCH_SIZE" \
    --grad_accum   "$GRAD_ACCUM" \
    --max_length   "$MAX_LENGTH" \
    --lr           "$LR" \
    --lora_r       "$LORA_R" \
    --lora_alpha   "$LORA_ALPHA" \
    --seed         "$SEED" \
    --device       "$DEVICE"

# 如需全量微调，追加 --full_ft（lr 会自动切到 2e-5）：
#     python llm_train_sft.py ... --full_ft
