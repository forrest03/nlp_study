"""统一解析 Qwen2-0.5B-Instruct 基座模型路径。"""
import os
from pathlib import Path

MODEL_NAME = "Qwen2-0.5B-Instruct"


def get_model_path() -> Path:
    env = os.getenv("GRPO_MODEL_PATH")
    if env:
        return Path(env)

    here = Path(__file__).resolve().parent
    # 从当前文件向上查找 八斗/pretrain_models/Qwen2-0.5B-Instruct
    for parent in here.parents:
        candidate = parent / "pretrain_models" / MODEL_NAME
        if candidate.is_dir():
            return candidate

    # 默认回落到 homework 目录上 4 层的 八斗 根目录
    badou_root = here.parents[4]
    return badou_root / "pretrain_models" / MODEL_NAME
