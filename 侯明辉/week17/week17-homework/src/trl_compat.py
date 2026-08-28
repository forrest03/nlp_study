"""
trl 兼容性补丁 —— trl 1.12 已经修复了主项目踩的所有 bug（2024 年 trl 0.21 + transformers 5.x 的兼容问题）。

本文件保留是为了：
  1. 复用主项目的导入风格（`import trl_compat` 永远先于 `from trl import`）
  2. 若未来 trl/transformers 又出现不兼容，monkeypatch 逻辑都在这里

当前 trl 1.12 + transformers 5.16 + peft 0.19 环境下，本模块是 no-op：
  - trl.import_utils.is_*_available() 已正确返回 bool
  - PreTrainedModel.warnings_issued 已存在
  - bf16 模型加载已正确处理

用法（保持主项目写法）：
  import trl_compat  # noqa: F401
  from trl import GRPOTrainer, GRPOConfig
"""


def _noop():
    """trl >= 1.12 不需要任何补丁。"""
    pass


_noop()