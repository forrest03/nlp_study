"""
trl 0.21 + transformers 5.x 兼容性修复（必须在 import trl 训练器之前执行）

用法（所有使用 trl 的脚本，第一行先 import 本模块）：
  import trl_compat  # noqa: F401  必须先于 trl 导入
  from trl import GRPOTrainer, GRPOConfig
"""


def _patch_trl_availability_flags():
    import trl.import_utils as tiu

    for name in dir(tiu):
        if not (name.startswith("is_") and name.endswith("_available")):
            continue
        fn = getattr(tiu, name)
        if not callable(fn):
            continue
        try:
            val = fn()
        except Exception:
            continue
        if isinstance(val, tuple):  # transformers 5.x 的元组返回值 → 取真实布尔位
            real = bool(val[0])
            setattr(tiu, name, lambda *a, _r=real, **k: _r)


def _patch_warnings_issued():
    """transformers 5.x 移除了 PreTrainedModel.warnings_issued（4.x 的警告去重字典），
    trl 0.21 的 GRPOTrainer 初始化时仍直接读写它。补一个类级别的空字典即可——
    trl 只是往里面写标记位，不影响训练逻辑。"""
    from transformers import PreTrainedModel

    if not hasattr(PreTrainedModel, "warnings_issued"):
        PreTrainedModel.warnings_issued = {}


_patch_trl_availability_flags()
_patch_warnings_issued()
