运行 Device: CPU
模型文件: 使用 Hugging Face CLI 命令下载模型到本地
运行: 
  1. 修改代码来通过模型名称标识符来自动加载本地模型, 不用指定磁盘路径
  2. 微调 bert 模型时, 池化分别使用 CLS, Max 和 Mean 层 (在不指定权重的前提下, Pool=Mean时, 样本数最少的分类也能获得50%的正确率, 而Pool=cls, 其正确率只有不到1%)
  3. 受限于 CPU, 没有进行 Qwen 的全量微调, SFT LoRA 也只训练了一个 epoch
