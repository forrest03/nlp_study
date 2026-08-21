# Qwen3.8-27B 模型配置与推理

## 模型概述

| 属性 | 值 |
|------|-----|
| 模型名称 | Qwen3.8-27B |
| 架构 | Qwen3_5ForConditionalGeneration |
| 模型类型 | 视觉-语言多模态（图片 + 视频） |
| 来源 | [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3.8-27B) |
| 发布日期 | 2026-08-15 |

## 核心架构参数

### 文本模型
| 参数 | 值 |
|------|-----|
| 层数 | 64 |
| 隐藏维度 | 5120 |
| 中间维度 | 17408 |
| 注意力头数 | 24 |
| KV 头数 (GQA) | 4 |
| 头维度 | 256 |
| 词表大小 | 248,320 |
| 最大上下文 | 262,144 (256K) |
| 激活函数 | SiLU |
| RoPE theta | 10,000,000 |

### 注意力层混合设计
- **64 层中**: 16 层全注意力 + 48 层线性注意力 (MLA)
- 每 4 层插入一个全注意力层（full_attention_interval=4）
- MLA 层: 16 个 key heads (dim=128), 48 个 value heads (dim=128)

### 视觉编码器
| 参数 | 值 |
|------|-----|
| 层数 | 27 |
| 隐藏维度 | 1152 |
| 中间维度 | 4304 |
| 注意力头数 | 16 |
| Patch 大小 | 16×16 |
| 输出维度 | 5120 |
| 空间合并 | 2×2 |

## 文件说明

### 模型定义代码（从 transformers 5.9.0 复制）
model 代码已内置在 transformers 库中，无需额外下载。这里复制了一份供参考：

```
Qwen3.8-27B_config/Qwen/Qwen3.8-27B/
├── modeling_qwen3_5.py            # 模型定义（Qwen3_5ForConditionalGeneration 等）
├── configuration_qwen3_5.py       # 配置类定义（Qwen3_5Config 等）
├── tokenization_qwen3_5.py        # 分词器定义
├── modular_qwen3_5.py             # 模块化组件定义
```

### 配置文件（从 ModelScope 下载）
```
Qwen3.8-27B_config/Qwen/Qwen3.8-27B/
├── config.json                    # 模型架构配置
├── configuration.json             # 模型配置（简化版）
├── generation_config.json         # 生成参数（temperature=1.0, top_k=20, top_p=0.95）
├── tokenizer.json                 # 分词器
├── tokenizer_config.json          # 分词器配置 + chat_template
├── vocab.json                     # 词汇表
├── merges.txt                     # BPE 合并规则
├── chat_template.jinja            # 对话模板（独立文件）
├── preprocessor_config.json       # 图像预处理配置
├── video_preprocessor_config.json # 视频预处理配置
├── model.safetensors.index.json   # 权重索引（18个分片，共约54GB）
```

## 特殊 Token 系统

| Token ID | 符号 | 用途 |
|----------|------|------|
| 248044 | `<|endoftext|>` | BOS / EOS / PAD |
| 248045 | `<|im_start|>` | 消息开始 |
| 248046 | `<|im_end|>` | 消息结束 |
| 248053 | `<|vision_start|>` | 视觉内容开始 |
| 248054 | `<|vision_end|>` | 视觉内容结束 |
| 248055 | `<|vision_pad|>` | 视觉占位 |
| 248056 | `<|image_pad|>` | 图片占位 |
| 248057 | `<|video_pad|>` | 视频占位 |
| 248058 | `<tool_call>` | 工具调用开始 |
| 248059 | `</tool_call>` | 工具调用结束 |
| 248066 | `<tool_response>` | 工具响应开始 |
| 248067 | `</tool_response>` | 工具响应结束 |
| 248068 | ` thinking` | 思考内容标记 |
| 248069 | ` response` | 回复内容标记 |
| 248070 | `<|audio_start|>` | 音频开始 |
| 248071 | `<|audio_end|>` | 音频结束 |

## 对话格式

```
<|im_start|>system
系统提示词<|im_end|>
<|im_start|>user
用户输入<|im_end|>
<|im_start|>assistant
 thinking
内部推理过程...
 response

最终回复<|im_end|>
```

### 多模态对话格式

```
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>描述这张图片<|im_end|>
```

### 工具调用格式

```
<|im_start|>assistant
<tool_call>
<function=get_weather>
<parameter=city>
北京
</parameter>
</function>
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"temperature": 25, "weather": "晴"}
</tool_response><|im_end|>
```

## 使用方式

### 1. 查看配置信息

```bash
cd /Users/zhouyang/myworkspace/badou-nlp/周扬/week16
python inference.py
```

### 2. 文本推理

```bash
python inference.py --text "你好，请介绍一下自己"
```

### 3. 图片推理

```bash
python inference.py --text "图中是什么?" --image path/to/image.jpg
```

### 4. 完整推理（需要权重）

```python
from transformers import AutoModelForConditionalGeneration, AutoTokenizer

MODEL_DIR = "Qwen3.8-27B_config/Qwen/Qwen3.8-27B"

model = AutoModelForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

messages = [{"role": "user", "content": "你好"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 生成参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| temperature | 1.0 | 采样温度 |
| top_k | 20 | Top-K 采样 |
| top_p | 0.95 | Nucleus 采样 |
| do_sample | True | 启用采样 |

## 环境要求

- transformers >= 5.8.0 (需支持 `qwen3_5` 模型类型)
- torch >= 2.0
- 显存: 约 54GB (bf16) / 27GB (int4量化)