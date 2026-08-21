"""
Qwen3.8-27B 核心推理程序

模型架构: Qwen3_5ForConditionalGeneration (视觉-语言多模态)
- 64层 Transformer，混合线性注意力(MLA) + 全注意力
- 5120 hidden_size, 24 attention heads, 4 KV heads (GQA)
- 262K 上下文窗口，支持 thinking/response 分离推理
- 原生支持图片、视频输入

使用方式:
  python inference.py --text "你好，请介绍一下自己"
  python inference.py --text "图中是什么?" --image path/to/image.jpg
  python inference.py --chat  # 交互式对话模式
"""

import argparse
import os

# 将配置目录添加到 Python 路径
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Qwen3.8-27B_config", "Qwen", "Qwen3.8-27B")


def load_model():
    """加载模型配置和 tokenizer（不加载权重，仅展示加载方式）"""
    from transformers import AutoTokenizer, AutoConfig, AutoProcessor

    print(f"加载配置: {CONFIG_DIR}")
    config = AutoConfig.from_pretrained(CONFIG_DIR, trust_remote_code=True)

    print(f"加载 tokenizer: {CONFIG_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG_DIR, trust_remote_code=True)

    # Qwen3.8 使用 Qwen2Tokenizer
    # 特殊 token 说明:
    #   <|im_start|> / <|im_end|> : 消息边界
    #   <|vision_start|> / <|vision_end|> : 视觉内容边界
    #   <|image_pad|> / <|video_pad|> : 图片/视频占位
    #   thinking / response : 思考/回复分离标记
    #   <tool_call> / <tool_response> : 工具调用/响应

    try:
        processor = AutoProcessor.from_pretrained(CONFIG_DIR, trust_remote_code=True)
    except ImportError:
        processor = None

    return config, tokenizer, processor


def build_chat_prompt(messages: list, enable_thinking: bool = True) -> str:
    """
    使用 chat_template 构建对话 prompt。

    消息格式:
      messages = [
          {"role": "system", "content": "你是一个有用的助手"},
          {"role": "user", "content": "用户问题"},
      ]

    返回: 完整的 prompt 字符串（包含 thinking token）
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG_DIR, trust_remote_code=True)

    # 使用内置的 chat_template（Jinja2 模板）
    # add_generation_prompt=True 会在末尾添加 assistant 的 thinking 前缀
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return prompt


def print_model_info(config):
    """打印模型关键信息"""
    print("=" * 60)
    print("Qwen3.8-27B 模型配置")
    print("=" * 60)

    tc = config.text_config
    print(f"  架构: {config.architectures[0]}")
    print(f"  模型类型: {config.model_type}")
    print(f"  隐藏层维度: {tc.hidden_size}")
    print(f"  中间层维度: {tc.intermediate_size}")
    print(f"  层数: {tc.num_hidden_layers}")
    print(f"  注意力头数: {tc.num_attention_heads}")
    print(f"  KV 头数 (GQA): {tc.num_key_value_heads}")
    print(f"  头维度: {tc.head_dim}")
    print(f"  词表大小: {tc.vocab_size}")
    print(f"  最大位置编码: {tc.max_position_embeddings:,}")
    print(f"  激活函数: {tc.hidden_act}")
    print(f"  RoPE theta: {tc.rope_parameters['rope_theta']:,}")
    print(f"  RoPE 类型: {tc.rope_parameters['rope_type']}")

    # 注意力层类型分布
    layer_types = getattr(tc, 'layer_types', [])
    full_count = sum(1 for t in layer_types if t == 'full_attention')
    linear_count = sum(1 for t in layer_types if t == 'linear_attention')
    print(f"  注意力类型: {full_count} 全注意力 + {linear_count} 线性注意力 (MLA)")
    print(f"  全注意力间隔: 每 {tc.full_attention_interval} 层一次")

    if hasattr(config, 'vision_config') and config.vision_config is not None:
        vc = config.vision_config
        print(f"\n视觉编码器:")
        print(f"  层数: {vc.depth}")
        print(f"  隐藏维度: {vc.hidden_size}")
        print(f"  注意力头数: {vc.num_heads}")
        print(f"  Patch 大小: {vc.patch_size}")
        print(f"  输出维度: {vc.out_hidden_size}")

    print(f"\n特殊 Token:")
    print(f"  BOS: {tc.bos_token_id} (<|endoftext|>)")
    print(f"  EOS: {tc.eos_token_id} (<|endoftext|>)")
    print(f"  PAD: {tc.pad_token_id}")
    print(f"  Image: {config.image_token_id} (<|image_pad|>)")
    print(f"  Video: {config.video_token_id} (<|video_pad|>)")


def demo_text():
    """纯文本推理示例"""
    print("\n" + "=" * 60)
    print("纯文本推理示例 (需要下载权重)")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
        {"role": "user", "content": "请用一句话介绍你自己。"},
    ]

    try:
        prompt = build_chat_prompt(messages)
        print(f"\n构建的 Prompt:\n{'-' * 40}")
        print(prompt[:500] + "...")
        print(f"\nPrompt 长度: {len(prompt)} 字符")

        # 实际推理需要权重文件，这里展示 tokenize 过程
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(CONFIG_DIR, trust_remote_code=True)
        tokens = tokenizer.encode(prompt)
        print(f"Token 数量: {len(tokens)}")
        print(f"前 20 个 token: {tokens[:20]}")

        # 解码验证
        decoded = tokenizer.decode(tokens[:20])
        print(f"解码验证: {decoded}")

    except Exception as e:
        print(f"错误: {e}")


def demo_vision():
    """多模态（图片）推理示例"""
    print("\n" + "=" * 60)
    print("多模态推理示例 (需要下载权重)")
    print("=" * 60)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "path/to/image.jpg"},
                {"type": "text", "text": "描述这张图片"},
            ],
        }
    ]

    try:
        prompt = build_chat_prompt(messages)
        print(f"\n构建的 Prompt:\n{'-' * 40}")
        # 找到视觉 token 的位置
        vision_start = prompt.find("<|vision_start|>")
        if vision_start >= 0:
            print(prompt[max(0, vision_start - 20):vision_start + 80])
    except Exception as e:
        print(f"错误: {e}")


def demo_tool_calling():
    """工具调用示例"""
    print("\n" + "=" * 60)
    print("工具调用示例 (需要下载权重)")
    print("=" * 60)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"}
                    },
                    "required": ["city"],
                },
            },
        }
    ]

    messages = [
        {"role": "system", "content": "你是一个助手，可以调用工具获取信息。"},
        {"role": "user", "content": "北京今天天气怎么样？"},
    ]

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(CONFIG_DIR, trust_remote_code=True)

        prompt = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        print(f"\n构建的 Prompt:\n{'-' * 40}")
        print(prompt[:800])
    except Exception as e:
        print(f"错误: {e}")


def demo_thinking():
    """思考模式示例"""
    print("\n" + "=" * 60)
    print("Thinking 思考模式示例")
    print("=" * 60)

    messages = [
        {"role": "user", "content": "请计算 123 * 456，并逐步展示计算过程。"},
    ]

    try:
        # 启用 thinking
        prompt_with = build_chat_prompt(messages, enable_thinking=True)
        print(f"\n启用 thinking:")
        print(f"  末尾: {prompt_with[-200:]}")

        # 关闭 thinking
        prompt_without = build_chat_prompt(messages, enable_thinking=False)
        print(f"\n关闭 thinking:")
        print(f"  末尾: {prompt_without[-200:]}")
    except Exception as e:
        print(f"错误: {e}")


def main():
    parser = argparse.ArgumentParser(description="Qwen3.8-27B 推理程序")
    parser.add_argument("--text", type=str, help="文本输入")
    parser.add_argument("--image", type=str, help="图片路径（可选）")
    parser.add_argument("--chat", action="store_true", help="交互式对话模式")
    args = parser.parse_args()

    config, tokenizer, _ = load_model()
    print_model_info(config)

    if args.text:
        if args.image:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": args.image},
                    {"type": "text", "text": args.text},
                ],
            }]
        else:
            messages = [{"role": "user", "content": args.text}]

        prompt = build_chat_prompt(messages)
        print(f"\n构建的 Prompt:\n{'-' * 40}")
        print(prompt[:500])

    elif args.chat:
        demo_text()
        demo_thinking()
        demo_vision()
        demo_tool_calling()
        print("\n提示: 实际推理需要下载模型权重文件 (safetensors)")
        print("      使用 AutoModelForConditionalGeneration.from_pretrained() 加载")
    else:
        # 默认展示所有示例
        demo_text()
        demo_thinking()
        demo_vision()
        demo_tool_calling()


if __name__ == "__main__":
    main()