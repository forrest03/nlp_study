"""
llm.py — LLM 客户端工厂

支持三个 provider（与 week11 保持一致）：
  - deepseek   ：DEEPSEEK_API_KEY
  - dashscope  ：DASHSCOPE_API_KEY
  - siliconflow：SILICONFLOW_API_KEY（兼容 SCY_API_KEY）

返回 (client, model_name) 元组，给 Runner 用。
"""

import os
import sys

from openai import OpenAI


PROVIDERS = {
    "deepseek": {
        "api_key":   lambda: os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url":  "https://api.deepseek.com",
        "model":     "deepseek-chat",
        "env_hint":  "DEEPSEEK_API_KEY",
    },
    "dashscope": {
        "api_key":   lambda: os.environ.get("DASHSCOPE_API_KEY", ""),
        "base_url":  "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model":     "qwen-plus",
        "env_hint":  "DASHSCOPE_API_KEY",
    },
    "siliconflow": {
        "api_key":   lambda: os.environ.get("SILICONFLOW_API_KEY", "") or os.environ.get("SCY_API_KEY", ""),
        "base_url":  "https://api.siliconflow.cn/v1",
        "model":     "Qwen/Qwen2.5-14B-Instruct",
        "env_hint":  "SILICONFLOW_API_KEY（兼容 SCY_API_KEY）",
    },
}


def build_client(provider: str) -> tuple[OpenAI, str]:
    """根据 provider 名构造 OpenAI 兼容 client。"""
    if provider not in PROVIDERS:
        raise ValueError(f"未知 provider：{provider}，可选：{list(PROVIDERS.keys())}")
    cfg = PROVIDERS[provider]
    api_key = cfg["api_key"]()
    if not api_key:
        print(f"错误：未设置 {cfg['env_hint']}", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
    return client, cfg["model"]


def preflight_check(client: OpenAI, model: str, provider: str) -> None:
    """
    启动时用极小请求探一下 API 联通性，401 等认证错误给出明确指引。
    通过：静默返回；不通过：打印诊断信息并 sys.exit(1)。
    """
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            timeout=10,
        )
    except Exception as e:
        msg = str(e)
        if "401" in msg or "Authentication" in msg or "api key" in msg.lower():
            print("\n❌ API key 无效或已过期（401 Authentication Fails）\n", file=sys.stderr)
            print(f"   当前 provider: {provider}", file=sys.stderr)
            print("   可能的原因：\n", file=sys.stderr)
            if provider == "deepseek":
                print("   1) 去 https://platform.deepseek.com/api_keys 重新生成 key", file=sys.stderr)
                print("   2) 或换 provider：--provider dashscope / siliconflow", file=sys.stderr)
            elif provider == "dashscope":
                print("   1) 去 https://dashscope.console.aliyun.com/apiKey 重新生成", file=sys.stderr)
                print("   2) 或换 provider：--provider deepseek / siliconflow", file=sys.stderr)
            elif provider == "siliconflow":
                print("   1) 去 https://cloud.siliconflow.cn/account/ak 重新生成", file=sys.stderr)
                print("   2) 或换 provider：--provider deepseek / dashscope", file=sys.stderr)
            sys.exit(1)
        else:
            # 其它错误（网络、超时等）原样抛
            print(f"\n❌ {provider} API 联通失败：{type(e).__name__}: {e}\n", file=sys.stderr)
            sys.exit(1)
