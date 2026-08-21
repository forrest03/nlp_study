"""DeepSeek 的极简 OpenAI 兼容客户端。"""

import time

try:
    from config import DEEPSEEK_API_KEY
except ImportError:
    DEEPSEEK_API_KEY = ""

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


DEEPSEEK_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
client = None


def get_client():
    """延迟创建客户端，启动程序时不立即请求网络。"""
    global client
    if OpenAI is None:
        raise RuntimeError("缺少 openai 依赖，请执行：pip install -r requirements.txt")
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("config.py 中没有设置 DEEPSEEK_API_KEY")
    if client is None:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_URL)
    return client


def llm_chat(system, user, stop=None):
    """调用一次大模型。失败时重试三次。"""
    for attempt in range(3):
        try:
            response = get_client().chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=900,
                stop=stop,
            )
            return response.choices[0].message.content
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
