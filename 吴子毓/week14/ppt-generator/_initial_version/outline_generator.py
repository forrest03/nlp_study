"""大纲生成器：调 LLM 根据主题生成 PPT 大纲（JSON）。
依赖：openai。环境变量：DASHSCOPE_API_KEY。
用法：python outline_generator.py "主题" [页数]
"""
import json, os, sys, re
from openai import OpenAI

MODEL = "deepseek-v4-flash-0731"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

SYSTEM_PROMPT = """你是 PPT 大纲设计专家。根据用户给的主题，生成结构化 PPT 大纲。
输出严格的 JSON 数组，每个元素是一页：
[
  {"section": "章节标识", "title": "页标题", "subtitle": "页副标题", "points": ["要点1", "要点2", "要点3"]}
]
规则：
1. 第一页是封面（section 填机构/作者，title 是主标题，subtitle 是副标题，points 放3个核心数字/亮点）
2. 后续每页一个主题，title 简洁（≤15字），subtitle 补充说明，points 3个要点（每个≤25字）
3. 要点具体、有数据/对比，不要空话
4. 只输出 JSON，不要解释文字"""

USER_TEMPLATE = """主题：{topic}
期望页数：{pages} 页（含封面）
请生成大纲。"""


def generate_outline(topic, pages=5):
    """调 LLM 生成大纲，返回 (outline, usage)。usage 用于统计 token。"""
    client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url=BASE_URL)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(topic=topic, pages=pages)},
        ],
        temperature=0.7,
    )
    content = resp.choices[0].message.content.strip()
    # 提取 JSON（LLM 可能带 ```json 标记）
    m = re.search(r'\[.*\]', content, re.S)
    if not m:
        raise ValueError(f"LLM 输出未找到 JSON: {content[:200]}")
    outline = json.loads(m.group(0))
    return outline, resp.usage


if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else "自进化 Agent：从对话失败中自动演化的 Skill 机制"
    pages = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    outline, usage = generate_outline(topic, pages)
    print(json.dumps(outline, ensure_ascii=False, indent=2))
    print(f"\n--- token 用量 ---", file=sys.stderr)
    print(f"prompt_tokens: {usage.prompt_tokens}", file=sys.stderr)
    print(f"completion_tokens: {usage.completion_tokens}", file=sys.stderr)
    print(f"total_tokens: {usage.total_tokens}", file=sys.stderr)
