"""大纲生成器：调 LLM 根据主题生成 PPT 大纲（JSON）。
依赖：openai。环境变量：DASHSCOPE_API_KEY。
用法：python outline_generator.py "主题" [页数]
"""
import json, os, sys, re
from openai import OpenAI

MODEL = "deepseek-v4-flash-0731"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

SYSTEM_PROMPT = """PPT大纲生成。仅输出JSON数组，每页：
{"section":"章节","title":"标题≤15字","subtitle":"副标题",
 "layout":"default|two_column|comparison|stack",
 "panels":[{"title":"栏标题","subtitle":"副标","body":"多行\\n","fill":"色键","border":"色键"}],
 "left":{"title":"左","rows":[{"label":"指标","value":"数值"}]},
 "right":{"title":"右","rows":[{"label":"指标","value":"数值"}]},
 "layers":[{"label":"层名","level":5,"color":"色键"}],
 "points":["要点1","要点2","要点3"]}
规则：1.首页封面(section填作者,points放3个核心数字)2.要点具体含数据不空话
3.左右分栏对比用 layout=two_column + panels(2个)
4.同一对象两版本对比用 layout=comparison + left/right
5.层栈/漏斗用 layout=stack + layers(高层=宽);其余用默认 points
6.fill/border/color 取色键: light/accent/gray/title/bg  7.只输出JSON无解释"""

USER_TEMPLATE = "主题:{topic} 页数:{pages}(含封面)"


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
