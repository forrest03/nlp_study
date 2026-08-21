"""
古诗学习卡生成器
=================
为一首古诗生成一张静态 HTML 学习卡片，包含：
  - 标题、作者、朝代、适合学段
  - 原文逐句排版（楷体、居中）
  - 重点字词注释
  - 白话译文、赏析
  - 背诵自测填空（答案折叠在 <details> 里，点击展开）

用法:
    python make_poem_card.py <data.json>                  # 输出到当前目录 <title>.html
    python make_poem_card.py <data.json> -o output.html   # 指定输出路径

JSON 数据格式:
{
  "title": "静夜思",
  "author": "李白",
  "dynasty": "唐",
  "grade_hint": "二年级",
  "lines": ["床前明月光，", "疑是地上霜。", "举头望明月，", "低头思故乡。"],
  "notes": [
    {"word": "疑", "meaning": "好像，仿佛"}
  ],
  "translation": "床前洒满了明亮的月光，好像铺上了一层白白的霜。……",
  "appreciation": "这首诗用眼前最常见的月光，写出了出门在外的人对家乡的想念。……",
  "quiz": [
    {"q": "床前明月＿", "a": "光"},
    {"q": "疑是地上＿", "a": "霜"}
  ]
}
"""
import argparse
import json
import html
import re
from pathlib import Path


TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} - 古诗卡</title>
<style>
  :root {{
    --bg: #f7f2ea;
    --card: #fffdf9;
    --ink: #2b2620;
    --muted: #8a8172;
    --accent: #9a3412;
    --accent-soft: #fbeee2;
    --border: #e8dfd0;
    --shadow: 0 10px 30px rgba(60, 45, 20, 0.10);
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                 "Microsoft YaHei", Roboto, sans-serif;
    background: var(--bg);
    color: var(--ink);
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
  }}
  .card {{
    width: 100%;
    max-width: 720px;
    background: var(--card);
    border-radius: 20px;
    box-shadow: var(--shadow);
    overflow: hidden;
  }}
  .header {{
    padding: 32px 36px 24px;
    background: linear-gradient(135deg, #7c2d12 0%, var(--accent) 100%);
    color: #fff;
  }}
  .title {{
    margin: 0;
    font-family: "Kaiti SC", "STKaiti", "KaiTi", "楷体", serif;
    font-size: 44px;
    font-weight: 700;
  }}
  .meta {{
    margin-top: 8px;
    font-size: 17px;
    opacity: 0.92;
  }}
  .grade-tag {{
    display: inline-block;
    margin-left: 10px;
    padding: 2px 10px;
    background: rgba(255, 255, 255, 0.18);
    border: 1px solid rgba(255, 255, 255, 0.4);
    border-radius: 999px;
    font-size: 13px;
  }}
  .body {{ padding: 28px 36px 36px; }}
  .poem {{
    margin: 4px 0 0;
    padding: 22px 16px;
    background: var(--accent-soft);
    border-radius: 12px;
    text-align: center;
  }}
  .poem .line {{
    font-family: "Kaiti SC", "STKaiti", "KaiTi", "楷体", serif;
    font-size: 26px;
    line-height: 1.9;
    letter-spacing: 6px;
  }}
  h2 {{
    margin: 28px 0 12px;
    font-size: 15px;
    font-weight: 600;
    color: var(--muted);
    letter-spacing: 2px;
  }}
  .notes .note {{
    padding: 8px 0;
    font-size: 16px;
    line-height: 1.6;
    border-bottom: 1px dashed var(--border);
  }}
  .notes .note:last-child {{ border-bottom: none; }}
  .notes .word {{
    color: var(--accent);
    font-weight: 600;
    margin-right: 4px;
  }}
  .prose {{
    padding: 14px 16px;
    background: #faf6ee;
    border-left: 4px solid var(--accent);
    border-radius: 8px;
    font-size: 16px;
    line-height: 1.8;
  }}
  .quiz {{ list-style: none; padding: 0; margin: 0; }}
  .quiz li {{
    padding: 12px 16px;
    margin-bottom: 10px;
    background: #fafafa;
    border: 1px solid var(--border);
    border-radius: 10px;
  }}
  .quiz .q {{
    font-family: "Kaiti SC", "STKaiti", "KaiTi", "楷体", serif;
    font-size: 17px;
    line-height: 1.6;
    letter-spacing: 2px;
  }}
  .quiz details {{ margin-top: 8px; }}
  .quiz summary {{
    cursor: pointer;
    font-size: 13px;
    color: var(--accent);
  }}
  .quiz .a {{
    margin-top: 6px;
    font-size: 16px;
    color: var(--accent);
    font-weight: 600;
  }}
  .footer {{
    margin-top: 28px;
    padding-top: 16px;
    border-top: 1px dashed var(--border);
    font-size: 12px;
    color: var(--muted);
    text-align: center;
  }}
</style>
</head>
<body>
  <div class="card">
    <div class="header">
      <h1 class="title">{title}</h1>
      <div class="meta">{dynasty} · {author}{grade_html}</div>
    </div>
    <div class="body">
      <div class="poem">
        {lines_html}
      </div>

      <h2>注释</h2>
      <div class="notes">
        {notes_html}
      </div>

      <h2>译文</h2>
      <div class="prose">{translation}</div>

      <h2>赏析</h2>
      <div class="prose">{appreciation}</div>

      <h2>背诵自测</h2>
      <ul class="quiz">
        {quiz_html}
      </ul>

      <div class="footer">古诗卡 · 一诗一卡，读背结合</div>
    </div>
  </div>
</body>
</html>
"""


def render_lines(lines):
    return "\n        ".join(
        f'<div class="line">{html.escape(line)}</div>' for line in lines
    )


def render_notes(notes):
    if not notes:
        return '<div class="note">（本诗无重点字词注释）</div>'
    items = []
    for note in notes:
        word = html.escape(note.get("word", "") or "？")
        meaning = html.escape(note.get("meaning", "") or "（待补充）")
        items.append(
            f'<div class="note"><span class="word">{word}</span>：{meaning}</div>'
        )
    return "\n        ".join(items)


def render_quiz(quiz):
    if not quiz:
        return '<li><div class="q">（暂无自测题）</div></li>'
    items = []
    for idx, item in enumerate(quiz, start=1):
        q = html.escape(item.get("q", "") or "（待补充题目）")
        a = html.escape(item.get("a", "") or "（待补充答案）")
        items.append(
            f'<li><div class="q">{idx}. {q}</div>'
            f"<details><summary>看答案</summary>"
            f'<div class="a">{a}</div></details></li>'
        )
    return "\n        ".join(items)


def render_grade_tag(grade_hint):
    if not grade_hint:
        return ""
    return f'<span class="grade-tag">适合 {html.escape(grade_hint)}</span>'


def build_html(data):
    return TEMPLATE.format(
        title=html.escape(data.get("title", "无题")),
        author=html.escape(data.get("author", "佚名")),
        dynasty=html.escape(data.get("dynasty", "")),
        grade_html=render_grade_tag(data.get("grade_hint", "")),
        lines_html=render_lines(data.get("lines", [])),
        notes_html=render_notes(data.get("notes", [])),
        translation=html.escape(data.get("translation", "") or "（待补充译文）"),
        appreciation=html.escape(data.get("appreciation", "") or "（待补充赏析）"),
        quiz_html=render_quiz(data.get("quiz", [])),
    )


def sanitize_filename(title):
    """去掉书名号、标点、空白，只留下能做文件名的字符。"""
    cleaned = re.sub(r"[《》〈〉\s]", "", title)
    return cleaned or "poem"


def main():
    parser = argparse.ArgumentParser(description="生成古诗学习卡 HTML")
    parser.add_argument("data", help="JSON 数据文件路径")
    parser.add_argument("-o", "--output",
                        help="输出 HTML 路径（默认当前目录下 <title>.html）")
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path.cwd() / f"{sanitize_filename(data.get('title', ''))}.html"
    out_path.write_text(build_html(data), encoding="utf-8")
    print(f"已生成: {out_path}")


if __name__ == "__main__":
    main()
