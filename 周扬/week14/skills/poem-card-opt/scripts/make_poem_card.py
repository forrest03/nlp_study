"""古诗卡 HTML 生成器: python make_poem_card.py <data.json> [-o out.html]"""
import argparse
import html
import json
import re
from pathlib import Path

# 与旧版输出逐字节一致（CSS/结构勿改）
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
    esc = html.escape
    return "\n        ".join(f'<div class="line">{esc(x)}</div>' for x in lines)


def render_notes(notes):
    if not notes:
        return '<div class="note">（本诗无重点字词注释）</div>'
    esc = html.escape
    return "\n        ".join(
        f'<div class="note"><span class="word">{esc(n.get("word", "") or "？")}</span>'
        f'：{esc(n.get("meaning", "") or "（待补充）")}</div>'
        for n in notes
    )


def render_quiz(quiz):
    if not quiz:
        return '<li><div class="q">（暂无自测题）</div></li>'
    esc = html.escape
    return "\n        ".join(
        f'<li><div class="q">{i}. {esc(q.get("q", "") or "（待补充题目）")}</div>'
        f'<details><summary>看答案</summary>'
        f'<div class="a">{esc(q.get("a", "") or "（待补充答案）")}</div></details></li>'
        for i, q in enumerate(quiz, 1)
    )


def build_html(data):
    g = data.get("grade_hint", "")
    grade_html = f'<span class="grade-tag">适合 {html.escape(g)}</span>' if g else ""
    return TEMPLATE.format(
        title=html.escape(data.get("title", "无题")),
        author=html.escape(data.get("author", "佚名")),
        dynasty=html.escape(data.get("dynasty", "")),
        grade_html=grade_html,
        lines_html=render_lines(data.get("lines", [])),
        notes_html=render_notes(data.get("notes", [])),
        translation=html.escape(data.get("translation", "") or "（待补充译文）"),
        appreciation=html.escape(data.get("appreciation", "") or "（待补充赏析）"),
        quiz_html=render_quiz(data.get("quiz", [])),
    )


def main():
    ap = argparse.ArgumentParser(description="生成古诗学习卡 HTML")
    ap.add_argument("data")
    ap.add_argument("-o", "--output")
    args = ap.parse_args()
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if args.output:
        out = Path(args.output)
    else:
        title = re.sub(r"[《》〈〉\s]", "", data.get("title", "")) or "poem"
        out = Path.cwd() / f"{title}.html"
    out.write_text(build_html(data), encoding="utf-8")
    print(f"已生成: {out}")


if __name__ == "__main__":
    main()
