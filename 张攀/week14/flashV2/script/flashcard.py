# -*- coding: utf-8 -*-
"""单词闪卡渲染模块 —— 供 harness 通过 importlib 动态加载调用。

JSON 数据由 harness 所调用的 LLM 生成后传入本模块，本模块只负责渲染 HTML。

导出（唯一公共接口）:
    run(word="", data=None, json_path=None, **kwargs) -> dict
        word:       英文单词（默认输出文件名 data/<word>.html）
        data:       单词数据 dict（符合 SKILL.md 的 JSON Schema），与 json_path 二选一
        json_path:  单词 JSON 文件路径，harness 传参时自动读取

返回: {"success": bool, "html_path": ..., "error": ...}
"""
import html
import json
import os
import re
import sys

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(SKILL_DIR, "data")

REQUIRED_KEYS = ("translation", "synonyms", "near_synonyms", "examples")


# ---------- 渲染 HTML ----------

def _esc(s):
    return html.escape(str(s or ""), quote=True)


def _highlight(text, word):
    """例句中高亮目标词（先转义再高亮，避免 XSS）。"""
    safe = _esc(text)
    if not word:
        return safe
    pattern = re.compile(r"(" + re.escape(word) + r")", re.IGNORECASE)
    return pattern.sub(r"<mark>\1</mark>", safe)


def _chips(arr, cls=""):
    if not arr:
        return '<div class="empty">暂无</div>'
    items = "".join(f'<span class="chip {cls}">{_esc(w)}</span>' for w in arr)
    return f'<div class="chips">{items}</div>'


def _render_definitions(defs):
    if not defs:
        return '<div class="empty">暂无释义</div>'
    items = []
    for d in defs:
        pos = _esc(d.get("partOfSpeech", "word"))
        en = _esc(d.get("definition", ""))
        zh = d.get("definition_zh") or d.get("definitionZh") or ""
        zh_html = f'<div class="zh">{_esc(zh)}</div>' if zh else ""
        items.append(
            f'<div class="def-item"><span class="pos">{pos}</span>'
            f'<span class="en">{en}</span>{zh_html}</div>'
        )
    return "".join(items)


def _render_examples(examples, word):
    if not examples:
        return '<div class="empty">暂无例句</div>'
    items = []
    for e in examples:
        en = _highlight(e.get("en", ""), word)
        zh = e.get("zh", "")
        zh_html = f'<div class="zh">{_esc(zh)}</div>' if zh else ""
        items.append(f'<div class="example"><div class="en">{en}</div>{zh_html}</div>')
    return "".join(items)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title} · 单词闪卡</title>
  <style>
    :root {{
      --bg1: #000000;
      --bg2: #1a1a2e;
      --card: #ffffff;
      --ink: #1f2330;
      --muted: #6b7280;
      --accent: #6a5af9;
      --accent-soft: #eee9ff;
      --line: #eceaf3;
      --pos: #ff7043;
      --shadow: 0 20px 50px rgba(0, 0, 0, 0.6);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; min-height: 100vh;
      font-family: -apple-system, "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif;
      background: linear-gradient(135deg, var(--bg1), var(--bg2));
      display: flex; align-items: center; justify-content: center;
      padding: 40px 16px; color: var(--ink);
    }}
    .stage {{ perspective: 1600px; width: 100%; max-width: 640px; }}
    .card {{
      position: relative; width: 100%; min-height: 560px;
      transform-style: preserve-3d;
      transition: transform .7s cubic-bezier(.4,.2,.2,1);
      cursor: pointer;
    }}
    .card.flipped {{ transform: rotateY(180deg); }}
    .face {{
      position: absolute; inset: 0; backface-visibility: hidden;
      background: var(--card); border-radius: 20px; box-shadow: var(--shadow);
      padding: 30px 30px 26px; overflow-y: auto;
    }}
    .face.back {{ transform: rotateY(180deg); }}
    .front .word-row {{ display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }}
    .front .word {{ font-size: 44px; font-weight: 800; letter-spacing: .5px; }}
    .front .phonetic {{ color: var(--muted); font-size: 17px; }}
    .audio-btn {{
      border: none; background: var(--accent-soft); color: var(--accent);
      width: 38px; height: 38px; border-radius: 50%; cursor: pointer; font-size: 17px;
      display: inline-flex; align-items: center; justify-content: center;
    }}
    .audio-btn:hover {{ background: var(--accent); color: #fff; }}
    .translation-box {{
      margin: 26px 0 0; padding: 18px 20px; border-radius: 14px;
      background: linear-gradient(135deg, var(--accent-soft), #f3efff);
      border: 1px solid var(--line);
    }}
    .translation-box .label {{ font-size: 12px; color: var(--muted); margin-bottom: 6px; letter-spacing: 1px; }}
    .translation-box .value {{ font-size: 24px; font-weight: 700; color: var(--accent); }}
    .quick-tags {{ margin-top: 18px; display: flex; flex-wrap: wrap; gap: 8px; }}
    .quick-tags .chip {{ font-size: 13px; padding: 5px 12px; border-radius: 20px; background: #f4f3f8; color: var(--ink); }}
    .flip-hint {{ position: absolute; bottom: 18px; left: 0; right: 0; text-align: center; color: var(--muted); font-size: 13px; }}
    .back .section {{ margin-bottom: 20px; }}
    .back .section:last-child {{ margin-bottom: 0; }}
    .back h3 {{
      margin: 0 0 10px; font-size: 14px; color: var(--accent);
      letter-spacing: 1px; display: flex; align-items: center; gap: 8px;
    }}
    .back h3::before {{ content: ""; width: 4px; height: 14px; background: var(--accent); border-radius: 2px; }}
    .def-item {{ margin-bottom: 10px; }}
    .def-item .pos {{
      display: inline-block; font-size: 11px; color: #fff; background: var(--pos);
      padding: 2px 8px; border-radius: 10px; margin-right: 8px; vertical-align: middle;
    }}
    .def-item .en {{ font-size: 14px; }}
    .def-item .zh {{ font-size: 13px; color: var(--muted); margin-top: 2px; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .chips .chip {{ font-size: 13px; padding: 5px 12px; border-radius: 20px; background: var(--accent-soft); color: var(--accent); }}
    .chips .chip.near {{ background: #fff3e8; color: #ff7043; }}
    .empty {{ color: var(--muted); font-size: 13px; font-style: italic; }}
    .example {{ margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px dashed var(--line); }}
    .example:last-child {{ border-bottom: none; margin-bottom: 0; padding-bottom: 0; }}
    .example .en {{ font-size: 14px; font-weight: 600; }}
    .example .en mark {{ background: #fff59d; padding: 0 3px; border-radius: 3px; }}
    .example .zh {{ font-size: 13px; color: var(--muted); margin-top: 3px; }}
  </style>
</head>
<body>
  <div class="stage">
    <div class="card" id="card">
      <div class="face front">
        <div class="word-row">
          <span class="word">{word}</span>
          {phonetic}
          {audio_btn}
        </div>
        <div class="translation-box">
          <div class="label">中文释义</div>
          <div class="value">{translation}</div>
        </div>
        <div class="quick-tags">{quick_syn}</div>
        <div class="flip-hint">点击卡片翻转查看详情 →</div>
      </div>
      <div class="face back">
        <div class="section">
          <h3>详细释义</h3>
          {definitions}
        </div>
        <div class="section">
          <h3>同义词</h3>
          {synonyms}
        </div>
        <div class="section">
          <h3>近义词</h3>
          {near_synonyms}
        </div>
        <div class="section">
          <h3>例句</h3>
          {examples}
        </div>
      </div>
    </div>
  </div>
  <script>
    document.getElementById('card').addEventListener('click', function() {{
      this.classList.toggle('flipped');
    }});
    {audio_script}
  </script>
</body>
</html>"""


def _generate_html(data):
    """根据 JSON 数据 dict 返回 HTML 字符串。"""
    word = data.get("word", "")
    phonetic = data.get("phonetic", "")
    audio = data.get("audio", "")
    translation = data.get("translation", "—") or "—"

    phonetic_html = f'<span class="phonetic">{_esc(phonetic)}</span>' if phonetic else ""
    audio_btn_html = ""
    audio_script = ""
    if audio:
        audio_btn_html = '<button class="audio-btn" id="audioBtn" title="发音">🔊</button>'
        audio_script = (
            "document.getElementById('audioBtn').addEventListener('click',function(e){"
            "e.stopPropagation();new Audio({audio_url}).play().catch(function(){});"
            "});"
        ).format(audio_url=json.dumps(audio))

    synonyms = data.get("synonyms", []) or []
    near_synonyms = data.get("near_synonyms") or data.get("nearSynonyms", []) or []
    examples = data.get("examples", []) or []
    definitions = data.get("definitions", []) or []

    return HTML_TEMPLATE.format(
        title=_esc(word),
        word=_esc(word),
        phonetic=phonetic_html,
        audio_btn=audio_btn_html,
        translation=_esc(translation),
        quick_syn=_chips(synonyms[:5]),
        definitions=_render_definitions(definitions),
        synonyms=_chips(synonyms[:5]),
        near_synonyms=_chips(near_synonyms[:5], "near"),
        examples=_render_examples(examples[:3], word),
        audio_script=audio_script,
    )


def _write_html(word, data):
    """渲染并写入 data/<word>.html，返回路径；失败返回 None。"""
    try:
        html_content = _generate_html(data)
    except Exception as e:  # noqa: BLE001
        print(f"[render error] {word}: {e}")
        return None
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, f"{word}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return out_path


# ---------- 唯一公共接口 ----------

def run(word="", data=None, json_path=None, **kwargs):
    """根据传入的 JSON 数据渲染单词闪卡 HTML。返回 dict。"""
    if data is None:
        if json_path:
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:  # noqa: BLE001
                return {"success": False, "error": f"读取 JSON 文件失败: {e}"}
        else:
            return {"success": False, "error": "缺少 JSON 数据：请通过 data（dict）或 json_path（文件）参数传入。"}

    if not isinstance(data, dict):
        return {"success": False, "error": "data 参数必须是 dict（符合 SKILL.md 的 JSON Schema）"}

    word = (word or data.get("word") or "").strip().lower()
    if not word:
        return {"success": False, "error": "缺少 word 参数，且 data 中无 word 字段。"}

    missing = [k for k in REQUIRED_KEYS if not data.get(k)]
    if missing:
        return {"success": False, "error": f"JSON 缺少字段: {missing}，与 SKILL.md 的 Schema 不符"}

    html_path = _write_html(word, data)
    if not html_path:
        return {"success": False, "error": f"渲染 data/{word}.html 失败"}
    return {"success": True, "word": word, "html_path": html_path}


if __name__ == "__main__":
    _word = sys.argv[1] if len(sys.argv) > 1 else input("单词: ").strip()
    result = run(word=_word, data={"word": _word, "translation": "示例", "synonyms": [], "near_synonyms": [], "examples": []})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result.get("success") else 1)
