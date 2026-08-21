"""
法规 .doc 文件结构化解析器
输入：.doc 文件路径
输出：{filename}.json — 结构化的条款 JSON
"""
import subprocess, sys, os, re, json, argparse
from pathlib import Path

def doc_to_text(doc_path: str) -> str:
    """用 macOS textutil 将 .doc 转为纯文本"""
    doc_path = os.path.abspath(doc_path)
    tmp = f"/tmp/parse_doc_{os.getpid()}.txt"
    subprocess.run(["textutil", "-convert", "txt", "-output", tmp, doc_path], check=True)
    with open(tmp, "r", encoding="utf-8") as f:
        text = f.read()
    os.remove(tmp)
    return text

def clean_text(text: str) -> str:
    """清理杂项：PAGE域代码、HYPERLINK域、多余空行"""
    text = re.sub(r'PAGE\s*\\\*\s*MERGEFORMAT\s*-\s*\d+\s*-', '', text)
    text = re.sub(r'HYPERLINK\s+"[^"]*"', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def parse_structure(text: str) -> dict:
    """解析法规文档为结构化 JSON"""
    doc = {
        "title": "",
        "subtitle": "",
        "chapters": [],
        "attachments": []
    }

    # ── 取标题和副标题 ──
    title_pat = re.compile(r'^(.+?)(?:管理办法|办法|规定|规则|条例|指引)\s*$', re.MULTILINE)
    m = title_pat.search(text[:300])
    if m:
        doc["title"] = m.group(0).strip()

    subtitle_pat = re.compile(r'[（(](\d{4}年\d{1,2}月\d{1,2}日.*?)[）)]')
    m = subtitle_pat.search(text)
    if m:
        doc["subtitle"] = m.group(1)

    # ── 分割章节 ──
    chapter_pat = re.compile(
        r'第([一二三四五六七八九十]+)章\s+(.+?)\s*\n',
        re.MULTILINE
    )
    chapter_starts = [(m.start(), m.group(0).strip(), m.group(1), m.group(2))
                      for m in chapter_pat.finditer(text)]

    if chapter_starts:
        for i, (start, full_line, ch_num, ch_title) in enumerate(chapter_starts):
            ch_num_arabic = cn_to_arabic(ch_num)
            content_start = start + len(full_line)
            if i + 1 < len(chapter_starts):
                content_end = chapter_starts[i + 1][0]
            else:
                attach_match = re.search(r'\n附件[：:]\s*\n', text[content_start:])
                if attach_match:
                    content_end = content_start + attach_match.start()
                    attach_text = text[content_start + attach_match.end():].strip()
                    doc["attachments"] = parse_attachments(attach_text)
                else:
                    content_end = len(text)
            chapter_body = text[content_start:content_end].strip()
            doc["chapters"].append(_build_chapter(ch_num_arabic, ch_num, ch_title.strip(), chapter_body))
    else:
        # ── 无"第X章"标记 → 全文作为扁平一章，"总则"类标题提取为章名 ──
        # 从标题行之后开始找正文
        body_start = 0
        if doc["title"]:
            m = re.search(re.escape(doc["title"]), text)
            if m:
                body_start = m.end()
        # 跳过分隔空行
        body = text[body_start:].strip()
        # 尝试按扁平标题（如"总  则"、"申请与许可"）切章
        flat_chapters = split_by_flat_headers(body)
        if flat_chapters and len(flat_chapters) > 1:
            for fc in flat_chapters:
                h_title, h_body = fc
                doc["chapters"].append(_build_chapter(
                    len(doc["chapters"]) + 1,
                    cn_to_arabic_str(len(doc["chapters"]) + 1),
                    h_title,
                    h_body
                ))
        else:
            # 全篇无章无节 → 直接解析全部条款
            doc["chapters"].append(_build_chapter(1, "一", "", body))

    # ── 附件检测（无章节时全文搜索） ──
    if not doc["attachments"]:
        attach_match = re.search(r'\n附件[：:]\s*\n', text)
        if attach_match:
            attach_text = text[attach_match.end():].strip()
            doc["attachments"] = parse_attachments(attach_text)

    return doc

def split_by_flat_headers(text: str) -> list:
    """将无'第X章'标记的正文按扁平标题切割，如'总  则'、'申请与许可'"""
    # 匹配独占一行的中文标题（2-10字，不含"第"开头，在行首，后跟空行或条款）
    header_pat = re.compile(
        r'^([\u4e00-\u9fff\uff00-\uffef]{2,10})\s*\n\s*(?=第[一二三四五六七八九十百零\d]+条)',
        re.MULTILINE
    )
    matches = list(header_pat.finditer(text))
    if not matches:
        return []

    result = []
    for i, m in enumerate(matches):
        h_title = m.group(1).strip()
        h_start = m.end()
        if i + 1 < len(matches):
            h_end = matches[i + 1].start()
        else:
            h_end = len(text)
        h_body = text[h_start:h_end].strip()
        result.append((h_title, h_body))
    return result

def _build_chapter(ch_num_arabic, ch_num_cn, ch_title, chapter_body):
    """构建 chapter 对象"""
    chapter = {
        "chapter_number": ch_num_arabic,
        "chapter_number_cn": f"第{ch_num_cn}章" if ch_num_cn else "",
        "chapter_title": ch_title,
        "sections": []
    }

    section_pat = re.compile(
        r'第([一二三四五六七八九十]+)节\s+(.+?)\s*\n',
        re.MULTILINE
    )
    section_matches = [(m.start(), m.group(0).strip(), m.group(1), m.group(2))
                       for m in section_pat.finditer(chapter_body)]

    if section_matches:
        for j, (s_start, s_full, s_num, s_title) in enumerate(section_matches):
            s_content_start = s_start + len(s_full)
            if j + 1 < len(section_matches):
                s_content_end = section_matches[j + 1][0]
            else:
                s_content_end = len(chapter_body)
            section_body = chapter_body[s_content_start:s_content_end].strip()
            section = {
                "section_number": cn_to_arabic(s_num),
                "section_number_cn": f"第{s_num}节",
                "section_title": s_title.strip(),
                "articles": parse_articles(section_body, ch_num_arabic)
            }
            chapter["sections"].append(section)
    else:
        chapter["sections"].append({
            "section_number": None,
            "section_title": None,
            "articles": parse_articles(chapter_body, ch_num_arabic)
        })

    return chapter


def parse_articles(body: str, ch_num: int) -> list:
    """解析条款，提取条号、内容、子项"""
    articles = []
    # 容忍段首缩进空白（全角空格等）
    article_pat = re.compile(
        r'第([一二三四五六七八九十百零\d]+)条\s+(.+?)(?=\n\s*第[一二三四五六七八九十百零\d]+条\s+|\Z)',
        re.DOTALL
    )
    for m in article_pat.finditer(body):
        art_num_raw = m.group(1)
        art_num = cn_to_arabic(art_num_raw) if any(c in '一二三四五六七八九十百零' for c in art_num_raw) else int(art_num_raw)
        art_body = m.group(2).strip()
        article = {
            "article_number": art_num,
            "article_number_cn": f"第{art_num_raw}条",
            "content": art_body,
            "clauses": parse_clauses(art_body)
        }
        articles.append(article)
    return articles

def parse_clauses(article_body: str) -> list:
    """解析条款内部的（一）（二）...子项"""
    clauses = []
    clause_pat = re.compile(
        r'[（(]([一二三四五六七八九十]+)[）)]\s*(.+?)(?=[（(][一二三四五六七八九十]+[）)]|\Z)',
        re.DOTALL
    )
    for m in clause_pat.finditer(article_body):
        c_num = m.group(1)
        c_body = m.group(2).strip()
        clauses.append({
            "clause_number": cn_to_arabic(c_num),
            "clause_number_cn": f"（{c_num}）",
            "content": c_body
        })
    return clauses

def parse_attachments(text: str) -> list:
    """解析附件：标题一行，URL 在下一行（HYPERLINK清理后残余的原始URL）"""
    attachments = []
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    seen = set()
    for i, line in enumerate(lines):
        if line.startswith('http'):
            title = lines[i-1] if i > 0 else ""
            url = line
            if url not in seen:
                seen.add(url)
                attachments.append({"title": title, "url": url})
    return attachments

def cn_to_arabic(cn: str) -> int:
    """中文数字转阿拉伯数字"""
    map_cn = {'零':0, '一':1, '二':2, '三':3, '四':4, '五':5,
              '六':6, '七':7, '八':8, '九':9, '十':10, '百':100}
    try:
        return int(cn)
    except ValueError:
        pass
    result = 0
    if '十' in cn:
        parts = cn.split('十')
        if parts[0] == '':
            result = 10
        else:
            result = map_cn.get(parts[0], 0) * 10
        if len(parts) > 1 and parts[1]:
            result += map_cn.get(parts[1], 0)
    else:
        result = map_cn.get(cn, 0)
    return result

def cn_to_arabic_str(n: int) -> str:
    """阿拉伯数字转中文数字字符串（1-99）"""
    nums = ['零','一','二','三','四','五','六','七','八','九','十']
    if n <= 10:
        return nums[n]
    if n < 20:
        return '十' + (nums[n-10] if n > 10 else '')
    tens = n // 10
    ones = n % 10
    return nums[tens] + '十' + (nums[ones] if ones else '')

def main():
    parser = argparse.ArgumentParser(description="解析法规 .doc 文件为结构化 JSON")
    parser.add_argument("doc_path", help=".doc 文件路径")
    parser.add_argument("-o", "--output", help="输出 JSON 路径（默认同目录同名.json）")
    args = parser.parse_args()

    doc_path = args.doc_path
    if not os.path.exists(doc_path):
        print(f"错误：文件不存在 {doc_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or (Path(doc_path).with_suffix('.json'))

    print(f"[1/3] 转换 .doc → 文本 ...")
    text = doc_to_text(doc_path)
    text = clean_text(text)

    print(f"[2/3] 解析结构 ...")
    doc = parse_structure(text)

    article_count = sum(
        len(section.get("articles", []))
        for ch in doc.get("chapters", [])
        for section in ch.get("sections", [])
    )
    clause_count = sum(
        len(article.get("clauses", []))
        for ch in doc.get("chapters", [])
        for section in ch.get("sections", [])
        for article in section.get("articles", [])
    )

    print(f"[3/3] 写入 JSON → {output_path}")
    print(f"      标题: {doc['title']}")
    print(f"      章节: {len(doc['chapters'])}章")
    print(f"      条款: {article_count}条")
    print(f"      子项: {clause_count}项")
    print(f"      附件: {len(doc['attachments'])}个")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    print(f"      完成 ✓")

if __name__ == "__main__":
    main()