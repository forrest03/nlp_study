#!/usr/bin/env python3
"""
法规 JSON → RAG chunks
基于 parse_doc.py 输出的结构化 JSON，按条级+上下文策略生成 chunk
"""
import json, sys, os, argparse
from pathlib import Path

# ── 配置 ──
SHORT_MERGE_THRESHOLD = 80     # 字符数，短于这个的条款合并到相邻chunk
LONG_SPLIT_THRESHOLD = 400     # 字符数，长于这个的拆分子chunk
MAX_CHUNK_CHARS = 500          # 单个chunk最大字符数


def build_chunks(doc: dict) -> list:
    """主 chunk 生成器"""
    chunks = []
    doc_title = doc.get("title", "")
    doc_subtitle = doc.get("subtitle", "")

    for ch in doc.get("chapters", []):
        ch_num = ch["chapter_number"]
        ch_title = ch["chapter_title"]
        ch_label = f"第{ch_num}章 {ch_title}"

        for sec in ch.get("sections", []):
            sec_num = sec.get("section_number")
            sec_title = sec.get("section_title")
            sec_label = f"第{sec_num}节 {sec_title}" if sec_num else ""

            articles = sec.get("articles", [])
            if not articles:
                continue

            # ── 第一遍：判断哪些条款需要合并/拆分 ──
            merged = merge_short_articles(articles, SHORT_MERGE_THRESHOLD)

            for item in merged:
                if isinstance(item, list):
                    if len(item) == 1:
                        chunk = build_single_chunk(item[0], ch_label, sec_label, doc_title)
                    else:
                        chunk = build_merged_chunk(item, ch_label, sec_label, doc_title)
                else:
                    art = item
                    art_len = len(art["content"])
                    if art_len > LONG_SPLIT_THRESHOLD and art.get("clauses"):
                        # 拆分长条款
                        sub_chunks = build_split_chunks(art, ch_label, sec_label, doc_title)
                        chunks.extend(sub_chunks)
                        continue
                    else:
                        chunk = build_single_chunk(art, ch_label, sec_label, doc_title)

                chunks.append(chunk)

    # ── 章级摘要 chunk ──
    for ch in doc.get("chapters", []):
        chunks.append(build_chapter_summary(ch, doc_title))

    return chunks


def merge_short_articles(articles: list, threshold: int) -> list:
    """极短相邻条款合并"""
    result = []
    buf = []
    for art in articles:
        if len(art["content"]) < threshold:
            buf.append(art)
        else:
            if buf:
                result.append(buf)
                buf = []
            result.append(art)
    if buf:
        result.append(buf)
    return result


def _ctx_header(ch_label: str, sec_label: str, doc_title: str) -> str:
    """生成 chunk 上下文前缀"""
    parts = [f"【文档】{doc_title}"]
    parts.append(f"【章节】{ch_label}")
    if sec_label:
        parts.append(f"【节】{sec_label}")
    return "\n".join(parts)


def build_single_chunk(art: dict, ch_label: str, sec_label: str, doc_title: str) -> dict:
    """单条款 chunk"""
    header = _ctx_header(ch_label, sec_label, doc_title)
    art_label = f"【条款】{art['article_number_cn']}"
    body = art["content"]

    text = f"{header}\n{art_label}\n{body}"

    return {
        "chunk_id": f"art_{art['article_number']}",
        "chunk_type": "article",
        "chapter": ch_label,
        "section": sec_label,
        "article_range": str(art["article_number"]),
        "text": text,
        "char_count": len(text),
        "article_numbers": [art["article_number"]],
        "metadata": {
            "doc_title": doc_title,
            "chapter_number": extract_ch_num(ch_label),
            "article_number": art["article_number"],
        }
    }


def build_merged_chunk(arts: list, ch_label: str, sec_label: str, doc_title: str) -> dict:
    """短条款合并 chunk"""
    header = _ctx_header(ch_label, sec_label, doc_title)
    nums = [a["article_number"] for a in arts]
    num_range = f"第{nums[0]}-{nums[-1]}条" if len(nums) > 1 else f"第{nums[0]}条"

    parts = [header, f"【条款】{num_range}"]
    for a in arts:
        parts.append(f"\n{a['article_number_cn']} {a['content']}")

    text = "\n".join(parts)

    return {
        "chunk_id": f"art_{nums[0]}_{nums[-1]}",
        "chunk_type": "merged",
        "chapter": ch_label,
        "section": sec_label,
        "article_range": f"{nums[0]}-{nums[-1]}",
        "text": text,
        "char_count": len(text),
        "article_numbers": nums,
        "metadata": {
            "doc_title": doc_title,
            "chapter_number": extract_ch_num(ch_label),
        }
    }


def build_split_chunks(art: dict, ch_label: str, sec_label: str, doc_title: str) -> list:
    """长条款按子项拆分"""
    clauses = art.get("clauses", [])
    if not clauses:
        return [build_single_chunk(art, ch_label, sec_label, doc_title)]

    chunks = []
    header = _ctx_header(ch_label, sec_label, doc_title)
    art_label = f"【条款】{art['article_number_cn']}"

    # 第一条拆出"引言"（条款正文中子项之前的部分）
    intro = split_intro(art["content"])
    if intro:
        text = f"{header}\n{art_label}\n{intro}"
        chunks.append({
            "chunk_id": f"art_{art['article_number']}_intro",
            "chunk_type": "sub_article",
            "chapter": ch_label,
            "section": sec_label,
            "article_range": str(art["article_number"]),
            "text": text,
            "char_count": len(text),
            "article_numbers": [art["article_number"]],
            "metadata": {
                "doc_title": doc_title,
                "chapter_number": extract_ch_num(ch_label),
                "article_number": art["article_number"],
                "clause_range": "intro",
            }
        })

    # 子项分组：每 N 个子项一组，确保每组不超过 MAX_CHUNK_CHARS
    group = []
    group_chars = 0
    for c in clauses:
        c_text = f"（{c['clause_number']}）{c['content']}"
        if group and group_chars + len(c_text) > MAX_CHUNK_CHARS:
            chunks.append(_make_clause_chunk(
                art, group, header, art_label, ch_label, sec_label, doc_title
            ))
            group = []
            group_chars = 0
        group.append(c)
        group_chars += len(c_text)

    if group:
        chunks.append(_make_clause_chunk(
            art, group, header, art_label, ch_label, sec_label, doc_title
        ))

    return chunks


def _make_clause_chunk(art, clauses, header, art_label, ch_label, sec_label, doc_title):
    """构建子项组 chunk"""
    c_range = f"（{clauses[0]['clause_number']}）-（{clauses[-1]['clause_number']}）"
    parts = [header, art_label, f"【子项】{c_range}"]
    for c in clauses:
        parts.append(f"（{c['clause_number']}）{c['content']}")
    text = "\n".join(parts)

    return {
        "chunk_id": f"art_{art['article_number']}_c{clauses[0]['clause_number']}_{clauses[-1]['clause_number']}",
        "chunk_type": "sub_article",
        "chapter": ch_label,
        "section": sec_label,
        "article_range": str(art["article_number"]),
        "text": text,
        "char_count": len(text),
        "article_numbers": [art["article_number"]],
        "metadata": {
            "doc_title": doc_title,
            "chapter_number": extract_ch_num(ch_label),
            "article_number": art["article_number"],
            "clause_range": f"{clauses[0]['clause_number']}-{clauses[-1]['clause_number']}",
        }
    }


def split_intro(content: str) -> str:
    """提取条款内容中子项编号之前的前导文本"""
    m = re.match(r'^(.+?)(?:[（(][一二三四五六七八九十]+[）)])', content, re.DOTALL)
    if m:
        intro = m.group(1).strip()
        # 只保留有意义的前导文本（多于15字）
        return intro if len(intro) > 15 else ""
    return ""


def build_chapter_summary(ch: dict, doc_title: str) -> dict:
    """生成章级摘要 chunk"""
    arts = []
    for sec in ch.get("sections", []):
        for a in sec.get("articles", []):
            arts.append(f"第{a['article_number']}条：{a['content'][:60]}...")

    ch_label = f"第{ch['chapter_number']}章 {ch['chapter_title']}"

    summary = f"【文档】{doc_title}\n【章节概要】{ch_label}\n"
    summary += f"本章共{len(arts)}条，涵盖以下内容：\n" + "\n".join(arts)

    return {
        "chunk_id": f"ch_summary_{ch['chapter_number']}",
        "chunk_type": "chapter_summary",
        "chapter": ch_label,
        "section": "",
        "article_range": "summary",
        "text": summary,
        "char_count": len(summary),
        "article_numbers": [],
        "metadata": {
            "doc_title": doc_title,
            "chapter_number": ch["chapter_number"],
        }
    }


def extract_ch_num(ch_label: str) -> int:
    """从'第N章 ...'提取数字"""
    import re as _re
    m = _re.match(r'第(\d+)章', ch_label)
    return int(m.group(1)) if m else 0


# ── CLI ──
import re

def main():
    ap = argparse.ArgumentParser(description="法规 JSON → RAG chunks")
    ap.add_argument("json_path", help="parse_doc.py 输出的 JSON 文件")
    ap.add_argument("-o", "--output", help="输出 JSONL 路径（默认同目录同名_chunks.jsonl）")
    ap.add_argument("--format", choices=["jsonl", "json"], default="jsonl",
                    help="输出格式：jsonl（每行一个chunk）或 json（数组）")
    args = ap.parse_args()

    with open(args.json_path, encoding="utf-8") as f:
        doc = json.load(f)

    chunks = build_chunks(doc)

    out = args.output or Path(args.json_path).with_suffix("").__str__() + "_chunks.jsonl"

    if args.format == "jsonl":
        with open(out, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    else:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

    # 统计
    types = {}
    for c in chunks:
        t = c["chunk_type"]
        types[t] = types.get(t, 0) + 1

    print(f"输入: {args.json_path}")
    print(f"输出: {out}")
    print(f"总计: {len(chunks)} chunks")
    for t, n in sorted(types.items()):
        print(f"  {t}: {n}")
    sizes = [c["char_count"] for c in chunks]
    print(f"字符: min={min(sizes)} avg={sum(sizes)//len(sizes)} max={max(sizes)}")


if __name__ == "__main__":
    main()