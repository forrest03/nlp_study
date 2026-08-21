"""
PDF 解析脚本：将原始年报 PDF 转换为结构化文本

教学重点（企业级 RAG 的真实挑战）：
  1. 数字 PDF vs 扫描件：处理方式完全不同
  2. 表格提取：年报里大量财务报表，直接按文字流提取会乱序
  3. 页眉/页脚噪声：每页都有公司名、页码，必须去除
  4. 章节识别：利用字体大小/加粗猜测标题层级
  5. 输出格式：保留元信息（页码、章节路径），供后续溯源用

依赖安装：
  pip install pdfplumber pymupdf pytesseract pillow
  # tesseract-ocr 需要单独安装并配置 PATH
  # Windows: https://github.com/UB-Mannheim/tesseract/wiki
"""

import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import pdfplumber          # 擅长表格提取
import fitz                # PyMuPDF，擅长文字+图片提取

# OCR 依赖可选（需要同时安装 pytesseract 包 + tesseract-ocr 二进制）
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent / "data"
RAW_DIR    = DATA_DIR / "raw_pdf"
PARSED_DIR = DATA_DIR / "parsed"
PARSED_DIR.mkdir(parents=True, exist_ok=True)

# 如果 tesseract 不在 PATH，手动指定（Windows 常见路径）
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class ParsedBlock:
    """
    一个解析块 = 年报里的一段连续内容（文字段落 or 表格）

    保留 page_num 和 section_path 非常重要——
    RAG 答案引用时能告诉用户"来自第38页，财务报告/资产负债表"
    """
    block_type:   str            # "text" | "table" | "title" | "code"
    content:      str            # 文字内容（代码块带 ``` 围栏）
    page_num:     int
    section_path: list[str]      # ["第三章 管理层讨论", "一、经营情况概述"]
    is_ocr:       bool = False   # 是否经过 OCR，质量可能有误
    code_lang:    Optional[str] = None  # 代码语言，如 python / go / json
    raw_table:    Optional[list] = field(default=None, repr=False)  # 原始表格数据


# ── 工具函数 ──────────────────────────────────────────────────────────────────

# 年报里常见的章节标题模式（粗略匹配，不求完美）
CHAPTER_PATTERNS = [
    re.compile(r"^第[一二三四五六七八九十百]+[章节]"),     # 第一章、第三节
    re.compile(r"^[一二三四五六七八九十]、"),               # 一、二、
    re.compile(r"^\d+\.\s"),                                # 1. 2.
    re.compile(r"^\d+\.\d+\s"),                             # 1.1 1.2（API 文档常见）
    re.compile(r"^(GET|POST|PUT|DELETE|PATCH)\s+/"),        # REST 接口行
    re.compile(r"^(接口|API|请求|响应|参数|错误码|附录)"),   # API 文档关键词标题
]

NOISE_PATTERNS = [
    re.compile(r"^.{1,40}年度报告\s*$"),    # 页眉：公司名+年度报告
    re.compile(r"^\d+\s*$"),                # 独立页码
    re.compile(r"^—\s*\d+\s*—$"),          # — 38 —
]

# ── 代码块识别 ────────────────────────────────────────────────────────────────

CODE_MARKER_RE = re.compile(r"^代码块\s*$")

# PDF 中语言标签 → markdown 围栏语言
LANG_LABEL_MAP = {
    "go": "go", "python": "python", "php": "php", "java": "java",
    "node.js": "javascript", "nodejs": "javascript", "javascript": "javascript",
    "ruby": "ruby", "lua": "lua", "c/c++": "cpp", "c#": "csharp",
    "c++": "cpp", "json": "json", "bash": "bash", "shell": "bash",
}

MONOSPACE_FONT_HINTS = (
    "sourcecode", "mono", "courier", "consolas", "menlo", "ocrb", "andale",
)

CODE_LINE_RE = re.compile(
    r"^\s*("
    r"import\b|from\b|#include|using\b|require\b|package\b|func\b|def\b|"
    r"function\b|class\b|const\b|let\b|var\b|public\b|private\b|return\b|"
    r"curl\b|GET\b|POST\b|PUT\b|DELETE\b|//|/\*|#|\"[\w]+\":|\{|\}|;"
    r")",
    re.IGNORECASE,
)

JSON_LINE_RE = re.compile(r'^\s*["{}\[\],]|^\s*"(Result|RequestId|Code|Message|Data)"')


def normalize_line(line: str) -> str:
    """去掉 PDF 零宽字符与多余空白。"""
    return line.replace("\u200b", "").replace("\ufeff", "").strip()


def is_monospace_font(font: str) -> bool:
    f = font.lower()
    return any(h in f for h in MONOSPACE_FONT_HINTS)


def line_monospace_ratio(spans: list[dict]) -> float:
    """一行中来自等宽字体的字符占比。"""
    if not spans:
        return 0.0
    mono = sum(len(s.get("text", "")) for s in spans if is_monospace_font(s.get("font", "")))
    total = sum(len(s.get("text", "")) for s in spans)
    return mono / total if total else 0.0


def detect_lang_label(line: str) -> Optional[str]:
    """识别单独一行的语言标签，如 Go / Python / node.js。"""
    clean = normalize_line(line).rstrip("​")
    if not clean or len(clean) > 20:
        return None
    key = clean.lower().replace(" ", "")
    if key in LANG_LABEL_MAP:
        return LANG_LABEL_MAP[key]
    # 去掉尾部零宽字符后再试
    for label, lang in LANG_LABEL_MAP.items():
        if key == label.replace(" ", ""):
            return lang
    return None


def is_code_line(line: str, spans: list[dict]) -> bool:
    """判断一行是否属于代码。"""
    clean = normalize_line(line)
    if not clean or CODE_MARKER_RE.match(clean):
        return False
    if re.match(r"^\d+$", clean):          # PDF 行号
        return True
    if line_monospace_ratio(spans) >= 0.6:
        return True
    if CODE_LINE_RE.search(clean):
        return True
    if JSON_LINE_RE.match(clean):
        return True
    return False


def strip_line_numbers(lines: list[str]) -> list[str]:
    """去掉 PDF 代码块中混入的行号（单独数字行或行首数字）。"""
    out = []
    for line in lines:
        clean = normalize_line(line)
        if re.match(r"^\d+$", clean):
            continue
        # "1 import (" → "import ("
        clean = re.sub(r"^(\d+)\s+(?=\S)", "", clean)
        if clean:
            out.append(clean)
    return out


def format_code_block(code: str, lang: Optional[str] = None) -> str:
    """将代码正文包成 markdown 围栏，便于 LLM 理解。"""
    body = strip_line_numbers(code.splitlines())
    body = "\n".join(body).strip()
    if not body:
        return ""
    fence_lang = lang or ""
    return f"```{fence_lang}\n{body}\n```"


def infer_json_lang(code: str) -> str:
    """JSON 示例自动识别。"""
    t = code.strip()
    if t.startswith("{") or t.startswith('"') or '"Result"' in t or '"RequestId"' in t:
        return "json"
    return ""


def table_looks_like_code(md: str) -> bool:
    """表格内容是否实为代码（PDF 误识别为表格）。"""
    if "代码块" not in md:
        return False
    hints = ("import", "function", "def ", "const ", "curl", '"Result"', "RequestId")
    return any(h in md for h in hints)


def extract_code_from_table_md(md: str) -> Optional[str]:
    """从误识别为表格的代码区域提取纯代码文本。"""
    lines = []
    for raw in md.splitlines():
        # 去掉 markdown 表格符号，保留单元格内容
        cells = [c.strip() for c in raw.strip("|").split("|")]
        text = " ".join(c for c in cells if c and c != "---")
        text = normalize_line(text)
        if not text or CODE_MARKER_RE.match(text):
            continue
        if re.match(r"^\d+$", text):
            continue
        text = re.sub(r"^(\d+)\s+(?=\S)", "", text)
        if text:
            lines.append(text)
    body = "\n".join(lines).strip()
    return body or None


def is_noise_line(line: str) -> bool:
    line = line.strip()
    if len(line) < 2:
        return True
    return any(p.match(line) for p in NOISE_PATTERNS)


def is_title_line(line: str, fontsize: Optional[float] = None, is_bold: bool = False) -> bool:
    """
    判断一行是否是标题。
    有字体信息时用字体大小，没有时用文字规律。
    """
    if fontsize and fontsize >= 14:
        return True
    if is_bold and len(line.strip()) < 50:
        return True
    return any(p.match(line.strip()) for p in CHAPTER_PATTERNS)


def table_to_markdown(table: list[list]) -> str:
    """把 pdfplumber 提取的表格转成 markdown 格式，方便 LLM 理解。"""
    if not table:
        return ""

    # 清洗单元格：None 变空字符串，去掉换行
    rows = []
    for row in table:
        cleaned = [str(cell or "").replace("\n", " ").strip() for cell in row]
        rows.append(cleaned)

    if not rows:
        return ""

    # 构建 markdown 表格
    header = rows[0]
    lines  = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows[1:]:
        # 对齐列数（有些 PDF 表格行列不整齐）
        while len(row) < len(header):
            row.append("")
        lines.append("| " + " | ".join(row[:len(header)]) + " |")

    return "\n".join(lines)


def detect_if_scanned(page: fitz.Page, text: str) -> bool:
    """
    启发式判断：文字极少但图片多 → 很可能是扫描页。
    年报中扫描件多见于附件（审计报告原件）。
    """
    if len(text.strip()) > 50:
        return False
    image_list = page.get_images(full=True)
    return len(image_list) > 0


def ocr_page(page: fitz.Page, dpi: int = 200) -> str:
    """对扫描页做 OCR（中文）。需要 pytesseract + tesseract-ocr 二进制。"""
    if not OCR_AVAILABLE:
        return "[扫描页，OCR 不可用（未安装 pytesseract/tesseract），内容跳过]"
    try:
        mat  = fitz.Matrix(dpi / 72, dpi / 72)
        clip = page.rect
        pix  = page.get_pixmap(matrix=mat, clip=clip)
        img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img, lang="chi_sim+eng")
        return text
    except Exception as e:
        logger.warning(f"  OCR 失败，跳过此页: {e}")
        return "[扫描页，OCR 失败，内容跳过]"


# ── 主解析逻辑 ────────────────────────────────────────────────────────────────

class AnnualReportParser:
    """
    年报 PDF 解析器。

    策略：
      - 用 pdfplumber 提取表格（它的表格算法更准）
      - 用 PyMuPDF (fitz) 提取带字体信息的文字（用于判断标题）
      - 对扫描页降级为 OCR
    """

    def __init__(self, pdf_path: Path, meta: dict = None):
        self.pdf_path = pdf_path
        self.meta     = meta or {}
        self.blocks: list[ParsedBlock] = []
        self._section_stack: list[str] = []

    def _flush_text(self, lines: list[str], page_num: int):
        if not lines:
            return
        content = "\n".join(normalize_line(l) for l in lines if normalize_line(l))
        if content:
            self.blocks.append(ParsedBlock(
                block_type="text",
                content=content,
                page_num=page_num,
                section_path=list(self._section_stack),
            ))

    def _flush_code(self, lines: list[str], page_num: int, lang: Optional[str]):
        if not lines:
            return
        raw = "\n".join(lines)
        lang = lang or infer_json_lang(raw) or None
        formatted = format_code_block(raw, lang)
        if formatted:
            self.blocks.append(ParsedBlock(
                block_type="code",
                content=formatted,
                page_num=page_num,
                section_path=list(self._section_stack),
                code_lang=lang,
            ))

    def _process_page_lines(self, page_dict: dict, page_num: int):
        """逐行解析，识别标题 / 代码块 / 普通段落。"""
        para_lines: list[str] = []
        code_lines: list[str] = []
        pending_lang: Optional[str] = None
        in_code = False

        def flush_para():
            nonlocal para_lines
            self._flush_text(para_lines, page_num)
            para_lines = []

        def flush_code():
            nonlocal code_lines, in_code, pending_lang
            if code_lines:
                self._flush_code(code_lines, page_num, pending_lang)
                pending_lang = None
            code_lines = []
            in_code = False

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                spans = line.get("spans", [])
                line_text = "".join(span["text"] for span in spans).strip()
                clean = normalize_line(line_text)

                if not clean or is_noise_line(clean):
                    continue

                fontsize = spans[0].get("size", 0) if spans else 0
                is_bold  = any("Bold" in span.get("font", "") for span in spans)

                # 语言标签（Go / Python …）
                lang = detect_lang_label(clean)
                if lang:
                    if in_code:
                        flush_code()
                    pending_lang = lang
                    continue

                # 代码块起始标记
                if CODE_MARKER_RE.match(clean):
                    flush_para()
                    flush_code()
                    in_code = True
                    continue

                # 标题
                if is_title_line(clean, fontsize, is_bold):
                    flush_para()
                    flush_code()
                    self._update_section(clean)
                    self.blocks.append(ParsedBlock(
                        block_type="title",
                        content=clean,
                        page_num=page_num,
                        section_path=list(self._section_stack),
                    ))
                    continue

                # 代码行
                if in_code or is_code_line(clean, spans):
                    if para_lines:
                        flush_para()
                    if not in_code:
                        in_code = True
                    code_lines.append(clean)
                    continue

                # 普通段落
                if in_code:
                    flush_code()
                para_lines.append(clean)

        flush_para()
        flush_code()

    def _postprocess_blocks(self) -> list[ParsedBlock]:
        """二次处理：拆分仍混在 text 里的代码，转换代码型表格。"""
        result: list[ParsedBlock] = []

        for blk in self.blocks:
            if blk.block_type == "table" and table_looks_like_code(blk.content):
                extracted = extract_code_from_table_md(blk.content)
                if extracted:
                    lang = infer_json_lang(extracted) or None
                    result.append(ParsedBlock(
                        block_type="code",
                        content=format_code_block(extracted, lang),
                        page_num=blk.page_num,
                        section_path=blk.section_path,
                        code_lang=lang,
                    ))
                    continue

            if blk.block_type != "text" or "代码块" not in blk.content:
                result.append(blk)
                continue

            # 拆分 text 中残留的「说明 + 代码块 + 代码」
            parts = re.split(r"(?=代码块\s*$)", blk.content, flags=re.MULTILINE)
            buf_text: list[str] = []
            pending_lang: Optional[str] = None

            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if part.startswith("代码块"):
                    code_body = part[len("代码块"):].strip()
                    if buf_text:
                        result.append(ParsedBlock(
                            block_type="text",
                            content="\n".join(buf_text),
                            page_num=blk.page_num,
                            section_path=blk.section_path,
                        ))
                        buf_text = []
                    if code_body:
                        # 尝试从开头拆语言标签
                        first_line, _, rest = code_body.partition("\n")
                        lang = detect_lang_label(first_line)
                        if lang:
                            pending_lang = lang
                            code_body = rest.strip() or code_body
                        lang = pending_lang or infer_json_lang(code_body) or None
                        formatted = format_code_block(code_body, lang)
                        if formatted:
                            result.append(ParsedBlock(
                                block_type="code",
                                content=formatted,
                                page_num=blk.page_num,
                                section_path=blk.section_path,
                                code_lang=lang,
                            ))
                        pending_lang = None
                else:
                    buf_text.append(part)

            if buf_text:
                result.append(ParsedBlock(
                    block_type="text",
                    content="\n\n".join(buf_text),
                    page_num=blk.page_num,
                    section_path=blk.section_path,
                ))

        return result

    def _update_section(self, title: str):
        """维护章节栈：根据缩进/编号层级推断层次。"""
        if re.match(r"^第[一二三四五六七八九十]+章", title):
            self._section_stack = [title]
        elif re.match(r"^第[一二三四五六七八九十]+节", title):
            self._section_stack = self._section_stack[:1] + [title]
        elif re.match(r"^[一二三四五六七八九十]、", title):
            self._section_stack = self._section_stack[:2] + [title]
        else:
            self._section_stack = self._section_stack[:3] + [title]

    def parse(self) -> list[ParsedBlock]:
        logger.info(f"开始解析: {self.pdf_path.name}")

        plumber_doc = pdfplumber.open(self.pdf_path)
        fitz_doc    = fitz.open(str(self.pdf_path))

        for page_num in range(len(fitz_doc)):
            fitz_page  = fitz_doc[page_num]
            plumb_page = plumber_doc.pages[page_num]

            raw_text   = fitz_page.get_text("text")
            is_scanned = detect_if_scanned(fitz_page, raw_text)

            if is_scanned:
                logger.debug(f"  第{page_num+1}页：检测到扫描件，启动 OCR")
                ocr_text = ocr_page(fitz_page)
                self.blocks.append(ParsedBlock(
                    block_type="text",
                    content=ocr_text,
                    page_num=page_num + 1,
                    section_path=list(self._section_stack),
                    is_ocr=True,
                ))
                continue

            # 提取表格（代码型表格跳过，改由行级解析 + 等宽字体识别）
            for table in plumb_page.extract_tables():
                if table:
                    md = table_to_markdown(table)
                    if md and not table_looks_like_code(md):
                        self.blocks.append(ParsedBlock(
                            block_type="table",
                            content=md,
                            page_num=page_num + 1,
                            section_path=list(self._section_stack),
                            raw_table=table,
                        ))

            page_dict = fitz_page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            self._process_page_lines(page_dict, page_num + 1)

        plumber_doc.close()
        fitz_doc.close()

        self.blocks = self._postprocess_blocks()
        code_n = sum(1 for b in self.blocks if b.block_type == "code")
        logger.info(f"  解析完成: {len(self.blocks)} 个块（其中代码块 {code_n} 个）")
        return self.blocks

    def save(self):
        """将解析结果保存为 JSON，保留所有元信息。"""
        stem     = self.pdf_path.stem
        out_path = PARSED_DIR / f"{stem}.json"

        output = {
            "meta":   self.meta,
            "source": str(self.pdf_path),
            "blocks": [asdict(b) for b in self.blocks],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"  已保存 → {out_path}")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def _resolve_pdf_path(item: dict) -> Path | None:
    """按 manifest 条目查找 PDF：优先 data/，其次 data/raw_pdf/。"""
    filename = item["filename"]
    for base in (DATA_DIR, RAW_DIR):
        candidate = base / filename
        if candidate.exists():
            return candidate
    return None


def main():
    manifest_path = DATA_DIR / "manifest.json"

    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = []
        for base in (DATA_DIR, RAW_DIR):
            for p in base.glob("*.pdf"):
                manifest.append({
                    "filename": p.name,
                    "doc_name": p.stem,
                    "version": "",
                    "doc_type": "api",
                })

    if not manifest:
        logger.error("没有找到任何 PDF，请将文档放入 data/ 并配置 manifest.json")
        return

    for item in manifest:
        pdf_path = _resolve_pdf_path(item)
        if pdf_path is None:
            logger.warning(f"文件不存在，跳过: {item['filename']}")
            continue

        parser = AnnualReportParser(pdf_path, meta=item)
        parser.parse()
        parser.save()

    logger.info(f"\n全部解析完成，结果在 {PARSED_DIR}")


if __name__ == "__main__":
    main()
