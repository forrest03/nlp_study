"""
PDF 解析脚本：将电子书 PDF 转换为结构化文本，供 RAG 系统使用

核心功能：
  1. 数字 PDF vs 扫描件：自动检测并处理
  2. 表格提取：将表格转为 markdown 格式
  3. 页眉/页脚噪声：自动去除常见噪声
  4. 章节识别：利用字体大小/加粗识别标题层级
  5. 结构化输出：保留元信息（页码、章节路径），供 RAG 溯源用

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
    pytesseract.pytesseract.tesseract_cmd = r'D:\software\Tesseract-OCR\tesseract.exe'
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR    = Path(__file__).parent.parent / "data" / "raw_pdf"
PARSED_DIR = Path(__file__).parent.parent / "data" / "parsed"
PARSED_DIR.mkdir(parents=True, exist_ok=True)

# 如果 tesseract 不在 PATH，手动指定（Windows 常见路径）
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class ParsedBlock:
    """
    一个解析块 = 电子书里的一段连续内容（文字段落 or 表格）

    保留 page_num 和 section_path 非常重要——
    RAG 答案引用时能告诉用户"来自第38页，第三章/第一节"
    """
    block_type:   str            # "text" | "table" | "title"
    content:      str            # 文字内容（表格转为 markdown）
    page_num:     int            # 页码
    section_path: list[str]      # ["第三章", "一、概述"]
    is_ocr:       bool = False   # 是否经过 OCR，质量可能有误
    raw_table:    Optional[list] = field(default=None, repr=False)  # 原始表格数据


# ── 工具函数 ──────────────────────────────────────────────────────────────────

# 电子书常见的章节标题模式
CHAPTER_PATTERNS = [
    re.compile(r"^第[一二三四五六七八九十百]+[章节篇]"),   # 第一章、第三节、第二篇
    re.compile(r"^第[一二三四五六七八九十百]+部分"),         # 第一部分
    re.compile(r"^[一二三四五六七八九十]+、"),               # 一、二、
    re.compile(r"^\d+\.\s"),                                # 1. 2.
    re.compile(r"^\d+\.\d+\s"),                            # 1.1 2.3
    re.compile(r"^\d+\.\d+\.\d+\s"),                      # 1.1.1 2.3.4
    re.compile(r"^Chapter\s+\d+"),                         # Chapter 1
    re.compile(r"^Section\s+\d+"),                         # Section 1
]

NOISE_PATTERNS = [
    re.compile(r"^.{1,60}年度报告\s*$"),     # 页眉：公司名+年度报告
    re.compile(r"^.{1,60}股份有限公司\s*$"), # 公司名
    re.compile(r"^\d+\s*$"),                 # 独立页码
    re.compile(r"^—\s*\d+\s*—$"),           # — 38 —
    re.compile(r"^\|\s*\d+\s*\|$"),          # | 38 |
    re.compile(r"^Page\s+\d+"),              # Page 38
]


def is_noise_line(line: str) -> bool:
    line = line.strip()
    if len(line) < 2:
        return True
    return any(p.match(line) for p in NOISE_PATTERNS)


def is_title_line(line: str, fontsize: Optional[float] = None, is_bold: bool = False) -> bool:
    line_stripped = line.strip()
    
    # 空行或过短的行不可能是标题
    if len(line_stripped) < 2:
        return False
    
    # 过长的行不太可能是标题
    if len(line_stripped) > 100:
        return False
    
    # 根据字体信息判断
    if fontsize:
        # 大字体且加粗 → 强烈暗示是标题
        if fontsize >= 16 and is_bold:
            return True
        # 中等字体且加粗且较短 → 可能是标题
        if fontsize >= 12 and is_bold and len(line_stripped) <= 40:
            return True
    
    # 根据文字模式判断（章节标题格式）
    return any(p.match(line_stripped) for p in CHAPTER_PATTERNS)


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

class PDFBookParser:
    """
    电子书 PDF 解析器，供 RAG 系统使用。

    策略：
      - 用 pdfplumber 提取表格（它的表格算法更准）
      - 用 PyMuPDF (fitz) 提取带字体信息的文字（用于判断标题）
      - 对扫描页降级为 OCR
      - 输出结构化数据列表，包含页码、章节等元信息
    """

    def __init__(self, pdf_path: Path, meta: dict = None):
        self.pdf_path = pdf_path
        self.meta     = meta or {}
        self.blocks: list[ParsedBlock] = []
        self._section_stack: list[str] = []

    def _update_section(self, title: str):
        """维护章节栈：根据编号层级推断层次。"""
        title_stripped = title.strip()
        if re.match(r"^第[一二三四五六七八九十百]+[章节篇]", title_stripped):
            self._section_stack = [title_stripped]           # 顶级章/篇
        elif re.match(r"^第[一二三四五六七八九十百]+部分", title_stripped):
            self._section_stack = [title_stripped]           # 顶级部分
        elif re.match(r"^Chapter\s+\d+", title_stripped):
            self._section_stack = [title_stripped]           # Chapter 级别
        elif re.match(r"^第[一二三四五六七八九十]+节", title_stripped):
            self._section_stack = self._section_stack[:1] + [title_stripped]  # 二级节
        elif re.match(r"^Section\s+\d+", title_stripped):
            self._section_stack = self._section_stack[:1] + [title_stripped]  # Section 级别
        elif re.match(r"^\d+\.\d+\.\d+", title_stripped):
            # 三级编号 1.1.1
            self._section_stack = self._section_stack[:2] + [title_stripped]
        elif re.match(r"^\d+\.\d+", title_stripped):
            # 二级编号 1.1
            self._section_stack = self._section_stack[:1] + [title_stripped]
        elif re.match(r"^[一二三四五六七八九十]+、", title_stripped):
            self._section_stack = self._section_stack[:2] + [title_stripped]  # 三级
        else:
            # 其他情况，作为最低级
            self._section_stack = self._section_stack[:3] + [title_stripped]

    def parse(self) -> list[ParsedBlock]:
        """
        解析 PDF 文件，返回结构化的数据块列表。

        Returns:
            list[ParsedBlock]: 包含页码、章节信息的结构化内容块列表
        """
        logger.info(f"开始解析: {self.pdf_path.name}")

        # 同时打开两个解析器
        plumber_doc = pdfplumber.open(self.pdf_path)
        fitz_doc    = fitz.open(str(self.pdf_path))

        for page_num in range(len(fitz_doc)):
            fitz_page   = fitz_doc[page_num]
            plumb_page  = plumber_doc.pages[page_num]

            # ── 1. 先用 PyMuPDF 获取带字体信息的文字 ──
            raw_text = fitz_page.get_text("text")
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

            # ── 2. 提取表格（用 pdfplumber）──
            for table in plumb_page.extract_tables():
                if table:
                    md = table_to_markdown(table)
                    if md:
                        self.blocks.append(ParsedBlock(
                            block_type="table",
                            content=md,
                            page_num=page_num + 1,
                            section_path=list(self._section_stack),
                            raw_table=table,
                        ))

            # ── 3. 提取文字（逐行处理）──
            page_dict = fitz_page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            current_para_lines = []

            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:   # 0=文字，1=图片
                    continue

                for line in block.get("lines", []):
                    line_text = "".join(
                        span["text"] for span in line.get("spans", [])
                    ).strip()

                    if not line_text or is_noise_line(line_text):
                        continue

                    # 判断是否标题
                    spans    = line.get("spans", [])
                    fontsize = spans[0].get("size", 0) if spans else 0
                    is_bold  = any("Bold" in span.get("font", "") for span in spans)

                    if is_title_line(line_text, fontsize, is_bold):
                        # 先把积累的段落存起来
                        if current_para_lines:
                            self.blocks.append(ParsedBlock(
                                block_type="text",
                                content="\n".join(current_para_lines),
                                page_num=page_num + 1,
                                section_path=list(self._section_stack),
                            ))
                            current_para_lines = []

                        self._update_section(line_text)
                        self.blocks.append(ParsedBlock(
                            block_type="title",
                            content=line_text,
                            page_num=page_num + 1,
                            section_path=list(self._section_stack),
                        ))
                    else:
                        current_para_lines.append(line_text)

            # 最后一段
            if current_para_lines:
                self.blocks.append(ParsedBlock(
                    block_type="text",
                    content="\n".join(current_para_lines),
                    page_num=page_num + 1,
                    section_path=list(self._section_stack),
                ))

        plumber_doc.close()
        fitz_doc.close()

        logger.info(f"  解析完成: {len(self.blocks)} 个块")
        return self.blocks

    def get_structured_data(self) -> list[dict]:
        """
        获取结构化数据列表，适合直接供 RAG 系统使用。

        Returns:
            list[dict]: 每个元素包含 block_type, content, page_num, section_path, is_ocr
        """
        return [asdict(block) for block in self.blocks]

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

def parse_pdf_files(input_dir: Optional[Path] = None, output_dir: Optional[Path] = None) -> list[dict]:
    """
    批量解析指定目录下的所有 PDF 文件。

    Args:
        input_dir: PDF 文件所在目录，默认为 RAW_DIR
        output_dir: 解析结果保存目录，默认为 PARSED_DIR

    Returns:
        list[dict]: 所有解析出的结构化数据块列表
    """
    if input_dir is None:
        input_dir = RAW_DIR
    if output_dir is None:
        output_dir = PARSED_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    # 扫描目录下所有 PDF 文件
    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        logger.error(f"没有找到任何 PDF 文件: {input_dir}")
        return []

    all_blocks = []

    for pdf_path in pdf_files:
        logger.info(f"处理文件: {pdf_path.name}")

        meta = {
            "filename": pdf_path.name,
            "source_path": str(pdf_path),
        }

        parser = PDFBookParser(pdf_path, meta=meta)
        blocks = parser.parse()
        parser.save()

        # 添加文件元信息
        for block in blocks:
            block_dict = asdict(block)
            block_dict["source_file"] = pdf_path.name
            all_blocks.append(block_dict)

    logger.info(f"\n全部解析完成，共处理 {len(pdf_files)} 个文件，生成 {len(all_blocks)} 个数据块")
    logger.info(f"结果保存至: {output_dir}")

    return all_blocks


def main():
    parse_pdf_files()


if __name__ == "__main__":
    main()