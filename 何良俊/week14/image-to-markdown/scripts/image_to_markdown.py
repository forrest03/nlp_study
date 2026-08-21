#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片识别 → Markdown（本地 Ollama 视觉模型 minicpm-v，非 OCR）

将一张或多张图片发送到本地 Ollama 的视觉理解模型（默认 minicpm-v），
识别图片内容并输出 Markdown 文本到 stdout，供会话继续使用。

依赖: Python 3.8+、Pillow（pip install Pillow）
环境: 本地 Ollama 已启动且已拉取模型（ollama pull minicpm-v），默认地址 http://127.0.0.1:11434

用法:
    python image_to_markdown.py <图片路径|URL> [更多图片...]
    python image_to_markdown.py shot.png --prompt "把所有表格提取为 markdown 表格"

环境变量（可选）:
    OLLAMA_BASE_URL   默认 http://127.0.0.1:11434
    OLLAMA_MODEL      默认 minicpm-v
    OLLAMA_API_STYLE  openai（/v1/chat/completions）| native（/api/chat），默认 openai
"""

import argparse
import base64
import io
import json
import os
import sys
import urllib.error
import urllib.request

# Windows 控制台可能用 GBK 编码，统一改成 UTF-8，避免打印中文报错
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

try:
    from PIL import Image
except ImportError:
    sys.stderr.write(
        "缺少 Pillow 依赖，请先安装: pip install Pillow\n"
        "或使用包管理器安装（如 conda install pillow）\n"
    )
    sys.exit(1)

MAX_EDGE = 1280  # 发送前图片最长边上限（px）
DEFAULT_PROMPT = "识别图片里所有信息，使用 markdown 输出全部内容，并保持排版的一致"
DEFAULT_MODEL = "minicpm-v"


def load_image(source: str) -> Image.Image:
    """从本地路径或 URL 读取图片。"""
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=60) as resp:
            return Image.open(io.BytesIO(resp.read()))
    with open(source, "rb") as f:
        return Image.open(io.BytesIO(f.read()))


def resize_to_max_edge(img: Image.Image, max_edge: int = MAX_EDGE) -> Image.Image:
    """等比缩放，使最长边不超过 max_edge；已达标则原样返回。"""
    w, h = img.size
    longest = max(w, h)
    if longest <= max_edge:
        return img
    ratio = max_edge / longest
    new_size = (max(1, round(w * ratio)), max(1, round(h * ratio)))
    return img.resize(new_size, Image.LANCZOS)


def to_jpeg_data_uri(img: Image.Image) -> str:
    """统一转成 JPEG 的 base64 data URI（RGBA/调色板图先铺白底转 RGB）。"""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _looks_truncated(text: str) -> bool:
    """启发式：本地模型输出过短 → 疑似没读出内容，值得云端兜底重跑。"""
    return len((text or "").strip()) < 30


def call_dashscope_vision(data_uris, prompt: str, model: str) -> str:
    """调用云端 qwen-vl-max（DashScope OpenAI 兼容端点）。"""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        sys.stderr.write(
            "--backend dashscope 需要 DASHSCOPE_API_KEY 环境变量"
            "（setx DASHSCOPE_API_KEY sk-xxx 后重启终端）\n"
        )
        sys.exit(2)
    content = [{"type": "image_url", "image_url": {"url": u}} for u in data_uris]
    content.append({"type": "text", "text": prompt})
    payload = {"model": model, "messages": [{"role": "user", "content": content}],
               "temperature": 0.2}
    req = urllib.request.Request(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        sys.stderr.write(f"DashScope HTTP {e.code}: {e.read().decode('utf-8', 'replace')}\n")
        sys.exit(1)
    except urllib.error.URLError as e:
        sys.stderr.write(f"网络错误: {e.reason}\n")
        sys.exit(1)
    try:
        raw = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        sys.stderr.write(f"意外的 API 响应: {json.dumps(body, ensure_ascii=False)[:500]}\n")
        sys.exit(1)
    if isinstance(raw, str):
        return raw
    return "".join(p.get("text", "") for p in raw if isinstance(p, dict) and p.get("type") == "text")


def call_vision(data_uris, prompt: str, backend: str, model: str,
                api_style: str, base_url: str, dashscope_model: str) -> str:
    """统一视觉入口：ollama / dashscope / auto（本地→过短则云端兜底）。"""
    if backend == "dashscope":
        return call_dashscope_vision(data_uris, prompt, dashscope_model)
    text = call_ollama_vision(data_uris, prompt, model, api_style, base_url)
    if backend == "auto" and _looks_truncated(text):
        if os.environ.get("DASHSCOPE_API_KEY"):
            sys.stderr.write("[提示] 本地输出过短，疑似漏识，用云端 qwen-vl-max 兜底重跑...\n")
            return call_dashscope_vision(data_uris, prompt, dashscope_model)
    return text


def call_ollama_vision(data_uris, prompt: str, model: str,
                       api_style: str, base_url: str) -> str:
    """调用本地 Ollama 视觉模型，返回模型输出的 Markdown 文本。"""
    if api_style == "native":
        images = [u.split(",", 1)[1] for u in data_uris]  # 去掉 data: 前缀
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "images": images,
            "stream": False,
            "options": {"num_predict": 4096},  # 加大输出上限，避免长内容截断
        }
        url = f"{base_url}/api/chat"
    else:  # openai 兼容
        content = [{"type": "image_url", "image_url": {"url": u}} for u in data_uris]
        content.append({"type": "text", "text": prompt})
        payload = {"model": model, "messages": [{"role": "user", "content": content}],
                   "stream": False, "max_tokens": 4096}
        url = f"{base_url}/v1/chat/completions"

    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        sys.stderr.write(f"Ollama HTTP {e.code}: {e.read().decode('utf-8', 'replace')}\n")
        sys.exit(1)
    except urllib.error.URLError as e:
        sys.stderr.write(
            f"无法连接本地 Ollama（{base_url}）: {e.reason}\n"
            "请确认 Ollama 已启动，且模型已拉取: ollama pull minicpm-v\n"
        )
        sys.exit(1)
    try:
        if api_style == "native":
            return body["message"]["content"]
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        sys.stderr.write(f"意外的 Ollama 响应: {json.dumps(body, ensure_ascii=False)[:500]}\n")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="图片识别（本地 Ollama 视觉模型）并输出 Markdown 文本，非 OCR"
    )
    parser.add_argument("images", nargs="+", help="图片路径或 URL，可多个")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="识别提示词")
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL),
        help="Ollama 视觉模型名，默认 minicpm-v",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        help="Ollama 服务地址，默认 http://127.0.0.1:11434",
    )
    parser.add_argument(
        "--api-style",
        choices=["openai", "native"],
        default=os.environ.get("OLLAMA_API_STYLE", "openai"),
        help="接口风格: openai（/v1/chat/completions）或 native（/api/chat），默认 openai",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "ollama", "dashscope"],
        default="auto",
        help="视觉后端: auto（本地，过短则云端兜底）/ ollama（纯本地）/ dashscope（纯云端），默认 auto",
    )
    parser.add_argument(
        "--dashscope-model", default="qwen-vl-max",
        help="云端视觉模型，默认 qwen-vl-max",
    )
    parser.add_argument(
        "--max-edge", type=int, default=MAX_EDGE, help="发送前图片最长边上限（像素），默认 1280"
    )
    args = parser.parse_args()
    base_url = args.ollama_url.rstrip("/")

    data_uris = []
    for src in args.images:
        is_url = src.startswith(("http://", "https://"))
        if not is_url and not os.path.isfile(src):
            sys.stderr.write(f"图片不存在: {src}\n")
            sys.exit(1)
        try:
            img = load_image(src)
            img = resize_to_max_edge(img, args.max_edge)
            data_uris.append(to_jpeg_data_uri(img))
        except Exception as e:
            sys.stderr.write(f"无法处理图片 {src}: {e}\n")
            sys.exit(1)
        print(
            f"[已处理] {src} -> {img.size[0]}x{img.size[1]} (最长边≤{args.max_edge})",
            file=sys.stderr,
        )

    markdown = call_vision(data_uris, args.prompt, args.backend, args.model,
                           args.api_style, base_url, args.dashscope_model)

    if len(data_uris) > 1:
        print(f"## 识别结果（共 {len(data_uris)} 张图片）\n")
    print(markdown)


if __name__ == "__main__":
    main()
