# -*- coding: utf-8 -*-
"""
共享的多格式资料读取工具。
支持文本、常见图片，以及 PDF 的文本提取/页面渲染。
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import List, Tuple

TEXT_EXTENSIONS = {".txt", ".md"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
PDF_EXTENSIONS = {".pdf"}


def read_text_file(file_path: str | Path) -> str:
    return Path(file_path).read_text(encoding="utf-8")


def encode_file_base64(file_path: str | Path) -> str:
    return base64.b64encode(Path(file_path).read_bytes()).decode("utf-8")


def guess_image_mime(file_path: str | Path) -> str:
    suffix = Path(file_path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(suffix, "image/jpeg")


def build_image_message_parts(file_path: str | Path, prompt_text: str) -> List[dict]:
    mime = guess_image_mime(file_path)
    encoded = encode_file_base64(file_path)
    return [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
    ]


def truncate_text(text: str, max_chars: int = 20000) -> str:
    normalized = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars] + "\n\n[内容过长，已截断]"


def extract_text_from_pdf(file_path: str | Path, max_pages: int = 10) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    file_path = Path(file_path)

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        texts = []
        for page in reader.pages[:max_pages]:
            page_text = (page.extract_text() or "").strip()
            if page_text:
                texts.append(page_text)
        if texts:
            return "\n\n".join(texts), warnings
    except ImportError:
        warnings.append("未安装 pypdf，无法走 PDF 文本提取分支。")
    except Exception as exc:
        warnings.append(f"pypdf 提取失败：{exc}")

    try:
        import fitz

        doc = fitz.open(str(file_path))
        try:
            texts = []
            for page_index in range(min(len(doc), max_pages)):
                page_text = doc.load_page(page_index).get_text("text").strip()
                if page_text:
                    texts.append(page_text)
            if texts:
                return "\n\n".join(texts), warnings
        finally:
            doc.close()
    except ImportError:
        warnings.append("未安装 PyMuPDF，无法走 PDF 文本兜底分支。")
    except Exception as exc:
        warnings.append(f"PyMuPDF 文本提取失败：{exc}")

    return "", warnings


def render_pdf_to_message_parts(
    file_path: str | Path,
    prompt_text: str,
    max_pages: int = 4,
    zoom: float = 2.0,
) -> Tuple[List[dict], List[str]]:
    warnings: List[str] = []

    try:
        import fitz
    except ImportError:
        warnings.append("未安装 PyMuPDF，无法把扫描版 PDF 渲染成图片。")
        return [], warnings

    file_path = Path(file_path)
    doc = fitz.open(str(file_path))
    try:
        page_count = min(len(doc), max_pages)
        parts: List[dict] = [
            {
                "type": "text",
                "text": f"{prompt_text}\n请按页阅读这份 PDF，目前附上前 {page_count} 页。",
            }
        ]
        matrix = fitz.Matrix(zoom, zoom)
        for page_index in range(page_count):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            encoded = base64.b64encode(pix.tobytes("png")).decode("utf-8")
            parts.append({"type": "text", "text": f"第 {page_index + 1} 页："})
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                }
            )
        return parts, warnings
    finally:
        doc.close()


def read_pdf_as_text_or_images(
    file_path: str | Path,
    prompt_text: str,
    max_text_chars: int = 20000,
) -> Tuple[str | List[dict], List[str]]:
    text, warnings = extract_text_from_pdf(file_path)
    if text.strip():
        return f"{prompt_text}\n\n{text[:max_text_chars]}", warnings

    parts, render_warnings = render_pdf_to_message_parts(file_path, prompt_text)
    warnings.extend(render_warnings)
    if parts:
        return parts, warnings

    raise RuntimeError(
        "无法从 PDF 提取文本，也无法渲染页面。请安装 pypdf/PyMuPDF，或者先将 PDF 转成图片后再入库。"
    )
