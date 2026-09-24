"""Extraction and summarisation helpers for issuer offering documents."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Any

import httpx

from news_agent_config import settings

MAX_TEXT_CHARS = 60_000
MAX_OCR_PAGES = 25


@dataclass(frozen=True)
class ExtractedDocument:
    text: str
    pages: int | None
    used_ocr: bool
    warnings: list[str]


def _clean_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


def extract_document_text(file_name: str, content: bytes) -> ExtractedDocument:
    """Extract DOCX/PDF text and OCR scanned PDF pages when needed.

    Optional OCR dependencies are kept inside the PDF fallback so a normal
    text-PDF/DOCX upload remains usable in minimal installations.
    """
    suffix = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    if suffix == "docx":
        try:
            from docx import Document

            document = Document(BytesIO(content))
            parts = [paragraph.text for paragraph in document.paragraphs]
            for table in document.tables:
                parts.extend(" | ".join(cell.text for cell in row.cells) for row in table.rows)
            text = _clean_text("\n".join(parts))
            return ExtractedDocument(text=text[:MAX_TEXT_CHARS], pages=None, used_ocr=False, warnings=[])
        except Exception as exc:
            raise ValueError(f"Не удалось прочитать Word-документ: {exc}") from exc

    if suffix != "pdf":
        raise ValueError("Поддерживаются только файлы PDF и DOCX.")

    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        pages = len(reader.pages)
        text = _clean_text("\n".join((page.extract_text() or "") for page in reader.pages))
    except Exception as exc:
        raise ValueError(f"Не удалось прочитать PDF: {exc}") from exc

    warnings: list[str] = []
    used_ocr = False
    if len(text) < 80:
        try:
            import fitz
            import pytesseract
            from PIL import Image

            pdf = fitz.open(stream=content, filetype="pdf")
            ocr_parts = []
            for page_index in range(min(len(pdf), MAX_OCR_PAGES)):
                page = pdf[page_index]
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image = Image.open(BytesIO(pixmap.tobytes("png")))
                ocr_parts.append(pytesseract.image_to_string(image, lang="rus+eng"))
            text = _clean_text("\n".join(ocr_parts))
            used_ocr = True
            if pages > MAX_OCR_PAGES:
                warnings.append(f"OCR выполнен только для первых {MAX_OCR_PAGES} из {pages} страниц.")
        except Exception as exc:
            warnings.append(
                "В PDF не найден текст, а OCR недоступен. Установите Tesseract с языками rus+eng "
                f"и зависимости OCR. Техническая причина: {exc}"
            )
    if not text:
        warnings.append("Текст документа не извлечён; результат требует ручной проверки.")
    return ExtractedDocument(text=text[:MAX_TEXT_CHARS], pages=pages, used_ocr=used_ocr, warnings=warnings)


def _fallback_summary(text: str, extraction: ExtractedDocument) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 20]
    return {
        "summary": "Автоматическое LLM-резюме недоступно. Ниже приведён извлечённый фрагмент для первичного просмотра.",
        "key_data": [],
        "attention_points": [
            "Проверьте условия выпуска, порядок выплат, досрочное погашение и ковенанты в первичном документе.",
            "Сверьте все цифры с оригиналом: извлечение текста или OCR может содержать ошибки.",
        ],
        "document_coverage": f"Извлечено {len(text):,} символов".replace(",", " "),
        "source_excerpt": "\n".join(lines[:12])[:3_000] or "Текст не удалось извлечь.",
        "extraction": asdict(extraction),
        "llm_used": False,
    }


def analyse_emission_document(file_name: str, content: bytes) -> dict[str, Any]:
    extraction = extract_document_text(file_name, content)
    if not extraction.text or not settings.openai_api_key:
        return _fallback_summary(extraction.text, extraction)

    prompt = {
        "file_name": file_name,
        "task": (
            "Проанализируй эмиссионный документ на русском. Верни ТОЛЬКО JSON с полями "
            "summary (строка), key_data (массив объектов {name, value, note}), "
            "attention_points (массив строк), document_coverage (строка). "
            "Выделяй только данные, прямо содержащиеся в документе. Не придумывай значения. "
            "Проверь эмитента, инструмент/серию/ISIN, объём, номинал, валюту, срок, ставку/купон, "
            "доходность, график выплат, амортизацию, оферту/досрочное погашение, обеспечение, ковенанты, "
            "рейтинги, цели привлечения и факторы риска. В attention_points укажи вопросы для инвестора."
        ),
        "document_text": extraction.text,
    }
    payload = {
        "model": settings.openai_model,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "Ты внимательный аналитик долгового рынка. Отвечай на русском и строго в JSON."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    }
    try:
        response = httpx.post(
            f"{settings.openai_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {settings.openai_api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=settings.request_timeout_seconds,
        )
        response.raise_for_status()
        result = json.loads(response.json()["choices"][0]["message"]["content"])
        result["extraction"] = asdict(extraction)
        result["source_excerpt"] = extraction.text[:3_000]
        result["llm_used"] = True
        return result
    except Exception as exc:
        result = _fallback_summary(extraction.text, extraction)
        result["extraction"]["warnings"].append(f"LLM-анализ недоступен: {exc}")
        return result
