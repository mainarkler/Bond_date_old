"""Token-aware extraction and two-stage analysis of issuer offering documents."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from io import BytesIO
from typing import Any

import httpx

from news_agent_config import settings

OCR_TEXT_MIN_CHARS = 80
MAX_OCR_PAGES = 100
MAX_BLOCK_TOKENS = 2_500
MAX_ANALYSIS_BLOCKS = 50


@dataclass(frozen=True)
class DocumentBlock:
    """A token-bounded document fragment sent to the recognition stage."""

    page: int | None
    index: int
    source: str
    text: str
    tokens: int


@dataclass(frozen=True)
class ExtractedDocument:
    text: str
    pages: int | None
    used_ocr: bool
    warnings: list[str]
    blocks: list[DocumentBlock] = field(default_factory=list)


def _clean_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


def _token_count(text: str) -> int:
    """Count model tokens when tiktoken is installed; retain a safe fallback."""
    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(settings.openai_model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Approximation is deliberately conservative for environments without tiktoken.
        return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


def _split_into_blocks(parts: list[tuple[int | None, str, str]]) -> list[DocumentBlock]:
    blocks: list[DocumentBlock] = []
    for page, source, raw_text in parts:
        paragraphs = [item.strip() for item in re.split(r"\n{2,}", _clean_text(raw_text)) if item.strip()]
        current: list[str] = []
        for paragraph in paragraphs:
            candidate = "\n\n".join([*current, paragraph])
            if current and _token_count(candidate) > MAX_BLOCK_TOKENS:
                text = "\n\n".join(current)
                blocks.append(DocumentBlock(page, len(blocks) + 1, source, text, _token_count(text)))
                current = [paragraph]
            else:
                current.append(paragraph)
        if current:
            text = "\n\n".join(current)
            # A single very long paragraph is split on words to keep API requests bounded.
            remaining = text.split()
            while remaining:
                chunk: list[str] = []
                while remaining:
                    candidate = " ".join([*chunk, remaining[0]])
                    if chunk and _token_count(candidate) > MAX_BLOCK_TOKENS:
                        break
                    chunk.append(remaining.pop(0))
                    if _token_count(" ".join(chunk)) > MAX_BLOCK_TOKENS:
                        break
                chunk_text = " ".join(chunk)
                if chunk_text:
                    blocks.append(DocumentBlock(page, len(blocks) + 1, source, chunk_text, _token_count(chunk_text)))
    return blocks


def _ocr_pdf_page(pdf: Any, page_index: int) -> str:
    import pytesseract
    from PIL import Image

    page = pdf[page_index]
    pixmap = page.get_pixmap(matrix=__import__("fitz").Matrix(2, 2), alpha=False)
    image = Image.open(BytesIO(pixmap.tobytes("png")))
    return pytesseract.image_to_string(image, lang="rus+eng")


def extract_document_text(file_name: str, content: bytes) -> ExtractedDocument:
    """Extract every PDF page independently, OCRing only pages lacking text."""
    suffix = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    warnings: list[str] = []
    if suffix == "docx":
        try:
            from docx import Document

            document = Document(BytesIO(content))
            parts = [(None, "paragraph", paragraph.text) for paragraph in document.paragraphs if paragraph.text.strip()]
            parts.extend(
                (None, "table", " | ".join(cell.text for cell in row.cells))
                for table in document.tables
                for row in table.rows
            )
        except Exception as exc:
            raise ValueError(f"Не удалось прочитать Word-документ: {exc}") from exc
        blocks = _split_into_blocks(parts)
        return ExtractedDocument("\n\n".join(block.text for block in blocks), None, False, warnings, blocks)

    if suffix != "pdf":
        raise ValueError("Поддерживаются только файлы PDF и DOCX.")

    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        pages = len(reader.pages)
        page_texts = [(index + 1, page.extract_text() or "") for index, page in enumerate(reader.pages)]
    except Exception as exc:
        raise ValueError(f"Не удалось прочитать PDF: {exc}") from exc

    used_ocr = False
    missing_text_pages = [page for page, text in page_texts if len(_clean_text(text)) < OCR_TEXT_MIN_CHARS]
    if missing_text_pages:
        try:
            import fitz

            pdf = fitz.open(stream=content, filetype="pdf")
            for page_number in missing_text_pages[:MAX_OCR_PAGES]:
                page_texts[page_number - 1] = (page_number, _ocr_pdf_page(pdf, page_number - 1))
            used_ocr = True
            if len(missing_text_pages) > MAX_OCR_PAGES:
                warnings.append(
                    f"OCR выполнен для первых {MAX_OCR_PAGES} из {len(missing_text_pages)} страниц без текстового слоя."
                )
        except Exception as exc:
            warnings.append(f"Не удалось выполнить OCR страниц без текста: {exc}")

    parts = [(page, "pdf_ocr" if page in missing_text_pages else "pdf_text", text) for page, text in page_texts]
    blocks = _split_into_blocks(parts)
    text = "\n\n".join(block.text for block in blocks)
    if not text:
        warnings.append("Текст документа не извлечён; результат требует ручной проверки.")
    return ExtractedDocument(text, pages, used_ocr, warnings, blocks)


def _fallback_summary(extraction: ExtractedDocument) -> dict[str, Any]:
    excerpt = extraction.text[:3_000] or "Текст не удалось извлечь."
    return {
        "summary": "Автоматическое LLM-резюме недоступно. Ниже приведён извлечённый фрагмент для первичного просмотра.",
        "key_data": [],
        "attention_points": [
            "Проверьте условия выпуска, порядок выплат, досрочное погашение и ковенанты в первичном документе.",
            "Сверьте все цифры с оригиналом: извлечение текста или OCR может содержать ошибки.",
        ],
        "document_coverage": f"Извлечено {len(extraction.blocks)} токенизированных блоков.",
        "source_excerpt": excerpt,
        "extraction": asdict(extraction),
        "llm_used": False,
    }


def _complete_json(system: str, prompt: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    """Call an OpenAI-compatible endpoint, with a JSON-mode compatibility retry."""
    payload = {
        "model": settings.openai_model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    }
    headers = {"Authorization": f"Bearer {settings.openai_api_key}", "Content-Type": "application/json"}
    url = f"{settings.openai_base_url.rstrip('/')}/chat/completions"
    response = httpx.post(url, headers=headers, json={**payload, "response_format": {"type": "json_object"}}, timeout=settings.request_timeout_seconds)
    if response.status_code == 400:
        # Some OpenAI-compatible gateways do not implement response_format.
        response = httpx.post(url, headers=headers, json=payload, timeout=settings.request_timeout_seconds)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    if isinstance(content, str):
        content = re.sub(r"^```json\s*|\s*```$", "", content.strip())
    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError("LLM вернула JSON не в формате объекта.")
    return result


def _recognise_blocks(blocks: list[DocumentBlock], warnings: list[str]) -> list[dict[str, Any]]:
    recognized: list[dict[str, Any]] = []
    for block in blocks[:MAX_ANALYSIS_BLOCKS]:
        try:
            recognized.append(
                _complete_json(
                    "Ты извлекаешь проверяемые факты из эмиссионного документа. Отвечай только JSON.",
                    {
                        "task": (
                            "Конвертируй блок в JSON: {block_index, page, facts: [{field, value, quote}], "
                            "risk_flags: [string]}. Не интерпретируй и не выдумывай. quote — короткая цитата из блока."
                        ),
                        "block": asdict(block),
                    },
                    max_tokens=1_500,
                )
            )
        except Exception as exc:
            warnings.append(f"Не распознан блок {block.index} (стр. {block.page or 'DOCX'}): {exc}")
    if len(blocks) > MAX_ANALYSIS_BLOCKS:
        warnings.append(f"Для LLM распознаны первые {MAX_ANALYSIS_BLOCKS} из {len(blocks)} блоков.")
    return recognized


def analyse_emission_document(file_name: str, content: bytes) -> dict[str, Any]:
    extraction = extract_document_text(file_name, content)
    if not extraction.text or not settings.openai_api_key:
        return _fallback_summary(extraction)

    warnings = extraction.warnings.copy()
    recognized_blocks = _recognise_blocks(extraction.blocks, warnings)
    if not recognized_blocks:
        extraction = ExtractedDocument(extraction.text, extraction.pages, extraction.used_ocr, warnings, extraction.blocks)
        return _fallback_summary(extraction)

    try:
        result = _complete_json(
            "Ты аналитик долгового рынка. На основе только переданного JSON подготовь итог на русском. Отвечай только JSON.",
            {
                "task": (
                    "Собери {summary, key_data: [{name, value, note}], attention_points, document_coverage}. "
                    "Не добавляй фактов, которых нет в распознанных блоках. Укажи эмитента, выпуск/ISIN, объём, "
                    "номинал, валюту, срок, купон, выплаты, амортизацию, оферту, обеспечение, ковенанты, рейтинги и риски, если они найдены."
                ),
                "recognized_blocks": recognized_blocks,
            },
            max_tokens=2_000,
        )
        extraction = ExtractedDocument(extraction.text, extraction.pages, extraction.used_ocr, warnings, extraction.blocks)
        result.update(
            extraction=asdict(extraction),
            source_excerpt=extraction.text[:3_000],
            llm_used=True,
            recognized_blocks=recognized_blocks,
        )
        return result
    except Exception as exc:
        warnings.append(f"Не удалось собрать итоговое LLM-резюме: {exc}")
        extraction = ExtractedDocument(extraction.text, extraction.pages, extraction.used_ocr, warnings, extraction.blocks)
        return _fallback_summary(extraction)
