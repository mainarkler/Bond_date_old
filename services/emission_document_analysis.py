"""Offline, token-aware analysis of issuer offering documents with a local GGUF LLM."""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

OCR_TEXT_MIN_CHARS = 80
MAX_OCR_PAGES = 100
MAX_BLOCK_TOKENS = 2_500
MAX_ANALYSIS_BLOCKS = 250


@dataclass(frozen=True)
class DocumentBlock:
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


@dataclass(frozen=True)
class LocalLLMConfig:
    model_path: str
    context_size: int = 8_192
    threads: int = 0


def _clean_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


def _token_count(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
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
            remaining = "\n\n".join(current).split()
            while remaining:
                chunk: list[str] = []
                while remaining:
                    candidate = " ".join([*chunk, remaining[0]])
                    if chunk and _token_count(candidate) > MAX_BLOCK_TOKENS:
                        break
                    chunk.append(remaining.pop(0))
                    if _token_count(" ".join(chunk)) > MAX_BLOCK_TOKENS:
                        break
                text = " ".join(chunk)
                if text:
                    blocks.append(DocumentBlock(page, len(blocks) + 1, source, text, _token_count(text)))
    return blocks


def _ocr_pdf_page(pdf: Any, page_index: int) -> str:
    import pytesseract
    from PIL import Image

    page = pdf[page_index]
    pixmap = page.get_pixmap(matrix=__import__("fitz").Matrix(2, 2), alpha=False)
    image = Image.open(BytesIO(pixmap.tobytes("png")))
    return pytesseract.image_to_string(image, lang="rus+eng")


def extract_document_text(file_name: str, content: bytes) -> ExtractedDocument:
    """Extract every PDF page independently, OCRing only pages that lack text."""
    suffix = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    warnings: list[str] = []
    if suffix == "docx":
        try:
            from docx import Document

            document = Document(BytesIO(content))
            parts = [(None, "paragraph", item.text) for item in document.paragraphs if item.text.strip()]
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
                warnings.append(f"OCR выполнен для первых {MAX_OCR_PAGES} из {len(missing_text_pages)} страниц без текстового слоя.")
        except Exception as exc:
            warnings.append(f"Не удалось выполнить OCR страниц без текста: {exc}")
    parts = [(page, "pdf_ocr" if page in missing_text_pages else "pdf_text", text) for page, text in page_texts]
    blocks = _split_into_blocks(parts)
    text = "\n\n".join(block.text for block in blocks)
    if not text:
        warnings.append("Текст документа не извлечён; результат требует ручной проверки.")
    return ExtractedDocument(text, pages, used_ocr, warnings, blocks)


def local_model_status(model_path: str | None = None) -> tuple[bool, str]:
    path = Path(model_path or os.getenv("LOCAL_LLM_MODEL_PATH", "models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"))
    if not path.is_file():
        return False, f"Локальная GGUF-модель не найдена: {path}"
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return False, "Не установлена зависимость llama-cpp-python для локальной модели."
    return True, str(path)


@lru_cache(maxsize=1)
def _load_local_model(model_path: str, context_size: int, threads: int) -> Any:
    from llama_cpp import Llama

    return Llama(model_path=model_path, n_ctx=context_size, n_threads=threads or None, verbose=False)


def _complete_json(system: str, prompt: dict[str, Any], max_tokens: int, config: LocalLLMConfig) -> dict[str, Any]:
    model = _load_local_model(config.model_path, config.context_size, config.threads)
    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": f"{system} Не используй сеть. Отвечай только валидным JSON."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    content = response["choices"][0]["message"]["content"]
    if isinstance(content, str):
        content = re.sub(r"^```json\s*|\s*```$", "", content.strip())
    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError("Локальная модель вернула JSON не в формате объекта.")
    return result


def _first_match(text: str, pattern: str, *, flags: int = re.IGNORECASE) -> str | None:
    match = re.search(pattern, text, flags)
    return _clean_text(match.group(1)) if match else None


def _local_rule_summary(extraction: ExtractedDocument) -> dict[str, Any]:
    """Produce useful structured output even when the optional GGUF model is absent."""
    text = extraction.text
    key_data: list[dict[str, str]] = []
    issuer = _first_match(
        text,
        r"(?:Эмитент[^.]{0,100}[—–-]\s*)?((?:Общество с ограниченной ответственностью|Публичное акционерное общество|Акционерное общество|ООО|ПАО|АО)\s*(?:«[^»\n]{2,120}»|\"[^\"\n]{2,120}\"|[^\n.]{2,120}))",
    )
    program = _first_match(text, r"Программа облигаций\s*[—–-]\s*([^\.]{3,220})")
    registration = _first_match(text, r"(?:государственн(?:ый|ого) регистрационн(?:ый|ого) номер|регистрационный номер)\s*[:№]?\s*([0-9A-ZА-Я][0-9A-ZА-Я\-/]{4,70})")
    isin = _first_match(text, r"\b(RU[0-9A-Z]{10})\b", flags=0)
    coupon = _first_match(text, r"(?:купонн(?:ая|ый) ставк[аи]|ставка купона|размер купона)\s*[:—–-]?\s*([^\.\n]{2,100})")
    maturity = _first_match(text, r"(?:дата погашения|погашени[ея])\s*[:—–-]?\s*([^\.\n]{2,120})")
    amount = _first_match(text, r"(?:объем выпуска|номинальн(?:ая|ый) стоимость|количество облигаций)\s*[:—–-]?\s*([^\.\n]{2,160})")
    for name, value in [
        ("Эмитент", issuer),
        ("Программа / выпуск", program),
        ("Регистрационный номер", registration),
        ("ISIN", isin),
        ("Купон", coupon),
        ("Погашение", maturity),
        ("Параметры выпуска", amount),
    ]:
        if value:
            key_data.append({"name": name, "value": value, "note": "Найдено локальным анализатором в тексте документа."})

    attention_points: list[str] = []
    checks = [
        (r"оферт", "Проверьте условия оферты и право досрочного выкупа."),
        (r"досрочн(?:ое|ого) погаш", "Проверьте основания и порядок досрочного погашения."),
        (r"амортизац", "Проверьте график амортизации номинала."),
        (r"ковенант", "Проверьте ковенанты и последствия их нарушения."),
        (r"обеспечени|залог|поручительств", "Проверьте состав, достаточность и порядок реализации обеспечения."),
        (r"риск", "Изучите раздел факторов риска и его влияние на выплаты."),
    ]
    for pattern, message in checks:
        if re.search(pattern, text, re.IGNORECASE):
            attention_points.append(message)
    if not attention_points:
        attention_points.append("Проверьте условия выпуска, порядок выплат, досрочное погашение и ковенанты в первичном документе.")

    summary_parts = [
        f"Локально обработан документ: {extraction.pages or 'не указано'} стр., {len(extraction.blocks)} токенизированных блоков."
    ]
    if issuer:
        summary_parts.append(f"В тексте найден эмитент: {issuer}.")
    if program:
        summary_parts.append(f"Указана программа/выпуск: {program}.")
    summary_parts.append("Сводка сформирована по всему извлечённому тексту без внешних сервисов; параметры требуют сверки с оригиналом.")
    return {
        "summary": " ".join(summary_parts),
        "key_data": key_data,
        "attention_points": attention_points,
        "document_coverage": f"Локально обработано {len(extraction.blocks)} блоков из {extraction.pages or 'DOCX'} страниц.",
        "source_excerpt": text[:3_000] or "Текст не удалось извлечь.",
        "extraction": asdict(extraction),
        "llm_used": False,
        "rule_based": True,
    }


def _fallback_summary(extraction: ExtractedDocument) -> dict[str, Any]:
    return _local_rule_summary(extraction)


def _recognise_blocks(blocks: list[DocumentBlock], warnings: list[str], config: LocalLLMConfig) -> list[dict[str, Any]]:
    recognized: list[dict[str, Any]] = []
    for block in blocks[:MAX_ANALYSIS_BLOCKS]:
        try:
            recognized.append(_complete_json(
                "Ты извлекаешь проверяемые факты из эмиссионного документа.",
                {"task": "Конвертируй блок в JSON {block_index, page, facts: [{field, value, quote}], risk_flags: [string]}. Не выдумывай фактов.", "block": asdict(block)},
                max_tokens=1_200,
                config=config,
            ))
        except Exception as exc:
            warnings.append(f"Не распознан блок {block.index} (стр. {block.page or 'DOCX'}): {exc}")
    if len(blocks) > MAX_ANALYSIS_BLOCKS:
        warnings.append(f"Для локальной модели распознаны первые {MAX_ANALYSIS_BLOCKS} из {len(blocks)} блоков.")
    return recognized


def analyse_emission_document(file_name: str, content: bytes, *, model_path: str | None = None) -> dict[str, Any]:
    """Run the fully offline local-model pipeline; no external API is called."""
    extraction = extract_document_text(file_name, content)
    available, status = local_model_status(model_path)
    if not extraction.text or not available:
        extraction = ExtractedDocument(extraction.text, extraction.pages, extraction.used_ocr, [*extraction.warnings, status], extraction.blocks)
        return _fallback_summary(extraction)
    config = LocalLLMConfig(model_path=status)
    warnings = extraction.warnings.copy()
    recognized_blocks = _recognise_blocks(extraction.blocks, warnings, config)
    if not recognized_blocks:
        return _fallback_summary(ExtractedDocument(extraction.text, extraction.pages, extraction.used_ocr, warnings, extraction.blocks))
    try:
        result = _complete_json(
            "Ты аналитик долгового рынка. На основе только переданного JSON подготовь итог на русском.",
            {"task": "Собери {summary, key_data: [{name, value, note}], attention_points, document_coverage}. Не добавляй фактов вне JSON-блоков.", "recognized_blocks": recognized_blocks},
            max_tokens=2_000,
            config=config,
        )
        extraction = ExtractedDocument(extraction.text, extraction.pages, extraction.used_ocr, warnings, extraction.blocks)
        result.update(extraction=asdict(extraction), source_excerpt=extraction.text[:3_000], llm_used=True, recognized_blocks=recognized_blocks)
        return result
    except Exception as exc:
        warnings.append(f"Не удалось собрать итоговое локальное LLM-резюме: {exc}")
        return _fallback_summary(ExtractedDocument(extraction.text, extraction.pages, extraction.used_ocr, warnings, extraction.blocks))
