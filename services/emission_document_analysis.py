"""Reliable offline extraction and structured summary for issuer documents."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Any

OCR_MIN_TEXT = 80
MAX_OCR_PAGES = 100


@dataclass
class DocumentPage:
    number: int | None
    text: str
    source: str


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _match(text: str, pattern: str, flags: int = re.IGNORECASE) -> str | None:
    found = re.search(pattern, text, flags)
    return _clean(found.group(1)) if found else None


def _ocr_page(pdf: Any, index: int) -> str:
    import pytesseract
    from PIL import Image

    pixmap = pdf[index].get_pixmap(matrix=__import__("fitz").Matrix(2, 2), alpha=False)
    return pytesseract.image_to_string(Image.open(BytesIO(pixmap.tobytes("png"))), lang="rus+eng")


def extract_document(file_name: str, content: bytes) -> tuple[list[DocumentPage], list[str]]:
    """Read all PDF/DOCX content; OCR is attempted only on textless PDF pages."""
    extension = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    warnings: list[str] = []
    if extension == "docx":
        try:
            from docx import Document

            document = Document(BytesIO(content))
            sections = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
            sections.extend(" | ".join(cell.text for cell in row.cells) for table in document.tables for row in table.rows)
            return [DocumentPage(None, _clean("\n".join(sections)), "docx")], warnings
        except Exception as exc:
            raise ValueError(f"Не удалось прочитать DOCX: {exc}") from exc
    if extension != "pdf":
        raise ValueError("Поддерживаются только PDF и DOCX.")
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        pages = [DocumentPage(index + 1, _clean(page.extract_text() or ""), "pdf_text") for index, page in enumerate(reader.pages)]
    except Exception as exc:
        raise ValueError(f"Не удалось прочитать PDF: {exc}") from exc
    empty = [page.number for page in pages if len(page.text) < OCR_MIN_TEXT]
    if empty:
        try:
            import fitz

            pdf = fitz.open(stream=content, filetype="pdf")
            for number in empty[:MAX_OCR_PAGES]:
                pages[number - 1] = DocumentPage(number, _clean(_ocr_page(pdf, number - 1)), "ocr")
            if len(empty) > MAX_OCR_PAGES:
                warnings.append(f"OCR выполнен для первых {MAX_OCR_PAGES} из {len(empty)} страниц без текста.")
        except Exception as exc:
            warnings.append(f"Не удалось распознать страницы PDF без текста: {exc}")
    return pages, warnings


def _key_data(text: str) -> list[dict[str, str]]:
    patterns = [
        ("Эмитент", r"((?:ООО|ПАО|АО|Общество с ограниченной ответственностью|Публичное акционерное общество)\s*(?:«[^»]{2,120}»|[^.\n]{2,120}))"),
        ("ISIN", r"\b(RU[0-9A-Z]{10})\b"),
        ("Регистрационный номер", r"(?:государственн(?:ый|ого) регистрационн(?:ый|ого) номер|регистрационный номер)\s*[:№]?\s*([0-9A-ZА-Я][0-9A-ZА-Я\-/]{4,70})"),
        ("Программа / выпуск", r"Программа облигаций\s*[—–-]\s*([^\.]{3,220})"),
        ("Купон", r"(?:ставка купона|купонн(?:ая|ый) ставк[аи]|размер купона)\s*[:—–-]?\s*([^\.\n]{2,100})"),
        ("Погашение", r"(?:дата погашения|срок погашения)\s*[:—–-]?\s*([^\.\n]{2,120})"),
        ("Объём / номинал", r"(?:объем выпуска|номинальн(?:ая|ый) стоимость)\s*[:—–-]?\s*([^\.\n]{2,160})"),
    ]
    return [{"name": name, "value": value, "note": "Найдено в извлечённом тексте."} for name, pattern in patterns if (value := _match(text, pattern))]


def analyse_emission_document(file_name: str, content: bytes) -> dict[str, Any]:
    """Build a deterministic offline result; no API key or model is required."""
    pages, warnings = extract_document(file_name, content)
    text = "\n\n".join(page.text for page in pages if page.text)
    data = _key_data(text)
    attention = []
    for pattern, message in [
        (r"оферт", "Проверьте условия оферты и порядок досрочного выкупа."),
        (r"амортизац", "Проверьте график амортизации номинала."),
        (r"ковенант", "Проверьте ковенанты и последствия их нарушения."),
        (r"обеспечени|залог|поручительств", "Проверьте достаточность и порядок реализации обеспечения."),
        (r"риск", "Изучите факторы риска и их влияние на выплаты."),
    ]:
        if re.search(pattern, text, re.IGNORECASE):
            attention.append(message)
    if not attention:
        attention.append("Проверьте условия выпуска, порядок выплат, досрочное погашение и ковенанты.")
    facts = "; ".join(f"{item['name']}: {item['value']}" for item in data[:4])
    summary = f"Обработано страниц: {len(pages)}. "
    summary += f"Выделенные параметры: {facts}." if facts else "Ключевые параметры автоматически не найдены; проверьте текст по страницам."
    return {
        "summary": summary,
        "key_data": data,
        "attention_points": attention,
        "coverage": f"Обработано {len(pages)} страниц, извлечено {len(text):,} символов.".replace(",", " "),
        "warnings": warnings,
        "pages": [asdict(page) for page in pages],
        "excerpt": text[:3_000],
    }
