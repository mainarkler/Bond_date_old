from __future__ import annotations

import io
import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz
import pytesseract
from PIL import Image
from docx import Document
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

OCR_LANG = os.getenv("EMISSION_OCR_LANG", "rus+eng")
LOCAL_MODEL_ID = os.getenv("EMISSION_LOCAL_MODEL", "Qwen/Qwen3-4B")
LOCAL_MAX_NEW_TOKENS = int(os.getenv("EMISSION_LOCAL_MAX_NEW_TOKENS", "3000"))
MAX_CHUNK_CHARS = int(os.getenv("EMISSION_ANALYSIS_CHUNK_CHARS", "14000"))


@dataclass
class SourcePage:
    page: int
    text: str
    method: str
    confidence: float | None = None


@dataclass
class DocumentData:
    filename: str
    file_type: str
    pages: list[SourcePage] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(f"[Страница {p.page}]\n{p.text}" for p in self.pages if p.text.strip())


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _ocr_page(page: fitz.Page) -> tuple[str, float | None]:
    pix = page.get_pixmap(matrix=fitz.Matrix(2.2, 2.2), alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    data = pytesseract.image_to_data(
        image,
        lang=OCR_LANG,
        config="--psm 6",
        output_type=pytesseract.Output.DICT,
    )
    parts: list[str] = []
    confidences: list[float] = []
    for text, conf in zip(data.get("text", []), data.get("conf", [])):
        token = str(text).strip()
        if not token:
            continue
        parts.append(token)
        try:
            value = float(conf)
            if value >= 0:
                confidences.append(value)
        except (TypeError, ValueError):
            pass
    return _clean_text(" ".join(parts)), (
        round(sum(confidences) / len(confidences), 1) if confidences else None
    )


def extract_pdf(data: bytes, filename: str) -> DocumentData:
    doc = fitz.open(stream=data, filetype="pdf")
    pages: list[SourcePage] = []
    for idx, page in enumerate(doc, start=1):
        text = _clean_text(page.get_text("text"))
        # A scanned page often has no text layer or only a few OCR-like artifacts.
        if len(re.sub(r"\s+", "", text)) < 80:
            try:
                text, confidence = _ocr_page(page)
                method = "OCR" if text else "empty"
            except (pytesseract.TesseractNotFoundError, RuntimeError) as exc:
                logger.warning("OCR failed on page %s: %s", idx, exc)
                text, confidence, method = text, None, "text-layer"
        else:
            confidence, method = None, "text-layer"
        pages.append(SourcePage(idx, text, method, confidence))
    doc.close()
    return DocumentData(filename, "PDF", pages)


def extract_docx(data: bytes, filename: str) -> DocumentData:
    with tempfile.NamedTemporaryFile(suffix=".docx") as tmp:
        tmp.write(data)
        tmp.flush()
        doc = Document(tmp.name)

        blocks: list[str] = []
        for paragraph in doc.paragraphs:
            value = _clean_text(paragraph.text)
            if value:
                blocks.append(value)

        for table_index, table in enumerate(doc.tables, start=1):
            rows = []
            for row in table.rows:
                cells = [_clean_text(cell.text) for cell in row.cells]
                rows.append(" | ".join(cells))
            if rows:
                blocks.append(f"[Таблица {table_index}]\n" + "\n".join(rows))

    text = "\n\n".join(blocks)
    return DocumentData(filename, "DOCX", [SourcePage(1, text, "docx")])


def extract_document(data: bytes, filename: str) -> DocumentData:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(data, filename)
    if suffix == ".docx":
        return extract_docx(data, filename)
    raise ValueError("Поддерживаются PDF и DOCX. Формат DOC (старый Word) пока не поддерживается.")


def _chunks(document: DocumentData) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for page in document.pages:
        block = f"[Страница {page.page}]\n{page.text}"
        if not page.text.strip():
            continue
        if current and size + len(block) > MAX_CHUNK_CHARS:
            chunks.append("\n\n".join(current))
            current, size = [], 0
        current.append(block)
        size += len(block)
    if current:
        chunks.append("\n\n".join(current))
    return chunks or ["Документ не содержит распознаваемого текста."]


SYSTEM_PROMPT = """
Ты анализируешь только один эмиссионный документ (проспект, решение о выпуске,
условия выпуска/размещения или иной документ эмитента). Не используй рыночные
данные, котировки, MOEX, внешние новости или сведения, которых нет в тексте.
Извлекай факты буквально и сохраняй страницу-источник.

Верни JSON с ключами:
issuer, security_type, issue_volume, nominal, currency, maturity,
placement_period, coupon, coupon_periods, amortization, offers,
early_redemption, buyback, security_guarantees, subordination, covenants,
default_events, payment_mechanics, claim_priority, taxes_legal_notes,
fees_expenses, restrictions, risk_factors, unusual_terms, other_material_terms,
source_references.

Для каждого поля указывай подробное содержание, если оно есть. Для сложных
условий (досрочное погашение, оферта, дефолт, ковенанты, амортизация и т.п.)
НЕ сокращай перечень: перечисляй все условия, сроки, триггеры, формулы и
процедуры, которые приведены в документе. Если данных нет, используй "не
указано в документе". source_references — массив объектов {page, topic}.
""".strip()


def _json_from_response(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _llm_complete(system_prompt: str, user_prompt: str) -> str:
    """Call Qwen3 through OpenRouter; no local model is loaded in Streamlit."""
    import requests

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        try:
            import streamlit as st
            api_key = str(st.secrets.get("OPENROUTER_API_KEY", "")).strip()
        except Exception:
            api_key = ""
    if not api_key:
        raise RuntimeError(
            "Не задан OPENROUTER_API_KEY. Добавьте ключ OpenRouter в Streamlit Secrets."
        )

    model = os.getenv(
        "EMISSION_LLM_MODEL",
        "qwen/qwen3-4b:free",
    )
    base_url = os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1/chat/completions",
    )
    timeout = int(os.getenv("EMISSION_LLM_TIMEOUT", "180"))

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": int(os.getenv("EMISSION_LLM_MAX_TOKENS", "3500")),
        "reasoning": {"enabled": False},
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/mainarkler/Bond_date_old",
        "X-Title": "Bond Date — Анализ эмиссионных документов",
    }

    try:
        response = requests.post(
            base_url,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Ошибка соединения с Qwen3 API: {exc}") from exc

    if response.status_code >= 400:
        try:
            error = response.json().get("error", {})
            detail = error.get("message") or response.text
        except ValueError:
            detail = response.text
        raise RuntimeError(
            f"Qwen3 API вернул HTTP {response.status_code}: {detail}"
        )

    try:
        result = response.json()
        message = result["choices"][0]["message"]
        response_content = message.get("content", "")
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Qwen3 API вернул неожиданный формат ответа.") from exc

    if not isinstance(response_content, str) or not response_content.strip():
        raise RuntimeError("Qwen3 API вернул пустой ответ.")

    return response_content.strip()

def _merge_extractions(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {}
    merged: dict[str, Any] = {}
    for item in items:
        for key, value in item.items():
            if value in (None, "", [], "не указано в документе"):
                merged.setdefault(key, value)
                continue
            if isinstance(value, list):
                existing = merged.setdefault(key, [])
                if not isinstance(existing, list):
                    existing = [existing]
                    merged[key] = existing
                for element in value:
                    if element not in existing:
                        existing.append(element)
            elif key not in merged or merged[key] in (None, "", "не указано в документе"):
                merged[key] = value
            elif isinstance(merged[key], str) and isinstance(value, str) and value not in merged[key]:
                merged[key] += "\n\n" + value
    return merged


def analyze_document(document: DocumentData, progress=None) -> dict[str, Any]:
    chunks = _chunks(document)
    extracted: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        if progress:
            progress(f"Извлечение условий: блок {index} из {len(chunks)}")
        prompt = (
            "Извлеки из следующего фрагмента только факты эмиссионного документа. "
            "Сохраняй номера страниц и не делай выводов за пределами текста.\n\n"
            + chunk
        )
        extracted.append(_json_from_response(_llm_complete(SYSTEM_PROMPT, prompt)))

    facts = _merge_extractions(extracted)
    if progress:
        progress("Формирование итогового аналитического заключения")

    final_system = """
Ты готовишь аналитическое резюме одного эмиссионного документа на основе
извлечённых фактов. Не добавляй рыночные данные и внешние сведения.
Разделяй факты документа и аналитические комментарии.

Верни JSON:
{
  "summary": "краткое описание выпуска",
  "key_features": [],
  "attention_points": [],
  "risks": [],
  "holder_favorable_mechanisms": [],
  "holder_unfavorable_terms": [],
  "analytical_conclusion": "нейтральное описание документа",
  "detailed_conditions": {
     "early_redemption": "...полный перечень условий...",
     "offers": "...полный перечень условий...",
     "default_events": "...полный перечень условий...",
     "covenants": "...полный перечень условий...",
     "amortization": "...полный перечень условий..."
  }
}
Для detailed_conditions сохраняй полные условия, а не ссылки "см. документ".
""".strip()

    analysis = _json_from_response(
        _llm_complete(
            final_system,
            "Факты документа:\n" + json.dumps(facts, ensure_ascii=False, indent=2),
        )
    )
    return {"facts": facts, "analysis": analysis, "chunks": len(chunks)}


def _font_paths() -> tuple[str, str]:
    candidates = [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
    ]
    for regular, bold in candidates:
        if Path(regular).exists() and Path(bold).exists():
            return regular, bold
    raise RuntimeError("Не найден шрифт с поддержкой кириллицы для PDF-отчёта.")


def _register_fonts() -> None:
    if "EmissionRegular" not in pdfmetrics.getRegisteredFontNames():
        regular, bold = _font_paths()
        pdfmetrics.registerFont(TTFont("EmissionRegular", regular))
        pdfmetrics.registerFont(TTFont("EmissionBold", bold))


def _safe(value: Any) -> str:
    if value is None or value == "":
        return "не указано в документе"
    if isinstance(value, list):
        return "\n".join(f"• {x}" for x in value) or "не указано в документе"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def build_pdf_report(document: DocumentData, result: dict[str, Any]) -> bytes:
    _register_fonts()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"Анализ эмиссионного документа — {document.filename}",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle("EmissionTitle", parent=styles["Title"], fontName="EmissionBold", fontSize=18, leading=22, alignment=TA_CENTER, spaceAfter=10)
    h1 = ParagraphStyle("EmissionH1", parent=styles["Heading1"], fontName="EmissionBold", fontSize=13, leading=16, spaceBefore=9, spaceAfter=6)
    body = ParagraphStyle("EmissionBody", parent=styles["BodyText"], fontName="EmissionRegular", fontSize=9.2, leading=13, spaceAfter=5)
    small = ParagraphStyle("EmissionSmall", parent=body, fontSize=7.5, leading=10)

    story: list[Any] = [
        Paragraph("Анализ эмиссионного документа", title),
        Paragraph(document.filename, body),
        Paragraph("Документ анализируется без использования рыночных данных и MOEX.", small),
    ]

    analysis = result.get("analysis", {})
    facts = result.get("facts", {})

    story += [Paragraph("1. Краткое резюме", h1), Paragraph(_safe(analysis.get("summary")), body)]

    story += [Paragraph("2. Основные параметры выпуска", h1)]
    fields = [
        ("Эмитент", "issuer"), ("Вид ценной бумаги", "security_type"),
        ("Объём выпуска", "issue_volume"), ("Номинал", "nominal"),
        ("Валюта", "currency"), ("Срок обращения / погашение", "maturity"),
        ("Период размещения", "placement_period"), ("Купон", "coupon"),
        ("Периоды купона", "coupon_periods"), ("Амортизация", "amortization"),
        ("Обеспечение / гарантии", "security_guarantees"),
        ("Субординация", "subordination"),
    ]
    table_data = [["Показатель", "Условие"]]
    for label, key in fields:
        table_data.append([label, _safe(facts.get(key))])
    table = Table(table_data, colWidths=[52 * mm, 118 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "EmissionRegular"),
        ("FONTNAME", (0, 0), (-1, 0), "EmissionBold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#182230")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d9dee5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f8fa")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [table]

    sections = [
        ("3. Оферты и досрочное погашение", [("Оферты", "offers"), ("Досрочное погашение", "early_redemption"), ("Обратный выкуп эмитентом", "buyback")]),
        ("4. Обеспечение, ковенанты и дефолты", [("Ковенанты", "covenants"), ("События дефолта", "default_events"), ("Очередность требований", "claim_priority")]),
        ("5. Механика выплат и прочие условия", [("Порядок выплат", "payment_mechanics"), ("Налоги и юридические положения", "taxes_legal_notes"), ("Комиссии и расходы", "fees_expenses"), ("Ограничения", "restrictions"), ("Иные существенные условия", "other_material_terms"), ("Нестандартные условия", "unusual_terms")]),
    ]
    for heading, rows in sections:
        story.append(Paragraph(heading, h1))
        for label, key in rows:
            story.append(Paragraph(f"<b>{label}</b>", body))
            story.append(Paragraph(_safe(facts.get(key)).replace("\n", "<br/>"), body))

    story.append(Paragraph("6. Факторы риска", h1))
    story.append(Paragraph(_safe(facts.get("risk_factors")).replace("\n", "<br/>"), body))

    for heading, key in [
        ("7. Ключевые особенности", "key_features"),
        ("8. На что обратить внимание", "attention_points"),
        ("9. Риски по результатам анализа документа", "risks"),
        ("10. Потенциально неблагоприятные для держателя условия", "holder_unfavorable_terms"),
        ("11. Механизмы защиты / потенциально благоприятные условия", "holder_favorable_mechanisms"),
    ]:
        story.append(Paragraph(heading, h1))
        story.append(Paragraph(_safe(analysis.get(key)).replace("\n", "<br/>"), body))

    story.append(Paragraph("12. Аналитическое заключение", h1))
    story.append(Paragraph(_safe(analysis.get("analytical_conclusion")).replace("\n", "<br/>"), body))

    story.append(Paragraph("13. Источники внутри документа", h1))
    refs = facts.get("source_references", [])
    if isinstance(refs, list) and refs:
        for ref in refs:
            if isinstance(ref, dict):
                story.append(Paragraph(f"Стр. {ref.get('page', '—')}: {ref.get('topic', '')}", small))
            else:
                story.append(Paragraph(str(ref), small))
    else:
        story.append(Paragraph("Ссылки на страницы не были выделены моделью.", small))

    doc.build(story)
    return buffer.getvalue()


def analyze_and_report(data: bytes, filename: str, progress=None) -> tuple[DocumentData, dict[str, Any], bytes]:
    document = extract_document(data, filename)
    if not any(page.text.strip() for page in document.pages):
        raise ValueError("Не удалось извлечь текст из документа. Проверьте качество скана.")
    result = analyze_document(document, progress=progress)
    report = build_pdf_report(document, result)
    return document, result, report
