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
LOCAL_MODEL_ID = os.getenv("EMISSION_LOCAL_MODEL", "Qwen/Qwen3-0.6B")
LOCAL_MAX_NEW_TOKENS = int(os.getenv("EMISSION_LOCAL_MAX_NEW_TOKENS", "1400"))
LOCAL_CONTEXT_TOKENS = int(os.getenv("EMISSION_LOCAL_CONTEXT_TOKENS", "6000"))
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


def _preprocess_ocr_image(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    gray = gray.resize((int(gray.width * 1.15), int(gray.height * 1.15)))
    return gray.point(lambda p: 255 if p > 210 else (0 if p < 115 else p))


def _ocr_page(page: fitz.Page) -> tuple[str, float | None]:
    """OCR a scanned PDF page with several Tesseract layouts."""
    pix = page.get_pixmap(matrix=fitz.Matrix(2.6, 2.6), alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    image = _preprocess_ocr_image(image)
    candidates: list[tuple[str, float | None]] = []

    for psm in (6, 4, 11):
        try:
            data = pytesseract.image_to_data(
                image,
                lang=OCR_LANG,
                config=f"--oem 1 --psm {psm}",
                output_type=pytesseract.Output.DICT,
                timeout=60,
            )
        except (pytesseract.TesseractError, RuntimeError) as exc:
            logger.warning("Tesseract failed, psm=%s: %s", psm, exc)
            continue

        parts: list[str] = []
        confidences: list[float] = []
        for token, conf in zip(data.get("text", []), data.get("conf", [])):
            token = str(token).strip()
            if not token:
                continue
            parts.append(token)
            try:
                value = float(conf)
                if value >= 0:
                    confidences.append(value)
            except (TypeError, ValueError):
                pass

        text = _clean_text(" ".join(parts))
        confidence = (
            round(sum(confidences) / len(confidences), 1)
            if confidences else None
        )
        if text:
            candidates.append((text, confidence))

    if not candidates:
        return "", None

    candidates.sort(
        key=lambda item: (len(re.sub(r"\s+", "", item[0])), item[1] or 0),
        reverse=True,
    )
    return candidates[0]


def extract_pdf(data: bytes, filename: str) -> DocumentData:
    doc = fitz.open(stream=data, filetype="pdf")
    pages: list[SourcePage] = []
    try:
        for idx, page in enumerate(doc, start=1):
            text = _clean_text(page.get_text("text"))
            if len(re.sub(r"\s+", "", text)) < 80:
                text, confidence = _ocr_page(page)
                method = "OCR" if text else "empty"
            else:
                confidence = None
                method = "text-layer"
            pages.append(SourcePage(idx, text, method, confidence))
    finally:
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


RELEVANT_TERMS = (
    "эмитент", "выпуск", "объем", "номинал", "погашен", "срок обращения",
    "купон", "ставка", "амортиза", "оферт", "досроч", "выкуп", "дефолт",
    "ковенант", "обеспеч", "гарант", "субординац", "очередност", "налог",
    "комисси", "расход", "ограничен", "риск", "размещен", "платеж",
    "дата", "формул", "уведомлен", "требован", "досрочн"
)


def _keyword_fragments(
    document: DocumentData,
    max_chars: int = 24000,
    window_chars: int = 900,
) -> tuple[str, list[dict[str, Any]]]:
    """Deterministically extract only text around material keywords.

    This stage uses no AI. It reduces the amount of text that reaches Qwen and
    preserves page references for every selected fragment.
    """
    hits: list[tuple[int, int, int, str, list[str]]] = []

    for page in document.pages:
        text = re.sub(r"\s+", " ", page.text or "").strip()
        if not text:
            continue
        lower = text.lower()
        positions: list[tuple[int, str]] = []

        for term in RELEVANT_TERMS:
            start = 0
            while True:
                pos = lower.find(term, start)
                if pos < 0:
                    break
                positions.append((pos, term))
                start = pos + max(1, len(term))

        if not positions:
            continue

        # Merge nearby keyword hits into larger readable fragments.
        positions.sort()
        ranges: list[tuple[int, int, list[str]]] = []
        for pos, term in positions:
            left = max(0, pos - window_chars)
            right = min(len(text), pos + len(term) + window_chars)
            if ranges and left <= ranges[-1][1] + 250:
                old_left, old_right, terms = ranges[-1]
                ranges[-1] = (old_left, max(old_right, right), terms + [term])
            else:
                ranges.append((left, right, [term]))

        for left, right, terms in ranges:
            fragment = text[left:right].strip()
            score = len(set(terms))
            if page.page == 1:
                score += 3
            hits.append((score, page.page, left, fragment, sorted(set(terms))))

    hits.sort(key=lambda item: (item[0], -item[1]), reverse=True)

    selected: list[str] = []
    selected_meta: list[dict[str, Any]] = []
    total = 0
    seen: set[tuple[int, str]] = set()

    for score, page_no, _, fragment, terms in hits:
        key = (page_no, fragment[:180])
        if key in seen:
            continue
        block = f"[Страница {page_no}]\n{fragment}"
        if total + len(block) > max_chars:
            continue
        seen.add(key)
        selected.append(block)
        selected_meta.append({
            "page": page_no,
            "keywords": terms,
            "score": score,
            "fragment": fragment,
        })
        total += len(block)

    selected.sort(key=lambda block: int(re.search(r"\d+", block).group()))
    selected_meta.sort(key=lambda item: item["page"])

    return "\n\n".join(selected), selected_meta


def _relevant_text(document: DocumentData, max_chars: int = 24000) -> str:
    """Compatibility wrapper returning keyword-selected text only."""
    text, _ = _keyword_fragments(document, max_chars=max_chars)
    return text or document.text[:max_chars]


SYSTEM_PROMPT = """
Ты анализируешь один эмиссионный документ. Используй только переданный текст.
Не используй рынок, MOEX, новости или внешние сведения.

Найди и структурируй только существенные условия выпуска. Для каждого факта
сохраняй номер страницы. Сложные условия не сокращай: перечисляй все
триггеры, сроки, даты, формулы, порядок уведомления, цену/сумму и процедуру,
если они есть в тексте. Если данных нет: "не указано в документе".

Верни JSON:
{
 "issuer": "...", "security_type": "...", "issue_volume": "...",
 "nominal": "...", "currency": "...", "maturity": "...",
 "placement_period": "...", "coupon": "...", "coupon_periods": "...",
 "amortization": "...", "offers": "...", "early_redemption": "...",
 "buyback": "...", "security_guarantees": "...", "subordination": "...",
 "covenants": "...", "default_events": "...", "payment_mechanics": "...",
 "claim_priority": "...", "taxes_legal_notes": "...", "fees_expenses": "...",
 "restrictions": "...", "risk_factors": "...", "unusual_terms": "...",
 "other_material_terms": "...", "source_references": [{"page": 1, "topic": "..."}]
}
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
    """Run Qwen3 locally. No API key or external LLM is used."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    global _QWEN_TOKENIZER, _QWEN_MODEL
    if "_QWEN_TOKENIZER" not in globals():
        _QWEN_TOKENIZER = None
        _QWEN_MODEL = None

    if _QWEN_TOKENIZER is None or _QWEN_MODEL is None:
        logger.info("Loading Qwen3 model: %s", LOCAL_MODEL_ID)
        _QWEN_TOKENIZER = AutoTokenizer.from_pretrained(
            LOCAL_MODEL_ID, trust_remote_code=True
        )
        kwargs = {"trust_remote_code": True}
        if torch.cuda.is_available():
            kwargs.update({"dtype": torch.bfloat16, "device_map": "auto"})
        else:
            kwargs.update({"dtype": torch.float32, "low_cpu_mem_usage": True})
        _QWEN_MODEL = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_ID, **kwargs)

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    prompt = _QWEN_TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = _QWEN_TOKENIZER(
        prompt, return_tensors="pt", truncation=True, max_length=LOCAL_CONTEXT_TOKENS
    )
    if torch.cuda.is_available():
        inputs = {k: v.to(_QWEN_MODEL.device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = _QWEN_MODEL.generate(
            **inputs,
            max_new_tokens=LOCAL_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=_QWEN_TOKENIZER.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return _QWEN_TOKENIZER.decode(generated, skip_special_tokens=True).strip()


def find_keyword_fragments(
    document: DocumentData,
    max_chars: int = 24000,
) -> tuple[str, list[dict[str, Any]]]:
    """First deterministic stage: find material fragments without loading the LLM."""
    return _keyword_fragments(document, max_chars=max_chars)


COMBINED_SYSTEM_PROMPT = """
Ты анализируешь один эмиссионный документ. Используй только переданный текст.
Не используй рынок, MOEX, новости или внешние сведения.

Сначала извлеки факты документа, затем на их основе сформируй краткий
нейтральный анализ. Для каждого факта сохраняй номер страницы.

Критически важно: сложные материальные условия НЕ сокращай. Если в документе
есть досрочное погашение, оферта, дефолт, ковенант, амортизация или иное
условие, перечисли все фактические триггеры, даты, сроки, формулы, порядок
уведомления, цену/сумму, ограничения и процедуру, которые указаны в тексте.
Не заменяй перечисление фразами вроде "при наступлении условий".

Если данных нет: "не указано в документе".

Верни только JSON:
{
  "facts": {
    "issuer": "...",
    "security_type": "...",
    "issue_volume": "...",
    "nominal": "...",
    "currency": "...",
    "maturity": "...",
    "placement_period": "...",
    "coupon": "...",
    "coupon_periods": "...",
    "amortization": "...",
    "offers": "...",
    "early_redemption": "...",
    "buyback": "...",
    "security_guarantees": "...",
    "subordination": "...",
    "covenants": "...",
    "default_events": "...",
    "payment_mechanics": "...",
    "claim_priority": "...",
    "taxes_legal_notes": "...",
    "fees_expenses": "...",
    "restrictions": "...",
    "risk_factors": "...",
    "unusual_terms": "...",
    "other_material_terms": "...",
    "source_references": [{"page": 1, "topic": "..."}]
  },
  "analysis": {
    "summary": "...",
    "key_features": [],
    "attention_points": [],
    "risks": [],
    "holder_favorable_mechanisms": [],
    "holder_unfavorable_terms": [],
    "analytical_conclusion": "...",
    "detailed_conditions": {
      "early_redemption": "...",
      "offers": "...",
      "default_events": "...",
      "covenants": "...",
      "amortization": "..."
    }
  }
}
""".strip()


def analyze_document(document: DocumentData, progress=None) -> dict[str, Any]:
    """Run keyword filtering first, then one Qwen3 call on selected fragments."""
    if progress:
        progress("Шаг 1/2: поиск важных частей по ключевым словам…")

    relevant, keyword_hits = find_keyword_fragments(document, max_chars=24000)
    if not relevant.strip():
        raise ValueError(
            "Не найдены существенные фрагменты по ключевым словам. "
            "Проверьте качество распознанного текста."
        )

    if progress:
        progress(
            f"Шаг 2/2: Qwen3 анализирует {len(keyword_hits)} отобранных фрагментов…"
        )

    result = _json_from_response(
        _llm_complete(
            COMBINED_SYSTEM_PROMPT,
            "Отобранные фрагменты распознанного документа:\n\n" + relevant,
        )
    )

    facts = result.get("facts")
    analysis = result.get("analysis")
    if not isinstance(facts, dict):
        facts = {}
    if not isinstance(analysis, dict):
        analysis = {}

    return {
        "facts": facts,
        "analysis": analysis,
        "selected_text": relevant,
        "keyword_hits": keyword_hits,
    }

def build_analysis_report(document: DocumentData, result: dict[str, Any]) -> bytes:
    return build_pdf_report(document, result)


def analyze_and_report(data: bytes, filename: str, progress=None) -> tuple[DocumentData, dict[str, Any], bytes]:
    """Backward-compatible combined path; UI should use separate stages."""
    document = extract_document(data, filename)
    if not any(page.text.strip() for page in document.pages):
        raise ValueError(
            "Не удалось извлечь текст. OCR не получил распознаваемый текст. "
            "Проверьте установку Tesseract и языковых пакетов "
            "tesseract-ocr-rus/tesseract-ocr-eng в Streamlit."
        )
    result = analyze_document(document, progress=progress)
    return document, result, build_analysis_report(document, result)

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
        raise ValueError(
            "Не удалось извлечь текст. OCR не получил распознаваемый текст. "
            "Проверьте установку Tesseract и языковых пакетов "
            "tesseract-ocr-rus/tesseract-ocr-eng в Streamlit."
        )
    result = analyze_document(document, progress=progress)
    report = build_pdf_report(document, result)
    return document, result, report
