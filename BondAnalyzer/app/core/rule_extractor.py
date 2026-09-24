from __future__ import annotations
import re
from .models import ExtractionResult, LoadedDocument, Source
from .section_detector import section_at, sections_for_pages

def _clean(value: str) -> str: return re.sub(r"\s+", " ", value).strip(" ;:,.\n")
def _snippet(text: str, start: int, end: int) -> str: return _clean(text[max(0,start-90):min(len(text),end+150)])[:360]
def _first(document, sections, patterns):
    for page in document.pages:
        for pattern in patterns:
            match = re.search(pattern, page.text, re.I | re.S)
            if match:
                value = _clean(match.group(1) if match.lastindex else match.group(0))
                return value, Source(page.page, section_at(sections, page.page, match.start()), _snippet(page.text, match.start(), match.end()))
    return None, None

def extract_rules(document: LoadedDocument) -> ExtractionResult:
    r = ExtractionResult(metadata={"engine": "rules", "llm_used": False}); sections = sections_for_pages(document.pages)
    patterns = {
      "issuer":[r"(?:Наименование\s+эмитента|Эмитент(?:ом\s+является)?)[\s:–-]+([^\n]{2,220})"],
      "security_type":[r"\b(облигаци[яйи][^\n,.]{0,90})"], "series":[r"(?:серия|серии)\s*[№N]?[\s:–-]*([A-Za-zА-Яа-я0-9._/-]{1,60})"],
      "issue_number":[r"(?:государственн(?:ый|ая) регистрационн(?:ый|ая) номер|номер выпуска)[\s:–-]*([A-Za-zА-Яа-я0-9._/-]{4,80})"],
      "isin":[r"\b([A-Z]{2}[A-Z0-9]{9}\d)\b"], "ticker":[r"(?:тикер|код ценной бумаги)[\s:–-]*([A-ZА-Я0-9._-]{2,20})"],
      "quantity":[r"(?:Количество облигаций|Количество ценных бумаг)[^\n\d]{0,80}([\d\s]{1,20})"],
      "nominal_value":[r"(?:Номинальная стоимость|Номинал)[^\n\d]{0,80}([\d\s]+(?:[,.]\d+)?\s*(?:RUB|руб\.?|рублей|российских рублей)?)"],
      "placement_price":[r"(?:Цена размещения(?: одной облигации)?)[^\n\d]{0,80}([\d\s]+(?:[,.]\d+)?\s*(?:RUB|руб\.?|рублей|российских рублей|%)[^\n]{0,40})"],
      "coupon_rate":[r"(?:купонная ставка|размер купона)[^\n\d]{0,80}([\d]+(?:[,.]\d+)?\s*%)"],
      "coupon_period":[r"(?:купонный период)[^\n\d]{0,80}([\d]+\s*(?:дн(?:ей|я)?|месяц(?:ев|а)?|лет))"],
      "maturity_date":[r"(?:дата погашения)[^\n\d]{0,80}(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})"],
      "offer_date":[r"(?:дата (?:оферты|приобретения))[^\n\d]{0,80}(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})"],
      "payment_method":[r"(?:порядок оплаты|оплата)[\s:–-]+([^\n]{4,180})"],
      "settlement":[r"\b(поставка против платежа|DVP)\b"],
      "amortization":[r"\b(амортизация|амортизационные выплаты|частичное погашение)\b"],
      "preemptive_right":[r"(?:преимущественное право)[\s:–-]+([^\n]{2,140})"],
    }
    for field, candidates in patterns.items():
        value, source = _first(document, sections, candidates)
        if value and source: r.parameters[field] = value; r.sources[field] = source; r.confidence[field] = 0.98
    for page in document.pages:
        for phrase, field, val in [("открытая подписка","placement_method","открытая подписка"),("закрытая подписка","placement_method","закрытая подписка"),("книга заявок","book_building","книга заявок"),("book-building","book_building","book-building"),("букбилдинг","book_building","букбилдинг")]:
            m=re.search(re.escape(phrase),page.text,re.I)
            if m and r.parameters.get(field) is None: r.parameters[field]=val; r.sources[field]=Source(page.page,section_at(sections,page.page,m.start()),_snippet(page.text,m.start(),m.end())); r.confidence[field]=0.98
    for page in document.pages:
        m=re.search(r"\b(RUB|руб\.|рублей|российских рублей)\b",page.text,re.I)
        if m: r.parameters["currency"]="RUB"; r.sources["currency"]=Source(page.page,section_at(sections,page.page,m.start()),_snippet(page.text,m.start(),m.end())); r.confidence["currency"]=0.98; break
    return r
