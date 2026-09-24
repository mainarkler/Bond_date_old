"""Best-effort Russian numbered-heading recognition."""
import re
from .models import PageText
_HEADING = re.compile(r"^\s*(\d+(?:\.\d+){0,4}\.?\s+[^\n]{3,120})\s*$", re.M)
def sections_for_pages(pages: list[PageText]) -> dict[int, list[tuple[int, str]]]:
    result = {}
    for page in pages:
        result[page.page] = [(m.start(), re.sub(r"\s+", " ", m.group(1)).strip()) for m in _HEADING.finditer(page.text)]
    return result
def section_at(sections: dict[int, list[tuple[int, str]]], page: int, offset: int) -> str | None:
    candidates = [title for start, title in sections.get(page, []) if start <= offset]
    return candidates[-1] if candidates else None
