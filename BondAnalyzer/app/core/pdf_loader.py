from pathlib import Path
import fitz
from .models import LoadedDocument, PageText

def load_pdf(path: str | Path) -> LoadedDocument:
    path = Path(path)
    with fitz.open(path) as pdf:
        pages = [PageText(index + 1, page.get_text("text").strip()) for index, page in enumerate(pdf)]
    if not any(page.text for page in pages):
        raise ValueError("В документе не найден текстовый слой. Для этого PDF требуется OCR.")
    return LoadedDocument(path.name, pages, {"format": "pdf"})
