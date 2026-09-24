from pathlib import Path
from .models import LoadedDocument
from .pdf_loader import load_pdf
from .docx_loader import load_docx

def load_document(path: str | Path) -> LoadedDocument:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf": return load_pdf(path)
    if suffix == ".docx": return load_docx(path)
    raise ValueError("Поддерживаются только файлы PDF и DOCX.")
