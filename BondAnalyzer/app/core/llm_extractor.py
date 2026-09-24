from .models import LoadedDocument, ExtractionResult
from .validator import parse_llm_json, validate_llm_result
from ..llm.llama_server import LocalLLM
from ..llm.prompts import build_prompt

def chunk_document(document: LoadedDocument, limit: int = 25000) -> list[str]:
    text=document.text
    if len(text)<=limit:return [text]
    chunks=[]; current=""
    for page in document.pages:
        piece=f"[PAGE {page.page}]\n{page.text}\n"
        if current and len(current)+len(piece)>limit: chunks.append(current); current=""
        current+=piece
    if current:chunks.append(current)
    return chunks

def extract_with_llm(document: LoadedDocument, root) -> ExtractionResult | None:
    llm=LocalLLM(root)
    if not llm.available: return None
    merged=ExtractionResult(metadata={"engine":"llama.cpp","llm_used":True})
    try:
      with llm:
       for chunk in chunk_document(document):
        result=validate_llm_result(parse_llm_json(llm.complete(build_prompt(chunk))),document)
        for field,source in result.sources.items():
         if merged.parameters[field] is None: merged.parameters[field]=result.parameters[field]; merged.sources[field]=source; merged.confidence[field]=result.confidence[field]
      return merged
    except (OSError,ValueError,RuntimeError): return None
