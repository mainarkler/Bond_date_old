from __future__ import annotations
import json, re
from datetime import datetime
from .models import FIELDS, ExtractionResult, LoadedDocument, Source
_DATE_FIELDS={"placement_start","placement_end","maturity_date","offer_date"}; _NUMERIC={"quantity","nominal_value","placement_price","coupon_rate","coupon_period"}
def parse_llm_json(raw: str) -> dict:
    try: data=json.loads(raw)
    except (TypeError,json.JSONDecodeError) as exc: raise ValueError("LLM returned invalid JSON") from exc
    if not isinstance(data,dict): raise ValueError("LLM JSON must be an object")
    return data
def _valid_date(value):
    if not isinstance(value,str): return False
    for fmt in ("%d.%m.%Y","%d/%m/%Y","%d-%m-%Y","%Y-%m-%d"):
        try: datetime.strptime(value,fmt); return True
        except ValueError: continue
    return False
def validate_llm_result(candidate: dict, document: LoadedDocument) -> ExtractionResult:
    result=ExtractionResult(metadata={"engine":"llama.cpp","llm_used":True}); page_map={p.page:p.text for p in document.pages}
    sources=candidate.get("sources",{})
    if not isinstance(sources,dict): return result
    for field in FIELDS:
        value=candidate.get(field)
        if value is None or not isinstance(value,(str,int,float,bool)): continue
        value=str(value).strip()
        if not value or (field in _DATE_FIELDS and not _valid_date(value)): continue
        if field in _NUMERIC and not re.search(r"\d",value): continue
        source=sources.get(field)
        if not isinstance(source,dict) or not isinstance(source.get("page"),int) or not isinstance(source.get("snippet"),str): continue
        page_text=page_map.get(source["page"],""); snippet=source["snippet"].strip()
        if not snippet or snippet not in page_text: continue
        result.parameters[field]=value; result.sources[field]=Source(source["page"], source.get("section") or None, snippet); result.confidence[field]=0.80
    conditions=candidate.get("special_conditions")
    if isinstance(conditions,list) and all(isinstance(x,str) for x in conditions): result.parameters["special_conditions"]=conditions
    return result
