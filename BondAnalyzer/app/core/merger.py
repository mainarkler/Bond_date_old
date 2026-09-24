from .models import ExtractionResult
def merge_results(rules: ExtractionResult, llm: ExtractionResult | None) -> ExtractionResult:
    merged=ExtractionResult(parameters=dict(rules.parameters),sources=dict(rules.sources),confidence=dict(rules.confidence),metadata=dict(rules.metadata))
    if not llm: return merged
    for field,value in llm.parameters.items():
        if field == "special_conditions":
            if not merged.parameters[field] and value: merged.parameters[field]=value
        elif merged.parameters.get(field) is None and value is not None and field in llm.sources:
            merged.parameters[field]=value; merged.sources[field]=llm.sources[field]; merged.confidence[field]=llm.confidence.get(field,.8)
    merged.metadata.update({"llm_used": bool(llm.sources), "engine":"rules + llama.cpp" if llm.sources else "rules"})
    return merged
