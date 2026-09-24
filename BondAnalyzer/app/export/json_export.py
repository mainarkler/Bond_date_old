import json
from pathlib import Path
from ..core.models import ExtractionResult
def export_json(path, filename: str, result: ExtractionResult):
 payload={"document":{"filename":filename},**result.as_dict()}
 Path(path).write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
