from pathlib import Path
from openpyxl import Workbook
from ..core.models import ExtractionResult, FIELDS
def export_xlsx(path, result: ExtractionResult):
 wb=Workbook(); ws=wb.active; ws.title='Bond analysis'; ws.append(['Parameter','Value','Confidence','Page','Section','Source'])
 for field in FIELDS:
  source=result.sources.get(field); ws.append([field,result.parameters.get(field),result.confidence.get(field),source.page if source else None,source.section if source else None,source.snippet if source else None])
 ws.freeze_panes='A2'; ws.auto_filter.ref=ws.dimensions
 for column,width in {'A':24,'B':34,'C':12,'D':10,'E':30,'F':80}.items():ws.column_dimensions[column].width=width
 Path(path).parent.mkdir(parents=True,exist_ok=True); wb.save(path)
