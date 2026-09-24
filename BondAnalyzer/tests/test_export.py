import json
from openpyxl import load_workbook
from app.core.models import ExtractionResult,Source
from app.export.json_export import export_json
from app.export.xlsx_export import export_xlsx
def test_exports(tmp_path):
 r=ExtractionResult(); r.parameters['issuer']='АО Тест'; r.sources['issuer']=Source(1,'1. Общие','Эмитент: АО Тест'); r.confidence['issuer']=.98
 jp=tmp_path/'result.json'; xp=tmp_path/'result.xlsx'; export_json(jp,'bond.pdf',r); export_xlsx(xp,r)
 assert json.loads(jp.read_text())['document']['filename']=='bond.pdf'; assert load_workbook(xp).active['A2'].value=='issuer'
