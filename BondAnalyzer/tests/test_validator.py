import pytest
from app.core.models import LoadedDocument,PageText
from app.core.validator import parse_llm_json,validate_llm_result
def test_invalid_json():
 with pytest.raises(ValueError):parse_llm_json('bad')
def test_unverified_source_dropped():
 r=validate_llm_result({'issuer':'АО Тест','sources':{'issuer':{'page':1,'snippet':'invented'}}},LoadedDocument('x',[PageText(1,'Эмитент: АО Тест')]))
 assert r.parameters['issuer'] is None
def test_verified_source_accepted():
 r=validate_llm_result({'issuer':'АО Тест','sources':{'issuer':{'page':1,'section':'1','snippet':'Эмитент: АО Тест'}}},LoadedDocument('x',[PageText(1,'Эмитент: АО Тест')]))
 assert r.parameters['issuer']=='АО Тест'
