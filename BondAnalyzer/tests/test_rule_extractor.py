from app.core.models import LoadedDocument,PageText
from app.core.rule_extractor import extract_rules
def test_rules_capture_values_and_sources():
 doc=LoadedDocument('x.pdf',[PageText(1,'1. Общие положения\nЭмитент: АО Тест\nКоличество облигаций: 1 000\nНоминальная стоимость: 1000 руб.\n'),PageText(2,'2. Порядок размещения\nЦена размещения одной облигации: 1000 RUB\nоткрытая подписка\nпоставка против платежа')])
 result=extract_rules(doc); assert result.parameters['issuer']=='АО Тест'; assert result.parameters['quantity']=='1 000'; assert result.parameters['placement_method']=='открытая подписка'; assert result.sources['placement_price'].page==2
def test_missing_value_is_null(): assert extract_rules(LoadedDocument('x',[PageText(1,'без реквизитов')])).parameters['isin'] is None
