import json
from ..core.models import FIELDS
SYSTEM_PROMPT="""Ты извлекаешь факты из документа об условиях размещения облигаций. Работай исключительно с переданным текстом, не придумывай значения. Для отсутствующего значения верни null. Верни только JSON с полями параметров и объектом sources. Для каждого найденного значения source обязан содержать page, section и snippet, причём snippet должен быть точной подстрокой документа."""
def build_prompt(text: str) -> str:
    schema={field:None for field in FIELDS}; schema["special_conditions"]=[]; schema["sources"]={}
    return f"{SYSTEM_PROMPT}\nСхема: {json.dumps(schema,ensure_ascii=False)}\n\nДОКУМЕНТ:\n{text}"
