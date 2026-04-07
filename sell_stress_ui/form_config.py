from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class FormField:
    id: str
    type: str
    label: str
    default: str | None
    required: bool


@dataclass(frozen=True)
class SellStressFormConfig:
    title: str
    description: str
    fields: dict[str, FormField]


def load_form_config(xml_path: str | Path) -> SellStressFormConfig:
    """Load XML UI schema. Kept generic so schema can be extended later."""
    root = ET.parse(xml_path).getroot()
    title = root.findtext("./metadata/title", default="Sell Stress")
    description = root.findtext("./metadata/description", default="")

    fields: dict[str, FormField] = {}
    for node in root.findall(".//field"):
        field = FormField(
            id=node.attrib["id"],
            type=node.attrib.get("type", "text"),
            label=node.attrib.get("label", node.attrib["id"]),
            default=node.attrib.get("default"),
            required=node.attrib.get("required", "false").lower() == "true",
        )
        fields[field.id] = field

    return SellStressFormConfig(title=title, description=description, fields=fields)
