"""Defense in depth for the local long-term memory (not a PII guarantee)."""
from __future__ import annotations

import re

# Drop potential contact data and private record-like details from memory.
SENSITIVE = re.compile(
    r"(?:[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})"
    r"|(?:https?://\S+)"
    r"|(?:\+?\d[\d\s\(\)\-]{8,}\d)"
    r"|(?:\b\d{6,}\b)"
    r"|(?:\b(?:passport|policy number|medical record|snils|inn|bank card|home address|"
    r"patient|diagnosis|credit card|otp|password|паспор\w*|снилс|инн|полис\w*|"
    r"адрес\w*|квартир\w*|диагноз\w*|болезн\w*|карт\w*\s+банка)\b)",
    flags=re.IGNORECASE,
)


def suitable_for_long_memory(value: str) -> bool:
    return bool(value and len(value) <= 4000 and not SENSITIVE.search(value))


def safe_memory_items(values: object, max_items: int, max_length: int) -> list[str]:
    if not isinstance(values, list):
        return []
    output: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        value = " ".join(value.split()).strip()
        if len(value) > max_length or not suitable_for_long_memory(value):
            continue
        if value not in output:
            output.append(value)
        if len(output) >= max_items:
            break
    return output
