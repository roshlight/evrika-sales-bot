"""Strict model envelope: invalid output cannot leak raw JSON or CRM internals."""
from __future__ import annotations
import json

FALLBACK = 'Не удалось подготовить корректный ответ. Пожалуйста, повторите вопрос; никаких заявок не отправлено.'
STAGES = {'greeting', 'qualification', 'recommendation', 'objection_handling',
          'closing', 'handoff', 'deferred', 'closed'}
CLIENT_FIELDS = {'grade', 'goal', 'support', 'start', 'constraints'}
SALES_FIELDS = {'recommended_product_id', 'alternative_product_id', 'open_question', 'summary'}


def validate_response(raw: str) -> tuple[list[str], dict]:
    if not isinstance(raw, str) or len(raw) > 30000:
        raise ValueError('Invalid response size')
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {'messages', 'crm_state'}:
        raise ValueError('Expected messages and crm_state')
    messages, state = value['messages'], value['crm_state']
    if not isinstance(messages, list) or not 1 <= len(messages) <= 3:
        raise ValueError('Expected 1–3 messages')
    if any(not isinstance(s, str) or not s.strip() or len(s) > 4000 for s in messages):
        raise ValueError('Invalid message')
    if any(any(ord(c) < 32 and c not in '\n\t' for c in s) for s in messages):
        raise ValueError('Control characters are forbidden')
    required = {'funnel_stage', 'lead_category', 'handoff_requested', 'client_data', 'sales_data'}
    if not isinstance(state, dict) or set(state) != required:
        raise ValueError('Invalid CRM shape')
    if state['funnel_stage'] not in STAGES:
        raise ValueError('Invalid funnel stage')
    if state['lead_category'] not in ('HOT', 'WARM', 'INFORMATIONAL', 'NON-TARGET', None):
        raise ValueError('Invalid lead category')
    if type(state['handoff_requested']) is not bool:
        raise ValueError('Invalid handoff request')
    for name, fields in [('client_data', CLIENT_FIELDS), ('sales_data', SALES_FIELDS)]:
        data = state[name]
        if not isinstance(data, dict) or set(data) != fields:
            raise ValueError('Invalid state fields')
        for key, item in data.items():
            if key.endswith('_product_id'):
                if item is not None and (type(item) is not int or not 1 <= item <= 51):
                    raise ValueError('Invalid product ID')
            elif item is not None and (not isinstance(item, str) or len(item) > 1000):
                raise ValueError('Invalid state value')
    return [s.strip() for s in messages], state


def parse_response(raw: str, max_bubbles: int = 3) -> tuple[list[str], dict | None]:
    try:
        messages, state = validate_response(raw)
        limit = max(1, min(3, max_bubbles))
        if len(messages) > limit:
            messages = messages[:limit - 1] + ['\n'.join(messages[limit - 1:])]
        return messages, state
    except (ValueError, TypeError, KeyError):
        return [FALLBACK], None
