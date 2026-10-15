"""Turn a single Bedrock reply into Telegram-style message bubbles + extract CRM state.

The model returns JSON with both "messages" (text bubbles) and "crm_state" (analytics).
This module is pure and safe to exercise without Telegram or AWS.
"""
from __future__ import annotations

import json
import random
from typing import Any


FALLBACK_MESSAGE = 'Одну секунду, я уточню информацию и вернусь к вам!'


def _extract_json(raw: str) -> Any:
    raw = raw.strip()
    if raw.startswith('```'):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == '```':
            raw = '\n'.join(lines[1:-1]).strip()
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        pass
    # Model sometimes adds text around the object: take the outermost {...}.
    start, end = raw.find('{'), raw.rfind('}')
    if 0 <= start < end:
        try:
            return json.loads(raw[start:end + 1])
        except (ValueError, TypeError):
            return None
    return None


def _looks_like_json(raw: str) -> bool:
    text = raw.strip().lstrip('`').strip()
    return text.startswith('{') or '"messages"' in raw or '"crm_state"' in raw


def parse_response(raw: str, max_bubbles: int = 3) -> tuple[list[str], dict | None]:
    """Parse full model response into (bubbles, crm_state).

    Returns (list_of_message_strings, crm_state_dict_or_None).
    Falls back gracefully on malformed JSON.
    """
    if not isinstance(raw, str) or not raw.strip():
        return [], None

    max_bubbles = max(1, min(4, max_bubbles))
    payload = _extract_json(raw)
    crm_state: dict | None = None

    if isinstance(payload, dict):
        crm_state = payload.get('crm_state')
        if not isinstance(crm_state, dict):
            crm_state = None

        if isinstance(payload.get('messages'), list):
            values = [x.strip()[:4000] for x in payload['messages']
                      if isinstance(x, str) and x.strip()]
        elif isinstance(payload.get('message'), str):
            values = [payload['message'].strip()[:4000]]
        else:
            values = [raw.strip()[:4000]]
    elif isinstance(payload, list):
        values = [x.strip()[:4000] for x in payload if isinstance(x, str) and x.strip()]
    elif _looks_like_json(raw):
        # Broken or truncated JSON must never reach the client as raw text.
        values = [FALLBACK_MESSAGE]
    else:
        values = [raw.strip()[:4000]]

    if not values:
        return [], crm_state

    if len(values) > max_bubbles:
        values = values[:max_bubbles - 1] + ['\n'.join(values[max_bubbles - 1:])[:4000]]

    return values, crm_state


def parse_bubbles(raw: str, max_bubbles: int = 3) -> list[str]:
    """Backward-compatible: return only message bubbles."""
    bubbles, _ = parse_response(raw, max_bubbles)
    return bubbles


def typing_seconds(
    text: str, chars_per_second: float = 14.0,
    min_seconds: float = 1.2, max_seconds: float = 7.0,
) -> float:
    """Jittered delay proportional to text length, hard-limited for sanity."""
    chars_per_second = max(1.0, chars_per_second)
    min_seconds = max(0.1, min_seconds)
    max_seconds = max(min_seconds, max_seconds)
    calculated = len(text) / chars_per_second * random.uniform(0.80, 1.25)
    return max(min_seconds, min(max_seconds, calculated + random.uniform(0.25, 0.85)))


def pre_message_pause(index: int) -> float:
    """Short thinking pause or an even shorter gap between message bubbles."""
    return random.uniform(0.5, 1.6) if index == 0 else random.uniform(0.4, 1.4)