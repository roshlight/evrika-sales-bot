"""Turn a single Bedrock reply into restrained, Telegram-style message bubbles.

The model, not a punctuation splitter, chooses where a natural message ends.
This module is pure and safe to exercise without Telegram or AWS.
"""
from __future__ import annotations

import json
import random
from typing import Any


def _extract_json(raw: str) -> Any:
    raw = raw.strip()
    if raw.startswith('```'):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == '```':
            raw = '\n'.join(lines[1:-1]).strip()
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None


def parse_bubbles(raw: str, max_bubbles: int = 3) -> list[str]:
    """Accept {messages:[...]}; fail safely to one text bubble on malformed JSON.

    The output cap prevents a cascade of notifications. No automatic sentence
    splitting: if the model wants one longer message, we keep one message.
    """
    if not isinstance(raw, str) or not raw.strip():
        return []
    max_bubbles = max(1, min(4, max_bubbles))
    payload = _extract_json(raw)
    if isinstance(payload, dict) and isinstance(payload.get('messages'), list):
        values = [x.strip()[:4000] for x in payload['messages'] if isinstance(x, str) and x.strip()]
    elif isinstance(payload, list):
        values = [x.strip()[:4000] for x in payload if isinstance(x, str) and x.strip()]
    elif isinstance(payload, dict) and isinstance(payload.get('message'), str):
        values = [payload['message'].strip()[:4000]]
    else:
        values = [raw.strip()[:4000]]
    if not values:
        return []
    if len(values) > max_bubbles:
        values = values[:max_bubbles - 1] + ['\n'.join(values[max_bubbles - 1:])[:4000]]
    return values


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
