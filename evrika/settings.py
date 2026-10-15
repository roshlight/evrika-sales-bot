"""Typed configuration for allowlisted Telegram chats and Bedrock."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int, minimum: int = 1, maximum: int = 100000) -> int:
    raw = os.getenv(name, str(default))
    try:
        number = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return number


def _float(name: str, default: float, minimum: float = 0, maximum: float = 60) -> float:
    raw = os.getenv(name, str(default))
    try:
        number = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return number


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, 'true' if default else 'false').strip().lower()
    if raw in ('true', '1', 'yes', 'on'):
        return True
    if raw in ('false', '0', 'no', 'off', ''):
        return False
    raise ValueError(f"{name} must be true or false")


def parse_targets(usernames: str, ids: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    parsed_names: list[str] = []
    for chunk in usernames.split(','):
        name = chunk.strip().lstrip('@')
        if not name:
            continue
        if len(name) > 32 or not all(c.isascii() and (c.isalnum() or c == '_') for c in name):
            raise ValueError(f"Invalid Telegram username {name!r} in TARGET_USERNAMES")
        if name.lower() not in [old.lower() for old in parsed_names]:
            parsed_names.append(name)
    parsed_ids: list[int] = []
    for chunk in ids.split(','):
        raw = chunk.strip()
        if not raw:
            continue
        try:
            number = int(raw)
        except ValueError as exc:
            raise ValueError("TARGET_USER_IDS must be comma-separated positive integers") from exc
        if number <= 0:
            raise ValueError("TARGET_USER_IDS must be comma-separated positive integers")
        if number not in parsed_ids:
            parsed_ids.append(number)
    return tuple(parsed_names), tuple(parsed_ids)


@dataclass(frozen=True)
class Settings:
    api_id: int
    api_hash: str
    target_usernames: tuple[str, ...]
    target_user_ids: tuple[int, ...]
    allow_all_private_chats: bool
    aws_region: str
    model_id: str
    max_output_tokens: int
    max_calls_hour: int
    max_calls_total: int
    recent_context: int
    summary_trigger: int
    debounce_seconds: float
    max_reply_bubbles: int
    typing_chars_per_second: float
    typing_min_seconds: float
    typing_max_seconds: float
    log_level: str

    @classmethod
    def from_env(cls, telegram_required: bool = True) -> "Settings":
        # Legacy singular fields remain supported when upgrading from v2.
        names, ids = parse_targets(
            ','.join(filter(None, [os.getenv('TARGET_USERNAMES', ''), os.getenv('TARGET_USERNAME', '')])),
            ','.join(filter(None, [os.getenv('TARGET_USER_IDS', ''), os.getenv('TARGET_USER_ID', '')])),
        )
        allow_all = _bool('ALLOW_ALL_PRIVATE_CHATS')
        api_hash = os.getenv('TELEGRAM_API_HASH', '').strip()
        if telegram_required and not (names or ids or allow_all):
            raise ValueError(
                "Set TARGET_USERNAMES (e.g. you,friend), TARGET_USER_IDS, "
                "or explicitly set ALLOW_ALL_PRIVATE_CHATS=true for testing"
            )
        if telegram_required and not api_hash:
            raise ValueError('Set TELEGRAM_API_HASH in .env')
        api_id = _int('TELEGRAM_API_ID', 0, minimum=0)
        if telegram_required and api_id == 0:
            raise ValueError('Set TELEGRAM_API_ID in .env')
        typing_min = _float('TYPING_MIN_SECONDS', 1.2, 0.1, 20)
        typing_max = _float('TYPING_MAX_SECONDS', 7.0, 0.1, 60)
        if typing_max < typing_min:
            raise ValueError('TYPING_MAX_SECONDS cannot be below TYPING_MIN_SECONDS')
        return cls(
            api_id=api_id, api_hash=api_hash,
            target_usernames=names, target_user_ids=ids,
            allow_all_private_chats=allow_all,
            aws_region=os.getenv('AWS_REGION', 'eu-north-1').strip(),
            model_id=os.getenv('BEDROCK_MODEL_ID', 'eu.anthropic.claude-sonnet-4-6').strip(),
            max_output_tokens=_int('MAX_OUTPUT_TOKENS', 1800, 64, 4096),
            max_calls_hour=_int('MAX_MODEL_CALLS_PER_HOUR', 30, 1, 10000),
            max_calls_total=_int('MAX_MODEL_CALLS_TOTAL', 300, 1, 100000),
            recent_context=_int('RECENT_CONTEXT_MESSAGES', 24, 6, 100),
            summary_trigger=_int('SUMMARY_TRIGGER_MESSAGES', 12, 2, 100),
            debounce_seconds=_float('DEBOUNCE_SECONDS', 2.4, 0.1, 15),
            max_reply_bubbles=_int('MAX_REPLY_BUBBLES', 3, 1, 4),
            typing_chars_per_second=_float('TYPING_CHARS_PER_SECOND', 14.0, 1, 60),
            typing_min_seconds=typing_min, typing_max_seconds=typing_max,
            log_level=os.getenv('LOG_LEVEL', 'INFO').upper(),
        )
