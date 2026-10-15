"""Polza chat-completions adapter; called only by explicit terminal mode."""
from __future__ import annotations

import http.client
import json
import os


class Polza:
    def __init__(self):
        self.key = os.environ.get('POLZA_API_KEY', '').strip()
        self.model = os.environ.get('POLZA_MODEL', '').strip()
        if not self.key or not self.model:
            raise ValueError('Set POLZA_API_KEY and POLZA_MODEL locally; never paste keys into chat')

    def generate(self, system, messages, max_tokens, temperature):
        converted = [{'role': 'system', 'content': system}]
        for message in messages:
            converted.append({'role': message['role'], 'content': '\n'.join(
                block['text'] for block in message['content'] if 'text' in block)})
        body = json.dumps({'model': self.model, 'messages': converted,
                           'max_tokens': max_tokens, 'temperature': temperature}, ensure_ascii=False)
        # Fixed HTTPS host, no redirects or retries (avoid forwarding keys or double billing).
        conn = http.client.HTTPSConnection('polza.ai', timeout=60)
        try:
            conn.request('POST', '/api/v1/chat/completions', body.encode('utf-8'),
                         {'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
            response = conn.getresponse()
            if response.status != 200:
                raise RuntimeError(f'Polza returned HTTP {response.status}; response body withheld')
            raw = response.read(1_000_001)
            if len(raw) > 1_000_000:
                raise RuntimeError('Polza response exceeded size limit')
            payload = json.loads(raw)
            choice = payload['choices'][0]
            if choice.get('finish_reason') != 'stop':
                raise RuntimeError('Model response did not finish normally')
            text = choice['message']['content']
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError('Model returned no text')
            usage = payload.get('usage') or {}
            return text, {'inputTokens': int(usage.get('prompt_tokens', 0)),
                          'outputTokens': int(usage.get('completion_tokens', 0))}
        except (OSError, ValueError, KeyError, IndexError, TypeError, http.client.HTTPException):
            raise RuntimeError('Polza connection or response failed; details withheld') from None
        finally:
            conn.close()
