"""Synchronous Bedrock caller; invoke via asyncio.to_thread in Telegram loop."""
from __future__ import annotations

import boto3
from botocore.config import Config

from evrika.prompts import extract_text


class Bedrock:
    def __init__(self, region: str, model_id: str):
        self.model_id = model_id
        self.client = boto3.client(
            "bedrock-runtime", region_name=region,
            config=Config(connect_timeout=10, read_timeout=55, retries={"max_attempts": 2}),
        )

    def generate(self, system: str, messages: list[dict], max_tokens: int, temperature: float) -> tuple[str, dict]:
        result = self.client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=messages,
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
        )
        text = extract_text(result)
        if not text:
            raise RuntimeError("Bedrock response does not contain text")
        return text, result.get("usage", {})