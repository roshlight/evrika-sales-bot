"""CLI entry point: python -m evrika [--preview | --test-bedrock | --usage]."""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv

from evrika.bedrock import Bedrock
from evrika.humanizer import parse_bubbles
from evrika.prompts import build_prompt, load_persona, load_knowledge_base
from evrika.settings import Settings
from evrika.store import Store

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Эврика: AI-консультант EdPalm")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true", help="Print full system prompt without network")
    mode.add_argument("--test-bedrock", action="store_true", help="Make one real Bedrock test request")
    mode.add_argument("--simulate", metavar="TEXT", help="Test persona on one message using Bedrock, no Telegram")
    mode.add_argument("--usage", action="store_true", help="Print global LLM usage counters")
    mode.add_argument("--check-aws", action="store_true", help="Inspect AWS credentials locally, without calling AWS")
    mode.add_argument("--clear-local-memory", action="store_true", help="Delete local dialogue and summary")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    config = Settings.from_env(
        telegram_required=not (
            args.preview or args.test_bedrock or args.simulate is not None
            or args.usage or args.clear_local_memory or args.check_aws
        )
    )
    if args.preview:
        persona = load_persona(ROOT / "persona.json")
        kb = load_knowledge_base(ROOT / "knowledge_base.json")
        print(build_prompt(persona, kb))
        return
    if args.usage or args.clear_local_memory:
        if args.usage:
            ledger = Store(ROOT / "data" / "usage.sqlite3")
            print(ledger.usage())
            ledger.close()
        else:
            count = 0
            for location in (ROOT / "data").glob("chat_[0-9]*.sqlite3"):
                chat = Store(location)
                chat.clear_chat()
                chat.close()
                count += 1
            legacy = ROOT / "data" / "memory.sqlite3"
            if legacy.exists():
                chat = Store(legacy)
                chat.clear_chat()
                chat.close()
                count += 1
            print(f"Cleared conversation data in {count} chat databases; global API counters kept")
        return
    if args.check_aws:
        import os
        bearer = bool(os.getenv('AWS_BEARER_TOKEN_BEDROCK', '').strip())
        key_id = bool(os.getenv('AWS_ACCESS_KEY_ID', '').strip())
        secret = bool(os.getenv('AWS_SECRET_ACCESS_KEY', '').strip())
        if bearer and (key_id or secret):
            print('WARNING: BOTH Bedrock bearer token and IAM credentials are set. Use one method.')
        elif key_id and secret:
            print('IAM access key ID and secret access key both configured (values hidden).')
        elif key_id or secret:
            print('INCOMPLETE IAM credentials: set BOTH AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.')
        elif bearer:
            print('Bedrock bearer token configured (value hidden).')
        elif os.getenv('AWS_PROFILE', '').strip():
            print('AWS_PROFILE is set; boto3 will try to load that profile when called.')
        else:
            print('No credentials in .env. Boto3 might use an external AWS profile or IAM role.')
        print('Region:', config.aws_region)
        print('Model:', config.model_id)
        print('Local configuration only: NO AWS request, NO permission or validity check.')
        return
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.simulate is not None:
        persona = load_persona(ROOT / "persona.json")
        kb = load_knowledge_base(ROOT / "knowledge_base.json")
        model_obj = Bedrock(config.aws_region, config.model_id)
        text, usage = model_obj.generate(
            build_prompt(persona, kb),
            [{"role": "user", "content": [{"text": args.simulate[:4000]}]}],
            config.max_output_tokens, 0.82,
        )
        for i, bubble in enumerate(parse_bubbles(text, config.max_reply_bubbles), 1):
            print(f"Bubble {i}: {bubble}")
        print("Token usage:", usage)
        return
    if args.test_bedrock:
        model_obj = Bedrock(config.aws_region, config.model_id)
        text, usage = model_obj.generate(
            "Answer in Russian, one simple sentence. This is a connectivity test.",
            [{"role": "user", "content": [{"text": "Say hello briefly."}]}],
            70, 0.1,
        )
        print("Connection OK:", text)
        print("Token usage:", usage)
        return
    from evrika.telegram_app import serve
    try:
        asyncio.run(serve(config))
    except KeyboardInterrupt:
        print("Stopped")


if __name__ == "__main__":
    main()