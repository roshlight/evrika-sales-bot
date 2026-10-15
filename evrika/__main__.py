"""Explicit terminal entry point. No arguments print help without API calls."""
from __future__ import annotations
import argparse
import asyncio
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Эврика: терминальный AI-консультант EdPalm")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--chat", action="store_true", help="Interactive Polza dialogue (paid API calls)")
    mode.add_argument("--preview", action="store_true", help="Print public prompt, no API call")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    if args.preview:
        from evrika.prompts import build_prompt, load_persona, load_knowledge_base
        print(build_prompt(load_persona(root / "persona.example.json"), load_knowledge_base(root / "knowledge_base.json")))
    elif args.chat:
        from dotenv import load_dotenv
        from evrika.terminal import chat
        load_dotenv(root / ".env")
        try:
            asyncio.run(chat())
        except KeyboardInterrupt:
            print("Остановлено")
        except (ValueError, RuntimeError) as exc:
            # Do not log exception bodies from provider/network libraries.
            print("Не удалось открыть прототип. Проверьте локальные настройки POLZA_API_KEY и POLZA_MODEL.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
