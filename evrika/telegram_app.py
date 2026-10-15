"""Legacy compatibility imports. Telegram deliberately disabled for terminal iteration."""
from pathlib import Path
from evrika.engine import Engine, ROOT

def chat_database_file(user_id):
    if type(user_id) is not int or user_id <= 0:
        raise ValueError("Expected positive Telegram user ID")
    return ROOT / "data" / f"chat_{user_id}.sqlite3"

async def serve(config):
    raise RuntimeError("Telegram is not enabled. Use python -m evrika --chat for the terminal prototype.")
