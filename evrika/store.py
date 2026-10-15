"""Private local SQLite: chat turns, cumulative memories, requests and flags."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


EMPTY_MEMORY = {"summary": "", "facts": [], "open_threads": [], "tone_notes": []}


class Store:
    def __init__(self, location: str | Path):
        location = Path(location)
        location.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(location)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL CHECK (role IN ('user','assistant')),
            text TEXT NOT NULL,
            telegram_message_id INTEGER UNIQUE,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS seen_telegram (
            telegram_message_id INTEGER PRIMARY KEY
        );
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            summary_json TEXT NOT NULL,
            summarized_until INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS flags (
            name TEXT PRIMARY KEY, value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            requested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'started'
        );
        """)
        self.conn.execute(
            "INSERT OR IGNORE INTO memory(id,summary_json,summarized_until) VALUES (1,?,0)",
            (json.dumps(EMPTY_MEMORY),),
        )
        self.conn.commit()

    def add_user(self, telegram_message_id: int, text: str) -> bool:
        with self.conn:
            seen = self.conn.execute(
                "INSERT OR IGNORE INTO seen_telegram(telegram_message_id) VALUES (?)",
                (telegram_message_id,),
            )
            if seen.rowcount == 0:
                return False
            self.conn.execute(
                "INSERT INTO messages(role,text,telegram_message_id) VALUES ('user',?,?)",
                (text[:4000], telegram_message_id),
            )
        return True

    def add_assistant(self, text: str) -> None:
        self.conn.execute("INSERT INTO messages(role,text) VALUES ('assistant',?)", (text[:4096],))
        self.conn.commit()

    def recent_messages(self, amount: int) -> list[tuple[int, str, str]]:
        rows = self.conn.execute(
            "SELECT id,role,text FROM messages ORDER BY id DESC LIMIT ?", (max(amount, 1),)
        ).fetchall()
        return [(int(row["id"]), row["role"], row["text"]) for row in reversed(rows)]

    def rows_between(self, lo: int, hi: int, max_rows: int = 80) -> list[tuple[int, str, str]]:
        rows = self.conn.execute(
            "SELECT id,role,text FROM messages WHERE id>? AND id<=? ORDER BY id LIMIT ?",
            (lo, hi, max_rows),
        ).fetchall()
        return [(int(row["id"]), row["role"], row["text"]) for row in rows]

    def load_memory(self) -> tuple[dict[str, Any], int]:
        row = self.conn.execute("SELECT summary_json,summarized_until FROM memory WHERE id=1").fetchone()
        return json.loads(row["summary_json"]), int(row["summarized_until"])

    def save_memory(self, value: dict[str, Any], summarized_until: int) -> None:
        old_state, old_marker = self.load_memory()
        if summarized_until < old_marker:
            raise ValueError("Summary checkpoint cannot go backwards")
        with self.conn:
            self.conn.execute(
                "UPDATE memory SET summary_json=?, summarized_until=? WHERE id=1",
                (json.dumps(value, ensure_ascii=False), summarized_until),
            )
            # Keep recent dialogue verbatim; compact and drop old raw text.
            # Seen IDs remain in a separate table to prevent replay duplicates.
            self.conn.execute("DELETE FROM messages WHERE id<=?", (summarized_until,))

    def clear_chat(self) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM messages")
            self.conn.execute("DELETE FROM seen_telegram")
            self.conn.execute("UPDATE memory SET summary_json=?,summarized_until=0 WHERE id=1",
                              (json.dumps(EMPTY_MEMORY),))
        self.set_flag("memory_epoch", str(int(self.get_flag("memory_epoch", "0")) + 1))

    def set_flag(self, name: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO flags(name,value) VALUES (?,?) "
            "ON CONFLICT(name) DO UPDATE SET value=excluded.value", (name, value)
        )
        self.conn.commit()

    def get_flag(self, name: str, default: str = "") -> str:
        row = self.conn.execute("SELECT value FROM flags WHERE name=?", (name,)).fetchone()
        return row["value"] if row else default

    def begin_request(self, kind: str, limit_per_hour: int, limit_total: int) -> int | None:
        hourly = self.conn.execute(
            "SELECT COUNT(*) FROM requests WHERE requested_at >= datetime('now', '-1 hour')"
        ).fetchone()[0]
        total = self.conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        if hourly >= limit_per_hour or total >= limit_total:
            return None
        cur = self.conn.execute("INSERT INTO requests(kind) VALUES (?)", (kind,))
        self.conn.commit()
        return int(cur.lastrowid)

    def complete_request(self, request_id: int, tokens: dict | None = None, status: str = "ok") -> None:
        tokens = tokens or {}
        self.conn.execute(
            "UPDATE requests SET input_tokens=?,output_tokens=?,status=? WHERE id=?",
            (int(tokens.get("inputTokens", 0)), int(tokens.get("outputTokens", 0)), status, request_id),
        )
        self.conn.commit()

    def usage(self) -> dict[str, int]:
        row = self.conn.execute(
            "SELECT COUNT(*) calls, COALESCE(SUM(input_tokens),0) input_tokens, "
            "COALESCE(SUM(output_tokens),0) output_tokens FROM requests"
        ).fetchone()
        return dict(row)

    def close(self) -> None:
        self.conn.close()
