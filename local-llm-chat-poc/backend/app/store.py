"""Conversation storage in one SQLite file, so the chat history survives a restart.

Pass no path to keep everything in memory (used by the tests).
"""

import sqlite3
import threading
import uuid
from datetime import datetime, timezone

from .schemas import Conversation, ConversationSummary, Message


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(path: str | None) -> sqlite3.Connection:
    # One connection shared by the request threads; every use is guarded by a lock.
    db = sqlite3.connect(path or ":memory:", check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    return db


class ConversationStore:
    def __init__(self, path: str | None = None) -> None:
        self._db = connect(path)
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, model TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    data TEXT NOT NULL  -- one Message as JSON
                );
                CREATE INDEX IF NOT EXISTS messages_by_conversation ON messages(conversation_id, id);
                """
            )

    def summaries(self) -> list[ConversationSummary]:
        """All conversations, newest first, without loading their messages."""
        with self._lock:
            rows = self._db.execute(
                "SELECT c.id, c.title, c.model, c.updated_at, COUNT(m.id) AS message_count FROM conversations c "
                "LEFT JOIN messages m ON m.conversation_id = c.id GROUP BY c.id ORDER BY c.updated_at DESC"
            ).fetchall()
        return [ConversationSummary(**dict(row)) for row in rows]

    def get(self, conversation_id: str) -> Conversation | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
            if row is None:
                return None
            messages = self._db.execute("SELECT data FROM messages WHERE conversation_id = ? ORDER BY id", (conversation_id,)).fetchall()
        return Conversation(**dict(row), messages=[Message.model_validate_json(m["data"]) for m in messages])

    def new(self, model: str, conversation_id: str | None = None) -> Conversation:
        """A conversation object that is only stored once its first turn completes."""
        return Conversation(
            id=conversation_id or uuid.uuid4().hex, title="New conversation", model=model, messages=[], created_at=now(), updated_at=now()
        )

    def add_turn(self, conversation: Conversation, user: Message, assistant: Message) -> None:
        if not conversation.messages:
            conversation.title = user.content[:60]
        conversation.messages.extend([user, assistant])
        conversation.model = assistant.model or conversation.model
        conversation.updated_at = now()
        with self._lock, self._db:
            self._save(conversation)
            self._db.executemany(
                "INSERT INTO messages (conversation_id, data) VALUES (?, ?)",
                [(conversation.id, m.model_dump_json(exclude_defaults=True)) for m in (user, assistant)],
            )

    def update(self, conversation: Conversation) -> None:
        """Save a changed title or model."""
        with self._lock, self._db:
            self._save(conversation)

    def delete(self, conversation_id: str) -> bool:
        with self._lock, self._db:
            return self._db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,)).rowcount > 0

    def _save(self, c: Conversation) -> None:
        self._db.execute(
            "INSERT INTO conversations (id, title, model, created_at, updated_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET title = excluded.title, model = excluded.model, updated_at = excluded.updated_at",
            (c.id, c.title, c.model, c.created_at, c.updated_at),
        )
