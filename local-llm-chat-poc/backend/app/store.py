"""In-memory conversation storage. Good enough for a POC; data is lost on restart.

To persist conversations, re-implement these five methods on top of SQLite/Postgres.
"""

import uuid
from datetime import datetime, timezone

from .schemas import Conversation, Message


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConversationStore:
    def __init__(self) -> None:
        self._items: dict[str, Conversation] = {}

    def list(self) -> list[Conversation]:
        return sorted(self._items.values(), key=lambda c: c.updated_at, reverse=True)

    def get(self, conversation_id: str) -> Conversation | None:
        return self._items.get(conversation_id)

    def new(self, model: str, conversation_id: str | None = None) -> Conversation:
        """A conversation object that is only stored once its first turn completes."""
        return Conversation(
            id=conversation_id or uuid.uuid4().hex, title="New conversation", model=model, messages=[], created_at=_now(), updated_at=_now()
        )

    def add_turn(self, conversation: Conversation, user: Message, assistant: Message) -> None:
        if not conversation.messages:
            conversation.title = user.content[:60]
        conversation.messages.extend([user, assistant])
        conversation.model = assistant.model or conversation.model
        conversation.updated_at = _now()
        self._items[conversation.id] = conversation

    def delete(self, conversation_id: str) -> bool:
        return self._items.pop(conversation_id, None) is not None
