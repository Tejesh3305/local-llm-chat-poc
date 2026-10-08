"""Automatic model choice: a few plain rules, so the result is easy to explain in a demo."""

import re

from .config import ModelConfig

# A fenced block, or words that almost only appear in programming questions.
CODE_HINTS = re.compile(
    r"```|c\+\+|c#|\b(code|coding|program|function|script|python|javascript|typescript|java|sql|html|css|regex|"
    r"api|json|bug|debug|exception|stack trace|compile|algorithm|git|bash|powershell)\b",
    re.IGNORECASE,
)


def choose_model(models: list[ModelConfig], default_id: str, message: str) -> tuple[ModelConfig, str]:
    """Pick from `models` (the ones that can be used right now). Returns the model and the reason."""
    if CODE_HINTS.search(message) and (model := next((m for m in models if "code" in m.tags), None)):
        return model, "programming question"
    default = next((m for m in models if m.id == default_id), models[0])
    return default, "general question"
