"""Chat messages that ask for a file ("make a Word doc of this", "convert the PDF to PowerPoint").

Plain rules, like the Auto model choice, so the behaviour is predictable in a demo. A message is a
file request when it has an action word and names a format; everything else is ordinary chat.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from .studio import FORMATS

ACTION = re.compile(r"\b(create|crate|make|generate|convert|export|turn|save|build|prepare|produce|draft|write|give me|need|want)\b", re.I)
# Copy the text across as it is, instead of having the model write something new
AS_IS = re.compile(r"\b(convert|export|save|turn|change|transform|copy)\b", re.I)
REWRITE = re.compile(r"\b(summar\w*|brief\w*|key points|short\w*|outline|highlights|overview|translate\w*|notes)\b", re.I)

FORMAT_WORDS = {
    "pptx": re.compile(r"\b(power\s?point|pptx?|slides?|slide\s?deck|deck|presentation)\b", re.I),
    "docx": re.compile(r"\.?\bdocx?\b|\bms[- ]?word\b|\bword\s*(doc|docs|document|file|format)s?\b|\b(in|to|as|into)\s+(a\s+)?word\b|\.word\b", re.I),
    # Not ".pdf": "Liam_Dao.pdf" is the name of the source, not a request for a PDF
    "pdf": re.compile(r"\b(to|as|into|in|a|an|new)\s+pdf\b|\bpdf\s*(file|report|document|version|format)\b", re.I),
}
# Said about an uploaded document ("the given doc", "this file") once the format words are taken out
DOCUMENT_WORDS = re.compile(r"\b(docs?|documents?|files?|given|attached|uploaded|pdf|it)\b", re.I)
EARLIER_ANSWER = re.compile(r"\b(above|previous|last (answer|reply|response)|your (answer|reply|response)|this (answer|reply|response|chat|conversation))\b", re.I)

FORMAT_LABELS = {"pdf": "PDF", "docx": "Word document", "pptx": "PowerPoint deck"}


@dataclass
class FileRequest:
    format: str  # "pdf" | "docx" | "pptx"
    as_is: bool  # copy the source text unchanged; otherwise the model writes the content
    mentions_document: bool  # the message talks about a document ("the given doc")
    mentions_answer: bool  # the message talks about an earlier answer ("the above")

    @property
    def label(self) -> str:
        return FORMAT_LABELS[self.format]


def file_request(message: str) -> FileRequest | None:
    if not ACTION.search(message):
        return None
    found = [(match.start(), fmt) for fmt, pattern in FORMAT_WORDS.items() if (match := pattern.search(message))]
    if not found:
        return None
    fmt = found[0][1] if len(found) == 1 else _target(message, found)
    rest = message
    for pattern in FORMAT_WORDS.values():
        rest = pattern.sub(" ", rest)
    mentions_document, mentions_answer = bool(DOCUMENT_WORDS.search(rest)), bool(EARLIER_ANSWER.search(message))
    # "Convert it", or "a Word doc of the given doc": the same text in another format. A summary, notes and
    # the like are written by the model, and so is a deck: a document copied onto slides as it is reads badly.
    as_is = fmt != "pptx" and not REWRITE.search(message) and bool(AS_IS.search(message) or mentions_document or mentions_answer)
    return FileRequest(fmt, as_is, mentions_document, mentions_answer)


def _target(message: str, found: list[tuple[int, str]]) -> str:
    """'convert the pdf file to word' names two formats: the one after the last to/into/as is the one wanted."""
    marks = list(re.finditer(r"\b(to|into|as)\b", message, re.I))
    if marks:
        tail = message[marks[-1].start():]
        hits = [(match.start(), fmt) for fmt, pattern in FORMAT_WORDS.items() if (match := pattern.search(tail))]
        if hits:
            return min(hits)[1]
    return max(found)[1]


def stem(filename: str) -> str:
    return Path(filename).stem.replace("_", " ").replace("-", " ").strip() or "Document"


def media_type(fmt: str) -> str:
    return FORMATS[fmt]
