"""Chat messages that ask for a file ("make a Word doc of this", "convert the PDF to PowerPoint").

Plain rules, like the Auto model choice, so the behaviour is predictable in a demo. A message is a
file request when it has an action word and names a format; everything else is ordinary chat.
"""

import difflib
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
# Said about an uploaded document ("the given doc", "this file") once the format words are taken out.
# Only used to ask which document is meant: a document is read only when it is picked with @.
DOCUMENT_WORDS = re.compile(r"\b(docs?|documents?|files?|given|attached|uploaded|pdf|resume|cv)\b", re.I)
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


PICTURE_VERBS = ("create", "generate", "make", "draw", "paint", "sketch", "render", "design", "produce", "imagine")
PICTURE_NOUNS = ("image", "picture", "photo", "photograph", "illustration", "drawing", "painting", "artwork", "poster", "wallpaper", "sketch")
EXACT_VERBS = {"drew", "show", "give", "want", "need"}  # "show me a picture of…", "I want an image of…"
EXACT_NOUNS = {"pic", "pics", "art", "img"}
LINKS = {"of", "showing", "with", "depicting", "for", "about", "that", "shows", "a", "an", "the"}
# "try again", "another one": repeat the previous request
REPEAT = re.compile(r"^\W*(please\s+)?(try\s+again|again|retry|redo|one\s+more|another(\s+one)?|regenerate|new\s+one|different\s+one)\W*(please)?\W*$", re.I)


def _like(word: str, choices: tuple[str, ...], cutoff: float) -> bool:
    """The word, or a misspelling of it: 'genearte' is 'generate', 'imgae' is 'image'."""
    word = word.lower().rstrip("s") if len(word) > 4 else word.lower()
    return any(word == choice or (len(word) >= 4 and difflib.SequenceMatcher(None, word, choice).ratio() >= cutoff) for choice in choices)


@dataclass
class PictureRequest:
    subject: str  # what to draw, e.g. "a cat"
    style: str  # one of the image styles: "photo", "poster", "terrain map", "landscape"


def picture_request(message: str) -> PictureRequest | None:
    """'create the image of cat', 'genearte image of dog', 'picture of a tank': a picture, not text."""
    words = re.findall(r"[A-Za-z]+", message)
    lowered = [word.lower() for word in words]
    noun_at = next((i for i, word in enumerate(lowered) if word in EXACT_NOUNS or _like(word, PICTURE_NOUNS, 0.8)), None)
    if noun_at is None:
        return None
    if lowered and lowered[0] in {"describe", "explain", "what", "which", "read", "analyse", "analyze", "summarise", "summarize", "is", "does", "why", "how"}:
        return None  # a question about a picture, not a request for one
    following = lowered[noun_at + 1] if noun_at + 1 < len(lowered) else ""
    if noun_at and lowered[noun_at - 1] in {"the", "this", "that", "my", "your", "these", "those"} and following not in {"of", "showing", "depicting", "for", "with"}:
        return None  # "make the image brighter": about a picture that already exists
    has_verb = any(word in EXACT_VERBS or _like(word, PICTURE_VERBS, 0.8) for word in lowered[:noun_at] + lowered[noun_at + 1 :])
    # "image of a dog" on its own also asks for one
    starts_with_picture = noun_at <= 2 and noun_at + 1 < len(lowered) and lowered[noun_at + 1] in {"of", "showing", "depicting"}
    if not (has_verb or starts_with_picture):
        return None

    after = words[noun_at + 1 :]
    while after and after[0].lower() in LINKS:
        after = after[1:]
    if after:
        subject = message[message.lower().find(after[0].lower(), sum(len(w) for w in words[: noun_at + 1])) :].strip(" .?!")
    else:  # "a dog image please": the words that are not the request itself
        subject = " ".join(w for w in words if not (w.lower() in EXACT_VERBS or _like(w, PICTURE_VERBS, 0.8) or _like(w, PICTURE_NOUNS, 0.8) or w.lower() in LINKS | {"please", "me"}))
    subject = subject or message.strip()
    words = message.lower()
    style = "poster" if "poster" in words else "terrain map" if re.search(r"\b(map|terrain|topograph)", words) else (
        "landscape" if re.search(r"\b(landscape|mountain|valley|scenery|ridge)", words) else "photo"
    )
    return PictureRequest(subject, style)


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
