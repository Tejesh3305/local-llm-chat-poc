"""Document library: read uploaded files, split them into pieces, embed the pieces and
find the ones closest to a question. Everything is stored in the local SQLite file.

The vectors are kept normalised, so the similarity of two pieces is a plain dot product.
A linear scan is fast enough for the few thousand pieces a POC holds.
"""

import asyncio
import io
import math
import re
import threading
import uuid
import zipfile
from array import array
from collections.abc import Awaitable, Callable
from xml.etree import ElementTree

from pypdf import PdfReader

from .schemas import DocumentInfo, Source
from .store import connect, now

# (texts, kind) -> one vector per text; kind is "document" or "query"
Embedder = Callable[[list[str], str], Awaitable[list[list[float]]]]

CHUNK_CHARS = 900
CHUNK_OVERLAP = 150
MAX_CHUNKS = 2000
EMBED_BATCH = 32
SUPPORTED = (".pdf", ".docx", ".txt", ".md")

WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class DocumentError(Exception):
    """A problem with an uploaded file, with a message that is safe to show to users."""


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    try:
        if name.endswith(".pdf"):
            return "\n\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
        if name.endswith(".docx"):
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                root = ElementTree.fromstring(archive.read("word/document.xml"))
            paragraphs = ("".join(t.text or "" for t in p.iter(f"{WORD_NS}t")) for p in root.iter(f"{WORD_NS}p"))
            return "\n\n".join(p for p in paragraphs if p.strip())
        if name.endswith((".txt", ".md")):
            return data.decode("utf-8", errors="replace")
    except DocumentError:
        raise
    except Exception as error:  # damaged or password-protected file
        raise DocumentError(f"'{filename}' could not be read. The file may be damaged or protected.") from error
    raise DocumentError(f"Unsupported file type. Upload one of: {', '.join(SUPPORTED)}")


def split_text(text: str) -> list[str]:
    """Pieces of about CHUNK_CHARS characters that start and end on paragraph or sentence breaks where possible."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        if end < len(text):
            window = text[start:end]
            # Prefer the last paragraph break, then the last sentence end, in the second half of the window.
            cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if cut > CHUNK_CHARS // 2:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP
        # Start the next piece at a word, not in the middle of one.
        space = text.find(" ", start, end)
        if space != -1:
            start = space + 1
    return chunks


def _normalise(vector: list[float]) -> bytes:
    length = math.sqrt(sum(x * x for x in vector)) or 1.0
    return array("f", (x / length for x in vector)).tobytes()


class DocumentLibrary:
    def __init__(self, path: str | None = None) -> None:
        self._db = connect(path)
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, filename TEXT NOT NULL, size_bytes INTEGER NOT NULL,
                    chunks INTEGER NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL, text TEXT NOT NULL, embedding BLOB NOT NULL
                );
                """
            )

    def documents(self) -> list[DocumentInfo]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [DocumentInfo(**dict(row)) for row in rows]

    def is_empty(self) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1 FROM documents LIMIT 1").fetchone() is None

    async def add(self, filename: str, data: bytes, embed: Embedder) -> DocumentInfo:
        """Read, split, embed and store a file. Raises DocumentError for files that cannot be used."""
        text = await asyncio.to_thread(extract_text, filename, data)
        pieces = split_text(text)
        if not pieces:
            raise DocumentError(f"No readable text found in '{filename}'. Scanned pages have no text to search.")
        if len(pieces) > MAX_CHUNKS:
            raise DocumentError(f"'{filename}' is too long for this POC ({len(pieces)} pieces, the limit is {MAX_CHUNKS}).")

        vectors: list[list[float]] = []
        for i in range(0, len(pieces), EMBED_BATCH):
            vectors += await embed(pieces[i : i + EMBED_BATCH], "document")

        info = DocumentInfo(id=uuid.uuid4().hex, filename=filename, size_bytes=len(data), chunks=len(pieces), created_at=now())
        with self._lock, self._db:
            self._db.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?)", (info.id, info.filename, info.size_bytes, info.chunks, info.created_at))
            self._db.executemany(
                "INSERT INTO chunks (document_id, position, text, embedding) VALUES (?, ?, ?, ?)",
                [(info.id, n, piece, _normalise(vector)) for n, (piece, vector) in enumerate(zip(pieces, vectors), start=1)],
            )
        return info

    def delete(self, document_id: str) -> bool:
        with self._lock, self._db:
            return self._db.execute("DELETE FROM documents WHERE id = ?", (document_id,)).rowcount > 0

    async def search(self, query: str, embed: Embedder, limit: int = 4, document_ids: list[str] | None = None) -> list[Source]:
        """The pieces most similar to the query, best first. `document_ids` restricts the search to those documents."""
        if self.is_empty():
            return []
        only = "WHERE c.document_id IN (%s)" % ", ".join("?" * len(document_ids)) if document_ids else ""
        with self._lock:
            rows = self._db.execute(
                f"SELECT c.document_id, d.filename, c.position, c.text, c.embedding FROM chunks c JOIN documents d ON d.id = c.document_id {only}",
                document_ids or [],
            ).fetchall()
        if not rows:  # the chosen documents were removed in the meantime
            return []
        query_vector = array("f", _normalise((await embed([query], "query"))[0]))

        def rank() -> list[Source]:
            scored = []
            for row in rows:
                vector = array("f", row["embedding"])
                if len(vector) == len(query_vector):  # skip pieces embedded with a different model
                    scored.append((sum(a * b for a, b in zip(query_vector, vector)), row))
            scored.sort(key=lambda item: item[0], reverse=True)
            return [
                Source(document_id=r["document_id"], filename=r["filename"], position=r["position"], text=r["text"], score=round(max(s, 0.0), 3))
                for s, r in scored[:limit]
            ]

        return await asyncio.to_thread(rank)


def with_excerpts(question: str, sources: list[Source]) -> str:
    """The question as sent to the model: the excerpts first, then what the user asked."""
    excerpts = "\n\n".join(f"[{n}] (from {s.filename})\n{s.text}" for n, s in enumerate(sources, start=1))
    return (
        "Answer the question using the excerpts from the user's documents below. "
        "Mention the excerpt number, like [1], for what you use. "
        "If the excerpts do not contain the answer, say so instead of guessing.\n\n"
        f"{excerpts}\n\nQuestion: {question}"
    )
