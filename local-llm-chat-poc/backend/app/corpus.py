"""Corpus search: BM25 over the documents in data/corpus.json. No model involved.

To use real documents, replace the list in data/corpus.json. Nothing here changes.
"""

import math
import re
from collections import Counter

STOP_WORDS = set(
    "a an and are as at be by can do does for from has have how if in is it its of on or our say says should "
    "that the their them there this to was we were what when where which who will with would".split()
)
SUFFIX = re.compile(r"(ations?|ating|ated|ates?|ments?|ings?|ed|es|s|e)$")


def stem(word: str) -> str:
    if len(word) < 5 or any(char.isdigit() for char in word):
        return word
    return SUFFIX.sub("", re.sub(r"ies$", "y", word))


def terms(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+(?:/[0-9]+)?", text.lower())
    return [stem(word) for word in words if len(word) > 1 and word not in STOP_WORDS]


class Corpus:
    def __init__(self, documents: list[dict], aliases: dict[str, str] | None = None):
        self.documents = documents
        # Query words that should also find other wording, including common Hindi terms
        self.aliases = aliases or {}
        self._index = []
        for doc in documents:
            words = terms(f"{doc['id']} {doc['title']} {doc['title']} {doc['text']}")  # title counted twice
            self._index.append((doc, Counter(words), len(words)))
        self._average = sum(length for _, _, length in self._index) / max(1, len(self._index))
        self._in_docs = Counter(word for _, counts, _ in self._index for word in counts)

    def get(self, doc_id: str) -> dict | None:
        return next((doc for doc in self.documents if doc["id"] == doc_id), None)

    def search(self, query: str, limit: int = 3, types: list[str] | None = None) -> list[dict]:
        """The best-matching documents, best first. `types` restricts the document types searched."""
        text = query
        for word, also in self.aliases.items():
            if word in query.lower():
                text += f" {also}"
        wanted = set(terms(text))
        hits = []
        for doc, counts, length in self._index:
            if types and doc["type"] not in types:
                continue
            score = 0.0
            for word in wanted:
                count = counts.get(word)
                if not count:
                    continue
                in_docs = self._in_docs[word]
                rarity = math.log(1 + (len(self._index) - in_docs + 0.5) / (in_docs + 0.5))
                score += rarity * count * 2.5 / (count + 1.5 * (0.25 + 0.75 * length / self._average))
            if score > 0:
                hits.append((score, doc))
        hits.sort(key=lambda hit: -hit[0])
        floor = max(1.2, hits[0][0] * 0.35) if hits else 0  # drop weak, incidental matches
        return [doc for score, doc in hits if score >= floor][:limit]


def extracts(documents: list[dict]) -> str:
    """Documents as text for a prompt."""
    return "\n\n".join(f"[{doc['id']}] {doc['title']}\n{doc['text']}" for doc in documents)


def reference(doc: dict) -> dict:
    """What the UI needs to show a document as a source."""
    return {"id": doc["id"], "type": doc["type"], "title": doc["title"]}
