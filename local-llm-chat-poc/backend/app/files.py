"""Files made in the chat (a Word copy of a document, a PDF of an answer), kept so the chat can link to them.

Each file lives in its own folder named by a random id: storage/files/<id>/<file name>.
"""

import shutil
import tempfile
import uuid
from pathlib import Path


class GeneratedFiles:
    def __init__(self, folder: str | None = None) -> None:
        self.folder = Path(folder) if folder else Path(tempfile.mkdtemp(prefix="sainik-files-"))
        self.folder.mkdir(parents=True, exist_ok=True)

    def save(self, data: bytes, filename: str, fmt: str) -> dict:
        file_id = uuid.uuid4().hex
        target = self.folder / file_id
        target.mkdir()
        (target / filename).write_bytes(data)
        return {"id": file_id, "filename": filename, "format": fmt, "size_bytes": len(data)}

    def get(self, file_id: str) -> Path | None:
        folder = self.folder / file_id
        if not folder.is_dir():
            return None
        return next(folder.iterdir(), None)

    def delete(self, file_id: str) -> None:
        shutil.rmtree(self.folder / file_id, ignore_errors=True)
