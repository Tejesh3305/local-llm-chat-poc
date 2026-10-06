"""Start the API with `python -m app` (host, port and reload come from .env)."""

import uvicorn

from .config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=settings.reload, log_level=settings.log_level)
