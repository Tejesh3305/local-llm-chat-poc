"""A tiny OpenAI-compatible server for testing without a GPU or a real model.

    python scripts/mock_llm_server.py            # serves http://127.0.0.1:11500/v1

Set OLLAMA_BASE_URL=http://127.0.0.1:11500/v1 in backend/.env to use it.
Replies echo the model name, which makes model-switching easy to verify.
"""

import asyncio
import json
import os
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

MODELS = os.getenv("MOCK_MODELS", "gemma4:e4b,nemotron-3-nano:4b").split(",")
PORT = int(os.getenv("MOCK_PORT", "11500"))

app = FastAPI()


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "mock"} for m in MODELS]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    model = body["model"]
    if model not in MODELS:
        return JSONResponse({"error": {"message": f'model "{model}" not found, try pulling it first'}}, status_code=404)

    user_text = next((m["content"] for m in reversed(body["messages"]) if m["role"] == "user"), "")
    words = f"[{model}] You said: {user_text}".split(" ")
    base = {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": int(time.time()), "model": model}

    async def events():
        for i, word in enumerate(words):
            await asyncio.sleep(0.03)
            delta = {"content": word + (" " if i < len(words) - 1 else "")}
            yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]})}\n\n"
        yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        usage = {"prompt_tokens": 10, "completion_tokens": len(words), "total_tokens": 10 + len(words)}
        yield f"data: {json.dumps({**base, 'choices': [], 'usage': usage})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
