"""A stand-in for Ollama, for working on the app without a GPU or a real model.

    python scripts/mock_llm_server.py            # serves http://127.0.0.1:11500

Set OLLAMA_BASE_URL=http://127.0.0.1:11500/v1 in backend/.env to use it.
Replies are canned and say so. They name the model, which makes switching easy to verify.

Environment variables:
    MOCK_PORT=11500
    MOCK_MODELS=gemma4:e2b-it-qat,llama3.2:3b    models reported as installed
    MOCK_LOAD_MS=3000                            pretend time to load a model into memory
    MOCK_TPS=40                                  pretend generation speed, tokens per second
"""

import asyncio
import json
import os
import re
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

MODELS = [name.strip() for name in os.getenv("MOCK_MODELS", "gemma4:e2b-it-qat,llama3.2:3b").split(",") if name.strip()]
PORT = int(os.getenv("MOCK_PORT", "11500"))
LOAD_MS = int(os.getenv("MOCK_LOAD_MS", "3000"))
TPS = float(os.getenv("MOCK_TPS", "40"))

app = FastAPI()
in_memory: str | None = None  # one model at a time, like OLLAMA_MAX_LOADED_MODELS=1


async def load(model: str) -> None:
    global in_memory
    if in_memory != model:
        await asyncio.sleep(LOAD_MS / 1000)
        in_memory = model


def reply_for(model: str, messages: list[dict]) -> str:
    asked = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    # A workflow's model step sends TASK / FACTS / REFERENCES: answer with the first facts, so the draft looks the part.
    facts = re.search(r"\nFACTS\n(.*?)\n\nREFERENCES", asked, re.S)
    if facts:
        return f"**[mock · {model}]** Canned draft, no model is running.\n\n" + "\n".join(facts.group(1).split("\n")[:5])
    if re.search(r"[ऀ-ॿ]", asked):
        return f"**[mock · {model}]** यह एक नमूना उत्तर है, कोई मॉडल नहीं चल रहा है।\n\nआपने पूछा: {asked}"
    earlier = sum(1 for m in messages if m["role"] == "user") - 1
    return (
        f"**[mock · {model}]** Canned reply, no model is running.\n\n"
        f"You asked: {asked}\n\n"
        f"I can see {earlier} earlier message{'' if earlier == 1 else 's'} from you in this chat.\n\n"
        "1. First point with `inline code`\n2. Second point\n\n"
        '```python\nprint("hello from the mock engine")\n```'
    )


# ---------- OpenAI-compatible API (what the backend uses for chat) ----------


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "mock"} for m in MODELS]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    model = body["model"]
    if model not in MODELS:
        return JSONResponse({"error": {"message": f'model "{model}" not found, try pulling it first'}}, status_code=404)

    whole = re.findall(r"\s*\S+", reply_for(model, body["messages"]))
    limit = body.get("max_tokens")
    pieces = whole[:limit] if limit else whole
    ended = "length" if len(pieces) < len(whole) else "stop"
    base = {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": int(time.time()), "model": model}

    async def events():
        await load(model)
        started = time.perf_counter()
        for index, piece in enumerate(pieces, 1):
            # Pace against the clock: very short sleeps return early on Windows and would run far faster than MOCK_TPS
            await asyncio.sleep(max(0.0, started + index / TPS - time.perf_counter()))
            yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'content': piece}, 'finish_reason': None}]})}\n\n"
        yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': ended}]})}\n\n"
        usage = {"prompt_tokens": 20, "completion_tokens": len(pieces), "total_tokens": 20 + len(pieces)}
        yield f"data: {json.dumps({**base, 'choices': [], 'usage': usage})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


# ---------- the parts of Ollama's own API the backend uses ----------


@app.get("/api/ps")
def running():
    return {"models": [{"name": in_memory, "model": in_memory, "size": 3_000_000_000, "size_vram": 3_000_000_000}] if in_memory else []}


@app.post("/api/generate")
async def generate(request: Request):
    global in_memory
    body = await request.json()
    model = body.get("model")
    if model not in MODELS:
        return JSONResponse({"error": f"model '{model}' not found"}, status_code=404)
    if body.get("keep_alive") == 0:  # how Ollama is asked to unload a model
        in_memory = None if in_memory == model else in_memory
        return {"model": model, "response": "", "done": True, "done_reason": "unload"}
    await load(model)
    return {"model": model, "response": "", "done": True, "done_reason": "load"}


if __name__ == "__main__":
    print(f"Mock engine on http://127.0.0.1:{PORT}  (canned replies, not a real model)")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
