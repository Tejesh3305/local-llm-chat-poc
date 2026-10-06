// Stand-in for Ollama, for working on the chat page on a machine with no GPU or models.
// Replies are canned and say so. No dependencies:
//
//     node dev/mock-ollama.mjs            # listens on http://127.0.0.1:11434
//
// Environment variables:
//     PORT=11434                          port to listen on
//     MOCK_MODELS=gemma4:e2b-it-qat,llama3.2:3b   models reported as installed
//     MOCK_LOAD_MS=3000                   pretend time to load a model into memory
//     MOCK_TPS=40                         pretend generation speed, tokens per second
//     MOCK_NO_CORS=1                      refuse browser requests, like Ollama without OLLAMA_ORIGINS

import http from 'node:http';

const PORT = Number(process.env.PORT || 11434);
const MODELS = (process.env.MOCK_MODELS || 'gemma4:e2b-it-qat,llama3.2:3b').split(',').map(s => s.trim()).filter(Boolean);
const LOAD_MS = Number(process.env.MOCK_LOAD_MS || 3000);
const TPS = Number(process.env.MOCK_TPS || 40);
const NO_CORS = process.env.MOCK_NO_CORS === '1';

let inMemory = null;   // one model at a time, like OLLAMA_MAX_LOADED_MODELS=1

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function readJson(request) {
  return new Promise(resolve => {
    let raw = '';
    request.on('data', chunk => { raw += chunk; });
    request.on('end', () => {
      try { resolve(JSON.parse(raw || '{}')); } catch { resolve({}); }
    });
  });
}

function json(response, status, body) {
  response.writeHead(status, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify(body));
}

async function load(model) {
  if (inMemory === model) return 0;
  await sleep(LOAD_MS);
  inMemory = model;
  return LOAD_MS;
}

function replyFor(model, messages) {
  const question = [...messages].reverse().find(m => m.role === 'user')?.content || '';
  const earlier = messages.filter(m => m.role === 'user').length - 1;
  if (/[ऀ-ॿ]/.test(question)) {
    return `**[mock · ${model}]** यह एक नमूना उत्तर है, कोई मॉडल नहीं चल रहा है।\n\nआपने पूछा: ${question}`;
  }
  return `**[mock · ${model}]** Canned reply, no model is running.\n\n` +
    `You asked: ${question}\n\n` +
    `I can see ${earlier} earlier message${earlier === 1 ? '' : 's'} from you in this chat.\n\n` +
    '1. First point with `inline code`\n2. Second point\n\n' +
    '```python\nprint("hello from the mock engine")\n```';
}

const server = http.createServer(async (request, response) => {
  const { pathname } = new URL(request.url, 'http://localhost');

  if (NO_CORS && request.headers.origin) {
    response.writeHead(403);
    return response.end();
  }
  response.setHeader('Access-Control-Allow-Origin', '*');
  response.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  response.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  if (request.method === 'OPTIONS') {
    response.writeHead(204);
    return response.end();
  }

  if (pathname === '/') return response.end('Ollama is running (mock)');
  if (pathname === '/api/version') return json(response, 200, { version: 'mock' });
  if (pathname === '/api/tags') return json(response, 200, { models: MODELS.map(name => ({ name, model: name, size: 3e9 })) });
  if (pathname === '/api/ps') {
    return json(response, 200, { models: inMemory ? [{ name: inMemory, model: inMemory, size: 3e9, size_vram: 3e9 }] : [] });
  }
  if (pathname === '/v1/models') return json(response, 200, { object: 'list', data: MODELS.map(id => ({ id, object: 'model' })) });

  if (request.method !== 'POST') return json(response, 404, { error: 'not found' });
  const body = await readJson(request);
  const model = body.model;
  const openai = pathname === '/v1/chat/completions';
  if (!MODELS.includes(model)) {
    const message = `model '${model}' not found`;
    return json(response, 404, openai ? { error: { message } } : { error: message });
  }

  if (pathname === '/api/generate') {
    await load(model);
    return json(response, 200, { model, response: '', done: true, done_reason: 'load' });
  }

  if (pathname === '/api/chat' || openai) {
    let open = true;
    response.on('close', () => { open = false; });
    const loadMs = await load(model);
    const pieces = replyFor(model, body.messages || []).match(/\s*\S+/g);
    response.writeHead(200, { 'Content-Type': openai ? 'text/event-stream' : 'application/x-ndjson' });
    const write = data => response.write(openai ? `data: ${JSON.stringify(data)}\n\n` : `${JSON.stringify(data)}\n`);

    for (const piece of pieces) {
      await sleep(1000 / TPS);
      if (!open) return;   // the page pressed Stop
      write(openai
        ? { choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] }
        : { model, message: { role: 'assistant', content: piece }, done: false });
    }
    if (openai) {
      write({ choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] });
      write({ choices: [], usage: { prompt_tokens: 20, completion_tokens: pieces.length } });
      response.write('data: [DONE]\n\n');
    } else {
      write({
        model, message: { role: 'assistant', content: '' }, done: true, done_reason: 'stop',
        load_duration: loadMs * 1e6, prompt_eval_count: 20,
        eval_count: pieces.length, eval_duration: Math.round((pieces.length / TPS) * 1e9),
      });
    }
    return response.end();
  }

  json(response, 404, { error: 'not found' });
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`Mock engine on http://127.0.0.1:${PORT}  (canned replies, not a real model)`);
  console.log(`Models: ${MODELS.join(', ')}${NO_CORS ? '  ·  browser requests refused' : ''}`);
});
