import type { ChatEvent, ModelsResponse } from './types';

// Empty = same origin (the Vite proxy forwards /api to the backend in development).
const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/+$/, '');

/** Turns FastAPI error bodies ({"detail": "..."} or a validation list) into one readable message. */
async function errorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body.detail === 'string') return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d: { msg: string }) => d.msg).join('; ');
  } catch {
    // not JSON
  }
  return `Request failed (HTTP ${response.status})`;
}

export async function fetchModels(): Promise<ModelsResponse> {
  const response = await fetch(`${API_BASE}/api/models`);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

/** Switch models: the backend unloads the other models, then loads this one into memory. */
export async function loadModel(id: string, signal: AbortSignal): Promise<void> {
  const response = await fetch(`${API_BASE}/api/models/${encodeURIComponent(id)}/load`, { method: 'POST', signal });
  if (!response.ok) throw new Error(await errorMessage(response));
}

/**
 * POST /api/chat and call onEvent for every Server-Sent Event.
 * Pass an AbortSignal to stop generation (the backend stops the model too).
 */
export async function streamChat(
  body: { message: string; model?: string; conversation_id?: string | null; thinking?: boolean },
  onEvent: (event: ChatEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) throw new Error(await errorMessage(response));

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    // SSE events are separated by a blank line; lines starting with ":" are keep-alive comments.
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() ?? '';
    for (const block of blocks) {
      for (const line of block.split(/\r?\n/)) {
        if (line.startsWith('data:')) onEvent(JSON.parse(line.slice(5).trim()) as ChatEvent);
      }
    }
  }
}
