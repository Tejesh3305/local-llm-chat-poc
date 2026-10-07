import type { Doc, InputValue, ModelsResponse, Situation, WorkflowResult, WorkflowsResponse } from './types';

// Empty = same origin (the Vite proxy forwards /api to the backend in development).
const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/+$/, '');

export const SERVER_DOWN = 'The app server is not answering. Start it with: python -m app';

export type Inputs = Record<string, InputValue>;

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

async function request(path: string, init?: RequestInit): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch (error) {
    if (init?.signal?.aborted) throw error;
    throw new Error(SERVER_DOWN);
  }
  if (!response.ok) throw new Error(await errorMessage(response));
  return response;
}

const post = (body: unknown, signal?: AbortSignal): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
  signal,
});

export const fetchModels = async (): Promise<ModelsResponse> => (await request('/api/models')).json();

/** Asks the runtime to load a model into memory ahead of the first message. */
export const loadModel = async (id: string): Promise<{ loaded: boolean }> =>
  (await request(`/api/models/${encodeURIComponent(id)}/load`, { method: 'POST' })).json();

export const fetchWorkflows = async (): Promise<WorkflowsResponse> => (await request('/api/workflows')).json();

/** The calculations and references for a workflow, with no model step. */
export const computeWorkflow = async (id: string, inputs: Inputs, signal?: AbortSignal): Promise<{ result: WorkflowResult; situation: Situation }> =>
  (await request(`/api/workflows/${id}/compute`, post({ inputs }, signal))).json();

export const fetchCorpus = async (): Promise<Doc[]> => (await (await request('/api/corpus')).json()).documents;

/**
 * POSTs and yields each Server-Sent Event as it arrives.
 * Aborting the signal stops generation (the backend stops the model too).
 */
export async function* streamEvents<T>(path: string, body: unknown, signal: AbortSignal): AsyncGenerator<T> {
  const response = await request(path, post(body, signal));
  if (!response.body) throw new Error('The server sent no stream.');
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    // SSE events are separated by a blank line; lines starting with ":" are keep-alive comments.
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() ?? '';
    for (const block of blocks) {
      for (const line of block.split(/\r?\n/)) {
        if (line.startsWith('data:')) yield JSON.parse(line.slice(5).trim()) as T;
      }
    }
  }
}
