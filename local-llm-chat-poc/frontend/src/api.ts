import type {
  ConversationSummary, Doc, DocumentInfo, DocumentInsights, ExportFormat, ImageEngine, InputValue, ModelsResponse, OutlineResponse, StudioImage,
  RepositoryModel, Situation, StoredConversation, WorkflowResult, WorkflowsResponse,
} from './types';

/** Sent as the model id to let the backend pick a model for each message. */
export const AUTO = 'auto';

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

/** Every model stored in the local runtime, with its size, format and what it can do. */
export const fetchRepository = async (): Promise<RepositoryModel[]> => (await request('/api/repository')).json();

// ---------- saved chats ----------

export const fetchConversations = async (): Promise<ConversationSummary[]> => (await request('/api/conversations')).json();

export const fetchConversation = async (id: string): Promise<StoredConversation> => (await request(`/api/conversations/${id}`)).json();

export const renameConversation = async (id: string, title: string): Promise<void> => {
  await request(`/api/conversations/${id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title }) });
};

export const deleteConversation = async (id: string): Promise<void> => {
  await request(`/api/conversations/${id}`, { method: 'DELETE' });
};

// ---------- uploaded documents ----------

export const fetchDocuments = async (): Promise<DocumentInfo[]> => (await request('/api/documents')).json();

/** Sends a PDF, DOCX or text file; the backend splits it and stores it for document search. */
export const uploadDocument = async (file: File): Promise<DocumentInfo> => {
  const form = new FormData();
  form.append('file', file);
  return (await request('/api/documents', { method: 'POST', body: form })).json();
};

export const deleteDocument = async (id: string): Promise<void> => {
  await request(`/api/documents/${id}`, { method: 'DELETE' });
};

// ---------- AI Studio ----------

/** The text as a PDF, Word or PowerPoint file. */
export const exportFile = async (format: ExportFormat, title: string, content: string): Promise<Blob> =>
  (await request('/api/studio/export', post({ format, title, content }))).blob();

/** How the text will be laid out, for the preview. */
export const fetchOutline = async (format: ExportFormat, title: string, content: string): Promise<OutlineResponse> =>
  (await request('/api/studio/outline', post({ format, title, content }))).json();

export const makeImage = async (prompt: string, style: string, engine: 'model' | 'demo', steps: number): Promise<StudioImage> =>
  (await request('/api/studio/image', post({ prompt, style, engine, steps }))).json();

export const fetchDetectEngine = async (): Promise<{ available: boolean; model: string; reason: string }> => (await request('/api/studio/detect-engine')).json();

/** Boxes around the objects in a picture: the ones named in `objects`, or military and everyday objects. */
export const detectObjects = async (image: string, objects: string) => (await request('/api/studio/detect', post({ image, objects }))).json();

export const fetchImageEngine = async (): Promise<ImageEngine> => (await request('/api/studio/image-engine')).json();

export const fetchInsights = async (id: string): Promise<DocumentInsights> => (await request(`/api/documents/${id}/insights`)).json();

/** Where a file made in the chat is downloaded from. */
export const fileUrl = (id: string) => `${API_BASE}/api/files/${id}`;

/** Offers a file to the viewer as a download. */
export function saveFile(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const link = Object.assign(document.createElement('a'), { href: url, download: name });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

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
