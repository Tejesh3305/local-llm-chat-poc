// Shapes returned by the FastAPI backend, and the few the UI keeps for itself.

export type Tone = 'good' | 'warn' | 'bad' | '';

// ---------- models ----------

export interface ModelInfo {
  id: string;
  label: string;
  description: string;
  vendor: string;
  licence: string;
  accent: string;
  provider: string;
  provider_name: string;
  provider_model: string;
  local: boolean; // the runtime's address is this computer
  is_default: boolean;
  available: boolean;
  loaded: boolean | null; // null: the runtime cannot be asked what is in memory
  gpu_percent: number | null;
  thinking: boolean | null; // default for the Think switch; null: the model has no such switch
  tags: string[]; // 'vision', 'code'
  use_cases: string[];
}

/** A model stored in the local runtime, whether or not the catalogue uses it. */
export interface RepositoryModel {
  name: string;
  provider: string;
  size_bytes: number | null;
  parameters: string | null;
  quantization: string | null;
  family: string | null;
  context_length: number | null;
  capabilities: string[];
  loaded: boolean;
  memory_bytes: number | null;
  role: 'chat' | 'embedding' | 'unused';
  catalogue_id: string | null;
}

export interface ModelsResponse {
  default_model: string;
  models: ModelInfo[];
  providers: Record<string, 'ok' | 'unreachable' | 'not_configured'>;
}

// ---------- corpus ----------

export interface DocRef {
  id: string;
  type: string;
  title: string;
}

export interface Doc extends DocRef {
  text: string;
}

// ---------- uploaded documents ----------

export interface DocumentInfo {
  id: string;
  filename: string;
  size_bytes: number;
  chunks: number;
  created_at: string;
}

/** A piece of an uploaded document that was given to the model for an answer. */
export interface Excerpt {
  document_id: string;
  filename: string;
  position: number;
  text: string;
  score: number;
}

// ---------- AI Studio ----------

export type ExportFormat = 'pdf' | 'docx' | 'pptx';

/** A file made in the chat, downloadable from /api/files/{id}. */
export interface GeneratedFile {
  id: string;
  filename: string;
  format: ExportFormat;
  size_bytes: number;
}

export interface OutlineResponse {
  title: string;
  sections: { heading: string; blocks: ['p' | 'bullet', string][] }[];
  slides: { kind: 'title' | 'content' | 'end'; title: string; subtitle?: string; points?: string[] }[];
  filename: string;
}

export interface DemoImage {
  svg: string;
  style: string;
  palette: string;
  seed: number;
  width: number;
  height: number;
}

export interface DocumentInsights {
  words: number;
  characters: number;
  sentences: number;
  reading_minutes: number;
  keywords: string[];
  dates: string[];
  numbers: number;
}

// Server-Sent Events sent by the AI Studio routes that use a model
export type StudioEvent =
  | { type: 'start'; model: string; task?: string; read_percent?: number }
  | { type: 'delta'; text: string }
  | ({ type: 'done' } & Stats)
  | { type: 'error'; message: string };

// ---------- saved chats ----------

export interface ConversationSummary {
  id: string;
  title: string;
  model: string;
  message_count: number;
  updated_at: string;
}

export interface StoredMessage {
  role: 'user' | 'assistant';
  content: string;
  model: string | null;
  documents: string[]; // user message: the documents picked with @, by file name
  route_reason: string | null;
  grounded: boolean;
  sources: DocRef[] | null;
  excerpts: Excerpt[];
  stats: Partial<Stats> | null;
  files?: GeneratedFile[];
}

export interface StoredConversation {
  id: string;
  title: string;
  model: string;
  messages: StoredMessage[];
}

// ---------- streaming ----------

export interface Stats {
  finish_reason: string | null;
  input_tokens: number | null;
  output_tokens: number | null;
  first_token_ms: number | null;
  generation_ms: number | null;
}

// Server-Sent Events sent by POST /api/chat
export type ChatEvent =
  | { type: 'start'; conversation_id: string; model: string; sources?: DocRef[]; route_reason?: string; action?: { type: 'file'; format: ExportFormat; label: string } }
  | { type: 'excerpts'; excerpts: Excerpt[] }
  | { type: 'file'; file: GeneratedFile }
  | { type: 'delta'; text: string }
  | ({ type: 'done' } & Stats)
  | { type: 'error'; message: string };

// Server-Sent Events sent by POST /api/workflows/{id}/run
export type RunEvent =
  | { type: 'result'; result: WorkflowResult; situation: Situation }
  | { type: 'start'; model: string }
  | { type: 'delta'; text: string }
  | ({ type: 'done'; skipped?: boolean } & Partial<Stats>)
  | { type: 'error'; message: string };

// ---------- workflows ----------

export type InputValue = string | number | string[];

export interface Field {
  path: string;
  label: string;
  type: 'number' | 'select' | 'time' | 'text' | 'chips';
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  options: string[];
  value: InputValue;
}

export interface Step {
  label: string;
  detail: string;
  kind: string;
}

export type Cell = string | { text: string; tone: Tone };

export interface Card {
  title: string;
  tiles?: { value: string; label: string; tone?: Tone }[];
  strip?: { label: string; good: boolean; title: string }[];
  note?: string;
  ranked?: [string, string][];
  table?: { head: string[]; rows: Cell[][] };
  pre?: string;
  list?: string[];
}

export interface WorkflowResult {
  steps: Step[];
  cards: Card[];
  refs: DocRef[];
  standard: string; // the brief in its fixed format, built from the calculations alone
}

export interface Pair {
  label: string;
  value: string;
  tone: Tone;
}

export interface Situation {
  as_of: string;
  indicators: Pair[];
  reports: { time: string; text: string; source: string; urgent?: boolean }[];
  forces: Pair[];
  logistics: Pair[];
}

export interface WorkflowInfo {
  id: string;
  letter: string;
  label: string;
  summary: string;
  inputs: Field[];
  brief: { enabled: boolean; title: string };
  plan: { label: string; kind: string }[];
}

export interface WorkflowsResponse {
  scenario: { name: string; title: string; as_of: string; questions: string[] };
  languages: string[];
  workflows: WorkflowInfo[];
  situation: Situation;
  corpus_count: number;
}

// ---------- UI ----------

/** One model reply on screen: a chat message or a workflow draft. Mutated while it streams; `rev` counts changes. */
export interface Reply {
  id: number;
  rev: number;
  model: ModelInfo;
  content: string;
  status: 'waiting' | 'streaming' | 'done' | 'stopped' | 'cancelled' | 'error';
  error: string;
  waitLabel: string;
  cold: boolean; // the model was not in memory when the request went out
  startedAt: number;
  pieces: number;
  firstAt: number;
  firstWordSec: number | null;
  tokens: number | null;
  tokensPerSec: number | null;
  approx: boolean;
  cut: boolean;
  sources?: DocRef[];
  routeReason?: string; // why the Auto choice picked this model
  excerpts?: Excerpt[]; // pieces of the uploaded documents that were used
  files?: GeneratedFile[]; // files made for this reply
}
