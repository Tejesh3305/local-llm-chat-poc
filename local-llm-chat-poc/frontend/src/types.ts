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
  | { type: 'start'; conversation_id: string; model: string; sources?: DocRef[] }
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
}
