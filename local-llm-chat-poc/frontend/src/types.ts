// Shapes returned by the FastAPI backend.

export interface ModelInfo {
  id: string;
  label: string;
  description: string;
  provider: string;
  is_default: boolean;
  available: boolean;
  /** In memory right now. */
  loaded: boolean;
  /** Default for the Think switch; null when the model has no such switch. */
  thinking: boolean | null;
}

export interface Stats {
  output_tokens: number | null;
  first_token_ms: number | null;
  tokens_per_second: number | null;
}

export interface ModelsResponse {
  default_model: string;
  models: ModelInfo[];
}

// Server-Sent Events sent by POST /api/chat
export type ChatEvent =
  | { type: 'start'; conversation_id: string; model: string }
  | { type: 'delta'; text: string }
  | ({ type: 'done'; finish_reason: string | null; input_tokens: number | null } & Stats)
  | { type: 'error'; message: string };

// A message as shown in the UI
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  model?: string;
  stats?: Stats;
}
