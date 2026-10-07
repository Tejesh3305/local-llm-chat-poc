import { useSyncExternalStore } from 'react';
import type { ModelInfo, Reply, Stats } from './types';

/*
 * Only one reply is generated at a time: the engine serves one request at a time,
 * and a second would just queue behind the first. Chat and workflows share this.
 */
let controller: AbortController | null = null;
let owner: string | null = null;
const listeners = new Set<() => void>();
const notify = () => listeners.forEach(listener => listener());

export const generation = {
  get owner() {
    return owner;
  },
  /** Claims the engine for `who` ('chat' or a workflow id) and returns the controller that Stop aborts. */
  begin(who: string): AbortController {
    controller = new AbortController();
    owner = who;
    notify();
    return controller;
  },
  end(finished: AbortController) {
    if (controller !== finished) return;
    controller = null;
    owner = null;
    notify();
  },
  stop() {
    controller?.abort();
  },
};

/** Who is generating right now ('chat', a workflow id) or null. */
export function useGenerating(): string | null {
  return useSyncExternalStore(
    listener => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => owner,
  );
}

// ---------- a reply while it streams ----------

let nextId = 1;

export function newReply(model: ModelInfo, waitLabel = ''): Reply {
  return {
    id: nextId++, rev: 0, model, content: '', status: 'waiting', error: '', waitLabel,
    cold: model.loaded === false, startedAt: performance.now(), pieces: 0, firstAt: 0,
    firstWordSec: null, tokens: null, tokensPerSec: null, approx: false, cut: false,
  };
}

export function addText(reply: Reply, text: string) {
  if (!reply.firstAt) {
    reply.firstAt = performance.now();
    reply.firstWordSec = (reply.firstAt - reply.startedAt) / 1000;
    reply.status = 'streaming';
  }
  reply.pieces++;
  reply.content += text;
  reply.rev++;
}

/** The reply finished normally; `stats` are the figures the backend measured. */
export function finish(reply: Reply, stats: Partial<Stats>) {
  if (!reply.content.trim()) return fail(reply, 'The model returned an empty reply.');
  reply.status = 'done';
  reply.tokens = stats.output_tokens ?? null;
  reply.cut = stats.finish_reason === 'length';
  if (stats.first_token_ms != null) reply.firstWordSec = stats.first_token_ms / 1000;
  if (reply.tokens && stats.generation_ms) reply.tokensPerSec = reply.tokens / (stats.generation_ms / 1000);
  else estimateSpeed(reply);
  reply.rev++;
}

export function fail(reply: Reply, message: string) {
  reply.status = 'error';
  reply.error = message;
  estimateSpeed(reply);
  reply.rev++;
}

/** Stop was pressed: keep what arrived, or mark the reply cancelled if nothing had. */
export function interrupt(reply: Reply) {
  reply.status = reply.content ? 'stopped' : 'cancelled';
  estimateSpeed(reply);
  reply.rev++;
}

/** Stopped replies carry no figures from the engine; estimate speed from what arrived. */
function estimateSpeed(reply: Reply) {
  if (reply.tokensPerSec != null || reply.pieces < 2) return;
  const seconds = (performance.now() - reply.firstAt) / 1000;
  if (seconds > 0.05) {
    reply.tokensPerSec = reply.pieces / seconds;
    reply.approx = true;
  }
}
