import { useCallback, useEffect, useReducer, useRef } from 'react';
import { streamEvents, type Inputs } from './api';
import { addText, fail, finish, generation, interrupt, newReply } from './generation';
import type { ChatEvent, ModelInfo, Reply } from './types';

export type ChatItem =
  | { kind: 'user'; id: number; text: string }
  | { kind: 'reply'; id: number; reply: Reply; question: string; grounded: boolean }
  | { kind: 'switch'; id: number; model: ModelInfo };

interface Options {
  model: ModelInfo | undefined;
  corpus: boolean; // answer from the scenario and its documents
  inputs: Inputs; // workflow input values, so Corpus answers see the same scenario as the workflows
  onSettled: () => void; // a reply ended, one way or another
}

/** All chat state in one hook: the conversation, sending, stop, retry and new chat. */
export function useChat({ model, corpus, inputs, onSettled }: Options) {
  const items = useRef<ChatItem[]>([]);
  const conversationId = useRef<string | null>(null);
  const nextId = useRef(1);
  const frame = useRef(0);
  const [version, bump] = useReducer((n: number) => n + 1, 0);
  const latest = useRef({ model, corpus, inputs, onSettled });
  latest.current = { model, corpus, inputs, onSettled };

  // While text streams in, repaint at most once per frame
  const touch = useCallback(() => {
    if (frame.current) return;
    frame.current = requestAnimationFrame(() => {
      frame.current = 0;
      bump();
    });
  }, []);

  const generate = useCallback(async (question: string, grounded: boolean) => {
    const { model, inputs, onSettled } = latest.current;
    if (!model || generation.owner) return;
    const reply = newReply(model);
    const item: ChatItem = { kind: 'reply', id: nextId.current++, reply, question, grounded };
    items.current = [...items.current, item];
    const controller = generation.begin('chat');
    bump();

    try {
      const body = { message: question, model: model.id, conversation_id: conversationId.current, grounded, inputs: grounded ? inputs : {} };
      for await (const event of streamEvents<ChatEvent>('/api/chat', body, controller.signal)) {
        if (event.type === 'start') {
          conversationId.current = event.conversation_id;
          reply.sources = event.sources;
        } else if (event.type === 'delta') {
          addText(reply, event.text);
          touch();
        } else if (event.type === 'done') {
          finish(reply, event);
        } else {
          fail(reply, event.message);
        }
      }
      if (reply.status === 'waiting' || reply.status === 'streaming') fail(reply, 'The connection closed before the reply finished.');
    } catch (error) {
      if (controller.signal.aborted) interrupt(reply);
      else fail(reply, error instanceof Error ? error.message : 'Request failed');
    } finally {
      generation.end(controller);
      // Stopped before any text: nothing to keep
      if (reply.status === 'cancelled') items.current = items.current.filter(entry => entry !== item);
      bump();
      onSettled();
    }
  }, [touch]);

  const send = useCallback((text: string) => {
    const question = text.trim();
    if (!question || generation.owner || !latest.current.model) return;
    // A failed turn that produced nothing is replaced by the new one
    const last = items.current.at(-1);
    if (last?.kind === 'reply' && last.reply.status === 'error' && !last.reply.content) items.current = items.current.slice(0, -1);
    items.current = [...items.current, { kind: 'user', id: nextId.current++, text: question }];
    void generate(question, latest.current.corpus);
  }, [generate]);

  /** Run the failed reply's question again, with the model selected now. */
  const retry = useCallback((item: ChatItem) => {
    if (item.kind !== 'reply' || generation.owner) return;
    items.current = items.current.filter(entry => entry !== item);
    void generate(item.question, item.grounded);
  }, [generate]);

  const newChat = useCallback(() => {
    if (generation.owner === 'chat') generation.stop();   // a workflow draft in progress is left alone
    conversationId.current = null;
    items.current = [];
    bump();
  }, []);

  // "Switched to ..." appears when the next reply will come from a different model than the last one
  useEffect(() => {
    if (!model) return;
    const list = items.current.at(-1)?.kind === 'switch' ? items.current.slice(0, -1) : items.current;
    const lastReply = list.findLast(entry => entry.kind === 'reply');
    const changed = lastReply?.kind === 'reply' && lastReply.reply.model.id !== model.id;
    items.current = changed ? [...list, { kind: 'switch', id: nextId.current++, model }] : list;
    bump();
  }, [model?.id]);   // eslint-disable-line react-hooks/exhaustive-deps

  // Only the latest reply can be retried
  const lastItem = items.current.at(-1);
  return { items: items.current, version, retryable: lastItem?.kind === 'reply' ? lastItem : null, send, retry, newChat };
}
