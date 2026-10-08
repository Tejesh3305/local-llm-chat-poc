import { useCallback, useEffect, useReducer, useRef } from 'react';
import { AUTO, fetchConversation, streamEvents, type Inputs } from './api';
import { addText, fail, finish, generation, interrupt, newReply, storedReply } from './generation';
import type { ChatEvent, DocumentInfo, ModelInfo, Reply } from './types';

/** Documents picked with @ for one question: only these are searched for it. */
export type Picked = Pick<DocumentInfo, 'id' | 'filename'>[];

export type ChatItem =
  | { kind: 'user'; id: number; text: string; documents: string[] }
  | { kind: 'reply'; id: number; reply: Reply; question: string; grounded: boolean; documents: boolean; picked: Picked }
  | { kind: 'switch'; id: number; model: ModelInfo };

interface Options {
  model: ModelInfo | undefined; // the selected model; with Auto on, the default one, until the backend has chosen
  auto: boolean; // let the backend pick a model for each message
  models: ModelInfo[];
  thinking: boolean | undefined; // undefined: use the model's default
  corpus: boolean; // answer from the scenario and its documents
  documents: boolean; // answer from the uploaded documents
  inputs: Inputs; // workflow input values, so Corpus answers see the same scenario as the workflows
  onSettled: () => void; // a reply ended, one way or another
}

/** All chat state in one hook: the conversation, sending, stop, retry, new chat and opening a saved chat. */
export function useChat(options: Options) {
  const { model, auto } = options;
  const items = useRef<ChatItem[]>([]);
  const conversationId = useRef<string | null>(null);
  const nextId = useRef(1);
  const frame = useRef(0);
  const [version, bump] = useReducer((n: number) => n + 1, 0);
  const latest = useRef(options);
  latest.current = options;

  // While text streams in, repaint at most once per frame
  const touch = useCallback(() => {
    if (frame.current) return;
    frame.current = requestAnimationFrame(() => {
      frame.current = 0;
      bump();
    });
  }, []);

  const generate = useCallback(async (question: string, grounded: boolean, documents: boolean, picked: Picked) => {
    const { model, auto, thinking, inputs, onSettled } = latest.current;
    if (!model || generation.owner) return;
    const reply = newReply(model);
    const item: ChatItem = { kind: 'reply', id: nextId.current++, reply, question, grounded, documents, picked };
    items.current = [...items.current, item];
    const controller = generation.begin('chat');
    bump();

    try {
      const body = {
        message: question, model: auto ? AUTO : model.id, conversation_id: conversationId.current,
        grounded, inputs: grounded ? inputs : {}, thinking: auto ? undefined : thinking,
        use_documents: documents || picked.length > 0, document_ids: picked.map(doc => doc.id),
      };
      for await (const event of streamEvents<ChatEvent>('/api/chat', body, controller.signal)) {
        if (event.type === 'start') {
          conversationId.current = event.conversation_id;
          reply.sources = event.sources;
          // With Auto, this is where the page learns which model answers
          const chosen = latest.current.models.find(entry => entry.id === event.model);
          if (chosen && chosen.id !== reply.model.id) {
            reply.model = chosen;
            reply.cold = chosen.loaded === false;
          }
          reply.routeReason = event.route_reason;
          reply.rev++;
          touch();
        } else if (event.type === 'excerpts') {
          reply.excerpts = event.excerpts;
          reply.rev++;
          touch();
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

  /** Send a question; `picked` (documents chosen with @) limits the answer to those documents. */
  const send = useCallback((text: string, picked: Picked = []) => {
    const question = text.trim();
    if (!question || generation.owner || !latest.current.model) return;
    // A failed turn that produced nothing is replaced by the new one
    const last = items.current.at(-1);
    if (last?.kind === 'reply' && last.reply.status === 'error' && !last.reply.content) items.current = items.current.slice(0, -1);
    items.current = [...items.current, { kind: 'user', id: nextId.current++, text: question, documents: picked.map(doc => doc.filename) }];
    void generate(question, latest.current.corpus, latest.current.documents, picked);
  }, [generate]);

  /** Run the failed reply's question again, with the model selected now and the same documents. */
  const retry = useCallback((item: ChatItem) => {
    if (item.kind !== 'reply' || generation.owner) return;
    items.current = items.current.filter(entry => entry !== item);
    void generate(item.question, item.grounded, item.documents, item.picked);
  }, [generate]);

  const newChat = useCallback(() => {
    if (generation.owner === 'chat') generation.stop();   // a workflow draft in progress is left alone
    conversationId.current = null;
    items.current = [];
    bump();
  }, []);

  /** Show a saved chat; the next message continues it. Throws if it cannot be fetched. */
  const open = useCallback(async (id: string) => {
    const { models, model } = latest.current;
    if (!model) return;
    if (generation.owner === 'chat') generation.stop();
    const conversation = await fetchConversation(id);
    const list: ChatItem[] = [];
    let question = '';
    for (const message of conversation.messages) {
      if (message.role === 'user') {
        question = message.content;
        list.push({ kind: 'user', id: nextId.current++, text: message.content, documents: message.documents ?? [] });
      } else {
        // A model that has since left the catalogue is shown under its id
        const author = models.find(entry => entry.id === message.model) ?? { ...model, id: message.model ?? model.id, label: message.model ?? model.label, vendor: '' };
        const reply = storedReply(author, message);
        list.push({ kind: 'reply', id: nextId.current++, reply, question, grounded: message.grounded, documents: message.excerpts.length > 0, picked: [] });
      }
    }
    conversationId.current = conversation.id;
    items.current = list;
    bump();
  }, []);

  // "Switched to ..." appears when the next reply will come from a different model than the last one
  useEffect(() => {
    if (!model) return;
    const list = items.current.at(-1)?.kind === 'switch' ? items.current.slice(0, -1) : items.current;
    const lastReply = list.findLast(entry => entry.kind === 'reply');
    const changed = !auto && lastReply?.kind === 'reply' && lastReply.reply.model.id !== model.id;
    items.current = changed ? [...list, { kind: 'switch', id: nextId.current++, model }] : list;
    bump();
  }, [model?.id, auto]);   // eslint-disable-line react-hooks/exhaustive-deps

  // Only the latest reply can be retried
  const lastItem = items.current.at(-1);
  return {
    items: items.current, version, conversationId: conversationId.current,
    retryable: lastItem?.kind === 'reply' ? lastItem : null, send, retry, newChat, open,
  };
}
