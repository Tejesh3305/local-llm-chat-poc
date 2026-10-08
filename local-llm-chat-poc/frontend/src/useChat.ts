import { useCallback, useRef, useState } from 'react';
import { streamChat } from './api';
import type { ChatMessage } from './types';

type Status = 'idle' | 'waiting' | 'streaming';

/** All chat state in one hook: messages, streaming status, errors, stop, retry, new chat. */
export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState<Status>('idle');
  const [error, setError] = useState<string | null>(null);
  const conversationId = useRef<string | null>(null);
  const abortController = useRef<AbortController | null>(null);

  const updateLast = (change: (message: ChatMessage) => ChatMessage) =>
    setMessages((list) => list.map((m, i) => (i === list.length - 1 ? change(m) : m)));

  const send = useCallback(async (text: string, model: string, thinking?: boolean) => {
    setError(null);
    setStatus('waiting');
    const controller = new AbortController();
    abortController.current = controller;
    setMessages((list) => [
      ...list,
      { id: crypto.randomUUID(), role: 'user', content: text },
      { id: crypto.randomUUID(), role: 'assistant', content: '', model },
    ]);

    try {
      await streamChat(
        { message: text, model, conversation_id: conversationId.current, thinking },
        (event) => {
          if (event.type === 'start') {
            conversationId.current = event.conversation_id;
            setStatus('streaming');
          } else if (event.type === 'delta') {
            updateLast((m) => ({ ...m, content: m.content + event.text }));
          } else if (event.type === 'done') {
            const { output_tokens, first_token_ms, tokens_per_second } = event;
            updateLast((m) => ({ ...m, stats: { output_tokens, first_token_ms, tokens_per_second } }));
          } else if (event.type === 'error') {
            setError(event.message);
          }
        },
        controller.signal,
      );
    } catch (err) {
      if (!controller.signal.aborted) setError(err instanceof Error ? err.message : 'Request failed');
    } finally {
      // Drop an assistant bubble that never received any text (error before first token).
      setMessages((list) => (list.at(-1)?.role === 'assistant' && !list.at(-1)?.content ? list.slice(0, -1) : list));
      setStatus('idle');
    }
  }, []);

  const stop = useCallback(() => abortController.current?.abort(), []);

  /** Re-send the last user message (after an error). */
  const retry = useCallback(
    (model: string, thinking?: boolean) => {
      const lastUser = [...messages].reverse().find((m) => m.role === 'user');
      if (!lastUser) return;
      setMessages((list) => list.slice(0, list.lastIndexOf(lastUser)));
      void send(lastUser.content, model, thinking);
    },
    [messages, send],
  );

  const newChat = useCallback(() => {
    abortController.current?.abort();
    conversationId.current = null;
    setMessages([]);
    setError(null);
  }, []);

  return { messages, status, error, send, stop, retry, newChat, clearError: () => setError(null) };
}
