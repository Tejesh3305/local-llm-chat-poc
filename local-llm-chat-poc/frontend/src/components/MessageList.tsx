import { useEffect, useRef } from 'react';
import type { ChatMessage, ModelInfo } from '../types';

interface Props {
  messages: ChatMessage[];
  waiting: boolean;
  models: ModelInfo[];
}

export function MessageList({ messages, waiting, models }: Props) {
  const bottom = useRef<HTMLDivElement>(null);
  useEffect(() => bottom.current?.scrollIntoView({ block: 'end' }), [messages]);

  if (messages.length === 0) {
    return <div className="messages empty">Ask anything. Answers are generated on this machine.</div>;
  }

  const label = (id?: string) => models.find((m) => m.id === id)?.label ?? id;
  const details = (m: ChatMessage) =>
    [
      label(m.model),
      m.stats?.output_tokens ? `${m.stats.output_tokens} tokens` : '',
      m.stats?.tokens_per_second ? `${m.stats.tokens_per_second} tokens/s` : '',
      m.stats?.first_token_ms != null ? `first word after ${(m.stats.first_token_ms / 1000).toFixed(1)} s` : '',
    ]
      .filter(Boolean)
      .join(' · ');

  return (
    <div className="messages">
      {messages.map((m, i) => (
        <div key={m.id} className={`message ${m.role}`}>
          {m.role === 'assistant' && !m.content && waiting && i === messages.length - 1 ? (
            <span className="typing">Thinking…</span>
          ) : (
            <div className="text">{m.content}</div>
          )}
          {m.role === 'assistant' && m.content && <div className="meta">{details(m)}</div>}
        </div>
      ))}
      <div ref={bottom} />
    </div>
  );
}
