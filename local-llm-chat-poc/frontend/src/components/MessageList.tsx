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

  return (
    <div className="messages">
      {messages.map((m, i) => (
        <div key={m.id} className={`message ${m.role}`}>
          {m.role === 'assistant' && !m.content && waiting && i === messages.length - 1 ? (
            <span className="typing">Thinking…</span>
          ) : (
            <div className="text">{m.content}</div>
          )}
          {m.role === 'assistant' && m.content && (
            <div className="meta">
              {label(m.model)}
              {m.outputTokens ? ` · ${m.outputTokens} tokens` : ''}
            </div>
          )}
        </div>
      ))}
      <div ref={bottom} />
    </div>
  );
}
