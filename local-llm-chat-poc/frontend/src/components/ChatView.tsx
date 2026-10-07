import { useEffect, useLayoutEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react';
import { SUGGESTIONS } from '../config';
import { generation } from '../generation';
import type { ModelInfo } from '../types';
import type { ChatItem } from '../useChat';
import { Icon } from './icons';
import { ReplyView } from './ReplyView';

export interface EngineStatus {
  kind: 'idle' | 'ready' | 'loading' | 'warn' | 'down';
  text: string;
}

interface Props {
  hidden: boolean;
  items: ChatItem[];
  version: number; // changes whenever the conversation does, including each streamed frame
  retryable: ChatItem | null;
  model: ModelInfo | undefined;
  generating: string | null;
  status: EngineStatus;
  corpus: boolean;
  corpusCount: number;
  scenario: { name: string; questions: string[] } | undefined;
  onToggleCorpus: () => void;
  onSend: (text: string) => void;
  onRetry: (item: ChatItem) => void;
  onOpenDoc: (id: string) => void;
}

export function ChatView(props: Props) {
  const { hidden, items, version, model, generating, status, corpus, scenario } = props;
  const [text, setText] = useState('');
  const [showJump, setShowJump] = useState(false);
  const [offline, setOffline] = useState(!navigator.onLine);
  const scroller = useRef<HTMLElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const pinned = useRef(true);   // the view follows new text only while the reader is at the bottom
  const lastTop = useRef(0);
  const busy = generating !== null;

  // Follow the conversation as it grows
  useLayoutEffect(() => {
    const view = scroller.current!;
    if (pinned.current) view.scrollTop = view.scrollHeight;
  }, [version, hidden]);

  useEffect(() => {
    const view = scroller.current!;
    const onScroll = () => {
      const fromBottom = view.scrollHeight - view.scrollTop - view.clientHeight;
      if (fromBottom < 90) pinned.current = true;
      else if (view.scrollTop < lastTop.current) pinned.current = false;   // only scrolling up lets go of the latest text
      lastTop.current = view.scrollTop;
      setShowJump(!pinned.current);
    };
    const observer = new ResizeObserver(() => {
      if (pinned.current) view.scrollTop = view.scrollHeight;
    });
    view.addEventListener('scroll', onScroll, { passive: true });
    observer.observe(view);
    return () => {
      view.removeEventListener('scroll', onScroll);
      observer.disconnect();
    };
  }, []);

  useEffect(() => {
    const update = () => setOffline(!navigator.onLine);
    addEventListener('online', update);
    addEventListener('offline', update);
    return () => {
      removeEventListener('online', update);
      removeEventListener('offline', update);
    };
  }, []);

  // Typing can start straight away, also after switching model or view
  useEffect(() => {
    if (!hidden) input.current?.focus();
  }, [hidden, model?.id, corpus, items.length === 0]);

  useLayoutEffect(() => {
    const area = input.current!;
    area.style.height = 'auto';
    area.style.height = `${area.scrollHeight}px`;
  }, [text]);

  const jump = () => {
    pinned.current = true;
    scroller.current!.scrollTop = scroller.current!.scrollHeight;
    setShowJump(false);
  };

  const send = (message: string) => {
    if (busy || !message.trim()) return;
    pinned.current = true;
    setShowJump(false);
    props.onSend(message);
  };

  const submit = () => {
    if (busy || !text.trim()) return;
    send(text);
    setText('');
  };

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    if (busy) generation.stop(); else submit();
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    // isComposing: Enter that confirms a Hindi (IME) word must not send the message
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing && event.keyCode !== 229) {
      event.preventDefault();
      submit();
    }
  };

  const name = model?.label ?? 'the model';
  const emptySub = corpus
    ? `The model answers from ${scenario?.name ?? 'the scenario'} and ${props.corpusCount} sample documents, and lists its sources.`
    : model && !model.local
      ? `Answers come from ${model.provider_name} on another machine.`
      : 'Answers come from a model running on this machine. No internet is used.';

  return (
    <div id="chatView" className="view" hidden={hidden}>
      <main id="scroller" className="scroller" ref={scroller}>
        <section id="empty" className="empty" hidden={items.length > 0}>
          <h1 id="emptyTitle">{corpus ? 'Ask about the scenario.' : 'Ask anything.'}</h1>
          <p id="emptySub">{emptySub}</p>
          <div id="suggestions" className="suggestions">
            {(corpus ? scenario?.questions ?? [] : SUGGESTIONS).map(question => (
              <button key={question} type="button" className="suggest" onClick={() => send(question)}>
                <span>{question}</span><Icon.up />
              </button>
            ))}
          </div>
        </section>
        <div id="thread" className="thread" role="log" aria-label="Conversation" aria-busy={generating === 'chat'} hidden={items.length === 0}>
          {items.map(item => {
            if (item.kind === 'user') return <div key={item.id} className="msg user"><div className="bubble">{item.text}</div></div>;
            if (item.kind === 'switch') {
              return (
                <div key={item.id} className="msg switch" data-accent={item.model.accent}>
                  <span className="switch-pill"><span className="dot" />Switched to <b>{item.model.label}</b><i>{item.model.vendor}</i></span>
                </div>
              );
            }
            return (
              <ReplyView
                key={item.id} reply={item.reply} rev={item.reply.rev} onOpenDoc={props.onOpenDoc}
                onRetry={item === props.retryable ? () => props.onRetry(item) : undefined}
              />
            );
          })}
        </div>
      </main>

      <footer className="dock">
        <div className="dock-inner">
          <button id="jumpBtn" className={showJump ? 'jump show' : 'jump'} type="button" aria-label="Jump to the latest message" onClick={jump}>
            <Icon.down />
          </button>
          <form id="composer" className="composer" onSubmit={onSubmit}>
            <button
              id="corpusBtn" className="tool" type="button" aria-pressed={corpus}
              title="Answer from the scenario and its sample documents, with sources" onClick={props.onToggleCorpus}
            >
              <Icon.book /><span>Corpus</span>
            </button>
            <textarea
              id="input" ref={input} rows={1} autoComplete="off" aria-label="Message" value={text}
              placeholder={corpus ? `Ask ${name} about the scenario…` : `Message ${name}…`}
              onChange={event => setText(event.target.value)} onKeyDown={onKeyDown}
            />
            <button id="sendBtn" className="send" type="submit" data-mode={busy ? 'stop' : 'send'} aria-label={busy ? 'Stop' : 'Send'} disabled={!busy && !text.trim()}>
              <span className="i-send"><Icon.up /></span><span className="i-stop"><Icon.stop /></span>
            </button>
          </form>
          <div className="statusline">
            <span id="engineStatus" className="status" data-state={status.kind}><span className="led" /><span id="statusText">{status.text}</span></span>
            <span className="status-side">
              <span id="netStatus" className="net" hidden={!offline}>Network off</span>
              <span className="hint">Enter to send · Shift+Enter for a new line</span>
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}
