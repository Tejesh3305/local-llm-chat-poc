import { useEffect, useLayoutEffect, useRef, useState, type ChangeEvent, type FormEvent, type KeyboardEvent } from 'react';
import { SUGGESTIONS } from '../config';
import { generation } from '../generation';
import type { ConversationSummary, DocumentInfo, ModelInfo } from '../types';
import type { ChatItem, Picked } from '../useChat';
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
  onSend: (text: string, picked: Picked) => void;
  onRetry: (item: ChatItem) => void;
  onOpenDoc: (id: string) => void;
  auto: boolean; // the backend picks the model for each message
  thinking: boolean | undefined; // undefined: the selected model has no Think switch
  onThinking: (on: boolean) => void;
  // saved chats
  conversations: ConversationSummary[];
  activeId: string | null;
  onOpenChat: (id: string) => void;
  onRenameChat: (id: string, title: string) => void;
  onDeleteChat: (id: string) => void;
  onNewChat: () => void;
  // uploaded documents
  documents: DocumentInfo[];
  uploading: boolean;
  onUpload: (file: File) => void;
  onDeleteDocument: (id: string) => void;
}

/** The "@word" being typed right before the caret, if any. `start` is the index of the @. */
function mentionAt(text: string, caret: number): { start: number; query: string } | null {
  const match = /(?:^|\s)@([^\s@]*)$/.exec(text.slice(0, caret));
  return match ? { start: caret - match[1].length - 1, query: match[1] } : null;
}

const fileSize = (bytes: number) => (bytes < 1_000_000 ? `${Math.max(1, Math.round(bytes / 1000))} KB` : `${(bytes / 1_000_000).toFixed(1)} MB`);

/** The left rail of the chat view: the saved chats and the uploaded documents. */
function ChatRail(props: Props) {
  const rename = (chat: ConversationSummary) => {
    const title = window.prompt('Rename chat', chat.title)?.trim();
    if (title && title !== chat.title) props.onRenameChat(chat.id, title.slice(0, 120));
  };
  return (
    <aside className="flow-rail chat-rail" aria-label="Saved chats and documents">
      <button type="button" className="btn primary rail-new" onClick={props.onNewChat} title="Start a new chat (Ctrl+Shift+O)">
        <Icon.plus /><span>New chat</span>
      </button>
      <div className="rail-title">Chats</div>
      {props.conversations.length === 0 && <p className="rail-empty">No saved chats yet. A chat is saved after its first reply.</p>}
      {props.conversations.map(chat => (
        <div key={chat.id} className="rail-row" aria-current={chat.id === props.activeId}>
          <button type="button" className="rail-open" title={chat.title} onClick={() => props.onOpenChat(chat.id)}>{chat.title}</button>
          <button type="button" className="rail-act" title="Rename" aria-label={`Rename ${chat.title}`} onClick={() => rename(chat)}><Icon.pencil /></button>
          <button
            type="button" className="rail-act" title="Delete" aria-label={`Delete ${chat.title}`}
            onClick={() => window.confirm(`Delete "${chat.title}"?`) && props.onDeleteChat(chat.id)}
          >
            <Icon.close />
          </button>
        </div>
      ))}

      <div className="rail-title rail-gap">Documents</div>
      {props.documents.length === 0 && <p className="rail-empty">Use Attach to add a PDF, DOCX or text file, then ask about it.</p>}
      {props.documents.map(file => (
        <div key={file.id} className="rail-row">
          <span className="rail-open" title={file.filename}>
            {file.filename}
            <small>{fileSize(file.size_bytes)} · {file.chunks} {file.chunks === 1 ? 'part' : 'parts'}</small>
          </span>
          <button
            type="button" className="rail-act" title="Remove" aria-label={`Remove ${file.filename}`}
            onClick={() => window.confirm(`Remove "${file.filename}" from the documents?`) && props.onDeleteDocument(file.id)}
          >
            <Icon.close />
          </button>
        </div>
      ))}
    </aside>
  );
}

export function ChatView(props: Props) {
  const { hidden, items, version, model, generating, status, corpus, scenario, auto, thinking, documents, uploading } = props;
  const [text, setText] = useState('');
  // Documents picked with @ for the next question, and the @ menu while it is open
  const [picked, setPicked] = useState<Picked>([]);
  const [mention, setMention] = useState<{ start: number; query: string } | null>(null);
  const [active, setActive] = useState(0);
  const dismissed = useRef(-1);   // the @ whose menu was closed with Esc, so it stays closed
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
  }, [hidden, model?.id, corpus, items.length === 0, props.activeId]);

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

  // A document removed from the rail can no longer be picked
  useEffect(() => {
    setPicked(list => (list.every(doc => documents.some(entry => entry.id === doc.id)) ? list : list.filter(doc => documents.some(entry => entry.id === doc.id))));
  }, [documents]);

  const options = mention
    ? documents.filter(doc => !picked.some(entry => entry.id === doc.id) && doc.filename.toLowerCase().includes(mention.query.toLowerCase()))
    : [];

  /** Open, update or close the @ menu from the text before the caret. */
  const track = (area: HTMLTextAreaElement) => {
    const found = area.selectionStart === area.selectionEnd ? mentionAt(area.value, area.selectionStart) : null;
    if (!found) dismissed.current = -1;   // the caret left that @, so a new one opens the menu again
    const next = found && found.start !== dismissed.current ? found : null;
    setMention(current => (current?.start === next?.start && current?.query === next?.query ? current : next));
    if (next?.query !== mention?.query || next?.start !== mention?.start) setActive(0);
  };

  /** Take the document from the menu: the "@word" leaves the text and the document becomes a chip. */
  const choose = (doc: DocumentInfo) => {
    if (!mention) return;
    const end = mention.start + 1 + mention.query.length;
    const rest = text.slice(end).replace(/^ /, '');
    setText(text.slice(0, mention.start) + rest);
    setPicked(list => [...list, { id: doc.id, filename: doc.filename }]);
    setMention(null);
    dismissed.current = -1;
    requestAnimationFrame(() => {
      const area = input.current!;
      area.focus();
      area.setSelectionRange(mention.start, mention.start);
    });
  };

  const send = (message: string) => {
    if (busy || !message.trim()) return;
    pinned.current = true;
    setShowJump(false);
    props.onSend(message, picked);
    setPicked([]);   // the picked documents apply to this one question
  };

  const submit = () => {
    if (busy || !text.trim()) return;
    send(text);
    setText('');
    setMention(null);
  };

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    if (busy) generation.stop(); else submit();
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    const composing = event.nativeEvent.isComposing || event.keyCode === 229;
    if (mention && !composing) {
      if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();   // Esc closes the menu; it must not also stop a reply
        dismissed.current = mention.start;
        setMention(null);
        return;
      }
      if (options.length && (event.key === 'ArrowDown' || event.key === 'ArrowUp')) {
        event.preventDefault();
        setActive(index => (index + (event.key === 'ArrowDown' ? 1 : options.length - 1)) % options.length);
        return;
      }
      if (options.length && (event.key === 'Enter' || event.key === 'Tab') && !event.shiftKey) {
        event.preventDefault();
        choose(options[Math.min(active, options.length - 1)]);
        return;
      }
    }
    // Backspace at the very start of an empty box takes back the last picked document
    if (event.key === 'Backspace' && !text && picked.length) {
      event.preventDefault();
      setPicked(list => list.slice(0, -1));
      return;
    }
    // isComposing: Enter that confirms a Hindi (IME) word must not send the message
    if (event.key === 'Enter' && !event.shiftKey && !composing) {
      event.preventDefault();
      submit();
    }
  };

  const name = auto ? 'the AI' : model?.label ?? 'the model';
  const emptySub = corpus
    ? `The model answers from ${scenario?.name ?? 'the scenario'} and ${props.corpusCount} sample documents, and lists its sources.`
    : documents.length
      ? 'Answers come from a model running on this machine. To ask about your documents, type @ and pick them; nothing else is read.'
      : model && !model.local
        ? `Answers come from ${model.provider_name} on another machine.`
        : 'Answers come from a model running on this machine. No internet is used.';

  const onPick = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';   // the same file can be picked again
    if (file) props.onUpload(file);
  };

  return (
    <div id="chatView" className="view chat" hidden={hidden}>
      <ChatRail {...props} />
      <div className="chat-main">
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
            if (item.kind === 'user') {
              return (
                <div key={item.id} className="msg user">
                  {item.documents.length > 0 && (
                    <div className="asked-from" title="Only these documents were searched for this question">
                      {item.documents.map(name => <span key={name} className="mention-chip"><Icon.file />{name}</span>)}
                    </div>
                  )}
                  <div className="bubble">{item.text}</div>
                </div>
              );
            }
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
          <div className="tools">
            <button
              id="corpusBtn" className="tool" type="button" aria-pressed={corpus}
              title="Answer from the scenario and its sample documents, with sources" onClick={props.onToggleCorpus}
            >
              <Icon.book /><span>Corpus</span>
            </button>
            {thinking !== undefined && (
              <button
                id="thinkBtn" className="tool" type="button" aria-pressed={thinking}
                title="Let the model reason before answering. Slower, but better for multi-step problems." onClick={() => props.onThinking(!thinking)}
              >
                <Icon.bulb /><span>Think</span>
              </button>
            )}
            <span className="tools-gap" />
            <label className="tool attach" data-busy={uploading} title="Add a PDF, DOCX, TXT or MD file to the documents">
              <Icon.clip /><span>{uploading ? 'Reading the file…' : 'Attach'}</span>
              <input type="file" accept=".pdf,.docx,.txt,.md" disabled={uploading} onChange={onPick} />
            </label>
          </div>
          {picked.length > 0 && (
            <div className="mentions">
              <span className="refs-label">Answer only from</span>
              {picked.map(doc => (
                <span key={doc.id} className="mention-chip">
                  <Icon.file />{doc.filename}
                  <button type="button" aria-label={`Remove ${doc.filename}`} title="Remove" onClick={() => setPicked(list => list.filter(entry => entry.id !== doc.id))}>
                    <Icon.close />
                  </button>
                </span>
              ))}
            </div>
          )}
          <form id="composer" className="composer" onSubmit={onSubmit}>
            {mention && (
              <div id="mentionMenu" className="mention-menu" role="listbox" aria-label="Pick a document">
                <div className="mention-head">Answer only from…</div>
                {options.map((doc, index) => (
                  <button
                    key={doc.id} type="button" role="option" className="mention-opt" aria-selected={index === active}
                    onMouseDown={event => event.preventDefault()}   // keep the caret in the message box
                    onMouseEnter={() => setActive(index)} onClick={() => choose(doc)}
                  >
                    <Icon.file /><span>{doc.filename}</span><small>{fileSize(doc.size_bytes)} · {doc.chunks} {doc.chunks === 1 ? 'part' : 'parts'}</small>
                  </button>
                ))}
                {options.length === 0 && (
                  <p className="mention-none">
                    {documents.length === 0 ? 'No documents yet. Use Attach to add one.'
                      : documents.length === picked.length ? 'Every document is already picked.'
                        : `No document matches "${mention.query}".`}
                  </p>
                )}
              </div>
            )}
            <textarea
              id="input" ref={input} rows={1} autoComplete="off" aria-label="Message" value={text}
              aria-controls={mention ? 'mentionMenu' : undefined} aria-expanded={!!mention}
              placeholder={
                picked.length ? `Ask ${name} about ${picked.length === 1 ? picked[0].filename : `these ${picked.length} documents`}…`
                  : corpus ? `Ask ${name} about the scenario…`
                    : documents.length ? `Message ${name}… (type @ to pick documents)`
                      : `Message ${name}…`
              }
              onChange={event => {
                setText(event.target.value);
                track(event.target);
              }}
              onSelect={event => track(event.currentTarget)}
              onBlur={() => setMention(null)}
              onKeyDown={onKeyDown}
            />
            <button id="sendBtn" className="send" type="submit" data-mode={busy ? 'stop' : 'send'} aria-label={busy ? 'Stop' : 'Send'} disabled={!busy && !text.trim()}>
              <span className="i-send"><Icon.up /></span><span className="i-stop"><Icon.stop /></span>
            </button>
          </form>
          <div className="statusline">
            <span id="engineStatus" className="status" data-state={status.kind}><span className="led" /><span id="statusText">{status.text}</span></span>
            <span className="status-side">
              <span id="netStatus" className="net" hidden={!offline}>Network off</span>
              <span className="hint">{documents.length ? '@ to pick documents · ' : ''}Enter to send · Shift+Enter for a new line</span>
            </span>
          </div>
        </div>
      </footer>
      </div>
    </div>
  );
}
