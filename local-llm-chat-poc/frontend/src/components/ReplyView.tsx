import { memo, useEffect, useLayoutEffect, useRef, useState, type MouseEvent } from 'react';
import { fileUrl } from '../api';
import { mdBlocks, patchBlocks } from '../markdown';
import type { DocRef, ExportFormat, Reply } from '../types';
import { Icon } from './icons';

const FORMAT_NAMES: Record<ExportFormat, string> = { pdf: 'PDF', docx: 'Word document', pptx: 'PowerPoint deck' };
const FORMAT_BADGES: Record<ExportFormat, string> = { pdf: 'PDF', docx: 'DOC', pptx: 'PPT' };
const fileSize = (bytes: number) => (bytes < 1_000_000 ? `${Math.max(1, Math.round(bytes / 1000))} KB` : `${(bytes / 1_000_000).toFixed(1)} MB`);

async function copyText(text: string) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const area = document.createElement('textarea');
    area.value = text;
    area.style.cssText = 'position:fixed;opacity:0';
    document.body.append(area);
    area.select();
    document.execCommand('copy');
    area.remove();
  }
}

/** Rendered markdown. The blocks are patched in place so earlier text is not rebuilt while a reply streams. */
export function Markdown({ text }: { text: string }) {
  const ref = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    patchBlocks(ref.current!, text ? mdBlocks(text) : []);
  }, [text]);

  // The Copy buttons of code blocks are part of the generated HTML
  const onClick = (event: MouseEvent) => {
    const button = (event.target as Element).closest('[data-copy="code"]');
    const label = button?.querySelector('span');
    if (!button || !label) return;
    copyText(button.closest('.code')!.querySelector('code')!.textContent ?? '').then(() => {
      label.textContent = 'Copied';
      setTimeout(() => { label.textContent = 'Copy'; }, 1400);
    });
  };
  return <div className="md" ref={ref} onClick={onClick} />;
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const copy = () => copyText(text).then(() => {
    setCopied(true);
    setTimeout(() => setCopied(false), 1400);
  });
  return (
    <button type="button" className="ghost" data-copy="reply" onClick={copy}>
      <Icon.copy /><span>{copied ? 'Copied' : 'Copy'}</span>
    </button>
  );
}

/** Documents as clickable chips: the sources of a reply or the references of a workflow. */
export function Refs({ label, refs, none, onOpen }: { label: string; refs: DocRef[]; none: string; onOpen: (id: string) => void }) {
  return (
    <div className="refs">
      <span className="refs-label">{label}</span>
      {refs.length === 0 && <span className="side-note">{none}</span>}
      {refs.map(doc => (
        <button key={doc.id} type="button" className="ref" title={doc.type} data-doc={doc.id} onClick={() => onOpen(doc.id)}>
          <b>{doc.id}</b><span>{doc.title}</span>
        </button>
      ))}
    </div>
  );
}

function Waiting({ reply }: { reply: Reply }) {
  const [, tick] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => tick(n => n + 1), 100);
    return () => clearInterval(timer);
  }, []);
  const label = reply.waitLabel || (reply.cold ? `Loading ${reply.model.label} into memory` : 'Thinking');
  return (
    <>
      <span className="shimmer">{label}</span>
      <span className="wait-time">{reply.waitLabel ? '' : `${((performance.now() - reply.startedAt) / 1000).toFixed(1)} s`}</span>
    </>
  );
}

const fmtSec = (seconds: number) => `${seconds.toFixed(seconds < 10 ? 2 : 1)} s`;

interface Props {
  reply: Reply;
  rev: number; // reply.rev: the reply object is mutated while it streams, so this is what tells memo to re-render
  onRetry?: () => void;
  onOpenDoc: (id: string) => void;
}

/** One model reply. Chat messages and workflow drafts both use it. */
export const ReplyView = memo(function ReplyView({ reply, onRetry, onOpenDoc }: Props) {
  const waiting = reply.status === 'waiting';
  const finished = !!reply.content && !waiting && reply.status !== 'streaming';
  return (
    <article className="msg assistant" data-accent={reply.model.accent} data-status={reply.status}>
      <div className="who">
        <span className="dot" />
        <span className="who-name">{reply.model.label}</span>
        <span className="who-vendor">{reply.model.vendor}</span>
        {reply.routeReason && <span className="tag" title="Auto chose this model for the message">Auto · {reply.routeReason}</span>}
      </div>
      <div className="wait" hidden={!waiting}>{waiting && <Waiting reply={reply} />}</div>
      <Markdown text={reply.content} />
      <div className="err" hidden={reply.status !== 'error'}>
        <span className="err-text">{reply.error}</span>
        {onRetry && <button type="button" className="btn small" data-retry onClick={onRetry}><Icon.retry />Retry</button>}
      </div>
      {reply.sources && reply.content && (
        <Refs label="Sources" refs={reply.sources} none="No document matched this question." onOpen={onOpenDoc} />
      )}
      {reply.files?.map(file => (
        <div key={file.id} className="file-card" data-format={file.format}>
          <span className="file-badge">{FORMAT_BADGES[file.format]}</span>
          <span className="file-name"><b>{file.filename}</b><small>{FORMAT_NAMES[file.format]} · {fileSize(file.size_bytes)} · made on this machine</small></span>
          {file.format === 'pdf' && <a className="btn small" href={fileUrl(file.id)} target="_blank" rel="noreferrer">Open</a>}
          <a className="btn small primary" href={fileUrl(file.id)} download={file.filename}><Icon.download />Download</a>
        </div>
      ))}
      {!!reply.excerpts?.length && reply.content && (
        <div className="excerpts">
          <span className="refs-label">From your documents</span>
          {reply.excerpts.map((excerpt, index) => (
            <details key={`${excerpt.document_id}-${excerpt.position}`} className="excerpt">
              <summary><b>[{index + 1}]</b><span>{excerpt.filename}</span><small>{excerpt.position ? `part ${excerpt.position}` : 'whole document'}</small></summary>
              <p>{excerpt.text}</p>
            </details>
          ))}
        </div>
      )}
      <div className="meta" hidden={!finished}>
        {finished && (
          <>
            {reply.firstWordSec != null && (
              <span className="chip">First word <b>{fmtSec(reply.firstWordSec)}</b>{reply.cold && <small>incl. model load</small>}</span>
            )}
            {reply.tokensPerSec != null && (
              <span className="chip">Speed <b>{reply.approx ? '≈ ' : ''}{reply.tokensPerSec.toFixed(1)} tokens/s</b></span>
            )}
            {!!reply.tokens && <span className="chip"><b>{reply.tokens}</b> tokens</span>}
            {reply.status === 'stopped' && <span className="chip">Stopped</span>}
            {reply.cut && <span className="chip">Cut off at the length limit</span>}
            <CopyButton text={reply.content} />
          </>
        )}
      </div>
    </article>
  );
});
