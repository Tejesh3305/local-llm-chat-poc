import { useEffect, useRef, useState, type ReactNode } from 'react';
import { fetchCorpus } from '../api';
import type { Doc, ModelInfo, ModelsResponse } from '../types';
import { Icon } from './icons';

/** A native <dialog> that opens and closes with `open`. Esc and the close button both call onClose. */
function Sheet({ id, title, open, onClose, children }: { id: string; title: string; open: boolean; onClose: () => void; children: ReactNode }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current!;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);
  return (
    <dialog id={id} ref={ref} aria-labelledby={`${id}Title`} onClose={onClose}>
      <div className="sheet">
        <div className="sheet-head">
          <h2 id={`${id}Title`}>{title}</h2>
          <button className="icon-btn" type="button" aria-label="Close" onClick={onClose}><Icon.close /></button>
        </div>
        {open && children}
      </div>
    </dialog>
  );
}

// ---------- documents ----------

/** null = closed, '' = the list of all documents, otherwise the id of the document shown. */
export type DocTarget = string | null;

export function DocDialog({ target, onShow }: { target: DocTarget; onShow: (target: DocTarget) => void }) {
  const [docs, setDocs] = useState<Doc[] | null>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    if (target === null || docs) return;
    fetchCorpus().then(setDocs).catch((failure: Error) => setError(failure.message));
  }, [target, docs]);

  const doc = docs?.find(entry => entry.id === target);
  const title = doc ? doc.title : docs ? `Sample corpus: ${docs.length} documents` : 'Sample corpus';
  return (
    <Sheet id="docDialog" title={title} open={target !== null} onClose={() => onShow(null)}>
      <div id="docBody">
        {!docs && <p className="card-note">{error || 'Loading…'}</p>}
        {doc && (
          <>
            <div className="doc-meta">
              <span className="tag">{doc.id}</span><span className="tag">{doc.type}</span>
              <span className="tag">Sample document, invented for the demo</span>
            </div>
            <p className="doc-text">{doc.text}</p>
            <button type="button" className="ghost" onClick={() => onShow('')}>All {docs!.length} documents</button>
          </>
        )}
        {docs && !doc && (
          <>
            <p className="card-note">Every document is invented for the demo. The backend's search runs over these texts.</p>
            <div className="doc-list">
              {docs.map(entry => (
                <button key={entry.id} type="button" className="ref" title={entry.type} data-doc={entry.id} onClick={() => onShow(entry.id)}>
                  <b>{entry.id}</b><span>{entry.title} ({entry.type})</span>
                </button>
              ))}
            </div>
          </>
        )}
      </div>
    </Sheet>
  );
}

// ---------- connection and theme ----------

interface StatusProps {
  open: boolean;
  onClose: () => void;
  theme: string;
  onTheme: (theme: string) => void;
  serverDown: boolean;
  engine: ModelsResponse | null;
  onCheck: () => void;
}

export function StatusDialog({ open, onClose, theme, onTheme, serverDown, engine, onCheck }: StatusProps) {
  const row = (ok: 'yes' | 'no' | '', text: string, code?: string) => (
    <li key={text} data-ok={ok}><span>{text}{code && <> <code>{code}</code></>}</span></li>
  );
  const model = (entry: ModelInfo) => {
    const status = engine!.providers[entry.provider];
    if (status !== 'ok') return row('no', `${entry.label}: ${entry.provider_name} is ${status === 'unreachable' ? 'not running' : 'not configured in backend/.env'}`);
    if (!entry.available) return row('no', `${entry.label} is not installed. Run`, `ollama pull ${entry.provider_model}`);
    const memory = !entry.loaded ? '' : entry.gpu_percent == null ? ' · in memory' : ` · in memory, ${entry.gpu_percent}% on GPU`;
    return row('yes', `${entry.label} installed${memory}`, entry.provider_model);
  };
  return (
    <Sheet id="settingsDialog" title="Connection" open={open} onClose={onClose}>
      <div className="field">
        <span>Theme</span>
        <div className="seg">
          {['light', 'dark'].map(name => (
            <button key={name} type="button" aria-pressed={theme === name} data-theme-value={name} onClick={() => onTheme(name)}>
              {name[0].toUpperCase() + name.slice(1)}
            </button>
          ))}
        </div>
      </div>
      <ul className="diag">
        {serverDown || !engine
          ? row(serverDown ? 'no' : '', serverDown ? 'The app server is not answering.' : 'Checking…')
          : [row('yes', 'App server connected'), ...engine.models.map(model)]}
      </ul>
      <div className="sheet-foot">
        <span className="spacer" />
        <button type="button" className="btn" onClick={onCheck}>Check again</button>
        <button type="button" className="btn primary" onClick={onClose}>Done</button>
      </div>
    </Sheet>
  );
}
