import { useCallback, useEffect, useState } from 'react';
import { fetchRepository } from '../api';
import type { ModelInfo, RepositoryModel } from '../types';
import { Icon } from './icons';

interface Props {
  hidden: boolean;
  models: ModelInfo[];
}

const gigabytes = (bytes: number | null | undefined) => (bytes == null ? '–' : `${(bytes / 1_000_000_000).toFixed(1)} GB`);

const ROLES: Record<RepositoryModel['role'], string> = {
  chat: 'Chat and workflows',
  embedding: 'Document search',
  unused: 'Not used by this app',
};

// Ollama's capability names in plain words.
const CAPABILITIES: Record<string, string> = {
  completion: 'Text',
  vision: 'Images',
  audio: 'Audio',
  tools: 'Tools',
  thinking: 'Step-by-step thinking',
  embedding: 'Document search',
};

/** Which messages the Auto choice sends to this model. */
function autoRule(model: ModelInfo): string {
  const rules = [model.tags.includes('code') ? 'programming questions' : '', model.is_default ? 'everything else' : ''].filter(Boolean);
  return rules.length ? rules.join(', ') : 'not picked automatically';
}

const thinkingText = (model: ModelInfo) => (model.thinking == null ? 'No switch' : model.thinking ? 'On by default' : 'Off by default, can be turned on per message');

/** One of the chat models: what it is, what it can do, what to use it for. */
function ModelCard({ model, stored }: { model: ModelInfo; stored?: RepositoryModel }) {
  const specs: [string, string][] = [
    ['Made by', model.vendor || '–'],
    ['Licence', model.licence || '–'],
    [`Name in ${model.provider_name}`, model.provider_model],
    ['Parameters', stored?.parameters ?? '–'],
    ['Size on disk', gigabytes(stored?.size_bytes)],
    ['Format', stored?.quantization ?? '–'],
    ['Maximum context', stored?.context_length ? `${stored.context_length.toLocaleString()} tokens` : '–'],
    ['Thinking', thinkingText(model)],
    ['Picked by Auto for', autoRule(model)],
  ];
  return (
    <article className="card lib-card" data-accent={model.accent}>
      <div className="card-head">
        <h3><span className="dot" />{model.label}</h3>
        <span className="lib-tags">
          {model.is_default && <span className="tag model">Default</span>}
          <span className="tag">{model.available ? 'Installed' : 'Not installed'}</span>
          {model.loaded && <span className="tag">In memory</span>}
        </span>
      </div>
      <p className="card-note">{model.description}</p>

      {model.use_cases.length > 0 && (
        <div>
          <h4>Recommended for</h4>
          <ul className="points">{model.use_cases.map(use => <li key={use}>{use}</li>)}</ul>
        </div>
      )}

      <div>
        <h4>Can work with</h4>
        <span className="lib-tags">
          {stored?.capabilities.length ? stored.capabilities.map(name => <span key={name} className="tag">{CAPABILITIES[name] ?? name}</span>) : <span className="side-note">Not reported</span>}
        </span>
      </div>

      <div>
        <h4>Specifications</h4>
        {specs.map(([label, value]) => <div key={label} className="kv"><span>{label}</span><b>{value}</b></div>)}
      </div>
    </article>
  );
}

/** The AI Library: the chat models of this app, and everything else stored in the local runtime. */
export function LibraryView({ hidden, models }: Props) {
  const [repository, setRepository] = useState<RepositoryModel[] | null>(null);
  const [error, setError] = useState('');

  const load = useCallback(() => {
    fetchRepository()
      .then(list => {
        setRepository(list);
        setError('');
      })
      .catch((failure: Error) => setError(failure.message));
  }, []);

  // Read again when the view opens, and when a model is installed, loaded or unloaded
  const state = models.map(model => `${model.id}:${model.available}:${model.loaded}`).join();
  useEffect(() => {
    if (!hidden) load();
  }, [hidden, state, load]);

  return (
    <div id="libraryView" className="view" hidden={hidden}>
      <main className="scroller">
        <div className="lib-col">
          <div className="flow-head">
            <div>
              <div className="flow-kicker">AI Library</div>
              <h1>Models on this machine</h1>
              <p>What each model is for, what it can do and how it is set up. The specifications are read live from the local runtime.</p>
            </div>
            <button type="button" className="btn" onClick={load}><Icon.retry /><span>Refresh</span></button>
          </div>
          {error && <div className="stale">{error}</div>}

          <div className="lib-grid">
            {models.map(model => <ModelCard key={model.id} model={model} stored={repository?.find(entry => entry.catalogue_id === model.id)} />)}
          </div>

          <div className="cards-head"><h3>Model repository</h3><span className="tag">Everything stored in the local runtime</span></div>
          {!repository && !error && <p className="card-note">Loading…</p>}
          {repository && (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr><th>Model</th><th>Used for</th><th>Size on disk</th><th>Parameters</th><th>Format</th><th>In memory</th></tr>
                </thead>
                <tbody>
                  {repository.map(entry => (
                    <tr key={`${entry.provider}/${entry.name}`}>
                      <td><b>{models.find(model => model.id === entry.catalogue_id)?.label ?? entry.name}</b><small className="lib-sub">{entry.name}</small></td>
                      <td>{ROLES[entry.role]}</td>
                      <td>{gigabytes(entry.size_bytes)}</td>
                      <td>{entry.parameters ?? '–'}</td>
                      <td>{entry.quantization ?? '–'}</td>
                      <td>{entry.loaded ? 'Yes' : 'No'}</td>
                    </tr>
                  ))}
                  {repository.length === 0 && <tr><td colSpan={6}>The runtime does not report any stored models.</td></tr>}
                </tbody>
              </table>
            </div>
          )}
          <p className="card-note">One chat model is kept in memory at a time: selecting or using a model unloads the other one.</p>
        </div>
      </main>
    </div>
  );
}
