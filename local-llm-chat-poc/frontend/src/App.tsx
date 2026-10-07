import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchModels, loadModel, SERVER_DOWN } from './api';
import { ChatView, type EngineStatus } from './components/ChatView';
import { DocDialog, StatusDialog, type DocTarget } from './components/Dialogs';
import { Icon } from './components/icons';
import { ModelPicker } from './components/ModelPicker';
import { WorkflowsView } from './components/WorkflowsView';
import { BRAND } from './config';
import { generation, useGenerating } from './generation';
import type { ModelInfo, ModelsResponse } from './types';
import { useChat } from './useChat';
import { useWorkflows } from './useWorkflows';

type View = 'chat' | 'flows';

interface Banner {
  kind: 'danger' | 'warn';
  text: string;
  command?: string;
}

/** What is wrong with the connection to the selected model, if anything. */
function bannerFor(serverDown: boolean, engine: ModelsResponse | null, model: ModelInfo | undefined): Banner | null {
  if (serverDown) return { kind: 'danger', text: SERVER_DOWN };
  if (!engine || !model) return null;
  const provider = engine.providers[model.provider];
  if (provider === 'unreachable') return { kind: 'danger', text: `Local engine not running. Start ${model.provider_name}.` };
  if (provider === 'not_configured') return { kind: 'danger', text: `${model.provider_name} has no address. Set it in backend/.env and restart the app server.` };
  if (!model.available) return { kind: 'warn', text: `${model.label} is not installed on this machine. Run`, command: `ollama pull ${model.provider_model}` };
  return null;
}

function statusFor(serverDown: boolean, engine: ModelsResponse | null, model: ModelInfo | undefined, loadingId: string | null): EngineStatus {
  if (serverDown) return { kind: 'down', text: 'App server not answering' };
  if (!engine || !model) return { kind: 'idle', text: 'Connecting…' };
  const provider = engine.providers[model.provider];
  if (provider === 'unreachable') return { kind: 'down', text: 'Local engine not running' };
  if (provider === 'not_configured') return { kind: 'down', text: `${model.provider_name} is not configured` };
  if (!model.available) return { kind: 'warn', text: `${model.label} is not installed` };
  if (loadingId === model.id) return { kind: 'loading', text: `Loading ${model.label} into memory…` };
  if (model.loaded === null) return { kind: 'ready', text: `${model.provider_name} connected` };
  if (!model.loaded) return { kind: 'ready', text: `${model.provider_name} running · ${model.label} loads with the next message` };
  const onGpu = model.gpu_percent != null && model.gpu_percent < 100 ? ` · ${model.gpu_percent}% on GPU` : '';
  return { kind: 'ready', text: `${model.label} ready in memory${onGpu}` };
}

export function App() {
  const [engine, setEngine] = useState<ModelsResponse | null>(null);
  const [serverDown, setServerDown] = useState(false);
  const [modelId, setModelId] = useState('');
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [view, setView] = useState<View>(location.hash === '#workflows' ? 'flows' : 'chat');
  const [corpus, setCorpus] = useState(false);
  const [theme, setTheme] = useState(document.documentElement.dataset.theme || 'light');
  const [doc, setDoc] = useState<DocTarget>(null);
  const [statusOpen, setStatusOpen] = useState(false);
  const generating = useGenerating();

  const refresh = useCallback(async () => {
    try {
      const next = await fetchModels();
      setEngine(next);
      setServerDown(false);
      return Object.values(next.providers).includes('ok');
    } catch {
      setServerDown(true);
      return false;
    }
  }, []);

  // Keep watching the engine: every 5 s while it is fine, every 2 s while something is wrong
  useEffect(() => {
    let timer = 0, stopped = false;
    const watch = async () => {
      const healthy = document.hidden ? true : await refresh();
      if (!stopped) timer = window.setTimeout(watch, healthy ? 5000 : 2000);
    };
    void watch();
    const onVisible = () => { if (!document.hidden) void refresh(); };
    document.addEventListener('visibilitychange', onVisible);
    return () => {
      stopped = true;
      clearTimeout(timer);
      document.removeEventListener('visibilitychange', onVisible);
    };
  }, [refresh]);

  const models = engine?.models ?? [];
  const selected = models.find(model => model.id === modelId) ?? models.find(model => model.is_default) ?? models[0];
  const engineUp = (model: ModelInfo) => engine?.providers[model.provider] === 'ok';

  const flows = useWorkflows(selected, refresh);
  const chat = useChat({ model: selected, corpus, inputs: flows.inputs, onSettled: refresh });

  // Start loading the selected model so the wait is over before the next message is sent.
  // It is tried once per selection, and again after the model was unavailable (engine down, or not yet pulled).
  // A model that the engine has unloaded for being idle is left alone until the next message.
  const loadTried = useRef('');
  useEffect(() => {
    if (!selected) return;
    if (!selected.available) {
      loadTried.current = '';
      return;
    }
    if (generating || selected.loaded !== false || loadTried.current === selected.id) return;
    loadTried.current = selected.id;
    setLoadingId(selected.id);
    loadModel(selected.id).catch(() => { /* the next message reports any problem */ }).finally(() => {
      setLoadingId(null);
      void refresh();
    });
  }, [selected?.id, selected?.available, selected?.loaded, generating]);   // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('nextxr-theme', theme); } catch { /* storage unavailable */ }
  }, [theme]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !document.querySelector('dialog[open]') && generation.owner) generation.stop();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, []);

  const switchView = (next: View) => {
    setView(next);
    history.replaceState(null, '', next === 'flows' ? '#workflows' : location.pathname + location.search);
  };

  const banner = bannerFor(serverDown, engine, selected);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span id="brandName" className="brand-name">{BRAND.name}</span>
          <div id="views" className="views" role="tablist" aria-label="View">
            <button type="button" role="tab" data-view="chat" aria-selected={view === 'chat'} onClick={() => switchView('chat')}>Chat</button>
            <button type="button" role="tab" data-view="flows" aria-selected={view === 'flows'} onClick={() => switchView('flows')}>Workflows</button>
          </div>
          <span id="brandTag" className="brand-tag">{BRAND.tagline}</span>
        </div>
        <ModelPicker models={models} selected={selected} loadingId={loadingId} engineUp={engineUp} onSelect={setModelId} />
        <div className="top-actions">
          <button id="newChatBtn" className="btn" type="button" hidden={view !== 'chat'} onClick={chat.newChat}><Icon.plus /><span>New chat</span></button>
          <button id="settingsBtn" className="icon-btn" type="button" aria-label="Connection and theme" title="Connection and theme" onClick={() => setStatusOpen(true)}>
            <Icon.sliders />
          </button>
        </div>
      </header>

      <div id="banner" className="banner" role="alert" data-kind={banner?.kind} hidden={!banner}>
        <span id="bannerText">{banner?.text}{banner?.command && <> <code>{banner.command}</code></>}</span>
        <button id="bannerAction" className="btn small" type="button" onClick={() => void refresh()}>Check again</button>
      </div>

      <ChatView
        hidden={view !== 'chat'} items={chat.items} version={chat.version} retryable={chat.retryable}
        model={selected} generating={generating} status={statusFor(serverDown, engine, selected, loadingId)}
        corpus={corpus} corpusCount={flows.data?.corpus_count ?? 0} scenario={flows.data?.scenario}
        onToggleCorpus={() => setCorpus(on => !on)} onSend={chat.send} onRetry={chat.retry} onOpenDoc={setDoc}
      />
      <WorkflowsView hidden={view !== 'flows'} flows={flows} model={selected} generating={generating} onOpenDoc={setDoc} onBrowse={() => setDoc('')} />

      <DocDialog target={doc} onShow={setDoc} />
      <StatusDialog
        open={statusOpen} onClose={() => setStatusOpen(false)} theme={theme} onTheme={setTheme}
        serverDown={serverDown} engine={engine} onCheck={() => void refresh()}
      />
    </div>
  );
}
