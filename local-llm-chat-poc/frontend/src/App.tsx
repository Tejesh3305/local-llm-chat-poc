import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchModels, loadModel } from './api';
import { ChatInput } from './components/ChatInput';
import { MessageList } from './components/MessageList';
import { ModelSelector, type ModelStatus } from './components/ModelSelector';
import type { ModelInfo } from './types';
import { useChat } from './useChat';

type Switching = { phase: 'loading' } | { phase: 'error'; message: string } | null;

export function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [model, setModel] = useState('');
  const [loadError, setLoadError] = useState<string | null>(null);
  const [switching, setSwitching] = useState<Switching>(null);
  // The user's Think choice for the selected model; null = use the model's default.
  const [thinkChoice, setThinkChoice] = useState<boolean | null>(null);
  const switchController = useRef<AbortController | null>(null);
  const chat = useChat();

  const refreshModels = useCallback(() => {
    fetchModels()
      .then((data) => {
        setModels(data.models);
        setModel((current) => current || data.default_model);
        setLoadError(null);
      })
      .catch((err: Error) => setLoadError(`Cannot reach the API: ${err.message}`));
  }, []);

  // On start, and after every answer: ask the backend which model is in memory now.
  useEffect(() => {
    if (chat.status === 'idle') refreshModels();
  }, [chat.status, refreshModels]);

  /** Select a model and make it the only one in memory (the backend unloads the previous one). */
  const selectModel = async (id: string) => {
    setModel(id);
    setThinkChoice(null);
    chat.clearError();
    switchController.current?.abort(); // a newer choice replaces a switch that is still running
    if (!models.find((m) => m.id === id)?.available) {
      setSwitching(null);
      return;
    }
    const controller = new AbortController();
    switchController.current = controller;
    setSwitching({ phase: 'loading' });
    try {
      await loadModel(id, controller.signal);
      if (!controller.signal.aborted) setSwitching(null);
    } catch (err) {
      if (!controller.signal.aborted) {
        setSwitching({ phase: 'error', message: err instanceof Error ? err.message : 'The model could not be loaded.' });
      }
    } finally {
      if (!controller.signal.aborted) refreshModels();
    }
  };

  const selected = models.find((m) => m.id === model);
  const busy = chat.status !== 'idle';
  // undefined for models without a thinking switch: nothing is sent and no switch is shown.
  const thinking = selected?.thinking == null ? undefined : (thinkChoice ?? selected.thinking);

  let modelStatus: ModelStatus | null = null;
  if (busy) modelStatus = { tone: 'busy', text: 'Generating…' };
  else if (selected) {
    if (!selected.available) modelStatus = { tone: 'problem', text: 'Not installed' };
    else if (switching?.phase === 'loading') modelStatus = { tone: 'busy', text: 'Loading model…' };
    else if (switching?.phase === 'error') modelStatus = { tone: 'problem', text: 'Failed to load' };
    else if (selected.loaded) modelStatus = { tone: 'ready', text: 'Ready' };
    else modelStatus = { tone: 'idle', text: 'Loads on first message' };
  }

  return (
    <div className="app">
      <header>
        <h1>Local LLM Chat</h1>
        <div className="controls">
          <ModelSelector
            models={models}
            value={model}
            status={modelStatus}
            disabled={busy}
            onChange={(id) => void selectModel(id)}
            thinking={thinking}
            onThinkingChange={setThinkChoice}
          />
          <button type="button" className="secondary" onClick={chat.newChat}>
            New chat
          </button>
        </div>
      </header>

      <MessageList messages={chat.messages} waiting={chat.status === 'waiting'} models={models} />

      {loadError && (
        <div className="banner error">
          {loadError} <button onClick={refreshModels}>Retry</button>
        </div>
      )}
      {chat.error && (
        <div className="banner error">
          {chat.error} <button onClick={() => chat.retry(model, thinking)}>Retry</button>
          <button onClick={chat.clearError}>Dismiss</button>
        </div>
      )}
      {switching?.phase === 'error' && !chat.error && (
        <div className="banner error">
          {switching.message} <button onClick={() => void selectModel(model)}>Retry</button>
        </div>
      )}
      {selected && !selected.available && !chat.error && (
        <div className="banner warning">"{selected.label}" is not installed in the local runtime yet.</div>
      )}

      <ChatInput busy={busy} onSend={(text) => chat.send(text, model, thinking)} onStop={chat.stop} />
    </div>
  );
}
