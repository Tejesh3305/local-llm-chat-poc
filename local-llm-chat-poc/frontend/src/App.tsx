import { useEffect, useState } from 'react';
import { fetchModels } from './api';
import { ChatInput } from './components/ChatInput';
import { MessageList } from './components/MessageList';
import { ModelSelector } from './components/ModelSelector';
import type { ModelInfo } from './types';
import { useChat } from './useChat';

export function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [model, setModel] = useState('');
  const [loadError, setLoadError] = useState<string | null>(null);
  const chat = useChat();

  const loadModels = () => {
    fetchModels()
      .then((data) => {
        setModels(data.models);
        setModel((current) => current || data.default_model);
        setLoadError(null);
      })
      .catch((err: Error) => setLoadError(`Cannot reach the API: ${err.message}`));
  };
  useEffect(loadModels, []);

  const selected = models.find((m) => m.id === model);
  const busy = chat.status !== 'idle';

  return (
    <div className="app">
      <header>
        <h1>Local LLM Chat</h1>
        <div className="controls">
          <ModelSelector models={models} value={model} onChange={setModel} />
          <button type="button" className="secondary" onClick={chat.newChat}>
            New chat
          </button>
        </div>
      </header>

      <MessageList messages={chat.messages} waiting={chat.status === 'waiting'} models={models} />

      {loadError && (
        <div className="banner error">
          {loadError} <button onClick={loadModels}>Retry</button>
        </div>
      )}
      {chat.error && (
        <div className="banner error">
          {chat.error} <button onClick={() => chat.retry(model)}>Retry</button>
          <button onClick={chat.clearError}>Dismiss</button>
        </div>
      )}
      {selected && !selected.available && !chat.error && (
        <div className="banner warning">"{selected.label}" is not installed in the local runtime yet.</div>
      )}

      <ChatInput busy={busy} onSend={(text) => chat.send(text, model)} onStop={chat.stop} />
    </div>
  );
}
