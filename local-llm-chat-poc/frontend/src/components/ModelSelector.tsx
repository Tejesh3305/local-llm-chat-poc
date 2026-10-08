import type { ModelInfo } from '../types';

export interface ModelStatus {
  tone: 'ready' | 'busy' | 'idle' | 'problem';
  text: string;
}

interface Props {
  models: ModelInfo[];
  value: string;
  status: ModelStatus | null;
  disabled: boolean;
  onChange: (id: string) => void;
  /** Whether the next message lets the model reason first; undefined hides the switch. */
  thinking?: boolean;
  onThinkingChange: (on: boolean) => void;
}

export function ModelSelector({ models, value, status, disabled, onChange, thinking, onThinkingChange }: Props) {
  return (
    <div className="model-selector">
      <label>
        Model
        <select value={value} onChange={(e) => onChange(e.target.value)} disabled={disabled || models.length === 0}>
          {models.map((m) => (
            <option key={m.id} value={m.id} title={m.description}>
              {m.label}
              {m.available ? '' : ' (not installed)'}
            </option>
          ))}
        </select>
      </label>
      {thinking !== undefined && (
        <button
          type="button"
          className="think-toggle"
          aria-pressed={thinking}
          title="Let the model reason before answering. Slower, but better for multi-step problems."
          onClick={() => onThinkingChange(!thinking)}
        >
          Think: {thinking ? 'on' : 'off'}
        </button>
      )}
      {status && (
        <span className={`model-status ${status.tone}`} role="status">
          <span className="dot" aria-hidden="true" />
          {status.text}
        </span>
      )}
    </div>
  );
}
