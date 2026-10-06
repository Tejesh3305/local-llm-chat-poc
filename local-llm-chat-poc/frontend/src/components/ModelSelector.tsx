import type { ModelInfo } from '../types';

interface Props {
  models: ModelInfo[];
  value: string;
  onChange: (id: string) => void;
}

export function ModelSelector({ models, value, onChange }: Props) {
  return (
    <label className="model-selector">
      Model
      <select value={value} onChange={(e) => onChange(e.target.value)} disabled={models.length === 0}>
        {models.map((m) => (
          <option key={m.id} value={m.id} title={m.description}>
            {m.label}
            {m.available ? '' : ' (not installed)'}
          </option>
        ))}
      </select>
    </label>
  );
}
