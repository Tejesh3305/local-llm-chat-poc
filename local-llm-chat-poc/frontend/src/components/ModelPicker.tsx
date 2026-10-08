import { useEffect, useRef, useState, type KeyboardEvent } from 'react';
import { AUTO } from '../api';
import type { ModelInfo } from '../types';
import { Icon } from './icons';

interface Props {
  models: ModelInfo[];
  selected: ModelInfo | undefined;
  auto: boolean; // the backend picks a model for each message
  loadingId: string | null; // model being loaded into memory ahead of use
  engineUp: (model: ModelInfo) => boolean;
  onSelect: (id: string) => void; // a model id, or AUTO
}

/** In plain words, which messages the Auto choice sends where. */
export function autoRule(models: ModelInfo[]): string {
  const coder = models.find(model => model.tags.includes('code'));
  const fallback = models.find(model => model.is_default) ?? models[0];
  if (!fallback) return '';
  return coder && coder.id !== fallback.id ? `Programming questions go to ${coder.label}, everything else to ${fallback.label}` : `Every message goes to ${fallback.label}`;
}

/** The active model at the top of the page, and the menu that switches it. */
export function ModelPicker({ models, selected, auto, loadingId, engineUp, onSelect }: Props) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const button = useRef<HTMLButtonElement>(null);
  const menu = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    (menu.current?.querySelector('[aria-selected="true"]') as HTMLElement | null)?.focus();
    const outside = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener('pointerdown', outside);
    return () => document.removeEventListener('pointerdown', outside);
  }, [open]);

  const choose = (id: string) => {
    setOpen(false);
    onSelect(id);
  };

  const onKeyDown = (event: KeyboardEvent) => {
    const options = [...(menu.current?.children ?? [])] as HTMLElement[];
    const at = options.indexOf(document.activeElement as HTMLElement);
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      options[(at + (event.key === 'ArrowDown' ? 1 : options.length - 1)) % options.length]?.focus();
    } else if (event.key === 'Escape') {
      event.stopPropagation();
      setOpen(false);
      button.current?.focus();
    } else if (event.key === 'Tab') {
      setOpen(false);
    }
  };

  const state = (model: ModelInfo): [string, string] => {
    if (engineUp(model) && !model.available) return ['Not installed', 'warn'];
    if (loadingId === model.id) return ['Loading…', ''];
    return [model.loaded ? 'In memory' : '', ''];
  };

  return (
    <div className="picker" ref={root}>
      <button
        id="modelBtn" ref={button} className="pick" type="button" data-accent={auto ? 'auto' : selected?.accent}
        aria-haspopup="listbox" aria-expanded={open} aria-label="Active model" disabled={!models.length}
        onClick={() => setOpen(value => !value)}
      >
        <span className="dot" />
        <span className="pick-name">{auto ? 'Auto' : selected?.label ?? 'No model'}</span>
        <span className="pick-vendor">{auto ? 'picks a model per message' : selected?.vendor}</span>
        <Icon.chevron />
      </button>
      <div id="modelMenu" ref={menu} className="menu" role="listbox" aria-label="Choose a model" hidden={!open} onKeyDown={onKeyDown}>
        <button type="button" className="opt" role="option" data-accent="auto" aria-selected={auto} onClick={() => choose(AUTO)}>
          <span className="dot" />
          <span className="opt-text">
            <span className="opt-name">Auto</span>
            <span className="opt-sub">{autoRule(models)}</span>
          </span>
          <span className="opt-state" />
          <span className="opt-check"><Icon.check /></span>
        </button>
        {models.map(model => {
          const [text, kind] = state(model);
          return (
            <button
              key={model.id} type="button" className="opt" role="option" data-accent={model.accent}
              aria-selected={!auto && model.id === selected?.id} onClick={() => choose(model.id)}
            >
              <span className="dot" />
              <span className="opt-text">
                <span className="opt-name">{model.label}</span>
                <span className="opt-sub">{[model.vendor, model.licence, model.provider_model].filter(Boolean).join(' · ')}</span>
              </span>
              <span className="opt-state" data-kind={kind}>{text}</span>
              <span className="opt-check"><Icon.check /></span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
