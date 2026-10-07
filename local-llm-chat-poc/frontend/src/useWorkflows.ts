import { useCallback, useEffect, useReducer, useRef, useState } from 'react';
import { computeWorkflow, fetchWorkflows, streamEvents, type Inputs } from './api';
import { addText, fail, finish, generation, interrupt, newReply } from './generation';
import type { InputValue, ModelInfo, Reply, RunEvent, Situation, WorkflowResult, WorkflowsResponse } from './types';

export type StepState = 'pending' | 'running' | 'done' | 'failed' | 'skipped';

/** What one workflow's page currently shows. */
export interface Panel {
  ran: boolean;
  running: boolean;
  result: WorkflowResult | null;
  resultKey: string; // the inputs the result was calculated from
  revealed: number; // calculation steps shown as done so far; the trace is revealed one step at a time
  modelStep: { state: StepState; detail: string } | null;
  brief: Reply | null; // the model's draft
  briefKey: string; // the inputs the draft was written from
  standard: string; // tag of the fixed-format brief when it is shown: '' | 'Standard format' | 'Standard format, no model'
  error: string;
}

const ALL = Number.POSITIVE_INFINITY;
const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
const emptyPanel = (): Panel => ({ ran: false, running: false, result: null, resultKey: '', revealed: ALL, modelStep: null, brief: null, briefKey: '', standard: '', error: '' });
const keyOf = (inputs: Inputs) => JSON.stringify(Object.entries(inputs).sort(([a], [b]) => a.localeCompare(b)));

/** Workflows state: the definitions from the backend, the input values, and each workflow's results and draft. */
export function useWorkflows(model: ModelInfo | undefined, onSettled: () => void) {
  const [data, setData] = useState<WorkflowsResponse | null>(null);
  const [loadError, setLoadError] = useState('');
  const [current, setCurrent] = useState('');
  const [language, setLanguage] = useState('');
  const [version, bump] = useReducer((n: number) => n + 1, 0);
  const inputs = useRef<Inputs>({});
  const situation = useRef<Situation | null>(null);
  const panels = useRef(new Map<string, Panel>());
  const frame = useRef(0);
  const pending = useRef<{ timer: number; controller: AbortController | null }>({ timer: 0, controller: null });
  const latest = useRef({ model, language, current, data, onSettled });
  latest.current = { model, language, current, data, onSettled };

  const load = useCallback(() => {
    fetchWorkflows()
      .then(response => {
        for (const workflow of response.workflows) if (!panels.current.has(workflow.id)) panels.current.set(workflow.id, emptyPanel());
        situation.current ??= response.situation;
        setData(response);
        setCurrent(id => id || response.workflows[0].id);
        setLanguage(name => name || response.languages[0]);
        setLoadError('');
      })
      .catch((error: Error) => setLoadError(error.message));
  }, []);
  useEffect(load, [load]);

  // While a draft streams in, repaint at most once per frame
  const touch = useCallback(() => {
    if (frame.current) return;
    frame.current = requestAnimationFrame(() => {
      frame.current = 0;
      bump();
    });
  }, []);

  /** The calculations again, with the inputs as they are now. No model involved. */
  const recompute = useCallback(async (id: string) => {
    const panel = panels.current.get(id);
    if (!panel) return;
    pending.current.controller?.abort();
    const controller = (pending.current.controller = new AbortController());
    const key = keyOf(inputs.current);
    try {
      const fresh = await computeWorkflow(id, inputs.current, controller.signal);
      situation.current = fresh.situation;
      if (panel.ran && !panel.running) {
        panel.result = fresh.result;
        panel.resultKey = key;
        panel.error = '';
      }
    } catch (error) {
      if (controller.signal.aborted) return;
      panel.error = error instanceof Error ? error.message : 'Request failed';
    }
    bump();
  }, []);

  const scheduleRecompute = useCallback(() => {
    clearTimeout(pending.current.timer);
    pending.current.timer = window.setTimeout(() => void recompute(latest.current.current), 120);
  }, [recompute]);

  const setInput = useCallback((path: string, value: InputValue) => {
    const fallback = latest.current.data?.workflows.flatMap(workflow => workflow.inputs).find(field => field.path === path)?.value;
    const next = { ...inputs.current };
    // A value put back to its scenario default is no longer an override
    if (JSON.stringify(value) === JSON.stringify(fallback)) delete next[path]; else next[path] = value;
    inputs.current = next;
    bump();
    scheduleRecompute();
  }, [scheduleRecompute]);

  const reset = useCallback(() => {
    inputs.current = {};
    bump();
    scheduleRecompute();
  }, [scheduleRecompute]);

  const show = useCallback((id: string) => {
    setCurrent(id);
    const panel = panels.current.get(id);
    if (panel?.ran && !panel.running && panel.resultKey !== keyOf(inputs.current)) void recompute(id);
  }, [recompute]);

  /**
   * Runs a workflow: the backend sends the calculations and references first, then streams the model's draft.
   * With reveal off (drafting again) the calculations on screen are refreshed without replaying the trace.
   */
  const run = useCallback(async (id: string, reveal = true) => {
    const { model, language, data, onSettled } = latest.current;
    const workflow = data?.workflows.find(entry => entry.id === id);
    const panel = panels.current.get(id);
    if (!workflow || !panel || panel.running || generation.owner) return;
    if (workflow.brief.enabled && !model) return;

    const sent = { ...inputs.current }, key = keyOf(sent);
    const reply = workflow.brief.enabled && model ? newReply(model, 'Waiting for the calculations') : null;
    Object.assign(panel, { ran: true, running: true, error: '', standard: '', brief: reply, modelStep: null });
    if (reveal) Object.assign(panel, { result: null, revealed: 0 });
    const controller = generation.begin(id);
    bump();

    let revealing: Promise<void> = Promise.resolve();
    const settle = () => {
      if (!reply) return;
      const name = reply.model.label;
      if (reply.status === 'done') panel.modelStep = { state: 'done', detail: `${name}${reply.tokens ? `, ${reply.tokens} tokens` : ''}` };
      else if (reply.status === 'stopped') panel.modelStep = { state: 'done', detail: `${name}, stopped early` };
      else {
        const cancelled = reply.status === 'cancelled';
        panel.modelStep = cancelled ? { state: 'skipped', detail: 'Stopped before the first word' } : { state: 'failed', detail: 'Model not available' };
        panel.standard = 'Standard format, no model';
        if (cancelled) panel.brief = null;
      }
    };

    try {
      const body = { inputs: sent, model: model?.id, language };
      for await (const event of streamEvents<RunEvent>(`/api/workflows/${id}/run`, body, controller.signal)) {
        if (event.type === 'result') {
          panel.result = event.result;
          panel.resultKey = key;
          situation.current = event.situation;
          const steps = event.result.steps.length;
          revealing = !reveal ? Promise.resolve() : (async () => {
            for (let index = 0; index < steps; index++) {
              panel.revealed = index;
              bump();
              await sleep(170);
            }
            panel.revealed = ALL;
            bump();
          })();
          bump();
        } else if (event.type === 'start') {
          await revealing;
          if (reply) {
            reply.waitLabel = '';
            reply.rev++;
            panel.modelStep = { state: 'running', detail: reply.model.label };
          }
          bump();
        } else if (event.type === 'delta') {
          if (reply) addText(reply, event.text);
          touch();
        } else if (event.type === 'done') {
          await revealing;
          if (event.skipped || !reply) panel.standard = 'Standard format';
          else finish(reply, event);
          settle();
        } else {
          await revealing;
          if (reply) fail(reply, event.message);
          settle();
        }
      }
      if (reply && (reply.status === 'waiting' || reply.status === 'streaming')) {
        fail(reply, 'The connection closed before the draft finished.');
        settle();
      }
    } catch (error) {
      await revealing;
      if (controller.signal.aborted) {
        if (reply) interrupt(reply);
      } else {
        const message = error instanceof Error ? error.message : 'Request failed';
        if (reply) fail(reply, message); else panel.error = message;
      }
      settle();
      // The request failed before anything arrived: the calculations do not need the model, so fetch them on their own
      if (!panel.result && !controller.signal.aborted) {
        try {
          const fresh = await computeWorkflow(id, sent);
          panel.result = fresh.result;
          panel.resultKey = key;
          situation.current = fresh.situation;
        } catch (failure) {
          panel.error = failure instanceof Error ? failure.message : 'Request failed';
        }
      }
    } finally {
      generation.end(controller);
      Object.assign(panel, { running: false, revealed: ALL, briefKey: key });
      if (!panel.result) panel.ran = false;
      bump();
      onSettled();
    }
  }, [touch]);

  return {
    data, loadError, reload: load, version,
    current, show,
    language, setLanguage,
    inputs: inputs.current, inputsKey: keyOf(inputs.current), setInput, reset,
    situation: situation.current,
    panel: (id: string) => panels.current.get(id) ?? emptyPanel(),
    run,
  };
}

export type Workflows = ReturnType<typeof useWorkflows>;
