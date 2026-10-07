import { useEffect, useRef, useState } from 'react';
import { generation } from '../generation';
import { inlineMd } from '../markdown';
import type { Card, Cell, Field, InputValue, ModelInfo, Pair, WorkflowInfo } from '../types';
import type { Panel, StepState, Workflows } from '../useWorkflows';
import { Icon } from './icons';
import { Markdown, Refs, ReplyView } from './ReplyView';

// ---------- inputs ----------

function NumberInput({ field, value, onChange }: { field: Field; value: number; onChange: (value: InputValue) => void }) {
  const [text, setText] = useState(String(value));
  const focused = useRef(false);
  // Follow changes made elsewhere (another workflow, Reset), but never fight the user's typing
  useEffect(() => {
    if (!focused.current) setText(String(value));
  }, [value]);
  return (
    <input
      type="number" min={field.min} max={field.max} step={field.step ?? 1} value={text}
      onFocus={() => { focused.current = true; }}
      onBlur={() => { focused.current = false; setText(String(value)); }}
      onChange={event => {
        setText(event.target.value);
        const number = parseFloat(event.target.value);
        if (Number.isFinite(number)) onChange(Math.min(field.max ?? number, Math.max(field.min ?? number, number)));
      }}
    />
  );
}

function InputField({ field, value, onChange }: { field: Field; value: InputValue; onChange: (value: InputValue) => void }) {
  if (field.type === 'chips') {
    const chosen = value as string[];
    return (
      <div className="in wide">
        <span>{field.label}</span>
        <div className="in-chips">
          {field.options.map(option => (
            <button
              key={option} type="button" aria-pressed={chosen.includes(option)}
              onClick={() => onChange(field.options.filter(name => (name === option ? !chosen.includes(name) : chosen.includes(name))))}
            >
              {option}
            </button>
          ))}
        </div>
      </div>
    );
  }
  return (
    <label className="in">
      <span>{field.label}</span>
      <div className="in-box">
        {field.type === 'number' ? (
          <NumberInput field={field} value={value as number} onChange={onChange} />
        ) : field.type === 'select' ? (
          <select value={value as string} onChange={event => onChange(event.target.value)}>
            {field.options.map(option => <option key={option} value={option}>{option[0].toUpperCase() + option.slice(1)}</option>)}
          </select>
        ) : (
          <input type={field.type} value={value as string} onChange={event => event.target.value && onChange(event.target.value)} />
        )}
        {field.unit && <em>{field.unit}</em>}
      </div>
    </label>
  );
}

// ---------- result cards ----------

const rich = (text: string) => ({ __html: inlineMd(text) });   // inlineMd escapes everything it is given

function CardView({ card, index }: { card: Card; index: number }) {
  const cell = (value: Cell, key: number) =>
    typeof value === 'string' ? <td key={key}>{value}</td> : <td key={key} data-tone={value.tone}>{value.text}</td>;
  return (
    <section className="card" style={{ animationDelay: `${index * 60}ms` }}>
      <div className="card-head"><h3>{card.title}</h3></div>
      {card.tiles && (
        <div className="tiles">
          {card.tiles.map(tile => <div key={tile.label} className="tile" data-tone={tile.tone || undefined}><b>{tile.value}</b><span>{tile.label}</span></div>)}
        </div>
      )}
      {card.strip && (
        <div className="strip">
          {card.strip.map(slot => <span key={slot.label} title={slot.title} data-tone={slot.good ? 'good' : undefined}>{slot.label}</span>)}
        </div>
      )}
      {card.note && <p className="card-note" dangerouslySetInnerHTML={rich(card.note)} />}
      {card.ranked && (
        <ol className="ranked">
          {card.ranked.map(([title, reason]) => <li key={title}><b>{title}</b><small>{reason}</small></li>)}
        </ol>
      )}
      {card.table && (
        <div className="table-wrap">
          <table>
            <thead><tr>{card.table.head.map((text, column) => <th key={column}>{text}</th>)}</tr></thead>
            <tbody>{card.table.rows.map((row, line) => <tr key={line}>{row.map(cell)}</tr>)}</tbody>
          </table>
        </div>
      )}
      {card.pre && <pre className="form">{card.pre}</pre>}
      {card.list && <ul className="points">{card.list.map(text => <li key={text} dangerouslySetInnerHTML={rich(text)} />)}</ul>}
    </section>
  );
}

// ---------- one workflow ----------

interface PanelProps {
  workflow: WorkflowInfo;
  panel: Panel;
  flows: Workflows;
  model: ModelInfo | undefined;
  generating: string | null;
  onOpenDoc: (id: string) => void;
}

function PanelView({ workflow, panel, flows, model, generating, onOpenDoc }: PanelProps) {
  const { result, brief } = panel;
  const canStop = panel.running && panel.modelStep?.state === 'running';   // the model step has started
  const blocked = !panel.running && generating !== null;
  const shown = !!result && panel.revealed >= result.steps.length;   // the trace has been revealed; show what it produced
  const stale = !!brief && !panel.running && !panel.standard && (brief.status === 'done' || brief.status === 'stopped') && panel.briefKey !== flows.inputsKey;

  // Steps: the calculation and search steps, then the model step if this workflow has one
  const steps: { label: string; kind: string; state: StepState; detail: string }[] = (result?.steps ?? workflow.plan.map(step => ({ ...step, detail: '' }))).map((step, index) => {
    const state: StepState = !panel.ran ? 'pending' : index < panel.revealed ? 'done' : index === panel.revealed ? 'running' : 'pending';
    return { ...step, state, detail: state === 'done' ? step.detail : '' };
  });
  if (workflow.brief.enabled) {
    steps.push({ label: 'Draft with the model', kind: 'Model', state: panel.modelStep?.state ?? 'pending', detail: panel.modelStep?.detail ?? '' });
  }

  const runLabel = canStop ? 'Stop' : panel.running ? 'Running…' : panel.ran ? 'Run again' : 'Run workflow';
  const onRun = () => (canStop ? generation.stop() : void flows.run(workflow.id));

  return (
    <div className="flow-col">
      <div className="scn">
        <b>{flows.data!.scenario.name}</b>
        <span className="scn-title">{flows.data!.scenario.title}</span>
        <span className="tag">Illustrative scenario, sample data</span>
        <button type="button" className="ghost" data-reset onClick={flows.reset}>Reset inputs</button>
      </div>

      <header className="flow-head">
        <div>
          <div className="flow-kicker">Output {workflow.letter}</div>
          <h1>{workflow.label}</h1>
          <p>{workflow.summary}</p>
        </div>
        <div className="flow-run">
          <label className="lang" hidden={!workflow.brief.enabled}>
            <span>Draft in</span>
            <select value={flows.language} onChange={event => flows.setLanguage(event.target.value)}>
              {flows.data!.languages.map(name => <option key={name}>{name}</option>)}
            </select>
          </label>
          <button
            type="button" className="btn primary" data-run onClick={onRun}
            disabled={panel.running ? !canStop : blocked || (workflow.brief.enabled && !model)}
            title={blocked ? 'The model is busy with another reply' : ''}
          >
            {runLabel}
          </button>
        </div>
      </header>

      <div className="inputs">
        {workflow.inputs.map(field => (
          <InputField key={field.path} field={field} value={flows.inputs[field.path] ?? field.value} onChange={value => flows.setInput(field.path, value)} />
        ))}
      </div>

      {panel.error && <div className="stale"><span>{panel.error}</span></div>}

      <ol className="steps">
        {steps.map(step => (
          <li key={step.label} className="step" data-state={step.state}>
            <span className="step-mark"><Icon.check /></span>
            <span className="step-label">{step.label}</span>
            <span className="step-detail">{step.detail}</span>
            <span className="step-kind">{step.kind}</span>
          </li>
        ))}
      </ol>

      {(brief || (panel.standard && shown)) && (
        <section className="brief">
          <div className="brief-head">
            <h3>{workflow.brief.title}</h3>
            <span className={panel.standard ? 'tag' : 'tag model'}>{panel.standard || 'Model draft'}</span>
          </div>
          {stale && (
            <div className="stale">
              <span>Inputs have changed since this draft.</span>
              <button type="button" className="btn small" data-redraft disabled={generating !== null} onClick={() => void flows.run(workflow.id, false)}>Draft again</button>
            </div>
          )}
          <div className="brief-body">
            {brief && <ReplyView reply={brief} rev={brief.rev} onOpenDoc={onOpenDoc} onRetry={() => void flows.run(workflow.id, false)} />}
            {panel.standard && result && <Markdown text={result.standard} />}
          </div>
        </section>
      )}

      {shown && result && (
        <>
          <Refs label="References from the sample corpus" refs={result.refs} none="No matching document." onOpen={onOpenDoc} />
          <div className="cards-head"><h3>Supporting calculations</h3><span className="tag">Calculated, no model</span></div>
          <div className="cards">{result.cards.map((card, index) => <CardView key={card.title} card={card} index={index} />)}</div>
        </>
      )}
    </div>
  );
}

// ---------- the view ----------

function Pairs({ title, pairs }: { title: string; pairs: Pair[] }) {
  return (
    <div className="side-sec">
      <h4>{title}</h4>
      {pairs.map(pair => <div key={pair.label} className="kv"><span>{pair.label}</span><b data-tone={pair.tone || undefined}>{pair.value}</b></div>)}
    </div>
  );
}

interface Props {
  hidden: boolean;
  flows: Workflows;
  model: ModelInfo | undefined;
  generating: string | null;
  onOpenDoc: (id: string) => void;
  onBrowse: () => void;
}

export function WorkflowsView({ hidden, flows, model, generating, onOpenDoc, onBrowse }: Props) {
  const main = useRef<HTMLDivElement>(null);
  const { data, situation } = flows;
  const workflow = data?.workflows.find(entry => entry.id === flows.current);
  const running = workflow ? flows.panel(workflow.id).running : false;

  // A new page, or a new run, starts from the top
  useEffect(() => {
    main.current?.scrollTo({ top: 0, behavior: running ? 'smooth' : 'auto' });
  }, [flows.current, running]);

  if (hidden || !data || !workflow) {
    return (
      <div id="flowView" className="flow" hidden={hidden}>
        {!hidden && <div className="flow-main"><div className="flow-col"><p className="card-note">{flows.loadError || 'Loading the workflows…'}</p></div></div>}
      </div>
    );
  }

  return (
    <div id="flowView" className="flow">
      <nav id="flowRail" className="flow-rail" aria-label="Workflows">
        <div className="rail-title">Workflows</div>
        {data.workflows.map(entry => {
          const panel = flows.panel(entry.id);
          return (
            <button key={entry.id} type="button" className="rail-item" data-flow={entry.id} aria-current={entry.id === workflow.id} onClick={() => flows.show(entry.id)}>
              <span className="rail-letter">{entry.letter}</span>
              <span>{entry.label}</span>
              <span className="rail-state" data-state={panel.running ? 'running' : panel.ran ? 'done' : ''} />
            </button>
          );
        })}
      </nav>

      <div id="flowMain" className="flow-main" ref={main}>
        <PanelView key={workflow.id} workflow={workflow} panel={flows.panel(workflow.id)} flows={flows} model={model} generating={generating} onOpenDoc={onOpenDoc} />
      </div>

      <aside id="flowSide" className="flow-side" aria-label="Situation">
        <div className="side-head"><b>Situation</b><span>as of {situation?.as_of}</span></div>
        <div className="side-note">Sample data for the exercise. The indicators are calculated from it.</div>
        {situation && (
          <>
            <Pairs title="Indicators" pairs={situation.indicators} />
            <div className="side-sec">
              <h4>Reports</h4>
              {situation.reports.map(report => (
                <div key={report.time} className="report" data-urgent={!!report.urgent}>{report.time}&nbsp; {report.text}<small>{report.source}</small></div>
              ))}
            </div>
            <Pairs title="Own forces" pairs={situation.forces} />
            <Pairs title="Logistics" pairs={situation.logistics} />
          </>
        )}
        <div className="side-sec">
          <h4>Sample corpus</h4>
          <button type="button" className="ghost" onClick={onBrowse}>Browse {data.corpus_count} documents</button>
        </div>
      </aside>
    </div>
  );
}
