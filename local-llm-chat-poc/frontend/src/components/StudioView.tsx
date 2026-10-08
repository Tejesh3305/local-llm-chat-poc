import { useEffect, useReducer, useRef, useState, type ChangeEvent, type DragEvent, type ReactNode } from 'react';
import { exportFile, fetchInsights, fetchOutline, makeImage, saveFile, streamEvents } from '../api';
import { addText, fail, finish, generation, interrupt, newReply } from '../generation';
import { applyFilter, FILTERS, loadImage, measure, toCanvas, type Filter } from '../imageTools';
import type { DemoImage, DocumentInfo, DocumentInsights, ExportFormat, ModelInfo, OutlineResponse, Reply, StudioEvent } from '../types';
import { Icon } from './icons';
import { ReplyView } from './ReplyView';

type Tool = 'pdf' | 'docx' | 'pptx' | 'image' | 'vision' | 'doclab';

const TOOLS: { id: Tool; group: string; label: string; icon: () => ReactNode }[] = [
  { id: 'pdf', group: 'Create', label: 'PDF report', icon: Icon.file },
  { id: 'docx', group: 'Create', label: 'Word document', icon: Icon.pencil },
  { id: 'pptx', group: 'Create', label: 'PowerPoint deck', icon: Icon.slides },
  { id: 'image', group: 'Create', label: 'Image generation', icon: Icon.wand },
  { id: 'vision', group: 'Analyse', label: 'Image processing', icon: Icon.image },
  { id: 'doclab', group: 'Analyse', label: 'Document processing', icon: Icon.scan },
];

interface Props {
  hidden: boolean;
  model: ModelInfo | undefined; // the selected model; with Auto, the default one
  models: ModelInfo[];
  generating: string | null;
  documents: DocumentInfo[];
  uploading: boolean;
  onUpload: (file: File) => void;
}

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
const noDoc = () => {};
const fileSize = (bytes: number) => (bytes < 1_000_000 ? `${Math.max(1, Math.round(bytes / 1000))} KB` : `${(bytes / 1_000_000).toFixed(1)} MB`);
const slug = (text: string) => text.replace(/[^A-Za-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 60) || 'document';
const FORMAT_NAMES: Record<ExportFormat, string> = { pdf: 'PDF', docx: 'Word', pptx: 'PowerPoint' };

/** **bold** as <b>, other marks dropped: enough for the previews. */
function inline(text: string): ReactNode[] {
  return text.replace(/`([^`]*)`/g, '$1').split(/\*\*([^*]+)\*\*/).map((piece, index) => (index % 2 ? <b key={index}>{piece}</b> : piece.replace(/\*/g, '')));
}

/** A request to the model whose reply streams in: drafts, image and document analysis. */
function useModelRun(models: ModelInfo[], fallback: ModelInfo | undefined) {
  const [reply, setReply] = useState<Reply | null>(null);
  const [meta, setMeta] = useState<{ task?: string; read?: number }>({});
  const [, bump] = useReducer((n: number) => n + 1, 0);

  const run = async (path: string, body: unknown, onText?: (text: string) => void): Promise<Reply | null> => {
    if (!fallback || generation.owner) return null;
    const current = newReply(fallback);
    setReply(current);
    setMeta({});
    const controller = generation.begin('studio');
    try {
      for await (const event of streamEvents<StudioEvent>(path, body, controller.signal)) {
        if (event.type === 'start') {
          const chosen = models.find(entry => entry.id === event.model);
          if (chosen && chosen.id !== current.model.id) {
            current.model = chosen;
            current.cold = chosen.loaded === false;
          }
          setMeta({ task: event.task, read: event.read_percent });
          current.rev++;
        } else if (event.type === 'delta') {
          addText(current, event.text);
          onText?.(current.content);
        } else if (event.type === 'done') {
          finish(current, event);
        } else {
          fail(current, event.message);
        }
        bump();
      }
      if (current.status === 'waiting' || current.status === 'streaming') fail(current, 'The connection closed before the reply finished.');
    } catch (error) {
      if (controller.signal.aborted) interrupt(current);
      else fail(current, error instanceof Error ? error.message : 'Request failed');
    } finally {
      generation.end(controller);
      bump();
    }
    return current;
  };
  return { reply, meta, run, clear: () => setReply(null) };
}

/** Export any text the model wrote as a file. */
function ExportButtons({ title, content }: { title: string; content: string }) {
  const [busy, setBusy] = useState<ExportFormat | null>(null);
  const [error, setError] = useState('');
  const save = async (format: ExportFormat) => {
    setBusy(format);
    setError('');
    try {
      saveFile(await exportFile(format, title, content), `${slug(title)}.${format}`);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Export failed');
    } finally {
      setBusy(null);
    }
  };
  return (
    <div className="studio-row">
      <span className="refs-label">Save as</span>
      {(['pdf', 'docx', 'pptx'] as ExportFormat[]).map(format => (
        <button key={format} type="button" className="btn small" disabled={busy !== null} onClick={() => void save(format)}>
          <Icon.download />{busy === format ? 'Preparing…' : FORMAT_NAMES[format]}
        </button>
      ))}
      {error && <span className="studio-error">{error}</span>}
    </div>
  );
}

function Head({ kicker, title, children, badge }: { kicker: string; title: string; children: ReactNode; badge: ReactNode }) {
  return (
    <div className="flow-head">
      <div>
        <div className="flow-kicker">{kicker}</div>
        <h1>{title}</h1>
        <p>{children}</p>
      </div>
      <div className="studio-badge">{badge}</div>
    </div>
  );
}

// ---------- PDF, Word and PowerPoint ----------

const SAMPLES: Record<ExportFormat, { title: string; content: string; topic: string }> = {
  pdf: {
    title: 'Winter Movement Assessment',
    topic: 'Assessment of winter road conditions on Axis Bravo for a convoy move next week: snowfall, bridge limits, avalanche risk and a recommendation.',
    content: `## Situation
Heavy snowfall of about **60 cm** is forecast over the next 24 hours. Axis Alpha remains blocked by avalanche debris.

## Findings
- Bridge 52B is classified for loads up to **30 t**; heavier vehicles need the bypass
- Snow vehicles are needed to lead the convoy on Axis Bravo
- Visibility in the ridge sector falls to 200 m before 0900 h

## Recommendation
Move the main body at **0530 h** on Axis Bravo with snow vehicles leading, and hold heavy vehicles until the bypass is cleared.`,
  },
  docx: {
    title: 'Movement Order: Convoy to Forward Base',
    topic: 'A formal movement order for a convoy of 12 vehicles to the forward base, with timings, route, responsibilities and communication instructions.',
    content: `## Subject
Movement of a convoy of 12 vehicles to the forward base.

## Instructions
The convoy will assemble at the transit camp at **0500 h** and depart at **0530 h** by Axis Bravo.

- Lead vehicle: snow clearance team
- Convoy commander: [placeholder]
- Halt at checkpoint 3 for a vehicle check

## Communication
Report departure, the halt and arrival on the HF net. Use the alternate net if the primary is not answering.`,
  },
  pptx: {
    title: 'Operations Briefing: Sector 4',
    topic: 'A short operations briefing for Sector 4: situation, weather, own forces, plan and coordination points.',
    content: `## Situation
- Activity reported near Grid 47B
- Forward elements observed along the ridge

## Weather
- Night minimum around −27 °C
- Flying window 0900 to 1400 h

## Plan
- Reserve moves to the ridge by 1200 h
- Casualty evacuation by helicopter from LIMA

## Coordination
- Hotline message to the counterpart by 0800 h
- Next update at 1800 h`,
  },
};

const KINDS: Record<ExportFormat, { kind: string; what: string; blurb: string }> = {
  pdf: { kind: 'report', what: 'report', blurb: 'Assessments and situation reports as a print-ready PDF, with header, page numbers and the unit marking.' },
  docx: { kind: 'letter', what: 'document', blurb: 'Letters, orders and memos as an editable Word file, using real headings and bullet styles so it can be revised.' },
  pptx: { kind: 'briefing', what: 'briefing', blurb: 'A briefing deck: a title slide, one slide per section with its points, and a closing slide.' },
};

function DocumentMaker({ format, hidden, model, models, generating }: { format: ExportFormat; hidden: boolean; model: ModelInfo | undefined; models: ModelInfo[]; generating: string | null }) {
  const sample = SAMPLES[format];
  const [topic, setTopic] = useState(sample.topic);
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [result, setResult] = useState<{ blob: Blob; url: string; outline: OutlineResponse } | null>(null);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState('');
  const draft = useModelRun(models, model);
  const drafting = generating === 'studio' && draft.reply?.status !== 'done' && !!draft.reply && ['waiting', 'streaming'].includes(draft.reply.status);
  const name = FORMAT_NAMES[format];

  useEffect(() => () => { if (result) URL.revokeObjectURL(result.url); }, [result]);

  const write = async () => {
    setContent('');
    setTitle('');
    const done = await draft.run('/api/studio/draft', { topic, kind: KINDS[format].kind, model: model?.id }, setContent);
    if (done && done.status === 'done') setContent(done.content);
  };

  const generate = async () => {
    setWorking(true);
    setError('');
    try {
      const [blob, outline] = await Promise.all([exportFile(format, title, content), fetchOutline(format, title, content)]);
      setResult({ blob, url: URL.createObjectURL(blob), outline });
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'The file could not be made.');
    } finally {
      setWorking(false);
    }
  };

  return (
    <div className="flow-col" hidden={hidden}>
      <Head kicker={`Create · ${name}`} title={`${name} generation`} badge={<span className="tag model">Draft by {model?.label ?? 'the model'} · file built on this machine</span>}>
        {KINDS[format].blurb}
      </Head>

      <section className="card">
        <div className="card-head"><h3>1. Draft with AI <small className="side-note">optional</small></h3></div>
        <p className="card-note">Describe what the {KINDS[format].what} should cover. The model writes it as structured text below, which you can edit before making the file.</p>
        <textarea className="studio-text" rows={3} value={topic} onChange={event => setTopic(event.target.value)} aria-label="What to write about" />
        <div className="studio-row">
          {drafting ? (
            <button type="button" className="btn" onClick={() => generation.stop()}><Icon.stop />Stop</button>
          ) : (
            <button type="button" className="btn primary" disabled={generating !== null || topic.trim().length < 3 || !model} onClick={() => void write()}>
              <Icon.wand />Draft with {model?.label ?? 'the model'}
            </button>
          )}
          {draft.reply && draft.reply.status !== 'waiting' && draft.reply.content && (
            <span className="side-note">
              {draft.reply.status === 'streaming' ? `Writing… ${draft.reply.content.split(/\s+/).length} words` : draft.reply.tokensPerSec ? `Drafted at ${draft.reply.tokensPerSec.toFixed(1)} tokens/s` : 'Drafted'}
            </span>
          )}
          {draft.reply?.status === 'waiting' && <span className="side-note shimmer">{draft.reply.cold ? `Loading ${draft.reply.model.label}` : 'Thinking'}</span>}
          {draft.reply?.status === 'error' && <span className="studio-error">{draft.reply.error}</span>}
        </div>
      </section>

      <section className="card">
        <div className="card-head">
          <h3>2. Content</h3>
          <button type="button" className="ghost" onClick={() => { setTitle(sample.title); setContent(sample.content); }}>Load sample</button>
        </div>
        <input className="studio-input" value={title} placeholder="Title (or start the text with '# Title')" onChange={event => setTitle(event.target.value)} aria-label="Title" />
        <textarea
          className="studio-text mono" rows={12} value={content} onChange={event => setContent(event.target.value)} aria-label="Content"
          placeholder={'## Section heading\nA paragraph of text, with **bold** words.\n- A bullet point\n- Another point'}
        />
        <p className="card-note">## starts a section{format === 'pptx' ? ' (one slide each)' : ''}, - starts a bullet point, **text** is bold.</p>
        <div className="studio-row">
          <button type="button" className="btn primary" disabled={!content.trim() || working || drafting} onClick={() => void generate()}>
            <Icon.file />{working ? 'Making the file…' : `Generate ${name}`}
          </button>
          {error && <span className="studio-error">{error}</span>}
        </div>
      </section>

      {result && (
        <section className="card">
          <div className="card-head">
            <h3>3. Result</h3>
            <span className="studio-row">
              <span className="side-note">{result.outline.filename} · {fileSize(result.blob.size)}</span>
              <button type="button" className="btn primary small" onClick={() => saveFile(result.blob, result.outline.filename)}><Icon.download />Download</button>
            </span>
          </div>
          {format === 'pdf' && <iframe className="pdf-frame" src={`${result.url}#toolbar=0&view=FitH`} title="PDF preview" />}
          {format === 'docx' && (
            <div className="page-preview">
              <div className="page">
                <div className="page-mark">SAINIK AI · 14 Corps Battlefield Intelligence System</div>
                <h2>{result.outline.title}</h2>
                {result.outline.sections.map((section, index) => (
                  <div key={index}>
                    {section.heading && <h3>{inline(section.heading)}</h3>}
                    {section.blocks.map(([kind, text], at) => (kind === 'bullet' ? <li key={at}>{inline(text)}</li> : <p key={at}>{inline(text)}</p>))}
                  </div>
                ))}
              </div>
            </div>
          )}
          {format === 'pptx' && (
            <div className="slide-grid">
              {result.outline.slides.map((slide, index) => (
                <div key={index} className="slide" data-kind={slide.kind}>
                  {slide.kind === 'content' ? (
                    <>
                      <div className="slide-bar">{slide.title}</div>
                      <ul>{slide.points?.map((point, at) => <li key={at}>{point}</li>)}</ul>
                    </>
                  ) : (
                    <>
                      <small>SAINIK AI</small>
                      <b>{slide.title}</b>
                      <span>{slide.subtitle}</span>
                    </>
                  )}
                  <i>{index + 1}</i>
                </div>
              ))}
            </div>
          )}
        </section>
      )}
    </div>
  );
}

// ---------- image generation (demo) ----------

const IMAGE_STYLES = ['terrain map', 'landscape', 'poster'];
const IMAGE_IDEAS = ['Snow-covered Himalayan ridge at dawn', 'Night patrol route through a forest valley', 'Desert convoy route with checkpoints', 'Winter readiness poster for Sector 4'];
const STAGES = ['Reading the prompt', 'Composing the scene', 'Rendering'];

async function svgToPng(image: DemoImage): Promise<Blob> {
  const picture = await loadImage(`data:image/svg+xml;charset=utf-8,${encodeURIComponent(image.svg)}`);
  const canvas = document.createElement('canvas');
  canvas.width = image.width;
  canvas.height = image.height;
  canvas.getContext('2d')!.drawImage(picture, 0, 0);
  return new Promise((resolve, reject) => canvas.toBlob(blob => (blob ? resolve(blob) : reject(new Error('PNG export failed'))), 'image/png'));
}

function ImageMaker({ hidden }: { hidden: boolean }) {
  const [prompt, setPrompt] = useState(IMAGE_IDEAS[0]);
  const [style, setStyle] = useState('landscape');
  const [stage, setStage] = useState(-1);
  const [current, setCurrent] = useState<(DemoImage & { prompt: string }) | null>(null);
  const [gallery, setGallery] = useState<(DemoImage & { prompt: string })[]>([]);
  const [error, setError] = useState('');

  const generate = async () => {
    setError('');
    setStage(0);
    try {
      const request = makeImage(prompt, style);
      for (let at = 0; at < STAGES.length; at++) {   // a short staged progress, as a real generator would report
        setStage(at);
        await sleep(550);
      }
      const image = { ...(await request), prompt };
      setCurrent(image);
      setGallery(list => [image, ...list.filter(entry => entry.svg !== image.svg)].slice(0, 8));
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'The image could not be made.');
    } finally {
      setStage(-1);
    }
  };

  const url = (image: DemoImage) => `data:image/svg+xml;charset=utf-8,${encodeURIComponent(image.svg)}`;

  return (
    <div className="flow-col" hidden={hidden}>
      <Head kicker="Create · Image" title="Image generation" badge={<span className="tag">Demo · drawn by code, no image model</span>}>
        Turns a short description into a picture. This demo draws terrain maps, landscapes and posters from the prompt's words (snow, night, desert, forest…); a real deployment would plug an image model in here.
      </Head>
      <section className="card">
        <textarea className="studio-text" rows={2} value={prompt} maxLength={500} onChange={event => setPrompt(event.target.value)} aria-label="Describe the picture" />
        <div className="studio-row">
          {IMAGE_IDEAS.map(idea => <button key={idea} type="button" className="chip-btn" onClick={() => setPrompt(idea)}>{idea}</button>)}
        </div>
        <div className="studio-row">
          <div className="seg">
            {IMAGE_STYLES.map(name => <button key={name} type="button" aria-pressed={style === name} onClick={() => setStyle(name)}>{name[0].toUpperCase() + name.slice(1)}</button>)}
          </div>
          <span className="tools-gap" />
          <button type="button" className="btn primary" disabled={stage >= 0 || prompt.trim().length < 2} onClick={() => void generate()}><Icon.wand />Generate image</button>
        </div>
        {stage >= 0 && (
          <div className="progress" aria-live="polite">
            <div className="progress-bar"><span style={{ width: `${((stage + 1) / STAGES.length) * 100}%` }} /></div>
            <span className="side-note shimmer">{STAGES[stage]}…</span>
          </div>
        )}
        {error && <span className="studio-error">{error}</span>}
      </section>
      {current && (
        <section className="card">
          <img className="studio-image" src={url(current)} alt={current.prompt} />
          <div className="studio-row">
            <span className="tag">{current.style}</span><span className="tag">palette: {current.palette}</span><span className="tag">seed {current.seed}</span>
            <span className="tools-gap" />
            <button type="button" className="btn small" onClick={() => saveFile(new Blob([current.svg], { type: 'image/svg+xml' }), `${slug(current.prompt)}.svg`)}><Icon.download />SVG</button>
            <button type="button" className="btn small" onClick={() => void svgToPng(current).then(blob => saveFile(blob, `${slug(current.prompt)}.png`), (failure: Error) => setError(failure.message))}><Icon.download />PNG</button>
          </div>
        </section>
      )}
      {gallery.length > 1 && (
        <section>
          <div className="cards-head"><h3>This session</h3></div>
          <div className="thumbs">
            {gallery.map(image => (
              <button key={image.svg.length + image.prompt + image.style} type="button" className="thumb" aria-pressed={image === current} title={image.prompt} onClick={() => setCurrent(image)}>
                <img src={url(image)} alt="" />
              </button>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

// ---------- image processing ----------

const IMAGE_TASKS: { id: string; label: string; detail: string }[] = [
  { id: 'describe', label: 'Describe', detail: 'What the picture shows, in detail' },
  { id: 'summary', label: 'Report summary', detail: 'Bullet points ready for a report' },
  { id: 'text', label: 'Read text', detail: 'Signs, labels and numbers in the picture' },
];

function ImageLab({ hidden, model, models, generating }: { hidden: boolean; model: ModelInfo | undefined; models: ModelInfo[]; generating: string | null }) {
  const [picture, setPicture] = useState<{ name: string; size: number; url: string; image: HTMLImageElement; facts: { width: number; height: number; brightness: number; colour: string } } | null>(null);
  const [filtered, setFiltered] = useState<{ filter: Filter; url: string } | null>(null);
  const [error, setError] = useState('');
  const [over, setOver] = useState(false);
  const reader = model?.tags.includes('vision') ? model : models.find(entry => entry.tags.includes('vision')) ?? model;
  const analysis = useModelRun(models, reader);
  const working = generating === 'studio' && !!analysis.reply && ['waiting', 'streaming'].includes(analysis.reply.status);

  const open = async (file: File | undefined) => {
    if (!file) return;
    setError('');
    if (!/^image\/(png|jpeg|webp|gif|bmp)$/.test(file.type)) {
      setError('Pick a PNG, JPEG, WebP, GIF or BMP picture.');
      return;
    }
    try {
      const url = URL.createObjectURL(file);
      const image = await loadImage(url);
      const facts = measure(toCanvas(image, 400));
      if (picture) URL.revokeObjectURL(picture.url);
      setPicture({ name: file.name, size: file.size, url, image, facts: { width: image.naturalWidth, height: image.naturalHeight, ...facts } });
      setFiltered(null);
      analysis.clear();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'The picture could not be opened.');
    }
  };

  const onDrop = (event: DragEvent) => {
    event.preventDefault();
    setOver(false);
    void open(event.dataTransfer.files[0]);
  };

  const filter = (id: Filter) => {
    if (!picture) return;
    const canvas = applyFilter(toCanvas(picture.image, 1600), id);
    setFiltered({ filter: id, url: canvas.toDataURL('image/png') });
  };

  const analyse = (task: string) => {
    if (!picture) return;
    const image = toCanvas(picture.image, 1024).toDataURL('image/jpeg', 0.85);   // small enough for the model, still sharp
    void analysis.run('/api/studio/image-analysis', { image, task, model: reader?.id });
  };

  return (
    <div className="flow-col" hidden={hidden}>
      <Head kicker="Analyse · Image" title="Image processing" badge={<span className="tag model">AI by {reader?.label ?? 'a vision model'} · filters on this machine</span>}>
        Open a photo, scan or screenshot. The model describes it, writes a report summary or reads the text in it; the filters enhance and transform it in the browser.
      </Head>
      <label className="drop" data-over={over} onDragOver={event => { event.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)} onDrop={onDrop}>
        <Icon.image />
        <span>{picture ? <><b>{picture.name}</b> · {picture.facts.width} × {picture.facts.height} px · {fileSize(picture.size)}. Drop or click to pick another.</> : <>Drop a picture here, or click to pick one</>}</span>
        <input type="file" accept="image/png,image/jpeg,image/webp,image/gif,image/bmp" onChange={(event: ChangeEvent<HTMLInputElement>) => { void open(event.target.files?.[0]); event.target.value = ''; }} />
      </label>
      {error && <div className="stale">{error}</div>}

      {picture && (
        <>
          <div className="compare">
            <figure><img src={picture.url} alt="Original" /><figcaption>Original</figcaption></figure>
            <figure>
              {filtered ? <img src={filtered.url} alt={filtered.filter} /> : <div className="compare-empty">Pick a filter to see the result here</div>}
              <figcaption>{filtered ? FILTERS.find(entry => entry.id === filtered.filter)?.label : 'Processed'}</figcaption>
            </figure>
          </div>
          <div className="studio-row">
            <span className="tag">brightness {picture.facts.brightness}%</span>
            <span className="tag"><span className="swatch" style={{ background: picture.facts.colour }} />average colour {picture.facts.colour}</span>
          </div>

          <section className="card">
            <div className="card-head"><h3>Enhance and transform</h3><span className="tag">On this machine, no model</span></div>
            <div className="task-grid">
              {FILTERS.map(entry => (
                <button key={entry.id} type="button" className="task" aria-pressed={filtered?.filter === entry.id} onClick={() => filter(entry.id)}>
                  <b>{entry.label}</b><small>{entry.detail}</small>
                </button>
              ))}
            </div>
            {filtered && (
              <div className="studio-row">
                <button type="button" className="btn small" onClick={() => void fetch(filtered.url).then(r => r.blob()).then(blob => saveFile(blob, `${slug(picture.name.replace(/\.\w+$/, ''))}-${filtered.filter}.png`))}>
                  <Icon.download />Download result
                </button>
              </div>
            )}
          </section>

          <section className="card">
            <div className="card-head"><h3>Ask the AI about the picture</h3><span className="tag model">{reader?.label}</span></div>
            <div className="task-grid">
              {IMAGE_TASKS.map(task => (
                <button key={task.id} type="button" className="task" disabled={generating !== null} onClick={() => analyse(task.id)}>
                  <b>{task.label}</b><small>{task.detail}</small>
                </button>
              ))}
            </div>
            {working && <button type="button" className="btn small" onClick={() => generation.stop()}><Icon.stop />Stop</button>}
            {analysis.reply && <ReplyView reply={analysis.reply} rev={analysis.reply.rev} onOpenDoc={noDoc} />}
            {analysis.reply?.status === 'done' && <ExportButtons title={`${analysis.meta.task ?? 'Image analysis'} - ${picture.name}`} content={analysis.reply.content} />}
          </section>
        </>
      )}
    </div>
  );
}

// ---------- document processing ----------

const DOC_TASKS: { id: string; label: string; detail: string }[] = [
  { id: 'summary', label: 'Executive summary', detail: 'The document in a few sentences, with a bottom line' },
  { id: 'key_points', label: 'Key points', detail: 'The most important points, most important first' },
  { id: 'entities', label: 'People, places and dates', detail: 'Names, organisations, places, dates and figures' },
  { id: 'actions', label: 'Actions and risks', detail: 'Tasks, deadlines and responsibilities, then risks' },
  { id: 'hindi', label: 'Summary in Hindi', detail: 'हिन्दी में सारांश' },
];

function DocLab({ hidden, model, models, generating, documents, uploading, onUpload }: Props) {
  const [chosen, setChosen] = useState('');
  const [facts, setFacts] = useState<DocumentInsights | null>(null);
  const [error, setError] = useState('');
  const analysis = useModelRun(models, model);
  const working = generating === 'studio' && !!analysis.reply && ['waiting', 'streaming'].includes(analysis.reply.status);
  const doc = documents.find(entry => entry.id === chosen);
  const count = useRef(documents.length);

  // Pick the newest document when one is added, and drop a choice whose document was removed
  useEffect(() => {
    if (documents.length > count.current || (!doc && documents.length)) setChosen(documents[0].id);
    if (!documents.length) setChosen('');
    count.current = documents.length;
  }, [documents]);   // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    setFacts(null);
    analysis.clear();
    if (!chosen) return;
    fetchInsights(chosen).then(setFacts, (failure: Error) => setError(failure.message));
  }, [chosen]);   // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="flow-col" hidden={hidden}>
      <Head kicker="Analyse · Document" title="Document processing" badge={<span className="tag model">AI by {model?.label ?? 'the model'}</span>}>
        Pick an uploaded document. The figures below are counted straight from the text; the AI then writes a summary, the key points, the names and dates, or the actions it sets out, ready to save as a file.
      </Head>
      <section className="card">
        <div className="studio-row">
          <select className="studio-input grow" value={chosen} onChange={event => setChosen(event.target.value)} disabled={!documents.length} aria-label="Document">
            {!documents.length && <option value="">No documents yet</option>}
            {documents.map(entry => <option key={entry.id} value={entry.id}>{entry.filename} ({fileSize(entry.size_bytes)})</option>)}
          </select>
          <label className="btn" data-busy={uploading}>
            <Icon.clip />{uploading ? 'Reading the file…' : 'Upload'}
            <input type="file" hidden accept=".pdf,.docx,.txt,.md" disabled={uploading} onChange={event => { const file = event.target.files?.[0]; event.target.value = ''; if (file) onUpload(file); }} />
          </label>
        </div>
        {error && <span className="studio-error">{error}</span>}
        {facts && doc && (
          <>
            <div className="tiles">
              <div className="tile"><b>{facts.words.toLocaleString()}</b><span>words</span></div>
              <div className="tile"><b>{facts.reading_minutes} min</b><span>to read</span></div>
              <div className="tile"><b>{doc.chunks}</b><span>{doc.chunks === 1 ? 'part' : 'parts'} indexed for search</span></div>
              <div className="tile"><b>{facts.numbers}</b><span>figures mentioned</span></div>
            </div>
            <div className="studio-row"><span className="refs-label">Frequent terms</span>{facts.keywords.map(word => <span key={word} className="tag">{word}</span>)}</div>
            {facts.dates.length > 0 && <div className="studio-row"><span className="refs-label">Dates</span>{facts.dates.map(date => <span key={date} className="tag">{date}</span>)}</div>}
          </>
        )}
      </section>

      {doc && (
        <section className="card">
          <div className="card-head"><h3>Ask the AI</h3><span className="tag model">{model?.label}</span></div>
          <div className="task-grid">
            {DOC_TASKS.map(task => (
              <button key={task.id} type="button" className="task" disabled={generating !== null}
                onClick={() => void analysis.run('/api/studio/document-analysis', { document_id: doc.id, task: task.id, model: model?.id })}>
                <b>{task.label}</b><small>{task.detail}</small>
              </button>
            ))}
          </div>
          {working && <button type="button" className="btn small" onClick={() => generation.stop()}><Icon.stop />Stop</button>}
          {analysis.meta.read != null && analysis.meta.read < 100 && (
            <div className="stale">The document is long, so the model read the first {analysis.meta.read}% of it.</div>
          )}
          {analysis.reply && <ReplyView reply={analysis.reply} rev={analysis.reply.rev} onOpenDoc={noDoc} />}
          {analysis.reply?.status === 'done' && <ExportButtons title={`${analysis.meta.task ?? 'Analysis'} - ${doc.filename.replace(/\.\w+$/, '')}`} content={analysis.reply.content} />}
        </section>
      )}
    </div>
  );
}

// ---------- the view ----------

export function StudioView(props: Props) {
  const { hidden, model, models, generating } = props;
  const [tool, setTool] = useState<Tool>('pdf');
  const groups = [...new Set(TOOLS.map(entry => entry.group))];

  return (
    <div id="studioView" className="view" hidden={hidden}>
      <div className="flow studio">
        <nav className="flow-rail" aria-label="AI Studio tools">
          {groups.map(group => (
            <div key={group} className="rail-group">
              <div className="rail-title">{group}</div>
              {TOOLS.filter(entry => entry.group === group).map(entry => (
                <button key={entry.id} type="button" className="rail-item" aria-current={tool === entry.id} onClick={() => setTool(entry.id)}>
                  <span className="rail-letter"><entry.icon /></span>{entry.label}
                </button>
              ))}
            </div>
          ))}
          <p className="rail-empty rail-gap">Everything runs on this machine. Generated files and pictures are not kept after the page is closed.</p>
        </nav>
        <main className="flow-main">
          {(['pdf', 'docx', 'pptx'] as ExportFormat[]).map(format => (
            <DocumentMaker key={format} format={format} hidden={tool !== format} model={model} models={models} generating={generating} />
          ))}
          <ImageMaker hidden={tool !== 'image'} />
          <ImageLab hidden={tool !== 'vision'} model={model} models={models} generating={generating} />
          <DocLab {...props} hidden={tool !== 'doclab'} />
        </main>
      </div>
    </div>
  );
}
