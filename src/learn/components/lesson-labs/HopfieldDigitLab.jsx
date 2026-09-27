import RemoteCodeBlock from '../content/RemoteCodeBlock.jsx';
import { useEffect, useMemo, useRef, useState } from 'react';
import { NeuralLab, NeuralTable } from './NeuralLessonElements.jsx';
import { NeuralNumber } from './NeuralNumberControl.jsx';

import { Bars, PixelImage, Values, f, vec } from './HopfieldPrimitives.jsx';
import { digitRead, prepareDigitBank, digitMse, occludeDigit } from '../../data/hopfield-memory-models.js';
const base = '/learn-code/modern-hopfield-networks/';
function useNear() {
  const ref = useRef(null),
    [near, setNear] = useState(false);
  useEffect(() => {
    if (!ref.current) return;
    if (!globalThis.IntersectionObserver) {
      setNear(true);
      return;
    }
    const observer = new IntersectionObserver(entries => {
      if (entries.some(e => e.isIntersecting)) {
        setNear(true);
        observer.disconnect();
      }
    }, {
      rootMargin: '350px'
    });
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  return [ref, near];
}
function useAsset(name, enabled, retry) {
  const [state, setState] = useState({});
  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    setState({
      name
    });
    fetch(base + name, {
      signal: controller.signal
    }).then(r => {
      if (!r.ok) throw Error('asset');
      return r.json();
    }).then(data => setState({
      name,
      data
    })).catch(e => {
      if (e.name !== 'AbortError') setState({
        name,
        error: true
      });
    });
    return () => controller.abort();
  }, [name, enabled, retry]);
  return state.name === name ? state : {};
}
export function DigitReadView({
  bank,
  model,
  raw,
  original,
  label = 'Current cue',
  showWeights = true
}) {
  const prepared = useMemo(() => prepareDigitBank(bank, model), [bank, model]),
    r = useMemo(() => digitRead(raw, bank, model, prepared), [raw, bank, model, prepared]),
    clean = original.pixels.map(x => x / 16),
    cue = raw.map(x => x / 16),
    sum = r.top.reduce((s, x) => s + x.weight, 0);
  return <><p className="hm-result" aria-live="polite">{label}: {r.winners.length === 10 ? 'All ten classes tie at0.1. No unique predicted class.' : r.winners.length === 1 ? `Predicted class ${r.winners[0]}; class ${original.label} mass ${f(r.classes[original.label], 8)}.` : `Tied classes ${r.winners.join(', ')}.`} Original source row {original.sourceId}, label {original.label}, is used only as an evaluation reference.</p><div className="hm-images"><PixelImage title="Original reference" pixels={original.pixels} raw caption={`source row${original.sourceId}, label${original.label}`} /><PixelImage title={label} pixels={raw} raw caption={`input MSE ${f(digitMse(cue, clean), 8)}`} /><PixelImage title="Weighted pixel read" pixels={r.readPixels} caption={`read MSE ${f(digitMse(r.readPixels, clean), 8)}`} />{r.top.map(({
        index,
        weight
      }) => <PixelImage key={index} title={`Memory row${bank.memory[index].sourceId}`} pixels={bank.memory[index].pixels} raw caption={`label${bank.memory[index].label}; weight ${f(weight, 8)}`} />)}</div><p>The three displayed memories carry {f(sum, 8)} of the total weight. All200 memories contribute to both outputs. Pixel MSE uses normalized intensities (0…1) relative to this original image; after arbitrary edits it remains a comparison, not a new ground-truth annotation.</p><Bars title={`${label}: ten computed class masses`} labels={r.classes.map((_, i) => `class${i}`)} values={r.classes} fixedMaximum={1} /><section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Exact pixel values and query representation</h4><Values caption="Normalized query and all64 retrieved pixels" rows={[["Unit query", vec(r.query)], ["Retrieved pixels", vec(r.readPixels)]]} /></section>{showWeights && <section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">All200 memory contributions</h4><NeuralTable caption="Current full memory read; original label is absent from the read function" headers={['Memory index', 'Source row', 'Class', 'Dot score', 'Weight']} rows={bank.memory.map((row, i) => [i, row.sourceId, row.label, f(r.scores[i], 8), f(r.weights[i], 10)])} /></section>}</>;
}
export function DigitMemoryLab() {
  const [ref, near] = useNear(),
    [modelId, setModelId] = useState('seed17'),
    [retry, setRetry] = useState(0),
    [rowIndex, setRowIndex] = useState(31),
    [raw, setRaw] = useState(null),
    [selected, setSelected] = useState(28),
    [pinned, setPinned] = useState(null),
    [fixedBeta, setFixedBeta] = useState(64),
    bankState = useAsset('digit-bank.json', near, retry),
    modelState = useAsset('model-' + modelId + '.json', near, retry),
    bank = bankState.data,
    model = useMemo(() => modelState.data ? {
      ...modelState.data,
      beta: modelId === 'fixed' ? fixedBeta : modelState.data.beta
    } : null, [modelState.data, modelId, fixedBeta]);
  const original = bank?.validation[rowIndex],
    pixels = raw || original?.pixels,
    changePixel = (i, n) => setRaw(pixels.map((v, j) => i === j ? n : v)),
    reset = () => {
      setModelId('seed17');
      setRowIndex(31);
      setRaw(null);
      setSelected(28);
      setPinned(null);
      setFixedBeta(64);
    };
  const keyboard = (event, i) => {
    const delta = {
      ArrowLeft: -1,
      ArrowRight: 1,
      ArrowUp: -8,
      ArrowDown: 8
    }[event.key];
    if (delta !== undefined) {
      event.preventDefault();
      const next = Math.max(0, Math.min(63, i + delta));
      setSelected(next);
      event.currentTarget.parentElement.querySelector(`[data-pixel="${next}"]`)?.focus();
    }
  };
  return <div ref={ref}><NeuralLab id="hopfield-digits" title="Edit real handwriting and inspect every returned association"><p>The bank contains200 original UCI Optdigits images; editable cues are the300 validation images. These are frozen selected models. Changing a model or retrying a download preserves your current pixels, original reference, selected cell and pinned comparison.</p><div className="hm-two"><label className="hm-model-select">Frozen association geometry<select value={modelId} onChange={e => setModelId(e.target.value)}><option value="seed17">Learned seed17, epoch100, β16</option><option value="seed41">Learned seed41, epoch25, β16</option><option value="fixed">Fixed pixel cosine geometry</option></select></label><NeuralNumber label="Validation index (0…299)" value={rowIndex} min={0} max={299} integer onChange={n => {
          setRowIndex(n);
          setRaw(null);
          setSelected(28);
        }} /></div>{modelId === 'fixed' && <NeuralNumber label="Fixed-geometry β (validation selected64)" value={fixedBeta} min={.1} max={256} onChange={setFixedBeta} />}<div className="hm-controls"><button onClick={reset}>Reset digit investigation</button></div>
 {!near && <p>Real images and the selected model load when this investigation approaches the screen.</p>}{near && (bankState.error || modelState.error) ? <div role="alert"><p>The selected memory data could not be loaded. Your edits are retained. Retry, or choose another model.</p><button onClick={() => setRetry(x => x + 1)}>Retry memory data</button></div> : near && (!bank || !model) ? <p role="status">Loading the selected frozen memory data. Current edits are retained.</p> : null}
 {bank && pixels && <><p>Selected cell: row {Math.floor(selected / 8) + 1}, column {selected % 8 + 1}. Each cell is an integer intensity0…16. Use arrow keys within the grid, then edit the selected value below.</p><div className="hm-two"><div className="hm-pixels" role="group" aria-label="Editable8 by8 handwriting pixels">{pixels.map((v, i) => <button type="button" data-pixel={i} key={i} tabIndex={i === selected ? 0 : -1} aria-pressed={i === selected} aria-label={`Row ${Math.floor(i / 8) + 1}, column ${i % 8 + 1}, intensity ${v}`} onClick={() => setSelected(i)} onKeyDown={e => keyboard(e, i)} style={{
              background: `rgb(${v / 16 * 255},${v / 16 * 255},${v / 16 * 255})`,
              color: v >= 8 ? 'black' : 'white'
            }}>{v}</button>)}</div><div><NeuralNumber label={`Pixel row ${Math.floor(selected / 8) + 1}, column ${selected % 8 + 1}`} value={pixels[selected]} min={0} max={16} integer onChange={n => changePixel(selected, n)} /><div className="hm-controls"><button onClick={() => changePixel(selected, 0)}>Set selected pixel0</button><button onClick={() => changePixel(selected, 16)}>Set selected pixel16</button></div></div></div><div className="hm-controls"><button onClick={() => setRaw([...original.pixels])}>Restore original cue</button><button onClick={() => setRaw(occludeDigit(original.pixels))}>Zero original columns4 and5</button><button onClick={() => setRaw(Array(64).fill(0))}>Blank cue null</button><button disabled={!model} onClick={() => setPinned({
            raw: [...pixels],
            original: structuredClone(original),
            model: structuredClone(model)
          })}>Pin current input and model</button></div></>}
 {bank && model && pixels && <DigitReadView bank={bank} model={model} raw={pixels} original={original} />}{bank && pinned && <details open><summary>Pinned comparison: {pinned.model.id}, β={pinned.model.beta}, original source row{pinned.original.sourceId}</summary><p>This complete pinned input and model stay fixed while the current model or pixels change.</p><DigitReadView bank={bank} model={pinned.model} raw={pinned.raw} original={pinned.original} label="Pinned cue" showWeights={false} /></details>}<p><a href={base + 'data-provenance.md'}>Dataset attribution and split provenance</a>. Only the500 memory/validation images needed here are loaded. The original source files and complete training program are available below; the browser performs no fitting.</p></NeuralLab></div>;
}
export function DigitStoriesFigure() {
  const [ref, near] = useNear(),
    [retry, setRetry] = useState(0),
    bankState = useAsset('digit-bank.json', near, retry),
    modelState = useAsset('model-seed17.json', near, retry),
    bank = bankState.data,
    model = modelState.data;
  return <div ref={ref}>{bank && model ? [2946, 3052].map(id => {
      const original = bank.validation.find(row => row.sourceId === id);
      return <div key={id}><h4>Actual validation source row{id}: clean versus occluded</h4><DigitReadView bank={bank} model={model} raw={original.pixels} original={original} label="Clean cue" showWeights={false} /><DigitReadView bank={bank} model={model} raw={occludeDigit(original.pixels)} original={original} label="Occluded cue" showWeights={false} /></div>;
    }) : <p>{bankState.error || modelState.error ? <><span>The measured image stories could not load. </span><button onClick={() => setRetry(x => x + 1)}>Retry image stories</button></> : near ? 'Loading the actual images and selected seed17 projection…' : 'The actual image stories load as they approach the screen.'}</p>}</div>;
}
export function HopfieldProgram({
  filename
}) {
  return <>
    <RemoteCodeBlock source={base + filename} language="python" filename={filename} title={"Read the complete explained program: " + (filename)} />
    <p><a href={base + filename} download>Download {filename}</a>. Save required source data beside the program as described above.</p>
  </>;
}
