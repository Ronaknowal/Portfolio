import { useEffect, useState } from 'react';
import { NeuralLab, NeuralNumber, NeuralSelect, NeuralTable, formatNeural as f } from './NeuralLessonElements.jsx';
import { initialImage, initialKernel, correlate2d, sharedFilterUpdate, windowGeometry, poolVector, cnnAxisLayers, receptiveTrace, observedAncestors, dilationOffsets, averagingProfile, transpose1d, shiftComparison } from '../../data/convolution-models.js';
import './convolution-labs.css';

const assetRoot = '/learn-code/convolution-pooling-receptive-fields/';
const clone = rows => rows.map(row => [...row]);
const rowText = values => `[${values.map(value => f(value, 4)).join(', ')}]`;

function EditableCell({ label, value, min, max, active, onChange }) {
  const [draft, setDraft] = useState(null);
  return <span className="convolution-editable-cell"><input type="number" min={min} max={max} step="0.5" value={draft ?? value} className={active ? 'is-selected' : ''} aria-label={label} aria-invalid={draft !== null} onBlur={() => setDraft(null)} onChange={event => {
    const text = event.target.value, next = Number(text);
    if (text.trim() && Number.isFinite(next) && next >= min && next <= max) { setDraft(null); onChange(next); }
    else setDraft(text);
  }} />{draft !== null && <small>Use {min}…{max}; current result retains {value}.</small>}</span>;
}

function Grid({ title, values, onChange, selected = [], onSelect, min = -5, max = 5, signed = false }) {
  const flat = values.flat(), scale = Math.max(...flat.map(Math.abs), 1e-12);
  return <figure className="convolution-grid-figure"><figcaption>{title}</figcaption>
    <div className="convolution-grid-scroll" tabIndex={values[0].length > 4 ? 0 : undefined} role={values[0].length > 4 ? 'region' : undefined} aria-label={values[0].length > 4 ? `${title}; scroll to inspect all columns` : undefined}><div className="convolution-grid" style={{ '--columns': values[0].length, minWidth: values[0].length > 4 ? values[0].length * 42 : undefined }}>
      {values.flatMap((row, r) => row.map((value, c) => {
        const active = selected.some(([sr, sc]) => sr === r && sc === c);
        const style = signed ? { backgroundColor: value === 0 ? '#202020' : value < 0 ? `rgba(143,168,200,${.1 + Math.abs(value) / scale * .6})` : `rgba(228,183,82,${.06 + Math.abs(value) / scale * .6})` } : undefined;
        if (onChange) return <EditableCell key={`${r}-${c}`} label={`${title}, row ${r + 1}, column ${c + 1}`} value={value} min={min} max={max} active={active} onChange={next => onChange(r, c, next)} />;
        if (onSelect) return <button key={`${r}-${c}`} aria-pressed={active} aria-label={`${title}, row ${r + 1}, column ${c + 1}: ${f(value)}`} onClick={() => onSelect(r, c)}>{f(value, 3)}</button>;
        return <span key={`${r}-${c}`} style={style} className={active ? 'is-selected' : ''} title={`Row ${r + 1}, column ${c + 1}: ${value}`}>{value !== 0 && Math.abs(value) < .01 ? value.toExponential(1) : f(value, 2)}</span>;
      }))}
    </div></div>{signed && <details><summary>Exact numeric values</summary><pre tabIndex={0}>{values.map(row => row.join(', ')).join('\n')}</pre></details>}</figure>;
}

function Rail({ values, title, selected = [], signed = false }) {
  return <figure className="convolution-rail"><figcaption>{title}</figcaption><ol>{values.map((value, index) => <li key={index} className={selected.includes(index) ? 'is-selected' : ''}><span>{index}</span><strong className={signed && value < 0 ? 'negative-value' : ''}>{f(value, 4)}</strong></li>)}</ol></figure>;
}

export function ConvolutionPatchLab() {
  const [image, setImage] = useState(clone(initialImage));
  const [kernel, setKernel] = useState(clone(initialKernel));
  const [position, setPosition] = useState([0, 1]);
  const [r, c] = position, output = correlate2d(image, kernel);
  const cells = [[r, c], [r, c + 1], [r + 1, c], [r + 1, c + 1]];
  const edit = setter => (row, column, value) => setter(previous => previous.map((items, index) => index === row ? items.map((item, j) => j === column ? value : item) : items));
  const products = kernel.flatMap((row, kr) => row.map((weight, kc) => ({ value: image[r + kr][c + kc], weight, product: image[r + kr][c + kc] * weight })));
  return <NeuralLab id="convolution-patch" title="One filter. Four places to use it.">
    <p>Edit the image or filter. Select an output cell to trace its four products. Outlines identify the participating image patch.</p>
    <div className="convolution-triptych"><Grid title="Image" values={image} onChange={edit(setImage)} selected={cells} /><Grid title="Shared filter" values={kernel} onChange={edit(setKernel)} min={-3} max={3} /><Grid title="Output" values={output} onSelect={(a, b) => setPosition([a, b])} selected={[position]} /></div>
    <div className="convolution-products">{products.map((term, i) => <span key={i}>{term.value} × ({term.weight})<strong>{f(term.product)}</strong></span>)}</div>
    <p className="convolution-result">Output ({r}, {c}): {products.map(term => `(${f(term.product)})`).join(' + ')} = <strong>{f(output[r][c])}</strong></p>
    <div className="neural-buttons"><button onClick={() => setKernel([[0, 0], [0, 0]])}>Zero filter</button><button onClick={() => setKernel([[1, 0], [0, -1]])}>Diagonal difference</button><button onClick={() => { setImage(clone(initialImage)); setKernel(clone(initialKernel)); setPosition([0, 1]); }}>Reset patch</button></div>
    <p>A pixel may belong to several windows. A zero coefficient makes that particular appearance contribute nothing, even though the connection exists.</p>
  </NeuralLab>;
}

export function ConvolutionChannelsLab() {
  const [inputs, setInputs] = useState([2, 3]), [weights, setWeights] = useState([[1, 1], [1, 1]]), [bias, setBias] = useState([0, 0]);
  const edit = (setter, index, value) => setter(previous => previous.map((item, i) => i === index ? value : item));
  const output = x => weights.map((row, index) => row.reduce((sum, weight, c) => sum + weight * x[c], bias[index]));
  return <NeuralLab id="convolution-channels" title="A 1×1 filter mixes measurements, not neighboring pixels">
    <div className="neural-controls">{inputs.map((value, i) => <NeuralNumber key={i} label={`Input channel ${i + 1}`} value={value} onChange={v => edit(setInputs, i, v)} min={-4} max={4} step={.5} />)}</div>
    <div className="neural-two">{weights.map((row, r) => <div key={r}><h4>Output channel {r + 1}</h4>{row.map((value, c) => <NeuralNumber key={c} label={`Weight ${r + 1} from input ${c + 1}`} value={value} min={-2} max={2} step={.5} onChange={v => setWeights(previous => previous.map((items, i) => i === r ? items.map((w, j) => j === c ? v : w) : items))} />)}<NeuralNumber label={`Bias ${r + 1}`} value={bias[r]} min={-2} max={2} step={.5} onChange={v => edit(setBias, r, v)} /><p className="convolution-result">{row.map((w, c) => `${w} × ${inputs[c]}`).join(' + ')} + {bias[r]} → <strong>{f(output(inputs)[r])}</strong></p></div>)}</div>
    <NeuralTable caption="Try a sum in the first row and a difference in the second. The zero case exposes biases." headers={['Input', 'Current output', 'Sum / difference']} rows={[[2, 3], [-1, 4], [0, 0]].map(x => [rowText(x), rowText(output(x)), rowText([x[0] + x[1], x[0] - x[1]])])} />
    <figure className="convolution-flow"><figcaption>Groups restrict the channel connections</figcaption><p>Inputs 0, 1 → output channels 0, 1, 2</p><p>Inputs 2, 3 → output channels 3, 4, 5</p><p>Four inputs and six outputs, groups = 2. There is no cross-group connection in this layer.</p></figure>
    <button onClick={() => { setInputs([2, 3]); setWeights([[1, 1], [1, 1]]); setBias([0, 0]); }}>Reset channels</button>
  </NeuralLab>;
}

export function ConvolutionUpdateLab() {
  const [targets, setTargets] = useState([0, 0]), [rate, setRate] = useState(.1);
  const result = sharedFilterUpdate(targets, rate);
  return <NeuralLab id="convolution-update" title="Two windows negotiate one shared update">
    <p>Fixed input [1, 3, 2], weights [1, −1]. The objective is half the sum of squared errors. Every edit recalculates one proposed simultaneous step.</p>
    <div className="neural-controls">{targets.map((value, i) => <NeuralNumber key={i} label={`Target at output ${i}`} value={value} onChange={v => setTargets(previous => previous.map((x, j) => j === i ? v : x))} min={-3} max={3} step={.1} />)}<NeuralNumber label="Step size" value={rate} onChange={setRate} min={0} max={.2} step={.01} /></div>
    <NeuralTable caption="Location contributions meet at shared parameters" headers={['Window', 'Output − target', 'To weight 0', 'To weight 1']} rows={result.contributions.map((row, i) => [`${i}: ${rowText(i === 0 ? [1, 3] : [3, 2])}`, f(result.errors[i]), ...row.map(v => f(v))]).concat([['Sum', '∂L/∂w', ...result.gradient.map(v => f(v))]])} />
    <Rail title="Overlapping contributions also add at the inputs: ∂L/∂x" values={result.inputGradient} signed />
    <div className="convolution-flow"><p>[1, −1] − {f(rate)} × {rowText(result.gradient)} → <strong>{rowText(result.nextWeights)}</strong></p><p>New outputs: {rowText(result.nextOutput)}</p><p className="convolution-result">Loss {f(result.loss)} → <strong>{f(result.nextLoss)}</strong>. {rate === 0 ? 'Zero rate preserves the weights.' : result.nextLoss > result.loss ? 'This step overshoots; try a smaller rate.' : result.nextLoss === result.loss ? 'The objective is unchanged.' : 'The combined objective decreases; individual errors need not all decrease.'}</p></div>
    <div className="neural-buttons"><button onClick={() => setTargets([-2, 1])}>Match current outputs</button><button onClick={() => { setTargets([0, 0]); setRate(.1); }}>Reset update</button></div>
  </NeuralLab>;
}

export function ConvolutionGeometryLab() {
  const defaults = { n: 8, k: 4, s: 1, d: 1, left: 1, right: 1 };
  const [settings, setSettings] = useState(defaults), [selection, setSelection] = useState(0);
  const geometry = windowGeometry(settings), current = Math.min(selection, Math.max(0, geometry.count - 1));
  const sampled = geometry.windows[current] || [];
  const cells = Array.from({ length: settings.n + settings.left + settings.right }, (_, i) => i - settings.left);
  return <NeuralLab id="convolution-geometry" title="Count starts, then inspect alignment">
    <div className="neural-controls">{[['n', 'Input width', 3, 16], ['k', 'Kernel taps', 1, 5], ['s', 'Stride', 1, 3], ['d', 'Dilation', 1, 3], ['left', 'Left padding', 0, 4], ['right', 'Right padding', 0, 4]].map(([key, label, min, max]) => <NeuralNumber key={key} label={label} value={settings[key]} onChange={value => setSettings(previous => ({ ...previous, [key]: value }))} min={min} max={max} step={1} integer />)}</div>
    {geometry.count > 0 ? <><NeuralNumber label="Output window index" value={current} onChange={setSelection} min={0} max={geometry.count - 1} step={1} integer /><figure className="convolution-coordinate-ruler"><figcaption>Input indices; P means supplied padding. Outlines mark sampled taps.</figcaption><div>{cells.map(index => <span key={index} className={`${sampled.includes(index) ? 'is-selected' : ''} ${index < 0 || index >= settings.n ? 'is-padding' : ''}`}><small>{index}</small>{index < 0 || index >= settings.n ? 'P' : '●'}</span>)}</div></figure><p className="convolution-result"><strong>{geometry.count} outputs</strong>; kernel spans {geometry.span} positions; first center {geometry.center}; unused tail {geometry.remainder}. Current taps: {rowText(sampled)}.</p></> : <p role="status">The filter span does not fit. Increase input/padding or reduce kernel/dilation. No output map exists for these sizes.</p>}
    <p>With width 8 and kernel 4, compare padding (1, 2) with (2, 1): both produce eight values, but their center grids differ. Matching a skip branch requires alignment as well as shape.</p>
    <button onClick={() => { setSettings(defaults); setSelection(0); }}>Reset geometry</button>
  </NeuralLab>;
}

export function ConvolutionPoolingLab() {
  const [values, setValues] = useState([1, 4, 3]);
  const [patches, setPatches] = useState([[[4, 2], [1, 1]], [[4, 0], [2, 2]]]);
  const max = poolVector(values), mean = poolVector(values, 2, 1, 'mean');
  return <NeuralLab id="convolution-pooling" title="Watch winners and shared gradients">
    <div className="neural-controls">{values.map((value, i) => <NeuralNumber key={i} label={`Pooling input ${i}`} value={value} min={-4} max={5} step={.5} onChange={v => setValues(previous => previous.map((x, j) => j === i ? v : x))} />)}</div>
    <div className="neural-two"><div><Rail title="Max outputs: strongest in each overlapping pair" values={max.outputs} /><Rail title="Max gradient: one route per window, sums at overlaps" values={max.gradient} /></div><div><Rail title="Average outputs: both values contribute" values={mean.outputs} /><Rail title="Average gradient: each window shares its gradient" values={mean.gradient} /></div></div>
    <p>Ties here select the first maximizing position. This is an explicit implementation choice for this exact small model.</p>
    <div className="neural-two">{patches.map((patch, p) => <div key={p}><Grid title={`Ambiguous patch ${p + 1}`} values={patch} min={0} max={4} onChange={(r, c, v) => setPatches(previous => previous.map((items, index) => index === p ? items.map((row, i) => i === r ? row.map((value, j) => j === c ? v : value) : row) : items))} /><p>Max {Math.max(...patch.flat())}; mean {f(patch.flat().reduce((a, b) => a + b, 0) / 4)}</p></div>)}</div>
    <p>Both initial patches have max 4 and mean 2, despite different pixels. Neither summary can recover which arrangement was present.</p>
    <NeuralTable caption="Adaptive average bins for [1, 2, 3, 4, 5] → three outputs" headers={['Indices', 'Values', 'Mean']} rows={[["[0, 2)", '[1, 2]', '1.5'], ["[1, 4)", '[2, 3, 4]', '3'], ["[3, 5)", '[4, 5]', '4.5']]} />
    <p>Index 1 belongs to two bins; so does index 3. For a sum of outputs the input gradient is [1/2, 5/6, 1/3, 5/6, 1/2].</p>
    <button onClick={() => { setValues([1, 4, 3]); setPatches([[[4, 2], [1, 1]], [[4, 0], [2, 2]]]); }}>Reset pooling</button>
  </NeuralLab>;
}

export function ConvolutionReceptiveLab() {
  const [stage, setStage] = useState(3), [index, setIndex] = useState(3), [first, setFirst] = useState(2), [second, setSecond] = useState(2);
  const trace = receptiveTrace(cnnAxisLayers), row = trace[stage], position = Math.min(index, row.n - 1);
  const ancestors = observedAncestors(cnnAxisLayers.slice(0, stage + 1), position);
  const center = row.a + position * row.j, offsets = dilationOffsets(first, second), bound = 2 * (first + second) + 1;
  return <NeuralLab id="convolution-receptive" title="A bounding interval is not the set of observed pixels">
    <NeuralSelect label="Trace through layer" value={stage} onChange={v => setStage(Number(v))} options={trace.map((r, i) => [i, `${i + 1}. ${r.name}`])} />
    <NeuralNumber label="Select output position" value={position} onChange={setIndex} min={0} max={row.n - 1} step={1} integer />
    <NeuralTable caption="The fixed 32-wide architecture; pooling expands reach too" headers={['Layer', 'Width', 'r / j / a']} rows={trace.map((r, i) => [`${i + 1}. ${r.name}`, r.n, `${r.r} / ${r.j} / ${r.a}`])} />
    <figure className="convolution-coordinate-ruler"><figcaption>Observed input indices reaching this output (outlined)</figcaption><div>{Array.from({ length: 32 }, (_, i) => <span key={i} className={ancestors.includes(i) ? 'is-selected' : ''}>{i}</span>)}</div></figure>
    <p className="convolution-result">Center {center}; bounding pixel-center interval [{center - (row.r - 1) / 2}, {center + (row.r - 1) / 2}]; <strong>{ancestors.length} observed pixels</strong>. Padding is constant and adds no observed image information.</p>
    <div className="neural-controls"><NeuralNumber label="First dilation" value={first} onChange={setFirst} min={1} max={2} step={1} integer /><NeuralNumber label="Second dilation" value={second} onChange={setSecond} min={1} max={2} step={1} integer /></div>
    <figure className="convolution-coordinate-ruler"><figcaption>Two three-tap, unit-stride layers: offsets around an interior output</figcaption><div>{Array.from({ length: bound }, (_, i) => i - (bound - 1) / 2).map(i => <span key={i} className={offsets.includes(i) ? 'is-selected' : ''}>{i}</span>)}</div></figure>
    <p>{offsets.length} connected offsets in a bound of width {bound}. {offsets.length < bound ? 'The unoutlined positions are holes.' : 'Every position inside this bound is connected.'}</p>
    <button onClick={() => { setStage(3); setIndex(3); setFirst(2); setSecond(2); }}>Reset receptive fields</button>
  </NeuralLab>;
}

export function ConvolutionInfluenceLab() {
  const [depth, setDepth] = useState(20), [threshold, setThreshold] = useState(.01);
  const result = averagingProfile(depth, threshold);
  return <NeuralLab id="convolution-influence" title="Many paths reach the center; fewer reach the edges">
    <p>Exact linear model: repeatedly apply [1, 1, 1]/3. There are no learned weights or nonlinear gates in this calculation.</p>
    <div className="neural-controls"><NeuralNumber label="Averaging depth" value={depth} onChange={setDepth} min={1} max={20} step={1} integer /><NeuralNumber label="Fraction of peak retained" value={threshold} onChange={setThreshold} min={.001} max={.2} step={.001} /></div>
    <figure className="convolution-profile"><figcaption>Input coefficient by offset (amber meets threshold)</figcaption><div>{result.coefficients.map((value, i) => <span key={i} title={`Offset ${i - depth}: ${value}`}><i style={{ height: `${value / result.peak * 100}%`, background: result.retained.includes(i - depth) ? '#e4b752' : '#777' }} /></span>)}</div><p>Offsets −{depth} … 0 … +{depth}; height scale 0 to {f(result.peak)}.</p></figure>
    <p className="convolution-result">Possible support width <strong>{2 * depth + 1}</strong>; width at or above {f(threshold * 100)}% of peak <strong>{result.width}</strong>. Coefficients sum to {f(result.coefficients.reduce((a, b) => a + b, 0))}.</p>
    <details><summary>Read every exact coefficient</summary><Rail title="Coefficient index; offset = index − depth" values={result.coefficients} /></details>
    <p>The recorded digit viewer below the experiment also exposes signed input gradients. Those depend on a trained model and particular image; this path-count example has a different meaning.</p>
    <button onClick={() => { setDepth(20); setThreshold(.01); }}>Reset influence</button>
  </NeuralLab>;
}

export function ConvolutionShiftLab() {
  const [boundary, setBoundary] = useState('circular'), [alpha, setAlpha] = useState(1);
  const values = shiftComparison(boundary), phase0 = 2 * alpha - 1, phase1 = -phase0;
  return <NeuralLab id="convolution-shift" title="A boundary and a sampling phase change the answer">
    <NeuralSelect label="Boundary rule" value={boundary} onChange={setBoundary} options={[["circular", 'Circular wrap'], ["zero", 'Zero outside']]} />
    <Rail title="Original impulse" values={values.source} /><Rail title="Shift input right, then filter [1, 0, −1]" values={values.first} signed /><Rail title="Filter first, then shift the output" values={values.second} signed />
    <p className="convolution-result">Maximum difference: <strong>{values.difference}</strong>. {values.difference === 0 ? 'These two operations commute for this circular fixture.' : 'The boundary changes the comparison even at stride one.'}</p>
    <NeuralNumber label="First averaging weight α (second is 1 − α)" value={alpha} onChange={setAlpha} min={0} max={1} step={.05} />
    <p>On the alternating signal [1, −1, 1, −1, 1, −1], filter with [α, 1 − α], then sample every second position.</p>
    <Rail title="Phase 0 outputs" values={[phase0, phase0, phase0]} signed /><Rail title="Phase 1 outputs" values={[phase1, phase1, phase1]} signed />
    <p>The equal-weight filter suppresses this particular alternating signal in both phases. It does not establish invariance for every signal or trained classifier.</p>
    <button onClick={() => { setBoundary('circular'); setAlpha(1); }}>Reset shifts</button>
  </NeuralLab>;
}

export function ConvolutionTransposeLab() {
  const [dual, setDual] = useState([2, -3]), [size, setSize] = useState(3);
  const scatter = transpose1d(dual, [1, -1]), coverage = transpose1d([1, 1, 1], Array(size).fill(1), 2).output;
  const left = -2 * dual[0] + dual[1], right = scatter.output.reduce((sum, value, i) => sum + value * [1, 3, 2][i], 0);
  return <NeuralLab id="convolution-transpose" title="Scatter each footprint and add the overlaps">
    <div className="neural-controls">{dual.map((value, i) => <NeuralNumber key={i} label={`Output-side value ${i}`} value={value} onChange={v => setDual(previous => previous.map((x, j) => j === i ? v : x))} min={-4} max={4} step={.5} />)}</div>
    {scatter.contributions.map((values, i) => <Rail key={i} title={`Footprint from value ${i}`} values={values} signed />)}<Rail title="Sum: C transpose times g" values={scatter.output} signed />
    <p className="convolution-result">〈Cx, g〉 = {f(left)}; 〈x, Cᵀg〉 = {f(right)} for x = [1, 3, 2]. The transpose preserves this dot-product identity, not the original input.</p>
    <NeuralNumber label="All-ones footprint size (stride 2)" value={size} onChange={setSize} min={2} max={4} step={1} integer />
    <Rail title="Three input ones: one-dimensional overlap counts" values={coverage} />
    <Grid title="Two-dimensional separable all-ones overlap counts" values={coverage.map(a => coverage.map(b => a * b))} signed />
    <p>Interior counts alternate for kernel 3. Kernel 4 equalizes the interior in this all-ones example; boundary coverage still differs and learned weights can still create artifacts.</p>
    <button onClick={() => { setDual([2, -3]); setSize(3); }}>Reset transpose</button>
  </NeuralLab>;
}

export function ConvolutionPatchMatrixFigure() {
  return <figure className="convolution-flow"><figcaption>Each matrix column is one flattened patch, in row-major order</figcaption><div className="neural-two"><Grid title="Unfolded columns" values={[[1, 2, 0, 1], [2, 0, 1, 3], [0, 1, 2, 1], [1, 3, 1, 0]]} /><Grid title="Fold overlap counts" values={[[1, 2, 1], [2, 4, 2], [1, 2, 1]]} /></div><p>[1, −1, 0, 1] × columns → [0, 5, 0, −2]. Folding the columns sums repeated pixels; it does not average them.</p><Grid title="Folded values: count × original image" values={[[1, 4, 0], [0, 4, 6], [2, 2, 0]]} /></figure>;
}

function RemoteContent({ file, json = false, children }) {
  const [data, setData] = useState(null), [error, setError] = useState(false), [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setError(false);
    fetch(assetRoot + file, { signal: controller.signal }).then(response => { if (!response.ok) throw new Error('Load failed'); return json ? response.json() : response.text(); }).then(setData).catch(error => { if (error.name !== 'AbortError') setError(true); });
    return () => controller.abort();
  }, [file, json, attempt]);
  if (error) return <p role="alert">This file could not load. <button onClick={() => setAttempt(value => value + 1)}>Retry</button> or use the download link.</p>;
  return data ? children(data) : <p role="status">Loading the selected lesson data…</p>;
}

export function ConvolutionProgram({ file, title }) {
  const [open, setOpen] = useState(false);
  return <details className="convolution-program" onToggle={event => setOpen(event.currentTarget.open)}><summary>{title}</summary><p><a href={assetRoot + file} download>Download {file}</a></p>{open && <RemoteContent file={file}>{source => <pre tabIndex={0} role="region" aria-label={title}><code>{source}</code></pre>}</RemoteContent>}</details>;
}

function MeasuredExplorer({ data }) {
  const [name, setName] = useState('cnn_max'), [seed, setSeed] = useState(1), [step, setStep] = useState(400), [specimen, setSpecimen] = useState(0), [layer, setLayer] = useState('first_maps'), [channel, setChannel] = useState(0);
  const runs = data.experiment.fits, run = runs.find(run => run.name === name && run.seed === seed), baseline = runs.find(run => run.name === 'mlp' && run.seed === seed);
  const selected = run.trace.find(row => row.step === step), reference = baseline.trace.find(row => row.step === step);
  const maps = runs.find(run => run.name === (name === 'mlp' ? 'cnn_max' : name) && run.seed === 1);
  const row = maps.visual_rows[specimen], map = row[layer][Math.min(channel, row[layer].length - 1)];
  return <div className="convolution-measured"><div className="neural-controls"><NeuralSelect label="Measured model" value={name} onChange={setName} options={[["mlp", 'Dense baseline'], ["cnn_max", 'CNN · max pool'], ["cnn_average", 'CNN · average pool'], ["cnn_global_average", 'CNN · global-average head']]} /><NeuralSelect label="Recorded seed" value={seed} onChange={value => setSeed(Number(value))} options={[1, 2, 3].map(v => [v, v])} /><NeuralSelect label="Recorded update" value={step} onChange={value => setStep(Number(value))} options={[0, 1, 25, 100, 200, 400].map(v => [v, v])} /></div>
    <NeuralTable caption={`Recorded update ${step}, seed ${seed}; development comparisons, not fresh browser training`} headers={['Model', 'Train CE', 'Dev CE', 'Dev correct / 120']} rows={[[name, f(selected.train.cross_entropy), f(selected.validation.cross_entropy), selected.validation.correct], ['Dense reference', f(reference.train.cross_entropy), f(reference.validation.cross_entropy), reference.validation.correct]]} />
    <ConvolutionTrainingTrace run={run} baseline={baseline} selectedStep={step} />
    <NeuralTable caption="Measured final checkpoint under zero-fill shifts; a stroke can be clipped" headers={['Input condition', 'CE', 'Correct / 120']} rows={[["Unshifted", run.trace.at(-1).validation], ['Right shift', run.shifted_right], ['Down shift', run.shifted_down]].map(([label, metric]) => [label, f(metric.cross_entropy), metric.correct])} />
    <h4>Look inside a saved model</h4><p>Maps are only available for the three retained seed-1 CNNs at update 400. The panel below uses <strong>{maps.name}, seed 1, update 400</strong>, independently of the evidence selector above.</p>
    <div className="neural-controls"><NeuralSelect label="Saved specimen source ID" value={specimen} onChange={v => setSpecimen(Number(v))} options={maps.visual_rows.map((row, i) => [i, `${row.source_id} · digit ${row.actual}`])} /><NeuralSelect label="Saved representation" value={layer} onChange={value => { setLayer(value); setChannel(0); }} options={[["first_maps", 'First post-ReLU · 8×8'], ["second_maps", 'Second post-ReLU · 4×4'], ["final_maps", 'Final pooled · 2×2']]} /><NeuralNumber label="Feature channel" value={Math.min(channel, row[layer].length - 1)} onChange={setChannel} min={0} max={row[layer].length - 1} step={1} integer /></div>
    <div className="neural-two"><Grid title={`Input · true digit ${row.actual}`} values={row.input} signed /><Grid title={`Saved ${layer}, channel ${Math.min(channel, row[layer].length - 1)}`} values={map} signed /></div>
    <Rail title={`Saved class probabilities; predicted digit ${row.predicted}`} values={row.probabilities} />
    <details><summary>Inspect weights and signed local input sensitivity</summary><p>The filter below is a recorded weight array. The saved maps above are post-ReLU; their missing biases/raw preactivations are not reconstructed here. Signed gradient colors: amber positive, blue negative, neutral zero. Each panel rescales to its own largest absolute value; numbers retain the exact relative scale.</p><Grid title="First-layer filter 0" values={maps.first_layer_kernels[0]} signed /><Grid title={`Derivative of logit ${row.predicted} at this exact image`} values={row.logit_input_gradient} signed /><p>{row.logit_input_gradient.flat().every(value => value === 0) ? 'All input derivatives are zero at this input.' : 'This is local logit sensitivity, not a causal explanation of the image.'}</p></details>
    <button onClick={() => { setName('cnn_max'); setSeed(1); setStep(400); setSpecimen(0); setLayer('first_maps'); setChannel(0); }}>Reset recorded view</button>
  </div>;
}

export function ConvolutionMeasuredLab() {
  const [open, setOpen] = useState(false);
  return <NeuralLab id="convolution-measured" title="Follow real digit pixels through a recorded model">
    <p><strong>Reference architecture: the CNN with a flattening head.</strong> The dense model has 64→32→10 layers; the global-average variant averages the final 2×2 maps and uses a 16→10 head.</p>
    <ol className="convolution-network"><li>Input · 1×8×8</li><li>Conv + ReLU · 8×8×8</li><li>Pool · 8×4×4</li><li>Conv + ReLU · 16×4×4</li><li>Pool · 16×2×2</li><li>Flatten 64 → 10 logits</li></ol>
    <p>The complete experiment and numeric records are downloadable. Opening the viewer loads this lesson’s saved measurements; it does not fit a network.</p>
    <button aria-expanded={open} onClick={() => setOpen(value => !value)}>{open ? 'Close recorded viewer' : 'Open recorded viewer'}</button>
    {open && <RemoteContent file="calculated-inputs.json" json>{data => <MeasuredExplorer data={data} />}</RemoteContent>}
  </NeuralLab>;
}

function ConvolutionTrainingTrace({ run, baseline, selectedStep }) {
  const [partition, setPartition] = useState('validation');
  const top = Math.max(...run.trace.map(row => row[partition].cross_entropy), ...baseline.trace.map(row => row[partition].cross_entropy)) * 1.08;
  const x = step => 44 + step / 400 * 276;
  const y = value => 196 - value / top * 168;
  return <figure className="convolution-training-trace"><figcaption>All six recorded checkpoints, seed {run.seed}</figcaption>
    <NeuralSelect label="Training trace partition" value={partition} onChange={setPartition} options={[["train", 'Training cross-entropy'], ["validation", 'Development cross-entropy']]} />
    <svg viewBox="0 0 340 232" role="img" aria-label={`Recorded ${partition} loss: amber ${run.name}, grey dense baseline. Exact data follows the plot.`}>
      <path d="M44 28 V196 H320" fill="none" stroke="#999" />
      {[0, .5, 1].map(t => <g key={t}><text x="38" y={y(t * top) + 5} textAnchor="end">{f(t * top, 2)}</text><text x={x(t * 400)} y="218" textAnchor="middle">{t * 400}</text></g>)}
      <line x1={x(selectedStep)} x2={x(selectedStep)} y1="28" y2="196" stroke="#888" strokeDasharray="3 4" />
      {[[baseline, '#aaa'], [run, '#e4b752']].map(([record, color], i) => <g key={i}><polyline fill="none" stroke={color} strokeWidth="2" strokeDasharray={i === 0 ? '5 4' : undefined} points={record.trace.map(row => `${x(row.step)},${y(row[partition].cross_entropy)}`).join(' ')} />{record.trace.map(row => <circle key={row.step} cx={x(row.step)} cy={y(row[partition].cross_entropy)} r={row.step === selectedStep ? 4 : 2.5} fill={color} />)}</g>)}
    </svg><p>Horizontal: parameter updates. Vertical: mean cross-entropy. Amber: {run.name}; dashed gray: dense baseline. Segments join measured points; intermediate updates were not recorded.</p>
    <details><summary>Exact values at every recorded update</summary><NeuralTable caption={`${partition} cross-entropy at actual checkpoints`} headers={['Update', run.name, 'Dense baseline']} rows={run.trace.map((row, i) => [row.step, f(row[partition].cross_entropy, 6), f(baseline.trace[i][partition].cross_entropy, 6)])} /></details>
  </figure>;
}
