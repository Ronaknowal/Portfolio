import { useId, useState } from 'react';
import { sensorValues, shapeText, coordinates, selectionCases, selectionResult, memoryCases, memoryState, broadcastCases, broadcastResult, reductionResult } from '../../data/numpy-foundations-model';
import './numpy-foundations-labs.css';

function ArrayGrid({ values, shape, label, selected = [], onSelect, sourceLabels, sensorLabels = false }) {
  const cols = shape.length >= 2 ? shape.at(-1) : Math.max(values.length, 1);
  const locations = coordinates(shape);
  return <figure className="numpy-array">
    <figcaption><strong>{label}</strong><span>{shape.length ? `shape ${shapeText(shape)}` : 'scalar · no axes'}</span></figcaption>
    {sensorLabels && <div className="numpy-sensor-labels"><span>A · °C</span><span>B · °C</span></div>}
    <div className="numpy-array__grid" style={{ '--numpy-columns': cols }}>
      {values.map((value, index) => {
        const coordinate = locations[index]?.join(', ') || 'scalar';
        const labelText = `${label} ${coordinate}: ${value}${selected.includes(index) ? ', highlighted' : ''}`;
        const contents = <><small>{shape.length ? `[${coordinate}]` : 'value'}</small><strong>{value}</strong>{sourceLabels && <small>{sourceLabels[index]}</small>}</>;
        return onSelect ? <button type="button" key={index} aria-label={labelText} aria-pressed={selected.includes(index)} className={selected.includes(index) ? 'is-selected' : ''} onClick={() => onSelect(index)}>{contents}</button>
          : <div key={index} aria-label={labelText} className={selected.includes(index) ? 'is-selected' : ''}>{contents}</div>;
      })}
    </div>
  </figure>;
}

function Lab({ name, eyebrow, title, children }) {
  const uid = useId();
  return <section className={`lesson-lab numpy-lab numpy-${name}-lab`} aria-labelledby={`${uid}-title`}><div className="lesson-eyebrow">{eyebrow}</div><h3 id={`${uid}-title`}>{title}</h3>{children}</section>;
}

export function NumpyDataDiagram() {
  return <figure className="numpy-data-diagram"><figcaption>One row is one time; one column is one sensor.</figcaption>
    <div className="numpy-data-layout"><span className="numpy-axis-label">axis 0<br />time<br />0 ↓ 1 ↓ 2</span><ArrayGrid values={sensorValues} shape={[3, 2]} label="X · temperature in °C" sensorLabels /></div>
    <p>Axis 1 moves across sensors: A → B. Cell <code>X[1, 0]</code> is 24 °C: time 1, sensor A. The visible grid organises coordinates; it is not a physical drawing of memory.</p>
  </figure>;
}

export function NumpySelectionLab() {
  const [choice, setChoice] = useState('column');
  const [revealed, setRevealed] = useState(false);
  const [cell, setCell] = useState(0);
  const result = selectionResult(choice);
  const choose = id => { setChoice(id); setRevealed(false); setCell(0); };
  return <Lab name="selection" eyebrow="COORDINATES · FOLLOW A VALUE" title="Which cells survive this selection?">
    <p>Predict the values and shape first. After revealing, select an output cell to trace its original coordinate. Compare the two column selections: the numbers match, but one keeps an extra axis.</p>
    <div className="lesson-controls"><label>Array selection<select value={choice} onChange={e => choose(e.target.value)}>{selectionCases.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label></div>
    <div className="numpy-code-line"><code>{result.code}</code></div>
    <div className="numpy-linked-arrays"><ArrayGrid values={sensorValues} shape={[3, 2]} label="Original X" sensorLabels selected={revealed ? [result.source[cell]] : []} />
      <span className="numpy-flow-arrow" aria-hidden="true">→</span>
      {revealed ? <ArrayGrid values={result.values} shape={result.shape} label="Selection result" selected={[cell]} onSelect={setCell} /> : <div className="numpy-await"><strong>Predict first</strong><p>How many axes remain? Which original cells belong in the result?</p></div>}
    </div>
    <div className="numpy-feedback" role="status">{revealed ? <><p>{result.why}</p><p><strong>Selected link:</strong> result {shapeText(coordinates(result.shape)[cell])} ← X[{Math.floor(result.source[cell] / 2)}, {result.source[cell] % 2}] = {result.values[cell]} °C.</p></> : <p>Every row has two readings. A slice keeps its axis; an integer index removes that axis.</p>}</div>
    <div className="numpy-actions"><button type="button" onClick={() => setRevealed(!revealed)}>{revealed ? 'Hide selection result' : 'Reveal selection result'}</button><button type="button" onClick={() => choose('column')}>Reset selection lab</button></div>
    <details><summary>Transfer: same values, different question</summary><p>Compare “Whole rows” with “Individual cells”. They happen to contain the same four numbers here. Explain why only one still records time–sensor pairs. Then compare transpose with reshape: both produce (2, 3), but do their cells mean the same thing?</p><details><summary>Explain the difference</summary><p>The row selection keeps a two-column table; the cell mask flattens selected values. Transpose makes each sensor a row. Reshape merely regroups [18, 20, 24, 26, 30, 32], mixing sensor identities. Matching shape or values alone cannot validate the meaning of an array.</p></details></details>
    <p className="lesson-note">This model supports only the displayed fixed numeric array and selections. Output coordinates are local to the new array; highlighted source coordinates identify their origin. Memory sharing is investigated separately below.</p>
  </Lab>;
}

export function NumpyMemoryLab() {
  const [kind, setKind] = useState('view');
  const [index, setIndex] = useState(1);
  const [written, setWritten] = useState(false);
  const state = memoryState(kind, written ? index : null);
  const choose = value => { setKind(value); setWritten(false); };
  return <Lab name="memory" eyebrow="VIEWS & COPIES · FOLLOW THE STORAGE" title="Does writing through picked change X?">
    <p>Start with a fresh X each time. Predict whether writing 99 into the selected sensor reading changes X too. The labels A and B identify separate data buffers, not Python variable names.</p>
    <div className="lesson-controls"><label>Selection storage<select value={kind} onChange={e => choose(e.target.value)}>{memoryCases.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label><label>Write to selected position<select value={index} onChange={e => { setIndex(Number(e.target.value)); setWritten(false); }}><option value={0}>picked[0]</option><option value={1}>picked[1]</option><option value={2}>picked[2]</option></select></label></div>
    <div className="numpy-code-line"><code>{state.code}<br />{`picked[${index}] = 99`}</code></div>
    <div className="numpy-linked-arrays"><ArrayGrid values={state.original} shape={[3, 2]} label="X → buffer A" selected={written && state.shares ? [index * 2] : []} sourceLabels={[0, 8, 16, 24, 32, 40].map(offset => `A + ${offset} bytes`)} /><span className="numpy-flow-arrow" aria-hidden="true" style={state.shares ? undefined : { transform: 'none' }}>{state.shares ? '↔' : '≠'}</span><ArrayGrid values={state.picked} shape={[3]} label={`picked → buffer ${state.buffer}`} selected={written ? [index] : []} sourceLabels={state.offsets.map(offset => `${state.buffer} + ${offset} bytes`)} /></div>
    <div className="numpy-storage-track"><strong>Buffer A · six float64 slots, 8 bytes each</strong><div>{state.original.map((value, i) => <span key={i} className={state.shares && i === index * 2 ? 'is-selected' : ''}><small>+{i * 8} B</small><b>{value}</b></span>)}</div></div>
    {!state.shares && <p className="numpy-memory-copy">Buffer B is a separate three-slot allocation. Equal numbers do not make its addresses the same as buffer A.</p>}
    <div className="numpy-feedback" role="status"><p>{written ? state.shares ? `Both names now expose 99 at buffer A + ${index * 16} bytes. X[${index}, 0] changed because the view points to that same storage slot.` : `Only buffer B changed. X[${index}, 0] remains ${sensorValues[index * 2]} because the selected data was copied before the write.` : 'Compare the address labels first. A view carries a different coordinate-to-address map to the SAME buffer; a copy owns independent numeric storage.'}</p><p><code>np.shares_memory(X, picked)</code>: {state.shares ? 'True' : 'False'}.</p></div>
    <div className="numpy-actions"><button type="button" disabled={written} onClick={() => setWritten(true)}>Write 99 through picked</button><button type="button" onClick={() => setWritten(false)}>Undo write</button><button type="button" onClick={() => { setKind('view'); setIndex(1); setWritten(false); }}>Reset memory lab</button></div>
    <details><summary>Transfer: where should you copy?</summary><p>You want to correct sensor A privately and preserve raw measurements. Which expression gives independent storage before the first write?</p><details><summary>Explain the solution</summary><p><code>picked = X[:, 0].copy()</code> makes independence explicit. <code>picked = X[:, 0]</code> followed by a write has already changed the raw array; copying afterwards cannot undo that change.</p></details></details>
    <p className="lesson-note">This model uses a contiguous (3, 2) float64 numeric array. A and B name buffers; their address offsets are measured in bytes. These are relative offsets, not actual machine addresses. Object dtypes, negative strides and general reshape decisions are outside this simulation.</p>
  </Lab>;
}

export function NumpyBroadcastLab() {
  const [choice, setChoice] = useState('sensor');
  const [revealed, setRevealed] = useState(false);
  const [cell, setCell] = useState(0);
  const result = broadcastResult(broadcastCases.find(item => item.id === choice));
  const active = result.cells[cell];
  const choose = id => { setChoice(id); setRevealed(false); setCell(0); };
  return <Lab name="broadcast" eyebrow="BROADCASTING · ONE RESULT, TWO INPUTS" title="Which offset reaches this exact reading?">
    <p>Predict whether the shapes work and the output shape. Reveal the result, then select different result cells. The highlighted operands and equation show exactly which values are reused.</p>
    <div className="lesson-controls"><label>Broadcast investigation<select value={choice} onChange={e => choose(e.target.value)}>{broadcastCases.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label></div>
    <div className="numpy-code-line"><code>{result.code}</code></div>
    <div className="numpy-alignment" aria-label="Shapes aligned at their last dimension"><div><span>Left</span>{result.aAligned.map((n, i) => <b key={i}>{n}</b>)}</div><div><span>Right</span>{result.bAligned.map((n, i) => <b key={i}>{n}</b>)}</div><p>Compare rightmost dimensions first. Missing leading axes are shown as 1.</p></div>
    <div className="numpy-operand-arrays"><ArrayGrid values={result.a} shape={result.aShape} label="Left operand" selected={revealed && active ? [active.aIndex] : []} /><span className="numpy-flow-arrow" aria-hidden="true">−</span><ArrayGrid values={result.b} shape={result.bShape} label="Right operand" selected={revealed && active ? [active.bIndex] : []} /></div>
    {revealed && result.valid && <div className="numpy-output"><ArrayGrid values={result.values} shape={result.shape} label="Broadcast result" selected={[cell]} onSelect={setCell} /><p className="numpy-equation">{result.a[active.aIndex]} − {result.b[active.bIndex]} = <strong>{active.value}</strong></p></div>}
    <div className="numpy-feedback" role="status">{!revealed ? <p>Do the axes match or have length one? If the operation works, which operand coordinate supplies output [0, 0]?</p> : result.valid ? <><p><strong>Output shape {shapeText(result.shape)}.</strong> {result.meaning}</p><p>Output {shapeText(active.coordinate)} uses left {shapeText(active.aCoordinate)} and right {shapeText(active.bCoordinate)}. A length-one input axis always uses coordinate 0; values are reused along the expanded output axis.</p></> : <p><strong>ValueError: no compatible broadcast.</strong> At aligned axis {result.mismatchAxis}, {result.aAligned[result.mismatchAxis]} and {result.bAligned[result.mismatchAxis]} differ and neither is 1. {result.meaning} Add the intended singleton sensor axis with <code>[:, None]</code>; do not tile or delete data to force a fit.</p>}</div>
    <div className="numpy-actions"><button type="button" onClick={() => setRevealed(!revealed)}>{revealed ? 'Hide broadcast result' : 'Reveal broadcast result'}</button><button type="button" onClick={() => choose('sensor')}>Reset broadcast lab</button></div>
    <details><summary>Transfer: the operation works, but is it the one you intended?</summary><p>Choose the square-result case. Predict a diagonal cell and a cell below the diagonal before revealing. Explain why there are nine results when v has only three values.</p><details><summary>Explain the result</summary><p>Shapes (3, 1) and (3,) align as (3, 1) and (1, 3). Both singleton axes expand, producing (3, 3). The diagonal subtracts each reading from itself (zero); below it, a later warmer reading minus an earlier one is positive. For three self-differences, use <code>v - v</code>.</p></details></details>
    <p className="lesson-note">This fixed-data model supports subtraction, scalars and up to two displayed axes. It maps operand coordinates; it does not allocate tiled copies, execute Python, or simulate NumPy's internal loops. Broadcasting avoids requiring repeated input storage, but a real computed output still needs storage.</p>
  </Lab>;
}

export function NumpyReductionLab() {
  const [axis, setAxis] = useState(0);
  const [keepdims, setKeepdims] = useState(false);
  const [revealed, setRevealed] = useState(false);
  const [cell, setCell] = useState(0);
  const result = reductionResult(axis, keepdims);
  const change = next => { setAxis(next); setCell(0); setRevealed(false); };
  return <Lab name="reduction" eyebrow="REDUCTIONS · FOLLOW THE CONTRIBUTING CELLS" title="Which values contribute to this mean?">
    <p>Predict how many means survive. Reveal and select a mean to see its contributing readings. Here all readings are temperatures in °C, so both a per-sensor mean and a per-time mean have an interpretable unit.</p>
    <div className="lesson-controls"><label>Axis to average over<select value={axis} onChange={e => change(Number(e.target.value))}><option value={0}>axis=0 · combine times</option><option value={1}>axis=1 · combine sensors</option></select></label><label>Keep the reduced axis<select value={String(keepdims)} onChange={e => { setKeepdims(e.target.value === 'true'); setRevealed(false); }}><option value="false">keepdims=False</option><option value="true">keepdims=True</option></select></label></div>
    <div className="numpy-code-line"><code>{`X.mean(axis=${axis}, keepdims=${keepdims ? 'True' : 'False'})`}</code></div>
    <div className="numpy-linked-arrays"><ArrayGrid values={sensorValues} shape={[3, 2]} label="Input readings" sensorLabels selected={revealed ? result.groups[cell] : []} /><span className="numpy-flow-arrow" aria-hidden="true">→</span>{revealed ? <ArrayGrid values={result.values} shape={result.shape} label="Mean result" selected={[cell]} onSelect={setCell} /> : <div className="numpy-await"><strong>Predict first</strong><p>Which axis disappears? Which real-world label should each surviving value have?</p></div>}</div>
    <div className="numpy-feedback" role="status">{revealed ? <><p className="numpy-equation">({result.groups[cell].map(i => sensorValues[i]).join(' + ')}) ÷ {result.groups[cell].length} = <strong>{result.values[cell]} °C</strong></p><p>{axis === 0 ? 'Time varies within each selected group; the sensor coordinate stays fixed. One result survives per sensor.' : 'Sensor varies within each selected group; the time coordinate stays fixed. One result survives per time.'} {keepdims ? `The reduced axis remains with length 1: ${shapeText(result.shape)}.` : `The reduced axis is removed: ${shapeText(result.shape)}.`}</p></> : <p>“Reduce axis {axis}” means combine values while moving along that axis. The other coordinate identifies one output group.</p>}</div>
    <div className="numpy-actions"><button type="button" onClick={() => setRevealed(!revealed)}>{revealed ? 'Hide reduction result' : 'Reveal reduction result'}</button><button type="button" onClick={() => { setAxis(0); setKeepdims(false); setCell(0); setRevealed(false); }}>Reset reduction lab</button></div>
    <details><summary>Transfer: subtract each time's mean from its two sensors</summary><p>Which axis and keepdims setting let the mean broadcast back to X?</p><details><summary>Explain the solution</summary><p><code>X - X.mean(axis=1, keepdims=True)</code> uses shape (3, 1), repeating each time's mean across its sensors. The result is three rows of [-1, 1]. Without keepdims, (3,) meets (3, 2) at incompatible trailing dimensions 3 and 2.</p></details></details>
    <p className="lesson-note">Only arithmetic means of this finite (3, 2) dataset are modelled. There are no missing values, weights, rounding changes or empty groups here; those need explicit handling in real data.</p>
  </Lab>;
}
