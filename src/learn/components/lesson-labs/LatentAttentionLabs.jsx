import RemoteCodeBlock from '../content/RemoteCodeBlock.jsx';
import { NeuralNumber } from './NeuralNumberControl.jsx';
import { useEffect, useMemo, useRef, useState } from 'react';
import { NeuralLab,  NeuralSelect, NeuralTable, NeuralPlot as BasePlot, formatNeural as f } from './NeuralLessonElements';
import { latentDefaults, latentRead, rotationOrder, rankOne, latentBudget, latentForecast, difference, matvec } from '../../data/latent-attention-models';
import './latent-attention-labs.css';
const assetBase = '/learn-assets/multi-head-latent-attention-mla/';
const vector = row => `[${row.map(value => f(value, 6)).join(', ')}]`;
const palette = ['#e4b752', '#83b4e8', '#c1a0dc', '#e6a18d'];
function NeuralPlot(props) { return <div className="mla-plot-scroll" tabIndex={0} role="region" aria-label={`${props.title}: horizontally scrollable chart`}><BasePlot {...props} /></div>; }
function MatrixEditor({
  label,
  matrix,
  onChange,
  limit = 6
}) {
  return <fieldset className="mla-matrix-editor"><legend>{label} — {matrix.length === 1 ? 'vector coordinates' : 'row: output, column: input'}</legend><div style={{
      gridTemplateColumns: `repeat(${matrix[0].length}, minmax(0,1fr))`
    }}>{matrix.flatMap((row, i) => row.map((value, j) => <NeuralNumber key={`${i}-${j}`} label={`${label} row ${i}, column ${j}`} value={value} min={-limit} max={limit} range={false} onChange={next => onChange(matrix.map((r, ri) => r.map((v, cj) => ri === i && cj === j ? next : v)))} />))}</div></fieldset>;
}
function VectorPlane({
  title,
  vectors,
  extent
}) {
  const bound = extent ?? Math.max(1, ...vectors.flatMap(v => v.value.map(Math.abs))) * 1.2;
  const x = v => 160 + v / bound * 118,
    y = v => 145 - v / bound * 118;
  return <figure className="mla-plane" tabIndex={0}><figcaption>{title}</figcaption><svg className="mla-diagram" viewBox="0 0 320 300" role="img" aria-label={vectors.map(v => `${v.label}: ${vector(v.value)}`).join('; ')}><path d="M22 145 H298 M160 18 V274" /><text x="295" y="165">x</text><text x="169" y="19">y</text><text x="165" y="162">0</text>{[-1, -.5, .5, 1].map(t => <g key={t}><text x={x(t * bound)} y="162" textAnchor="middle">{f(t * bound, 1)}</text><text x="150" y={y(t * bound) + 3} textAnchor="end">{f(t * bound, 1)}</text></g>)}{vectors.map((v, i) => <g key={v.label}><path d={`M160 145 L${x(v.value[0])} ${y(v.value[1])}`} style={{
          stroke: palette[i],
          strokeWidth: 2.5,
          strokeDasharray: i % 2 ? '5 3' : undefined
        }} /><circle cx={x(v.value[0])} cy={y(v.value[1])} r={4 + i} style={{
          fill: 'none',
          stroke: palette[i],
          strokeWidth: 2
        }} /></g>)}</svg><ul className="mla-key">{vectors.map((v, i) => <li key={v.label}><i style={{
          background: palette[i]
        }} />{v.label}: {vector(v.value)}</li>)}</ul></figure>;
}
function LatentReassociationFigure({
  expanded,
  absorbed,
  nonlinear = false
}) {
  return <figure className="mla-figure" tabIndex={0}><figcaption>{nonlinear ? 'Nonlinear contrast: moving ReLU changes the operation' : 'Moving the value map across a weighted sum'}</figcaption><div className="mla-scroll"><svg className="mla-diagram mla-wide" viewBox="0 0 690 235" role="img" aria-label={`${nonlinear ? 'Expanded applies ReLU to every mapped value before mixing; absorbed applies ReLU after mapping the mixed latent. These are different nonlinear operations.' : 'Expanded maps every latent then mixes values; absorbed mixes latents then maps once.'} Outputs ${vector(expanded)} and ${vector(absorbed)}.`}>
    {[[40, 'EXPANDED', nonlinear ? 'c → ReLU(UV c)' : 'each c → UV c', 'Σ weight × value', expanded], [140, 'ABSORBED', 'Σ weight × c', nonlinear ? 'ReLU(UV × mix)' : 'UV × mixed latent', absorbed]].map(([y, label, first, second, output]) => <g key={label}>
      <text x="15" y={y - 15}>{label}</text><rect x="15" y={y} width="130" height="43" /><text x="80" y={y + 27} textAnchor="middle">same c and weights</text><path d={`M145 ${y + 21} H188`} /><text x="168" y={y + 17}>→</text>
      <rect x="190" y={y} width="135" height="43" /><text x="257" y={y + 27} textAnchor="middle">{first}</text><path d={`M325 ${y + 21} H368`} /><text x="347" y={y + 17}>→</text>
      <rect x="370" y={y} width="140" height="43" /><text x="440" y={y + 27} textAnchor="middle">{second}</text><path d={`M510 ${y + 21} H553`} /><text x="532" y={y + 17}>→</text><rect x="555" y={y} width="122" height="43" /><text x="616" y={y + 27} textAnchor="middle">{vector(output.map(v => Number(v.toFixed(3))))}</text>
    </g>)}
    <text x="345" y="220" textAnchor="middle">One fixed linear UV can move; a nonlinear activation generally cannot.</text>
  </svg></div></figure>;
}
function ForecastSensitivityFigure({ baseline, current }) {
  const delta = current.map((v, i) => v - baseline[i]), bound = Math.max(1e-7, ...delta.map(Math.abs)) * 1.35;
  const x = v => 185 + 120 * v / bound, y = v => 140 - 100 * v / bound;
  return <figure className="mla-figure" tabIndex={0}><figcaption>Local forecast sensitivity: displacement from the original-prefix forecast</figcaption><div className="mla-scroll" tabIndex={0} role="region" aria-label="Enlarged forecast difference, horizontally scrollable"><svg className="mla-diagram mla-sensitivity" viewBox="0 0 360 290" role="img" aria-label={`Original-prefix forecast ${vector(baseline)}; edited-prefix forecast ${vector(current)}. Difference ${vector(delta)}. Same current rank and score scale for both.`}>
    <path d="M65 140 H305 M185 40 V240" />{[-1, 0, 1].map(v => <g key={v}><text x={x(v * bound)} y="259" textAnchor="middle">{(v * bound).toExponential(1)}</text><text x="58" y={y(v * bound) + 4} textAnchor="end">{(v * bound).toExponential(1)}</text></g>)}
    <path d={`M185 140 L${x(delta[0])} ${y(delta[1])}`} style={{ stroke: palette[1], strokeWidth: 2 }} /><circle cx="185" cy="140" r="7" style={{ fill: '#171717', stroke: palette[0], strokeWidth: 2 }} /><rect x={x(delta[0]) - 4} y={y(delta[1]) - 4} width="8" height="8" style={{ fill: palette[1], stroke: palette[1] }} />
    <text x="185" y="280" textAnchor="middle">Δx: edited minus original forecast</text><text x="185" y="22" textAnchor="middle">vertical: Δy; enlarged difference axes</text>
  </svg></div><p>Amber circle: original-prefix forecast {vector(baseline)}. Blue square: edited-prefix forecast {vector(current)}. Difference [{delta.map(v => v.toExponential(6)).join(', ')}]. Both use the current rank and scale. This enlarged view shows sensitivity, not a change in prediction quality; the full trajectory above retains its coordinate scale.</p></figure>;
}
export function LatentStorageFigure() {
  return <figure className="mla-figure" tabIndex={0}><figcaption>One position, one layer: shared coordinates generate distinct head representations</figcaption><div className="mla-scroll" tabIndex={0}><svg className="mla-diagram mla-wide" viewBox="0 0 690 340" role="img" aria-label="Input h splits into normalized latent c and positioned rotary key kR. These two fields are cached. Distinct UK and UV maps reconstruct each head's keys and values from c.">
    <rect x="15" y="135" width="90" height="48" /><text x="60" y="164" textAnchor="middle">input h</text>
    <path d="M105 145 L170 88 M105 172 L170 252" />
    <rect x="170" y="62" width="130" height="52" /><text x="235" y="83" textAnchor="middle">down-map</text><text x="235" y="103" textAnchor="middle">then RMSNorm</text>
    <rect x="170" y="226" width="130" height="52" /><text x="235" y="248" textAnchor="middle">rotary-key map</text><text x="235" y="268" textAnchor="middle">then R(position)</text>
    <path d="M300 88 H338 M300 252 H338" />
    <rect x="326" y="28" width="135" height="279" rx="8" style={{
          fill: 'none',
          strokeDasharray: '6 4',
          stroke: palette[0]
        }} /><text x="394" y="49" textAnchor="middle">PERSISTENT</text>
    <rect x="340" y="66" width="108" height="44" /><text x="394" y="93" textAnchor="middle">latent c</text>
    <rect x="340" y="229" width="108" height="44" /><text x="394" y="256" textAnchor="middle">rotary key kR</text>
    {[0, 1].map(h => <g key={h}><path d={`M448 88 L495 ${79 + h * 105}`} /><rect x="496" y={50 + h * 105} width="174" height="58" /><text x="583" y={72 + h * 105} textAnchor="middle">head {h}: UK{h} c / UV{h} c</text><text x="583" y={94 + h * 105} textAnchor="middle">distinct key and value</text></g>)}
    <text x="583" y="242" textAnchor="middle">Expanded K/V may be temporary.</text><text x="583" y="263" textAnchor="middle">All heads read the shared kR.</text><text x="350" y="328" textAnchor="middle">GQA directly shares K/V. MLA shares coordinates used by different maps.</text>
  </svg></div></figure>;
}
export function LatentScoreFigure() {
  return <figure className="mla-figure" tabIndex={0}><figcaption>Content and position meet before a single softmax</figcaption><div className="mla-score-rails"><div><strong>Content rail</strong><code>qCᵀ UK c</code><span>head-specific content comparison</span></div><div><strong>Position rail</strong><code>qRᵀ kR</code><span>positioned query and shared key</span></div></div><svg className="mla-diagram" viewBox="0 0 600 125" role="img" aria-label="Content and rotary dot products are added, divided by square root of dk plus dr, masked, then jointly normalized by one softmax."><path d="M150 0 V22 L270 42 M450 0 V22 L330 42 M300 70 V95" /><rect x="160" y="35" width="280" height="37" /><text x="300" y="59" textAnchor="middle">add → divide by √(dk + dr) → mask</text><text x="300" y="118" textAnchor="middle">ONE softmax over the legal positions</text></svg><p>For hand-example head 0, content [1, 0, 1] plus rotary [−1, 0, 1], divided by 2, gives [0, 0, 1]. Separate softmax operations would define a different read.</p></figure>;
}
export function LatentPathsLab() {
  const [state, setState] = useState(latentDefaults),
    [head, setHead] = useState(0),
    [entity, setEntity] = useState('latent'),
    [slot, setSlot] = useState(1),
    [basis, setBasis] = useState(false),
    [nonlinear, setNonlinear] = useState(false),
    [outputMap, setOutputMap] = useState([[1, 0, .5, 0], [0, 1, 0, .5]]);
  let absorbed, expanded, error;
  try {
    absorbed = latentRead(state, {
      changedBasis: basis,
      nonlinear
    });
    expanded = latentRead(state, {
      expanded: true,
      changedBasis: basis,
      nonlinear
    });
  } catch (e) {
    error = e.message;
  }
  const headEntities = ['queries', 'queryRotary', 'keyUp', 'valueUp'];
  const selected = headEntities.includes(entity) ? head : slot;
  const current = state[entity][selected];
  const matrix = Array.isArray(current[0]) ? current : [current];
  const edit = next => setState(previous => ({
    ...previous,
    [entity]: previous[entity].map((row, i) => i === selected ? Array.isArray(current[0]) ? next : next[0] : row)
  }));
  const reset = () => {
    setState(latentDefaults());
    setHead(0);
    setEntity('latent');
    setSlot(1);
    setBasis(false);
    setNonlinear(false);
    setOutputMap([[1, 0, .5, 0], [0, 1, 0, .5]]);
  };
  return <NeuralLab title="Move linear maps; keep the same computation" id="mla-paths"><p>Fresh constructed inputs. The two routes receive the same arrays. Select any latent, query, rotary vector or map to edit it.</p>
    <div className="mla-controls"><NeuralSelect label="Head to inspect" value={head} onChange={v => setHead(Number(v))} options={[[0, 'Head 0'], [1, 'Head 1']]} /><NeuralSelect label="Array to edit" value={entity} onChange={setEntity} options={['latent', 'queries', 'keyUp', 'valueUp', 'queryRotary', 'keyRotary'].map(v => [v, v])} /><NeuralSelect label="Memory record" value={slot} onChange={v => setSlot(Number(v))} options={state.latent.map((_, i) => [i, `Record ${i}`])} /><NeuralNumber label="Query logical position" value={state.queryPosition} min={0} max={16} step={1} integer onChange={v => setState(s => ({
        ...s,
        queryPosition: v
      }))} /><NeuralNumber label="Selected key logical position" value={state.positions[slot]} min={0} max={16} step={1} integer onChange={v => setState(s => ({
        ...s,
        positions: s.positions.map((p, i) => i === slot ? v : p)
      }))} /><NeuralNumber label="Radians per position" value={state.frequency} min={0} max={Math.PI} range={false} onChange={v => setState(s => ({
        ...s,
        frequency: v
      }))} /></div>
    <MatrixEditor label={`${entity} ${selected}`} matrix={matrix} onChange={edit} />
    <div className="mla-actions"><button onClick={() => setBasis(v => !v)}>{basis ? 'Use original coordinates' : 'Apply compensated basis diag(2, 0.5)'}</button><button onClick={() => setNonlinear(v => !v)}>{nonlinear ? 'Restore linear values' : 'Compare moving ReLU across the sum'}</button><button disabled={state.latent.length >= 8} onClick={() => setState(s => ({
        ...s,
        latent: [...s.latent, [0, 0]],
        keyRotary: [...s.keyRotary, [1, 0]],
        positions: [...s.positions, Math.min(16, s.positions.at(-1) + 1)]
      }))}>Append editable record</button><button disabled={state.latent.length <= 2} onClick={() => {
        setSlot(v => Math.min(v, state.latent.length - 2));
        setState(s => ({
          ...s,
          latent: s.latent.slice(0, -1),
          keyRotary: s.keyRotary.slice(0, -1),
          positions: s.positions.slice(0, -1)
        }));
      }}>Remove last record</button><button onClick={reset}>Reset latent paths</button></div>
    {error ? <p role="status">{error}</p> : <><div className="mla-path-columns"><section><h4>Expand before reading</h4><p>{nonlinear ? 'c → UK c, ReLU(UV c) → scores → softmax → weighted values' : 'c → UK c, UV c → scores → softmax → weighted values'}</p><NeuralTable caption={`Head ${head}: expanded records`} headers={['Position', 'Content key', 'Value']} rows={expanded[head].keys.map((k, i) => [state.positions[i], vector(k), vector(expanded[head].values[i])])} /></section><section><h4>Read before expanding</h4><p>{nonlinear ? 'q → UKᵀ q → scores → softmax → weighted c → ReLU(UV × mix)' : 'q → UKᵀ q → scores → softmax → weighted c → UV'}</p><p>Effective query: {vector(absorbed[head].effective)}</p><p>Mixed latent: {vector(absorbed[head].mixture)}</p><p>Current basis: {basis ? 'scaled c with inverse-scaled up-maps' : 'original'}</p></section></div>
      <NeuralTable caption={`Head ${head}: two signed score terms, then one distribution`} headers={['Position', 'Content', 'Rotary', 'Logit', 'Weight']} rows={absorbed[head].weights.map((w, i) => [state.positions[i], f(absorbed[head].content[i]), f(absorbed[head].rotary[i]), state.positions[i] <= state.queryPosition ? f(absorbed[head].logits[i]) : 'masked', f(w, 7)])} />
      <LatentReassociationFigure nonlinear={nonlinear} expanded={expanded[head].output} absorbed={absorbed[head].output} /><VectorPlane title="Two routes, same output when the maps are linear" vectors={[{
        label: 'Expanded',
        value: expanded[head].output
      }, {
        label: 'Absorbed',
        value: absorbed[head].output
      }]} />
      <MatrixEditor label="Output map: joined heads to two output coordinates" matrix={outputMap} onChange={setOutputMap} /><LatentOutputFigure map={outputMap} expanded={expanded.map(r => r.output).flat()} absorbed={absorbed.map(r => r.output).flat()} />
      <p>Maximum head-output difference: <strong>{difference(absorbed.map(r => r.output), expanded.map(r => r.output)).toExponential(3)}</strong>. {nonlinear ? 'ReLU is being moved across a sum: this is an intentional changed operation, not an exact MLA branch.' : 'The weighted sum distributes through a fixed linear value map. A compensated invertible basis changes coordinates without discarding information.'}</p>
    </>}
  </NeuralLab>;
}
export function LatentRotationLab() {
  const [matrix, setMatrix] = useState([[1, 1], [0, 2]]),
    [input, setInput] = useState([1, -1]),
    [angle, setAngle] = useState(Math.PI / 3);
  const result = rotationOrder(matrix, input, angle);
  const extent = Math.max(1, ...[input, result.projected, result.rotated, result.projectThenRotate, result.rotateThenProject].flat().map(Math.abs)) * 1.2;
  return <NeuralLab title="Projection then rotation is an order of operations" id="mla-rotation"><MatrixEditor label="Projection U" matrix={matrix} onChange={setMatrix} limit={4} /><div className="mla-controls">{input.map((v, i) => <NeuralNumber key={i} label={`Input coordinate ${i}`} value={v} min={-4} max={4} range={false} onChange={next => setInput(x => x.map((a, j) => j === i ? next : a))} />)}<NeuralNumber label="Rotation angle in radians" value={angle} min={-Math.PI} max={Math.PI} range={false} onChange={setAngle} /></div><p>Angle: {f(angle * 180 / Math.PI, 3)}°. Both planes use the same scale.</p><div className="mla-path-columns"><VectorPlane title="U first, then R" extent={extent} vectors={[{
        label: 'Input',
        value: input
      }, {
        label: 'After projection U',
        value: result.projected
      }, {
        label: 'After rotation R',
        value: result.projectThenRotate
      }]} /><VectorPlane title="R first, then U" extent={extent} vectors={[{
        label: 'Input',
        value: input
      }, {
        label: 'After rotation R',
        value: result.rotated
      }, {
        label: 'After projection U',
        value: result.rotateThenProject
      }]} /></div><NeuralTable caption="Commutator RU − UR" headers={['Row', 'Column 0', 'Column 1']} rows={result.commutator.map((row, i) => [i, ...row.map(v => f(v))])} /><p>Matrix difference: {f(Math.max(...result.commutator.flat().map(Math.abs)), 8)}. Selected-vector difference: {f(difference(result.projectThenRotate, result.rotateThenProject), 8)}. A zero vector can hide unequal maps; test the matrices as well as the selected input.</p><div className="mla-actions"><button onClick={() => setMatrix([[2, 0], [0, 2]])}>Use isotropic projection 2I</button><button onClick={() => setAngle(0)}>Set zero rotation</button><button onClick={() => {
        setMatrix([[1, 1], [0, 2]]);
        setInput([1, -1]);
        setAngle(Math.PI / 3);
      }}>Reset operation order</button></div></NeuralLab>;
}
const budgetDefaults = {
  batch: 3,
  layers: 24,
  length: 8192,
  heads: 24,
  content: 64,
  value: 64,
  latent: 192,
  rotary: 32,
  bytes: 2,
  queries: 1
};
export function LatentBudgetLab() {
  const [state, setState] = useState(budgetDefaults), [chosenKv, setChosenKv] = useState(8);
  const divisors = Array.from({ length: state.heads }, (_, i) => i + 1).filter(n => state.heads % n === 0);
  const kvHeads = divisors.includes(chosenKv) ? chosenKv : divisors.filter(n => n <= chosenKv).at(-1);
  const result = latentBudget(state);
  const fields = [['batch', 'Requests B', 64], ['layers', 'Layers N', 256], ['length', 'Stored positions L', 262144], ['heads', 'Query heads H', 256], ['content', 'Content width dk', 2048], ['value', 'Value width dv', 2048], ['latent', 'Latent width dc', 2048], ['queries', 'New query rows T', state.length]];
  return <NeuralLab title="A smaller record can require more arithmetic" id="mla-budget"><div className="mla-controls">{fields.map(([key, label, max]) => <NeuralNumber key={key} label={label} value={state[key]} min={key === 'rotary' ? 2 : 1} max={max} step={key === 'rotary' ? 2 : 1} range={false} integer onChange={v => {
        if (key === 'rotary' && v % 2) return;
        setState(s => ({
          ...s,
          [key]: v,
          ...(key === 'length' ? {
            queries: Math.min(s.queries, v)
          } : {})
        }));
      }} />)}<NeuralSelect label="Rotary width dr (even)" value={state.rotary} onChange={v => setState(s => ({
        ...s,
        rotary: Number(v)
      }))} options={Array.from({
        length: 128
      }, (_, i) => [(i + 1) * 2, (i + 1) * 2])} /><NeuralSelect label="Bytes per coordinate" value={state.bytes} onChange={v => setState(s => ({
        ...s,
        bytes: Number(v)
      }))} options={[[1, '1-byte payload (metadata excluded)'], [2, '2-byte payload'], [4, '4-byte payload']]} /><NeuralSelect label="GQA cached KV heads (divides query heads)" value={kvHeads} onChange={v => setChosenKv(Number(v))} options={divisors.map(n => [n, `${n} KV heads`])} /></div><div className="mla-record"><span><b>{state.latent}</b> content coordinates c</span><span><b>{state.rotary}</b> rotary coordinates kR</span></div><p>Schematic fields, with sizes counted numerically: one record has {result.record} numbers. Multiply by {state.batch} requests × {state.layers} layers × {state.length} stored positions × {state.bytes} bytes. Head count is absent from this persistent record.</p><NeuralTable caption="Calculated cache representations under the current dimensions" headers={['Representation', 'Exact bytes', 'GiB']} rows={[["Compact MLA", result.exact.compact], ['Literal expanded same MLA function', result.exact.expanded], ['Expanded content + one rotary key', result.exact.sharedRotary], ['Plain MHA with dk/dv (different parameterization)', result.exact.mha], [`GQA-${kvHeads} with dk/dv (different parameterization)`, result.exact.mqa * BigInt(kvHeads)], ['MQA with dk/dv (different parameterization)', result.exact.mqa]].map(([name, bytes]) => [name, bytes.toLocaleString('en-US'), f(Number(bytes) / 2 ** 30, 6)])} /><NeuralTable caption="Calculated arithmetic: one layer, multiply-add = two operations" headers={['Operation', 'Count']} rows={[["Expanded attention core", result.exact.expandedOps], ['Absorbed attention core', result.exact.absorbedOps], ['Reconstruct all historical K/V from latents', result.exact.reconstruction]].map(([name, count]) => [name, count.toLocaleString('en-US')])} /><p>Absorbed/expanded core ratio (rounded): {f(result.absorbedOps / result.expandedOps)}. This comparison omits projections, softmax and device behavior; these numbers are operations, not measured time.</p><div className="mla-actions"><button disabled={state.heads > 128} onClick={() => setState(s => ({
        ...s,
        heads: s.heads * 2
      }))}>Double query heads</button><button onClick={() => { setState(budgetDefaults); setChosenKv(8); }}>Reset latent budget</button></div><QuantizedMlaRecord /></NeuralLab>;
}
export function LatentRankLab() {
  const [matrix, setMatrix] = useState([[3, 1], [0, 2]]),
    [input, setInput] = useState([2, -3]),
    [full, setFull] = useState(false);
  const result = rankOne(matrix, input),
    reduced = full ? result.full : result.output;
  return <NeuralLab title="A small discarded matrix direction can carry the input" id="mla-rank"><MatrixEditor label="Matrix M" matrix={matrix} onChange={setMatrix} limit={10} /><div className="mla-controls">{input.map((v, i) => <NeuralNumber key={i} label={`Input x coordinate ${i}`} value={v} min={-10} max={10} range={false} onChange={next => setInput(x => x.map((a, j) => j === i ? next : a))} />)}<NeuralSelect label="Retained directions" value={full ? '2' : '1'} onChange={v => setFull(v === '2')} options={[[1, 'Leading one'], [2, 'Both: exact control']]} /></div><div className="mla-path-columns"><VectorPlane title="Input versus the retained singular direction" vectors={[{
        label: 'Input',
        value: input
      }, {
        label: 'Retained direction (unit length)',
        value: result.direction
      }]} /><VectorPlane title="Full map versus truncated map output" vectors={[{
        label: 'M x',
        value: result.full
      }, {
        label: full ? 'Full two-direction output' : 'M v vᵀ x',
        value: reduced
      }]} /></div><NeuralTable caption="Different objectives have different units" headers={['Quantity', 'Value']} rows={[["Singular values", vector(result.singular)], ['Squared parameter reconstruction error', f(full ? 0 : result.matrixError, 8)], ['Output error norm on this input', f(Math.hypot(...result.full.map((v, i) => v - reduced[i])), 8)], ['Top direction', vector(result.direction)]]} /><p>{result.tied ? 'The singular values tie. The displayed first axis is a deterministic choice among optimal directions; no unique direction is implied.' : 'The leading singular direction minimizes the matrix reconstruction objective. The input can point elsewhere, so this does not minimize its output error.'}</p><div className="mla-actions"><button onClick={() => setInput(result.direction.map(v => v * 3))}>Put input along retained direction</button><button onClick={() => setMatrix([[0, 0], [0, 0]])}>Use zero matrix</button><button onClick={() => {
        setMatrix([[3, 1], [0, 2]]);
        setInput([2, -3]);
        setFull(false);
      }}>Reset rank investigation</button></div></NeuralLab>;
}
export function LatentSoftmaxFigure() {
  const logits = [[0, 0, 0], [0, 1, 2], [0, 2, 4]];
  const weights = logits.map(row => {
    const terms = row.map(v => Math.exp(v - Math.max(...row))),
      sum = terms.reduce((a, b) => a + b, 0);
    return terms.map(v => v / sum);
  });
  return <figure className="mla-figure" tabIndex={0}><figcaption>A nonlinear normalization can increase matrix rank</figcaption><div className="mla-path-columns"><NeuralTable caption="Logits: rank 1 (row multiples)" headers={['Row', '0', '1', '2']} rows={logits.map((row, i) => [i, ...row])} /><NeuralTable caption="Row-softmax: rank 3" headers={['Row', '0', '1', '2']} rows={weights.map((row, i) => [i, ...row.map(v => f(v, 6))])} /></div><p>The logits factor as [0, 1, 2]ᵀ[0, 1, 2]. Exponentiation and row normalization break that linear dependence; the probability matrix has nonzero determinant ≈ 0.0244307.</p></figure>;
}
export function LatentForecastLab() {
  const host = useRef(null),
    [visible, setVisible] = useState(false),
    [payload, setPayload] = useState(null),
    [error, setError] = useState(''),
    [attempt, setAttempt] = useState(0);
  const [boundary, setBoundary] = useState(27),
    [frame, setFrame] = useState(19),
    [edits, setEdits] = useState({}),
    [rank, setRank] = useState('original'),
    [head, setHead] = useState(0),
    [shift, setShift] = useState(0),
    [wrong, setWrong] = useState(false);
  useEffect(() => {
    const observer = new IntersectionObserver(entries => {
      if (entries.some(e => e.isIntersecting)) {
        setVisible(true);
        observer.disconnect();
      }
    }, {
      rootMargin: '160px'
    });
    observer.observe(host.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!visible) return;
    const controller = new AbortController();
    setError('');
    fetch(assetBase + 'runtime.json', {
      signal: controller.signal
    }).then(r => {
      if (!r.ok) throw new Error('Model download failed.');
      return r.json();
    }).then(setPayload).catch(e => {
      if (e.name !== 'AbortError') setError('The saved model could not be loaded. Check your connection, then retry.');
    });
    return () => controller.abort();
  }, [visible, attempt]);
  const result = useMemo(() => {
    if (!payload) return null;
    const points = payload.points.slice(0, boundary).map((point, i) => edits[i] ?? point);
    const basis = rank === 'original' ? null : payload.completeBasis.map(row => row.slice(0, Number(rank)));
    const options = {
      basis,
      shift,
      wrongScale: wrong
    };
    const absorbed = latentForecast(payload.weights, points, options),
      expanded = latentForecast(payload.weights, points, {
        ...options,
        expanded: true
      });
    const original = latentForecast(payload.weights, points, {
      shift
    });
    const correctScale = latentForecast(payload.weights, points, {
      basis,
      shift
    });
    const baseline = latentForecast(payload.weights, payload.points.slice(0, boundary), options);
    let cache = null;
    const incremental = [];
    for (const point of points) {
      const next = latentForecast(payload.weights, [point], {
        ...options,
        cache
      });
      cache = next.cache;
      incremental.push(next.predictions[0]);
    }
    return {
      points,
      absorbed,
      expanded,
      original,
      correctScale,
      baseline,
      cacheError: difference(incremental, absorbed.predictions)
    };
  }, [payload, boundary, edits, rank, shift, wrong]);
  const selected = Math.min(frame, boundary - 1),
    point = result?.points[selected];
  const reset = () => {
    setBoundary(27);
    setFrame(19);
    setEdits({});
    setRank('original');
    setHead(0);
    setShift(0);
    setWrong(false);
  };
  return <div ref={host}><NeuralLab title="Inspect an actual normalized latent cache" id="mla-forecast"><p>UCI Libras source row 77, initially 27 observed points. The selected model is fixed. Rank changes project its original normalized eight-coordinate latent; they do not train a new architecture.</p>{error ? <p role="alert">{error} <button onClick={() => setAttempt(v => v + 1)}>Retry model download</button></p> : !result ? <p role="status">Loading the selected small model…</p> : <>
    <div className="mla-controls"><NeuralNumber label="Observed prefix length" value={boundary} min={2} max={40} step={1} integer onChange={setBoundary} /><NeuralNumber label="Observed frame to edit" value={selected} min={0} max={boundary - 1} step={1} integer onChange={setFrame} />{[0, 1].map(axis => <NeuralNumber key={axis} label={`Observed ${axis ? 'y' : 'x'} at frame ${selected}`} value={point[axis]} min={0} max={1} range={false} onChange={v => setEdits(previous => ({
            ...previous,
            [selected]: point.map((x, d) => d === axis ? v : x)
          }))} />)}<NeuralSelect label="Cached content representation" value={rank} onChange={setRank} options={[["original", 'Original eight coordinates'], ...Array.from({
            length: 8
          }, (_, i) => [String(i + 1), i === 7 ? 'Full 8-direction basis: no loss' : `${i + 1} singular directions`])]} /><NeuralNumber label="Common logical position shift" value={shift} min={0} max={128} step={1} integer onChange={setShift} /><NeuralSelect label="Score scale" value={wrong ? 'wrong' : 'correct'} onChange={v => setWrong(v === 'wrong')} options={[["correct", 'Original 1/√6'], ['wrong', 'Default based on latent width']]} /><NeuralSelect label="Head to inspect" value={head} onChange={v => setHead(Number(v))} options={[0, 1, 2, 3].map(h => [h, `Head ${h}`])} /></div>
    <div className="mla-actions"><button onClick={() => setEdits(previous => ({
            ...previous,
            [selected]: [point[0], 1 - point[1]]
          }))}>Reflect selected y</button><button onClick={reset}>Reset latent forecast</button></div>
    <NeuralPlot title="Observed trajectory and one-step forecast" xLabel="x coordinate" yLabel="y coordinate" xDomain={[Math.min(0, ...result.absorbed.predictions.at(-1).slice(0, 1)), Math.max(1, result.absorbed.predictions.at(-1)[0])]} yDomain={[Math.min(0, result.absorbed.predictions.at(-1)[1]), Math.max(1, result.absorbed.predictions.at(-1)[1])]} series={[{
          label: 'Observed prefix',
          color: palette[0],
          values: result.points
        }, {
          label: 'Current next forecast',
          color: palette[1],
          dashed: true,
          values: [result.points.at(-1), result.absorbed.predictions.at(-1)]
        }, {
          label: 'Original-prefix forecast, same rank/scale',
          color: palette[3],
          dashed: true,
          values: [payload.points[boundary - 1], result.baseline.predictions.at(-1)]
        }, {
          label: 'Recorded next point (not input)',
          color: palette[2],
          dashed: true,
          values: [payload.points[boundary - 1], payload.points[boundary]]
        }]} />
    <ForecastSensitivityFigure baseline={result.baseline.predictions.at(-1)} current={result.absorbed.predictions.at(-1)} /><NeuralTable caption="Same-state execution versus changed representation" headers={['Computation', 'Next predicted point']} rows={[["Expanded path, current rank/scale", vector(result.expanded.predictions.at(-1))], ['Absorbed path, current rank/scale', vector(result.absorbed.predictions.at(-1))], ['Original full model, correct scale, current points', vector(result.original.predictions.at(-1))], ['Current rank, correct scale, current points', vector(result.correctScale.predictions.at(-1))], ['Recorded next point: reference only', vector(payload.points[boundary])]]} />
    <p>Expanded/absorbed maximum difference: {difference(result.absorbed.predictions, result.expanded.predictions).toExponential(3)}. Full/incremental difference: {result.cacheError.toExponential(3)}. Current-vs-correct-scale maximum difference: {difference(result.absorbed.predictions, result.correctScale.predictions).toExponential(3)}.</p>
    <p>Float32 cache: [{boundary}, {rank === 'original' ? 8 : rank}] latent + [{boundary}, 2] rotary = <strong>{boundary * ((rank === 'original' ? 8 : Number(rank)) + 2) * 4} bytes</strong>, excluding metadata. Inputs before the first edit retain maximum prediction difference {Math.max(0, ...result.absorbed.predictions.slice(0, Object.keys(edits).length ? Math.min(...Object.keys(edits).map(Number)) : boundary).flatMap((p, i) => p.map((v, d) => Math.abs(v - result.baseline.predictions[i][d])))).toExponential(3)}.</p>
    <LatentCacheCoordinates latent={result.absorbed.cache.latents[selected]} rotary={result.absorbed.cache.rotaryKeys[selected]} position={selected} head={head} contentScore={result.absorbed.traces.at(-1)[head].contentScores[selected]} rotaryScore={result.absorbed.traces.at(-1)[head].rotaryScores[selected]} />
    <NeuralTable caption={`Cache record ${selected}: two persistent fields`} headers={['Field', 'Current coordinates']} rows={[["Normalized full latent before any projection", vector(result.absorbed.fullLatents[selected])], ['Stored latent after selected basis', vector(result.absorbed.cache.latents[selected])], ['Stored rotated shared key', vector(result.absorbed.cache.rotaryKeys[selected])]]} />
    <NeuralPlot title={`Head ${head}: actual final-query attention`} xLabel="observed frame" yLabel="weight" xDomain={[0, boundary - 1]} yDomain={[0, 1]} series={[{
          label: 'Current weights',
          color: palette[0],
          values: result.absorbed.traces.at(-1)[head].weights.map((w, i) => [i, w])
        }]} />
    <p>Mixed latent: {vector(result.absorbed.traces.at(-1)[head].mixture)}. Head output: {vector(result.absorbed.traces.at(-1)[head].output)}.</p>
    <section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Inspect score contributions for every stored position</h4><NeuralTable caption="Current head scores and weights" headers={['Frame', 'Content dot', 'Rotary dot', 'Scaled logit', 'Weight']} rows={result.absorbed.traces.at(-1)[head].weights.map((w, i) => [i, f(result.absorbed.traces.at(-1)[head].contentScores[i]), f(result.absorbed.traces.at(-1)[head].rotaryScores[i]), f(result.absorbed.traces.at(-1)[head].logits[i]), f(w, 8)])} /></section>
  </>}</NeuralLab></div>;
}
export function LatentProgram({
  file
}) {
  return <>
    <RemoteCodeBlock source={assetBase + file} language="python" filename={file} title={"Read complete " + (file)} />
  </>;
}

function LatentOutputFigure({ map, expanded, absorbed }) {
  const left = matvec(map, expanded), right = matvec(map, absorbed);
  return <figure className="mla-figure" tabIndex={0}><figcaption>Join both head outputs, then apply the output map</figcaption><div className="mla-scroll" tabIndex={0}><svg className="mla-diagram mla-output-diagram" viewBox="0 0 720 255" role="img" aria-label={`Joined absorbed head vector ${vector(absorbed)}; output map gives ${vector(right)}. Expanded route gives ${vector(left)}.`}>
    {absorbed.map((v, i) => <g key={i}><rect x="15" y={18 + 55 * i} width="142" height="36" /><text x="86" y={40 + 55 * i} textAnchor="middle">head {Math.floor(i / 2)}[{i % 2}] = {f(v, 4)}</text>{map.map((row, j) => <path key={j} d={`M157 ${36 + 55 * i} L245 ${76 + 105 * j}`} style={{ stroke: palette[i], strokeDasharray: i % 2 ? '4 3' : undefined }} />)}</g>)}
    {map.map((row, j) => <g key={j}><rect x="247" y={51 + j * 105} width="257" height="51" /><text x="375" y={73 + j * 105} textAnchor="middle">output row {j}: {vector(row)}</text><text x="375" y={92 + j * 105} textAnchor="middle">dot with joined head vector</text><path d={`M504 ${76 + j * 105} H547`} /><text x="524" y={70 + j * 105}>→</text><rect x="550" y={57 + j * 105} width="152" height="37" /><text x="626" y={81 + j * 105} textAnchor="middle">contribution {f(right[j], 5)}</text></g>)}
  </svg></div><NeuralTable caption="Output-coordinate dot products: same map on both execution routes" headers={['Coordinate', 'Map row', 'Expanded contribution', 'Absorbed contribution']} rows={map.map((row, i) => [i, vector(row), f(left[i], 8), f(right[i], 8)])} /><p>Joined expanded {vector(expanded)}; joined absorbed {vector(absorbed)}. Output-map edits change the final attention contribution, while head scores, weights and outputs stay fixed. This contribution is before any surrounding residual addition.</p></figure>;
}
function QuantizedMlaRecord() {
  const segments = [['512 FP8 latent coordinates', 512], ['scale metadata', 16], ['64 BF16 rotary coordinates', 128]];
  return <figure className="mla-figure" tabIndex={0}><figcaption>A specific documented V3.2 sparse FlashMLA record: 656 bytes</figcaption><div className="mla-scroll" tabIndex={0}><svg className="mla-diagram mla-quantized-record" viewBox="0 0 680 210" role="img" aria-label="512 bytes latent payload plus 16 scale bytes plus 128 rotary bytes equals 656 bytes. Specific V3.2 sparse format, not a universal MLA record.">{segments.map(([name, size], i) => { const start = segments.slice(0, i).reduce((sum, row) => sum + row[1], 0); return <g key={name}><rect x={12 + start} y="20" width={size} height="40" style={{ fill: palette[i] }} /><text x="16" y={98 + 35 * i}>{name}: {size} bytes</text></g>; })}<text x="16" y="205">512 + 16 + 128 = 656 bytes; metadata is not a latent coordinate</text></svg></div><p>This layout is the documented V3.2 sparse FlashMLA serving format described in the manuscript. It is separate from the unquantized calculator; other versions and cache formats have different fields.</p></figure>;
}
function LatentCacheCoordinates({ latent, rotary, position, head, contentScore, rotaryScore }) {
  const extent = Math.max(1e-8, ...latent.map(Math.abs), ...rotary.map(Math.abs));
  return <figure className="mla-figure" tabIndex={0}><figcaption>Stored record {position}: two signed coordinate fields feed head {head}</figcaption><div className="mla-scroll" tabIndex={0}><svg className="mla-diagram mla-cache-coordinates" viewBox="0 0 720 305" role="img" aria-label={`Stored latent ${vector(latent)} and rotated key ${vector(rotary)}, common signed extent ${extent}. Final query head${head} produces content dot${contentScore} and rotary dot${rotaryScore}.`}>{[[latent, 'cached content latent c', contentScore], [rotary, 'cached rotated key kR', rotaryScore]].map(([values, name, score], row) => <g key={name}><text x="12" y={26 + row * 133}>{name}</text><path d={`M12 ${72 + row * 133} H366`} style={{ stroke: '#bbb', strokeDasharray: '3 3' }} />{values.map((v, i) => <g key={i}><rect x={17 + i * 43} y={72 + row * 133 - Math.max(0, v) / extent * 28} width="32" height={Math.abs(v) / extent * 28} style={{ fill: v < 0 ? palette[2] : palette[0] }} /><text x={33 + i * 43} y={39 + row * 133} textAnchor="middle">{i}</text><text x={33 + i * 43} y={119 + row * 133} textAnchor="middle">{f(v, 2)}</text></g>)}<path d={`M373 ${72 + row * 133} H423`} /><text x="395" y={64 + row * 133}>→</text><rect x="429" y={48 + row * 133} width="274" height="49" /><text x="566" y={69 + row * 133} textAnchor="middle">head {head}: {row ? 'rotated query dot kR' : 'effective query dot c'}</text><text x="566" y={88 + row * 133} textAnchor="middle">signed score {f(score, 7)}</text></g>)}<text x="12" y="292">Dashed rails are zero; common bar scale ±{f(extent, 5)}. Both scores enter the same final-query logit.</text></svg></div></figure>;
}
