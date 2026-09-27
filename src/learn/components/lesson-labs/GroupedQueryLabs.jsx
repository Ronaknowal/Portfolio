import RemoteCodeBlock from '../content/RemoteCodeBlock.jsx';
import { NeuralNumber } from './NeuralNumberControl.jsx';
import { GqaPromptDiagram, GqaPayloadDiagram } from './GroupedQueryDiagrams.jsx';
import { useEffect, useMemo, useRef, useState } from 'react';
import { NeuralLab, NeuralSelect, NeuralTable, NeuralPlot as BaseNeuralPlot, formatNeural as f } from './NeuralLessonElements';
import { groupedDefaults, groupedRead, cacheBudget, conversionRead, groupedForecast, groupedRollout } from '../../data/grouped-query-models';
import './grouped-query-labs.css';
import studyHistory from '../../data/grouped-query-study';
const vector = row => `[${row.map(value => f(value, 6)).join(', ')}]`;
const assetBase = '/learn-assets/grouped-query-attention-gqa-multi-query-attention-mqa/';
const copy = value => structuredClone(value);
function NeuralPlot(props) { return <div className="gqa-plot-scroll" tabIndex={0} role="region" aria-label={`${props.title}: horizontally scrollable chart`}><BaseNeuralPlot {...props} /></div>; }
const palette = ['#e4b752', '#83b4e8', '#c1a0dc', '#e6a18d'];
export function GqaCacheTimeline() {
  return <><GqaPromptDiagram /><figure className="gqa-figure" tabIndex={0}><figcaption>One layer: each new query reads every legal stored record</figcaption>
    <div className="gqa-diagram-scroll" tabIndex={0} role="region" aria-label="Three-step causal cache timeline">
      <svg className="gqa-diagram" viewBox="0 0 660 265" role="img" aria-label="At positions 0, 1 and 2, project one new Q and one K/V record. The query reads the old and new records; only K/V persist.">
        {[0, 1, 2].map(t => <g key={t} transform={`translate(${t * 215 + 10},0)`}>
          <text x="100" y="22" textAnchor="middle">Position {t}</text>
          <rect x="25" y="38" width="150" height="34" rx="4" /><text x="100" y="60" textAnchor="middle">new input → projections</text>
          <path d="M60 72 V102 M140 72 V102" /><text x="60" y="120" textAnchor="middle">Q{t}</text><text x="140" y="120" textAnchor="middle">K{t} / V{t}</text>
          {Array.from({
            length: t + 1
          }, (_, j) => <g key={j}><rect x={10 + j * 64} y="152" width="59" height="34" rx="3" /><text x={39 + j * 64} y="174" textAnchor="middle">KV{j}</text><path d={`M60 127 L${39 + j * 64} 150`} strokeDasharray="4 3" /></g>)}
          <text x="100" y="211" textAnchor="middle">Q{t} → mixed output</text><text x="100" y="237" textAnchor="middle">retain {t + 1} K/V records</text>
        </g>)}
      </svg>
    </div><p>Solid branches create new projections. Dashed branches read stored vectors. Earlier K/V remain; the old query has finished its job. Each layer owns its own cache.</p>
  </figure></>;
}
export function GqaWiring({
  mapping: supplied,
  selected = 0,
  onSelect
}) {
  const [count, setCount] = useState(2);
  const mapping = supplied ?? [0, 1, 2, 3].map(h => Math.floor(h / (4 / count)));
  const groups = Math.max(...mapping) + 1;
  return <figure className="gqa-figure" tabIndex={0}><figcaption>Fixed head assignment: four readers, {groups} stored descriptions</figcaption>
    {!supplied && <NeuralSelect label="Distinct KV heads" value={count} onChange={value => setCount(Number(value))} options={[[4, 'MHA: four'], [2, 'GQA: two'], [1, 'MQA: one']]} />}
    <svg className="gqa-diagram" viewBox="0 0 420 230" role="img" aria-label={mapping.map((g, h) => `Query ${h} reads KV group ${g}`).join('; ')}>
      {mapping.map((g, h) => <g key={h}><path d={`M130 ${32 + h * 52} C215 ${32 + h * 52},215 ${115 + (g - (groups - 1) / 2) * 50},280 ${115 + (g - (groups - 1) / 2) * 50}`} style={{
          stroke: palette[g],
          strokeWidth: h === selected ? 3 : 1
        }} /><rect x="30" y={16 + h * 52} width="100" height="32" rx="4" /><text x="80" y={37 + h * 52} textAnchor="middle">Query {h}</text></g>)}
      {Array.from({
        length: groups
      }, (_, g) => <g key={g}><rect x="280" y={99 + (g - (groups - 1) / 2) * 50} width="110" height="32" rx="4" style={{
          stroke: palette[g]
        }} /><text x="335" y={120 + (g - (groups - 1) / 2) * 50} textAnchor="middle">KV group {g}</text></g>)}
    </svg>
    {onSelect && <NeuralSelect label="Inspect query head" value={selected} onChange={v => onSelect(Number(v))} options={mapping.map((g, h) => [h, `Query ${h} → KV ${g}`])} />}
    <p>The lines assign heads to groups. The distribution over token positions is computed separately from each query and its group's keys.</p>
  </figure>;
}
function VectorMixture({
  values,
  weights,
  output
}) {
  const scale = 25,
    x = value => 150 + value * scale,
    y = value => 145 - value * scale;
  return <figure className="gqa-figure" tabIndex={0}><figcaption>Values and their weighted mixture in two coordinates</figcaption>
    <svg className="gqa-diagram gqa-vector-diagram" viewBox="0 0 310 300" role="img" aria-label={`Mixed value ${vector(output)}. Exact contributions appear in the table.`}>
      <path d="M20 145 H290 M150 20 V275" />
      <text x="288" y="164">x</text><text x="158" y="19">y</text><text x="155" y="162">0</text>
      {[-4, -2, 2, 4].map(v => <g key={v}><text x={x(v)} y="161" textAnchor="middle">{v}</text><text x="143" y={y(v) + 4} textAnchor="end">{v}</text></g>)}
      {values.map((v, i) => <g key={i}><path d={`M150 145 L${x(v[0])} ${y(v[1])}`} style={{
          stroke: palette[i % 4],
          strokeDasharray: '4 3'
        }} /><circle cx={x(v[0])} cy={y(v[1])} r="4" style={{
          fill: palette[i % 4]
        }}><title>Position {i}: value {vector(v)}, weight {f(weights[i])}</title></circle></g>)}
      <path d={`M150 145 L${x(output[0])} ${y(output[1])}`} style={{
        stroke: '#fff',
        strokeWidth: 3
      }} /><circle cx={x(output[0])} cy={y(output[1])} r="6" fill="#fff" />
    </svg><p>Dashed vectors are stored values. The solid white vector is their weighted sum. Probabilities choose the mixture; vector length is a coordinate magnitude.</p>
  </figure>;
}
export function GqaReadLab() {
  const [state, setState] = useState(groupedDefaults), [head, setHead] = useState(0), [entity, setEntity] = useState('query'), [slot, setSlot] = useState(0);
  const [mask, setMask] = useState('logical'), [queryRow, setQueryRow] = useState(0);
  const logicalPosition = state.queryPosition + queryRow;
  const legal = state.positions.map((position, i) => mask === 'logical' ? position <= logicalPosition : i <= queryRow);
  const compute = allowed => allowed.some(Boolean) ? groupedRead({ ...state, positions: allowed.map(v => v ? 0 : 2), queryPosition: 1 }) : null;
  const rows = compute(legal), reference = compute(state.positions.map(p => p <= logicalPosition)), row = rows?.[head], group = state.mapping[head];
  const baseline = groupedRead(groupedDefaults());
  const current = entity === 'query' ? state.query[head] : state[entity][group][slot];
  const edit = (axis, value) => setState(previous => { const next = copy(previous); (entity === 'query' ? next.query[head] : next[entity][group][slot])[axis] = value; return next; });
  const reverse = () => { setState(s => ({ ...s, positions: [...s.positions].reverse(), keys: s.keys.map(r => [...r].reverse()), values: s.values.map(r => [...r].reverse()) })); setSlot(state.positions.length - 1 - slot); };
  const append = () => { setState(s => ({ ...s, positions: [...s.positions, Math.min(256, Math.max(...s.positions) + 1)], keys: s.keys.map(r => [...r, [1, 1]]), values: s.values.map(r => [...r, [4, -4]]) })); setSlot(state.positions.length); };
  const remove = () => { setState(s => ({ ...s, positions: s.positions.filter((_, i) => i !== slot), keys: s.keys.map(r => r.filter((_, i) => i !== slot)), values: s.values.map(r => r.filter((_, i) => i !== slot)) })); setSlot(Math.max(0, slot - 1)); };
  const reset = () => { setState(groupedDefaults()); setHead(0); setSlot(0); setEntity('query'); setMask('logical'); setQueryRow(0); };
  const difference = rows && reference ? Math.max(...rows.flatMap((r, h) => r.output.map((v, d) => Math.abs(v - reference[h].output[d])))) : null;
  return <NeuralLab title="Different readers of one shared memory" id="gqa-read">
    <p>Constructed arithmetic inputs. Edit one query, or a key/value in the selected reader's group. All four outputs update immediately. Each of the two query rows uses these same four editable query vectors; its logical position can admit different records.</p>
    <GqaWiring mapping={state.mapping} selected={head} onSelect={setHead} />
    <div className="gqa-controls"><NeuralSelect label="Entity to edit" value={entity} onChange={setEntity} options={['query', 'keys', 'values'].map(v => [v, v])} /><NeuralSelect label="Memory slot to edit" value={slot} onChange={v => setSlot(Number(v))} options={state.positions.map((p, i) => [i, `Slot ${i}, logical ID ${p}`])} />
      {[0, 1].map(axis => <NeuralNumber key={axis} range={false} label={`${entity} ${axis === 0 ? 'x' : 'y'}`} value={current[axis]} min={-4} max={4} onChange={v => edit(axis, v)} />)}
      <NeuralNumber label="Selected key logical position" value={state.positions[slot]} min={0} max={256} integer step={1} range={false} onChange={v => setState(s => ({ ...s, positions: s.positions.map((p, i) => i === slot ? v : p) }))} />
      <NeuralNumber label="First query logical position" value={state.queryPosition} min={0} max={255} integer step={1} range={false} onChange={v => setState(s => ({ ...s, queryPosition: v }))} />
      <NeuralSelect label="Inspect query row" value={queryRow} onChange={v => setQueryRow(Number(v))} options={[[0, `Row 0: logical ID ${state.queryPosition}`], [1, `Row 1: logical ID ${state.queryPosition + 1}`]]} />
      <NeuralSelect label="Edited-memory mask relation" value={mask} onChange={setMask} options={[["logical", 'Key ID ≤ query ID'], ['upper-left', 'Local upper-left triangle']]} />
    </div>
    <div className="gqa-actions"><button disabled={state.positions.length === 8} onClick={append}>Append future record</button><button disabled={state.positions.length === 1} onClick={remove}>Remove selected record</button><button onClick={reverse}>Reverse slots with complete records</button><button onClick={() => setState(s => ({ ...s, mapping: [0, 1, 0, 1] }))}>Use interleaved routing</button><button onClick={() => setState(s => ({ ...s, keys: [...s.keys].reverse(), values: [...s.values].reverse(), mapping: s.mapping.map(g => 1 - g) }))}>Relabel both groups consistently</button><button onClick={reset}>Reset shared read</button></div>
    <p>Append initializes both groups' new key to [1, 1] and value to [4, −4], with ID one beyond the largest stored ID (bounded at 256). Edit all of them directly. A future record stays excluded until its ID is legal. {state.positions.length} physical records per KV group; compact K/V payload {2 * 2 * state.positions.length * 2 * 8} bytes in this float64 calculation, versus {2 * 4 * state.positions.length * 2 * 8} bytes if copied to all four heads.</p>
    <NeuralTable caption="Two query positions over the same edited memory" headers={['Query row / logical ID', ...state.positions.map((p, i) => `Slot ${i} / key ${p}`)]} rows={[0, 1].map(r => [`${r} / ${state.queryPosition + r}`, ...state.positions.map((p, i) => (mask === 'logical' ? p <= state.queryPosition + r : i <= r) ? 'Allowed' : 'Masked')])} />
    {row ? <div className="gqa-columns"><VectorMixture values={state.values[group]} weights={row.weights} output={row.output} /><NeuralTable caption={`Query ${head} at logical ${logicalPosition}: actual attention`} headers={['Slot / ID', 'Legality', 'Scaled score', 'Weight', 'Contribution']} rows={row.weights.map((w, i) => [`${i} / ${state.positions[i]}`, legal[i] ? 'Allowed' : 'Masked', f(row.scores[i]), f(w, 6), vector(row.contributions[i])])} /></div> : <p role="status">No legal key for this query. No attention probabilities or mixed output are defined. Lower a key ID or raise the query ID.</p>}
    {rows && <NeuralTable caption="All reader outputs and difference from the original fixture" headers={['Query → group', 'Output', 'Difference']} rows={rows.map((r, h) => [`${h} → ${r.group}`, vector(r.output), vector(r.output.map((v, d) => v - baseline[h].output[d]))])} />}
    <p>Maximum difference from the explicit logical-position reference: <strong>{difference === null ? 'not defined: one relation has no legal key' : f(difference, 10)}</strong>. A local-index triangle can read a different history. Moving complete records preserves the logical relation's output.</p>
    <p>A value edit changes mixtures without changing weights. A key edit changes only dot products that use that key; an orthogonal query gives an exact unchanged score. Relabeling both groups and their connections preserves the function.</p>
  </NeuralLab>;
}
const budgetDefaults = {
  batch: 1,
  layers: 80,
  length: 32768,
  queryHeads: 64,
  kvHeads: 8,
  keyWidth: 128,
  valueWidth: 128,
  bytes: 2,
  modelWidth: 512
};
export function GqaBudgetLab() {
  const [state, setState] = useState(budgetDefaults),
    [fraction, setFraction] = useState(.6),
    [factor, setFactor] = useState(8);
  const budget = cacheBudget(state);
  const fields = [['batch', 'Requests', 64], ['layers', 'Layers', 256], ['length', 'Occupied tokens', 262144], ['keyWidth', 'Key width', 256], ['valueWidth', 'Value width', 256], ['modelWidth', 'Model width', 8192]];
  return <NeuralLab title="Build the cache payload one axis at a time" id="gqa-budget">
    <div className="gqa-controls">{fields.map(([key, label, max]) => <NeuralNumber key={key} label={label} value={state[key]} min={1} max={max} step={1} integer range={false} onChange={v => setState(s => ({
        ...s,
        [key]: v
      }))} />)}
      <NeuralSelect label="Query heads" value={state.queryHeads} onChange={v => setState(s => ({
        ...s,
        queryHeads: Number(v),
        kvHeads: Number(v) % s.kvHeads === 0 ? s.kvHeads : 1
      }))} options={Array.from({ length: 128 }, (_, i) => [i + 1, i + 1])} />
      <NeuralSelect label="KV heads" value={state.kvHeads} onChange={v => setState(s => ({
        ...s,
        kvHeads: Number(v)
      }))} options={Array.from({ length: state.queryHeads }, (_, i) => i + 1).filter(v => state.queryHeads % v === 0).map(v => [v, v])} />
      <NeuralSelect label="Bytes per stored number" value={state.bytes} onChange={v => setState(s => ({
        ...s,
        bytes: Number(v)
      }))} options={[[2, '2: FP16/BF16'], [4, '4: float32']]} />
    </div>
    <p>KV choices are divisors of the query-head count. Changing query heads keeps a compatible KV count; otherwise it selects one shared KV head so the grouping stays defined.</p>
    <GqaPayloadDiagram state={state} /><div className="gqa-factor-strip">{[['requests', state.batch], ['layers', state.layers], ['tokens', state.length], ['KV heads', state.kvHeads], ['K + V width', state.keyWidth + state.valueWidth], ['bytes', state.bytes]].map(([label, value]) => <span key={label}><b>{value}</b><small>{label}</small></span>)}</div>
    <NeuralTable caption="Calculated tensor payload, excluding metadata and other model state" headers={['Quantity', 'Current value']} rows={[["Total bytes", budget.total.toLocaleString('en-US')], ['Binary GiB', f(budget.total / 2 ** 30, 6)], ['Decimal GB', f(budget.total / 1e9, 6)], ['Bytes per occupied token (all requests/layers)', budget.perToken.toLocaleString('en-US')], ['Same dimensions with MHA', `${f(budget.mha / 2 ** 30)} GiB`], ['Bias-free attention projection parameters', budget.parameters.toLocaleString('en-US')], ['Score + value-mix operations, one layer and decode query', budget.denseOps.toLocaleString('en-US')]]} />
    <p>Reducing KV heads from {state.queryHeads} to {state.kvHeads} saves a factor {state.queryHeads / state.kvHeads} in these stored tensors. All {state.queryHeads} queries still perform their own score and value mixing.</p>
    <h4>Separate assumed timing model</h4><div className="gqa-controls"><NeuralNumber label="Affected time fraction" value={fraction} min={0} max={1} step={.05} onChange={setFraction} /><NeuralNumber label="Assumed acceleration of that part" value={factor} min={1} max={32} step={1} onChange={setFactor} /></div>
    <p>Calculated total speedup: <strong>{f(1 / (1 - fraction + fraction / factor), 6)}×</strong>. This is 1 / ({f(1 - fraction)} + {f(fraction)} / {factor}), an assumption-based calculation, not measured latency.</p>
    <button onClick={() => {
      setState(budgetDefaults);
      setFraction(.6);
      setFactor(8);
    }}>Reset cache budget</button>
  </NeuralLab>;
}
export function GqaMaskLab() {
  const [position, setPosition] = useState(3),
    [length, setLength] = useState(5),
    [mode, setMode] = useState('logical');
  const [keyShift, setKeyShift] = useState(0),
    [reverse, setReverse] = useState(false);
  const keys = Array.from({
    length
  }, (_, i) => ({
    id: i + keyShift,
    key: i % 2 ? [0, 1] : [1, 0],
    value: [i + 1, length - i]
  }));
  if (reverse) keys.reverse();
  const matrices = [position, position + 1].map((q, row) => keys.map((key, slot) => mode === 'logical' ? key.id <= q : slot <= row));
  const results = matrices.map(legal => {
    if (!legal.some(Boolean)) return null;
    const fixture = {
      query: [[Math.sqrt(2), 0]],
      keys: [keys.map(k => k.key)],
      values: [keys.map(k => k.value)],
      mapping: [0],
      positions: keys.map((_, i) => legal[i] ? 0 : 2),
      queryPosition: 1
    };
    return groupedRead(fixture)[0];
  });
  return <NeuralLab title="A one-row query can live at position 32" id="gqa-mask">
    <p>Local query row and logical sequence position are different coordinates. The two query rows below use identical vectors; the legal history is the only difference.</p>
    <div className="gqa-controls"><NeuralNumber label="First logical query position" value={position} min={0} max={16} step={1} integer onChange={setPosition} /><NeuralNumber label="Stored key count" value={length} min={3} max={8} step={1} integer onChange={setLength} /><NeuralNumber label="First key logical position" value={keyShift} min={0} max={16} step={1} integer onChange={setKeyShift} /><NeuralSelect label="Mask relation" value={mode} onChange={setMode} options={[["logical", 'Key ID ≤ query ID'], ['upper-left', 'Local upper-left triangle']]} /></div>
    <NeuralTable caption="Legal keys and live outputs" headers={['Query ID', ...keys.map(k => `Key ID ${k.id}`), 'Output']} rows={matrices.map((legal, r) => [position + r, ...legal.map(v => v ? 'Allowed' : 'Masked'), results[r] ? vector(results[r].output) : 'No legal key'])} />
    <div className="gqa-actions"><button onClick={() => setReverse(v => !v)}>Reverse physical slots with records</button><button onClick={() => {
        setPosition(3);
        setLength(5);
        setKeyShift(0);
        setMode('logical');
        setReverse(false);
      }}>Reset offset mask</button></div>
    <p>Moving whole records between physical slots preserves the logical-ID computation. A local-index triangle instead changes which records participate when slots move. An empty legal set is shown explicitly.</p>
  </NeuralLab>;
}
export function GqaConversionLab() {
  const [query, setQuery] = useState([1, 2]),
    [keys, setKeys] = useState([[2, 0], [0, 2]]),
    [values, setValues] = useState([[1, 3], [5, -1]]), [offsets, setOffsets] = useState([[0, 0], [0, 0]]);
  const mean = conversionRead(query, keys, values)[0];
  const rows = conversionRead(query, keys, values, { keys: mean.averagedKeys.map((v, i) => v + offsets[0][i]), values: mean.averagedValues.map((v, i) => v + offsets[1][i]) });
  const update = (setter, h, slot, value) => setter(previous => previous.map((row, i) => i === h ? row.map((v, j) => j === slot ? value : v) : row));
  return <NeuralLab title="A nearby parameter can produce a different answer" id="gqa-conversion">
    <p>Two original scalar heads are merged into one K/V head. Their queries remain distinct. Edit the original entries; the means, distributions and outputs are recalculated.</p>
    {[0, 1].map(h => <fieldset key={h}><legend>Original head {h}</legend><div className="gqa-controls"><NeuralNumber label={`Query ${h}`} value={query[h]} min={-6} max={6} range={false} onChange={v => setQuery(q => q.map((x, i) => i === h ? v : x))} />{[0, 1].map(slot => <div key={slot}><NeuralNumber label={`Head ${h} key ${slot}`} value={keys[h][slot]} min={-6} max={6} range={false} onChange={v => update(setKeys, h, slot, v)} /><NeuralNumber label={`Head ${h} value ${slot}`} value={values[h][slot]} min={-6} max={6} range={false} onChange={v => update(setValues, h, slot, v)} /></div>)}</div></fieldset>)}
    <div className="gqa-factor-strip"><span><small>Mean keys</small><b>{vector(rows[0].averagedKeys)}</b></span><span><small>Mean values</small><b>{vector(rows[0].averagedValues)}</b></span></div>
    <fieldset><legend>Shared candidate: move away from the minimizing mean</legend><p>These controls edit offsets from the current means, so zero restores mean pooling even after an original head changes.</p><div className="gqa-controls">{[0, 1].flatMap(kind => [0, 1].map(slot => <NeuralNumber key={`${kind}-${slot}`} label={`${kind === 0 ? 'Shared key' : 'Shared value'} ${slot} offset from mean`} value={offsets[kind][slot]} min={-6} max={6} range={false} onChange={v => update(setOffsets, kind, slot, v)} />))}</div><button onClick={() => setOffsets([[0, 0], [0, 0]])}>Use exact means</button></fieldset>
    <NeuralTable caption="Squared distance from both original heads (keys and values are separate coordinates)" headers={['Space', 'Minimum at mean', 'Current candidate', 'Excess']} rows={[['Keys', rows[0].minimumKeyDistance, rows[0].keyDistance], ['Values', rows[0].minimumValueDistance, rows[0].valueDistance]].map(([name, minimum, actual]) => [name, f(minimum, 6), f(actual, 6), f(actual - minimum, 6)])} />
    <p>Current shared keys {vector(rows[0].sharedKeys)}; shared values {vector(rows[0].sharedValues)}. This measures the constructed key/value vectors, not a trained projection matrix or forecast error.</p>
    <NeuralTable caption="Parameter averaging is before softmax and value mixing" headers={['Head', 'Old weights', 'New weights', 'Old output', 'New output', 'Output change']} rows={rows.map((r, h) => [h, vector(r.originalWeights), vector(r.convertedWeights), f(r.original, 6), f(r.converted, 6), f(r.outputDelta, 6)])} />
    <div className="gqa-actions"><button onClick={() => {
        setOffsets([[0, 0], [0, 0]]);
        setKeys(k => [k[0], [...k[0]]]);
        setValues(v => [v[0], [...v[0]]]);
      }}>Tie K/V heads: preservation control</button><button onClick={() => {
        setOffsets([[0, 0], [0, 0]]);
        setQuery([1, 2]);
        setKeys([[2, 0], [0, 2]]);
        setValues([[1, 3], [5, -1]]);
      }}>Reset conversion</button></div>
    <p>Mean pooling minimizes squared distance to the original parameters. Softmax acts on the merged scores afterward, so it need not preserve either prediction. Already tied K/V heads give exact preservation.</p>
  </NeuralLab>;
}
export function GqaTrainingHistory() {
  const series = Object.entries(studyHistory).reverse().map(([heads, run], i) => ({
    label: `${heads} KV heads; selected update ${run.selected}`,
    color: palette[i],
    values: run.history.map(([step, mse]) => [step, Math.log10(mse)])
  }));
  const values = series.flatMap(row => row.values.map(pair => pair[1]));
  return <figure className="gqa-figure" tabIndex={0}><figcaption>Measured continuation: recovery after conversion</figcaption>
    <NeuralPlot title="Validation loss over the declared 45 additional updates" xLabel="additional update" yLabel="log10 transformed-coordinate MSE" xDomain={[0, 45]} yDomain={[Math.floor(Math.min(...values)), Math.ceil(Math.max(...values))]} series={series} />
    <p>Each curve begins at its zero-update state. All branches continue from the same MHA parent, with a fresh optimizer. A value of −3 on this axis means MSE 0.001; these are measured losses, not cache-derived quality estimates.</p>
    <section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Exact recorded validation losses and selected steps</h4><NeuralTable caption="Recorded transformed-coordinate validation MSE" headers={['Extra update', 'MHA (4 KV)', 'GQA (2 KV)', 'MQA (1 KV)']} rows={studyHistory['4'].history.map(([step], i) => [step, ...['4', '2', '1'].map(heads => `${f(studyHistory[heads].history[i][1], 8)}${studyHistory[heads].selected === step ? ' · selected' : ''}`)])} /></section>
  </figure>;
}
export function GqaGradientFigure() {
  const [a, setA] = useState(.2),
    [b, setB] = useState(.7),
    [u, setU] = useState(3),
    [v, setV] = useState(-1);
  return <NeuralLab title="A shared parameter adds the contributions of its uses" id="gqa-gradient"><div className="gqa-controls">{[[a, setA, 'Reader 0 attention', 0, 1], [b, setB, 'Reader 1 attention', 0, 1], [u, setU, 'Reader 0 upstream derivative', -5, 5], [v, setV, 'Reader 1 upstream derivative', -5, 5]].map(([value, setter, label, min, max]) => <NeuralNumber key={label} label={label} value={value} min={min} max={max} range={false} onChange={setter} />)}</div><div className="gqa-gradient"><span>Reader 0<br />{f(a)} × {f(u)} = {f(a * u)}</span><b>↘</b><span>shared value<br />gradient = <strong>{f(a * u + b * v)}</strong></span><b>↗</b><span>Reader 1<br />{f(b)} × {f(v)} = {f(b * v)}</span></div><p>The arrows meet at the same parameter. Opposite signs can cancel; no extra average is inserted. Any loss averaging is already in the upstream derivatives.</p><button onClick={() => {
      setA(.2);
      setB(.7);
      setU(3);
      setV(-1);
    }}>Reset gradient</button></NeuralLab>;
}
export function GqaForecastLab() {
  const host = useRef(null),
    [visible, setVisible] = useState(false);
  const [branch, setBranch] = useState('2'),
    [payload, setPayload] = useState(null),
    [error, setError] = useState(''),
    [attempt, setAttempt] = useState(0);
  const [edits, setEdits] = useState({}),
    [boundary, setBoundary] = useState(32),
    [frame, setFrame] = useState(23),
    [head, setHead] = useState(0),
    [steps, setSteps] = useState(5),
    [shift, setShift] = useState(0);
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
    setPayload(null);
    setError('');
    fetch(`${assetBase}runtime-${branch}.json`, {
      signal: controller.signal
    }).then(r => {
      if (!r.ok) throw new Error('Model download failed.');
      return r.json();
    }).then(setPayload).catch(e => {
      if (e.name !== 'AbortError') setError('The saved model could not be loaded. Check your connection, then retry.');
    });
    return () => controller.abort();
  }, [branch, attempt, visible]);
  const result = useMemo(() => {
    if (!payload) return null;
    const points = payload.points.slice(0, boundary).map((point, i) => edits[i] ?? point);
    const live = groupedRollout(payload.weights, points, Number(branch), steps, shift);
    const baseline = groupedForecast(payload.weights, payload.points.slice(0, boundary), Number(branch), shift);
    let cache = null;
    const incremental = [];
    for (const point of points) {
      const step = groupedForecast(payload.weights, [point], Number(branch), shift, cache);
      cache = step.cache;
      incremental.push(step.predictions[0]);
    }
    const error = Math.max(...incremental.flatMap((p, i) => p.map((x, d) => Math.abs(x - live.predictions[i][d]))));
    return {
      ...live,
      points,
      baseline,
      error
    };
  }, [payload, edits, boundary, branch, steps, shift]);
  const reset = () => {
    setBranch('2');
    setEdits({});
    setBoundary(32);
    setFrame(23);
    setHead(0);
    setSteps(5);
    setShift(0);
  };
  const currentFrame = Math.min(frame, boundary - 1);
  const point = result?.points[currentFrame];
  return <div ref={host}><NeuralLab title="Forecast an actual recorded movement from its observed prefix" id="gqa-forecast">
    <p>UCI Libras source row 77. The saved selected model runs on the current observed points. Generated points feed back into later rollout steps; recorded future points are evaluation references only.</p>
    <NeuralSelect label="Selected trained branch" value={branch} onChange={setBranch} options={[[4, 'MHA: 4 KV heads'], [2, 'GQA: 2 KV heads'], [1, 'MQA: 1 KV head']]} />
    {error ? <p role="alert">{error} <button onClick={() => setAttempt(v => v + 1)}>Retry model download</button></p> : !result ? <p role="status">Loading the selected small model…</p> : <>
      <div className="gqa-controls"><NeuralNumber label="Observed prefix length" value={boundary} min={2} max={40} step={1} integer onChange={setBoundary} /><NeuralNumber label="Observed frame to edit" value={currentFrame} min={0} max={boundary - 1} step={1} integer onChange={setFrame} /><NeuralNumber label="Generated steps" value={steps} min={1} max={5} step={1} integer onChange={setSteps} /><NeuralNumber label="Common logical position shift" value={shift} min={0} max={128} step={1} integer onChange={setShift} />
      {[0, 1].map(axis => <NeuralNumber key={axis} label={`Observed ${axis ? 'y' : 'x'} at frame ${currentFrame}`} value={point[axis]} min={0} max={1} range={false} onChange={v => setEdits(previous => ({
            ...previous,
            [currentFrame]: point.map((x, d) => d === axis ? v : x)
          }))} />)}</div>
      <div className="gqa-actions"><button onClick={() => setEdits(previous => ({
            ...previous,
            [currentFrame]: [1 - point[0], point[1]]
          }))}>Reflect selected x</button><button onClick={reset}>Reset forecast</button></div>
      <div className="gqa-columns"><NeuralPlot title="Observed and generated trajectory" xLabel="x coordinate" yLabel="y coordinate" xDomain={[Math.min(0, ...result.rollout.map(p => p[0])), Math.max(1, ...result.rollout.map(p => p[0]))]} yDomain={[Math.min(0, ...result.rollout.map(p => p[1])), Math.max(1, ...result.rollout.map(p => p[1]))]} series={[{
            label: 'Observed prefix',
            color: palette[0],
            values: result.points
          }, {
            label: 'Generated continuation',
            color: palette[1],
            dashed: true,
            values: [result.points.at(-1), ...result.rollout]
          }, {
            label: 'Recorded future (not input)',
            color: palette[2],
            dashed: true,
            values: [payload.points[boundary - 1], ...payload.points.slice(boundary, boundary + steps)]
          }]} />
      <NeuralPlot title="Observation boundary along time" xLabel="zero-based frame" yLabel="x coordinate" xDomain={[0, boundary + steps - 1]} yDomain={[Math.min(0, ...result.rollout.map(p => p[0])), Math.max(1, ...result.rollout.map(p => p[0]))]} series={[{
            label: `Observed: frames 0–${boundary - 1}`,
            color: palette[0],
            values: result.points.map((p, i) => [i, p[0]])
          }, {
            label: `Generated: frames ${boundary}–${boundary + steps - 1}`,
            color: palette[1],
            dashed: true,
            values: result.rollout.map((p, i) => [i + boundary, p[0]])
          }]} /></div>
      <NeuralTable caption="Current input-bound forecast and compact-cache check" headers={['Quantity', 'Value']} rows={[["Next predicted point", vector(result.predictions.at(-1))], ['Original-prefix prediction at the same boundary', vector(result.baseline.predictions.at(-1))], ['Recorded next point, excluded from input', vector(payload.points[boundary])], ['K and V shape, each', `[1, ${branch}, ${boundary}, 6]`], ['Float32 K/V payload', `${2 * Number(branch) * boundary * 6 * 4} bytes`], ['Full versus incremental maximum difference', result.error.toExponential(3)], ['Earlier predictions before first edited frame', Object.keys(edits).length ? `max difference ${Math.max(0, ...result.predictions.slice(0, Math.min(...Object.keys(edits).map(Number))).flatMap((p, i) => p.map((v, d) => Math.abs(v - result.baseline.predictions[i][d])))).toExponential(3)}` : 'No edited input']]} />
      <NeuralSelect label="Inspect final query head" value={head} onChange={v => setHead(Number(v))} options={[0, 1, 2, 3].map(h => [h, `Head ${h} → KV ${Math.floor(h / (4 / Number(branch)))}`])} />
      <NeuralPlot title={`Actual attention of query head ${head}`} xLabel="observed frame" yLabel="attention weight" xDomain={[0, boundary - 1]} yDomain={[0, 1]} series={[{
          label: `Head ${head} weights`,
          color: palette[0],
          values: result.attention.at(-1)[head].map((w, i) => [i, w])
        }]} />
      <p>Head output: <code>{vector(result.heads.at(-1)[head])}</code>. These attention weights describe this head's read, rather than a causal attribution of the whole forecast.</p>
      <section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Inspect every observed point, head weight and generated point</h4><NeuralTable caption="Current observed inputs and final-query weights" headers={['Frame', 'Observed point', 'Weight']} rows={result.points.map((p, i) => [i, vector(p), f(result.attention.at(-1)[head][i], 8)])} /><NeuralTable caption="Generated continuation, with no true future fed back" headers={['Frame', 'Generated point']} rows={result.rollout.map((p, i) => [boundary + i, vector(p)])} /></section>
    </>}
  </NeuralLab></div>;
}
export function GqaProgram({
  file = 'author-calculations.py'
}) {
  return <>
    <RemoteCodeBlock source={assetBase + file} language="python" filename={file} title={"Read the complete " + (file) + " program"} />
  </>;
}
