import { useMemo, useState } from 'react';
import { NeuralLab, NeuralTable } from './NeuralLessonElements.jsx';
import { NeuralNumber } from './NeuralNumberControl.jsx';
import { XSignedWrites, XVector, XValues, XFigure, XArrow, XNode, XLedger, XMatrix, XAddressPlane, XCausalGrid, XChunkJoin, XPlot, f, vec } from './XlstmPrimitives.jsx';
import { scalarScan, matrixScan, denseMatrixRead, chunkMatrixRead } from '../../data/xlstm-memory-models.js';
import { workedScalar, workedMatrix, chunkRows, MatrixFloorFigure } from './XlstmFigures.jsx';
import { xlstmExamples as examples } from '../../data/xlstm-example-inputs.js';
const scalarDefault = () => examples.scalar.values.map((value, i) => ({
  value,
  writeLog: Math.log(examples.scalar.writes[i]),
  retention: examples.scalar.retention[i]
}));
export function ScalarLedgerLab() {
  const [rows, setRows] = useState(scalarDefault),
    [output, setOutput] = useState(.6),
    [offset, setOffset] = useState(0),
    [representation, setRepresentation] = useState('stable'),
    [step, setStep] = useState(3),
    [pinned, setPinned] = useState(null),
    stable = useMemo(() => scalarScan(rows, output, true, offset), [rows, output, offset]),
    raw = useMemo(() => offset === 0 ? scalarScan(rows, output, false) : null, [rows, output, offset]),
    index = Math.min(step, rows.length - 1),
    showRaw = representation === 'raw' && raw,
    trace = showRaw ? raw : stable,
    current = trace[index],
    edit = (i, key, value) => setRows(old => old.map((row, j) => j === i ? {
      ...row,
      [key]: value
    } : row));
  return <NeuralLab id="xlstm-scalar" title="Choose the evidence retained by a scalar memory"><p>Each row contributes a signed candidate and a positive write weight. Edit its log-weight or retention and follow the actual surviving mass. The normalized estimate stays between the written candidates; the output gate separately controls exposure.</p>{rows.map((r, i) => <div className="xl-record" key={i}><p>Observation {i + 1}; raw write weight exp({f(r.writeLog, 6)})={f(Math.exp(r.writeLog), 6)} before any common scale shift.</p><div className="neural-controls"><NeuralNumber label={`Observation ${i + 1} candidate`} value={r.value} min={-1} max={1} onChange={n => edit(i, 'value', n)} /><NeuralNumber label={`Observation ${i + 1} write log-weight`} value={r.writeLog} min={-12} max={12} onChange={n => edit(i, 'writeLog', n)} /><NeuralNumber label={`Observation ${i + 1} retention`} value={r.retention} min={.01} max={1} onChange={n => edit(i, 'retention', n)} /></div></div>)}<NeuralNumber label="Scalar output gate" value={output} min={0} max={1} onChange={setOutput} /><p>Retention zero is a limiting reset outside these finite-log controls; the editable interval is0.01…1.</p><p className="xl-result" aria-live="polite">Final estimate {f(stable.at(-1).estimate, 9)}; exposed output {f(stable.at(-1).hidden, 9)}. Stored final c′={f(stable.at(-1).cell, 8)}, n′={f(stable.at(-1).normalizer, 8)}, m={f(stable.at(-1).log_scale, 8)}.</p><div className="xl-controls"><button disabled={rows.length >= 12} onClick={() => setRows(old => [...old, {
        value: .4,
        writeLog: 0,
        retention: .8
      }])}>Add observation</button><button disabled={rows.length <= 2} onClick={() => setRows(old => old.slice(0, -1))}>Remove last</button><button onClick={() => setRows(old => [...old].reverse())}>Reverse observations</button><button onClick={() => edit(rows.length - 1, 'writeLog', Math.log(.75))}>Set last write to0.75</button><button onClick={() => {
        setRows(old => old.map(r => ({
          ...r,
          value: .6
        })));
        setOutput(.8);
      }}>Agreeing-evidence null</button><button onClick={() => setOutput(0)}>Zero-output valve</button><button onClick={() => {
        setRows(structuredClone(workedScalar));
        setOutput(.75);
        setOffset(0);
        setStep(2);
      }}>Solved three-row example</button><button onClick={() => setOffset(old => old === 0 ? 1000 : 0)}>Common log shift: {offset === 0 ? 'add1000' : 'restore0'}</button><button onClick={() => setPinned({
        rows: structuredClone(rows),
        output,
        offset,
        final: stable.at(-1)
      })}>Pin full ledger</button><button onClick={() => {
        setRows(scalarDefault());
        setOutput(.6);
        setOffset(0);
        setRepresentation('stable');
        setStep(3);
        setPinned(null);
      }}>Reset scalar investigation</button></div>{pinned && <div className="xl-result xl-pinned"><p>Pinned gate {pinned.output}, common offset {pinned.offset}, rows {pinned.rows.map(r => `(z=${r.value},a=${f(r.writeLog)},f=${r.retention})`).join('; ')}. Pinned final output {f(pinned.final.hidden, 9)}; current difference {f(stable.at(-1).hidden - pinned.final.hidden, 9)}.</p></div>}<div className="xl-two"><NeuralNumber label="Inspect scalar step" value={index + 1} min={1} max={rows.length} integer onChange={n => setStep(n - 1)} /><label className="xl-select">Intermediate representation<select value={representation} onChange={e => setRepresentation(e.target.value)}><option value="stable">Stored stabilized totals</option><option value="raw">Raw totals</option></select></label></div>{offset !== 0 && <p>Common write-log offset +1000 is active. Raw totals would require an unrepresentable exponential multiplier; the visible ledger uses stabilized values and keeps its log scale explicitly. No raw Infinity is plotted.</p>}<XLedger rows={rows} trace={trace} step={index} output={output} scaled={!showRaw} /><XValues caption="Selected step representation" rows={[["Effective write i′ / retention f′", `${f(stable[index].write, 9)} / ${f(stable[index].retain, 9)}`], ["Stored log scale", f(stable[index].log_scale, 9)], ["Raw scale multiplier", `exp(${f(stable[index].log_scale, 9)})`], ["Displayed totals", `${f(current.cell, 9)} / ${f(current.normalizer, 9)}`]]} /><XPlot title="Exposed output after every actual observation" xLabel="step" yLabel="output h" series={[{
      label: 'stabilized',
      color: '#e6bb60',
      points: stable.map((r, i) => [i + 1, r.hidden])
    }, ...(raw ? [{
      label: 'raw reference',
      color: '#87bdf1',
      dashed: true,
      points: raw.map((r, i) => [i + 1, r.hidden])
    }] : [])]} /></NeuralLab>;
}
const matrixDefault = () => examples.matrix.keys.map((key, i) => ({
  key: [...key],
  value: [...examples.matrix.values[i]],
  query: [...examples.matrix.queries[i]],
  writeLog: Math.log(examples.matrix.write[i]),
  retention: examples.matrix.forget[i]
}));
export function MatrixAddressLab() {
  const [rows, setRows] = useState(matrixDefault),
    [step, setStep] = useState(2),
    [scaled, setScaled] = useState(false),
    [cell, setCell] = useState([0, 0]),
    [pinned, setPinned] = useState(null),
    index = Math.min(step, rows.length - 1),
    raw = useMemo(() => matrixScan(rows, {
      stabilized: false
    }), [rows]),
    stable = useMemo(() => matrixScan(rows), [rows]),
    r = (scaled ? stable : raw)[index],
    row = rows[index],
    edit = (key, value) => setRows(old => old.map((r, i) => i === index ? {
      ...r,
      [key]: value
    } : r));
  return <NeuralLab id="xlstm-matrix" title="Edit a key, a value or a query without confusing their roles"><p>Select a write below and edit its actual coordinates. The complete prefix is recomputed. The result can leave the range of stored values because address alignment is signed.</p><NeuralNumber label="Selected matrix write / read" value={index + 1} min={1} max={rows.length} integer onChange={n => setStep(n - 1)} /><div className="xl-fit-grid"><XVector label={`Write ${index + 1} key`} value={row.key} onChange={x => edit('key', x)} /><XVector label={`Write ${index + 1} value`} value={row.value} onChange={x => edit('value', x)} /><XVector label={`Read ${index + 1} query (already scaled)`} value={row.query} onChange={x => edit('query', x)} /></div><div className="neural-controls"><NeuralNumber label="Selected matrix write log-weight" value={row.writeLog} min={-4} max={4} onChange={x => edit('writeLog', x)} /><NeuralNumber label="Selected matrix retention" value={row.retention} min={.05} max={1} onChange={x => edit('retention', x)} /></div><div className="xl-controls"><button disabled={rows.length >= 8} onClick={() => {
        setRows(old => [...old, {
          key: [1, .5],
          value: [.5, -.5],
          query: [.5, 1],
          writeLog: 0,
          retention: .7
        }]);
        setStep(rows.length);
      }}>Add write</button><button disabled={rows.length <= 2} onClick={() => setRows(old => old.slice(0, -1))}>Remove last write</button><button onClick={() => edit('key', row.key.map(x => -x))}>Reverse selected key</button><button onClick={() => setRows(old => old.map(x => ({
        ...x,
        value: [0, 0]
      })))}>Zero all values</button><button onClick={() => setRows(old => old.map(x => ({
        ...x,
        query: [0, 0]
      })))}>Zero all queries</button><button onClick={() => setScaled(x => !x)}>Show {scaled ? 'raw' : 'stabilized'} representation</button><button onClick={() => setPinned({
        rows: structuredClone(rows),
        final: raw.at(-1)
      })}>Pin complete address history</button><button onClick={() => {
        setRows(structuredClone(workedMatrix));
        setStep(2);
      }}>Solved three-write example</button><button onClick={() => {
        setRows(matrixDefault());
        setStep(2);
        setScaled(false);
        setCell([0, 0]);
        setPinned(null);
      }}>Reset matrix investigation</button></div><p className="xl-result" aria-live="polite">Selected read {vec(r.read)}; numerator {vec(r.numerator)} divided by {f(r.denominator, 9)}. Signed mass {f(r.mass, 9)}; floor {f(r.floor, 9)} is {Math.abs(r.mass) <= r.floor ? 'active' : 'inactive'}. Raw/stabilized read difference {vec(raw[index].read.map((v, j) => v - stable[index].read[j]))}.</p>{pinned && <div className="xl-result xl-pinned"><p>Pinned final read {vec(pinned.final.read)}; current final {vec(raw.at(-1).read)}; difference {vec(raw.at(-1).read.map((x, i) => x - pinned.final.read[i]))}.</p><details><summary>Complete pinned history</summary><NeuralTable caption="Pinned matrix inputs" headers={['Write', 'Key', 'Value', 'Query', 'Log write', 'Retention']} rows={pinned.rows.map((r, i) => [i + 1, vec(r.key), vec(r.value), vec(r.query), f(r.writeLog), r.retention])} /></details></div>}<div className="xl-two"><XAddressPlane keys={rows.slice(0, index + 1).map(x => x.key)} query={row.query} /><XMatrix title={`Stored C${scaled ? '′' : ''} at step ${index + 1}`} values={r.cell} selected={cell} /></div><div className="xl-matrix-selector"><NeuralNumber label="Inspect key row" value={cell[0] + 1} min={1} max={2} integer onChange={n => setCell([n - 1, cell[1]])} /><NeuralNumber label="Inspect value column" value={cell[1] + 1} min={1} max={2} integer onChange={n => setCell([cell[0], n - 1])} /></div><p>Selected new outer-product entry k{cell[0] + 1}×v{cell[1] + 1}={f(row.key[cell[0]] * row.value[cell[1]], 8)}; effective write multiplies it by {f(r.write, 8)}. The stored entry also retains {f(r.retain, 8)} of its previous scaled entry.</p><XMatrix title="Current weighted outer-product write" values={row.key.map(k => row.value.map(v => r.write * k * v))} selected={cell} /><XSignedWrites contributions={r.contributions} denominator={r.denominator} /><NeuralTable caption="Every surviving write contributes signed mass and payload" headers={['Source', 'Surviving weight', 'Key·query', 'Signed coefficient', 'Value contribution']} rows={r.weights.map((w, i) => [i + 1, f(w, 8), f(rows[i].key.reduce((s, v, j) => s + v * row.query[j], 0), 8), f(r.coefficients[i], 8), vec(r.contributions[i])])} /><XPlot title="Actual signed reads may exceed the value range" xLabel="step" yLabel="returned coordinate" series={[0, 1].map(j => ({
      label: `read coordinate ${j + 1}`,
      color: j ? '#87bdf1' : '#e6bb60',
      points: raw.map((r, i) => [i + 1, r.read[j]])
    }))} /><section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">All current input rows and normalizer states</h4><NeuralTable caption="Complete address calculation" headers={['Step', 'Key', 'Value', 'Query', 'n', 'Raw read']} rows={rows.map((r, i) => [i + 1, vec(r.key), vec(r.value), vec(r.query), vec(raw[i].normalizer), vec(raw[i].read)])} /></section><section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Worked one-dimensional active-floor counterexample</h4><MatrixFloorFigure /></section></NeuralLab>;
}
const emptyIncoming = () => ({
  cell: [[0, 0], [0, 0], [0, 0]],
  normalizer: [0, 0, 0]
});
export function CausalChunkLab() {
  const [rows, setRows] = useState(() => structuredClone(chunkRows)),
    [size, setSize] = useState(3),
    [selected, setSelected] = useState(4),
    [incoming, setIncoming] = useState(emptyIncoming),
    [reset, setReset] = useState(false),
    [pinned, setPinned] = useState(null),
    index = Math.min(selected, rows.length - 1),
    row = rows[index],
    reference = useMemo(() => matrixScan(rows, {
      stabilized: false,
      initial: incoming
    }), [rows, incoming]),
    chunk = useMemo(() => chunkMatrixRead(rows, size, incoming, reset), [rows, size, incoming, reset]),
    dense = useMemo(() => denseMatrixRead(rows), [rows]),
    r = chunk.outputs[index],
    errors = chunk.outputs.map((r, i) => Math.max(...r.read.map((x, j) => Math.abs(x - reference[i].read[j])))),
    edit = (key, value) => setRows(old => old.map((r, i) => i === index ? {
      ...r,
      [key]: value
    } : r));
  return <NeuralLab id="xlstm-chunks" title="Move chunk boundaries and keep the full memory"><p>Chunking changes execution grouping. A deliberate boundary reset changes the history. Edit actual query/key/value records; their gates are already supplied, so all three schedules apply to the same declared operator.</p><div className="neural-controls"><NeuralNumber label="Chunk size" value={size} min={1} max={12} integer onChange={setSize} /><NeuralNumber label="Selected causal record" value={index + 1} min={1} max={rows.length} integer onChange={n => setSelected(n - 1)} /></div><div className="xl-fit-grid"><XVector label="Selected chunk query" value={row.query} onChange={x => edit('query', x)} /><XVector label="Selected chunk key" value={row.key} onChange={x => edit('key', x)} /><XVector label="Selected chunk value" value={row.value} onChange={x => edit('value', x)} /></div><div className="neural-controls"><NeuralNumber label="Selected chunk write log-weight" value={row.writeLog} min={-2} max={2} onChange={x => edit('writeLog', x)} /><NeuralNumber label="Selected chunk retention" value={row.retention} min={.05} max={1} onChange={x => edit('retention', x)} /></div><div className="xl-controls"><button disabled={rows.length >= 12} onClick={() => setRows(old => [...old, {
        query: [1, 0, 0],
        key: [.5, .5, 0],
        value: [1, -1],
        writeLog: 0,
        retention: .8
      }])}>Add record</button><button disabled={rows.length <= 2} onClick={() => setRows(old => old.slice(0, -1))}>Remove last record</button><button onClick={() => setReset(x => !x)}>{reset ? 'Restore complete carry' : 'Reset state at every boundary'}</button><button onClick={() => setIncoming(structuredClone(examples.incoming))}>Nonzero incoming-state fixture</button><button onClick={() => setIncoming(emptyIncoming())}>Empty incoming state</button><button onClick={() => setRows(old => old.map((r, i) => i < 4 ? r : {
        ...r,
        value: r.value.map(v => -v)
      }))}>Reverse future values from record5</button><button onClick={() => setRows(old => old.map(r => ({
        ...r,
        value: [0, 0]
      })))}>Zero new values</button><button onClick={() => setPinned({
        rows: structuredClone(rows),
        incoming: structuredClone(incoming),
        size,
        reset,
        outputs: chunk.outputs.map(r => r.read)
      })}>Pin current computation</button><button onClick={() => {
        setRows(structuredClone(chunkRows));
        setSize(3);
        setSelected(4);
        setIncoming(emptyIncoming());
        setReset(false);
        setPinned(null);
      }}>Reset chunk investigation</button></div><p className="xl-result" aria-live="polite">Maximum absolute output difference from the unsplit raw recurrence: {Math.max(...errors).toExponential(5)}. {reset ? 'Boundary resets intentionally discard prior state.' : 'Full incoming state is preserved.'} Current final read {vec(chunk.outputs.at(-1).read)}.</p><details><summary>Edit every incoming state coordinate</summary><div className="xl-fit-grid">{incoming.cell.map((values, k) => <XVector key={k} label={`Incoming C key row${k + 1}`} value={values} onChange={value => setIncoming(old => ({
          ...old,
          cell: old.cell.map((r, i) => i === k ? value : r)
        }))} />)}</div><XVector label="Incoming n" value={incoming.normalizer} onChange={normalizer => setIncoming(old => ({
        ...old,
        normalizer
      }))} /></details><p>With nonzero incoming C, zeroing new values still permits a nonzero read of retained old information. The bounded chunk reference uses moderate raw arithmetic; no extreme-log stabilization claim is made.</p><XFigure title="Current chronological groups and their actual boundaries" width={Math.max(480, rows.length * 50 + 40)} height={165}>{rows.map((_, i) => <g key={i}><rect x={20 + i * 50} y="30" width="40" height="50" fill={i === index ? '#4a3b20' : '#222'} stroke="#777" /><text x={40 + i * 50} y="60" textAnchor="middle">{i + 1}</text>{i > 0 && i % size === 0 && <line x1={15 + i * 50} y1="10" x2={15 + i * 50} y2="125" stroke="#87bdf1" strokeDasharray="5 4" />}{i < rows.length - 1 && <XArrow x1={60 + i * 50} y1={55} x2={70 + i * 50} y2={55} />}</g>)}<text x="15" y="146">Dashed boundaries: {reset ? 'state discarded' : 'C,n carried'}; last group uses its actual length.</text></XFigure><XChunkJoin result={r} /><XCausalGrid rows={rows} coefficients={dense.map(r => r.coefficients)} selected={index} /><p>The dense tile shows input-write coefficients only, with its own row scale. Any entered prehistory contributes through the separate incoming-state rail above.</p><p>{incoming.cell.flat().every(v => v === 0) && incoming.normalizer.every(v => v === 0) ? `Dense versus recurrent maximum read difference with this empty history: ${Math.max(...dense.map((r, i) => Math.max(...r.read.map((v, j) => Math.abs(v - reference[i].read[j]))))).toExponential(5)}.` : "The dense empty-history outputs exclude your entered prehistory; compare them only after selecting Empty incoming state. The chunk/recurrent comparison above includes the same entered history in both routes."}</p><XPlot title="A schedule comparison measures numerical difference, not accuracy" xLabel="record" yLabel="maximum absolute read difference" series={[{
      label: reset ? 'reset contrast' : 'complete carry',
      color: '#e6bb60',
      points: errors.map((v, i) => [i + 1, v])
    }]} /><XMatrix title={`Incoming C at selected chunk start${r.start + 1}`} values={r.incoming.cell} /><p>Incoming normalizer {vec(r.incoming.normalizer)}. For the selected query, incoming numerator {vec(r.incomingNumerator)} and local numerator {vec(r.localNumerator)} combine before normalization.</p>{pinned && <div className="xl-result xl-pinned"><p>Pinned size{pinned.size}, {pinned.reset ? 'reset' : 'carry'}, final {vec(pinned.outputs.at(-1))}. Current final difference {vec(chunk.outputs.at(-1).read.map((v, i) => v - pinned.outputs.at(-1)[i]))}.</p><details><summary>Full pinned inputs and incoming state</summary><p>C={pinned.incoming.cell.map(vec).join('; ')}, n={vec(pinned.incoming.normalizer)}</p><NeuralTable caption="Pinned causal records" headers={['Record', 'Query', 'Key', 'Value', 'Write log', 'Retention']} rows={pinned.rows.map((r, i) => [i + 1, vec(r.query), vec(r.key), vec(r.value), f(r.writeLog), f(r.retention)])} /></details></div>}<section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Every current input and output</h4><NeuralTable caption="All causal records" headers={['Record', 'Query', 'Key', 'Value', 'Chunk read', 'Recurrent with entered history', 'Dense empty-history read']} rows={rows.map((r, i) => [i + 1, vec(r.query), vec(r.key), vec(r.value), vec(chunk.outputs[i].read), vec(reference[i].read), vec(dense[i].read)])} /></section></NeuralLab>;
}
