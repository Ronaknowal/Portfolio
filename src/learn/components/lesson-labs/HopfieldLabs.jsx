import { useMemo, useState } from 'react';
import { NeuralLab, NeuralTable } from './NeuralLessonElements.jsx';
import { NeuralNumber } from './NeuralNumberControl.jsx';
import { Bits, VectorEditor, MemoryPlane, MemoryPlot, Bars, MemoryFigure, Arrow, Node, Values, f, vec } from './HopfieldPrimitives.jsx';
import { SignedMemoryFigure, CobwebFigure } from './HopfieldFigures.jsx';
import { binaryRecall, continuousTrace, energyContours, distance, associate } from '../../data/hopfield-memory-models.js';
const binaryDefault = () => ({
  patterns: [[1, 1, -1, -1]],
  cue: [-1, 1, -1, -1],
  order: [0, 1, 2, 3]
});
export function BinaryMemoryLab() {
  const [view, setView] = useState(binaryDefault),
    [step, setStep] = useState(1),
    [pinned, setPinned] = useState(null),
    result = useMemo(() => binaryRecall(view.patterns, view.cue, view.order), [view]),
    index = Math.min(step, result.trace.length - 1),
    current = result.trace[index],
    previous = result.trace[Math.max(0, index - 1)],
    edit = patch => {
      setView(v => ({
        ...v,
        ...patch
      }));
      setStep(1);
    };
  const swap = (i, value) => {
    const other = view.order.indexOf(value),
      order = [...view.order];
    [order[i], order[other]] = [order[other], order[i]];
    edit({
      order
    });
  };
  return <NeuralLab id="hopfield-binary" title="Repair a cue one coordinate at a time"><p>Flip any cue or stored bit. The weight matrix and complete bounded recall update immediately. Changing a visit slot swaps it with the existing slot, so every coordinate is still visited once per sweep.</p>
 <Bits label="Cue" value={view.cue} onChange={cue => edit({
      cue
    })} /><div className="hm-lab-list">{view.patterns.map((row, i) => <Bits key={i} label={`Stored pattern ${i + 1}`} value={row} onChange={next => edit({
        patterns: view.patterns.map((old, j) => i === j ? next : old)
      })} />)}</div><div className="hm-controls"><button disabled={view.patterns.length === 3} onClick={() => edit({
        patterns: [...view.patterns, [1, -1, 1, -1]]
      })}>Add a pattern</button><button disabled={view.patterns.length === 1} onClick={() => edit({
        patterns: view.patterns.slice(0, -1)
      })}>Remove last pattern</button></div>
 <div className="neural-controls">{view.order.map((v, i) => <NeuralNumber key={i} label={`Visit slot ${i + 1}: coordinate`} value={v + 1} min={1} max={4} integer range={false} onChange={n => swap(i, n - 1)} />)}</div>
 <p className="hm-result" aria-live="polite">Final {vec(result.final)}: {result.category}. {result.converged ? 'A complete sweep made no change.' : 'Eight-sweep limit reached.'} Final energy {f(result.trace.at(-1).energy)}; initial {f(result.trace[0].energy)}.</p>
 <div className="hm-controls"><button onClick={() => edit({
        cue: [1, -1, -1, -1]
      })}>Worked cue</button><button onClick={() => edit({
        cue: [-1, -1, -1, -1]
      })}>Ambiguous cue</button><button onClick={() => edit({
        order: [...view.order].reverse()
      })}>Reverse visit order</button><button onClick={() => edit({
        cue: [...view.patterns[0]]
      })}>Stored-cue null</button><button onClick={() => setPinned({
        view: structuredClone(view),
        result
      })}>Pin this bank and cue</button><button onClick={() => {
        setView(binaryDefault());
        setStep(1);
        setPinned(null);
      }}>Reset binary investigation</button></div>
 {pinned && <div className="hm-result hm-pinned"><p>Pinned cue {vec(pinned.view.cue)}, visit order {vec(pinned.view.order.map(x => x + 1))}, bank {pinned.view.patterns.map(vec).join('; ')}.</p><p>Pinned final {vec(pinned.result.final)}, E={f(pinned.result.trace.at(-1).energy)}. Current final energy difference {f(result.trace.at(-1).energy - pinned.result.trace.at(-1).energy)}.</p></div>}
 <NeuralNumber label="Inspect coordinate update (0 is initial)" value={index} min={0} max={result.trace.length - 1} integer onChange={setStep} /><div className="hm-controls"><button disabled={index === 0} onClick={() => setStep(index - 1)}>Back one update</button><button disabled={index === result.trace.length - 1} onClick={() => setStep(index + 1)}>Next update</button></div><p>Displayed state {vec(current.state)}. {index > 0 ? `Coordinate ${current.coordinate + 1}: ${current.before} → ${current.state[current.coordinate]}, field ${f(current.field)}; energy change ${f(current.energy - previous.energy)}.` : 'Initial state before any coordinate visit.'} A zero field keeps the existing sign.</p>
 <SignedMemoryFigure state={previous.state} weights={result.w} active={current.coordinate ?? view.order[0]} /><MemoryPlot title="Energy at every actual coordinate update" xLabel="coordinate update" yLabel="energy" series={[{
      label: 'E(s)',
      color: '#e6bb60',
      points: result.trace.map((t, i) => [i, t.energy])
    }]} /><section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Complete computed trace</h4><NeuralTable caption="No skipped flat updates" headers={['Update', 'Coordinate', 'State', 'Field', 'Energy']} rows={result.trace.map((t, i) => [i, t.coordinate === null ? 'initial' : t.coordinate + 1, vec(t.state), t.field === null ? '—' : f(t.field), f(t.energy)])} /></section>
 <MemoryFigure title="A different update regime: simultaneous updates can cycle" width={560} height={125}><Node x={15} y={25} width={140} lines={['[+1,−1]', 'E=+1']} /><Arrow x1={155} y1={47} x2={220} y2={47} /><Node x={220} y={25} width={140} lines={['[−1,+1]', 'E=+1']} /><Arrow x1={360} y1={47} x2={420} y2={47} /><Node x={420} y={25} width={125} lines={['[+1,−1]', 'E=+1']} /><text x="15" y="110">Fixed W=[[0,1],[1,0]]; update both signs from the old state.</text></MemoryFigure></NeuralLab>;
}
const continuousDefault = () => ({
  memories: [{
    id: 1,
    value: [1, 0]
  }, {
    id: 2,
    value: [-1, 0]
  }],
  cue: [-.35, .6],
  beta: 2,
  nextId: 3
});
export function ContinuousMemoryLab() {
  const [view, setView] = useState(continuousDefault),
    [step, setStep] = useState(0),
    [pinned, setPinned] = useState(null),
    memories = useMemo(() => view.memories.map(r => r.value), [view.memories]),
    trace = useMemo(() => continuousTrace(memories, view.cue, view.beta), [memories, view.cue, view.beta]),
    contours = useMemo(() => energyContours(memories, view.beta), [memories, view.beta]),
    current = trace[step],
    last = trace[12],
    distances = memories.map(m => distance(m, last.q)),
    nearest = distances.indexOf(Math.min(...distances)),
    edit = patch => {
      setView(v => ({
        ...v,
        ...patch
      }));
      setStep(0);
    },
    special = memories.length === 2 && memories.some(m => m[0] === 1 && m[1] === 0) && memories.some(m => m[0] === -1 && m[1] === 0);
  return <NeuralLab id="hopfield-continuous" title="Change a memory landscape and follow twelve reads"><p>Move the cue or any memory with the coordinate fields. The next read and all twelve updates are visible immediately; selecting a step inspects the same computed route.</p><VectorEditor label="Continuous cue" value={view.cue} onChange={cue => edit({
      cue
    })} /><div className="hm-two">{view.memories.map((r, i) => <VectorEditor key={r.id} label={`Memory ID ${r.id}`} value={r.value} onChange={value => edit({
        memories: view.memories.map((old, j) => i === j ? {
          ...old,
          value
        } : old)
      })} />)}</div><div className="hm-controls"><button disabled={memories.length === 6} onClick={() => edit({
        memories: [...view.memories, {
          id: view.nextId,
          value: [0, 1]
        }],
        nextId: view.nextId + 1
      })}>Add memory</button><button disabled={memories.length === 1} onClick={() => edit({
        memories: view.memories.slice(0, -1)
      })}>Remove last memory</button><button onClick={() => edit({
        memories: [view.memories[0]]
      })}>One-memory case</button><button onClick={() => edit({
        memories: [view.memories[0], {
          id: view.nextId,
          value: [...view.memories[0].value]
        }],
        nextId: view.nextId + 1
      })}>Duplicate memory null</button><button onClick={() => edit({
        cue: [0, 0]
      })}>Symmetric zero cue</button></div><NeuralNumber label="Continuous inverse temperature β" value={view.beta} min={.1} max={8} onChange={beta => edit({
      beta
    })} /><p className="hm-result" aria-live="polite">First read {vec(trace[1].q)}; after12 updates {vec(last.q)}. Nearest is memory ID {view.memories[nearest].id}, distance {f(distances[nearest], 6)} ({distances[nearest] <= .05 ? 'within the display threshold0.05' : 'a mixture / separated from every memory at threshold0.05'}). Last step norm {f(distance(trace[11].q, last.q), 8)}; this finite endpoint is not a convergence proof.</p>
 <div className="hm-controls"><button onClick={() => setPinned({
        view: structuredClone(view),
        trace
      })}>Pin current landscape and cue</button><button onClick={() => {
        setView(continuousDefault());
        setStep(0);
        setPinned(null);
      }}>Reset continuous investigation</button></div>{pinned && <div className="hm-result hm-pinned">Pinned β={pinned.view.beta}, cue {vec(pinned.view.cue)}, memories {pinned.view.memories.map(r => `ID${r.id}:${vec(r.value)}`).join('; ')}. Pinned endpoint {vec(pinned.trace[12].q)}; current endpoint displacement {f(distance(last.q, pinned.trace[12].q), 6)}.</div>}
 <NeuralNumber label="Selected continuous update" value={step} min={0} max={12} integer onChange={setStep} /><div className="hm-two"><MemoryPlane title={`Energy landscape: inspect update ${step}`} memories={memories} q={view.cue} read={current.read} path={trace.map(t => t.q)} selected={step} contours={contours} /><Bars title={`Weights produced from state ${step}`} labels={view.memories.map(r => `Memory ID${r.id}`)} values={current.weights} /></div><p>Geometric M1…Mn labels follow the current row order; stable memory IDs above retain their identity after edits. Eight contours are calculated on a61×41 grid over [−2,2]²; step selection reuses this unchanged grid.</p><NeuralTable caption={`Selected state ${step}: dot → scaled score → weight`} headers={['Memory ID', 'Coordinates', 'Dot', 'β×dot', 'Weight']} rows={view.memories.map((r, i) => [r.id, vec(r.value), f(current.scores[i]), f(current.logits[i]), f(current.weights[i], 8)])} /><Values caption="Energy and local sensitivity at the selected state" rows={[["Selected state", vec(current.q)], ["Next read", vec(current.read)], ["E(current) → E(next)", `${f(current.energy, 8)} → ${f(current.nextEnergy, 8)}`], ["Actual change / guaranteed upper change", `${f(current.nextEnergy - current.energy, 8)} / ${f(-.5 * current.step ** 2, 8)}`], ["Local Jacobian rows", current.jacobian.map(vec).join('; ')], ["Local Jacobian norm", f(current.sensitivity, 8)]]} /><MemoryPlot title="Actual continuous objective along all twelve updates" xLabel="update" yLabel="energy" series={[{
      label: 'E',
      color: '#e6bb60',
      points: trace.map((t, i) => [i, t.energy])
    }]} />{special && <section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Compare the canonical opposite-pair cobwebs</h4><p>These two labeled comparison plots keep their own β=.5 and β=2 with starting horizontal coordinate .2; the current editable route is shown above.</p><CobwebFigure /></section>}</NeuralLab>;
}
const associationDefault = () => ({
  q: [-.3, .4],
  beta: 2,
  records: [{
    id: 1,
    key: [1, 0],
    value: [1, 0]
  }, {
    id: 2,
    key: [0, 1],
    value: [0, 1]
  }, {
    id: 3,
    key: [-1, 0],
    value: [0, 1]
  }],
  nextId: 4,
  mode: 'classes'
});
export function AssociationLab() {
  const [view, setView] = useState(associationDefault),
    [pinned, setPinned] = useState(null),
    r = associate(view.records, view.q, view.beta),
    edit = patch => setView(v => ({
      ...v,
      ...patch
    })),
    keyRead = r.weights.reduce((s, p, i) => s.map((x, j) => x + p * view.records[i].key[j]), [0, 0]);
  return <NeuralLab id="hopfield-association" title="Change where you look and what each record returns"><p>Keys control scores. Values supply the payload. Editing a value changes its contribution while leaving every key score and weight unchanged.</p><VectorEditor label="Association query" value={view.q} onChange={q => edit({
      q
    })} /><NeuralNumber label="Association β" value={view.beta} min={.1} max={8} onChange={beta => edit({
      beta
    })} /><div className="hm-controls"><button onClick={() => edit({
        mode: 'classes',
        records: view.records.map((row, i) => ({
          ...row,
          value: i === 0 ? [1, 0] : [0, 1]
        }))
      })}>One-hot class values</button><button onClick={() => edit({
        mode: 'payload'
      })}>Free payload values</button><button onClick={() => edit({
        mode: 'payload',
        records: view.records.map(row => ({
          ...row,
          value: [4, -2]
        }))
      })}>Equal-payload null</button><button onClick={() => edit({
        records: [...view.records].reverse()
      })}>Reverse complete records</button><button disabled={view.records.length >= 6} onClick={() => edit({
        records: [...view.records, {
          id: view.nextId,
          key: [.4, .4],
          value: view.mode === 'classes' ? [0, 1] : [1, -1]
        }],
        nextId: view.nextId + 1
      })}>Add record</button><button disabled={view.records.length <= 2} onClick={() => edit({
        records: view.records.slice(0, -1)
      })}>Remove last record</button></div>
 {view.records.map((record, i) => <div className="hm-two" key={record.id}><VectorEditor label={`Record ${record.id} key`} value={record.key} onChange={key => edit({
        records: view.records.map((row, j) => i === j ? {
          ...row,
          key
        } : row)
      })} />{view.mode === 'payload' ? <VectorEditor label={`Record ${record.id} value`} value={record.value} min={-5} max={5} onChange={value => edit({
        records: view.records.map((row, j) => i === j ? {
          ...row,
          value
        } : row)
      })} /> : <div className="hm-controls"><button onClick={() => edit({
          records: view.records.map((row, j) => i === j ? {
            ...row,
            value: row.value[0] === 1 ? [0, 1] : [1, 0]
          } : row)
        })}>Record {record.id}: class {record.value[0] === 1 ? 1 : 2} (toggle)</button></div>}</div>)}
 <p className="hm-result" aria-live="polite">Current returned {view.mode === 'classes' ? 'class masses' : 'payload'} {vec(r.output)}. The key-space average is {vec(keyRead)}. {view.mode === 'classes' ? 'Class masses sum to1; several smaller records can outweigh one strong competing record.' : 'A free payload need not be a probability or a key-space state.'}</p><div className="hm-two"><Bars title="Which keys receive weight?" labels={view.records.map(row => `Record${row.id}`)} values={r.weights} /><MemoryFigure title="Every record writes into both output coordinates" width={560} height={view.records.length * 58 + 70}>{view.records.map((row, i) => <g key={row.id}><text x="5" y={30 + i * 58}>ID{row.id}: p={f(r.weights[i], 4)}</text>{[0, 1].map(j => {
            const v = r.contributions[i][j],
              zero = 300 + j * 135;
            return <g key={j}><line x1={zero} y1={8 + i * 58} x2={zero} y2={37 + i * 58} stroke="#aaa" /><rect x={zero + Math.min(0, v * 20)} y={14 + i * 58} width={Math.abs(v * 20)} height="16" fill={v < 0 ? '#e9979f' : '#e6bb60'} /><text x={zero - 35} y={49 + i * 58} className="hm-small">v{j + 1}×p={f(v, 4)}</text></g>;
          })}</g>)}<text x="5" y={view.records.length * 58 + 42}>Sum columns → {vec(r.output)} (signed bars:20 pixels per unit)</text></MemoryFigure></div>
 <MemoryPlane title="Current keys, query and the key-space weighted average" memories={view.records.map(row=>row.key)} q={view.q} read={keyRead} extent={Math.max(1,...view.records.flatMap(row=>row.key.map(Math.abs)),...view.q.map(Math.abs))}/><p>Plane labels in current order: {view.records.map((row,i)=>`M${i+1}=record ${row.id}`).join("; ")}. The diamond averages keys. Arbitrary returned payloads are shown separately above.</p><NeuralTable caption="Complete current association calculation" headers={['ID', 'Key', 'Score', 'Weight', 'Value', 'Weighted contribution']} rows={view.records.map((row, i) => [row.id, vec(row.key), f(r.scores[i], 6), f(r.weights[i], 8), vec(row.value), vec(r.contributions[i])])} /><div className="hm-controls"><button onClick={() => setPinned({
        view: structuredClone(view),
        result: r
      })}>Pin this association</button><button onClick={() => {
        setView(associationDefault());
        setPinned(null);
      }}>Reset association investigation</button></div>{pinned && <div className="hm-result hm-pinned"><p>Pinned q={vec(pinned.view.q)}, β={pinned.view.beta}; returned {vec(pinned.result.output)}. Current output difference {vec(r.output.map((x, i) => x - pinned.result.output[i]))}.</p><details><summary>Pinned record identities and inputs</summary><NeuralTable caption="Complete pinned association" headers={['ID', 'Key', 'Value', 'Weight']} rows={pinned.view.records.map((row, i) => [row.id, vec(row.key), vec(row.value), f(pinned.result.weights[i], 8)])} /></details></div>}<p>Complete-record permutation preserves this read up to floating-point roundoff. With all values [4,−2], query edits can change every weight while the returned payload remains [4,−2]. No fixed-bank energy descent is claimed for these distinct values.</p></NeuralLab>;
}
