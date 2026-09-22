import { useState } from 'react';
import { LessonTable } from './LessonElements';
import { clusteringState, cutSweep, filterState, formatSpectral as fmt, spectralNames } from '../../data/spectral-graph-models.js';
import './spectral-graph-labs.css';
const colors = ['#e2b55a', '#8dcce8', '#b9cba0'];
const sixPositions = [[55, 65], [55, 205], [190, 135], [350, 135], [485, 65], [485, 205]];
export function SpectralAnchorFigure() {
  const positions = [[55, 45], [205, 45], [130, 140], [130, 245], [55, 340], [205, 340]];
  const edges = [[0, 1], [0, 2], [1, 2], [2, 3], [3, 4], [3, 5], [4, 5]];
  return <figure className="spectral-inline spectral-anchor" aria-label="Six documents and their first graph signal">
    <svg viewBox="0 0 260 380" role="img" aria-label="ABC have value plus one, DEF minus one; only the weak C to D bridge joins unequal values">
      {edges.map(([i, j]) => <line key={`${i}-${j}`} x1={positions[i][0]} y1={positions[i][1]} x2={positions[j][0]} y2={positions[j][1]} stroke={i === 2 && j === 3 ? '#e2b55a' : '#8a8d86'} strokeWidth={i === 2 && j === 3 ? 1.5 : 3} />)}
      <text x="147" y="197">w = 0.2</text>
      {positions.map(([x, y], i) => <g key={i}><circle cx={x} cy={y} r="28" fill={i < 3 ? colors[0] : colors[1]} /><text x={x} y={y - 3} textAnchor="middle" className="anchor-name">{spectralNames[i]}</text><text x={x} y={y + 17} textAnchor="middle" className="anchor-value">{i < 3 ? '+1' : '−1'}</text></g>)}
    </svg>
    <figcaption>Six node values, seven edges. All triangle edges have weight 1; the thinner C–D bridge has weight 0.2. Only that bridge joins unequal values in this signal. Positions arrange the example and do not encode distance.</figcaption>
  </figure>;
}
export function SpectralNodeGraph({
  graph,
  signal,
  sides,
  label,
  selected = -1
}) {
  const six = graph.degree.length === 6;
  const positions = six ? sixPositions : [[50, 50], [50, 165], [160, 105], [255, 50], [255, 165], [365, 105], [460, 50], [460, 165], [570, 105]];
  function drawing(points, viewBox, className, minWidth) {
    return <svg viewBox={viewBox} className={className} role="img" aria-label={label} style={{
      minWidth
    }}>
      {graph.adjacency.flatMap((row, i) => row.map((weight, j) => j > i && weight > 0 ? <line key={`${i}-${j}`} x1={points[i][0]} y1={points[i][1]} x2={points[j][0]} y2={points[j][1]} stroke={sides && sides[i] !== sides[j] ? '#f1dfbf' : '#7a7f83'} strokeDasharray={sides && sides[i] !== sides[j] ? '6 4' : undefined} strokeWidth={1 + 2 * Math.min(weight, 3)} /> : null))}
      {points.map(([x, y], i) => <g key={i}>
        <circle cx={x} cy={y} r="25" fill={sides ? colors[sides[i] ?? 0] : signal[i] < 0 ? colors[1] : colors[0]} stroke={i === selected ? '#fff0d0' : '#111'} strokeWidth={i === selected ? 4 : 1} />
        <text x={x} y={y + 6} textAnchor="middle" className="spectral-node-name">{spectralNames[i]}</text>
        {signal && <text x={x} y={y + 47} textAnchor="middle">{fmt(signal[i], 3)}</text>}
      </g>)}
    </svg>;
  }
  return <div className="spectral-graph-scroll" tabIndex={0} role="region" aria-label={`${label} drawing`}>
    {drawing(positions, `0 0 ${six ? 540 : 620} ${six ? 270 : 215}`, six ? 'spectral-graph-wide' : '', six ? 400 : 530)}
    {six && drawing([[55, 45], [205, 45], [130, 140], [130, 245], [55, 340], [205, 340]], '0 0 260 410', 'spectral-graph-compact', 0)}
  </div>;
}
export function SpectralCutLab() {
  const [kind, setKind] = useState('bridge');
  const [weight, setWeight] = useState(0.2);
  const [index, setIndex] = useState(0);
  const state = cutSweep(weight, kind);
  const cut = state.candidates[Math.min(index, state.candidates.length - 1)];
  const nodes = new Set(cut?.nodes ?? []);
  function reset() {
    setKind('bridge');
    setWeight(0.2);
    setIndex(0);
  }
  return <section className="lesson-lab spectral-investigation" aria-label="Spectral cut sweep">
    <h3>Move the cut along the normalized coordinates</h3>
    <p>Each node gets yᵢ=v₂ᵢ/√dᵢ. Sort these numbers, place a threshold between unequal neighbors, and inspect the actual crossing edges. Inspect which threshold separates the weakly linked triangles.</p>
    <div className="lesson-controls">
      <label>Graph<select aria-label="Cut graph" value={kind} onChange={event => {
          setKind(event.target.value);
          setIndex(0);
        }}><option value="bridge">Two equal triangles</option><option value="unequal">Unequal degrees and a tail</option></select></label>
      <label>Joining edge: {weight.toFixed(2)}<input aria-label="Cut bridge weight" type="range" min="0" max="2" step="0.05" value={weight} onChange={event => {
          setWeight(Number(event.target.value));
          setIndex(0);
        }} /></label>
    </div>
    <p className="lesson-note">{kind === 'bridge' ? `Edges AB, AC, BC, DE, DF and EF have weight 1; CD has weight ${weight}.` : `Edge weights: AB=3, AC=1, BC=2, CD=${weight}, DE=1, EF=0.5. No other edges.`} Line width is 1+2×weight in drawing units.</p>
    <div className="spectral-order" aria-label="Nodes in ascending normalized coordinate order">{state.order.map(i => <div key={i} className={nodes.has(i) ? 'in-cut' : ''}><strong>{spectralNames[i]}</strong><span>{fmt(state.coordinates[i])}</span><small>{nodes.has(i) ? 'in S' : 'outside S'}</small></div>)}</div>
    <div className="lesson-controls">
      <label>Threshold<select aria-label="Sweep threshold" value={Math.min(index, state.candidates.length - 1)} onChange={event => setIndex(Number(event.target.value))}>{state.candidates.map((candidate, i) => <option key={i} value={i}>{candidate.nodes.map(node => spectralNames[node]).join('')} | remaining · t={fmt(candidate.threshold)}</option>)}</select></label>
      <button type="button" onClick={() => setIndex(state.candidates.indexOf(state.best))}>Choose lowest sweep conductance</button>
      <button type="button" onClick={reset}>Reset cut</button>
    </div>
    <SpectralNodeGraph graph={state} signal={state.coordinates} sides={state.degree.map((_, i) => nodes.has(i) ? 0 : 1)} label="Selected cut with dashed crossing edges" />
    {cut && <div aria-live="polite" className="lesson-results"><strong>S={cut.nodes.map(i => spectralNames[i]).join(', ')}</strong><br />Crossing weight {fmt(cut.cut)}; volumes {fmt(cut.volume)} and {fmt(cut.otherVolume)}.<br />Conductance {fmt(cut.conductance)} = crossing weight / smaller volume.<br />Ncut {fmt(cut.normalizedCut)}; RatioCut {fmt(cut.ratioCut)}.</div>}
    <p className="lesson-note">Dashed edges cross the selected cut; positions only arrange the graph. Near-equal coordinates within 10⁻⁹ stay together. Signs and numerical bases can flip. At zero bridge, the zero eigenspace is not unique; inspect the actual components. This finite sweep evaluates candidates, not every possible partition.</p>
    <LessonTable caption="Compare every retained threshold" headers={['Set S', 'Crossing weight', 'Volume S', 'Conductance']} rows={state.candidates.map(candidate => [candidate.nodes.map(i => spectralNames[i]).join(''), fmt(candidate.cut), fmt(candidate.volume), fmt(candidate.conductance)])} />
  </section>;
}
function EmbeddingPlot({
  state,
  frame,
  selected
}) {
  const px = x => 190 + 150 * x;
  const py = y => 190 - 150 * y;
  const groups = [];
  state.points.forEach((point, i) => {
    const group = groups.find(item => Math.hypot(item.point[1] - point[1], item.point[2] - point[2]) < 0.055);
    if (group) group.nodes.push(i);else groups.push({
      point,
      nodes: [i]
    });
  });
  return <div className="spectral-embedding-scroll" tabIndex={0} role="region" aria-label="Projected spectral row coordinates">
    <svg viewBox="0 0 400 400" role="img" aria-label="Two coordinates of three-dimensional spectral rows; close labels grouped">
      <line x1="30" y1="190" x2="365" y2="190" stroke="#686969" /><line x1="190" y1="25" x2="190" y2="360" stroke="#686969" />
      <circle cx="190" cy="190" r="150" stroke="#464746" strokeDasharray="4 5" fill="none" />
      <text x="355" y="218" textAnchor="end">coordinate 2</text><text x="200" y="25">coordinate 3</text>
      {state.points.map((point, i) => <g key={i}>
        {frame.labels && <line x1={px(point[1])} y1={py(point[2])} x2={px(frame.centroids[frame.labels[i]][1])} y2={py(frame.centroids[frame.labels[i]][2])} stroke={colors[frame.labels[i]]} strokeOpacity="0.45" />}
        <circle cx={px(point[1])} cy={py(point[2])} r={i === selected ? 7 : 4} fill={frame.labels ? colors[frame.labels[i]] : '#e2b55a'} stroke={i === selected ? '#fff' : 'none'} />
      </g>)}
      {groups.map((group, i) => <text className="spectral-group-label" key={i} x={px(group.point[1])} y={py(group.point[2]) + (group.point[2] > 0 ? -17 : 22)} textAnchor="middle">{group.nodes.map(node => spectralNames[node]).join(' ')}</text>)}
      {frame.centroids.map((point, i) => <g key={i}><title>Centroid c{i + 1}: coordinates {fmt(point[1], 3)}, {fmt(point[2], 3)}</title><rect x={px(point[1]) - 7} y={py(point[2]) - 7} width="14" height="14" fill="none" stroke={colors[i]} strokeWidth="2" /></g>)}
    </svg>
    <div className="spectral-centroid-key" aria-label="Square centroid markers and their plotted coordinates">{frame.centroids.map((point, i) => <span key={i}><i style={{ borderColor: colors[i] }} aria-hidden="true" />c{i + 1}: ({fmt(point[1], 3)}, {fmt(point[2], 3)})</span>)}</div>
  </div>;
}
export function SpectralEmbeddingLab() {
  const [weight, setWeight] = useState(0.08);
  const [seed, setSeed] = useState('spread');
  const [step, setStep] = useState(0);
  const [selected, setSelected] = useState(2);
  const state = clusteringState(weight, seed);
  const frame = state.frames[Math.min(step, state.frames.length - 1)];
  function reset() {
    setWeight(0.08);
    setSeed('spread');
    setStep(0);
    setSelected(2);
  }
  return <section className="lesson-lab spectral-investigation" aria-label="Spectral row embedding and clustering">
    <h3>One row becomes one point</h3>
    <p>There are nine nodes and three intended groups. Compute the three lowest normalized modes, then normalize each node’s three-number row. Inspect whether updating a center changes the points or only their grouping.</p>
    <div className="lesson-controls">
      <label>Joining edges: {weight.toFixed(2)}<input aria-label="Embedding bridge weight" type="range" min="0" max="1.5" step="0.02" value={weight} onChange={event => {
          setWeight(Number(event.target.value));
          setStep(0);
        }} /></label>
      <label>Initial centers<select aria-label="Embedding seed centers" value={seed} onChange={event => {
          setSeed(event.target.value);
          setStep(0);
        }}><option value="spread">Spread: A, D, G</option><option value="nearby">Nearby: A, B, D</option></select></label>
      <label>Inspect node<select aria-label="Embedding inspected node" value={selected} onChange={event => setSelected(Number(event.target.value))}>{state.degree.map((_, i) => <option key={i} value={i}>{spectralNames[i]}</option>)}</select></label>
    </div>
    <p className="lesson-note">Every edge inside ABC, DEF and GHI has weight 1. Joining edges C–D and F–G both use the selected weight. Node positions only arrange the graph; colors show the current assignment once one exists.</p>
    <SpectralNodeGraph graph={state} sides={frame.labels ?? state.degree.map(() => 0)} selected={selected} label="Nine-node graph with current cluster memberships" />
    <div className="spectral-row-equation"><strong>Node {spectralNames[selected]}</strong><span>raw row [{state.raw[selected].map(value => fmt(value, 3)).join(', ')}]</span><span>÷ row norm {fmt(state.rowNorms[selected])}</span><span>point [{state.points[selected].map(value => fmt(value, 3)).join(', ')}]</span></div>
    <EmbeddingPlot state={state} frame={frame} selected={selected} />
    <p className="lesson-note">Circles represent node rows; squares are centers. The plot shows coordinates 2 and 3; coordinate 1 is omitted. Near-coincident labels are grouped for readability. Clustering and loss use all three coordinates, so projected distances alone cannot decide membership. The dashed unit circle bounds this projection of unit-length rows.</p>
    <div className="lesson-controls"><button type="button" disabled={step === 0} onClick={() => setStep(step - 1)}>Previous clustering step</button><button type="button" disabled={step >= state.frames.length - 1} onClick={() => setStep(step + 1)}>Next clustering step</button><button type="button" onClick={reset}>Reset embedding</button></div>
    <div className="lesson-results" aria-live="polite"><strong>{frame.phase}</strong> · frame {step + 1}/{state.frames.length}<p>{frame.note}</p>{frame.loss !== null && <p>Sum of squared distances: {fmt(frame.loss, 6)}.</p>}{frame.empty.length > 0 && <p>Empty centers: {frame.empty.map(i => `c${i + 1}`).join(', ')}. This run currently has fewer than three occupied groups; use spread seeds or a different restart.</p>}</div>
    <LessonTable caption="All normalized row coordinates and current membership" headers={['Node', 'Coordinate 1', 'Coordinate 2', 'Coordinate 3', 'Center']} rows={state.points.map((point, i) => [spectralNames[i], ...point.map(value => fmt(value)), frame.labels ? `c${frame.labels[i] + 1}` : 'unassigned'])} />
    <p>Change the nearby initialization at a disconnected graph and look for an empty center. These deterministic presets expose an algorithmic failure mode; they are not a benchmark of initialization quality.</p>
  </section>;
}
export function SpectralFilterLab() {
  const [weight, setWeight] = useState(0.2);
  const [signal, setSignal] = useState('noisy');
  const [filter, setFilter] = useState('heat');
  const [amount, setAmount] = useState(1);
  const state = filterState(weight, signal, filter, amount);
  function reset() {
    setWeight(0.2);
    setSignal('noisy');
    setFilter('heat');
    setAmount(1);
  }
  return <section className="lesson-lab spectral-investigation" aria-label="Graph spectral filtering">
    <h3>Price each mode, then reconstruct the signal</h3>
    <p>Keep the graph fixed while comparing filters. Inspect which coefficients heat suppresses most. Then remove the bridge and ask which part of the signal can survive indefinitely.</p>
    <div className="lesson-controls">
      <label>Signal<select aria-label="Filter signal" value={signal} onChange={event => setSignal(event.target.value)}><option value="noisy">Groups plus local variation</option><option value="groups">Two constant groups</option><option value="spike">One-node impulse</option><option value="constant">Constant two</option></select></label>
      <label>Filter<select aria-label="Spectral filter" value={filter} onChange={event => {
          setFilter(event.target.value);
          setAmount(event.target.value === 'cutoff' ? 2 : 1);
        }}><option value="heat">Heat: exp(−tλ)</option><option value="ridge">Ridge: 1/(1+αλ)</option><option value="cutoff">Keep first m modes</option></select></label>
      <label>{filter === 'cutoff' ? 'Modes kept m' : filter === 'heat' ? 'Time t' : 'Penalty α'}: {amount}<input aria-label="Filter amount" type="range" min={filter === 'cutoff' ? 1 : 0} max={filter === 'cutoff' ? 6 : 10} step={filter === 'cutoff' ? 1 : 0.25} value={amount} onChange={event => setAmount(Number(event.target.value))} /></label>
      <label>Bridge: {weight.toFixed(2)}<input aria-label="Filter bridge weight" type="range" min="0" max="2" step="0.05" value={weight} onChange={event => setWeight(Number(event.target.value))} /></label>
    </div>
    <div className="spectral-mode-bank" aria-label="Coefficient times gain equals retained coefficient">{state.coefficients.map((coefficient, i) => <div key={i}><strong>u{i + 1}</strong><span>λ≈{fmt(state.values[i], 3)}</span><span>{fmt(coefficient, 3)}</span><span>× {fmt(state.gains[i], 3)}</span><div className="spectral-gain-track"><span style={{
            width: `${100 * state.gains[i]}%`
          }} /></div><strong>= {fmt(coefficient * state.gains[i], 3)}</strong></div>)}</div>
    <p className="lesson-note">Each bar’s filled fraction is its gain from 0 to 1, not its coefficient or probability. Mode coefficients are signed; gains scale their magnitude before reconstruction. Values around 10⁻¹⁵ here reflect floating-point roundoff. Null-mode gains use the component count from the graph, not a rounded eigenvalue.</p>
    <div className="spectral-signal-comparison" aria-label="Node signal before and after filtering">{state.signal.map((value, i) => <div key={i}><strong>{spectralNames[i]}</strong><span>{fmt(value, 3)}</span><span aria-hidden="true">↓</span><strong>{fmt(state.output[i], 3)}</strong><div className="spectral-signed-bar"><span style={{
            left: `${50 + Math.min(0, state.output[i]) * 20}%`,
            width: `${Math.abs(state.output[i]) * 20}%`,
            background: state.output[i] < 0 ? colors[1] : colors[0]
          }} /></div></div>)}</div>
    <p className="lesson-note">Top numbers are input values, bottom numbers are output values. Each small axis spans −2.5 to +2.5, with zero in its center. These fixtures are calculated toy signals; “local variation” is not known measurement noise.</p>
    <div className="lesson-results" aria-live="polite">Graph energy: {fmt(state.energyBefore)} → {fmt(state.energyAfter)}.<br />Squared change ‖output−input‖²: {fmt(state.discarded)}.{state.repeatedBoundary && <p><strong>The cutoff splits a repeated eigenspace.</strong> Its result can depend on the eigensolver’s chosen basis. Keep the entire equal-eigenvalue block, or use a scalar gain depending only on λ.</p>}</div>
    <div className="lesson-controls"><button type="button" onClick={reset}>Reset filter</button></div>
    <LessonTable caption="Numerical inspection of mode reconstruction" headers={['Mode', 'Eigenvalue', 'Coefficient', 'Gain', 'Retained']} rows={state.coefficients.map((coefficient, i) => [`u${i + 1}`, fmt(state.values[i], 6), fmt(coefficient, 6), fmt(state.gains[i], 6), fmt(coefficient * state.gains[i], 6)])} />
  </section>;
}
export function SpectralCoordinateFigure() {
  return <figure className="spectral-inline"><div className="spectral-correspondence"><div><strong>Combinatorial L</strong><span>uᵀu = 1</span><span>uᵀ1 = 0</span><span>equal mass per node</span></div><div><strong>Normalized 𝓛</strong><span>vᵀv = 1</span><span>vᵀ√d = 0</span><span>yᵢ = vᵢ / √dᵢ</span></div><div><strong>Same normalized quotient</strong><span>yᵀDy = 1</span><span>yᵀD1 = 0</span><span>degree-weighted mass</span></div></div><figcaption>The normalization changes the geometry of balance. The middle and right columns are the same normalized problem in two coordinate systems; the left column is a different objective.</figcaption></figure>;
}
export function SpectralWalkFigure() {
  return <figure className="spectral-inline"><div className="spectral-walk"><div><strong>Step 0</strong><span>A: 1 · B: 0</span></div><span aria-hidden="true">→</span><div><strong>Step 1</strong><span>A: 0 · B: 1</span></div><span aria-hidden="true">→</span><div><strong>Step 2</strong><span>A: 1 · B: 0</span></div></div><figcaption>A unit-weight two-node graph is connected, but its ordinary walk swaps all probability each step. The alternating mode has transition eigenvalue −1. Giving each node a half chance to stay makes that mode’s eigenvalue zero.</figcaption></figure>;
}
export function SpectralResistanceFigure() {
  return <figure className="spectral-inline"><div className="spectral-circuit"><div><strong>Inject 1 A at C</strong><span>ABC triangle</span><span>all three potentials = 5 V</span></div><div><strong>C — D</strong><span>conductance 0.2 S</span><span>current 1 A →</span><span>drop 5 V</span></div><div><strong>Extract 1 A at D</strong><span>DEF triangle</span><span>all three potentials = 0 V</span></div></div><figcaption>One bridge is the only route for this current. Its resistance is 1/0.2=5 Ω. Other edges carry zero current in this particular injection experiment; the internal triangles still matter for different endpoints. These are declared circuit units, not the document-affinity units.</figcaption></figure>;
}
