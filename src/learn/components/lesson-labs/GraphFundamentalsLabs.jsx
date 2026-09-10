import { useId, useState } from 'react';
import { Investigation } from './LessonInvestigation.jsx';
import { foundationEdges, formatGraphNumber, graphAveragingTrace, graphConnectivity, graphEnergy, graphFoundationLabels, graphHarmonicInterpolation, graphMatrices, graphMatrixVector, graphNormalizations, graphWalks } from '../../data/graph-fundamentals-models.js';
import './graph-fundamentals-labs.css';
const labels = graphFoundationLabels;
const format = formatGraphNumber;
const positions = [[42, 58], [159, 34], [278, 58], [91, 171], [235, 171]];
function Range({
  label,
  value,
  setValue,
  min = 0,
  max = 6,
  step = 1
}) {
  return <label className="gf-range"><span>{label} <strong>{format(value)}</strong></span>
    <input aria-label={label} type="range" min={min} max={max} step={step} value={value} onChange={event => setValue(Number(event.target.value))} />
  </label>;
}
function Matrix({
  values,
  title,
  rowLabels = labels.slice(0, values.length),
  columnLabels = rowLabels,
  selected = null
}) {
  return <div className="gf-matrix" role="region" aria-label={title} tabIndex={0}>
    <table><caption>{title}</caption><thead><tr><th scope="col">row ↓ / col →</th>{columnLabels.map(label => <th scope="col" key={label}>{label}</th>)}</tr></thead>
      <tbody>{values.map((row, i) => <tr key={i}><th scope="row">{rowLabels[i]}</th>{row.map((value, j) => <td key={j} className={`${value ? 'gf-nonzero' : ''} ${selected?.[0] === i && selected?.[1] === j ? 'gf-selected' : ''}`}>{typeof value === 'boolean' ? Number(value) : format(value, 3)}</td>)}</tr>)}</tbody>
    </table>
  </div>;
}
function Network({
  edges,
  nodeLabels = labels,
  signal = null,
  directed = false,
  anchors = [],
  selected = [],
  title
}) {
  const arrowId = `gf-arrow-${useId().replaceAll(':', '')}`;
  const hasReverse = (u, v) => directed && edges.some(([a, b, w]) => w > 0 && a === v && b === u);
  const lineEdges = edges.filter(([,, weight]) => weight > 0);
  const hasLoop = edges.some(([u, v, weight]) => u === v && weight > 0);
  const viewBox = hasLoop ? '0 -32 320 256' : nodeLabels.length === 3 && !signal ? '0 0 320 100' : '0 0 320 224';
  return <svg className="gf-network" viewBox={viewBox} role="img" aria-label={title}>
    <title>{title}</title>
    <defs><marker id={arrowId} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="currentColor" /></marker></defs>
    {lineEdges.map(([u, v, weight], index) => {
      const [x1, y1] = positions[u];
      const [x2, y2] = positions[v];
      if (u === v) return <g key={index}><path d={`M${x1 - 12},${y1 - 14} C${x1 - 38},${y1 - 60} ${x1 + 38},${y1 - 60} ${x1 + 12},${y1 - 14}`} fill="none" stroke="var(--gf-gold)" strokeWidth="2" /><text x={x1} y={y1 - 49} textAnchor="middle">loop {weight}</text></g>;
      const length = Math.hypot(x2 - x1, y2 - y1);
      const ux = (x2 - x1) / length;
      const uy = (y2 - y1) / length;
      const offset = hasReverse(u, v) ? 17 : 0;
      const middle = [(x1 + x2) / 2 - uy * offset, (y1 + y2) / 2 + ux * offset];
      const path = `M${x1 + ux * 19},${y1 + uy * 19} Q${middle[0]},${middle[1]} ${x2 - ux * 23},${y2 - uy * 23}`;
      return <g key={index} className="gf-edge"><path d={path} markerEnd={directed ? `url(#${arrowId})` : undefined} fill="none" stroke="currentColor" strokeWidth={1.5 + Math.min(weight, 3)} />
        <rect x={middle[0] - 11} y={middle[1] - 11} width="22" height="20" rx="3" /><text x={middle[0]} y={middle[1] + 4} textAnchor="middle">{weight}</text>
      </g>;
    })}
    {nodeLabels.map((label, index) => {
      const [x, y] = positions[index];
      return <g key={label} className={`${selected.includes(index) ? 'gf-active-node' : ''} ${anchors.includes(index) ? 'gf-anchor-node' : ''}`}>
        {anchors.includes(index) && <rect x={x - 22} y={y - 22} width="44" height="44" rx="3" />}
        <circle cx={x} cy={y} r="18" /><text x={x} y={y + 6} textAnchor="middle">{label}</text>
        {signal && <text className="gf-signal-label" x={x} y={y + 42} textAnchor="middle">{signal[index] === null ? '?' : format(signal[index], 2)}</text>}
      </g>;
    })}
  </svg>;
}
function Groups({
  components
}) {
  return <span>{components.map((component, index) => <span className="gf-group" key={index}>{component.map(vertex => labels[vertex]).join(' · ')}</span>)}</span>;
}
function NodeValues({
  values,
  nodeLabels = labels.slice(0, values.length)
}) {
  return <div className="gf-node-values">{nodeLabels.map((label, i) => <span key={label} data-node-value={label}><strong>{label}</strong>{format(values[i])}</span>)}</div>;
}
export function GraphMatrixWalkLab() {
  const [directed, setDirected] = useState(false);
  const [bridge, setBridge] = useState(false);
  const [extra, setExtra] = useState(false);
  const [start, setStart] = useState(0);
  const [target, setTarget] = useState(2);
  const [steps, setSteps] = useState(2);
  const edges = foundationEdges(bridge ? 1 : 0, extra);
  if (directed) edges.push([2, 1, 1]);
  const graph = graphMatrices(5, edges, directed);
  const connectivity = graphConnectivity(graph.adjacency);
  const walks = graphWalks(graph.adjacency, start, target, steps);
  const reset = () => {
    setDirected(false);
    setBridge(false);
    setExtra(false);
    setStart(0);
    setTarget(2);
    setSteps(2);
  };
  return <Investigation id="graph-matrix-walks" kicker="ONE RELATION · LINKED VIEWS" title="Which matrix cells describe this journey?">
    <p>Predict the A-to-C entry of A² before inspecting the walks. The drawn weights are coupling strengths; every view below uses the same active graph.</p>
    <div className="gf-controls">
      <label><input type="checkbox" checked={directed} onChange={event => setDirected(event.target.checked)} /> Directed arrows, including C→B</label>
      <label><input type="checkbox" checked={bridge} onChange={event => setBridge(event.target.checked)} /> Add C–D bridge, weight 1</label>
      <label><input type="checkbox" checked={extra} onChange={event => setExtra(event.target.checked)} /> Add A–C edge, weight 1</label>
    </div>
    <div className="gf-linked"><Network edges={edges} directed={directed} selected={[start, target]} title="Five-vertex weighted graph; line positions are layout, labels are weights" /><Matrix values={graph.adjacency} title="A: weighted adjacency" selected={[start, target]} /></div>
    <p className="gf-caption">Row is source, column is destination. An undirected edge supplies two symmetric cells. Changing to arrows uses the shown directions; it also adds the explicitly named C→B edge.</p>
    <div className="gf-controls gf-selects">
      <label>From<select aria-label="Walk start" value={start} onChange={event => setStart(Number(event.target.value))}>{labels.map((label, i) => <option value={i} key={label}>{label}</option>)}</select></label>
      <label>To<select aria-label="Walk destination" value={target} onChange={event => setTarget(Number(event.target.value))}>{labels.map((label, i) => <option value={i} key={label}>{label}</option>)}</select></label>
      <label>Steps<select aria-label="Walk length" value={steps} onChange={event => setSteps(Number(event.target.value))}>{[0, 1, 2, 3].map(value => <option key={value}>{value}</option>)}</select></label>
      <button type="button" onClick={reset}>Reset graph and walk</button>
    </div>
    <p className="gf-result" aria-live="polite">A<sup>{steps}</sup>[{labels[start]}, {labels[target]}] = <strong>{format(walks.weightSum)}</strong>. {walks.walks.length} vertex-sequence {walks.walks.length === 1 ? 'walk contributes' : 'walks contribute'}.</p>
    <div className="gf-walks">{walks.walks.length ? walks.walks.map((walk, index) => <div key={index}><span>{walk.vertices.map(vertex => labels[vertex]).join(' → ')}</span><span>weight product {format(walk.weight)}</span></div>) : <p>No walk of this exact length. That alone does not rule out another length.</p>}</div>
    <details><summary>Inspect the full matrix power and connectivity</summary><Matrix values={walks.power} title={`A to power ${steps}`} selected={[start, target]} /><p>Weak / undirected components: <Groups components={connectivity.weakComponents} /></p><p>Strong components: <Groups components={connectivity.strongComponents} /></p><p>Each vertex reaches itself in zero steps. Matrix-power values count weighted walks of the selected length; the component calculation considers all reachable lengths.</p></details>
  </Investigation>;
}
export function FoundationGraphFigure() {
  const edges = foundationEdges();
  const graph = graphMatrices(5, edges);
  return <figure className="gf-inline" data-graph-figure="foundation-matrix">
    <div className="gf-linked"><Network edges={edges} title="The initial two-component weighted station network" /><Matrix values={graph.adjacency} title="Same graph: adjacency A" selected={[1, 0]} /></div>
    <figcaption>B's row has weight 2 toward A and weight 1 toward C. D and E occupy their own matrix block. The empty C–D cells represent the absent bridge.</figcaption>
  </figure>;
}
export function DirectedComponentsFigure() {
  const edges = [[0, 1, 1], [1, 2, 1], [2, 1, 1]];
  const result = graphConnectivity(graphMatrices(3, edges, true).adjacency);
  return <figure className="gf-inline" data-graph-figure="directed-components"><div className="gf-linked"><Network edges={edges} nodeLabels={labels.slice(0, 3)} directed title="A points to B, and B and C have reciprocal arrows" /><Matrix values={result.reachable} title="Can row reach column? 1=yes" /></div>
    <figcaption>A can reach B and C, but neither can return to A. There is one weak component A·B·C and two strong components: A and B·C. The diagonal includes the zero-step walk.</figcaption>
  </figure>;
}
export function LaplacianEnergyLab() {
  const [signal, setSignal] = useState([3, 1, 0, 2, 2]);
  const [bridge, setBridge] = useState(0);
  const [selected, setSelected] = useState(1);
  const result = graphEnergy(5, foundationEdges(bridge), signal);
  const maxEnergy = Math.max(1, ...result.edgeTerms.map(edge => edge.energy));
  return <Investigation id="graph-edge-energy" kicker="LOCAL DIFFERENCES · GLOBAL ENERGY" title="Where does the disagreement live?">
    <p>Predict the signed result at B. A larger neighbor pulls its contribution one way; a smaller neighbor pulls the other. The edge energies stay nonnegative even when the signed contributions cancel.</p>
    <div className="gf-linked"><Network edges={result.activeEdges} signal={signal} selected={[selected]} title="Node signal values and weighted edges for the disagreement calculation" /><div className="gf-controls gf-signal-controls">{labels.map((label, i) => <Range key={label} label={`Signal at ${label}`} value={signal[i]} setValue={value => setSignal(previous => previous.map((entry, index) => index === i ? value : entry))} />)}</div></div>
    <div className="gf-controls"><Range label="Bridge C–D weight" value={bridge} max={3} setValue={setBridge} /><label>Inspect row<select aria-label="Laplacian row" value={selected} onChange={event => setSelected(Number(event.target.value))}>{labels.map((label, i) => <option key={label} value={i}>{label}</option>)}</select></label>
      <button type="button" onClick={() => {
        setSignal([3, 3, 3, 1, 1]);
        setBridge(0);
      }}>Make each component constant</button>
      <button type="button" onClick={() => {
        setSignal([3, 1, 0, 2, 2]);
        setBridge(0);
        setSelected(1);
      }}>Reset energy</button>
    </div>
    <div className="gf-row-equation">{result.adjacency[selected].map((weight, neighbor) => weight > 0 && <span key={neighbor}>{format(weight)} × ({format(signal[selected])} − {format(signal[neighbor])})<small>edge {labels[selected]}–{labels[neighbor]}: {format(weight * (signal[selected] - signal[neighbor]))}</small></span>)}<strong>Σ = {format(result.action[selected])}</strong></div>
    <p>All coordinates of Lx, with signed values:</p><NodeValues values={result.action} />
    <div className="gf-energy-bars" aria-label="Energy per undirected edge">{result.edgeTerms.map((edge, i) => <div key={i}><span>{labels[edge.source]}–{labels[edge.target]}</span><div><i style={{
            width: `${100 * edge.energy / maxEnergy}%`
          }} /></div><span>{format(edge.weight)} × {format(edge.drop)}² = <strong>{format(edge.energy)}</strong></span></div>)}</div>
    <p className="gf-caption">Bar lengths share the current scale 0 to {format(maxEnergy)} energy units. Each undirected edge appears once.</p>
    <p className="gf-result" aria-live="polite">Edge sum = <strong>{format(result.edgeEnergy)}</strong>; independently multiplying xᵀLx gives <strong>{format(result.quadraticEnergy)}</strong>.</p>
    <p>Components: <Groups components={result.components} />. Try the constant-per-component preset, then add the bridge. Different component values become a disagreement only when a positive coupling connects them.</p>
  </Investigation>;
}
export function IncidenceFlowFigure() {
  const result = graphEnergy(5, foundationEdges(), [3, 1, 0, 2, 2]);
  return <figure className="gf-inline" data-graph-figure="incidence-flow">
    <div className="gf-incidence-flow"><div><strong>Node potentials x</strong><NodeValues values={[3, 1, 0, 2, 2]} /><p>C takes tail minus head</p></div><div><strong>Edge drops Cx</strong><NodeValues values={result.edgeTerms.map(edge => edge.drop)} nodeLabels={['A→B', 'B→C', 'D→E']} /><p>W multiplies by conductance</p></div><div><strong>Edge currents WCx</strong><NodeValues values={result.edgeTerms.map(edge => edge.flow)} nodeLabels={['A→B', 'B→C', 'D→E']} /><p>Cᵀ gathers signed outflow</p></div></div>
    <div className="gf-flow-result"><strong>Back at nodes: Lx</strong><NodeValues values={result.action} /></div>
    <figcaption>The A→B current is 4; B sends 1 toward C while receiving 4 from A, so its net outward current is −3. The sum of all five node outflows is zero: every internal current enters one endpoint and leaves the other. Arrows here choose a sign convention for undirected couplings.</figcaption>
  </figure>;
}
export function GraphNormalizationLab() {
  const [loop, setLoop] = useState(false);
  const [operator, setOperator] = useState('symmetric');
  const [vector, setVector] = useState('ones');
  const edges = [[0, 1, 2], [1, 2, 1], ...(loop ? [[1, 1, 1]] : [])];
  const result = graphNormalizations(4, edges);
  const names = {
    laplacian: 'L = D − A',
    symmetric: 'S = H L H',
    randomWalk: 'R = D† L',
    transition: 'P: hold at isolates',
    naiveIdentity: 'Naive I − H A H'
  };
  const matrix = result[operator];
  const values = vector === 'ones' ? [1, 1, 1, 1] : result.degrees.map(Math.sqrt);
  const action = graphMatrixVector(matrix, values);
  return <Investigation id="graph-normalization" kicker="SAME EDGES · DIFFERENT OPERATORS" title="What should happen to the isolated vertex?">
    <p>Predict the D diagonal before selecting the naive formula. Here Hii is 1/√di when di is positive and zero otherwise. The highlighted D diagonal makes the isolated-vertex convention visible.</p>
    <div className="gf-controls"><label><input type="checkbox" checked={loop} onChange={event => setLoop(event.target.checked)} /> Add a unit self-loop at B</label>
      <label>Operator<select aria-label="Graph normalization operator" value={operator} onChange={event => setOperator(event.target.value)}>{Object.entries(names).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
      <label>Input signal<select aria-label="Normalization signal" value={vector} onChange={event => setVector(event.target.value)}><option value="ones">All ones</option><option value="root">Square-root degree</option></select></label>
      <button type="button" onClick={() => {
        setLoop(false);
        setOperator('symmetric');
        setVector('ones');
      }}>Reset normalization</button>
    </div>
    <div className="gf-linked"><Network edges={edges} nodeLabels={labels.slice(0, 4)} title="Weighted path A-B-C, isolated D, optional self-loop at B" /><Matrix values={matrix} title={names[operator]} selected={[3, 3]} /></div>
    <p>Row-sum degree:</p><NodeValues values={result.degrees} />
    <p>Input signal:</p><NodeValues values={values} />
    <p className="gf-result" aria-live="polite">Operator × signal:</p><NodeValues values={action} />
    <p className="gf-caption">Rounded numerical readouts; tiny nonzero floating residuals remain visible. S annihilates √degree but generally not the all-ones signal. The isolate also supplies its own zero direction; √degree alone has a zero there.</p>
    <p>{operator === 'naiveIdentity' ? 'The isolate now has diagonal 1. That formula differs from H L H at D, so it describes another operator.' : operator === 'transition' ? 'The D row holds its value with PDD=1. Every row sums to one, including the isolate.' : 'The isolate has an all-zero generator row. It neither contributes disagreement nor exchanges with another vertex.'} {loop ? 'The loop cancels from L, but it changes B’s degree and therefore the normalized operators.' : 'With no self-loops, positive-degree diagonal entries of S are one.'}</p>
  </Investigation>;
}
function Trajectories({
  states
}) {
  const all = states.flat();
  const low = Math.min(0, ...all);
  const high = Math.max(6, ...all);
  const x = step => 74 + step / Math.max(1, states.length - 1) * 218;
  const y = value => 178 - (value - low) / (high - low) * 142;
  return <svg className="gf-trajectories" viewBox="0 0 320 240" role="img" aria-label="Calculated node signals against discrete update step">
    <title>Calculated trajectories: horizontal axis is update step, vertical axis is signal value</title>
    {[low, (low + high) / 2, high].map(value => <g key={value}><line x1="74" x2="292" y1={y(value)} y2={y(value)} /><text x="67" y={y(value) + 5} textAnchor="end">{format(value, 1)}</text></g>)}
    <text x="74" y="202" textAnchor="middle">0</text><text x="292" y="202" textAnchor="middle">{Math.max(1, states.length - 1)}</text><text x="183" y="228" textAnchor="middle">update step</text><text x="74" y="19">signal value</text>
    {states[0].map((_, vertex) => <g key={vertex} className={`gf-series gf-series-${vertex}`}><polyline points={states.map((values, step) => `${x(step)},${y(values[vertex])}`).join(' ')} />{states.map((values, step) => <circle key={step} cx={x(step)} cy={y(values[vertex])} r="2.8" />)}</g>)}
  </svg>;
}
export function GraphAveragingLab() {
  const [method, setMethod] = useState('exchange');
  const [steps, setSteps] = useState(0);
  const [stepSize, setStepSize] = useState(0.25);
  const result = graphAveragingTrace(method, steps, stepSize);
  const current = result.states.at(-1);
  const summary = result.summaries.at(-1);
  return <Investigation id="graph-averaging" kicker="ACTUAL ITERATIONS · PRESERVED QUANTITIES" title="Which average survives the update?">
    <p>This separate fixture is the unweighted path A—B—C, plus isolated D. Initial values are 6, 0, 0, 4. Predict the first two steps and which mean stays fixed.</p>
    <div className="gf-controls"><label>Update rule<select aria-label="Averaging rule" value={method} onChange={event => {
          setMethod(event.target.value);
          setSteps(0);
        }}><option value="exchange">Conservative exchange</option><option value="neighbor">Neighbor averaging</option><option value="lazy">Lazy neighbor averaging</option></select></label>
      {method === 'exchange' && <label>Step size τ<select aria-label="Exchange step size" value={stepSize} onChange={event => {
          setStepSize(Number(event.target.value));
          setSteps(0);
        }}><option value={0.25}>0.25</option><option value={0.5}>0.5</option><option value={0.75}>0.75: test beyond bound</option></select></label>}
      <button type="button" disabled={steps === 0} onClick={() => setSteps(steps - 1)}>Previous update</button>
      <button type="button" disabled={steps === 24} onClick={() => setSteps(steps + 1)}>Apply one update</button>
      <button type="button" onClick={() => setSteps(24)}>Inspect step 24</button>
      <button type="button" onClick={() => {
        setMethod('exchange');
        setSteps(0);
        setStepSize(0.25);
      }}>Reset averaging</button>
    </div>
    <div className="gf-linked"><Network edges={[[0, 1, 1], [1, 2, 1]]} nodeLabels={labels.slice(0, 4)} signal={current} title={`Node values after ${steps} calculated updates`} /><Trajectories states={result.states} /></div>
    <div className="gf-legend">{labels.slice(0, 4).map((label, i) => <span key={label} className={`gf-series-${i}`}>{label}{i === 3 ? ' (isolate)' : ''}</span>)}</div>
    <p className="gf-result" aria-live="polite">Step {steps}: A–B–C ordinary mean <strong>{format(summary.ordinaryMean)}</strong>; degree-weighted mean <strong>{format(summary.weightedMean)}</strong>. D stays {format(current[3])}.</p>
    <p>{method === 'exchange' ? 'Ordinary mean 2 is conserved on A–B–C. For this graph τ≤0.5 guarantees convex-combination rows; τ=0.75 loses that guarantee and this example eventually oscillates with growing amplitude.' : method === 'neighbor' ? 'The degree-weighted mean 1.5 is conserved. After the first update this example alternates between [0,3,0] and [3,0,3]; connectivity alone did not guarantee convergence.' : 'Keeping half of each previous value preserves the same degree-weighted mean 1.5 and damps the alternating mode.'}</p>
    <p className="gf-caption">Computed linear model, not measured time. Changing the rule or τ restarts the trajectory. Colors identify nodes; the table supplies exact node labels and numeric steps.</p>
    <details><summary>Inspect update matrix and every calculated state</summary><Matrix values={result.update} title="Update matrix" /><Matrix values={result.states} title="Signal at each update" rowLabels={result.states.map((_, i) => String(i))} columnLabels={labels.slice(0, 4)} /></details>
  </Investigation>;
}
export function GraphHarmonicLab() {
  const [left, setLeft] = useState(6);
  const [right, setRight] = useState(0);
  const [bridge, setBridge] = useState(false);
  const [secondAnchor, setSecondAnchor] = useState(false);
  const anchors = {
    0: left,
    2: right,
    ...(secondAnchor ? {
      3: 2
    } : {})
  };
  const result = graphHarmonicInterpolation(5, foundationEdges(bridge ? 1 : 0), anchors);
  return <Investigation id="graph-harmonic-anchors" kicker="BOUNDARY VALUES · A UNIQUE ANSWER?" title="What can the missing values actually be?">
    <p>A and C are measured anchors. B is chosen to minimize weighted disagreement. Before changing anything, decide whether the disconnected D–E pair can have a unique value.</p>
    <div className="gf-controls"><Range label="Anchor A" value={left} max={8} setValue={setLeft} /><Range label="Anchor C" value={right} max={8} setValue={setRight} />
      <label><input type="checkbox" checked={bridge} onChange={event => setBridge(event.target.checked)} /> Connect C–D with weight 1</label>
      <label><input type="checkbox" checked={secondAnchor} onChange={event => setSecondAnchor(event.target.checked)} /> Anchor D at 2</label>
      <button type="button" onClick={() => {
        setLeft(6);
        setRight(0);
        setBridge(false);
        setSecondAnchor(false);
      }}>Reset anchors</button>
    </div>
    <Network edges={result.activeEdges} signal={result.values} anchors={Object.keys(anchors).map(Number)} title="Anchored vertices have square outlines; question marks indicate an undetermined component" />
    <NodeValues values={result.values} />
    <p className="gf-result" aria-live="polite">{result.unique ? 'Every component has an anchor: the unknown values have a unique harmonic solution.' : <>Unanchored component: <Groups components={result.unanchored} />. Its level is undetermined; the other component can still be solved.</>}</p>
    <p>At B, the equation is 2(B − {format(left)}) + (B − {format(right)}) = 0, giving B = {format(result.values[1])}. Its two incident couplings are unchanged when the bridge is added.</p>
    <details><summary>Inspect the reduced system and outflow residuals</summary><Matrix values={result.reduced} title="Reduced L for supported unknowns" rowLabels={result.unknown.map(vertex => labels[vertex])} /><p>Right side: [{result.rightSide.map(value => format(value)).join(', ')}].</p><p>All Lx coordinates:</p><NodeValues values={result.residuals} /><p>Unanchored unknowns remain unspecified. At solved unknowns the net outflow is zero; anchors may supply or absorb current, so their residuals need not be zero.</p></details>
    <p className="gf-caption">Computed minimizer of a stated coupling-energy model. Square outlines indicate fixed values, not estimated confidence. A real sensor needs evidence that smoothness is an appropriate assumption.</p>
  </Investigation>;
}
export function GraphAvailabilityFigure() {
  return <figure className="gf-inline" data-graph-figure="availability"><div className="gf-cutoff-flow">
    <div><strong>Available by time 5</strong><span className="gf-time-edge">A — B <small>time 2</small></span><span className="gf-time-edge">B — C <small>time 3</small></span><p>C may be a test vertex while its already-known relationships remain allowed in a declared transductive task.</p></div>
    <div className="gf-future"><strong>Prediction cutoff: 5</strong><span className="gf-time-edge">A — C <small>time 6 · future target</small></span><p>This future relationship cannot be used to construct features for the time-5 prediction. Removing a label alone does not remove its edge evidence.</p></div>
  </div><figcaption>The drawing is an original data-availability example. Whether an edge is permitted follows the task's information contract, not merely whether an endpoint is called “test.”</figcaption></figure>;
}
