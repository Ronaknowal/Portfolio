import { useMemo, useState } from 'react';
import { squareElementNames as names, squareGroup, compositionState, moveSquareData, vertexImage, orbitState, fixedColoringCount, permutationCycles, cosetProductState, equivarianceState, parseSensorReadings, modularState, formatAlgebraNumber as number } from '../../data/abstract-algebra-models.js';
import './abstract-algebra-labs.css';
const colors = ['#efb944', '#83c6a8', '#b7a0d5'];
const positions = [[180, 115], [110, 45], [40, 115], [110, 185]];
function SelectElement({
  label,
  value,
  onChange
}) {
  return <label>{label}<select aria-label={label} value={value} onChange={event => onChange(Number(event.target.value))}>
    {names.map((name, index) => <option key={name} value={index}>{name}</option>)}
  </select></label>;
}
function Table({
  caption,
  headers,
  rows
}) {
  return <div className="algebra-scroll" role="region" aria-label={caption} tabIndex={0}>
    <table><caption>{caption}</caption><thead><tr>{headers.map(header => <th scope="col" key={header}>{header}</th>)}</tr></thead>
      <tbody>{rows.map((row, index) => <tr key={index}>{row.map((value, column) => <td key={column}>{value}</td>)}</tr>)}</tbody>
    </table>
  </div>;
}
export function SquareData({
  values = ['A', 'B', 'C', 'D'],
  title,
  selected = null,
  coloring = false
}) {
  const numeric = !coloring && values.every(value => typeof value === 'number');
  return <figure className="algebra-square"><figcaption>{title}</figcaption>
    <svg viewBox="0 0 220 230" role="img" aria-label={`${title}. Values at positions 0 to 3: ${values.join(', ')}.`}>
      <path d="M110 24V205M20 115H200" className="algebra-axis" />
      <path d="M180 115L110 45L40 115L110 185Z" className="algebra-outline" />
      {positions.map(([x, y], index) => <g key={index}>
        {numeric ? <rect x={x - 36} y={y - 19} width="72" height="38" rx="2" fill="#161711" stroke="#686956" /> : <circle cx={x} cy={y} r={selected === index ? 23 : 20} fill={coloring ? colors[values[index]] : '#161711'} stroke={selected === index ? '#efb944' : '#686956'} strokeWidth={selected === index ? 3 : 1} />}
        <text x={x} y={y + 6} textAnchor="middle" className={coloring ? 'algebra-color-label' : ''} style={numeric && String(values[index]).length > 5 ? {
          fontSize: 15
        } : undefined}>{values[index]}</text>
        <text x={numeric ? x : index === 0 ? x + 30 : index === 2 ? x - 30 : x} y={numeric ? index === 3 ? y + 39 : y - 27 : index === 1 ? y - 28 : index === 3 ? y + 39 : y + 6} textAnchor="middle" className="algebra-position">{index}</text>
      </g>)}
    </svg>
  </figure>;
}
function Route({
  title,
  states,
  stepNames,
  tracked = [],
  active = null,
  coloring = false
}) {
  return <div className="algebra-route"><h4>{title}</h4><div className="algebra-stages">
    {states.map((values, index) => <div key={index} className={active === index ? 'algebra-active-stage' : ''}>
      {index > 0 && <p className="algebra-operation">{stepNames[index - 1]} <span aria-hidden="true">→</span></p>}
      <SquareData values={values} title={index === 0 ? 'Start' : index === 1 ? 'After first move' : 'Final result'} selected={tracked[index]} coloring={coloring} />
    </div>)}
  </div></div>;
}
export function SquareCompositionLab() {
  const [g, setG] = useState(1);
  const [h, setH] = useState(4);
  const [vertex, setVertex] = useState(1);
  const [step, setStep] = useState(2);
  const state = compositionState(g, h, vertex);
  const reset = () => {
    setG(1);
    setH(4);
    setVertex(1);
    setStep(2);
  };
  return <section className="algebra-lab" data-algebra-lab="composition" aria-label="Square composition investigation">
    <h3>Watch the same label take two routes</h3>
    <p>The outline and position numbers stay fixed. A, B, C and D travel with the square. The gold ring follows the label that started at your selected vertex. Predict its destinations before comparing the lanes.</p>
    <div className="algebra-controls"><SelectElement label="Outer move g" value={g} onChange={setG} /><SelectElement label="Inner move h" value={h} onChange={setH} />
      <label>Track starting vertex<select aria-label="Track starting vertex" value={vertex} onChange={event => setVertex(Number(event.target.value))}>{[0, 1, 2, 3].map(value => <option key={value}>{value}</option>)}</select></label>
    </div>
    <Route title={`${names[g]}${names[h]}: apply h, then g`} states={state.firstStates} stepNames={[`Apply ${names[h]}`, `Apply ${names[g]}`]} tracked={state.firstRoute} active={step} />
    <Route title={`${names[h]}${names[g]}: apply g, then h`} states={state.secondStates} stepNames={[`Apply ${names[g]}`, `Apply ${names[h]}`]} tracked={state.secondRoute} active={step} />
    <div className="algebra-buttons"><button onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0}>Previous stage</button><button onClick={() => setStep(Math.min(2, step + 1))} disabled={step === 2}>Next stage</button><button onClick={reset}>Reset composition</button></div>
    <p role="status">Highlighted stage {step}: tracked vertex is {state.firstRoute[step]} in the first lane and {state.secondRoute[step]} in the second. Products: gh={names[state.product]}, hg={names[state.reversed]}. The inverse of gh is {names[state.inverse]}.</p>
    <details><summary>Inspect the complete composition table</summary><p>Read the row after the column. The highlighted cell is your current gh. This table records results, not a proof of associativity. On a narrow screen, focus the table and scroll sideways.</p>
      <Table caption="D4: row after column" headers={['g after h', ...names]} rows={state.table.map((row, i) => [names[i], ...row.map((value, j) => i === g && j === h ? <strong className="algebra-selected-value">{names[value]}</strong> : names[value])])} />
    </details>
  </section>;
}
export function GroupActionFigure() {
  const top = [0, 1, 2, 3];
  const bottom = [4, 7, 6, 5];
  return <figure className="algebra-figure">
    <h3>Eight transformations; two configurations</h3>
    <p>The first graph's nodes are group elements. An arrow labelled r sends g to rg; s sends g to sg. A two-headed s edge records both directions.</p>
    <p className="algebra-scroll-hint">Scroll sideways to see all eight moves when the graph is wider than the screen.</p>
    <div className="algebra-scroll" role="region" aria-label="Eight-element Cayley graph; scroll horizontally on a narrow screen" tabIndex={0}>
      <svg className="algebra-cayley" viewBox="0 0 600 310" role="img" aria-label="Top r cycle e,r,r squared,r cubed; bottom reverse r cycle s,r cubed s,r squared s,rs. Vertical s edges join each aligned pair.">
        <defs><marker id="algebra-arrow-r" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#efb944" /></marker><marker id="algebra-arrow-s" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#83c6a8" /></marker></defs>
        {[0, 1, 2].map(index => <g key={index}><path d={`M${89 + 140 * index} 82H${181 + 140 * index}`} className="algebra-r" markerEnd="url(#algebra-arrow-r)" /><text x={135 + 140 * index} y="69">r</text><path d={`M${181 + 140 * index} 220H${105 + 140 * index}`} className="algebra-r" markerEnd="url(#algebra-arrow-r)" /><text x={135 + 140 * index} y="247">r</text></g>)}
        <path d="M510 63C510 16 70 16 70 63" className="algebra-r" markerEnd="url(#algebra-arrow-r)" /><text x="287" y="24">r</text>
        <path d="M70 238C70 287 510 287 510 238" className="algebra-r" markerEnd="url(#algebra-arrow-r)" /><text x="287" y="302">r</text>
        {top.map((g, index) => <g key={g}><path d={`M${70 + 140 * index} 103V197`} className="algebra-s" markerStart="url(#algebra-arrow-s)" markerEnd="url(#algebra-arrow-s)" /><text x={82 + 140 * index} y="155">s</text>
          <circle cx={70 + 140 * index} cy="82" r="22" /><text x={70 + 140 * index} y="89" textAnchor="middle">{names[g]}</text>
          <circle cx={70 + 140 * index} cy="220" r="26" /><text x={70 + 140 * index} y="227" textAnchor="middle">{names[bottom[index]]}</text>
        </g>)}
      </svg>
    </div>
    <div className="algebra-action-pair"><SquareData values={[1, 0, 1, 0]} title="Opposite marks: state A" coloring /><div><svg className="algebra-action-graph" viewBox="0 0 220 150" role="img" aria-label="Action graph with nodes A and B, an r arrow in both directions and an s loop at each node">
      <path d="M40 69C4 6 96 6 60 69" className="algebra-s" markerEnd="url(#algebra-arrow-s)" /><path d="M160 69C124 6 216 6 180 69" className="algebra-s" markerEnd="url(#algebra-arrow-s)" />
      <path d="M75 88H145" className="algebra-r" markerStart="url(#algebra-arrow-r)" markerEnd="url(#algebra-arrow-r)" />
      <circle cx="50" cy="88" r="22" /><circle cx="170" cy="88" r="22" />
      <text x="50" y="95" textAnchor="middle">A</text><text x="170" y="95" textAnchor="middle">B</text><text x="110" y="75" textAnchor="middle">r</text><text x="17" y="45" textAnchor="middle">s</text><text x="137" y="45" textAnchor="middle">s</text>
    </svg><p>r swaps A and B.<br />s fixes each state.</p></div><SquareData values={[0, 1, 0, 1]} title="Opposite marks: state B" coloring /></div>
    <figcaption>In the action graph there are only two distinct nodes, A and B: r gives a two-way edge and s gives a loop at each node. The action forgets which of four transformations produced a given state. In the Cayley graph, all eight transformations remain different. The same letter r labels different kinds of movement because the node sets differ.</figcaption>
  </figure>;
}
export function ColoringOrbitLab() {
  const [colorCount, setColorCount] = useState(2);
  const [colorsAtVertices, setColors] = useState([1, 0, 1, 0]);
  const [rotationsOnly, setRotationsOnly] = useState(false);
  const [g, setG] = useState(4);
  const orbit = orbitState(colorsAtVertices, rotationsOnly);
  const tally = fixedColoringCount(colorCount, rotationsOnly);
  const cycles = permutationCycles(g);
  const transformed = moveSquareData(g, colorsAtVertices);
  const fixed = transformed.every((value, index) => value === colorsAtVertices[index]);
  return <section className="algebra-lab" data-algebra-lab="orbits" aria-label="Coloring orbit investigation">
    <h3>Separate different moves from different results</h3>
    <p>Colors are named 0, 1 and optionally 2. Only vertex positions move; exchanging color names is not an allowed symmetry. Predict the orbit size for opposite marks, then try adjacent marks.</p>
    <div className="algebra-controls"><label>Available colors<select aria-label="Available colors" value={colorCount} onChange={event => {
          const next = Number(event.target.value);
          setColorCount(next);
          setColors(values => values.map(value => value % next));
        }}><option value="2">Two named colors</option><option value="3">Three named colors</option></select></label>
      <label>Allowed symmetries<select aria-label="Allowed symmetries" value={rotationsOnly ? 'rotations' : 'full'} onChange={event => {
          const only = event.target.value === 'rotations';
          setRotationsOnly(only);
          if (only) setG(value => value % 4);
        }}><option value="full">D4: rotations and reflections</option><option value="rotations">C4: rotations only</option></select></label>
    </div>
    <div className="algebra-controls algebra-paint-controls">{colorsAtVertices.map((value, index) => <label key={index}>Vertex {index}<select aria-label={`Color at vertex ${index}`} value={value} onChange={event => setColors(values => values.map((old, i) => i === index ? Number(event.target.value) : old))}>{Array.from({
            length: colorCount
          }, (_, color) => <option key={color}>{color}</option>)}</select></label>)}</div>
    <div className="algebra-buttons"><button onClick={() => setColors([1, 0, 1, 0])}>Opposite marks</button><button onClick={() => setColors([1, 1, 0, 0])}>Adjacent marks</button><button onClick={() => setColors([0, 0, 0, 0])}>Constant coloring</button><button onClick={() => {
        setColorCount(2);
        setColors([1, 0, 1, 0]);
        setRotationsOnly(false);
        setG(4);
      }}>Reset colorings</button></div>
    <p role="status">{orbit.group.length} transformations produce {orbit.orbitSize} distinct states. The current coloring's stabilizer is {'{'}{orbit.stabilizer.map(value => names[value]).join(', ')}{'}'}. Product: {orbit.orbitSize} × {orbit.stabilizer.length} = {orbit.group.length}.</p>
    <div className="algebra-orbit-collection">{orbit.states.map((state, index) => <div key={state.colors.join(',')}><SquareData values={state.colors} title={`Distinct state ${index + 1}`} coloring /><p>Reached by {state.transformations.map(value => names[value]).join(', ')}.</p></div>)}</div>
    <h4>Why do some moves fix more colorings?</h4>
    <label>Inspect one transformation<select aria-label="Inspect one transformation" value={g} onChange={event => setG(Number(event.target.value))}>{orbit.group.map(value => <option value={value} key={value}>{names[value]}</option>)}</select></label>
    <p>Vertex cycles of {names[g]}: {cycles.map(cycle => `(${cycle.join(' → ')} → ${cycle[0]})`).join(' ')}. A fixed coloring uses one color along each whole cycle. There are {colorCount}<sup>{cycles.length}</sup>={colorCount ** cycles.length} such colorings among all {tally.totalColorings}. Your particular coloring is {fixed ? 'fixed' : 'changed'} by this move.</p>
    <Table caption="Fixed-coloring count across the complete chosen group" headers={['Move', 'Cycles', 'Fixed colorings']} rows={tally.rows.map(row => [names[row.g], row.cycles.map(cycle => `(${cycle.join(' ')})`).join(' '), row.fixed])} />
    <p className="algebra-result">All configurations, up to the chosen group: {tally.numerator} ÷ {tally.group.length} = <strong>{tally.orbitCount} classes</strong>. This counts every possible coloring, not only the orbit you painted above.</p>
  </section>;
}
export function CosetProductLab() {
  const [normal, setNormal] = useState(false);
  const [choice, setChoice] = useState(0);
  const state = cosetProductState(normal, choice);
  return <section className="algebra-lab" data-algebra-lab="cosets" aria-label="Coset representative investigation">
    <h3>Change the representative; should the answer change?</h3>
    <p>Attempt to multiply the input blocks H and rH by multiplying one representative of each. The right representative stays r. Change the left representative inside H and inspect the output block.</p>
    <div className="algebra-controls"><label>Subgroup H<select aria-label="Subgroup H" value={normal ? 'rotations' : 'reflection'} onChange={event => {
          setNormal(event.target.value === 'rotations');
          setChoice(0);
        }}><option value="reflection">Reflection subgroup</option><option value="rotations">Rotation subgroup</option></select></label>
      <label>Representative from H<select aria-label="Representative from H" value={choice} onChange={event => setChoice(Number(event.target.value))}>{state.subgroup.map((g, index) => <option value={index} key={g}>{names[g]}</option>)}</select></label>
    </div>
    <div className="algebra-product-witness"><div><small>Left input block H</small><strong>{'{'}{state.subgroup.map(g => names[g]).join(', ')}{'}'}</strong><span>Choose {names[state.representative]}</span></div><span aria-hidden="true">×</span><div><small>Right input block rH</small><strong>{'{'}{state.expected.map(g => names[g]).join(', ')}{'}'}</strong><span>Choose r</span></div><span aria-hidden="true">→</span><div><small>Representative product</small><strong>{names[state.representative]}r = {names[state.product]}</strong><span>Find its block below</span></div></div>
    <div className="algebra-coset-blocks">{state.cosets.map(coset => <div key={coset.representative} className={coset.members.join(',') === state.output.join(',') ? 'algebra-output-block' : ''}><strong>{coset.representative === 0 ? 'H' : `${names[coset.representative]}H`}</strong><span>{'{'}{coset.members.map(g => names[g]).join(', ')}{'}'}</span>{coset.members.join(',') === state.output.join(',') && <small>Current output</small>}</div>)}</div>
    <p role="status">{state.sameOutput ? 'This choice gives the same output block as choosing e.' : 'The output block changed although the left input block stayed H. This attempted quotient multiplication is not well-defined.'} {normal ? 'Every representative choice works for this normal subgroup; the general proof explains why.' : 'Compare e and s to expose the failed contract.'}</p>
    <button onClick={() => {
      setNormal(false);
      setChoice(0);
    }}>Reset representatives</button>
  </section>;
}
export function EquivarianceLab() {
  const [draft, setDraft] = useState('1, 2, 4, 8');
  const [values, setValues] = useState([1, 2, 4, 8]);
  const [mode, setMode] = useState('tied');
  const [g, setG] = useState(4);
  const [weights, setWeights] = useState([2, 0.5, 0]);
  const [error, setError] = useState('');
  const state = useMemo(() => equivarianceState(values, mode, g, weights), [values, mode, g, weights]);
  const apply = event => {
    event.preventDefault();
    try {
      setValues(parseSensorReadings(draft));
      setError('');
    } catch (problem) {
      setError(problem.message);
    }
  };
  const reset = () => {
    setDraft('1, 2, 4, 8');
    setValues([1, 2, 4, 8]);
    setMode('tied');
    setG(4);
    setWeights([2, 0.5, 0]);
    setError('');
  };
  return <section className="algebra-lab" data-algebra-lab="equivariance" aria-label="Equivariant sensor map investigation">
    <h3>Transform then calculate, or calculate then transform?</h3>
    <p>Four vertex channels store voltages. Every arrow uses the same permutation action and the same matrix W. Weights are dimensionless. Predict which maps pass reflection as well as rotation.</p>
    <form onSubmit={apply}><label>Sensor readings (V)<input aria-label="Sensor readings" value={draft} onChange={event => setDraft(event.target.value)} /></label><p>Four decimals in [−20, 20], in increments of 0.25. Changes apply on the button; a rejected draft preserves the current calculation.</p><button type="submit">Apply readings</button></form>
    {error && <p role="alert">{error}</p>}
    <div className="algebra-controls"><label>Processing map<select aria-label="Processing map" value={mode} onChange={event => setMode(event.target.value)}><option value="tied">Self / neighbor / opposite</option><option value="raw">Raw matrix</option><option value="rotations">Raw matrix: C4 average</option><option value="averaged">Raw matrix: D4 average</option><option value="shift">One-position shift</option></select></label><SelectElement label="Data transformation g" value={g} onChange={setG} /></div>
    {mode === 'tied' && <div className="algebra-controls">{['Self weight', 'Neighbor weight', 'Opposite weight'].map((label, index) => <label key={label}>{label}: {weights[index]}<input type="range" aria-label={label} min="-4" max="4" step="0.25" value={weights[index]} onChange={event => setWeights(old => old.map((value, i) => i === index ? Number(event.target.value) : value))} /></label>)}</div>}
    <Table caption="Actual W: output rows and source columns 0 to 3" headers={['Output / source', '0', '1', '2', '3']} rows={state.matrix.map((row, index) => [index, ...row.map(number)])} />
    <Route title={`Upper route: transform by ${names[g]}, then apply W`} states={[values, state.transformedInput, state.transformThenMap]} stepNames={[`Apply ${names[g]}`, 'Apply W']} />
    <Route title={`Lower route: apply W, then transform by ${names[g]}`} states={[values, state.mapped, state.mapThenTransform]} stepNames={['Apply W', `Apply ${names[g]}`]} />
    <p role="status">Upper minus lower at vertices 0 to 3: [{state.difference.map(number).join(', ')}] V. Current-input defect: {number(state.inputSpecificDefect)} V. Largest matrix-entry defect over all eight transformations: {number(state.allInputDefect)}. {state.allInputDefect === 0 ? 'The matrix identities certify every real input vector for D4.' : 'At least one transformation fails for some input, even if this current input happens to pass.'}</p>
    <details><summary>Inspect the all-input certificate and one output</summary><p>Each row below is the largest absolute entry of WPg−PgW. Exact zero is meaningful for these bounded binary-fraction weights; no tolerance hides a failure.</p><Table caption="All-input matrix defects" headers={['g', 'Maximum entry defect']} rows={state.commutators.map(row => [names[row.g], number(row.maximum)])} /><p>Original output at vertex 0: {state.matrix[0].map((weight, index) => `${number(weight)} × ${number(values[index])}`).join(' + ')} = {number(state.mapped[0])} V.</p></details>
    <p>Mean of W applied to the original input: {number(state.poolBefore)} V. Mean of W applied to the transformed input: {number(state.poolAfter)} V. Pooling a permuted output is always unchanged; for the complete input-to-mean calculation, the preceding W must satisfy the appropriate contract too.</p>
    <button onClick={reset}>Reset sensor map</button>
  </section>;
}
export function ModularDivisionFigure() {
  const states = [modularState(5, 2, 4), modularState(6, 2, 4)];
  return <figure className="algebra-figure"><h3>Same multiplier; different information loss</h3><div className="algebra-modular-pair">{states.map(state => <div key={state.modulus}><h4>Multiply by 2 modulo {state.modulus}</h4><div className="algebra-residue-map">{state.outputs.map((output, input) => <div key={input} className={output === state.target ? 'algebra-hit' : ''}><span>{input}</span><span aria-hidden="true">→</span><strong>{output}</strong></div>)}</div><p>Solutions of 2x≡4: {state.solutions.join(', ')}. {state.injective ? 'Every output has one preimage. The inverse multiplier is 3.' : 'Different inputs collide. A unique inverse multiplier does not exist.'}</p></div>)}</div><figcaption>Residues are the integers 0 through n−1. Arrows show actual modular multiplication. The two highlighted preimages modulo 6 explain why ordinary cancellation would discard a solution.</figcaption></figure>;
}
