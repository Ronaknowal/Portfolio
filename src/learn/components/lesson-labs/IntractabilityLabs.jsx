import { useState } from 'react';
import { LessonTable } from './LessonElements';
import { formulaPresets, coverPresets, literalName, formatFormula, parseFormula, evaluateFormula, truthTable, reduceFormulaToClique, inspectSelection, firstCliqueChoices, encodingCounts, formatCoverEdges, parseCoverEdges, isCover, matchingCover, exactCover, minimumCover } from '../../data/intractability-models.js';
import './intractability-labs.css';
const name = vertex => String.fromCharCode(65 + vertex);
const names = vertices => vertices.length ? vertices.map(name).join(', ') : '∅';
const truth = value => value ? 'T' : 'F';
function FormulaEditor({
  clauses,
  onApply
}) {
  const [draft, setDraft] = useState(formatFormula(clauses));
  const [error, setError] = useState('');
  function apply(text) {
    try {
      const value = parseFormula(text);
      onApply(value);
      setDraft(text);
      setError('');
    } catch (failure) {
      setError(failure.message);
    }
  }
  return <details className="intract-edit"><summary>Change the formula</summary>
    <p>One clause per line; 1 means x1, −1 means not x1. Use 1–3 clauses and 1–3 literals per clause, drawn from ±1, ±2, ±3. Whitespace separates literals. Repeated occurrences stay distinct. Changes apply only on Apply formula.</p>
    <label>Clause draft<textarea aria-label="Clause draft" value={draft} maxLength={300} rows={3} onChange={event => setDraft(event.target.value)} /></label>
    <div className="intract-controls"><button onClick={() => apply(draft)}>Apply formula</button>
      <button onClick={() => apply(formatFormula(formulaPresets.satisfiable))}>Default formula</button>
      <button onClick={() => apply(formatFormula(formulaPresets.contradictory))}>Contradictory clauses</button>
      <button onClick={() => apply(formatFormula(formulaPresets.repeated))}>Repeated literals</button>
    </div>{error && <p role="alert">{error} Applied formula is unchanged.</p>}
  </details>;
}
export function CertificateLab() {
  const [clauses, setClauses] = useState(formulaPresets.satisfiable);
  const [assignment, setAssignment] = useState([false, false, false]);
  const evaluated = evaluateFormula(clauses, assignment);
  function apply(value) {
    setClauses(value);
    setAssignment([false, false, false]);
  }
  return <section className="intract-lab" aria-label="Certificate checker" data-lab="certificate">
    <h3>A certificate flows through the clauses</h3>
    <p>Inspect which clauses change when x3 becomes true. Each line is an OR; the final gate requires every line. The switches change one candidate, not the formula.</p>
    <div className="intract-switches">{assignment.map((value, index) => <button key={index} aria-pressed={value} aria-label={`Toggle x${index + 1}`} onClick={() => setAssignment(assignment.map((old, i) => i === index ? !old : old))}><strong>x{index + 1}</strong><span>{value ? 'True' : 'False'}</span></button>)}</div>
    <div className="intract-circuit">{evaluated.rows.map((row, index) => <div className="intract-clause" key={index}>
      <span className="intract-clause-name">C{index + 1}</span>
      <div className="intract-literals">{row.literals.map((literal, i) => <span className={row.values[i] ? 'is-true' : 'is-false'} key={i}>{literalName(literal)}<small>{truth(row.values[i])}</small></span>)}</div>
      <span className="intract-or">OR →</span><strong>{truth(row.satisfied)}</strong>
    </div>)}</div>
    <p className="intract-result" aria-live="polite">AND of all clauses → <strong>{evaluated.satisfied ? 'Accepted certificate' : 'Rejected certificate'}</strong>. {evaluated.satisfied ? 'This assignment proves that a solution exists.' : 'This assignment fails. Other assignments may still work.'}</p>
    <button onClick={() => setAssignment([false, false, false])}>Reset assignment</button>
    <details><summary>Compare all eight certificates</summary>
      <p>This finite search examines all assignments to the browser's three variables, including unused variables. Its result concerns this tiny instance; it proves no general lower bound.</p>
      <LessonTable caption="Every Boolean assignment to x1, x2, x3" headers={['x1 x2 x3', 'Clause results', 'Accepted?']} rows={truthTable(clauses).map(row => [row.assignment.map(truth).join(' '), row.rows.map(item => truth(item.satisfied)).join(' '), row.satisfied ? 'Yes' : 'No'])} />
    </details>
    <FormulaEditor clauses={clauses} onApply={apply} />
  </section>;
}
export function ReductionFlowFigure() {
  return <figure className="intract-figure">
    <div className="intract-oracle-boundary"><strong>Constructed solver for problem A</strong>
      <div className="intract-flow"><div><b>A input x</b><span>e.g. a formula</span></div><span aria-hidden="true">→</span><div><b>Polynomial converter f</b><span>graph + target k</span></div><span aria-hidden="true">→</span><div><b>Hypothetical B solver</b><span>yes / no</span></div></div>
      <p>Answer for A = answer for B on f(x)</p>
    </div>
    <figcaption>The conversion arrow A ≤p B points toward the solver you would use. A fast B solver would make A fast. A fast A solver alone provides no B solver.</figcaption>
  </figure>;
}
function occurrenceEdgePath(first, second) {
  if (second.clause - first.clause < 2) {
    return `M ${first.x} ${first.y} L ${second.x} ${second.y}`;
  }
  // First-to-third-clause edges pass below the middle group instead of through its nodes.
  const routeY = 214 + (first.occurrence * 3 + second.occurrence) * 4;
  return `M ${first.x} ${first.y} L 111 ${first.y} L 111 ${routeY} L 223 ${routeY} L 223 ${second.y} L ${second.x} ${second.y}`;
}
function OccurrenceGraph({
  clauses,
  selected
}) {
  const state = inspectSelection(clauses, selected);
  const byId = new Map(state.graph.vertices.map(vertex => [vertex.id, vertex]));
  return <div className="intract-svg-scroll" tabIndex={0} role="region" aria-label="Literal occurrence graph; equivalent selections and pair checks follow">
    <svg viewBox="0 0 335 265" role="img" aria-label="Vertices grouped by clause; compatibility edges join different clauses without opposite literals">
      {clauses.map((_, clause) => <text key={clause} x={55 + clause * 112} y="19" textAnchor="middle" className="intract-svg-heading">Clause {clause + 1}</text>)}
      {state.graph.edges.map(([from, to]) => {
        const a = byId.get(from);
        const b = byId.get(to);
        return <path key={`${from}-${to}`} d={occurrenceEdgePath(a, b)} fill="none" className="intract-edge" />;
      })}
      {state.pairs.map(pair => <path key={`${pair.first.id}-${pair.second.id}`} d={occurrenceEdgePath(pair.first, pair.second)} fill="none" className={pair.compatible ? 'intract-selected-edge' : 'intract-conflict'} />)}
      {state.graph.vertices.map(vertex => <g key={vertex.id}><rect x={vertex.x - 28} y={vertex.y - 18} width="56" height="36" rx="5" className={selected.includes(vertex.id) ? 'intract-selected-node' : 'intract-node'} /><text x={vertex.x} y={vertex.y + 5} textAnchor="middle">{literalName(vertex.literal)}</text><title>{`Clause ${vertex.clause + 1}, occurrence ${vertex.occurrence + 1}: ${literalName(vertex.literal)}`}</title></g>)}
    </svg>
  </div>;
}
export function ReductionLab() {
  const [clauses, setClauses] = useState(formulaPresets.satisfiable);
  const [selected, setSelected] = useState([null, null, null]);
  const result = inspectSelection(clauses, selected);
  const [oracle, setOracle] = useState(undefined);
  function apply(value) {
    setClauses(value);
    setSelected(value.map(() => null));
    setOracle(undefined);
  }
  return <section className="intract-lab" aria-label="SAT to clique reduction" data-lab="reduction">
    <h3>Choose one compatible occurrence from each clause</h3>
    <p>Each rectangle is a literal occurrence. Faint lines are graph edges. A green selected connection exists; a red dashed connection is forbidden because the two literals contradict. There are no edges within a clause. Connections from clause 1 to clause 3 bend below the middle group; crossings are not extra vertices. Choose below; repeated names remain separate vertices.</p>
    <OccurrenceGraph clauses={clauses} selected={selected} />
    <div className="intract-choice-columns">{clauses.map((clause, c) => <fieldset key={c}><legend>Clause {c + 1}</legend>{clause.map((literal, occurrence) => <button key={occurrence} aria-pressed={selected[c] === `${c}:${occurrence}`} onClick={() => setSelected(selected.map((id, index) => index === c ? `${c}:${occurrence}` : id))}>{literalName(literal)} <small>occurrence {occurrence + 1}</small></button>)}</fieldset>)}</div>
    <p className="intract-result" aria-live="polite">Target: {clauses.length} mutually connected vertices. {result.clique ? `Valid clique → recovered assignment ${result.assignment.map(truth).join(', ')} for x1, x2, x3. Every clause is true.` : result.complete ? 'This selection is not a clique. It does not settle whether a different selection works.' : 'Choose one occurrence in every clause.'}</p>
    {result.pairs.length > 0 && <LessonTable caption="Selected pair compatibility" headers={['Occurrence pair', 'Edge / reason']} rows={result.pairs.map(pair => [`C${pair.first.clause + 1}.${pair.first.occurrence + 1} ${literalName(pair.first.literal)} ↔ C${pair.second.clause + 1}.${pair.second.occurrence + 1} ${literalName(pair.second.literal)}`, pair.compatible ? 'Edge exists: consistent choices in different clauses' : 'No edge: a variable and its negation'])} />}
    <div className="intract-controls"><button onClick={() => {
        setSelected(clauses.map(() => null));
        setOracle(undefined);
      }}>Clear choices</button><button onClick={() => {
        const answer = firstCliqueChoices(clauses);
        setOracle(answer);
        if (answer !== null) setSelected(answer);
      }}>Search this tiny graph</button></div>
    {oracle !== undefined && <p role="status">{oracle === null ? 'Exhaustive finite search found no target clique; this formula is unsatisfiable.' : 'Exhaustive finite search found the displayed clique.'} This bounded search is not an efficient general clique algorithm.</p>}
    <FormulaEditor clauses={clauses} onApply={apply} />
  </section>;
}
export function EncodingFigure() {
  const [exponent, setExponent] = useState(10);
  const result = encodingCounts(exponent);
  return <figure className="intract-figure" data-lab="encoding">
    <h3>A short number can request a very long table</h3>
    <label>Target T = 2 to this power <select aria-label="Target exponent" value={exponent} onChange={event => setExponent(Number(event.target.value))}>{[4, 10, 20, 30, 40].map(value => <option key={value} value={value}>{value}</option>)}</select></label>
    <div className="intract-encoding"><div><span>Input digits for T</span><code>{result.binary}</code><strong>{result.bits} binary digits</strong></div><div><span>DP array indices 0…T</span><strong>{Number(result.slots).toLocaleString('en-US')} slots</strong><span>T = {Number(result.target).toLocaleString('en-US')}</span></div></div>
    <figcaption>These are exact integer counts, computed with BigInt; no large table is allocated. Doubling T adds one binary digit but roughly doubles the array. The full input also includes every item, its encoding and delimiters.</figcaption>
  </figure>;
}
const coverPositions = [[55, 50], [160, 30], [270, 65], [265, 170], [160, 205], [55, 170]];
export function CoverTradeoffLab() {
  const [edges, setEdges] = useState(coverPresets.trianglePath);
  const [draft, setDraft] = useState(formatCoverEdges(coverPresets.trianglePath));
  const [selected, setSelected] = useState([]);
  const [budget, setBudget] = useState(2);
  const [error, setError] = useState('');
  const approximation = matchingCover(edges);
  const exact = exactCover(edges, budget);
  const optimum = minimumCover(edges);
  const valid = isCover(edges, selected);
  function apply(text) {
    try {
      const graph = parseCoverEdges(text);
      setEdges(graph);
      setDraft(text);
      setSelected([]);
      setBudget(2);
      setError('');
    } catch (failure) {
      setError(failure.message);
    }
  }
  return <section className="intract-lab" aria-label="Vertex cover tradeoffs" data-lab="cover">
    <h3>Separate a feasible answer from a proof of its quality</h3>
    <p>Select vertices so every edge touches at least one. Thick amber edges form a disjoint matching from a deterministic scan of the applied edge order. Each forces a different vertex into any cover. Green vertices are your selected set.</p>
    <div className="intract-svg-scroll" tabIndex={0} role="region" aria-label="Cover graph; edge coverage table follows"><svg viewBox="0 0 325 245" role="img" aria-label="Selected vertices and disjoint matching edges in the applied graph">
      {edges.map(([u, v], index) => <line key={`${u}-${v}`} x1={coverPositions[u][0]} y1={coverPositions[u][1]} x2={coverPositions[v][0]} y2={coverPositions[v][1]} className={approximation.matching.includes(index) ? 'intract-matching-edge' : selected.includes(u) || selected.includes(v) ? 'intract-selected-edge' : 'intract-edge'} />)}
      {coverPositions.map(([x, y], vertex) => <g key={vertex}><circle cx={x} cy={y} r="21" className={selected.includes(vertex) ? 'intract-selected-node' : 'intract-node'} /><text x={x} y={y + 6} textAnchor="middle">{name(vertex)}</text><title>{name(vertex)}: {selected.includes(vertex) ? 'selected' : 'not selected'}</title></g>)}
    </svg></div>
    <div className="intract-controls">{coverPositions.map((_, vertex) => <button key={vertex} aria-label={`Toggle vertex ${name(vertex)}`} aria-pressed={selected.includes(vertex)} onClick={() => setSelected(selected.includes(vertex) ? selected.filter(value => value !== vertex) : [...selected, vertex].sort())}>{name(vertex)}</button>)}</div>
    <p className="intract-result" aria-live="polite">Selected: {names(selected)} ({selected.length}). {valid ? 'Every edge is covered.' : `${edges.filter(([u, v]) => !selected.includes(u) && !selected.includes(v)).length} edges remain uncovered.`}</p>
    <div className="intract-bounds"><div><span>Matching lower bound</span><strong>{approximation.matching.length} ≤ OPT</strong></div><div><span>Your feasible upper bound</span><strong>{valid ? `OPT ≤ ${selected.length}` : 'Not yet a cover'}</strong></div></div>
    <p>Matching edges: {approximation.matching.length ? approximation.matching.map(index => edges[index].map(name).join('–')).join(', ') : 'none'}. Their endpoints give a cover of size {approximation.selected.length}, at most twice the optimum.</p>
    <div className="intract-controls"><button onClick={() => setSelected([...approximation.selected].sort())}>Use matching endpoints</button><button onClick={() => setSelected([])}>Clear selected vertices</button></div>
    <details><summary>Inspect every edge's coverage</summary><LessonTable caption="Applied edges and cover certificate" headers={['Edge', 'In disjoint matching?', 'Covered by selected set?']} rows={edges.length ? edges.map(([u, v], index) => [`${name(u)}–${name(v)}`, approximation.matching.includes(index) ? 'Yes' : 'No', selected.includes(u) || selected.includes(v) ? 'Yes' : 'No']) : [['No edges', 'Empty matching', 'Vacuously covered']]} /></details>
    <div className="intract-budget"><h4>Ask an exact, budgeted question</h4><label>Maximum selected vertices k <select aria-label="Cover budget" value={budget} onChange={event => setBudget(Number(event.target.value))}>{[0, 1, 2, 3, 4, 5, 6].map(value => <option key={value}>{value}</option>)}</select></label>
      <p aria-live="polite">{exact.selected === null ? `No cover of size at most ${budget}.` : `A cover within budget: ${names(exact.selected)}.`} The edge-branch search visited {exact.events.length} calls.</p>
      <button disabled={exact.selected === null} onClick={() => setSelected([...exact.selected].sort())}>Use exact budget answer</button>
      <details><summary>See the finite branch tree</summary><ol className="intract-branch">{exact.events.map((event, index) => <li key={index} style={{
            '--branch-depth': event.depth
          }}><code>{names(event.selected)}</code> · {event.left} left · {event.remaining.length} uncovered edges {event.remaining.length === 0 ? '→ accept' : event.left === 0 ? '→ reject' : `→ branch on ${event.remaining[0].map(name).join('–')}`}</li>)}</ol><p>Indentation is recursion depth. Both endpoints are explored when needed; a successful branch ends this decision search. Call counts are exact for this code and input order, not timings.</p></details>
      <details><summary>Reveal this tiny graph's optimum</summary><p>Minimum cover size {optimum.length}; one witness is {names(optimum)}. The display obtains it by increasing the budget and running complete bounded search. It is practical here because there are only six vertices.</p></details>
    </div>
    <details className="intract-edit"><summary>Change the graph</summary><p>Fixed vertices A–F, at most 15 distinct undirected edges. Duplicate orientations collapse; self-loops are rejected. Empty input keeps six isolated vertices. Edits apply only on Apply graph.</p>
      <label>Edge draft<textarea aria-label="Cover edge draft" maxLength={400} rows={5} value={draft} onChange={event => setDraft(event.target.value)} /></label><div className="intract-controls"><button onClick={() => apply(draft)}>Apply graph</button><button onClick={() => apply(formatCoverEdges(coverPresets.trianglePath))}>Default graph</button><button onClick={() => apply(formatCoverEdges(coverPresets.star))}>Star graph</button><button onClick={() => apply('')}>Empty graph</button></div>{error && <p role="alert">{error} Applied graph is unchanged.</p>}
    </details>
  </section>;
}
export function PathRestrictionFigure() {
  return <figure className="intract-figure"><div className="intract-path">{[5, 1, 6, 8, 4].map((weight, index) => <div key={index}><span>position {index}</span><strong className={index % 2 === 0 ? 'is-chosen' : ''}>{weight}</strong></div>)}</div><figcaption>Only neighbors conflict. Choosing positions 0,2,4 gives weight 5+6+4=15. The path lets a prefix summarize every future conflict by its final boundary; arbitrary extra conflict edges invalidate that summary.</figcaption></figure>;
}
