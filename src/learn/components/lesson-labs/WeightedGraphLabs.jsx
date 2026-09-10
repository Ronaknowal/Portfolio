import { useId, useMemo, useState } from 'react';
import { GRAPH_LABELS, ROUTE_TEXT, NEGATIVE_TEXT, FOREST_TEXT, DEPENDENCY_TEXT, DEPENDENCY_CYCLE_TEXT, JOB_DURATIONS, parseWeightedEdges, dijkstraTrace, bellmanFordTrace, recoverRoute, spanningForestTrace, dependencyState, criticalSchedule } from '../../data/weighted-graph-models.js';
import './weighted-graph-labs.css';
const label = vertex => GRAPH_LABELS[vertex];
const cost = value => value === Infinity ? '∞' : value === -Infinity ? '−∞' : String(value);
const positions = [[45, 40], [215, 40], [45, 185], [215, 185], [45, 330], [215, 330]];
function GraphDrawing({
  edges,
  directed = true,
  selected = [],
  current = null,
  vertexStates = [],
  title = 'Weighted graph',
  weighted = true,
  count = 6
}) {
  const arrow = useId().replace(/:/g, '');
  return <div className="weighted-graph-picture">
    <svg viewBox={`0 0 260 ${count === 3 ? 235 : 375}`} role="img" aria-label={title}>
      <defs><marker id={arrow} viewBox="0 0 8 8" refX="7" refY="4" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" orient="auto-start-reverse"><path d="M0 0L8 4L0 8Z" fill="context-stroke" /></marker></defs>
      {edges.map(edge => {
        const [x1, y1] = positions[edge.u];
        const [x2, y2] = positions[edge.v];
        const dx = x2 - x1;
        const dy = y2 - y1;
        const length = Math.hypot(dx, dy);
        const reciprocal = directed && edges.some(other => other.u === edge.v && other.v === edge.u);
        const bend = reciprocal ? 29 : Math.abs(dy) > 200 ? 57 : 0;
        const cx = (x1 + x2) / 2 - dy / length * bend;
        const cy = (y1 + y2) / 2 + dx / length * bend;
        const firstLength = Math.hypot(cx - x1, cy - y1);
        const lastLength = Math.hypot(x2 - cx, y2 - cy);
        const sx = x1 + (cx - x1) / firstLength * 21;
        const sy = y1 + (cy - y1) / firstLength * 21;
        const ex = x2 - (x2 - cx) / lastLength * (directed ? 27 : 21);
        const ey = y2 - (y2 - cy) / lastLength * (directed ? 27 : 21);
        const tx = (sx + 2 * cx + ex) / 4;
        const ty = (sy + 2 * cy + ey) / 4;
        const className = edge.id === current ? 'is-current' : selected.includes(edge.id) ? 'is-selected' : '';
        return <g key={edge.id} className={className}>
          <path className="weighted-graph-edge" d={`M${sx},${sy} Q${cx},${cy} ${ex},${ey}`} markerEnd={directed ? `url(#${arrow})` : undefined} />
          {weighted && <g className="weighted-graph-weight"><rect x={tx - 16} y={ty - 11} width="32" height="22" rx="3" /><text x={tx} y={ty + 4}>{edge.weight}</text></g>}
        </g>;
      })}
      {positions.slice(0, count).map(([x, y], vertex) => <g key={vertex} className={`weighted-graph-vertex ${vertexStates[vertex] || ''}`}>
        <title>{label(vertex)}{vertexStates[vertex] ? `: ${vertexStates[vertex]}` : ''}</title><circle cx={x} cy={y} r="20" /><text x={x} y={y + 5}>{label(vertex)}</text>
      </g>)}
    </svg>
  </div>;
}
function EdgeTable({
  edges,
  directed = true,
  weighted = true
}) {
  return <details><summary>Read the exact edge list</summary><div className="weighted-graph-scroll"><table><thead><tr><th>Edge ID</th><th>Connection</th>{weighted && <th>Weight</th>}</tr></thead><tbody>{edges.map(edge => <tr key={edge.id}><td>{edge.id}</td><td>{label(edge.u)} {directed ? '→' : '—'} {label(edge.v)}</td>{weighted && <td>{edge.weight}</td>}</tr>)}</tbody></table>{!edges.length && <p>No edges. All six vertices still exist.</p>}</div></details>;
}
function Steps({
  step,
  length,
  onChange
}) {
  return <div className="weighted-graph-controls"><button type="button" onClick={() => onChange(step - 1)} disabled={step === 0}>Previous</button><span>State {step + 1} / {length}</span><button type="button" onClick={() => onChange(step + 1)} disabled={step === length - 1}>Next step</button><button type="button" onClick={() => onChange(0)} disabled={step === 0}>Restart trace</button></div>;
}
function VertexSelect({
  title,
  value,
  onChange
}) {
  return <label>{title}<select aria-label={title} value={value} onChange={event => onChange(Number(event.target.value))}>{GRAPH_LABELS.map((name, vertex) => <option key={name} value={vertex}>{name}</option>)}</select></label>;
}
function EdgeEditor({
  draft,
  setDraft,
  onApply,
  error,
  weighted = true,
  nonnegative = false,
  directed = true,
  onReset
}) {
  return <details className="weighted-graph-editor"><summary>Edit the graph</summary>
    <p>A–F always exist. One {weighted ? 'source, destination and integer weight' : 'prerequisite and dependent'} per line; at most 12 edges. {weighted ? `Weights ${nonnegative ? '0' : '−25'}…25. ` : ''}No self-loops or duplicate {directed ? 'ordered' : 'unordered'} pairs in this bounded editor. {directed ? 'Opposite directions may coexist.' : 'Each line is one undirected edge.'} Drafts apply together and restart the investigation.</p>
    <form onSubmit={event => {
      event.preventDefault();
      onApply();
    }}><label>Edge draft<textarea aria-label="Edge draft" rows="7" value={draft} onChange={event => setDraft(event.target.value)} spellCheck="false" maxLength={1000} /></label>
      <div className="weighted-graph-controls"><button type="submit">Apply graph</button><button type="button" onClick={onReset}>Reset example</button></div>
      {error && <p className="weighted-graph-error" role="alert">{error} The applied graph is unchanged.</p>}
    </form>
  </details>;
}
function useGraphEditor(initial, options) {
  const [draft, setDraft] = useState(initial);
  const [edges, setEdges] = useState(() => parseWeightedEdges(initial, options));
  const [error, setError] = useState('');
  const apply = (text = draft) => {
    try {
      setEdges(parseWeightedEdges(text, options));
      setError('');
      return true;
    } catch (failure) {
      setError(failure.message);
      return false;
    }
  };
  const reset = () => {
    setDraft(initial);
    apply(initial);
  };
  return {
    draft,
    setDraft,
    edges,
    error,
    apply,
    reset
  };
}
export function DijkstraFrontierLab() {
  const editor = useGraphEditor(ROUTE_TEXT, {
    nonnegative: true
  });
  const [source, setSource] = useState(0);
  const [target, setTarget] = useState(4);
  const [step, setStep] = useState(0);
  const trace = useMemo(() => dijkstraTrace(6, editor.edges, source), [editor.edges, source]);
  const state = trace[Math.min(step, trace.length - 1)];
  const route = state.settled[target] ? recoverRoute(state.parents, source, target) : null;
  return <section className="weighted-graph-lab" aria-label="Dijkstra priority frontier investigation">
    <header><span className="weighted-graph-eyebrow">Investigate · tentative versus final</span><h3>One best distance, several physical entries</h3><p>Predict when B becomes final. Follow A→B10 against A→C1→B1, then continue far enough to see old entries discarded.</p></header>
    <div className="weighted-graph-controls"><VertexSelect title="Source" value={source} onChange={value => {
        setSource(value);
        setStep(0);
      }} /><VertexSelect title="Route target" value={target} onChange={setTarget} /></div>
    <div className="weighted-graph-split">
      <GraphDrawing edges={editor.edges} current={state.edgeId} selected={route?.edgeIds || []} vertexStates={state.settled.map(done => done ? 'final' : '')} title="Directed costs; amber current edge, green finalized route" />
      <div><h4>Physical priority entries</h4><p className="weighted-graph-small">Priority-ordered view; actual heap storage is not sorted. Equal costs use increasing ticket numbers.</p><ol className="weighted-graph-frontier">{state.queue.map(entry => <li key={entry.ticket} className={entry.distance !== state.distances[entry.vertex] ? 'is-stale' : ''}><strong>{label(entry.vertex)} · {entry.distance}</strong><span>ticket {entry.ticket} · {entry.distance !== state.distances[entry.vertex] ? 'stale' : 'current'}</span></li>)}</ol>{!state.queue.length && <p>The physical queue is empty.</p>}
        <table><thead><tr><th>Vertex</th><th>Best</th><th>Parent</th><th>Status</th></tr></thead><tbody>{GRAPH_LABELS.map((name, vertex) => <tr key={name}><th>{name}</th><td>{cost(state.distances[vertex])}</td><td>{state.parents[vertex] ? label(state.parents[vertex].vertex) : '—'}</td><td>{state.settled[vertex] ? 'final' : state.distances[vertex] < Infinity ? 'tentative' : 'unknown'}</td></tr>)}</tbody></table>
      </div>
    </div>
    <p className="weighted-graph-event" role="status">{state.message}</p>
    <Steps step={step} length={trace.length} onChange={setStep} />
    <p><strong>Selected target {label(target)}:</strong> {route ? `${route.vertices.map(label).join(' → ')}; cost ${state.distances[target]}. Green edges recover this finalized route.` : state.kind === 'done' ? 'unreachable from this source.' : 'no finalized route yet. A tentative distance is still a candidate.'}</p>
    <p className="weighted-graph-small">Arrow direction matters. Line lengths are layout only; printed weights determine costs. Amber marks the inspected edge; green outlines mark finalized vertices. The status table repeats that meaning in words.</p>
    <EdgeTable edges={editor.edges} />
    <EdgeEditor {...editor} nonnegative onApply={() => {
      if (editor.apply()) setStep(0);
    }} onReset={() => {
      editor.reset();
      setStep(0);
      setSource(0);
      setTarget(4);
    }} />
    <details><summary>Try a changed contract</summary><p>Make A→B cost 0 and predict its extraction. Choose F as source to test unreachable vertices. Then attempt a negative edge: input validation rejects a theorem the model can no longer promise.</p></details>
  </section>;
}
export function BellmanFordPassLab() {
  const editor = useGraphEditor(NEGATIVE_TEXT, {});
  const [source, setSource] = useState(0);
  const [step, setStep] = useState(0);
  const trace = useMemo(() => bellmanFordTrace(6, editor.edges, source), [editor.edges, source]);
  const state = trace[step];
  return <section className="weighted-graph-lab" aria-label="Bellman Ford edge budget investigation">
    <header><span className="weighted-graph-eyebrow">Investigate · generations and negative cycles</span><h3>Only the previous row may supply a candidate</h3><p>B→C→B costs −3. E→F→E costs −2. Predict which cycle can affect source A, and whether downstream D belongs to the cycle itself.</p></header>
    <VertexSelect title="Source" value={source} onChange={value => {
      setSource(value);
      setStep(0);
    }} />
    <div className="weighted-graph-split"><GraphDrawing edges={editor.edges} vertexStates={state.distances.map(value => value === -Infinity ? 'unbounded' : value === Infinity ? 'unreached' : state.pass === 6 ? 'finite' : 'candidate')} title="Directed graph with candidate, unreached and final negative-cycle classifications" />
      <div><h4>Distance generations</h4><div className="weighted-graph-scroll weighted-graph-generations" tabIndex="0" aria-label="Distance generations, scroll horizontally if needed"><table><thead><tr><th>Budget</th>{GRAPH_LABELS.map(name => <th key={name}>{name}</th>)}</tr></thead><tbody>{trace.slice(0, step + 1).map(row => <tr key={row.pass} className={row.pass === state.pass ? 'is-current-row' : ''}><th>{row.pass === 6 ? 'detect' : `≤${row.pass}`}</th>{row.distances.map((value, vertex) => <td key={vertex}>{cost(value)}</td>)}</tr>)}</tbody></table></div>
        <p className="weighted-graph-small">≤k means at most k edges. Before detection, a number is a budgeted candidate and ∞ means no route within that budget. The detection row classifies unrestricted walks: ∞ is unreachable; −∞ means arbitrarily low costs, not an attained path length.</p>
        {state.previous && <details><summary>Inspect this pass's edge candidates</summary><table><thead><tr><th>Edge</th><th>Previous + weight</th><th>Candidate</th></tr></thead><tbody>{state.candidates.map(candidate => {
                const edge = editor.edges[candidate.edgeId];
                return <tr key={edge.id}><td>{label(edge.u)}→{label(edge.v)}</td><td>{cost(candidate.fromCost)} + ({edge.weight})</td><td>{cost(candidate.cost)}</td></tr>;
              })}</tbody></table><p>Each destination keeps the minimum of its carried value and every incoming candidate. Candidates do not read newly written cells.</p></details>}
      </div></div>
    <p className="weighted-graph-small">Dashed rings mark unbounded-below destinations after detection. Read all vertex values in the generation table; diagram coordinates never determine costs.</p>
    <p className="weighted-graph-event" role="status">{state.message}</p><Steps step={step} length={trace.length} onChange={setStep} />
    {state.pass === 6 && <p><strong>Detection seeds:</strong> {state.seeds.map(label).join(', ') || 'none'}. <strong>Downstream affected region:</strong> {state.affected.map(label).join(', ') || 'none'}. These lists do not claim that every listed vertex lies on a negative cycle.</p>}
    <EdgeTable edges={editor.edges} /><EdgeEditor {...editor} onApply={() => {
      if (editor.apply()) setStep(0);
    }} onReset={() => {
      editor.reset();
      setSource(0);
      setStep(0);
    }} />
    <details><summary>Explain the difference, then change it</summary><p>From A, E and F stay unreachable despite their negative cycle. Switch the source to E. Finally remove C→B: the negative edge B→C remains, but its reachable negative cycle disappears. Five edges suffice for a finite optimal simple route whenever no relevant negative cycle exists.</p></details>
  </section>;
}
export function SpanningForestLab() {
  const editor = useGraphEditor(FOREST_TEXT, {
    directed: false
  });
  const [method, setMethod] = useState('kruskal');
  const [step, setStep] = useState(0);
  const trace = useMemo(() => spanningForestTrace(6, editor.edges, method), [editor.edges, method]);
  const state = trace[Math.min(step, trace.length - 1)];
  const groups = [...new Set(state.components)].map(root => GRAPH_LABELS.filter((_, vertex) => state.components[vertex] === root).join(' '));
  return <section className="weighted-graph-lab" aria-label="Minimum spanning forest investigation">
    <header><span className="weighted-graph-eyebrow">Investigate · a cheapest connecting forest</span><h3>A global edge order or one growing boundary</h3><p>Predict why edge A—B of weight 4 may be rejected although it is cheaper than a later accepted edge of weight 5. Track the connection already present, not just the number.</p></header>
    <label>Growth rule<select aria-label="Growth rule" value={method} onChange={event => {
        setMethod(event.target.value);
        setStep(0);
      }}><option value="kruskal">Kruskal · global order</option><option value="prim">Prim · crossing frontier</option></select></label>
    <div className="weighted-graph-split"><GraphDrawing edges={editor.edges} directed={false} selected={state.accepted} current={state.edgeId} vertexStates={method === 'prim' ? state.visited.map(value => value ? 'inside' : '') : []} title="Undirected graph with accepted edges green and current candidate amber" />
      <div><h4>Accepted connections</h4><p className="weighted-graph-total">{state.total}<span>total edge weight</span></p><p>{state.accepted.length} edges · {groups.length} {groups.length === 1 ? 'component' : 'components'}</p><ul className="weighted-graph-components">{groups.map(group => <li key={group}>{group}</li>)}</ul><h4>{method === 'kruskal' ? 'Remaining global candidates' : 'Physical frontier entries'}</h4><ol className="weighted-graph-edge-queue">{state.queue.map((id, index) => {
            const edge = editor.edges.find(item => item.id === id);
            return <li key={`${id}-${index}`}>{label(edge.u)}—{label(edge.v)} <strong>{edge.weight}</strong></li>;
          })}</ol>{!state.queue.length && <p>No queued edges.</p>}<p className="weighted-graph-small">The list is ordered for inspection. Prim entries may become internal before they are popped; the state checks them when removed.</p></div>
    </div>
    <p className="weighted-graph-event" role="status">{state.message}</p><Steps step={step} length={trace.length} onChange={setStep} />
    {method === 'prim' && <p><strong>Visited by Prim:</strong> {GRAPH_LABELS.filter((_, vertex) => state.visited[vertex]).join(', ') || 'none'}. Green vertex outlines mark these vertices across all started components.</p>}
    <p>Green edges are accepted; amber is the current candidate, including a rejected one. Components are defined by accepted edges. The complete default forest costs 13 with either method.</p>
    <EdgeTable edges={editor.edges} directed={false} /><EdgeEditor {...editor} directed={false} onApply={() => {
      if (editor.apply()) setStep(0);
    }} onReset={() => {
      editor.reset();
      setMethod('kruskal');
      setStep(0);
    }} />
    <details><summary>Make the assumption change visible</summary><p>Delete all edges touching F; expect two components and four accepted edges. Try tied weights or a negative edge: the spanning-forest objective remains valid. Explain why a negative weight that breaks Dijkstra's proof does not break the cut exchange argument.</p></details>
  </section>;
}
export function TopologicalDependencyLab() {
  const editor = useGraphEditor(DEPENDENCY_TEXT, {
    weighted: false
  });
  const [order, setOrder] = useState([]);
  const state = dependencyState(6, editor.edges, order);
  const cycleEdges = state.cycle ? state.cycle.slice(1).map((vertex, index) => editor.edges.find(edge => edge.u === state.cycle[index] && edge.v === vertex).id) : [];
  return <section className="weighted-graph-lab" aria-label="Topological ready frontier investigation">
    <header><span className="weighted-graph-eyebrow">Investigate · readiness and blocked work</span><h3>You choose among jobs whose prerequisites are gone</h3><p>Each arrow means “finish the source before the destination.” Emit B before A, then explain why C is still waiting. A ready choice changes the valid order without changing feasibility.</p></header>
    <div className="weighted-graph-split"><GraphDrawing edges={editor.edges} weighted={false} selected={cycleEdges} vertexStates={GRAPH_LABELS.map((_, vertex) => order.includes(vertex) ? 'emitted' : state.ready.includes(vertex) ? 'ready' : 'waiting')} title="Dependency graph, with any returned directed cycle highlighted" />
      <div><h4>Ready now</h4><div className="weighted-graph-ready">{state.ready.map(vertex => <button key={vertex} type="button" onClick={() => setOrder([...order, vertex])}>Emit {label(vertex)}</button>)}</div>{!state.ready.length && <p>{state.complete ? 'Every job has been emitted.' : 'No remaining job is ready.'}</p>}<h4>Output order</h4><ol className="weighted-graph-order">{order.map(vertex => <li key={vertex}>{label(vertex)}</li>)}</ol>{!order.length && <p>Empty output.</p>}<table><thead><tr><th>Job</th><th>Remaining prerequisites</th><th>Status</th></tr></thead><tbody>{GRAPH_LABELS.map((name, vertex) => <tr key={name}><th>{name}</th><td>{state.indegrees[vertex]}</td><td>{order.includes(vertex) ? 'emitted' : state.ready.includes(vertex) ? 'ready' : 'waiting'}</td></tr>)}</tbody></table></div></div>
    <p className="weighted-graph-small">Green outlines mark ready jobs; filled nodes have been emitted. Any highlighted edges form the returned cycle witness. The status table identifies each state in words.</p>
    <p className="weighted-graph-event" role="status">{state.complete ? 'Complete: every dependency points forward in the output.' : state.cycle ? `Cycle witness: ${state.cycle.map(label).join(' → ')}. Residual vertices: ${state.remaining.map(label).join(', ')}. Residual does not mean “on this cycle.”` : `Choose any of ${state.ready.map(label).join(', ')}; each currently has zero remaining indegree.`}</p>
    <div className="weighted-graph-controls"><button type="button" disabled={!order.length} onClick={() => setOrder(order.slice(0, -1))}>Undo emission</button><button type="button" disabled={!order.length} onClick={() => setOrder([])}>Restart order</button><button type="button" onClick={() => {
        editor.setDraft(DEPENDENCY_CYCLE_TEXT);
        editor.apply(DEPENDENCY_CYCLE_TEXT);
        setOrder([]);
      }}>Load cycle example</button></div>
    <EdgeTable edges={editor.edges} weighted={false} /><EdgeEditor {...editor} weighted={false} onApply={() => {
      if (editor.apply()) setOrder([]);
    }} onReset={() => {
      editor.reset();
      setOrder([]);
    }} />
    <details><summary>Why does the blocked example matter?</summary><p>With E→C added, C→E→C is a cycle and F is blocked downstream. F need not lie on any cycle. The native DFS program returns actual active-stack vertices, rather than labelling the entire Kahn residual a cycle.</p></details>
  </section>;
}
export function RouteVersusNetworkFigure() {
  const edges = [{
    id: 0,
    u: 0,
    v: 1,
    weight: 2
  }, {
    id: 1,
    u: 1,
    v: 2,
    weight: 2
  }, {
    id: 2,
    u: 0,
    v: 2,
    weight: 3
  }];
  const mst = spanningForestTrace(3, edges).at(-1);
  const routes = dijkstraTrace(3, [...edges, ...edges.map(edge => ({
    ...edge,
    id: edge.id + 3,
    u: edge.v,
    v: edge.u
  }))], 0).at(-1);
  return <figure className="weighted-graph-figure weighted-graph-objectives"><div><h4>Pay once for a connecting network</h4><GraphDrawing edges={edges} directed={false} count={3} selected={mst.accepted} title="Minimum spanning tree: AB2 and BC2; AC3 unused" /><p>Install A—B and B—C: <strong>total {mst.total}</strong>. A→C follows two edges, costing 4.</p></div><div><h4>Minimize routes starting at A</h4><GraphDrawing edges={edges} directed={false} count={3} selected={[0, 2]} title="Shortest-path tree from A: AB2 and AC3" /><p>Keep A—B and A—C: <strong>total 5</strong>. Best A→C costs <strong>{routes.distances[2]}</strong>.</p></div><figcaption>Same original graph and weights, different optimization questions. The selected edge sets were checked by the local models; tree weight and source-to-target distance are different quantities.</figcaption></figure>;
}
export function CriticalPathFigure() {
  const edges = parseWeightedEdges(DEPENDENCY_TEXT, {
    weighted: false
  });
  const schedule = criticalSchedule(6, edges, JOB_DURATIONS);
  return <figure className="weighted-graph-figure"><h4>An unlimited-capacity build schedule</h4><p>All jobs are available at time 0 except for dependencies; durations are fixed. B→D→E→F is a critical chain of total 11.</p><div className="weighted-graph-timeline">{GRAPH_LABELS.map((name, vertex) => <div className="weighted-graph-time-row" key={name}><span>{name}</span><div className="weighted-graph-time-track"><span className={schedule.chain.includes(vertex) ? 'is-critical' : ''} style={{
            marginLeft: `${schedule.starts[vertex] / schedule.makespan * 100}%`,
            width: `${JOB_DURATIONS[vertex] / schedule.makespan * 100}%`
          }} aria-label={`${name}: start ${schedule.starts[vertex]}, finish ${schedule.finishes[vertex]}`} /></div><span>{schedule.starts[vertex]}–{schedule.finishes[vertex]}</span></div>)}</div><figcaption>Exact modeled time, not measured performance. The bars share a 0–11 time axis; amber bars mark one longest prerequisite chain. A and B overlap. C finishes at 7, but E must wait until D finishes at 8. The Python program below returns the same starts, finishes and chain.</figcaption></figure>;
}
