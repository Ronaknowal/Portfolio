import { useId, useState } from 'react';
import { GRAPH_VERTICES, GRAPH_EDGES, GRID_DEFAULT_WALLS, GRID_DEFAULT_SOURCES, GRID_DEFAULT_TARGET, parseGraphEdges, buildGraph, graphLayout, graphTraversalTrace, graphPath, directedCycleExamples, gridWavefrontTrace } from '../../data/graph-traversal-models.js';
import './graph-traversal-labs.css';
const edgeText = edges => edges.map(edge => edge.join('-')).join(', ');
const cellKey = cell => cell.join(',');
function GraphInvestigation({
  name,
  eyebrow,
  title,
  children
}) {
  const id = useId();
  return <section className="graph-traversal-lab" data-lab={name} aria-labelledby={id}><p className="graph-lab-eyebrow">{eyebrow}</p><h3 id={id}>{title}</h3>{children}</section>;
}
function GraphTraceControls({
  trace,
  step,
  setStep,
  reset
}) {
  return <div className="graph-trace-controls"><button type="button" onClick={() => setStep(step - 1)} disabled={step === 0}>Back</button><span>Step {step + 1} / {trace.length}</span><button type="button" onClick={() => setStep(step + 1)} disabled={step === trace.length - 1}>{step === trace.length - 1 ? 'Trace complete' : 'Next: ' + trace[step + 1].phase}</button><button type="button" onClick={reset}>Reset investigation</button></div>;
}
function GraphFeedback({
  state
}) {
  return <div className="graph-feedback" role="status"><strong>{state.phase}.</strong> {state.note}</div>;
}
function GraphPicture({
  graph,
  state = {},
  selected = null,
  path = null,
  compact = false,
  caption = 'Vertex and edge relationships'
}) {
  const marker = useId(),
    layout = graphLayout(graph, { compact }),
    positions = Object.fromEntries(layout.nodes.map(node => [node.vertex, node]));
  const discovered = new Set(state.discovered || []),
    finished = new Set(state.finished || []),
    active = new Set((state.frames || []).map(frame => frame.vertex));
  const match = (edge, from, to) => edge && (edge[0] === from && edge[1] === to || !graph.directed && edge[0] === to && edge[1] === from);
  const pathPairs = path?.slice(1).map((to, index) => [path[index], to]) || [];
  return <div className="graph-picture"><p className="graph-diagram-heading">{caption} · {graph.directed ? 'arrows permit one direction' : 'lines permit both directions'}</p><div className="graph-diagram-scroll" tabIndex={0} role="region" aria-label={caption + '; scroll horizontally to inspect the full graph'}><svg viewBox={`0 0 ${layout.width} ${layout.height}`} style={{
        width: compact ? '100%' : layout.width,
        minWidth: compact ? 280 : layout.width,
        maxWidth: compact ? layout.width : undefined
      }} role="img" aria-label={`${graph.directed ? 'Directed' : 'Undirected'} graph. Vertices ${graph.vertices.join(', ')}. Edges ${graph.edges.map(edge => edge.join(graph.directed ? ' to ' : ' with ')).join('; ') || 'none'}. ${selected ? `Selected ${selected}, outgoing adjacency ${graph.adjacency[selected].join(', ') || 'none'}.` : ''}`}>
    <defs><marker id={marker} markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L9,4.5 L0,9 Z" fill="#d3c6a2" /></marker></defs>
    {layout.edges.map(edge => {
          const from = positions[edge.from],
            to = positions[edge.to],
            loop = edge.from === edge.to,
            dx = to.x - from.x,
            dy = to.y - from.y,
            length = Math.hypot(dx, dy) || 1;
          const start = [from.x + dx / length * 25, from.y + dy / length * 25],
            end = [to.x - dx / length * 27, to.y - dy / length * 27];
          const curve = edge.reciprocal ? [(from.x + to.x) / 2 - dy / length * 28, (from.y + to.y) / 2 + dx / length * 28] : null;
          const d = loop ? `M ${from.x - 17},${from.y - 17} C ${from.x - 65},${from.y - 72} ${from.x + 65},${from.y - 72} ${from.x + 17},${from.y - 17}` : curve ? `M ${start} Q ${curve} ${end}` : `M ${start} L ${end}`;
          const selectedEdge = selected && (edge.from === selected || !graph.directed && edge.to === selected),
            current = match(state.edge, edge.from, edge.to),
            onPath = pathPairs.some(pair => match(pair, edge.from, edge.to));
          return <g key={edge.from + '-' + edge.to}><path className={`graph-edge${selectedEdge ? ' is-selected' : ''}${onPath ? ' is-route' : ''}${current ? ' is-current' : ''}`} d={d} markerEnd={graph.directed ? `url(#${marker})` : undefined} />{loop && <text className="graph-loop-label" x={from.x} y={from.y - 66} textAnchor="middle">self-loop</text>}</g>;
        })}
    {layout.nodes.map(node => <g key={node.vertex} className={`graph-vertex${discovered.has(node.vertex) ? ' is-discovered' : ''}${finished.has(node.vertex) ? ' is-finished' : ''}${active.has(node.vertex) ? ' is-stack-active' : ''}${state.activeVertex === node.vertex ? ' is-current' : ''}`}>
      {selected === node.vertex && <circle className="graph-selection-ring" cx={node.x} cy={node.y} r="31" />}<circle cx={node.x} cy={node.y} r="24" /><text className="graph-vertex-name" x={node.x} y={node.y + 7} textAnchor="middle">{node.vertex}</text>
      {state.method === 'bfs' && <text className="graph-vertex-state" x={node.x} y={node.y + 44} textAnchor="middle">{state.distance[node.vertex] === null ? 'unreached' : `d = ${state.distance[node.vertex]}`}</text>}
      {state.method === 'dfs' && <text className="graph-vertex-state" x={node.x} y={node.y + 44} textAnchor="middle">{active.has(node.vertex) ? 'active' : finished.has(node.vertex) ? 'finished' : 'unreached'}</text>}
    </g>)}
  </svg></div><p className="graph-note">Placement and line length organize the drawing; they are not edge weights or geographic distances. A crossing without a labelled circle is not a vertex. {graph.directed && 'Opposite directed edges curve separately; self-loops return to the same vertex. '}{compact ? 'This small graph fits as a whole so the active and finished states can be compared together.' : 'Labels retain their native size with local scrolling.'}</p></div>;
}
export function GraphRepresentationLab() {
  const [draft, setDraft] = useState(edgeText(GRAPH_EDGES)),
    [direction, setDirection] = useState('undirected'),
    [graph, setGraph] = useState(() => buildGraph()),
    [selected, setSelected] = useState('A'),
    [error, setError] = useState('');
  const run = event => {
    event.preventDefault();
    const parsed = parseGraphEdges(draft);
    if (!parsed.valid) {
      setError(parsed.error);
      return;
    }
    setGraph(buildGraph(parsed.edges, direction === 'directed'));
    setError('');
  };
  const reset = () => {
    setDraft(edgeText(GRAPH_EDGES));
    setDirection('undirected');
    setGraph(buildGraph());
    setSelected('A');
    setError('');
  };
  return <GraphInvestigation name="graph-representation" eyebrow="ONE RELATION, THREE STORAGE VIEWS" title="Where does one edge appear?">
    <p>Predict which matrix cells A-B changes. In an undirected graph both A → B and B → A are valid moves. Turn on direction and the same text describes only A → B.</p>
    <form className="graph-controls" onSubmit={run}><label className="graph-wide-control">Edges · comma-separated, such as A-B<input value={draft} onChange={event => setDraft(event.target.value)} /></label><label>Direction<select value={direction} onChange={event => setDirection(event.target.value)}><option value="undirected">Undirected</option><option value="directed">Directed</option></select></label><button type="submit">Apply graph</button><button type="button" onClick={reset}>Reset graph</button></form>
    {error && <p className="graph-error" role="alert">{error} The active graph was kept.</p>}
    <p className="graph-note">Draft edits apply with Apply graph. Active: {graph.directed ? 'directed' : 'undirected'}, {graph.vertices.length} explicit vertices and {graph.edges.length} distinct edges. Empty edge input keeps all eight isolated vertices. Duplicate pairs collapse; a self-loop appears once as an adjacency membership and once on the matrix diagonal.</p>
    <GraphPicture graph={graph} selected={selected} />
    <div className="graph-representations"><div><h4>Adjacency lists · select a row</h4><ul className="graph-adjacency-list">{graph.vertices.map(vertex => <li key={vertex}><button type="button" aria-pressed={selected === vertex} onClick={() => setSelected(vertex)} aria-label={`Inspect outgoing neighbors of ${vertex}`}><strong>{vertex}</strong><span>→ {graph.adjacency[vertex].join(', ') || '∅'}</span></button></li>)}</ul></div><div><h4>Adjacency matrix · row → column</h4><div className="graph-table-scroll" tabIndex={0} role="region" aria-label="Adjacency matrix"><table className="graph-adjacency-matrix"><thead><tr><th scope="col">from / to</th>{graph.vertices.map(vertex => <th key={vertex} scope="col">{vertex}</th>)}</tr></thead><tbody>{graph.vertices.map((vertex, index) => <tr key={vertex} className={selected === vertex ? 'is-selected' : ''}><th scope="row">{vertex}</th>{graph.matrix[index].map((value, column) => <td key={column} className={value ? 'has-edge' : ''}>{value}</td>)}</tr>)}</tbody></table></div></div></div>
    <p className="graph-selected-adjacency">From <strong>{selected}</strong>, one step may go to <strong>{graph.adjacency[selected].join(', ') || 'no vertex'}</strong>. The selected list, matrix row and highlighted outgoing edges describe the same adjacency relation.</p>
    <details><summary>Transfer: direction, isolation and repeated edges</summary><p>Apply A-B, B-A, A-A with direction enabled. The two opposite arrows are separate directed edges and A's loop occupies diagonal cell A,A. Turn direction off: A-B and B-A collapse to one undirected edge. H remains present even though no edge mentions it. Matrix entries record existence, not parallel-edge multiplicity or weights.</p></details>
  </GraphInvestigation>;
}
export function GraphSearchLab() {
  const [method, setMethod] = useState('bfs'),
    [source, setSource] = useState('A'),
    [direction, setDirection] = useState('undirected'),
    [target, setTarget] = useState('E'),
    [trace, setTrace] = useState(() => graphTraversalTrace()),
    [step, setStep] = useState(0);
  const state = trace[step],
    path = graphPath(state, target),
    dfs = state.method === 'dfs';
  const run = event => {
    event.preventDefault();
    setTrace(graphTraversalTrace(buildGraph(GRAPH_EDGES, direction === 'directed'), source, method));
    setStep(0);
  };
  const reset = () => {
    setMethod('bfs');
    setSource('A');
    setDirection('undirected');
    setTarget('E');
    setTrace(graphTraversalTrace());
    setStep(0);
  };
  return <GraphInvestigation name="graph-search" eyebrow="DISCOVERED, PENDING AND FINISHED ARE DIFFERENT" title="What work does the frontier remember?">
    <p>Predict A's first three discoveries. BFS uses the oldest pending vertex. DFS suspends an actual frame with its next neighbor, explores one child and then resumes.</p>
    <form className="graph-controls" onSubmit={run}><label>Search<select value={method} onChange={event => setMethod(event.target.value)}><option value="bfs">BFS · FIFO queue</option><option value="dfs">DFS · call frames</option></select></label><label>Starting vertex<select value={source} onChange={event => setSource(event.target.value)}>{GRAPH_VERTICES.map(vertex => <option key={vertex}>{vertex}</option>)}</select></label><label>Edge direction<select value={direction} onChange={event => setDirection(event.target.value)}><option value="undirected">Undirected</option><option value="directed">Directed in listed order</option></select></label><button type="submit">Start selected search</button></form>
    <p className="graph-note">Search/source/direction drafts apply with Start selected search. Active: {state.method.toUpperCase()} from {state.source}, {state.graph.directed ? 'directed' : 'undirected'}. This investigation uses the lesson's fixed six-edge graph independently of edits in the representation lab. Neighbor labels are inspected in ascending order.</p>
    <GraphPicture graph={state.graph} state={state} path={path} />
    <div className="graph-search-state"><div><h4>{dfs ? 'Call stack · top frame first' : 'FIFO queue · front first'}</h4>{dfs ? <ol className="graph-frame-stack">{[...state.frames].reverse().map((frame, index) => <li key={frame.vertex}><strong>{frame.vertex}{index === 0 ? ' · TOP' : ''}</strong><span>next neighbor index {frame.nextNeighborIndex}</span><small>{state.graph.adjacency[frame.vertex][frame.nextNeighborIndex] === undefined ? 'next action: return' : `next neighbor: ${state.graph.adjacency[frame.vertex][frame.nextNeighborIndex]}`}</small></li>)}</ol> : <ol className="graph-queue">{state.queue.map((vertex, index) => <li key={vertex}><strong>{vertex}</strong><small>{index === 0 ? 'FRONT' : `d=${state.distance[vertex]}`}</small></li>)}</ol>}{!(dfs ? state.frames.length : state.queue.length) && <p className="graph-empty">No pending {dfs ? 'frames' : 'vertices'}.</p>}</div><div><h4>Discovery / entry order</h4><p className="graph-order">{state.entryOrder.join(' → ')}</p><h4>{dfs ? 'Finish / return order' : 'Finished adjacency lists'}</h4><p className="graph-order">{state.finishOrder.join(' → ') || 'none yet'}</p></div></div>
    <div className="graph-table-scroll" tabIndex={0} role="region" aria-label="Search parents and discovery state"><table className="graph-search-table"><thead><tr><th>Vertex</th><th>Parent</th><th>{dfs ? 'DFS tree depth' : 'BFS distance'}</th>{dfs && <><th>Entry time</th><th>Finish time</th></>}<th>State</th></tr></thead><tbody>{state.graph.vertices.map(vertex => <tr key={vertex}><th scope="row">{vertex}</th><td>{state.parent[vertex] ?? (vertex === state.source ? 'source' : '—')}</td><td>{(dfs ? state.depth[vertex] : state.distance[vertex]) ?? '—'}</td>{dfs && <><td>{state.entryTime[vertex] ?? '—'}</td><td>{state.finishTime[vertex] ?? '—'}</td></>}<td>{state.finished.includes(vertex) ? 'finished' : state.discovered.includes(vertex) ? 'discovered' : 'unreached'}</td></tr>)}</tbody></table></div>
    <label className="graph-route-control">Show the parent route to<select value={target} onChange={event => setTarget(event.target.value)}>{GRAPH_VERTICES.map(vertex => <option key={vertex}>{vertex}</option>)}</select></label>
    <p className="graph-route-result">{path ? `${path.join(' → ')} · ${path.length - 1} edge${path.length === 2 ? '' : 's'}. ${dfs ? 'A DFS parent route; shortest distance is not promised.' : 'A shortest edge-count route, assigned at first discovery.'}` : state.result === 'complete' ? `${target} is unreachable from ${state.source}.` : `${target} has not been discovered yet. Step before concluding it is unreachable.`}</p>
    <GraphFeedback state={state} /><GraphTraceControls trace={trace} step={step} setStep={setStep} reset={reset} />
    <details><summary>Transfer: same reachability, different routes</summary><p>Finish BFS and DFS from A with undirected edges. BFS discovers A, B, C, D, E; DFS enters A, B, D, C, E. Select C: BFS's route is A → C, while DFS's parent route is A → B → D → C. Start at F to reach only F and G, or H to reach only H. Turn on direction and start at E: the incoming D → E arrow does not permit E → D.</p></details>
    <p className="graph-note">BFS marks on discovery before enqueueing. DFS frames retain the next uninspected neighbor, matching recursive calls and finish events rather than pushing every neighbor at once. Green-filled vertices are discovered; dashed frame outlines mark active DFS calls; state labels and tables distinguish finished work. First-parent ties depend on neighbor order. Geometry is not a shortest-path cost.</p>
  </GraphInvestigation>;
}
export function GridWavefrontLab() {
  const [walls, setWalls] = useState(GRID_DEFAULT_WALLS),
    [target, setTarget] = useState(GRID_DEFAULT_TARGET),
    [sourceMode, setSourceMode] = useState('single'),
    [editing, setEditing] = useState('walls'),
    [trace, setTrace] = useState(() => gridWavefrontTrace()),
    [step, setStep] = useState(0),
    [error, setError] = useState('');
  const state = trace[step],
    blocked = new Set(state.walls.map(cellKey)),
    roots = new Set(state.sources.map(cellKey)),
    frontier = new Set(state.frontier.map(cellKey)),
    path = new Set((state.path || []).map(cellKey));
  const apply = (nextWalls, nextTarget, nextMode) => {
    const sources = nextMode === 'single' ? [GRID_DEFAULT_SOURCES[0]] : GRID_DEFAULT_SOURCES;
    setTrace(gridWavefrontTrace({
      walls: nextWalls,
      target: nextTarget,
      sources
    }));
    setStep(0);
    setWalls(nextWalls);
    setTarget(nextTarget);
    setSourceMode(nextMode);
    setError('');
  };
  const chooseCell = (row, column) => {
    const cell = [row, column],
      key = cellKey(cell);
    if (editing === 'target') {
      if (blocked.has(key)) {
        setError('Choose an open target cell, or remove its wall first.');
        return;
      }
      apply(walls, cell, sourceMode);
    } else {
      if (roots.has(key) || key === cellKey(target)) {
        setError('Keep sources and the target open. Move the target before placing a wall there.');
        return;
      }
      const nextWalls = blocked.has(key) ? walls.filter(item => cellKey(item) !== key) : [...walls, cell];
      apply(nextWalls, target, sourceMode);
    }
  };
  const chooseMode = mode => {
    const newRoots = mode === 'single' ? [GRID_DEFAULT_SOURCES[0]] : GRID_DEFAULT_SOURCES;
    apply(walls.filter(cell => !newRoots.some(source => cellKey(source) === cellKey(cell))), target, mode);
  };
  const reset = () => {
    setEditing('walls');
    apply(GRID_DEFAULT_WALLS, GRID_DEFAULT_TARGET, 'single');
  };
  return <GraphInvestigation name="grid-wavefront" eyebrow="A GRAPH WITHOUT A STORED EDGE LIST" title="Watch equal-cost distance spread one layer at a time">
    <p>Predict how many moves the lower-right target needs from the upper-left source. Each open cell is a vertex; an edge is one open up, left, right or down move.</p>
    <div className="graph-controls"><label>Sources<select value={sourceMode} onChange={event => chooseMode(event.target.value)}><option value="single">One source · (0,0)</option><option value="multiple">Two sources · (0,0) and (4,4)</option></select></label><label>Click a cell to<select value={editing} onChange={event => setEditing(event.target.value)}><option value="walls">Toggle a wall</option><option value="target">Choose the target</option></select></label><button type="button" onClick={() => apply([], target, sourceMode)}>Clear all walls</button></div>
    <p className="graph-note">Cell actions and source changes apply immediately and restart the wavefront. Switching to two sources opens (4,4) if necessary. Rows and columns start at 0. Keyboard focus and Enter/Space perform the same cell action.</p>
    {error && <p className="graph-error" role="alert">{error} The grid was kept.</p>}
    <div className="grid-wavefront-board" role="group" aria-label="Five by five grid; each cell is an edit button">{state.distance.flatMap((row, rowIndex) => row.map((distance, columnIndex) => {
        const key = cellKey([rowIndex, columnIndex]),
          wall = blocked.has(key),
          source = roots.has(key),
          isTarget = key === cellKey(state.target),
          status = wall ? 'wall' : source ? 'source' : distance === null ? state.result === 'complete' ? 'unreachable' : 'unreached' : `distance ${distance}`;
        return <button type="button" key={key} className={`grid-wavefront-cell${wall ? ' is-wall' : ''}${source ? ' is-source' : ''}${isTarget ? ' is-target' : ''}${frontier.has(key) ? ' is-frontier' : ''}${path.has(key) ? ' is-path' : ''}`} onClick={() => chooseCell(rowIndex, columnIndex)} aria-label={`Row ${rowIndex}, column ${columnIndex}: ${status}${isTarget ? ', target' : ''}; ${editing === 'walls' ? 'toggle wall' : 'choose target'}`} aria-pressed={editing === 'walls' ? wall : isTarget}><small>{rowIndex},{columnIndex}</small><strong>{wall ? '#' : distance ?? '·'}</strong><span>{source && isTarget ? 'S / T' : source ? 'S' : isTarget ? 'T' : frontier.has(key) ? 'front' : ''}</span></button>;
      }))}</div>
    <p className="grid-wavefront-legend"># = wall; · = open and not reached; S = source; T = target. Numbers count moves from the nearest source. Thick amber borders mark the frontier; dotted inner outlines and green fill mark the selected parent route.</p>
    <div className="grid-distance-result"><span>Target ({state.target.join(',')})</span><strong>{state.distance[state.target[0]][state.target[1]] === null ? state.result === 'complete' ? 'unreachable' : 'not reached yet' : `${state.distance[state.target[0]][state.target[1]]} moves`}</strong><p>{state.path ? state.path.map(cell => `(${cell.join(',')})`).join(' → ') : 'A parent path appears when the target is discovered.'}</p></div>
    <GraphFeedback state={state} /><GraphTraceControls trace={trace} step={step} setStep={setStep} reset={reset} />
    <details><summary>Transfer: multiple sources and a blocked region</summary><p>The default single-source answer is 8 moves. With two sources the default target (4,4) is itself a source, so its distance is 0. Choose target (0,4) and compare which source reaches it sooner. In single-source mode, block both (0,1) and (1,0): the source is isolated and an open target is unreachable. Clear the walls to compare with Manhattan distance. Diagonal movement, weighted terrain and counting visited cells instead of moves change the contract.</p></details>
    <p className="graph-note">Each step expands a whole equal-distance layer. Every cell is marked on first discovery; ties use sources in row/column order and neighbors up, left, right, down. Multiple equally short parent routes can exist. A displayed shortest route includes both endpoints, so it contains one more cell than moves. No Euclidean distances, diagonal edges or varying movement costs are modeled.</p>
  </GraphInvestigation>;
}
export function DirectedCycleFigure() {
  const {
    dag,
    cycle,
    dagRevisit,
    backEdge
  } = directedCycleExamples();
  const dagTrace = graphTraversalTrace(dag, 'A', 'dfs'),
    cycleTrace = graphTraversalTrace(cycle, 'A', 'dfs');
  const dagState = dagTrace.find(state => state.phase === 'Do not enter a discovered vertex again' && state.edge?.join('') === dagRevisit.join(''));
  const cycleState = cycleTrace.find(state => state.phase === 'Do not enter a discovered vertex again' && state.edge?.join('') === backEdge.join(''));
  return (
    <figure className="graph-inline-figure">
      <figcaption><strong>A visited vertex can be finished or still on the active path</strong></figcaption>
      <h4>Directed acyclic diamond · C → D reaches finished work</h4>
      <GraphPicture graph={dag} state={dagState} compact caption="Revisit without a cycle" />
      <p>DFS explores A → B → D and finishes D before entering C. The later C → D edge points to a finished vertex. There is no directed route from D back to C or A; a visited-only rule would report a false cycle.</p>
      <h4>Back edge · D → A reaches an active ancestor</h4>
      <GraphPicture graph={cycle} state={cycleState} compact caption="An edge back to the active frame stack" />
      <p>While A, B and D are active, D → A closes A → B → D → A. The active stack, not just general discovery, identifies this directed cycle. The illustrated states come from the same DFS frame model as the lab. Undirected cycle detection needs a separate parent-edge rule; do not copy this directed test blindly.</p>
    </figure>
  );
}
