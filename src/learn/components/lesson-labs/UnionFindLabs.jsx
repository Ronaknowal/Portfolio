import { useId, useMemo, useState } from 'react';
import { BALANCED_UNIONS, CHAIN_UNIONS, activateIslandCell, activeIslandGroups, buildUnionFind, createIslandGrid, createUnionFind, forestLayout, parentPath, representative, traceFind, traceUnion, unionFindGroups } from '../../data/union-find-models.js';
import './union-find-labs.css';
function Forest({
  state,
  focus = [],
  label = 'Parent forest',
  compact = false
}) {
  const marker = `uf-arrow-${useId().replace(/:/g, '')}`;
  const layout = forestLayout(state);
  return <div className={`uf-forest${compact ? ' uf-forest--compact' : ''}`} tabIndex={0} role="region" aria-label={`${label}; horizontally scrollable if needed`}>
    <svg viewBox={`0 0 ${layout.width} ${layout.height}`} style={{
      width: compact ? '100%' : layout.width
    }} role="img" aria-label={`${label}. Each arrow points from a child to its parent. Roots have double outlines and point to themselves.`}>
      <defs><marker id={marker} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="currentColor" /></marker></defs>
      {state.parent.map((parent, node) => {
        if (parent === node) return null;
        const a = layout.positions[node];
        const b = layout.positions[parent];
        const distance = Math.hypot(b.x - a.x, b.y - a.y);
        const ux = (b.x - a.x) / distance;
        const uy = (b.y - a.y) / distance;
        return <line key={`edge-${node}`} x1={a.x + ux * 21} y1={a.y + uy * 21} x2={b.x - ux * 25} y2={b.y - uy * 25} markerEnd={`url(#${marker})`} className={focus.includes(node) && focus.includes(parent) ? 'uf-edge uf-edge--active' : 'uf-edge'} />;
      })}
      {state.parent.map((parent, node) => {
        const {
          x,
          y
        } = layout.positions[node];
        return <g key={node} className={focus.includes(node) ? 'uf-node uf-node--active' : 'uf-node'}>
          <circle cx={x} cy={y} r="20" />
          {parent === node && <circle cx={x} cy={y} r="25" className="uf-root-ring" />}
          <text x={x} y={y + 6}>{node}</text>
          {parent === node && <text x={x} y={y - 32} className="uf-root-label">root</text>}
        </g>;
      })}
    </svg>
  </div>;
}
function Groups({
  state
}) {
  return <div className="uf-groups" aria-label="Component memberships">
    {unionFindGroups(state).map(group => <span key={group.root}><b>root {group.root}</b> {'{'}{group.members.join(', ')}{'}'}</span>)}
  </div>;
}
function ParentTable({
  state
}) {
  return <details className="uf-details"><summary>Exact parent and component-size table</summary>
    <div className="uf-table-scroll" tabIndex={0} role="region" aria-label="Parent table">
      <table><caption>Size is authoritative only at a root. A dash hides stale non-root entries.</caption><thead><tr><th>Element</th><th>Parent</th><th>Representative</th><th>Root size</th></tr></thead>
        <tbody>{state.parent.map((parent, node) => <tr key={node}><th>{node}</th><td>{parent}</td><td>{representative(state, node)}</td><td>{parent === node ? state.size[node] : '—'}</td></tr>)}</tbody>
      </table>
    </div>
  </details>;
}
function StepControls({
  frames,
  step,
  setStep
}) {
  return <div className="uf-controls uf-step-controls">
    <button type="button" onClick={() => setStep(step - 1)} disabled={step === 0}>Previous step</button>
    <span>Step {step + 1} / {frames.length}</span>
    <button type="button" onClick={() => setStep(step + 1)} disabled={step === frames.length - 1}>Next step</button>
    <button type="button" onClick={() => setStep(frames.length - 1)} disabled={step === frames.length - 1}>Show result</button>
  </div>;
}
export function UnionFindEquivalenceFigure() {
  const flat = buildUnionFind([[0, 1], [1, 2], [2, 3]], 'size', 4);
  const chain = buildUnionFind([[0, 1], [1, 2], [2, 3]], 'unweighted', 4);
  return <figure className="uf-inline">
    <div className="uf-equivalence">
      <div><h3>Same group, shallow forest</h3><Forest state={flat} label="Shallow forest for elements 0, 1, 2, 3" compact /></div>
      <div><h3>Same group, longer route</h3><Forest state={chain} label="Chain forest for elements 0, 1, 2, 3" compact /></div>
    </div>
    <figcaption>The input links are 0—1, 1—2 and 2—3. Both structures represent {'{0, 1, 2, 3}'}. The first forest contains a parent link 2→0 even though 2—0 was never an input edge. Double outlines mark self-parent roots; arrows point upward to parents. Root 0 in the first forest and root 3 in the second are equally valid group names.</figcaption>
  </figure>;
}
export function UnionFindLab() {
  const heading = useId();
  const [committed, setCommitted] = useState(createUnionFind);
  const [frames, setFrames] = useState(() => [{
    state: createUnionFind(),
    focus: [],
    message: 'Eight singleton components. Add a link and inspect both root searches before the merge.'
  }]);
  const [step, setStep] = useState(0);
  const [left, setLeft] = useState('0');
  const [right, setRight] = useState('1');
  const [policy, setPolicy] = useState('size');
  const [error, setError] = useState('');
  const current = frames[step];
  const depth = Math.max(...current.state.parent.map((_, node) => parentPath(current.state, node).length - 1));
  function apply(event) {
    event.preventDefault();
    if (!/^\d+$/.test(left) || !/^\d+$/.test(right)) {
      setError('Enter whole-number element IDs from 0 through 7.');
      return;
    }
    try {
      const result = traceUnion(committed, Number(left), Number(right), policy);
      setFrames(result.frames);
      setStep(0);
      setCommitted(result.state);
      setError('');
    } catch (issue) {
      setError(issue.message);
    }
  }
  function reset(preset) {
    const state = preset === 'balanced' ? buildUnionFind(BALANCED_UNIONS, policy) : preset === 'chain' ? buildUnionFind(CHAIN_UNIONS, policy) : createUnionFind();
    setCommitted(state);
    setFrames([{
      state,
      focus: [],
      message: preset === 'empty' ? 'Reset: eight singleton components.' : `Applied ${preset === 'balanced' ? 'pairwise balanced' : 'consecutive-link'} construction using ${policy === 'size' ? 'size weighting' : 'first-root below second-root'}; no compression.`
    }]);
    setStep(0);
    setError('');
  }
  return <section className="uf-lab" aria-labelledby={heading} data-union-find-lab="unions">
    <p className="uf-eyebrow">INVESTIGATION · LINKS BECOME GROUPS</p><h3 id={heading}>A link joins whole components</h3>
    <p>Inspect the two roots and whether the count will fall. The figure contains parent pointers, not a drawing of the input network. Compression is off here so the attachment policy is visible.</p>
    <form onSubmit={apply} className="uf-controls"><label>First element<input value={left} onChange={event => setLeft(event.target.value)} inputMode="numeric" maxLength={4} /></label><label>Second element<input value={right} onChange={event => setRight(event.target.value)} inputMode="numeric" maxLength={4} /></label><label>Attachment policy<select aria-label="Attachment policy" value={policy} onChange={event => setPolicy(event.target.value)}><option value="size">Smaller below larger</option><option value="unweighted">First root below second</option></select></label><button type="submit" disabled={step !== frames.length - 1}>Apply union</button></form>
    {error && <p role="alert" className="uf-error">{error}</p>}
    <div className="uf-controls"><button type="button" onClick={() => reset('empty')}>Reset singletons</button><button type="button" onClick={() => reset('balanced')}>Pairwise construction</button><button type="button" onClick={() => reset('chain')}>Consecutive links</button></div>
    <p className="uf-note">The policy selector affects the next union or construction. Consecutive links are (0,1), (1,2), …, (6,7). Finish a trace before adding another link. Rewinding shows old snapshots; it does not delete an edge.</p>
    <p className="uf-status" aria-live="polite">{current.message}</p>
    <div className="uf-readout"><strong>{current.state.count} components</strong><span>Maximum pointer depth: {depth} edges</span></div>
    <Forest state={current.state} focus={current.focus} />
    <p className="uf-note">Same-component test: follow each element to its self-parent root and compare those roots.</p>
    <Groups state={current.state} /><StepControls frames={frames} step={step} setStep={setStep} /><ParentTable state={current.state} />
  </section>;
}
export function PathCompressionLab() {
  const heading = useId();
  const seed = useMemo(() => buildUnionFind(), []);
  const [base, setBase] = useState(seed);
  const [node, setNode] = useState('7');
  const [trace, setTrace] = useState(() => traceFind(seed, 7));
  const [step, setStep] = useState(0);
  const current = trace.frames[step];
  function investigate() {
    setTrace(traceFind(base, Number(node)));
    setStep(0);
  }
  function repeat() {
    setBase(trace.state);
    setTrace(traceFind(trace.state, Number(node)));
    setStep(0);
  }
  return <section className="uf-lab" aria-labelledby={heading} data-union-find-lab="compression">
    <p className="uf-eyebrow">INVESTIGATION · A LOOKUP REPAIRS ITS ROUTE</p><h3 id={heading}>Shorten the path, preserve the group</h3>
    <p>The seed was built by seven size-weighted unions: (0,1), (2,3), (0,2), (4,5), (6,7), (4,6), (0,4). Inspect the path from 7, then watch full compression rewrite it.</p>
    <div className="uf-controls"><label>Find element<select aria-label="Find element" value={node} onChange={event => setNode(event.target.value)}>{seed.parent.map((_, index) => <option key={index}>{index}</option>)}</select></label><button type="button" onClick={investigate}>Trace selected find</button><button type="button" onClick={repeat} disabled={step !== trace.frames.length - 1}>Find again on result</button><button type="button" onClick={() => {
        setBase(seed);
        setNode('7');
        setTrace(traceFind(seed, 7));
        setStep(0);
      }}>Reset balanced seed</button></div>
    <p className="uf-note">Changing the selection prepares a new lookup; apply it with “Trace selected find.” This trace is for element {trace.path[0]}. To retain its rewrites, use “Find again on result.”</p>
    <div className="uf-compression-path"><b>Path before this lookup</b><span>{trace.path.join(' → ')}</span><span>{trace.hops} upward hops</span></div>
    <p className="uf-status" aria-live="polite">{current.message}</p>
    <Forest state={current.state} focus={current.focus} label="Path compression forest" />
    <div className="uf-readout"><strong>Representative: {trace.root}</strong><span>A new lookup from {trace.path[0]} now takes {parentPath(current.state, trace.path[0]).length - 1} upward hops.</span></div>
    <Groups state={current.state} /><StepControls frames={trace.frames} step={step} setStep={setStep} /><ParentTable state={current.state} />
    <p className="uf-note">Hop counts measure this tiny pointer trace, not elapsed time or the full operation cost. Recording snapshots and drawing the forest are extra teaching-interface work.</p>
  </section>;
}
export function IslandUnionLab() {
  const heading = useId();
  const [state, setState] = useState(createIslandGrid);
  const [message, setMessage] = useState('All cells are closed: zero active islands. Open a cell with the mouse, Enter or Space.');
  const groups = activeIslandGroups(state);
  const labels = new Map();
  groups.forEach(group => group.members.forEach(node => labels.set(node, Math.min(...group.members))));
  function open(index) {
    const result = activateIslandCell(state, index);
    setState(result.state);
    setMessage(result.message);
  }
  function ring() {
    let fresh = createIslandGrid();
    for (const index of [6, 7, 8, 11, 13, 16, 17, 18]) fresh = activateIslandCell(fresh, index).state;
    setState(fresh);
    setMessage('Ring preset: one island surrounds closed cell (2, 2). Open that cell and inspect the count; all four neighbors already connect.');
  }
  return <section className="uf-lab uf-island-lab" aria-labelledby={heading} data-union-find-lab="islands">
    <p className="uf-eyebrow">INVESTIGATION · CONNECTIONS HIDDEN IN A GRID</p><h3 id={heading}>Open land, then join its neighbors</h3>
    <p>Side contact connects; diagonal contact does not. A cell displays its (row,column) and component label. Labels use the smallest cell ID for stable naming; the DSU root may be different.</p>
    <div className="uf-controls"><button type="button" onClick={() => {
        setState(createIslandGrid());
        setMessage('Reset: zero active islands.');
      }}>Reset grid</button><button type="button" onClick={ring}>Load ring around center</button></div>
    <p className="uf-status" aria-live="polite">{message}</p>
    <div className="uf-island-layout"><div className="uf-grid" role="group" aria-label="Five by five island grid">
      {state.active.map((active, index) => <button type="button" key={index} className={active ? 'uf-cell uf-cell--open' : 'uf-cell'} onClick={() => open(index)} aria-label={`Cell ${Math.floor(index / 5)}, ${index % 5}: ${active ? `open, component ${labels.get(index)}` : 'closed; open cell'}`} aria-pressed={active}><small>{Math.floor(index / 5)},{index % 5}</small><b>{active ? labels.get(index) : '·'}</b></button>)}
    </div><div><p className="uf-island-count"><strong>{state.count}</strong> active {state.count === 1 ? 'island' : 'islands'}</p><p>{state.active.filter(Boolean).length} open cells</p><p className="uf-note">New count = old count + 1 − successful unions. Closed cells do not count as singleton islands.</p></div></div>
    <details className="uf-details"><summary>Exact active component memberships</summary><ul>{groups.length ? groups.map(group => <li key={group.root}>Label {Math.min(...group.members)}; DSU root {group.root}; cells {group.members.map(node => `(${Math.floor(node / 5)},${node % 5})`).join(', ')}</li>) : <li>No active cells.</li>}</ul></details>
    <p className="uf-note">Opening an already open cell is a no-op. This lab supports additions; use reset for a new scenario. Ordinary Union-Find cannot close a cell and discover whether its island splits.</p>
  </section>;
}
