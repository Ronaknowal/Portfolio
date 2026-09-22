import { useId, useState } from 'react';
import { TREE_SAMPLE, buildTree, treeMetrics, treeLayout, validateBST, inorderKeys, parseTreeKeys, parseTreeTarget, searchTrace, insertionTrace, traversalTrace, deletionTrace, invalidBoundsExample, rotationExample } from '../../data/tree-models.js';
import './tree-labs.css';

function TreePanel({ name, eyebrow, title, children }) {
  const uid = useId();
  return <section className="tree-lab" data-lab={name} aria-labelledby={uid}><p className="tree-eyebrow">{eyebrow}</p><h3 id={uid}>{title}</h3>{children}</section>;
}

function TraceControls({ step, trace, setStep, reset }) {
  return <div className="tree-trace-controls"><button type="button" disabled={step === 0} onClick={() => setStep(step - 1)}>Back</button><span>Step {step + 1} / {trace.length}</span><button type="button" disabled={step === trace.length - 1} onClick={() => setStep(step + 1)}>{step === trace.length - 1 ? 'Trace complete' : 'Next: ' + trace[step + 1].phase}</button><button type="button" onClick={reset}>Reset investigation</button></div>;
}

function TreeDiagram({ tree, state = {}, caption = 'Tree structure', annotations = {}, identity = true, inspect = true }) {
  const uid = useId(), [enlarged, setEnlarged] = useState(false);
  const layout = treeLayout(tree), positions = Object.fromEntries(layout.nodes.map(node => [node.id, node]));
  const highlighted = new Set([...(state.visitedIds || []), ...(state.frontier || []).map(frame => frame.nodeId)]);
  const emitted = new Set(state.outputIds || []);
  const describe = node => `${node.id}, key ${node.key}, depth ${node.depth}, left ${node.leftId === null ? 'empty' : positions[node.leftId]?.key}, right ${node.rightId === null ? 'empty' : positions[node.rightId]?.key}${node.id === tree.rootId ? ', root' : ''}`;
  return <div className="tree-diagram">
    {inspect && <div className="tree-diagram-tools"><span>{caption}</span><button type="button" aria-pressed={enlarged} onClick={() => setEnlarged(!enlarged)}>{enlarged ? 'Fit whole tree' : 'Enlarge tree labels'}</button></div>}
    <div className="tree-diagram-scroll" tabIndex={enlarged ? 0 : undefined} role="region" aria-label={caption + (enlarged ? '; scroll horizontally to inspect' : '')}>
      <svg viewBox={`0 0 ${layout.width} ${layout.height}`} style={{ minWidth: enlarged ? layout.width : undefined }} role="img" aria-label={tree.nodes.length ? `${caption}. ${layout.nodes.map(describe).join('; ')}` : `${caption}: empty root, no nodes`}>
        <defs><marker id={uid} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill="currentColor" /></marker></defs>
        {layout.edges.map(edge => {
          const from = positions[edge.fromId], to = positions[edge.toId], dx = to.x - from.x, dy = to.y - from.y, length = Math.hypot(dx, dy);
          const active = state.edge?.fromId === edge.fromId && state.edge?.toId === edge.toId;
          return <g key={`${edge.fromId}-${edge.side}`} className={active ? 'tree-edge is-active' : 'tree-edge'}><line x1={from.x + dx / length * 23} y1={from.y + dy / length * 23} x2={to.x - dx / length * 25} y2={to.y - dy / length * 25} markerEnd={`url(#${uid})`} /><text x={(from.x + to.x) / 2 + (edge.side === 'left' ? -8 : 8)} y={(from.y + to.y) / 2 - 4} textAnchor={edge.side === 'left' ? 'end' : 'start'}>{edge.side === 'left' ? 'L' : 'R'}</text></g>;
        })}
        {layout.nodes.map(node => {
          const active = state.activeId === node.id, successor = state.successorId === node.id, target = state.targetId === node.id;
          return <g key={node.id} className={`tree-node${active ? ' is-active' : ''}${highlighted.has(node.id) ? ' is-path' : ''}${emitted.has(node.id) ? ' is-emitted' : ''}${successor ? ' is-successor' : ''}`}>
            {target && <circle className="tree-target-ring" cx={node.x} cy={node.y} r="28" />}
            <circle cx={node.x} cy={node.y} r="22" /><text className="tree-node-key" x={node.x} y={node.y + 6} textAnchor="middle">{node.key}</text>
            {identity && <text className="tree-node-id" x={node.x} y={node.y + 39} textAnchor="middle">{node.id}</text>}
            {annotations[node.id] && <text className="tree-annotation" x={node.x} y={node.y + (identity ? 54 : 42)} textAnchor="middle">{annotations[node.id]}</text>}
          </g>;
        })}
        {!tree.nodes.length && <text className="tree-empty-label" x={layout.width / 2} y="62" textAnchor="middle">root → empty</text>}
      </svg>
    </div>
    <p className="tree-legend">L = left child; R = right child. {identity && 'n1, n2… are stable node identities; the number inside is the key. '}Amber marks the current comparison or action. Position organizes the drawing, not memory addresses.</p>
    {inspect && <details className="tree-structure-details"><summary>Inspect exact keys, links and depths</summary><div className="tree-table-scroll" tabIndex={0} role="region" aria-label="Tree links table"><table><caption>root → {tree.rootId ?? 'empty'}</caption><thead><tr>{['Node', 'Key', 'Left', 'Right', 'Depth'].map(label => <th key={label} scope="col">{label}</th>)}</tr></thead><tbody>{layout.nodes.map(node => <tr key={node.id}><th scope="row">{node.id}</th><td>{node.key}</td><td>{node.leftId ?? 'empty'}</td><td>{node.rightId ?? 'empty'}</td><td>{node.depth}</td></tr>)}</tbody></table></div></details>}
  </div>;
}

function Feedback({ state }) {
  return <div className={`tree-feedback${state.transient ? ' is-transient' : ''}`} role="status"><strong>{state.phase}.</strong> {state.note}</div>;
}

export function TreeSearchLab() {
  const [draftKeys, setDraftKeys] = useState(TREE_SAMPLE.join(', ')), [tree, setTree] = useState(() => buildTree());
  const [target, setTarget] = useState('7'), [operation, setOperation] = useState('search'), [trace, setTrace] = useState(() => searchTrace(buildTree(), 7)), [step, setStep] = useState(0), [error, setError] = useState('');
  const state = trace[step], metrics = treeMetrics(state.tree);
  const run = (activeTree = tree, chosenOperation = operation, targetText = target) => {
    const parsed = parseTreeTarget(targetText);
    if (!parsed.valid) { setError(parsed.error); return; }
    setError(''); setStep(0); setTrace(chosenOperation === 'insert' ? insertionTrace(activeTree, parsed.key) : searchTrace(activeTree, parsed.key));
  };
  const rebuild = values => {
    const parsed = parseTreeTarget(target);
    if (!parsed.valid) { setError(parsed.error); return; }
    const nextTree = buildTree(values); setTree(nextTree); setDraftKeys(values.join(', ')); run(nextTree);
  };
  const apply = event => { event.preventDefault(); const parsed = parseTreeKeys(draftKeys); if (!parsed.valid) { setError(parsed.error); return; } rebuild(parsed.values); };
  const reset = () => { const initial = buildTree(); setTree(initial); setDraftKeys(TREE_SAMPLE.join(', ')); setTarget('7'); setOperation('search'); setTrace(searchTrace(initial, 7)); setStep(0); setError(''); };
  return <TreePanel name="tree-search" eyebrow="COMPARE → CHOOSE ONE SUBTREE" title="Where can this key still be?">
    <p>Inspect the path to 7 as you step. The interval keeps every ancestor restriction; going right from 6 does not erase the earlier requirement to stay below 8.</p>
    <form onSubmit={apply} className="tree-build-form"><label>Insertion order (up to 12 integer entries)<input value={draftKeys} onChange={event => setDraftKeys(event.target.value)} /></label><button type="submit">Build this tree</button></form>
    <div className="tree-preset-buttons"><button type="button" onClick={() => rebuild(TREE_SAMPLE)}>Sample insertion order</button><button type="button" onClick={() => rebuild([...TREE_SAMPLE].sort((a, b) => a - b))}>Same keys, ascending order</button><button type="button" onClick={() => rebuild([])}>Empty tree</button></div>
    <form className="tree-controls" onSubmit={event => { event.preventDefault(); run(); }}><label>Operation<select value={operation} onChange={event => setOperation(event.target.value)}><option value="search">Search for a key</option><option value="insert">Insert one key</option></select></label><label>Target key<input inputMode="numeric" value={target} onChange={event => setTarget(event.target.value)} /></label><button type="submit">Trace this operation</button></form>
    {error && <p className="tree-error" role="alert">{error} The displayed tree and trace were kept.</p>}
    <p className="tree-note">Edits apply only with Build this tree or Trace this operation. Active trace: <strong>{state.key}</strong>, {trace.at(-1).result === 'inserted' || trace.at(-1).result === 'duplicate' || trace.at(-1).result === 'limit' ? 'insertion' : 'search'}. Each operation starts from the last built tree; insertion changes this trace's result, not the saved insertion order.</p>
    <div className="tree-readouts"><span>{metrics.size} nodes</span><span>height {metrics.height} edges</span><span>{state.comparisonCount} node comparisons</span></div>
    <TreeDiagram tree={state.tree} state={state} caption="Current ordered search" />
    <div className="tree-bounds"><span>Permitted interval on the active path</span><strong>{state.lower ?? '−∞'} &lt; key &lt; {state.upper ?? '+∞'}</strong><small>Bounds are strict because duplicates are ignored. Empty endpoints mean no bound from that side.</small></div>
    <p className="tree-path">Compared node keys: {state.visitedIds.length ? state.visitedIds.filter(id => tree.nodes.some(node => node.id === id)).map(id => tree.nodes.find(node => node.id === id).key).join(' → ') : 'none yet'}</p>
    <Feedback state={state} /><TraceControls step={step} trace={trace} setStep={setStep} reset={reset} />
    <details><summary>Transfer: shape, duplicate and absent key</summary><p>Search for 14 using both insertion orders. The sample needs 3 comparisons; ascending order needs 9. Insert 5 in the sample and follow the strict interval (4, 6). Then insert 6: no new node is created. A valid BST need not be balanced.</p></details>
    <p className="tree-note">This ordered-set model supports integer keys −99…99 and at most 12 visible nodes. Duplicates leave the tree unchanged. Height counts edges; empty height is −1. Comparison counts are exact for this operation, not wall-clock benchmarks.</p>
  </TreePanel>;
}

export function TreeTraversalLab() {
  const tree = buildTree(), [order, setOrder] = useState('inorder'), [step, setStep] = useState(0);
  const trace = traversalTrace(tree, order), state = trace[step], queue = order === 'level-order';
  const frontier = queue ? state.frontier : [...state.frontier].reverse();
  return <TreePanel name="tree-traversal" eyebrow="PENDING WORK IS DIFFERENT FROM OUTPUT" title="When does a visited node enter the answer?">
    <p>The tree stays fixed. Inspect the first three output keys, then compare the frontier with the answer. A call can begin at 8 while inorder output begins at 1.</p>
    <div className="tree-controls"><label>Traversal order<select value={order} onChange={event => { setOrder(event.target.value); setStep(0); }}><option value="preorder">Preorder: node, left, right</option><option value="inorder">Inorder: left, node, right</option><option value="postorder">Postorder: left, right, node</option><option value="level-order">Level order: FIFO queue</option></select></label></div>
    <TreeDiagram tree={tree} state={state} caption="The same tree, a different visit moment" />
    <div className="tree-traversal-state"><div><h4>{queue ? 'Queue · front first' : 'Call stack · top first'}</h4><ol className={queue ? 'tree-frontier tree-frontier--queue' : 'tree-frontier'}>{frontier.map((frame, index) => <li key={frame.nodeId}><strong>{tree.nodes.find(node => node.id === frame.nodeId).key}</strong><span>{frame.nodeId} · {index === 0 ? queue ? 'FRONT' : 'TOP' : frame.phase}</span>{!queue && <small>{frame.phase}</small>}</li>)}</ol>{!frontier.length && <p className="tree-empty">No pending {queue ? 'nodes' : 'calls'}.</p>}</div><div><h4>Output · emitted keys only</h4><ol className="tree-output">{state.output.map((key, index) => <li key={state.outputIds[index]}><strong>{key}</strong><small>#{index + 1}</small></li>)}</ol>{!state.output.length && <p className="tree-empty">Nothing emitted yet.</p>}</div></div>
    <Feedback state={state} /><TraceControls step={step} trace={trace} setStep={setStep} reset={() => { setOrder('inorder'); setStep(0); }} />
    <details><summary>Check the four destinations</summary><p>Preorder begins 8, 3, 1. Inorder gives 1, 3, 4, 6, 7, 8, 10, 13, 14. Postorder ends 13, 14, 10, 8. Level order begins 8, 3, 10. Explain why only inorder is guaranteed to be sorted by the BST rule.</p></details>
    <p className="tree-note">The recursive traces display nonempty calls and their resume phases; immediate calls on empty links are omitted. Output storage is separate from auxiliary stack/queue storage. The level-order queue enqueues left before right. No tree keys or links change.</p>
  </TreePanel>;
}

const deletionPresets = {
  sample: { values: TREE_SAMPLE, key: 8 },
  successor: { values: [20, 10, 40, 30, 50, 35], key: 20 },
  single: { values: [8], key: 8 },
  one: { values: [8, 3], key: 8 },
  empty: { values: [], key: 8 },
};

export function TreeDeletionLab() {
  const [preset, setPreset] = useState('sample'), [target, setTarget] = useState('8'), [trace, setTrace] = useState(() => deletionTrace(buildTree(), 8)), [step, setStep] = useState(0), [error, setError] = useState('');
  const state = trace[step], validation = validateBST(state.tree);
  const choose = name => { const item = deletionPresets[name]; setPreset(name); setTarget(String(item.key)); setTrace(deletionTrace(buildTree(item.values), item.key)); setStep(0); setError(''); };
  const run = event => { event.preventDefault(); const parsed = parseTreeTarget(target); if (!parsed.valid) { setError(parsed.error); return; } setTrace(deletionTrace(buildTree(deletionPresets[preset].values), parsed.key)); setStep(0); setError(''); };
  return <TreePanel name="tree-deletion" eyebrow="RECONNECT THE STRUCTURE, KEEP THE ORDER" title="Which node disappears when you delete a key?">
    <p>Inspect the final root and surviving identities as you step. With two children, this implementation copies the successor's key into the target, then removes the successor node. Copying a key is different from moving the whole node.</p>
    <div className="tree-controls"><label>Starting tree<select value={preset} onChange={event => choose(event.target.value)}><option value="sample">Sample tree · delete root 8</option><option value="successor">Deeper successor with right child</option><option value="single">One-node tree</option><option value="one">Root with one child</option><option value="empty">Empty tree</option></select></label></div>
    <form className="tree-controls" onSubmit={run}><label>Key to delete<input inputMode="numeric" value={target} onChange={event => setTarget(event.target.value)} /></label><button type="submit">Trace deletion</button></form>
    {error && <p className="tree-error" role="alert">{error} The active trace was kept.</p>}
    <p className="tree-note">Active deletion: key {state.key}. Changing the input applies when you press Trace deletion. Each investigation starts from the selected preset.</p>
    <TreeDiagram tree={state.tree} state={state} caption="Deletion: identities, keys and links" />
    <div className="tree-identity-readout"><span>Target identity: <strong>{state.targetId ?? 'locating'}</strong></span><span>Successor identity: <strong>{state.successorId ?? 'not selected'}</strong></span><span>Removed identity: <strong>{state.removedId ?? 'none yet'}</strong></span></div>
    <p className={state.transient ? 'tree-transient-note' : 'tree-validity'}>{validation.valid ? 'Strict global BST ordering holds in this snapshot.' : state.transient ? 'Intermediate write: duplicate key is visible. This is not yet the completed set; finish reconnecting before using the result.' : validation.errors.join(' ')}</p>
    <Feedback state={state} /><TraceControls step={step} trace={trace} setStep={setStep} reset={() => choose('sample')} />
    <details><summary>Transfer: try the three deletion shapes</summary><p>In the sample, delete 1 (leaf), 14 (one child) and 3 (two children). Then use the deeper-successor preset: deleting 20 copies 30 into n1 and reconnects the successor's right child 35 as the left child of 40. Neither subtree may be lost. Delete a missing key and explain why nothing changes.</p></details>
    <p className="tree-note">This lab models one explicit key-copy deletion algorithm. Other correct implementations transplant node objects instead. IDs remain visible to distinguish those policies. The temporary duplicate is an intermediate mutation, not a published valid tree; concurrent access, memory reclamation and balancing are outside this model.</p>
  </TreePanel>;
}

export function TreeAnatomyFigure() {
  const tree = buildTree(), metrics = treeMetrics(tree);
  return <figure className="tree-inline-figure"><figcaption><strong>One root, child links, and several possible stopping points</strong></figcaption><TreeDiagram tree={tree} identity={false} inspect={false} caption="Sample tree anatomy" /><p>Root 8 has no parent and depth 0. Nodes 1, 4, 7 and 13 are leaves: both child links are empty. Leaf 13 has depth 3. Node 3 also roots its own smaller subtree. Depth counts edges from the whole tree's root; the longest root-to-leaf route has {metrics.height} edges, so the tree's height is {metrics.height}. There are {metrics.size} nodes and {metrics.size - 1} parent–child edges.</p></figure>;
}

export function TreeBoundsFigure() {
  const tree = invalidBoundsExample();
  return <figure className="tree-inline-figure"><figcaption><strong>A parent-only check can miss a broken ancestor rule</strong></figcaption><TreeDiagram tree={tree} state={{ activeId: 'n3', visitedIds: ['n1', 'n2', 'n3'] }} identity={false} inspect={false} caption="Invalid binary search tree: key 12 lies in 10's left subtree" /><div className="tree-bounds-chain"><span>Left from 10<br /><strong>key &lt; 10</strong></span><span aria-hidden="true">→</span><span>Right from 5<br /><strong>5 &lt; key &lt; 10</strong></span><span aria-hidden="true">→</span><span className="is-invalid">Key 12<br /><strong>12 &lt; 10 is false</strong></span></div><p>12 is greater than its parent 5, but it is still inside 10's left subtree. Checking only the immediate parent accepts this mistake; carrying both ancestor bounds catches it. This picture is deliberately invalid.</p></figure>;
}

export function TreeRotationFigure() {
  const { before, after, transferredId } = rotationExample();
  return <figure className="tree-inline-figure"><figcaption><strong>A right rotation changes links while preserving sorted order</strong></figcaption><div className="tree-rotation-pair"><div><h4>Before · root 30</h4><TreeDiagram tree={before} state={{ activeId: transferredId }} identity={false} inspect={false} caption="Before right rotation" /></div><div><h4>After · root 20</h4><TreeDiagram tree={after} state={{ activeId: transferredId }} identity={false} inspect={false} caption="After right rotation" /></div></div><p>Promote 20. Make 30 its right child. Transfer 20's former right subtree, rooted at 25, to 30's left link: 20 &lt; 25 &lt; 30 still holds. Keys and node identities are preserved; only the root reference and child links change.</p><p className="tree-inorder-proof">Before: {inorderKeys(before).join(' → ')}<br />After: {inorderKeys(after).join(' → ')}</p><p>A rotation is a local operation. An AVL or red–black tree also needs rules deciding when and which rotations restore its balancing invariant; one rotation alone is not a complete balancing algorithm.</p></figure>;
}
