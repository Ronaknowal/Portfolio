import { useId, useState } from 'react';
import { HEAP_SAMPLE, HEAP_RAW_SAMPLE, TOP_K_SAMPLE, TRIE_SAMPLE, STREAM_LIMIT, heapOperationTrace, heapViolations, heapLayout, heapConstructionProfile, topKStreamTrace, parseHeapValues, parseHeapKey, buildTrie, trieWords, trieOperationTrace, trieLayout, parseTrieWords, parseTrieQuery } from '../../data/heap-trie-models.js';
import './heap-trie-labs.css';
const wordLabel = word => word === '' ? 'ε' : word;
function Investigation({
  name,
  eyebrow,
  title,
  children
}) {
  const id = useId();
  return <section className="heap-trie-lab" data-lab={name} aria-labelledby={id}><p className="heap-trie-eyebrow">{eyebrow}</p><h3 id={id}>{title}</h3>{children}</section>;
}
function TraceButtons({
  trace,
  step,
  setStep,
  reset
}) {
  return <div className="heap-trie-steps"><button type="button" onClick={() => setStep(step - 1)} disabled={step === 0}>Back</button><span>Step {step + 1} / {trace.length}</span><button type="button" onClick={() => setStep(step + 1)} disabled={step === trace.length - 1}>{step === trace.length - 1 ? 'Trace complete' : 'Next: ' + trace[step + 1].phase}</button><button type="button" onClick={reset}>Reset investigation</button></div>;
}
function Feedback({
  state
}) {
  return <div className="heap-trie-feedback" role="status"><strong>{state.phase}.</strong> {state.note}</div>;
}
function HeapPicture({
  values,
  activeIndices = [],
  comparedIndices = [],
  swappedIndices = [],
  selectedIndex = null,
  onSelect,
  occurrenceIndices
}) {
  const layout = heapLayout(values),
    highlighted = new Set(activeIndices),
    compared = new Set(comparedIndices),
    swapped = new Set(swappedIndices);
  const selection = selectedIndex !== null && selectedIndex < values.length ? selectedIndex : null;
  const description = layout.nodes.map(node => `index ${node.index} stores ${node.value}; parent ${node.index === 0 ? 'none' : Math.floor((node.index - 1) / 2)}; left ${2 * node.index + 1 < values.length ? 2 * node.index + 1 : 'empty'}; right ${2 * node.index + 2 < values.length ? 2 * node.index + 2 : 'empty'}`).join('; ');
  return <div className="heap-array-tree">
    <p className="heap-trie-mini-heading">One array, drawn two ways {onSelect && '· choose an index to link the views'}</p>
    <div className="heap-array-scroll" role="region" aria-label="Heap array, scroll to inspect indices" tabIndex={0}><ol className="heap-array-strip">{values.map((value, index) => <li key={index} className={`${highlighted.has(index) ? 'is-active ' : ''}${compared.has(index) ? 'is-compared ' : ''}${swapped.has(index) ? 'is-swapped ' : ''}${selection === index ? 'is-selected' : ''}`}>
      {onSelect ? <button type="button" aria-label={`Inspect index ${index}, value ${value}`} aria-pressed={selection === index} onClick={() => onSelect(index)}><small>i = {index}</small><strong>{value}</strong>{occurrenceIndices && <small>#{occurrenceIndices[index] + 1}</small>}</button> : <span><small>i = {index}</small><strong>{value}</strong>{occurrenceIndices && <small>#{occurrenceIndices[index] + 1}</small>}</span>}
    </li>)}</ol>{!values.length && <p className="heap-trie-empty">[ ] · no array positions</p>}</div>
    <div className="heap-trie-diagram-scroll" role="region" tabIndex={0} aria-label="Heap tree; scroll horizontally for the full structure"><svg style={{
        width: layout.width,
        minWidth: layout.width
      }} viewBox={`0 0 ${layout.width} ${layout.height}`} role="img" aria-label={values.length ? `Complete binary heap positions. ${description}` : 'Empty heap, no root'}>
      {layout.edges.map(edge => {
          const parent = layout.nodes[edge.parent],
            child = layout.nodes[edge.child];
          return <line key={edge.child} className={highlighted.has(edge.parent) && highlighted.has(edge.child) ? 'heap-edge is-active' : 'heap-edge'} x1={parent.x} y1={parent.y + 23} x2={child.x} y2={child.y - 24} />;
        })}
      {layout.nodes.map(node => <g key={node.index} className={`heap-node${highlighted.has(node.index) ? ' is-active' : ''}${compared.has(node.index) ? ' is-compared' : ''}${swapped.has(node.index) ? ' is-swapped' : ''}`}>
        {selection === node.index && <circle className="heap-selection-ring" cx={node.x} cy={node.y} r="30" />}
        <circle cx={node.x} cy={node.y} r="23" /><text className="heap-node-value" x={node.x} y={node.y + 6} textAnchor="middle">{node.value}</text><text className="heap-node-index" x={node.x} y={node.y + 43} textAnchor="middle">i = {node.index}</text>
      </g>)}
      {!values.length && <text className="heap-empty-root" x={layout.width / 2} y="50" textAnchor="middle">root → empty</text>}
    </svg></div>
    {selection !== null && <p className="heap-index-inspection">Selected index <strong>{selection}</strong>, value <strong>{values[selection]}</strong>. Parent: {selection ? Math.floor((selection - 1) / 2) : 'none (root)'}. Left child: {2 * selection + 1 < values.length ? 2 * selection + 1 : 'empty'}. Right child: {2 * selection + 2 < values.length ? 2 * selection + 2 : 'empty'}.</p>}
    <p className="heap-trie-note">i labels a position, not a value's identity. Amber marks active positions; dashed outlines identify compared positions; thick green outlines mark a completed swap. Dotted outer ring links the selected array cell to its tree position. The diagram scrolls locally so labels stay readable.</p>
  </div>;
}
export function HeapOperationsLab() {
  const [draft, setDraft] = useState(HEAP_SAMPLE.join(', ')),
    [operation, setOperation] = useState('push'),
    [key, setKey] = useState('0'),
    [trace, setTrace] = useState(() => heapOperationTrace()),
    [step, setStep] = useState(0),
    [selected, setSelected] = useState(0),
    [error, setError] = useState('');
  const state = trace[step],
    violations = heapViolations(state.heap);
  const run = event => {
    event.preventDefault();
    const values = parseHeapValues(draft),
      target = operation === 'push' ? parseHeapKey(key) : {
        valid: true,
        key: 0
      };
    if (!values.valid || !target.valid) {
      setError(values.error || target.error);
      return;
    }
    setTrace(heapOperationTrace(values.values, operation, target.key));
    setStep(0);
    setSelected(0);
    setError('');
  };
  const preset = (values, chosenOperation = 'push') => {
    setDraft(values.join(', '));
    setOperation(chosenOperation);
    setKey('0');
    setTrace(heapOperationTrace(values, chosenOperation, 0));
    setStep(0);
    setSelected(0);
    setError('');
  };
  return <Investigation name="heap-operations" eyebrow="COMPLETE SHAPE + PARENT ORDER" title="Repair one path, not the whole array">
    <p>Predict where pushing 0 will end. Follow each comparison and swap in both views. The array index stays in place while its value changes.</p>
    <form className="heap-trie-controls" onSubmit={run}><label className="heap-trie-wide-control">Starting array · up to 12 integers<input value={draft} onChange={event => setDraft(event.target.value)} /></label><label>Operation<select value={operation} onChange={event => setOperation(event.target.value)}><option value="push">Push one value</option><option value="pop">Pop the minimum</option><option value="build">Build a heap</option></select></label>{operation === 'push' && <label>Value to push<input inputMode="numeric" value={key} onChange={event => setKey(event.target.value)} /></label>}<button type="submit">Run heap operation</button></form>
    <div className="heap-trie-presets"><button type="button" onClick={() => preset(HEAP_SAMPLE)}>Valid min-heap</button><button type="button" onClick={() => preset(HEAP_RAW_SAMPLE, 'build')}>Unordered array → build</button><button type="button" onClick={() => preset([])}>Empty heap</button></div>
    {error && <p className="heap-trie-error" role="alert">{error} The active trace was kept.</p>}
    <p className="heap-trie-note">Draft edits apply with Run heap operation; presets start their own trace. Active: <strong>{state.operation}{state.operation === 'push' ? ` ${state.key}` : ''}</strong>. Each run starts from the displayed starting-array input; completing an operation does not silently rewrite it.</p>
    <HeapPicture values={state.heap} {...state} selectedIndex={selected} onSelect={setSelected} />
    <div className="heap-trie-metrics"><span>{state.heap.length} entries</span><span>{state.comparisonCount} priority comparisons</span><span>{state.swapCount} swaps</span>{state.popped !== null && <span>Returned minimum: {state.popped}</span>}</div>
    <p className={violations.length ? 'heap-trie-warning' : 'heap-trie-valid'}>{violations.length ? `Order still needs repair at ${violations.map(edge => `${edge.parent} → ${edge.child}`).join(', ')}. Intermediate states are not completed heaps.` : 'Every current parent is ≤ each existing child.'}</p>
    <Feedback state={state} /><TraceButtons trace={trace} step={step} setStep={setStep} reset={() => preset(HEAP_SAMPLE)} />
    <details><summary>Transfer: make the child choice matter</summary><p>Pop from 1, 4, 2, 8, 7, 3, 9. The root replacement 9 must swap with right child 2, not left child 4. Then build from 7, 2, 9, 1, 5. Explain why repair starts at index 1 rather than at a leaf. Equal children choose the left in this model, and equal parent/child priorities require no swap.</p></details>
    <p className="heap-trie-note">Indices are zero-based: parent(i) = floor((i − 1) / 2) for i &gt; 0; children are 2i + 1 and 2i + 2 when present. Integer priorities −99…99; no stable ordering of equal-priority jobs is implied. Counts describe this exact teaching algorithm, not runtime or Python heapq's internal comparison count.</p>
  </Investigation>;
}
export function TopKStreamLab() {
  const [draft, setDraft] = useState(TOP_K_SAMPLE.join(', ')),
    [draftK, setDraftK] = useState('3'),
    [trace, setTrace] = useState(() => topKStreamTrace()),
    [step, setStep] = useState(0),
    [error, setError] = useState('');
  const state = trace[step],
    retained = new Set(state.heap.map(entry => entry.sourceIndex)),
    discarded = new Set(state.discarded.map(entry => entry.sourceIndex));
  const run = event => {
    event.preventDefault();
    const parsed = parseHeapValues(draft, STREAM_LIMIT),
      k = Number(draftK);
    if (!parsed.valid || !/^\d+$/.test(draftK) || !Number.isInteger(k) || k < 1 || k > 8) {
      setError(parsed.error || 'Choose an integer k from 1 through 8.');
      return;
    }
    setTrace(topKStreamTrace(parsed.values, k));
    setStep(0);
    setError('');
  };
  const reset = () => {
    setDraft(TOP_K_SAMPLE.join(', '));
    setDraftK('3');
    setTrace(topKStreamTrace());
    setStep(0);
    setError('');
  };
  return <Investigation name="top-k-stream" eyebrow="KEEP THE BOUNDARY THAT CAN STILL MATTER" title="The smallest retained value guards the largest k">
    <p>Predict whether the second 9 replaces another value. We retain occurrences, so two separate 9s can both belong in the largest three.</p>
    <form className="heap-trie-controls" onSubmit={run}><label className="heap-trie-wide-control">Stream · up to 16 integers<input value={draft} onChange={event => setDraft(event.target.value)} /></label><label>k · number of occurrences<input inputMode="numeric" value={draftK} onChange={event => setDraftK(event.target.value)} /></label><button type="submit">Read this stream</button></form>
    {error && <p className="heap-trie-error" role="alert">{error} The active stream was kept.</p>}
    <p className="heap-trie-note">Edits apply with Read this stream. Active k = {state.k}; {state.processed} of {state.stream.length} occurrences committed. Inspecting a candidate does not yet include it in the prefix result.</p>
    <ol className="top-k-stream-ribbon" aria-label="Stream occurrence decisions">{state.stream.map((value, index) => <li key={index} className={`${retained.has(index) ? 'is-retained ' : ''}${discarded.has(index) ? 'is-discarded ' : ''}${state.current?.sourceIndex === index ? 'is-current' : ''}`}><small>#{index + 1}</small><strong>{value}</strong><span>{retained.has(index) ? 'kept' : discarded.has(index) ? 'discarded' : state.current?.sourceIndex === index ? 'candidate' : 'unread'}</span></li>)}</ol>
    <div className="top-k-boundary"><span>{state.processed < state.k ? 'Still filling the retained set' : 'kth-largest boundary of the committed prefix'}</span><strong>{state.kth === null ? 'not available yet' : state.kth}</strong><small>{state.processed < state.k ? `Need ${state.k - state.processed} more occurrence${state.k - state.processed === 1 ? '' : 's'} before a kth-largest value exists.` : 'The minimum retained value. A larger candidate replaces it; a smaller or equal candidate can be discarded.'}</small></div>
    <h4>Retained occurrences · min-heap, not ranked order</h4><HeapPicture values={state.heap.map(entry => entry.value)} occurrenceIndices={state.heap.map(entry => entry.sourceIndex)} activeIndices={state.processed >= state.k ? [0] : []} />
    <div className="top-k-discarded"><h4>Discarded occurrences · out of contention</h4>{state.discarded.length ? <p>{state.discarded.map(entry => `#${entry.sourceIndex + 1}: ${entry.value}`).join(' · ')}</p> : <p>None discarded yet.</p>}</div>
    <Feedback state={state} /><TraceButtons trace={trace} step={step} setStep={setStep} reset={reset} />
    <details><summary>Explain the boundary, then change the constraint</summary><p>At the end of the default stream, the retained values are 5, 9 and 9, so the third-largest occurrence has value 5. Why can a discarded value never return when k stays fixed and the stream only grows? Try k = 8: this six-value stream never has an eighth-largest value. For the smallest k values instead, which heap order would expose the item to replace?</p></details>
    <p className="heap-trie-note">Each commit performs a complete heap repair; use the preceding heap lab to inspect its individual swaps. Occurrence labels preserve duplicates. Candidate ties keep the existing equal boundary occurrence here; this is not a promise of stable priority-queue ordering. This bounded model retains discarded history to teach decisions; a production top-k algorithm need not store discarded values.</p>
  </Investigation>;
}
function TriePicture({
  trie,
  state = {},
  title = 'Shared character paths',
  inspect = true
}) {
  const layout = trieLayout(trie),
    positions = Object.fromEntries(layout.nodes.map(node => [node.id, node])),
    visited = new Set(state.visitedIds || []);
  const description = layout.nodes.map(node => `prefix ${wordLabel(node.prefix)}, ${node.terminal ? 'terminal: stored word' : 'not terminal'}, children ${Object.keys(node.children).sort().join(', ') || 'none'}`).join('; ');
  return <div className="trie-picture"><p className="heap-trie-mini-heading">{title} · read character edges from left to right</p><div className="heap-trie-diagram-scroll" tabIndex={0} role="region" aria-label={title + '; scroll horizontally to inspect character paths'}><svg style={{
        width: layout.width,
        minWidth: layout.width
      }} viewBox={`0 0 ${layout.width} ${layout.height}`} role="img" aria-label={description}>
    {layout.edges.map(edge => {
          const from = positions[edge.fromId],
            to = positions[edge.toId],
            active = state.edge?.fromId === from.id && state.edge?.toId === to.id;
          return <g key={edge.toId} className={active ? 'trie-edge is-active' : 'trie-edge'}><line x1={from.x + 16} y1={from.y} x2={to.x - 17} y2={to.y} /><text x={(from.x + to.x) / 2} y={(from.y + to.y) / 2 - 9} textAnchor="middle">{edge.character}</text></g>;
        })}
    {layout.nodes.map(node => <g key={node.id} className={`trie-node${state.activeId === node.id ? ' is-active' : ''}${visited.has(node.id) ? ' is-visited' : ''}${node.terminal ? ' is-terminal' : ''}`}><circle cx={node.x} cy={node.y} r="15" />{node.terminal && <circle className="trie-terminal-ring" cx={node.x} cy={node.y} r="20" />}<text x={node.x} y={node.y + 5} textAnchor="middle" className="trie-terminal-mark">{node.terminal ? '✓' : node.prefix === '' ? 'ε' : '·'}</text><text x={node.x} y={node.y + 38} textAnchor="middle" className="trie-prefix-label">{wordLabel(node.prefix)}</text></g>)}
  </svg></div><p className="heap-trie-note">Letters label edges; the text below a node is the prefix consumed so far. A double ring and ✓ mean a stored word. A dot means a prefix only. ε is the empty prefix. The drawing scrolls locally to keep letters readable; layout distance is not a measured quantity.</p>
    {inspect && <details><summary>Inspect exact prefixes and terminal flags</summary><div className="heap-trie-table-scroll" tabIndex={0} role="region" aria-label="Trie exact state"><table><thead><tr><th>Prefix</th><th>Stored word?</th><th>Character → next prefix</th></tr></thead><tbody>{layout.nodes.map(node => <tr key={node.id}><th scope="row">{wordLabel(node.prefix)}</th><td>{node.terminal ? 'yes' : 'no'}</td><td>{Object.entries(node.children).sort().map(([letter, id]) => `${letter} → ${wordLabel(positions[id].prefix)}`).join('; ') || 'none'}</td></tr>)}</tbody></table></div></details>}
  </div>;
}
export function TriePrefixLab() {
  const [draftWords, setDraftWords] = useState(TRIE_SAMPLE.join(', ')),
    [query, setQuery] = useState('car'),
    [operation, setOperation] = useState('exact'),
    [trace, setTrace] = useState(() => trieOperationTrace()),
    [step, setStep] = useState(0),
    [error, setError] = useState('');
  const state = trace[step],
    stored = trieWords(state.trie);
  const run = event => {
    event.preventDefault();
    const words = parseTrieWords(draftWords),
      parsed = parseTrieQuery(query);
    if (!words.valid || !parsed.valid) {
      setError(words.error || parsed.error);
      return;
    }
    setTrace(trieOperationTrace(buildTrie(words.words), parsed.query, operation));
    setStep(0);
    setError('');
  };
  const reset = () => {
    setDraftWords(TRIE_SAMPLE.join(', '));
    setQuery('car');
    setOperation('exact');
    setTrace(trieOperationTrace());
    setStep(0);
    setError('');
  };
  return <Investigation name="trie-prefix" eyebrow="FOLLOW A PREFIX, THEN ASK WHETHER IT ENDS A WORD" title="A shared path is not the same as a stored word">
    <p>Predict exact lookup for ca, then try prefix lookup for ca. Both follow the same edges. The terminal marker decides whether ca itself is stored.</p>
    <form className="heap-trie-controls" onSubmit={run}><label className="heap-trie-wide-control">Starting words · comma-separated<input value={draftWords} onChange={event => setDraftWords(event.target.value)} /></label><label>Operation<select value={operation} onChange={event => setOperation(event.target.value)}><option value="exact">Exact word lookup</option><option value="prefix">Prefix suggestions</option><option value="insert">Insert a word</option><option value="delete">Delete a word</option></select></label><label>Word or prefix<input value={query} onChange={event => setQuery(event.target.value)} /></label><button type="submit">Trace these characters</button></form>
    {error && <p className="heap-trie-error" role="alert">{error} The active trie was kept.</p>}
    <p className="heap-trie-note">Inputs apply with Trace these characters. Each run starts from the starting word set; insertion/deletion changes this trace's result. Active operation: <strong>{state.operation}</strong>, query <strong>{wordLabel(state.query)}</strong>. Blank query is the empty string; use ε in the word list to store it.</p>
    <div className="trie-query-strip" aria-label="Query character consumption"><span>query</span>{[...state.query].map((character, index) => <strong key={index} className={index < [...state.consumed].length ? 'is-consumed' : ''}>{character}<small>{index < [...state.consumed].length ? 'read' : 'next'}</small></strong>)}{!state.query && <strong>ε<small>0 letters</small></strong>}<span>consumed: <b>{wordLabel(state.consumed)}</b></span></div>
    <TriePicture trie={state.trie} state={state} />
    <div className="trie-word-results"><h4>Stored words in this snapshot</h4><p>{stored.length ? stored.map(wordLabel).join(' · ') : 'No stored words.'}</p>{state.operation === 'prefix' && state.result !== null && <><h4>Suggestions for {wordLabel(state.query)}</h4><p>{state.matches.length ? state.matches.map(wordLabel).join(' · ') : 'No stored words match.'}</p></>}</div>
    {state.removedIds.length > 0 && <p className="heap-trie-warning">Pruned identities: {state.removedIds.join(', ')}. Surviving prefixes keep their identities and child links.</p>}
    <Feedback state={state} /><TraceButtons trace={trace} step={step} setStep={setStep} reset={reset} />
    <details><summary>Transfer: delete a prefix without losing its longer word</summary><p>Delete car: cart must remain. Delete cart from the original set: prune only its t suffix, because car is still terminal. Insert ca: reuse its path and add a terminal flag. Finally start with ε, car and look up the empty string; only the root's terminal marker makes that an exact stored word. Empty-prefix suggestions enumerate every stored word.</p></details>
    <p className="heap-trie-note">Browser words are case-sensitive lowercase ASCII a–z, at most 8 entries of 6 letters and 32 visible nodes. This explicitly narrower input surface is not a general Unicode or normalization demonstration. Sparse child links represent only present characters. Suggestions are alphabetic here; finding a prefix and ranking its completions are separate tasks. Clearing a marker precedes pruning, so an intermediate unused path may still be visible.</p>
  </Investigation>;
}
export function HeapConstructionFigure() {
  const profile = heapConstructionProfile(15);
  return <figure className="heap-trie-inline-figure"><figcaption><strong>Most positions have little or no distance left to sink</strong></figcaption><div className="heap-construction-levels">{[...profile.rows].reverse().map(row => <div key={row.height}><span>height {row.height}</span><div className="heap-height-positions">{row.indices.map(index => <span key={index}>i={index}</span>)}</div><strong>{row.count} × {row.height} = {row.downwardBudget}</strong></div>)}</div><p>These are the actual subtree heights of a complete 15-position tree, not timings. Eight leaves contribute zero possible downward steps. Four parents can sink at most one level each; two can sink two; the root can sink three. The total budget is <strong>{profile.totalDownwardBudget}</strong> downward moves, versus the loose 15 × 3 bound that incorrectly treats every node like the root.</p><p>Bottom-up construction starts at index 6, then 5 down to 0. At each internal node, both child subtrees already satisfy heap order. A downward level uses at most two priority comparisons in this implementation; a swap may stop sooner. Across growing complete trees, the height-weighted total is O(n), which is why heap construction is linear.</p></figure>;
}
export function TrieSharingFigure() {
  const trie = buildTrie();
  return <figure className="heap-trie-inline-figure"><figcaption><strong>car, cart and cat share characters, not complete-word status</strong></figcaption><TriePicture trie={trie} inspect={false} title="Four stored words, shared prefix nodes" /><p>All three c-words share c → a. car is a stored word and also the prefix of cart, so its node has both a terminal marker and an outgoing t edge. ca has a path but no terminal marker. dog starts a different root branch. The {trie.nodes.length} visible nodes include the empty root; no node is allocated for a missing alphabet character.</p></figure>;
}
