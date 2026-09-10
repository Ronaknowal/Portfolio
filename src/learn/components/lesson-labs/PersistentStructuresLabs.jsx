import { useEffect, useId, useRef, useState } from 'react';
import { DEFAULT_PERSISTENT_VALUES, HISTORY_WRITES, assignPersistent, createPersistentStore, historyLookup, parsePersistentValues, persistentLayout, persistentQuery, prefixRankModel, reachablePersistentNodes } from '../../data/persistent-structures-models.js';
import './persistent-structures-labs.css';
function NumberField({
  label,
  value,
  onChange,
  min,
  max
}) {
  return <label>
    {label}
    <input type="number" value={value} min={min} max={max} step="1" onChange={event => onChange(event.target.value)} />
  </label>;
}
export function StackSharingFigure() {
  const marker = useId().replace(/:/g, '');
  return <figure className="persistent-visual persistent-stack">
    <svg viewBox="0 0 330 240" role="img" aria-label="Two stack heads, value 9 and value 7, both reference the same tail node with value 3, which references value 1 then empty.">
      <defs><marker id={marker} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 Z" /></marker></defs>
      <path d="M82 66 L145 120 M248 66 L185 120 M165 149 L165 181" markerEnd={`url(#${marker})`} />
      <text x="82" y="20">root A</text>
      <text x="248" y="20">root B</text>
      <rect x="42" y="30" width="80" height="40" />
      <rect x="208" y="30" width="80" height="40" />
      <text x="82" y="56">9 · n3</text>
      <text x="248" y="56">7 · n4</text>
      <rect x="125" y="112" width="80" height="40" className="shared" />
      <text x="165" y="138">3 · n2</text>
      <rect x="125" y="185" width="80" height="40" className="shared" />
      <text x="165" y="211">1 · n1</text>
      <text x="56" y="141">shared</text>
      <text x="262" y="211">→ empty</text>
    </svg>
    <figcaption>Four physical nodes represent two three-element stacks. Both arrows meet the <em>same</em> n2 object. Popping either new head returns n2; no tail is copied or changed.</figcaption>
  </figure>;
}
function initialStore() {
  return assignPersistent(createPersistentStore(), 0, 2, 9).store;
}
export function PathCopyFigure() {
  const store = initialStore();
  const marker = useId().replace(/:/g, '');
  const nodes = [{
    id: 8,
    x: 55,
    y: 60
  }, {
    id: 11,
    x: 275,
    y: 60
  }, {
    id: 2,
    x: 165,
    y: 127,
    shared: true
  }, {
    id: 7,
    x: 55,
    y: 194
  }, {
    id: 10,
    x: 275,
    y: 194
  }, {
    id: 6,
    x: 165,
    y: 261,
    shared: true
  }, {
    id: 3,
    x: 55,
    y: 328
  }, {
    id: 9,
    x: 275,
    y: 328
  }];
  const positions = new Map(nodes.map(node => [node.id, node]));
  return <figure className="persistent-visual persistent-copy-figure">
    <svg viewBox="0 0 330 370" role="img" aria-label="Assigning index 2 from 4 to 9 creates nodes n9, n10 and n11 along the right then left path. Both roots share subtree n2 for indices 0 through 1; both right parents share subtree n6 for indices 3 through 4. Unchanged subtrees are collapsed.">
      <defs><marker id={marker} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 Z" /></marker></defs>
      <text x="55" y="20">v0 root</text>
      <text x="275" y="20">v1 root</text>
      {nodes.filter(node => !node.shared).flatMap(node => [store.nodes[node.id].left, store.nodes[node.id].right].filter(id => positions.has(id)).map(id => {
        const target = positions.get(id);
        return <path key={`${node.id}-${id}`} d={`M${node.x} ${node.y + 25} L${target.x} ${target.y - 28}`} markerEnd={`url(#${marker})`} />;
      }))}
      {nodes.map(({
        id,
        x,
        y,
        shared
      }) => <g key={id} transform={`translate(${x},${y})`}>
        <rect x="-45" y="-26" width="90" height="52" rx="4" className={shared ? 'shared' : id >= 9 ? 'copied' : ''} />
        <text y="-4">n{id} · {store.nodes[id].sum}</text>
        <text y="18">[{store.nodes[id].low},{store.nodes[id].high})</text>
        {shared && <text className="shared-label" y="45">same subtree</text>}
      </g>)}
    </svg>
    <figcaption>Labels give object ID, sum and owned interval. Amber boxes are new; green boxes are one shared subtree each, collapsed here. Both versions keep their own changed path while their unchanged neighbors are identical objects.</figcaption>
  </figure>;
}
function PhysicalGraph({
  store,
  versions,
  queryCover,
  newIds
}) {
  const marker = useId().replace(/:/g, '');
  const scrollRef = useRef(null);
  const reached = reachablePersistentNodes(store, versions);
  const {
    positions,
    width,
    height
  } = persistentLayout(store, reached);
  const roots = versions.map(version => store.versions[version].root).filter(root => root !== null);
  const rootCenter = roots.length ? roots.reduce((sum, root) => sum + positions.get(root).x, 0) / roots.length : 0;
  function centerRoots() {
    if (scrollRef.current) scrollRef.current.scrollLeft = Math.max(0, rootCenter - scrollRef.current.clientWidth / 2);
  }
  useEffect(centerRoots, [rootCenter, width]);
  return <>
    <button onClick={centerRoots}>Center root handles</button>
    <div ref={scrollRef} className="persistent-scroll" tabIndex={0} role="region" aria-label="Physical node graph; scroll horizontally to inspect every node">
    <svg className="persistent-dag" viewBox={`0 0 ${width} ${height}`} style={{
        width
      }} role="img" aria-label={`Physical node graph for versions ${versions.join(' and ')}. Shared node IDs appear once; all edges reference immutable children. The identity table below gives every connection.`}>
        <defs><marker id={marker} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 Z" /></marker></defs>
        {[...reached].map(id => {
          const node = store.nodes[id];
          const from = positions.get(id);
          return [node.left, node.right].filter(child => child !== null).map(child => {
            const to = positions.get(child);
            return <path key={`${id}-${child}`} d={`M${from.x} ${from.y + 30} L${to.x} ${to.y - 34}`} markerEnd={`url(#${marker})`} />;
          });
        })}
        {versions.map((version, position) => {
          const root = store.versions[version].root;
          if (root === null) return null;
          const target = positions.get(root);
          const x = target.x + (position ? 35 : -35);
          return <g key={version}>
          <text x={x} y={28}>v{version} root</text>
          <path d={`M${x} 38 L${target.x} ${target.y - 34}`} markerEnd={`url(#${marker})`} />
        </g>;
        })}
        {[...reached].map(id => {
          const node = store.nodes[id];
          const point = positions.get(id);
          return <g key={id} transform={`translate(${point.x},${point.y})`} className={newIds.includes(id) ? 'new-node' : 'old-node'}>
          <rect x="-32" y="-32" width="64" height="64" rx="5" className={queryCover.includes(id) ? 'query-node' : ''} />
          <text y="-11">n{id}</text>
          <text y="9">Σ {node.sum}</text>
          <text y="28" className="interval">[{node.low},{node.high})</text>
        </g>;
        })}
      </svg>
  </div>
  </>;
}
export function PathCopyLab() {
  const [store, setStore] = useState(initialStore);
  const [source, setSource] = useState('0');
  const [selected, setSelected] = useState('1');
  const [index, setIndex] = useState('2');
  const [value, setValue] = useState('9');
  const [low, setLow] = useState('1');
  const [high, setHigh] = useState('4');
  const [draft, setDraft] = useState(DEFAULT_PERSISTENT_VALUES.join(', '));
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('v1 branches from v0: index 2 changes from 4 to 9. Three new nodes.');
  const [newIds, setNewIds] = useState([9, 10, 11]);
  const [retained, setRetained] = useState([0, 1]);
  const [query, setQuery] = useState(() => persistentQuery(initialStore(), 1, 1, 4));
  const [activeQuery, setActiveQuery] = useState({
    version: 1,
    low: 1,
    high: 4
  });
  function apply(action) {
    try {
      action();
      setError('');
    } catch (failure) {
      setError(failure.message);
    }
  }
  function reset(values = DEFAULT_PERSISTENT_VALUES) {
    const next = createPersistentStore(values);
    setStore(next);
    setSource('0');
    setSelected('0');
    setIndex('0');
    setValue('9');
    setLow('0');
    setHigh(String(values.length));
    setNewIds([]);
    setRetained([0]);
    setQuery(persistentQuery(next, 0, 0, values.length));
    setActiveQuery({
      version: 0,
      low: 0,
      high: values.length
    });
    setNotice('Fresh v0. The initial tree is built once.');
  }
  const shown = [...new Set([Number(source), activeQuery.version])];
  const reachable = reachablePersistentNodes(store, retained);
  return <section className="persistent-lab" aria-label="Path copying and version ownership investigation">
    <h3>Change one path. Keep both histories.</h3>
    <p>The graph initially shows v0 and v1 at once. Amber outlines identify newly allocated objects; a thick green border marks nodes used by the active interval sum. IDs are object identities, not array indices. Pan the graph to follow every arrow; the identity table is an equivalent text view.</p>
    <div className="persistent-controls">
      <label>Source version<select aria-label="Source version" value={source} onChange={event => setSource(event.target.value)}>{store.versions.map((_, i) => <option key={i} value={i}>v{i}</option>)}</select></label>
      <NumberField label="Assignment index" value={index} onChange={setIndex} min={0} max={store.size - 1} />
      <NumberField label="New integer value" value={value} onChange={setValue} min={-99} max={99} />
    </div>
    <button onClick={() => apply(() => {
      if (!index.trim() || !value.trim()) throw new Error('Enter both index and value.');
      const result = assignPersistent(store, Number(source), Number(index), Number(value));
      const version = result.store.versions.length - 1;
      setStore(result.store);
      setSelected(String(version));
      setRetained([...retained, version]);
      setNewIds(result.copied.map(pair => pair[1]));
      setActiveQuery({
        version,
        low: 0,
        high: store.size
      });
      setLow('0');
      setHigh(String(store.size));
      setQuery(persistentQuery(result.store, version, 0, store.size));
      setNotice(`Created v${version} from v${source}. ${result.copied.length} new nodes; v${source} remains unchanged.`);
    })}>Create branch</button>
    <p role="status">{notice}</p>
    <div className="persistent-controls">
      <label>Query version<select aria-label="Query version" value={selected} onChange={event => setSelected(event.target.value)}>{store.versions.map((_, i) => <option key={i} value={i}>v{i}</option>)}</select></label>
      <NumberField label="Range start, inclusive" value={low} onChange={setLow} min={0} max={store.size} />
      <NumberField label="Range end, exclusive" value={high} onChange={setHigh} min={0} max={store.size} />
    </div>
    <button onClick={() => apply(() => {
      if (!low.trim() || !high.trim()) throw new Error('Enter both range endpoints.');
      const next = persistentQuery(store, Number(selected), Number(low), Number(high));
      setQuery(next);
      setActiveQuery({
        version: Number(selected),
        low: Number(low),
        high: Number(high)
      });
    })}>Run historical query</button>
    <p className="persistent-result">Active query: v{activeQuery.version} [{activeQuery.low},{activeQuery.high}) → <strong>{query.sum}</strong>; cover {query.cover.map(id => `n${id}`).join(' + ') || 'empty'}.</p>
    <PhysicalGraph store={store} versions={shown} queryCover={query.cover} newIds={newIds} />
    <details>
      <summary>Inspect physical identities and version ancestry</summary>
      <div className="persistent-scroll" tabIndex={0}><table>
          <caption>Each row is one immutable physical object</caption>
          <thead><tr>
              <th>ID</th>
              <th>Interval</th>
              <th>Sum</th>
              <th>Left</th>
              <th>Right</th>
            </tr></thead>
          <tbody>{store.nodes.map(node => <tr key={node.id}>
              <th>n{node.id}</th>
              <td>[{node.low},{node.high})</td>
              <td>{node.sum}</td>
              <td>{node.left === null ? '—' : `n${node.left}`}</td>
              <td>{node.right === null ? '—' : `n${node.right}`}</td>
            </tr>)}</tbody>
        </table></div>
      <ul>{store.versions.map((version, i) => <li key={i}>v{i} → root n{version.root}; {version.parent === null ? 'initial' : `parent version v${version.parent}`}; {version.label}.</li>)}</ul>
    </details>
    <details>
      <summary>Which nodes survive if we release root handles?</summary>
      <p>Toggle conceptual external references. The union of their reachable objects counts each shared node once. This browser keeps the arena for inspection; this is an ownership calculation, not a measurement of actual garbage collection.</p>
      <div className="persistent-handles">{store.versions.map((_, i) => <label key={i}><input type="checkbox" checked={retained.includes(i)} onChange={() => setRetained(retained.includes(i) ? retained.filter(id => id !== i) : [...retained, i])} />Keep v{i}</label>)}</div>
      <p className="persistent-result">{reachable.size} reachable / {store.nodes.length} allocated. {store.nodes.length - reachable.size} would be reclaimable with no other references.</p>
      <div className="persistent-allocation" aria-label="Reachable physical node identities">{store.nodes.map(node => <span key={node.id} className={reachable.has(node.id) ? 'live' : 'released'}>n{node.id} {reachable.has(node.id) ? 'kept' : 'free'}</span>)}</div>
    </details>
    <details>
      <summary>Try a different initial array</summary>
      <label>1–8 integers, each −99…99<input value={draft} onChange={event => setDraft(event.target.value)} /></label>
      <button onClick={() => apply(() => reset(parsePersistentValues(draft)))}>Build fresh array</button>
    </details>
    <button onClick={() => apply(() => {
      reset();
      setDraft(DEFAULT_PERSISTENT_VALUES.join(', '));
    })}>Reset versions</button>
    {error && <p role="alert">{error} The active model is unchanged.</p>}
  </section>;
}
export function HistoryLookupLab() {
  const [index, setIndex] = useState('0');
  const [snapshot, setSnapshot] = useState('2');
  const history = HISTORY_WRITES[Number(index)];
  const result = historyLookup(history, Number(snapshot));
  return <section className="persistent-lab" aria-label="Per-index historical predecessor lookup">
    <h3>Find the last write that had happened</h3>
    <div className="persistent-controls">
      <label>Array index<select aria-label="Array index" value={index} onChange={event => setIndex(event.target.value)}>{[0, 1, 2].map(i => <option key={i}>{i}</option>)}</select></label>
      <label>Saved snapshot<select aria-label="Saved snapshot" value={snapshot} onChange={event => setSnapshot(event.target.value)}>{[0, 1, 2, 3, 4].map(i => <option key={i}>{i}</option>)}</select></label>
    </div>
    <div className="persistent-timeline" aria-label="Write history ordered by snapshot">{history.map(([time, value], i) => <div key={time} className={i === result.position ? 'chosen' : time > Number(snapshot) ? 'future' : ''}>
        <span>snapshot {time}</span>
        <strong>{value}</strong>
        <small>{i === result.position ? 'last eligible' : time > Number(snapshot) ? 'too late' : 'earlier write'}</small>
      </div>)}</div>
    <p className="persistent-result">get(index {index}, snapshot {snapshot}) = <strong>{result.value}</strong></p>
    <p>The selected write stays valid until the next write for this same index. Snapshot 4 need not store another copy of every value. Index 2 has never changed.</p>
    <details>
      <summary>Inspect the binary-search comparisons</summary>
      <ol>{result.trace.map((step, i) => <li key={i}>Search history positions [{step.low},{step.high}); middle {step.middle} has time {history[step.middle][0]}. {step.eligible ? 'Eligible: search to its right for a later eligible write.' : 'Too late: discard it and everything to its right.'}</li>)}</ol>
    </details>
    <button onClick={() => {
      setIndex('0');
      setSnapshot('2');
    }}>Reset lookup</button>
  </section>;
}
export function PrefixRankLab() {
  const [draft, setDraft] = useState('5, 1, 4, 1, 3, 5');
  const [low, setLow] = useState('1');
  const [high, setHigh] = useState('5');
  const [rank, setRank] = useState('3');
  const [active, setActive] = useState({
    values: [5, 1, 4, 1, 3, 5],
    low: 1,
    high: 5,
    rank: 3
  });
  const [model, setModel] = useState(() => prefixRankModel(active.values, 1, 5, 3));
  const [step, setStep] = useState(0);
  const [error, setError] = useState('');
  const current = model.steps[step];
  function apply() {
    try {
      if (![low, high, rank].every(value => value.trim())) throw new Error('Enter both endpoints and a rank.');
      const next = {
        values: parsePersistentValues(draft),
        low: Number(low),
        high: Number(high),
        rank: Number(rank)
      };
      setModel(prefixRankModel(next.values, next.low, next.high, next.rank));
      setActive(next);
      setStep(0);
      setError('');
    } catch (failure) {
      setError(failure.message);
    }
  }
  return <section className="persistent-lab" aria-label="Prefix history order statistic investigation">
    <h3>Subtract histories, then follow the rank</h3>
    <label>Array values<input value={draft} onChange={event => setDraft(event.target.value)} /></label>
    <div className="persistent-controls">
      <NumberField label="Subarray start" value={low} onChange={setLow} min={0} max={7} />
      <NumberField label="Subarray end, exclusive" value={high} onChange={setHigh} min={1} max={8} />
      <NumberField label="Rank, starting at 1" value={rank} onChange={setRank} min={1} max={8} />
    </div>
    <button onClick={apply}>Apply query</button>
    <p>Active array: {active.values.join(', ')}. Subarray [{active.low},{active.high}) = {active.values.slice(active.low, active.high).join(', ')}; requested rank {active.rank}.</p>
    <div className="persistent-scroll" tabIndex={0}><table className="persistent-histogram">
        <caption>Value buckets: later prefix − earlier prefix = subarray occurrences</caption>
        <thead><tr>
            <th>Value</th>
            {model.alphabet.map(value => <th key={value}>{value}</th>)}
          </tr></thead>
        <tbody>{[['prefix ' + active.high, model.later], ['− prefix ' + active.low, model.earlier], ['= subarray', model.counts]].map(([label, counts]) => <tr key={label}>
            <th>{label}</th>
            {counts.map((count, i) => <td key={i} className={i >= current.begin && i < current.end ? 'active-bucket' : ''}>
              <span aria-hidden="true">{'●'.repeat(count) || '·'}</span>
              <strong>{count}</strong>
            </td>)}
          </tr>)}</tbody>
      </table></div>
    <p className="persistent-result" role="status">{current.direction === 'answer' ? `One value bucket remains: ${current.value}. That is occurrence rank ${active.rank} in the original subarray.` : `Current value interval [${model.alphabet[current.begin]}, ${model.alphabet[current.end - 1]}], local rank ${current.k}. Left half contains ${current.leftCount} occurrences. ${current.direction === 'left' ? `Keep rank ${current.k} and descend left.` : `Skip those ${current.leftCount}; descend right with rank ${current.k - current.leftCount}.`}`}</p>
    <div className="persistent-controls">
      <button disabled={step === 0} onClick={() => setStep(step - 1)}>Previous rank step</button>
      <button disabled={step === model.steps.length - 1} onClick={() => setStep(step + 1)}>Next rank step</button>
      <button onClick={() => setStep(0)}>Reset descent</button>
    </div>
    <p>The buckets come from exact integer frequencies. The Python implementation obtains a half's count from two tree nodes; this small browser model also shows the expanded histogram so you can check the subtraction directly.</p>
    {error && <p role="alert">{error} The active query is unchanged.</p>}
  </section>;
}
