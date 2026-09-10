import "./workflow-figures.css";

export function ThreadRacePicture({ state, advance }) {
  return <figure className="wv-figure wv-lab-figure" data-visual="thread-execution-lanes">
    <div className="wv-shared-counter"><span>One shared counter</span><output>{state.value}</output><small>{state.locked ? `Shared lock: ${state.owner ? `held by ${state.owner}` : 'available'}` : 'No lock protects this operation'}</small></div>
    <div className="wv-thread-connectors"><span>↙ read from / write back ↗</span><span>↙ read from / write back ↗</span></div>
    <div className="wv-worker-lanes">{Object.entries(state.workers).map(([id, worker]) => {
      const waiting = state.locked && state.owner && state.owner !== id && worker.phase < 3;
      return <div className="wv-worker" key={id} data-waiting={Boolean(waiting)}><h4>Worker {id}</h4><div className="wv-local-value"><span>Private snapshot</span><output>{worker.local ?? '—'}</output></div>
        <ol className="wv-operation-lane">{['Read shared value', 'Compute local + 1', 'Write local value'].map((label, index) => <li key={label} data-phase={index < worker.phase ? 'done' : index === worker.phase ? 'next' : 'pending'}><span>{index < worker.phase ? '✓' : index === worker.phase ? '→' : '·'}</span><span>{label}<small>{index < worker.phase ? 'done' : index === worker.phase ? waiting ? 'waiting for lock' : 'next operation' : 'pending'}</small></span></li>)}</ol>
        <button type="button" disabled={worker.phase === 3} onClick={() => advance(id)}>Advance {id}</button><p className="wv-worker-status">{worker.phase === 3 ? 'Finished' : waiting ? `Cannot enter: ${state.owner} holds the lock` : 'Choose when this worker advances'}</p>
      </div>;
    })}</div>
    <figcaption>Each lane keeps its own progress and copied value. Both lanes write to the single counter above. A compute step changes only that worker's copy; a write can overwrite an earlier result. Position shows program order, not elapsed time.</figcaption>
  </figure>;
}
