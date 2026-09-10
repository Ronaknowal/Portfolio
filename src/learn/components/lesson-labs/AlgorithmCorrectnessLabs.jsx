import { useState } from 'react';
import { compactionTrace, euclidTrace, parseProofValues, partitionTrace, searchObligations, searchProofTrace } from '../../data/algorithm-correctness-models.js';
import './algorithm-correctness-labs.css';
function StepControls({
  position,
  count,
  setPosition,
  reset
}) {
  return <div className="proof-controls">
    <button type="button" onClick={() => setPosition(position - 1)} disabled={position === 0}>Previous state</button>
    <button type="button" onClick={() => setPosition(position + 1)} disabled={position === count - 1}>Next state</button>
    <button type="button" onClick={() => setPosition(count - 1)} disabled={position === count - 1}>Finish trace</button>
    <button type="button" onClick={reset}>Reset investigation</button>
    <span>State {position + 1} of {count}</span>
  </div>;
}
function Check({
  valid,
  children
}) {
  return <span className={valid ? 'proof-check is-valid' : 'proof-check is-false'}>
    {valid ? '✓ ' : '✗ '}{children}
  </span>;
}
function validateTarget(draft) {
  if (!/^-?\d+$/.test(draft.trim())) throw new Error('Use an integer target from −20 to 20.');
  const target = Number(draft);
  if (target < -20 || target > 20) throw new Error('Use an integer target from −20 to 20.');
  return target;
}
export function SearchInvariantLab() {
  const [draft, setDraft] = useState('4, 9, 2, 9');
  const [targetDraft, setTargetDraft] = useState('9');
  const [values, setValues] = useState([4, 9, 2, 9]);
  const [target, setTarget] = useState(9);
  const [skip, setSkip] = useState(false);
  const [claim, setClaim] = useState('prefix');
  const [position, setPosition] = useState(0);
  const [error, setError] = useState('');
  const states = searchProofTrace(values, target, skip);
  const state = states[position];
  const obligations = searchObligations(values, target, claim, skip);
  const expected = values.indexOf(target);
  function apply(event) {
    event.preventDefault();
    try {
      const nextValues = parseProofValues(draft);
      const nextTarget = validateTarget(targetDraft);
      setValues(nextValues);
      setTarget(nextTarget);
      setPosition(0);
      setError('');
    } catch (failure) {
      setError(failure.message);
    }
  }
  function reset() {
    setDraft('4, 9, 2, 9');
    setTargetDraft('9');
    setValues([4, 9, 2, 9]);
    setTarget(9);
    setSkip(false);
    setClaim('prefix');
    setPosition(0);
    setError('');
  }
  const witnessMessages = {
    initialization: witness => `At i=${witness.index}, the initialized state does not satisfy this claim.`,
    preservation: witness => `An admitted state i=${witness.index} advances to i=${witness.next}, where the claim is false.`,
    matchExit: witness => `An admitted state returns ${witness.index}, but an earlier match exists at ${witness.earlier}.`,
    absentExit: witness => `At admitted exit i=${witness.index}, return −1 is wrong: position ${witness.missed} matches.`
  };
  return <section className="proof-lab" data-lab="search-invariant" aria-label="Search invariant investigation">
    <h3>Which positions have actually been ruled out?</h3>
    <p>Predict the first match. Follow the growing rejected prefix, then try the faulty two-position jump. Changing the claim below changes the proof question, not the running algorithm. Use up to eight integers from −20 to 20; an empty list is allowed.</p>
    <form onSubmit={apply} className="proof-inputs">
      <label>Search values<input value={draft} onChange={event => setDraft(event.target.value)} placeholder="Empty list is allowed" /></label>
      <label>Target<input value={targetDraft} onChange={event => setTargetDraft(event.target.value)} /></label>
      <button type="submit">Apply search input</button>
    </form>
    {error && <p role="alert">{error} The last applied input remains active.</p>}
    <label className="proof-checkbox"><input type="checkbox" checked={skip} onChange={event => {
        setSkip(event.target.checked);
        setPosition(0);
      }} />Try the faulty jump: i = min(i + 2, n)</label>
    <p className="proof-state-label">Target {target} · i = {state.index} · n − i = {state.variant}</p>
    <div className="proof-array-scroll" tabIndex={0} role="region" aria-label="Rejected and unexamined search positions">
      <div className="proof-array">{values.map((value, index) => <div key={index} className={'proof-cell ' + (index < state.index ? value === target ? 'is-false' : 'is-known' : 'is-unknown') + (index === state.index ? ' is-current' : '')}>
        <small>index {index}</small><strong>{value}</strong><small>{index < state.index ? 'rejected' : index === state.index ? 'next' : 'unknown'}</small>
      </div>)}{values.length === 0 && <p>Empty list: the rejected prefix and remaining region are both empty.</p>}</div>
    </div>
    <p data-result="search-action">{state.action}</p>
    <Check valid={state.prefixValid}>No rejected position contains the target</Check>
    {state.finished && <p data-result="search-result">Returned {state.result}. <Check valid={state.result === expected}>First-occurrence contract {state.result === expected ? 'holds' : 'fails'}</Check></p>}
    <StepControls position={position} count={states.length} setPosition={setPosition} reset={reset} />
    <div className="proof-obligations">
      <label>Candidate invariant<select aria-label="Candidate invariant" value={claim} onChange={event => setClaim(event.target.value)}>
        <option value="prefix">Bounds and no match before i</option>
        <option value="bounds">Bounds only: 0 ≤ i ≤ n</option>
        <option value="whole">Bounds and no match anywhere</option>
      </select></label>
      <p>This finite check examines all {obligations.examined} index boundaries for the applied list, including hypothetical states permitted by the claim. It does not prove the algorithm for all lists. A failed initialization makes later conditional checks insufficient.</p>
      <dl>{[['initialization', 'Initialize'], ['preservation', 'Preserve across a continuing step'], ['matchExit', 'Justify a matching return'], ['absentExit', 'Justify an absent return']].map(([key, label]) => <div key={key}><dt>{label}</dt><dd className={obligations[key] ? 'proof-failure' : ''}>
        {obligations[key] ? witnessMessages[key](obligations[key]) : 'No counterexample among the applicable states for this input.'}
      </dd></div>)}</dl>
    </div>
    <details><summary>Try a changed proof</summary><p>Choose bounds only. The real correct trace still works, but the claim permits i=n even when a match exists. It cannot justify the failure return. Now use the prefix claim with the faulty jump: preservation exposes the missed match. Restore the ordinary update and try an absent target and an empty list.</p></details>
  </section>;
}
function OccurrenceRow({
  items,
  label,
  read,
  write,
  active,
  zone
}) {
  return <div className="proof-array-scroll" tabIndex={0} role="region" aria-label={label}>
    <p className="proof-row-label">{label}</p>
    <div className="proof-array">{items.map((item, index) => <div key={index} className={'proof-cell ' + (zone ? zone(index) : '') + (active?.includes(index) ? ' is-current' : '')}>
      <small>slot {index}</small><strong>{item.value}</strong><small>from #{item.origin}</small>
      {(read === index || write === index) && <span className="proof-pointer">{read === index ? 'read ' : ''}{write === index ? 'write' : ''}</span>}
    </div>)}{items.length === 0 && <p>No occurrences. The empty regions already meet the exit contract.</p>}</div>
  </div>;
}
export function CompactionInvariantLab() {
  const [draft, setDraft] = useState('5, 0, 5, 2, 0, 7');
  const [removedDraft, setRemovedDraft] = useState('0');
  const [values, setValues] = useState([5, 0, 5, 2, 0, 7]);
  const [removed, setRemoved] = useState(0);
  const [position, setPosition] = useState(0);
  const [error, setError] = useState('');
  const trace = compactionTrace(values, removed);
  const state = trace.states[position];
  function apply(event) {
    event.preventDefault();
    try {
      const nextValues = parseProofValues(draft);
      const nextRemoved = validateTarget(removedDraft);
      setValues(nextValues);
      setRemoved(nextRemoved);
      setPosition(0);
      setError('');
    } catch (failure) {
      setError(failure.message);
    }
  }
  function reset() {
    setDraft('5, 0, 5, 2, 0, 7');
    setRemovedDraft('0');
    setValues([5, 0, 5, 2, 0, 7]);
    setRemoved(0);
    setPosition(0);
    setError('');
  }
  return <section className="proof-lab" data-lab="compaction-invariant" aria-label="Stable compaction invariant investigation">
    <h3>Overwrite only storage that is no longer needed</h3>
    <p>Each # label is an original occurrence, so the two 5s remain distinguishable. Predict the destination of the next kept occurrence. The upper row is a teaching reference; the native algorithm does not make that copy. Use up to eight integers from −20 to 20, or an empty list.</p>
    <form onSubmit={apply} className="proof-inputs">
      <label>Compaction values<input value={draft} onChange={event => setDraft(event.target.value)} /></label>
      <label>Value to remove<input value={removedDraft} onChange={event => setRemovedDraft(event.target.value)} /></label>
      <button type="submit">Apply compaction input</button>
    </form>
    {error && <p role="alert">{error} The last input remains active.</p>}
    <OccurrenceRow items={trace.original} label="Original occurrences · fixed proof reference" active={state.source === null ? [] : [state.source]} />
    <p className="proof-state-label">read = {state.read} · write = {state.write} · n − read = {state.variant}</p>
    <OccurrenceRow items={state.working} label="Actual array · kept prefix / reusable gap / unread suffix" read={state.read} write={state.write} active={state.destination === null ? [] : [state.destination]} zone={index => index < state.write ? 'is-known' : index < state.read ? 'is-unused' : 'is-unknown'} />
    <p data-result="compaction-action">{state.action}</p>
    <div className="proof-checks"><Check valid={state.boundsValid}>write ≤ read ≤ n</Check><Check valid={state.prefixValid}>Kept prefix has exactly the retained original occurrences in order</Check><Check valid={state.unreadValid}>Unread suffix is unchanged</Check></div>
    <p data-result="compaction-result">Logical result so far: [{state.working.slice(0, state.write).map(item => item.value).join(', ')}]. {state.read === values.length ? `Return ${state.write}; slots from ${state.write} onward are unspecified by this contract.` : 'It describes only the original positions already processed.'}</p>
    <StepControls position={position} count={trace.states.length} setPosition={setPosition} reset={reset} />
    <details><summary>Transfer: are trailing duplicate values an error?</summary><p>No. This contract returns a count and a valid prefix; it does not shorten the physical list. A second zero-filling loop can establish a stronger full-list result without changing the retained prefix. Compare all-kept, all-removed and empty inputs.</p></details>
  </section>;
}
export function PartitionInvariantLab() {
  const [draft, setDraft] = useState('1, 2, 0');
  const [values, setValues] = useState([1, 2, 0]);
  const [faulty, setFaulty] = useState(false);
  const [position, setPosition] = useState(0);
  const [error, setError] = useState('');
  const states = partitionTrace(values, faulty);
  const state = states[position];
  function apply(event) {
    event.preventDefault();
    try {
      setValues(parseProofValues(draft, {
        categories: true
      }));
      setPosition(0);
      setError('');
    } catch (failure) {
      setError(failure.message);
    }
  }
  function reset() {
    setDraft('1, 2, 0');
    setValues([1, 2, 0]);
    setFaulty(false);
    setPosition(0);
    setError('');
  }
  function zone(index) {
    if (index < state.low) return 'is-zero';
    if (index < state.middle) return 'is-one';
    if (index < state.high) return 'is-unknown';
    return 'is-two';
  }
  return <section className="proof-lab" data-lab="partition-invariant" aria-label="Four-region partition investigation">
    <h3>What do you know about the swapped-in occurrence?</h3>
    <p>The regions represent established claims, not just colors in the input. Follow the unknown region until it vanishes; then compare the faulty extra increment on the same input. Use up to eight values, each 0, 1 or 2; an empty list is allowed.</p>
    <form onSubmit={apply} className="proof-inputs">
      <label>Partition categories<input value={draft} onChange={event => setDraft(event.target.value)} /></label>
      <button type="submit">Apply partition input</button>
    </form>
    {error && <p role="alert">{error} The last valid categories remain active.</p>}
    <label className="proof-checkbox"><input type="checkbox" checked={faulty} onChange={event => {
        setFaulty(event.target.checked);
        setPosition(0);
      }} />Try the faulty increment after swapping from high</label>
    <p className="proof-state-label">low = {state.low} · middle = {state.middle} · high = {state.high} · {state.variant < 0 ? `high − middle = ${state.variant}: invalid region boundaries` : `unknown count = ${state.variant}`}</p>
    <div className="proof-zone-key"><span className="is-zero">0s: [0,{state.low})</span><span className="is-one">1s: [{state.low},{state.middle})</span><span className="is-unknown">Unknown: [{state.middle},{state.high})</span><span className="is-two">2s: [{state.high},{values.length})</span></div>
    <OccurrenceRow items={state.working} label="Category occurrences · region boundaries above are exclusive on the right" active={state.changed} zone={zone} />
    <p data-result="partition-action">{state.action}</p>
    <div className="proof-checks">{[['bounds', 'Ordered boundaries'], ['zero', 'Every claimed 0 is a 0'], ['one', 'Every claimed 1 is a 1'], ['two', 'Every claimed 2 is a 2'], ['occurrences', 'Every original occurrence remains exactly once']].map(([key, label]) => <Check key={key} valid={state.checks[key]}>{label}</Check>)}</div>
    {position === states.length - 1 && <p data-result="partition-result">Trace ended: [{state.working.map(item => item.value).join(', ')}]. {Object.values(state.checks).every(Boolean) ? 'The invariant and empty unknown region establish the ordered result.' : 'The loop ended, but its invariant failed: termination did not establish correctness.'}</p>}
    <StepControls position={position} count={states.length} setPosition={setPosition} reset={reset} />
    <details><summary>Transfer: why is the other swap different?</summary><p>When low &lt; middle, the occurrence at low is already known to be a 1. Moving it to middle is safe before both boundaries advance. The occurrence at high−1 comes from the unknown region, so the same argument is unavailable.</p></details>
  </section>;
}
export function EuclidTerminationLab() {
  const [firstDraft, setFirstDraft] = useState('84');
  const [secondDraft, setSecondDraft] = useState('30');
  const [pair, setPair] = useState([84, 30]);
  const [position, setPosition] = useState(0);
  const [error, setError] = useState('');
  const states = euclidTrace(...pair);
  const state = states[position];
  function apply(event) {
    event.preventDefault();
    try {
      if (![firstDraft, secondDraft].every(value => /^\d+$/.test(value.trim()))) throw new Error('Use two nonnegative integers from 0 to 96.');
      const next = [Number(firstDraft), Number(secondDraft)];
      euclidTrace(...next);
      setPair(next);
      setPosition(0);
      setError('');
    } catch (failure) {
      setError(failure.message);
    }
  }
  function reset() {
    setFirstDraft('84');
    setSecondDraft('30');
    setPair([84, 30]);
    setPosition(0);
    setError('');
  }
  return <section className="proof-lab" data-lab="euclid-termination" aria-label="Euclid invariant and termination investigation">
    <h3>Preserve the answer while shrinking the question</h3>
    <p>Predict the remainder, then advance to (b,remainder). The divisor list stays the same while the second coordinate decreases. Use integers from 0 to 96 with at least one positive value, so every divisor stays visible; (0,0) is handled separately in the native example.</p>
    <form onSubmit={apply} className="proof-inputs">
      <label>First integer<input value={firstDraft} onChange={event => setFirstDraft(event.target.value)} /></label>
      <label>Second integer<input value={secondDraft} onChange={event => setSecondDraft(event.target.value)} /></label>
      <button type="submit">Apply Euclid input</button>
    </form>
    {error && <p role="alert">{error} The last pair remains active.</p>}
    <p className="proof-division" data-result="euclid-equation">{state.terminal ? `b = 0: return a = ${state.a}` : `${state.a} = ${state.quotient} × ${state.b} + ${state.remainder}`}</p>
    {!state.terminal && <>
      <div className="proof-remainder-bar" role="img" aria-label={`${state.quotient * state.b} units in whole multiples of ${state.b}, followed by remainder ${state.remainder}. Total ${state.a}.`}>
        <span className="proof-multiples" style={{
          width: `${100 * state.quotient * state.b / Math.max(1, state.a)}%`
        }} />
        <span className="proof-remainder" style={{
          width: `${100 * state.remainder / Math.max(1, state.a)}%`
        }} />
      </div>
      <p>Whole multiples: {state.quotient * state.b}. Remainder: {state.remainder}. {state.a === 0 ? 'The dividend is zero, so both contributions have zero length.' : `Bar lengths are exact portions of a=${state.a}.`} Each new row has its own total, so this is not a cross-step magnitude chart.</p>
      <p data-result="euclid-descent">Next pair ({state.b},{state.remainder}); 0 ≤ {state.remainder} &lt; {state.b}. The measure b strictly decreases.</p>
    </>}
    <p>Positive common divisors of the current pair:</p>
    <div className="proof-divisors" data-result="euclid-divisors">{state.commonDivisors.map(divisor => <span key={divisor}>{divisor}</span>)}</div>
    <p>Compare with the original pair ({pair.join(', ')}): [{states[0].commonDivisors.join(', ')}]. Preserving these divisors preserves their greatest member.</p>
    <p className="proof-measure-history">Second-coordinate history: {states.slice(0, position + 1).map(entry => entry.b).join(' → ')}</p>
    <StepControls position={position} count={states.length} setPosition={setPosition} reset={reset} />
    <details><summary>Transfer: reverse the two starting inputs</summary><p>Try (30,84). The first quotient is zero and the remainder is 30. The measure still falls from 84 to 30. If b starts at zero, no division occurs and the loop already meets its return condition.</p></details>
  </section>;
}
export function ProofBoundaryFigure() {
  return <figure className="proof-inline" aria-label="Search proof follows initialization, continuing edges and separate exits">
    <div className="proof-flow-entry">Initialize i=0 → <strong>Boundary: no match before i</strong></div>
    <div className="proof-flow-branches">
      <div><strong>i = n</strong><span>All positions are rejected</span><b>Return −1</b></div>
      <div><strong>i &lt; n, a[i] = target</strong><span>Earlier positions rejected; current position matches</span><b>Return i</b></div>
      <div><strong>i &lt; n, a[i] ≠ target</strong><span>Reject this one position; i ← i+1</span><b>↶ Re-establish the boundary claim</b></div>
    </div>
    <figcaption>The preservation argument follows the continuing edge. Each return has its own obligation. The changing region is the meaning of the invariant; the current value of i is not constant.</figcaption>
  </figure>;
}
export function AssignmentHistoryFigure() {
  return <figure className="proof-inline" aria-label="Old-value dependency in a swap">
    <table className="proof-history"><caption>Start left=3, right=8</caption><thead><tr><th>Program point</th><th>Faulty overwrite</th><th>Save before replacing</th></tr></thead><tbody>
      <tr><th>Read old left</th><td>Not saved</td><td>saved=3</td></tr>
      <tr><th>left ← right</th><td>left=8, right=8</td><td>left=8, right=8, saved=3</td></tr>
      <tr><th>Assign right</th><td>right ← left gives 8</td><td>right ← saved gives 3</td></tr>
    </tbody></table>
    <figcaption>Once the only copy of 3 is overwritten, later assignments cannot recover it from left. The temporary is a dependency, not incidental syntax.</figcaption>
  </figure>;
}
export function DescentFigure() {
  return <figure className="proof-inline" aria-label="Strict real decrease versus lexicographic natural-number decrease">
    <div className="proof-descent"><strong>Positive exact reals</strong><p>1 → 1/2 → 1/4 → 1/8 → …</p><span>Strictly decreasing and bounded below, but never reaches 0. This alone cannot prove a loop with guard x&gt;0 stops.</span></div>
    <div className="proof-descent"><strong>Natural-number pairs, compared left first</strong><p>(2,1) → (2,0) → (1,2) → (1,1)</p><span>The second coordinate can reset upward when the first drops. With both coordinates nonnegative, this order admits no infinite descent.</span></div>
    <figcaption>The number system and comparison rule are part of a termination proof. Floating-point underflow would describe a different program from exact real halving.</figcaption>
  </figure>;
}
export function SortedContractFigure() {
  return <figure className="proof-inline" aria-label="Sorted output alone does not establish a sorting contract">
    <p>Original occurrences: <strong>[2,1,2]</strong></p>
    <div className="proof-contract-comparison"><div><strong>[1,2]</strong><span>Sorted: yes</span><span>All original multiplicities: no</span><b>One copy of 2 was lost</b></div><div><strong>[1,2,2]</strong><span>Sorted: yes</span><span>All original multiplicities: yes</span><b>Both obligations hold</b></div></div>
    <figcaption>A set comparison also loses multiplicity information. Correctness is relative to the whole specification, including what must be preserved.</figcaption>
  </figure>;
}
