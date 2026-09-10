import { useState } from 'react';
import { amplification, expectedSelectionWork, productFixtures, productProbe, rejectionMap, reservoirDistribution, reservoirTrace, sampleBudget, shuffleTrace, quickselectTrace } from '../../data/randomized-algorithm-models.js';
import './randomized-algorithm-labs.css';
const letters = 'ABCDEFGH';
const percent = value => `${(100 * value).toFixed(value < 0.01 ? 4 : 2)}%`;
function Cells({
  values,
  label,
  activeEnd,
  selected
}) {
  return <div className="random-cells" aria-label={label}>{values.map((value, index) => <span key={index} className={`${activeEnd !== undefined && index > activeEnd ? 'is-fixed' : ''} ${selected === index ? 'is-selected' : ''}`}>
      <small>{index}</small><strong>{value}</strong>{activeEnd !== undefined && index > activeEnd && <small>fixed</small>}
    </span>)}</div>;
}
export function ProbabilityTreeFigure() {
  return <figure className="random-figure" data-figure="probability-tree">
    <div className="random-tree"><strong>Two independent fair bits</strong><div><span>first 0 · 1/2</span><span>first 1 · 1/2</span></div><div>{['00', '01', '10', '11'].map(bits => <span key={bits}>{bits}<small>1/4</small></span>)}</div></div>
    <figcaption>Four equally likely paths. The event “at least one 1” contains 01, 10 and 11: probability 3/4. Exactly one 1 contains two paths: probability 1/2.</figcaption>
  </figure>;
}
export function UniformChoiceLab() {
  const [target, setTarget] = useState(3);
  const [reject, setReject] = useState(true);
  const model = rejectionMap(8, target, reject);
  return <section className="random-lab" data-lab="uniform-choice" aria-label="Uniform integer mapping investigation">
    <p className="lesson-eyebrow">MAP THE RAW OUTCOMES</p>
    <h3>Which output gets an extra ticket?</h3>
    <p>Predict the counts, then change the target range. Each of the eight raw cells has probability 1/8 before rejection.</p>
    <div className="random-controls"><label>Number of outputs<select value={target} onChange={event => setTarget(Number(event.target.value))}>{[2, 3, 4, 5, 6, 7].map(value => <option key={value}>{value}</option>)}</select></label><label>Mapping<select aria-label="Mapping" value={reject ? 'reject' : 'modulo'} onChange={event => setReject(event.target.value === 'reject')}><option value="reject">Reject leftover raw cells</option><option value="modulo">Modulo only — inspect the bias</option></select></label><button onClick={() => {
        setTarget(3);
        setReject(true);
      }}>Reset</button></div>
    <div className="random-mapping">{model.outcomes.map(({
        raw,
        result
      }) => <div key={raw} className={result === null ? 'is-rejected' : ''}><small>raw {raw}</small><span aria-hidden="true">↓</span><strong>{result === null ? 'retry' : `out ${result}`}</strong></div>)}</div>
    <div className="random-bars" aria-label="Exact output probabilities">{model.counts.map((count, index) => <div key={index}><span>output {index}</span><i style={{
          width: `${count / model.accepted * 100}%`
        }} /><strong>{count}/{model.accepted}</strong></div>)}</div>
    <p aria-live="polite" data-result="mapping">{model.accepted}/8 raw outcomes accepted. Expected attempts: {model.expectedAttempts.toFixed(3)}. {reject ? 'These probabilities condition on acceptance; rejection retries with fresh independent bits.' : 'These are the unconditional modulo probabilities. Equal-looking output labels do not imply equal probability.'}</p>
    <p className="lesson-note">Exact enumeration of eight outcomes, not a measured histogram. Changing the target resets the mathematical experiment immediately; no random trials are hidden.</p>
  </section>;
}
export function WeightedTicketsFigure() {
  return <figure className="random-figure" data-figure="weighted-tickets">
    <div className="random-ticket-line">{['A', 'B', 'B', 'B', 'C', 'C'].map((label, ticket) => <span key={ticket} className={`ticket-${label}`}><small>{ticket}</small><strong>{label}</strong></span>)}</div>
    <div className="random-intervals"><span>A: [0,1) · 1 ticket</span><span>B: [1,4) · 3 tickets</span><span>C: [4,6) · 2 tickets</span></div>
    <figcaption>Weights [1,3,2] give six equally likely integer tickets. Prefix ends [1,4,6] route a ticket to the first end strictly greater than it. Ticket 4 belongs to C, not B.</figcaption>
  </figure>;
}
export function ShuffleLab() {
  const [choices, setChoices] = useState([]);
  const [choice, setChoice] = useState(0);
  const state = shuffleTrace(choices).at(-1);
  const done = state.activeEnd < 1;
  function reset() {
    setChoices([]);
    setChoice(0);
  }
  return <section className="random-lab" data-lab="shuffle" aria-label="Fisher Yates shuffle investigation">
    <p className="lesson-eyebrow">FIX ONE POSITION</p><h3>Keep the completed suffix out of the next draw</h3>
    <p>{done ? "The shuffle is complete. Reset to explore a different path." : `Choose which active index swaps into position ${state.activeEnd}.`} Your manual choice follows one possible path; uniform sampling would choose each permitted index equally.</p>
    <Cells values={state.values} activeEnd={done ? -1 : state.activeEnd} label="Active prefix and completed suffix" selected={done ? undefined : choice} />
    <div className="random-controls"><label>Chosen active index<select value={choice} disabled={done} onChange={event => setChoice(Number(event.target.value))}>{Array.from({
            length: Math.max(1, state.activeEnd + 1)
          }, (_, index) => <option key={index} value={index}>{index} · {state.values[index]}</option>)}</select></label><button disabled={done} onClick={() => {
        setChoices([...choices, choice]);
        setChoice(0);
      }}>Swap and fix</button><button disabled={!choices.length} onClick={() => {
        setChoices(choices.slice(0, -1));
        setChoice(0);
      }}>Previous</button><button onClick={reset}>Reset</button></div>
    <p aria-live="polite" data-result="shuffle">{done ? `Complete permutation: ${state.values.join('')}.` : `${4 - state.activeEnd - 1} positions fixed; ${state.activeEnd + 1} equally eligible occurrences remain.`} Path: {choices.length ? choices.join(' → ') : 'no draws yet'}.</p>
    <p className="lesson-note">Each complete labeled permutation has probability 1/(4×3×2)=1/24 under independent uniform draws. A value staying where it was is a valid outcome; excluding self-swaps changes the distribution.</p>
  </section>;
}
export function SubsetCounterexampleFigure() {
  return <figure className="random-figure" data-figure="subset-counterexample"><div className="random-pair-samples"><div><strong>A B</strong><span>probability 1/2</span></div><div><strong>C D</strong><span>probability 1/2</span></div></div><figcaption>Every item appears with probability 1/2, yet AC, AD, BC and BD never appear. Uniform inclusion alone does not prove a uniform two-item subset; all six pairs should have probability 1/6.</figcaption></figure>;
}
export function ReservoirLab() {
  const [capacity, setCapacity] = useState(2);
  const [choices, setChoices] = useState([]);
  const [draw, setDraw] = useState(0);
  const state = reservoirTrace(capacity, choices).at(-1);
  const done = state.seen === 6;
  const distribution = reservoirDistribution(state.seen, capacity);
  function reset() {
    setChoices([]);
    setDraw(0);
  }
  return <section className="random-lab" data-lab="reservoir" aria-label="Reservoir sampling investigation">
    <p className="lesson-eyebrow">A SMALL MEMORY FOR A GROWING STREAM</p><h3>Let every seen position compete for a slot</h3>
    <p>{done ? "All six records have arrived. Reset to compare another path." : `Predict whether the next record enters. Draw j uniformly from 0 through ${state.seen}; only j less than k replaces a slot.`} Here you select a possible j to inspect the mechanism.</p>
    <div className="random-controls"><label>Reservoir size k<select value={capacity} onChange={event => {
          setCapacity(Number(event.target.value));
          reset();
        }}>{[1, 2, 3].map(value => <option key={value}>{value}</option>)}</select></label><label>Next draw j<select value={draw} disabled={done} onChange={event => setDraw(Number(event.target.value))}>{Array.from({
            length: state.seen + 1
          }, (_, value) => <option key={value} value={value}>{value}{value < capacity ? ` → slot ${value}` : ' → discard'}</option>)}</select></label><button disabled={done} onClick={() => {
        setChoices([...choices, draw]);
        setDraw(0);
      }}>Process next record</button><button disabled={!choices.length} onClick={() => {
        setChoices(choices.slice(0, -1));
        setDraw(0);
      }}>Previous</button><button onClick={reset}>Reset</button></div>
    <div className="random-stream">{'ABCDEF'.split('').map((letter, index) => <span key={letter} className={index < state.seen ? 'is-seen' : index === state.seen ? 'is-incoming' : ''}>{letter}<small>{index < state.seen ? 'seen' : index === state.seen ? 'next' : 'later'}</small></span>)}</div>
    <p>Reservoir slots (slot order is not sorted sample order)</p><Cells values={state.sample.map(index => letters[index])} label="Retained occurrence identities" />
    <p aria-live="polite" data-result="reservoir">Seen {state.seen}; retained {state.sample.map(index => letters[index]).join('')}. {state.draw === null ? 'Initial fill requires no random choice.' : state.draw < capacity ? `Draw ${state.draw} replaced ${letters[state.replaced]} with ${letters[state.seen - 1]}.` : `Draw ${state.draw} discarded ${letters[state.seen - 1]}.`}</p>
    <details><summary>Inspect all possible subsets at this prefix</summary><p>This enumerates every uniform draw branch, independently of your selected path. Each of {distribution.length} unordered subsets has probability {percent(distribution[0].probability)}.</p><div className="random-subsets">{distribution.map(entry => <span key={entry.sample.join(',')} className={entry.sample.join(',') === [...state.sample].sort((left, right) => left - right).join(',') ? 'is-selected' : ''}>{entry.sample.map(index => letters[index]).join('')}<small>{percent(entry.probability)}</small></span>)}</div></details>
    <p className="lesson-note">A–F identify occurrences. Repeated record values still occupy different stream positions. Displayed percentages are rounded. Finite branch enumeration supports this fixture; the induction in the text proves the general invariant.</p>
  </section>;
}
export function SelectionLab() {
  const [rank, setRank] = useState(4);
  const [choices, setChoices] = useState([]);
  const [choice, setChoice] = useState(0);
  const values = [8, 1, 6, 3, 9, 2, 7, 4, 5];
  const state = quickselectTrace(values, rank, choices).at(-1);
  const done = state.result !== null;
  function reset() {
    setChoices([]);
    setChoice(0);
  }
  return <section className="random-lab" data-lab="random-selection" aria-label="Randomized selection investigation">
    <p className="lesson-eyebrow">SAME ANSWER, DIFFERENT WORK</p><h3>Spend a partition only on the surviving side</h3>
    <p>Try extreme pivots, then reset and choose a central one. The work counter counts elements scanned by three-way partition, not milliseconds or individual Python comparisons.</p>
    <div className="random-controls"><label>Wanted sorted rank (zero-based)<select value={rank} onChange={event => {
          setRank(Number(event.target.value));
          reset();
        }}>{values.map((_, index) => <option key={index}>{index}</option>)}</select></label><label>Pivot occurrence<select value={choice} disabled={done} onChange={event => setChoice(Number(event.target.value))}>{(done ? [state.result] : state.active).map((value, index) => <option key={index} value={index}>index {index} · value {value}</option>)}</select></label><button disabled={done} onClick={() => {
        setChoices([...choices, choice]);
        setChoice(0);
      }}>Partition</button><button disabled={!choices.length} onClick={() => {
        setChoices(choices.slice(0, -1));
        setChoice(0);
      }}>Previous</button><button onClick={reset}>Reset</button></div>
    <Cells values={state.active} label="Current active values" selected={choice} />
    {state.pivot !== null && <div className="random-partitions">{[['less than', state.lower], ['equal to', state.equal], ['greater than', state.upper]].map(([label, partition]) => <div key={label}><small>{label} {state.pivot}</small><strong>{partition.join(', ') || 'empty'}</strong></div>)}</div>}
    <p data-result="selection" aria-live="polite">{done ? `Exact result ${state.result}.` : `Retained ${state.active.length} values; local rank ${state.wanted}.`} Scanned elements: {state.work}.</p>
    <p>Before any choices, uniform pivots on these nine distinct values give expected scanned work {expectedSelectionWork(9, rank).toFixed(3)}. {done ? 'Your selected path is complete.' : `Conditional expected additional work now: ${expectedSelectionWork(state.active.length, state.wanted).toFixed(3)}.`}</p>
    <p className="lesson-note">Expectations are computed over every possible future pivot rank by the recurrence in the text. They are averages over random choices for fixed data, not a bound on your chosen path. Sorted rank is defined using duplicates as separate occurrences in the native algorithm.</p>
  </section>;
}
function Matrix({
  values,
  label
}) {
  return <div className="random-matrix"><strong>{label}</strong><div>{values.map((row, index) => <span key={index}>{row.map((value, column) => <b key={column}>{value}</b>)}</span>)}</div></div>;
}
export function VerificationLab() {
  const [fixtureName, setFixtureName] = useState('cancellation');
  const [bits, setBits] = useState([1, 1]);
  const [rounds, setRounds] = useState(3);
  const [repeat, setRepeat] = useState(false);
  const state = productProbe(fixtureName, bits);
  const allProbes = [[0, 0], [0, 1], [1, 0], [1, 1]].map(probe => ({
    probe,
    passes: productProbe(fixtureName, probe).passes
  }));
  const singleFailure = allProbes.filter(probe => probe.passes).length / 4;
  const wrongProduct = fixtureName !== 'correct';
  return <section className="random-lab" data-lab="random-verification" aria-label="Matrix product verification investigation">
    <p className="lesson-eyebrow">PROBE THE CLAIM</p><h3>A passing probe is different from a proved product</h3>
    <div className="random-controls"><label>Claimed product<select value={fixtureName} onChange={event => setFixtureName(event.target.value)}>{Object.entries(productFixtures).map(([key, fixture]) => <option key={key} value={key}>{fixture.title}</option>)}</select></label>{bits.map((bit, index) => <label key={index}>Probe coordinate r{index}<select value={bit} onChange={event => setBits(bits.map((value, coordinate) => coordinate === index ? Number(event.target.value) : value))}><option>0</option><option>1</option></select></label>)}<button onClick={() => {
        setFixtureName('cancellation');
        setBits([1, 1]);
        setRounds(3);
        setRepeat(false);
      }}>Reset</button></div>
    <div className="random-matrices"><Matrix values={state.left} label="A" /><Matrix values={state.right} label="B" /><Matrix values={state.claimed} label="claimed C" /></div>
    <div className="random-probe-path"><div><span>r = [{bits.join(', ')}]</span><span>↓ multiply by B</span><strong>B r = [{state.intermediate.join(', ')}]</strong><span>↓ multiply by A</span><strong>A(B r) = [{state.actual.join(', ')}]</strong></div><div><span>same r</span><span>↓ multiply by claimed C</span><strong>C r = [{state.claimedProbe.join(', ')}]</strong><span>↓ subtract</span><strong>residual = [{state.residual.join(', ')}]</strong></div></div>
    <p aria-live="polite" data-result="verification">{state.passes ? 'Probe passes: equality has not been established by this single result.' : 'Mismatch found: this probe certifies that the integer product is wrong.'}</p>
    <div className="random-witnesses">{allProbes.map(({
        probe,
        passes
      }) => <button key={probe.join('')} onClick={() => setBits(probe)} aria-pressed={probe.join('') === bits.join('')}><strong>[{probe.join(',')}]</strong><span>{passes ? 'passes' : 'detects error'}</span></button>)}</div>
    <div className="random-controls"><label>Number of tests<select value={rounds} onChange={event => setRounds(Number(event.target.value))}>{[1, 2, 3, 5, 10].map(value => <option key={value}>{value}</option>)}</select></label><label>Randomness across tests<select value={repeat ? 'same' : 'independent'} onChange={event => setRepeat(event.target.value === 'same')}><option value="independent">Fresh independent vectors</option><option value="same">Reuse the same random vector</option></select></label></div>
    <p data-result="amplification">{wrongProduct ? `For this wrong product: single-test false acceptance ${percent(singleFailure)}; ${rounds} tests false acceptance ${percent(repeat ? singleFailure : singleFailure ** rounds)}.` : 'This product is correct. All probes pass; passing is not an error.'}</p>
    <p className="lesson-note">The four vectors are exhaustively evaluated with exact small integers. The selected vector is manual. Repetition readouts refer to the declared ideal random experiment, not to repeating your button clicks. Reducing an even integer error modulo 2 would hide it entirely; this model does not do that.</p>
  </section>;
}
export function AmplificationFigure() {
  const model = amplification(3, 0.25);
  return <figure className="random-figure" data-figure="majority-errors"><div className="random-bars">{model.distribution.map(({
        errors,
        probability
      }) => <div key={errors} className={errors >= 2 ? 'is-error-bar' : ''}><span>{errors} wrong votes</span><i style={{
          width: `${probability * 100}%`
        }} /><strong>{Math.round(probability * 64)}/64</strong></div>)}</div><figcaption>Three independent binary decisions, each wrong with probability 1/4. Majority is wrong on two or three errors: (9+1)/64=5/32. Requiring all to be wrong would give 1/64, which is the wrong event for majority.</figcaption></figure>;
}
export function SampleBudgetLab() {
  const [epsilon, setEpsilon] = useState(0.1);
  const [delta, setDelta] = useState(0.05);
  const budget = sampleBudget(epsilon, delta);
  return <section className="random-lab" data-lab="sample-budget" aria-label="Sampling accuracy budget investigation"><p className="lesson-eyebrow">CHOOSE PRECISION BEFORE SAMPLING</p><h3>What does halving the error tolerance cost?</h3><p>Estimate the mean of independent 0/1 observations with one common unknown mean. The tolerance ε is an absolute fraction, and δ bounds the chance of an error at least ε.</p><div className="random-controls"><label>Absolute tolerance ε<select value={epsilon} onChange={event => setEpsilon(Number(event.target.value))}>{[0.2, 0.1, 0.05, 0.02].map(value => <option key={value}>{value}</option>)}</select></label><label>Failure budget δ<select value={delta} onChange={event => setDelta(Number(event.target.value))}>{[0.1, 0.05, 0.01, 0.001].map(value => <option key={value}>{value}</option>)}</select></label><button onClick={() => {
        setEpsilon(0.1);
        setDelta(0.05);
      }}>Reset</button></div><div className="random-budget-pair"><div><span>tolerance ε = {epsilon}</span><strong>{budget} samples</strong></div><div><span>half the tolerance = {epsilon / 2}</span><strong>{sampleBudget(epsilon / 2, delta)} samples</strong></div></div><p data-result="budget" aria-live="polite">Sufficient budget: {budget}. At that fixed sample count, Hoeffding bounds the failure probability by {percent(Math.min(1, 2 * Math.exp(-2 * budget * epsilon ** 2)))}.</p><p className="lesson-note">Analytic upper bound, not an observed frequency or a minimum necessary sample count. Correlated duplicates do not create independent observations; repeatedly checking and stopping when the estimate looks good is a different procedure.</p></section>;
}
export function ReplayFigure() {
  return <figure className="random-figure" data-figure="replay-state"><div className="random-replay"><span>Fixed input + generator state</span><b>↓</b><span>Same algorithm + same draw order</span><b>↓</b><span>Replay the same choices</span></div><figcaption>Recording only a seed omits input order, generator/version, prior draws and scheduling. A reproduced mistake is still a mistake; replay is for diagnosis, not a proof of uniformity or independence.</figcaption></figure>;
}
