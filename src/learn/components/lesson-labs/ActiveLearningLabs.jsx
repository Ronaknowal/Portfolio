import { useMemo, useState } from 'react';
import { LessonTable } from './LessonElements.jsx';
import { CoveragePlane, EntropyBars, ProbabilityStrip, ThresholdRulers, formatActiveValue } from './ActiveLearningFigures.jsx';
import { activeLearningFixtures as fixtures, committeeDecomposition, compareNumbers, coveringDistances,
  entropyBatch, farthestFirst, parseNumericRows, thresholdQuestion, thresholdState,
  validateGeometry, validateThresholdInputs } from '../../data/active-learning-models.js';
import './active-learning.css';

const copy = value => structuredClone(value);
const numberInput = value => value === '' ? NaN : Number(value);
const near = (actual, predicted) => Math.abs(actual - predicted) <= 0.005 + 1e-12;

function ErrorMessage({ message }) {
  return message ? <p role="alert" className="active-error">{message}</p> : null;
}



export function ThresholdInvestigation() {
  const [thresholdText, setThresholdText] = useState(fixtures.hypotheses.map(row => `${row.threshold}, ${row.weight}`).join('\n'));
  const [observationText, setObservationText] = useState('-1, 0\n9, 1');
  const [queryText, setQueryText] = useState('0 1 2 3 4 5 6 7 8');
  const [oracleMode, setOracleMode] = useState('simulated');
  const [oracleThreshold, setOracleThreshold] = useState('5.5');
  const [manualAnswer, setManualAnswer] = useState('0');
  const [configuration, setConfiguration] = useState(null);
  const [history, setHistory] = useState([]);
  const [query, setQuery] = useState('4');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const clearAcquisition = () => { setResult(null); setError(''); };
  const editSetup = setter => event => { setter(event.target.value); setConfiguration(null); setHistory([]); clearAcquisition(); };
  const begin = () => {
    try {
      const hypotheses = parseNumericRows(thresholdText, 2, 'Threshold/weight').map(([threshold, weight]) => ({ threshold, weight })).sort((left, right) => left.threshold - right.threshold);
      const observations = observationText.trim() ? parseNumericRows(observationText, 2, 'Observation').map(([x, label]) => ({ x, label })) : [];
      const queries = queryText.trim().split(/[\s,]+/).map(value => numberInput(value));
      validateThresholdInputs(hypotheses, observations, queries);
      const trueThreshold = numberInput(oracleThreshold);
      if (oracleMode === 'simulated' && !hypotheses.some(row => row.threshold === trueThreshold)) throw new Error('Choose a simulated threshold from your candidate list.');
      if (oracleMode === 'simulated' && observations.some(row => Number(row.x >= trueThreshold) !== row.label)) throw new Error('Seed answers contradict the selected simulated threshold. Edit the seeds or use manual mode.');
      setConfiguration({ hypotheses, observations, queries, trueThreshold, oracleMode });
      setHistory([]);
      const unusedQueries = queries.filter(value => !observations.some(row => row.x === value));
      setQuery(unusedQueries.length ? String(unusedQueries.includes(4) ? 4 : unusedQueries[0]) : '');
      clearAcquisition();
    } catch (cause) { setError(cause.message); }
  };
  const observations = configuration ? [...configuration.observations, ...history.map(row => ({ x: row.query, label: row.answer }))] : [];
  const remaining = configuration ? thresholdState(configuration.hypotheses, observations) : [];
  const eligible = configuration?.queries.filter(value => !observations.some(row => row.x === value)) ?? [];
  const acquire = () => {
    if (!configuration || !remaining.length || !eligible.includes(Number(query))) return;
    const score = thresholdQuestion(configuration.hypotheses, observations, Number(query));
    const answer = configuration.oracleMode === 'manual' ? Number(manualAnswer) : Number(Number(query) >= configuration.trueThreshold);
    const entry = { query: Number(query), answer, before: remaining.map(row => row.threshold), after: score.groups[answer].map(row => row.threshold) };
    setHistory([...history, entry]);
    setResult({ ...score, answer, query: Number(query) });
    
  };
  const preview = configuration && remaining.length && eligible.includes(Number(query))
    ? thresholdQuestion(configuration.hypotheses, observations, Number(query)) : null;
  const nextQuestion = () => {
    const available = configuration.queries.filter(value => !observations.some(row => row.x === value));
    setQuery(available.length ? String(available[0]) : ''); clearAcquisition();
  };
  const reset = () => { setConfiguration(null); setHistory([]); clearAcquisition(); };
  const normalizeWeights = () => {
    try {
      const rows = parseNumericRows(thresholdText, 2, 'Threshold/weight');
      if (rows.some(([, weight]) => weight <= 0)) throw new Error('Weights must be positive.');
      const largest = Math.max(...rows.map(row => row[1]));
      const total = rows.reduce((sum, row) => sum + row[1] / largest, 0);
      setThresholdText(rows.map(([value, weight]) => `${value}, ${(weight / largest) / total}`).join('\n'));
      setConfiguration(null); setHistory([]); clearAcquisition();
    } catch (cause) { setError(cause.message); }
  };
  const preset = name => {
    setThresholdText((name === 'uneven' ? [0.5, 1.5, 4.5, 7.5] : fixtures.hypotheses.map(row => row.threshold)).map(value => `${value}, 1`).join('\n'));
    setObservationText('-1, 0\n9, 1'); setQueryText(name === 'uneven' ? '1 3 8' : '0 1 2 3 4 5 6 7 8');
    setOracleThreshold(name === 'uneven' ? '4.5' : '5.5'); setOracleMode('simulated'); reset();
  };
  return <section className="active-investigation" aria-labelledby="threshold-investigation-title" data-live-exploration="active-learning" data-active-lab="threshold">
    <h3 id="threshold-investigation-title">Investigation · spend a label on a useful question</h3>
    <p>Change the candidate rules and inspect how many would survive each answer. A ruler's solid region predicts 1; its dashed query line marks the proposed input.</p>
    <div className="active-controls"><button onClick={() => preset('default')}>Eight-threshold preset</button><button onClick={() => preset('uneven')}>Uneven-family preset</button><button onClick={reset}>Reset run</button></div>
    {!configuration && <><div className="active-input-grid">
      <label>Threshold, positive weight (one pair per line)<textarea value={thresholdText} onChange={editSetup(setThresholdText)} /></label>
      <label>Seed observations: x, answer<textarea value={observationText} onChange={editSetup(setObservationText)} /></label>
      <label>Eligible query values<textarea value={queryText} onChange={editSetup(setQueryText)} /></label>
      <div className="active-controls"><label>Oracle mode<select aria-label="Oracle mode" value={oracleMode} onChange={editSetup(setOracleMode)}><option value="simulated">Simulated threshold</option><option value="manual">Enter manual answers</option></select></label>
        {oracleMode === 'simulated' && <label>Simulation threshold (concealed after start)<input type="number" step="any" value={oracleThreshold} onChange={editSetup(setOracleThreshold)} /></label>}</div>
    </div><p className="active-small">Use 2–16 distinct thresholds in [−10,10] and up to 24 distinct query values in [−12,12]. Weights express relative plausibility; scores divide by the remaining total. This simulator does not infer real labels.</p>
      <div className="active-controls"><button onClick={normalizeWeights}>Normalize entered weights</button><button onClick={begin}>Start run</button></div></>}
    {configuration && <>
      <p>Labels: {configuration.observations.length} seed + {history.length} newly acquired = {observations.length} total. {remaining.length} hypotheses remain.</p>
      <p className="active-small">Observed answers: {observations.map(row => `(${row.x}, ${row.label})`).join('; ') || 'none'}.</p>
      <ThresholdRulers hypotheses={configuration.hypotheses} query={Number(query) || 0} survivors={remaining} showPredictions />
      {preview && <p role="status">For query {query}, answer 0 would retain {preview.counts[0]} candidates; answer 1 would retain {preview.counts[1]}. Expected survivors: {formatActiveValue(preview.expectedCount)}. Changing the query updates this comparison before you acquire a label.</p>}
      {remaining.length === 0 ? <p role="status" className="active-error">No listed threshold is consistent. The observed answers above conflict with the candidate family; edit the setup or reset before acquiring again.</p> : eligible.length === 0 && !result ? <p>No unused query remains. Reset or edit the setup to investigate another family.</p> : <>
        {!result && <div className="active-controls"><label>Unused query<select aria-label="Unused query" value={query} onChange={event => { setQuery(event.target.value); clearAcquisition(); }}>{eligible.map(value => <option key={value} value={value}>{value}</option>)}</select></label>
          {configuration.oracleMode === 'manual' && <label>Manual oracle answer<select aria-label="Manual oracle answer" value={manualAnswer} onChange={event => { setManualAnswer(event.target.value); clearAcquisition(); }}><option value="0">0</option><option value="1">1</option></select></label>}
          
          
        </div>}
        
        <div className="active-controls"><button disabled={!remaining.length || !eligible.includes(Number(query))} onClick={acquire}>Acquire this answer</button>{result && <button disabled={!eligible.length} onClick={nextQuestion}>Choose another question</button>}<button onClick={reset}>Edit setup</button></div>
      </>}
      {result && <div className="active-result" role="status"> At x={result.query}, computed counts for answers 0 / 1 are {result.counts.join(' / ')}. The acquired answer is {result.answer}.
        <p>Under the current hypothesis weights: P(0)={formatActiveValue(result.probabilities[0])}, P(1)={formatActiveValue(result.probabilities[1])}; expected survivors {formatActiveValue(result.expectedCount)}. This expectation is over the hypotheses; the actual retained count follows the acquired answer.</p>
        <p>{result.counts[result.answer] === result.remaining.length ? 'This legal query eliminated no hypothesis.' : result.counts[result.answer] === 0 ? 'The answer contradicts every remaining candidate; it identifies a problem with the assumptions or annotation.' : 'The crossed-out rules disagree with the newly observed answer.'}</p>
      </div>}
      {history.length > 0 && <LessonTable caption="Acquired history with original hypothesis identities" headers={['Query', 'Answer', 'Before', 'After']} rows={history.map(row => [row.query, row.answer, row.before.join(', '), row.after.join(', ') || 'none'])} />}
    </>}
    <ErrorMessage message={error} />
  </section>;
}

export function CommitteeInvestigation() {
  const [rows, setRows] = useState(copy(fixtures.opposing));

  const [error, setError] = useState('');
  const clear = () => { setError(''); };
  const changeRows = next => { setRows(next); clear(); };
  const edit = (member, column, value) => changeRows(rows.map((row, index) => index === member ? row.map((cell, position) => position === column ? numberInput(value) : cell) : row));
  const normalize = member => {
    const row = rows[member];
    const total = row.reduce((sum, value) => sum + value, 0);
    if (row.some(value => !Number.isFinite(value) || value < 0) || total <= 0) { setError('Enter finite nonnegative cells with a positive total before normalizing.'); return; }
    changeRows(rows.map((values, index) => index === member ? values.map(value => value / total) : values));
  };
  const calculation = useMemo(() => {
    try { const decomposition = committeeDecomposition(rows); return { result: { ...decomposition, actualRelation: compareNumbers(decomposition.disagreement, committeeDecomposition(fixtures.opposing).disagreement) } }; }
    catch (cause) { return { error: cause.message }; }
  }, [rows]);
  const result = calculation.result;
  return <section className="active-investigation" aria-labelledby="committee-investigation-title" data-live-exploration="active-learning" data-active-lab="committee">
    <h3 id="committee-investigation-title">Investigation · agreement can hide inside uncertainty</h3>
    <p>Compare your edited committee with the fixed two-member reference [0.95,0.05] and [0.05,0.95]. Change member distributions to see how disagreement D and its entropy terms change in nats.</p>
    <div className="active-controls"><button onClick={() => changeRows(copy(fixtures.opposing))}>Opposing reference</button><button onClick={() => changeRows(copy(fixtures.shared))}>Shared ambiguity</button><button onClick={() => changeRows(copy(fixtures.identical))}>Identical confident members</button><button onClick={() => changeRows(copy(fixtures.opposing))}>Reset committee</button></div>
    <div className="active-controls"><label>Class columns<select aria-label="Class columns" value={rows[0].length} onChange={event => changeRows(rows.map(row => Array.from({ length: Number(event.target.value) }, (_, index) => row[index] ?? 0)))}>{[2, 3, 4].map(count => <option key={count}>{count}</option>)}</select></label><button disabled={rows.length >= 6} onClick={() => changeRows([...rows, rows[0].map(() => 1 / rows[0].length)])}>Add member</button></div>
    {rows.map((row, member) => <div className="active-editor-member" key={member}><strong>Member {member + 1}</strong><label>Redistribute probability toward class 0<input aria-label={`Member ${member + 1} class 0 probability slider`} type="range" min="0" max="1" step="0.01" value={Number.isFinite(row[0]) ? row[0] : 0} onChange={event => {
        const first = Number(event.target.value), rest = row.slice(1).map(value => Number.isFinite(value) && value >= 0 ? value : 0), total = rest.reduce((sum, value) => sum + value, 0);
        changeRows(rows.map((values, index) => index === member ? [first, ...rest.map(value => (1 - first) * (total > 0 ? value / total : 1 / rest.length))] : values));
      }} /></label>
      <div className="active-controls">{row.map((value, column) => <label key={column}>Class {column}<input aria-label={`Member ${member + 1} class ${column}`} type="number" min="0" max="1" step="any" value={Number.isFinite(value) ? value : ''} onChange={event => edit(member, column, event.target.value)} /></label>)}
        <button onClick={() => normalize(member)}>Normalize member {member + 1}</button><button disabled={rows.length <= 2} onClick={() => changeRows(rows.filter((_, index) => index !== member))}>Remove member {member + 1}</button></div>
      {row.every(value => Number.isFinite(value) && value >= 0 && value <= 1) && Math.abs(row.reduce((sum, value) => sum + value, 0) - 1) <= 1e-9 && <ProbabilityStrip values={row} label={`Entered member ${member + 1}`} />}
    </div>)}
    <div className="active-controls">
      </div>
    
    <div className="active-controls"></div>
    <ErrorMessage message={error || calculation.error} />
    {result && <div className="active-result" role="status"> D is {result.actualRelation} than the reference. Calculated D is {formatActiveValue(result.disagreement)} nats.
      <ProbabilityStrip values={result.mean} label="Computed committee mean" /><EntropyBars result={result} classes={rows[0].length} />
      <LessonTable caption="Member entropies used in the average" headers={['Member', 'Entropy, nats']} rows={result.memberEntropies.map((value, index) => [index + 1, formatActiveValue(value)])} />
      <p className="active-small">Vote entropy: {formatActiveValue(result.voteEntropy)} nats; votes [{result.votes.join(', ')}]. Exact argmax ties vote for the lowest class index. All member weights are equal. These supplied distributions are not trained posterior samples.</p>
    </div>}
  </section>;
}

export function BatchInvestigation() {
  const [anchors, setAnchors] = useState(copy(fixtures.anchors));
  const [candidates, setCandidates] = useState(copy(fixtures.candidates));
  const [budget, setBudget] = useState(2);
  const [chosen, setChosen] = useState([]);

  const [step, setStep] = useState(0);
  const [error, setError] = useState('');
  const clear = () => { setChosen([]); setStep(0); setError(''); };
  const edit = (collection, id, field, value) => {
    const next = (collection === 'anchor' ? anchors : candidates).map(point => point.id === id ? { ...point, [field]: numberInput(value) } : point);
    (collection === 'anchor' ? setAnchors : setCandidates)(next); clear();
  };
  const preset = name => {
    setAnchors(copy(fixtures.anchors));
    setCandidates(name === 'coincident' ? ['A', 'B', 'C'].map(id => ({ id, x: 0, y: 0, probability: 0.5 })) : fixtures.candidates.map(point => ({ ...point, y: name === 'changed' && point.id === 'C' ? 2 : point.y })));
    setBudget(2); clear();
  };
  const add = collection => {
    const points = collection === 'anchor' ? anchors : candidates;
    const prefix = collection === 'anchor' ? 'L' : 'U';
    let serial = 1;
    while ([...anchors, ...candidates].some(point => point.id === `${prefix}${serial}`)) serial += 1;
    const next = [...points, { id: `${prefix}${serial}`, x: 0, y: 0, ...(collection === 'candidate' ? { probability: 0.5 } : {}) }];
    (collection === 'anchor' ? setAnchors : setCandidates)(next); clear();
  };
  const remove = (collection, id) => {
    (collection === 'anchor' ? setAnchors : setCandidates)((collection === 'anchor' ? anchors : candidates).filter(point => point.id !== id)); clear();
  };
  const calculation = useMemo(() => {
    try {
      validateGeometry(anchors, candidates, budget);
      const geometric = farthestFirst(anchors, candidates, budget), uncertain = entropyBatch(candidates, budget);
      return { result: { selectedIds: chosen, user: coveringDistances(anchors, candidates, chosen), geometric, uncertain, entropyRadius: coveringDistances(anchors, candidates, uncertain).radius } };
    } catch (cause) { return { error: cause.message }; }
  }, [anchors, candidates, budget, chosen]);
  const result = calculation.result;
  const validCoordinates = [...anchors, ...candidates].every(point => [point.x, point.y].every(value => Number.isFinite(value) && Math.abs(value) <= 10));
  const selectedAtStep = result ? step === 0 ? [] : result.geometric.steps[step - 1].selected : [];
  return <section className="active-investigation" aria-labelledby="batch-investigation-title" data-live-exploration="active-learning" data-active-lab="batch">
    <h3 id="batch-investigation-title">Investigation · buy coverage, not duplicate locations</h3>
    <p>Edit the coordinates or supplied probabilities. Select a batch and watch its covering radius change live. Compare batches of the same size with the budgeted alternatives. Geometry keeps the same scale on both axes.</p>
    <div className="active-controls"><button onClick={() => preset('default')}>Original geometry</button><button onClick={() => preset('changed')}>Move C nearer</button><button onClick={() => preset('coincident')}>Coincident null case</button><button onClick={() => preset('default')}>Reset batch</button></div>
    <h4>Existing anchors</h4>{anchors.map(point => <div className="active-edit-row" key={point.id}><strong>{point.id}</strong>{['x', 'y'].map(field => <label key={field}>{field}<input aria-label={`${point.id} ${field}`} type="number" min="-10" max="10" step="any" value={Number.isFinite(point[field]) ? point[field] : ''} onChange={event => edit('anchor', point.id, field, event.target.value)} /></label>)}<button disabled={anchors.length === 1} onClick={() => remove('anchor', point.id)}>Remove {point.id}</button></div>)}
    <div className="active-controls"><button disabled={anchors.length >= 4} onClick={() => add('anchor')}>Add anchor</button></div>
    <h4>Candidate inputs</h4>{candidates.map(point => <div className="active-edit-row" key={point.id}><strong>{point.id}</strong>{['x', 'y', 'probability'].map(field => <label key={field}>{field === 'probability' ? 'P(class 1)' : field}<input aria-label={`${point.id} ${field}`} type="number" min={field === 'probability' ? 0 : -10} max={field === 'probability' ? 1 : 10} step="any" value={Number.isFinite(point[field]) ? point[field] : ''} onChange={event => edit('candidate', point.id, field, event.target.value)} /></label>)}<button disabled={candidates.length <= 2} onClick={() => remove('candidate', point.id)}>Remove {point.id}</button></div>)}
    <p className="active-small">Entropy differences within 10⁻¹² nats count as numerical ties; identifier order breaks them.</p>
    <div className="active-controls"><button disabled={candidates.length >= 24} onClick={() => add('candidate')}>Add candidate</button><label>Batch budget<input type="number" min="1" max={Math.min(6, candidates.length)} step="1" value={Number.isFinite(budget) ? budget : ''} onChange={event => { setBudget(numberInput(event.target.value)); clear(); }} /></label></div>
    {validCoordinates && <CoveragePlane anchors={anchors} candidates={candidates} selected={selectedAtStep} showDistances={Boolean(result)} />}
    <fieldset><legend>Choose the distinct IDs you would label</legend><div className="active-controls">{candidates.map(point => <label key={point.id}><span><input type="checkbox" checked={chosen.includes(point.id)} onChange={event => { setChosen(event.target.checked ? [...chosen, point.id] : chosen.filter(id => id !== point.id)); }} /> {point.id}</span></label>)}</div></fieldset>
    <div className="active-controls"></div>
    <div className="active-controls"></div><ErrorMessage message={error || calculation.error} />
    {result && <div className="active-result" role="status"> Your [{result.selectedIds.join(', ')}] radius is {formatActiveValue(result.user.radius)}.
      <LessonTable caption="Same inputs and budget; different batch objectives" headers={['Method', 'Selected IDs', 'Covering radius']} rows={[
        ['Your batch', result.selectedIds.join(', '), formatActiveValue(result.user.radius)],
        ['Highest entropy', result.uncertain.join(', '), formatActiveValue(result.entropyRadius)],
        ['Farthest first', result.geometric.selected.join(', '), formatActiveValue(result.geometric.radius)],
      ]} />
      <p>{compareNumbers(result.user.radius, result.geometric.radius) === 'same' ? 'Your batch and farthest-first have equal coverage on these inputs.' : 'The two batches leave different maximum distances to a center.'} Neither distance establishes label accuracy.</p>
      <div className="active-controls"><button disabled={step === 0} onClick={() => setStep(step - 1)}>Previous geometric step</button><button disabled={step === result.geometric.steps.length} onClick={() => setStep(step + 1)}>Next geometric step</button></div>
      <p>Geometric step {step} / {result.geometric.steps.length}: {step === 0 ? 'only the existing anchors are centers.' : `add ${result.geometric.steps[step - 1].id}; radius becomes ${formatActiveValue(result.geometric.steps[step - 1].radius)}.`} Ties use identifier order; selected identifiers are always masked.</p>
      <details><summary>Inspect your batch's own distances</summary><LessonTable caption="Your chosen batch: nearest centers" headers={['Candidate', 'Nearest center', 'Distance']} rows={result.user.assignments.map(row => [row.id, row.center.id, formatActiveValue(row.distance)])} /></details>
    </div>}
  </section>;
}
