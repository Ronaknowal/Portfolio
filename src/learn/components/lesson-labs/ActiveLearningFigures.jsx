import { useState } from 'react';
import { CodeBlock } from '../content/Code.jsx';
import { LessonTable } from './LessonElements.jsx';
import { activeLearningFixtures, committeeDecomposition, coveringDistances, farthestFirst, uncertaintyScores } from '../../data/active-learning-models.js';
import experiment from '../../data/active-learning-experiment.json';
import { activeLearningProgram } from '../../data/active-learning-examples.js';
import './active-learning.css';

export const formatActiveValue = value => Number(value.toFixed(6)).toString();
const classColors = ['#dbb75d', '#82ac9b', '#b59ec4', '#93aac9'];

export function ProbabilityStrip({ values, label }) {
  return <div>
    <p className="active-small">{label}</p>
    <div className="active-probability" role="img" aria-label={`${label}: ${values.map((value, index) => `class ${index} probability ${value}`).join(', ')}`}>
      {values.map((value, index) => <span key={index} style={{ width: `${100 * value}%`, background: classColors[index] }} />)}
    </div>
    <div className="active-probability-key">{values.map((value, index) => <span key={index}><i aria-hidden="true" style={{ background: classColors[index] }} />Class {index}: {formatActiveValue(value)}</span>)}</div>
  </div>;
}

export function AnnotationBoundaryFigure() {
  return <figure className="active-figure" data-active-figure="annotation-data-boundary">
    <h4>One requested answer crosses the boundary</h4>
    <ol className="active-flow">
      <li>U: inspect inputs<span>Class answer concealed</span></li><li>Select an ID<span>Only inputs and fitted model</span></li>
      <li>Pending query<span>Ask the annotator</span></li><li>Acquire the answer<span>Now the label may enter L</span></li><li>Refit the model<span>Include the newest label</span></li>
    </ol>
    <div className="active-boundary"><div><strong>Development lane</strong>Compare predeclared strategies at matched budgets.</div><div><strong>Final test lane</strong>Assess the selected procedure after selection is complete.</div></div>
    <figcaption>Conceptual data-access diagram. Evaluation labels stay outside the acquisition and fitting loop; their annotation cost still counts.</figcaption>
  </figure>;
}

export function ThresholdRulers({ hypotheses, query, survivors, showPredictions = false }) {
  const minimum = Math.min(-1, query, ...hypotheses.map(row => row.threshold));
  const maximum = Math.max(9, query, ...hypotheses.map(row => row.threshold));
  const position = value => 8 + 184 * (value - minimum) / (maximum - minimum);
  return <div>{hypotheses.map(row => {
    const retained = !survivors || survivors.some(candidate => candidate.threshold === row.threshold);
    return <div className={`active-ruler ${retained ? '' : 'eliminated'}`} key={row.threshold}>
      <span>t={row.threshold}</span>
      <svg viewBox="0 0 200 22" aria-hidden="true"><line x1="8" y1="11" x2="192" y2="11" stroke="#506154" />
        <line x1={position(row.threshold)} x2="192" y1="11" y2="11" stroke={retained ? '#91b29e' : '#506154'} strokeWidth="5" />
        <circle cx={position(row.threshold)} cy="11" r="4" fill="#d9d8bb" />
        <line x1={position(query)} x2={position(query)} y1="0" y2="22" stroke="#e2b55a" strokeWidth="2" strokeDasharray="3 2" />
      </svg>
      <span>{showPredictions ? `y=${Number(query >= row.threshold)}` : `w=${formatActiveValue(row.weight)}`}</span>
    </div>;
  })}</div>;
}

export function ThresholdFigure() {
  return <figure className="active-figure" data-active-figure="threshold-version-space"><h4>Query x = 4, then observe 0</h4>
    <ThresholdRulers hypotheses={activeLearningFixtures.hypotheses} query={4} survivors={activeLearningFixtures.hypotheses.filter(row => row.threshold > 4)} showPredictions />
    <figcaption>Exact worked example. Each green segment is the region where its threshold predicts 1; the dashed amber line is the same query on every ruler. Crossing out four incompatible rules leaves t = 4.5, 5.5, 6.5 or 7.5.</figcaption>
  </figure>;
}

export function UncertaintyFigure() {
  const rows = [[0.5, 0.5, 0], [0.45, 0.3, 0.25], [0.6, 0.2, 0.2]];
  return <figure className="active-figure" data-active-figure="multiclass-uncertainty-strips"><h4>The same probabilities, three different questions</h4>
    {rows.map((values, index) => { const scores = uncertaintyScores(values); return <div key={index}>
      <ProbabilityStrip values={values} label={`Candidate ${'ABC'[index]}`} />
      <p className="active-small">Least confidence {formatActiveValue(scores.leastConfidence)} · margin {formatActiveValue(scores.margin)} · entropy {formatActiveValue(scores.entropy)} nats</p>
    </div>; })}
    <figcaption>Calculated from the supplied probabilities. Margin chooses A's top-two tie; entropy and least confidence choose B. Strip lengths encode probabilities, not model accuracy.</figcaption>
  </figure>;
}

export function EntropyBars({ result, classes = 2 }) {
  const maximum = Math.log(classes);
  return <div>{[['H(mean)', result.predictiveEntropy], ['Mean H(member)', result.meanMemberEntropy], ['Difference D', result.disagreement]].map(([label, value]) => <div key={label}>
    <div className="active-entropy-row"><span>{label}</span><div className="active-entropy-track"><span style={{ width: `${100 * value / maximum}%` }} /></div></div>
    <p className="active-small">{label}: {formatActiveValue(value)} nats</p>
  </div>)}<p className="active-small">Common bar scale: 0 to ln {classes} = {formatActiveValue(maximum)} nats.</p></div>;
}

export function CommitteeFigure() {
  return <figure className="active-figure" data-active-figure="committee-entropy-decomposition"><h4>Same mean prediction; different reasons for uncertainty</h4>
    <div className="active-comparison">{[['Shared ambiguity', activeLearningFixtures.shared], ['Opposing confidence', activeLearningFixtures.opposing]].map(([label, rows]) => {
      const result = committeeDecomposition(rows);
      return <div key={label}><h4>{label}</h4>{rows.map((values, index) => <ProbabilityStrip key={index} values={values} label={`Member ${index + 1}`} />)}
        <ProbabilityStrip values={result.mean} label="Mean distribution" /><EntropyBars result={result} />
      </div>;
    })}</div><figcaption>Calculated entropy decomposition of authored distributions. Only the opposing members disagree; equal means do not imply equal disagreement.</figcaption>
  </figure>;
}

export function CoveragePlane({ anchors, candidates, selected = [], showDistances = false }) {
  const all = [...anchors, ...candidates];
  const minimum = Math.min(-0.5, ...all.flatMap(point => [point.x, point.y])) - 0.5;
  const maximum = Math.max(4.5, ...all.flatMap(point => [point.x, point.y])) + 0.5;
  const x = value => 26 + 260 * (value - minimum) / (maximum - minimum);
  const y = value => 286 - 260 * (value - minimum) / (maximum - minimum);
  const assignment = showDistances ? coveringDistances(anchors, candidates, selected) : null;
  return <div className="active-plane">
    <svg viewBox="0 0 320 320" role="img" aria-label="Candidate geometry with equal x and y scales; exact coordinates and identifiers are listed below.">
      <rect x="26" y="26" width="260" height="260" fill="none" stroke="#506453" />
      <line x1={x(0)} x2={x(0)} y1="26" y2="286" stroke="#506453" strokeDasharray="3 3" />
      <line x1="26" x2="286" y1={y(0)} y2={y(0)} stroke="#506453" strokeDasharray="3 3" />
      <text x="26" y="306" fontSize="11">{formatActiveValue(minimum)}</text><text x="286" y="306" textAnchor="end" fontSize="11">{formatActiveValue(maximum)} x</text>
      <text x="26" y="17" fontSize="11">y: {formatActiveValue(minimum)} to {formatActiveValue(maximum)}</text>
      {assignment?.assignments.map(row => <line key={row.id} x1={x(row.point.x)} y1={y(row.point.y)} x2={x(row.center.x)} y2={y(row.center.y)} stroke="#a5b197" strokeWidth="1.5" />)}
      {anchors.map(point => <rect key={point.id} x={x(point.x) - 5} y={y(point.y) - 5} width="10" height="10" fill="#eee2b3"><title>{point.id}: ({point.x}, {point.y})</title></rect>)}
      {candidates.map((point, index) => <circle key={point.id} cx={x(point.x)} cy={y(point.y)} r={selected.includes(point.id) ? 7 : 4} fill={classColors[index % 4]} stroke={selected.includes(point.id) ? '#f0ebd4' : '#172219'} strokeWidth="1.5"><title>{point.id}: ({point.x}, {point.y}){selected.includes(point.id) ? ', selected' : ''}</title></circle>)}
    </svg>
    <div className="active-point-key">{candidates.map((point, index) => <span key={point.id}><i style={{ color: classColors[index % 4] }} aria-hidden="true">●</i> {point.id} ({point.x}, {point.y}){selected.includes(point.id) ? ' selected' : ''}</span>)}</div>
    <p className="active-small">Squares: existing anchors. Ringed circles: chosen batch. Nearby or coincident inputs keep their true locations; the coordinate list preserves each identity.</p>
    {assignment && <LessonTable caption="Nearest-center distances in the authored coordinate plane" headers={['Candidate', 'Nearest center', 'Distance']} rows={assignment.assignments.map(row => [row.id, row.center.id, formatActiveValue(row.distance)])} />}
  </div>;
}

export function CoverageFigure() {
  const { anchors, candidates } = activeLearningFixtures;
  const result = farthestFirst(anchors, candidates, 2);
  return <figure className="active-figure" data-active-figure="geometric-covering-radius"><h4>Covering a space after selecting C and D</h4>
    <CoveragePlane anchors={anchors} candidates={candidates} selected={result.selected} showDistances />
    <figcaption>Calculated Euclidean geometry. The longest remaining connection is B → L1, radius {formatActiveValue(result.radius)}. Selecting A and B instead leaves radius 4. These are coverage distances, not errors from a fitted classifier.</figcaption>
  </figure>;
}

export function AcquisitionCurves() {
  const [strategy, setStrategy] = useState('entropy');
  const [run, setRun] = useState('mean');
  const [revealed, setRevealed] = useState(0);
  const allScores = experiment.strategies.flatMap(name => experiment.traces[name].flatMap(trace => trace.development_correct));
  const lower = Math.max(0, Math.floor(Math.min(...allScores) / 5) * 5 - 5);
  const x = index => 38 + 252 * index / 30;
  const y = score => 210 - 180 * (score - lower) / (80 - lower);
  const means = name => Array.from({ length: 31 }, (_, index) => experiment.traces[name].reduce((sum, trace) => sum + trace.development_correct[index], 0) / 5);
  const selectedTrace = experiment.traces[strategy][run === 'mean' ? 0 : Number(run)];
  const series = run === 'mean' ? experiment.strategies.map((name, index) => ({ label: name, values: means(name), color: classColors[index], dash: ['', '6 3', '2 3', '8 3 2 3'][index] })) : [{ label: `${strategy}, seed ${selectedTrace.seed}`, values: selectedTrace.development_correct, color: '#e2b55a' }];
  return <figure className="active-figure" data-active-figure="measured-acquisition-curves"><h4>Measured progress: every acquired label enters the next fit</h4>
    <svg viewBox="0 0 320 258" role="img" aria-label="Measured development counts, training labels 6 to 36. Exact checkpoints follow in the table.">
      {[lower, (lower + 80) / 2, 80].map(value => <g key={value}><line x1="38" x2="290" y1={y(value)} y2={y(value)} stroke="#3c4b40" /><text x="32" y={y(value) + 4} textAnchor="end" fontSize="11">{value}</text></g>)}
      <text x="38" y="16" fontSize="11">Correct / 80 development rows</text>
      {series.map(row => <polyline key={row.label} points={row.values.map((value, index) => `${x(index)},${y(value)}`).join(' ')} fill="none" stroke={row.color} strokeWidth="1.8" strokeDasharray={row.dash} />)}
      {[0, 10, 20, 30].map(index => <text key={index} x={x(index)} y="228" fontSize="11" textAnchor="middle">{6 + index}</text>)}
      <text x="164" y="249" fontSize="11" textAnchor="middle">Total fitting labels (six seeds + queries)</text>
    </svg>
    <div className="active-probability-key">{series.map(row => <span key={row.label}><i style={{ background: row.color }} />{row.label}{row.dash ? ` · dash ${row.dash}` : ' · solid'}</span>)}</div>
    <div className="active-controls"><label>Trace strategy<select aria-label="Trace strategy" value={strategy} onChange={event => { setStrategy(event.target.value); setRevealed(0); }}>{experiment.strategies.map(name => <option key={name}>{name}</option>)}</select></label>
      <label>Curve view<select aria-label="Curve view" value={run} onChange={event => { setRun(event.target.value); setRevealed(0); }}><option value="mean">All strategy means</option>{experiment.run_seeds.map((seed, index) => <option key={seed} value={index}>One run: seed {seed}</option>)}</select></label></div>
    <details><summary>All 31 measured checkpoints</summary><LessonTable caption="Development correct out of 80 at every actual fit" headers={['Training labels', ...series.map(row => row.label)]} rows={Array.from({ length: 31 }, (_, index) => [6 + index, ...series.map(row => formatActiveValue(row.values[index]))])} /></details>
    <details><summary>Replay acquired labels for {strategy}, seed {selectedTrace.seed}</summary>
      <p className="active-small">Initial six pool indices: {selectedTrace.initial_pool_rows.join(', ')}. This replays saved evidence; it does not train in the browser.</p>
      <p>Newly acquired: {revealed} · fitted-label count after refit: {6 + revealed}</p>
      <div className="active-controls"><button disabled={revealed === 30} onClick={() => setRevealed(revealed + 1)}>Acquire next recorded label</button><button onClick={() => setRevealed(0)}>Reset replay</button></div>
      {revealed > 0 && <LessonTable caption="Only acquired oracle responses" headers={['Query', 'Source row', 'Acquired label', 'Pre-query P(class 1)']} rows={selectedTrace.queries.slice(0, revealed).map((query, index) => [index + 1, query.source_row, query.label, formatActiveValue(query.model_probability[1])])} />}
    </details>
    <figcaption>Recorded UCI Banknote experiment, five paired initial-label sets, fixed 320/80/80 split. Means are across runs on the same rows; line segments connect actual checkpoints. The vertical axis is cropped to expose observed differences; a higher point means more correct development decisions. No smoothing, browser fitting or confidence-interval claim.</figcaption>
  </figure>;
}

export function AnnotationTimeline() {
  return <figure className="active-figure" data-active-figure="annotation-record-timeline"><h4>An answer has a state before it becomes training data</h4>
    <ol className="active-flow"><li>Candidate<span>Eligible, not yet requested</span></li><li>Pending<span>ID reserved; no duplicate query</span></li><li>Answered<span>Label + annotator + input version</span></li><li>Accepted<span>Or send disagreement for adjudication</span></li><li>Fitted<span>Record which model used it</span></li></ol>
    <figcaption>Conceptual workflow. An abstained or unreadable response branches to review; it is not automatically converted into class 0.</figcaption>
  </figure>;
}

export function ExperimentDownloads() {
  return <aside className="active-downloads"><h4>Reproduce the complete experiment</h4>
    <p className="active-small"><a href="/learn/downloads/active-learning/banknote-active-learning.py" download>Complete Python program</a> · <a href="/learn/downloads/active-learning/banknote-subset.csv" download>480-row data file</a> · <a href="/learn/downloads/active-learning/data-provenance.md" download>Data provenance and split</a></p>
    <details><summary>Read the complete executable program</summary><CodeBlock language="python">{activeLearningProgram}</CodeBlock></details>
  </aside>;
}
