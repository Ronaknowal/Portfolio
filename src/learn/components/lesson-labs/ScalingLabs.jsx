import { useId, useState } from 'react';
import {
  crossFitEncoding, knnImpute, limits, nearestCandidate, transformRecord,
} from '../../data/scaling-models';
import { featureNames, rows, split } from '../../data/scaling-data';
import { FlowDefs, preparation } from './ScalingFigures.jsx';
import {
  Investigation, NumberField, Plot, Prediction, Reason, SelectField, Table, fixed, fraction, measured, round, withinTolerance,
} from './ScalingShared.jsx';
import './scaling-labs.css';

const numericOf = row => [row[2], row[3], row[4], row[5]];
const sexOf = row => (row[6] === null ? null : ['female', 'male'][row[6]]);
const recordBounds = [[5, 300], [5, 300], [5, 300], [1500, 8000]];
// All editable records share these bounds, so the prediction range neither
// excludes a valid transformed coordinate nor gives away the current answer.
const predictionLimit = Math.ceil(Math.max(20, ...preparation.scalers.flatMap((ruler, index) =>
  [...recordBounds[index], preparation.imputer.medians[index]].map(value => Math.abs((value - ruler.center) / ruler.scale)))));

/* ------------------------------------------------------------------ I1 */

const rulerPresets = {
  raw: { label: 'Raw units: 1 mm, 1 g', divisors: [1, 1] },
  domain: { label: 'Declared: 1 mm, 100 g', divisors: [1, 100] },
  doubled: { label: 'Both divisors doubled: 2 mm, 200 g', divisors: [2, 200] },
};
const rulerStart = {
  query: [40, 4000], a: [41, 4100], b: [43, 4001], divisors: [1, 1],
};

/** §2 · I1. Which candidate is nearer depends on the divisors, and the
 * contributions say why. */
export function RulerLab() {
  const state = useInvestigationWithHistory(rulerStart);
  const active = state.active;
  const answerFor = inputs => nearestCandidate(inputs.query, [
    { name: 'A', point: inputs.a }, { name: 'B', point: inputs.b },
  ], inputs.divisors).winner;
  const outcome = nearestCandidate(active.query, [
    { name: 'A', point: active.a }, { name: 'B', point: active.b },
  ], active.divisors);
  const revealed = Boolean(state.result);
  // The transformed plane is dimensionless in both directions, so a circle of
  // equal scaled distance really is a circle there. The physical plot below is
  // anisotropic and carries no circles.
  const offsets = [active.a, active.b].map(point => [
    (point[0] - active.query[0]) / active.divisors[0],
    (point[1] - active.query[1]) / active.divisors[1],
  ]);
  const reach = Math.max(1e-6, ...offsets.flat().map(Math.abs)) * 1.35;
  const size = 260;
  const middle = size / 2;
  const toPlane = value => middle + (value / reach) * (middle - 26);
  const fromPlane = value => middle - (value / reach) * (middle - 26);
  const billSpan = [Math.min(active.query[0], active.a[0], active.b[0]) - 1.5, Math.max(active.query[0], active.a[0], active.b[0]) + 1.5];
  const massSpan = [Math.min(active.query[1], active.a[1], active.b[1]) - 120, Math.max(active.query[1], active.a[1], active.b[1]) + 120];
  const names = ['bill length (mm)', 'body mass (g)'];
  const canTranslate = [state.draft.query, state.draft.a, state.draft.b]
    .every(point => point[0] + 5 <= 80 && point[1] + 500 <= 7000);

  return <Investigation
    title="Investigation 1 — Choose the ruler, then inspect the neighbour"
    question={`A new measurement Q sits at ${active.query[0]} mm and ${round(active.query[1])} g. Candidates A and B are fixed numbers you can edit. Before applying, record which candidate the rule will call nearer, then look at each feature's contribution to the squared distance.`}
    note="These are constructed measurements for arithmetic, not rows claimed to come from the penguin file. Changing a divisor expresses a choice about which differences should count; nothing moves in the physical world."
    onReset={state.reset}>
    <div className="sc-controls">
      <NumberField label="Q bill length (mm)" value={state.draft.query[0]} min={20} max={80} step="0.5" decimals={2}
        onChange={value => state.edit({ query: [value, state.draft.query[1]] })} />
      <NumberField label="Q body mass (g)" value={state.draft.query[1]} min={2000} max={7000} step="50" decimals={2}
        onChange={value => state.edit({ query: [state.draft.query[0], value] })} />
      <NumberField label="A bill length (mm)" value={state.draft.a[0]} min={20} max={80} step="0.5" decimals={2}
        onChange={value => state.edit({ a: [value, state.draft.a[1]] })} />
      <NumberField label="A body mass (g)" value={state.draft.a[1]} min={2000} max={7000} step="50" decimals={2}
        onChange={value => state.edit({ a: [state.draft.a[0], value] })} />
      <NumberField label="B bill length (mm)" value={state.draft.b[0]} min={20} max={80} step="0.5" decimals={2}
        onChange={value => state.edit({ b: [value, state.draft.b[1]] })} />
      <NumberField label="B body mass (g)" value={state.draft.b[1]} min={2000} max={7000} step="50" decimals={2}
        onChange={value => state.edit({ b: [state.draft.b[0], value] })} />
      <NumberField label="Bill divisor (mm per unit)" value={state.draft.divisors[0]} min={limits.divisor.minimum} max={limits.divisor.maximum}
        step="0.5" decimals={4} onChange={value => state.edit({ divisors: [value, state.draft.divisors[1]] })} />
      <NumberField label="Mass divisor (g per unit)" value={state.draft.divisors[1]} min={limits.divisor.minimum} max={limits.divisor.maximum}
        step="10" decimals={4} onChange={value => state.edit({ divisors: [state.draft.divisors[0], value] })} />
    </div>
    <p className="sc-caption">
      A divisor must be positive: dividing by nothing is not a ruler, so the field refuses anything below {limits.divisor.minimum} instead of
      quietly substituting 1. The measurement fields are bounded to plausible penguin values for the same reason. Try the presets as entry points,
      then invent your own.
    </p>
    <div className="sc-buttons">
      {Object.entries(rulerPresets).map(([key, preset]) => (
        <button key={key} type="button" onClick={() => state.edit({ divisors: preset.divisors })}>{preset.label}</button>
      ))}
      <button type="button" disabled={!canTranslate} onClick={() => state.edit({
        query: [state.draft.query[0] + 5, state.draft.query[1] + 500],
        a: [state.draft.a[0] + 5, state.draft.a[1] + 500],
        b: [state.draft.b[0] + 5, state.draft.b[1] + 500],
      })}>Translate everything by +5 mm, +500 g</button>
    </div>
    {!canTranslate && <p className="sc-caption">Translation has reached a measurement bound. Reset or lower the measurements before translating again.</p>}
    <Prediction
      prompt={`With divisors ${round(state.draft.divisors[0])} mm and ${round(state.draft.divisors[1])} g, which candidate will be nearer to Q?`}
      options={[['A', 'A is nearer'], ['B', 'B is nearer'], ['tie', 'They tie']]}
      state={state} answerFor={answerFor}
      label={inputs => `divisors ${round(inputs.divisors[0])} and ${round(inputs.divisors[1])} gave ${answerFor(inputs)}`}
      describe={outcome.winner === 'tie'
        ? 'The two totals agree to within the stated numerical tolerance, so this is recorded as a tie rather than resolved by some other rule.'
        : `The deciding contribution is the ${Math.abs(outcome.scored[0].contributions[0].contribution - outcome.scored[1].contributions[0].contribution)
          > Math.abs(outcome.scored[0].contributions[1].contribution - outcome.scored[1].contributions[1].contribution) ? 'bill' : 'mass'} term.`} />

    {revealed && <>
      <Table caption={`Each feature's contribution to the squared distance, under divisors ${round(active.divisors[0])} and ${round(active.divisors[1])}`}
        headings={['candidate', 'feature', 'difference', 'divisor', '(difference ÷ divisor)²', 'total']}
        rows={outcome.scored.flatMap(candidate => candidate.contributions.map((part, index) => [
          index === 0 ? candidate.name : '', names[index], round(part.difference, 4), round(part.divisor, 4),
          round(part.contribution, 6), index === 0 ? round(candidate.total, 6) : '',
        ]))} />
      <div className="sc-bars" role="img" aria-label={`Squared distance totals: A ${round(outcome.scored[0].total, 6)}, B ${round(outcome.scored[1].total, 6)}.`}>
        {outcome.scored.map((candidate, index) => (
          <div className="sc-bar-row" key={candidate.name}>
            <span>{candidate.name} total</span>
            <span className="sc-bar-track">
              <span className={`sc-bar-fill${index === 1 ? ' is-b' : ''}`}
                style={{ width: `${100 * candidate.total / (Math.max(outcome.scored[0].total, outcome.scored[1].total) || 1)}%` }} />
            </span>
            <span className="sc-bar-value">{round(candidate.total, 6)}</span>
          </div>
        ))}
      </div>
      <div className="sc-panels">
        <div className="sc-panel">
          <h4>Scaled difference plane, centred on Q</h4>
          <svg viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`An equal-aspect plane of scaled differences from Q. A sits at ${round(offsets[0][0], 4)}, ${round(offsets[0][1], 4)} and B at ${round(offsets[1][0], 4)}, ${round(offsets[1][1], 4)}. The dashed circles have radii equal to each candidate's scaled distance, ${round(Math.sqrt(outcome.scored[0].total), 4)} and ${round(Math.sqrt(outcome.scored[1].total), 4)}.`}>
            <line className="sc-axis" x1="18" x2={size - 18} y1={middle} y2={middle} />
            <line className="sc-axis" x1={middle} x2={middle} y1="18" y2={size - 18} />
            {outcome.scored.map(candidate => (
              <circle key={candidate.name} className="sc-ring" cx={middle} cy={middle}
                r={Math.min(middle - 6, (Math.sqrt(candidate.total) / reach) * (middle - 26))} />
            ))}
            {offsets.map((offset, index) => <g key={index}>
              <line className={`sc-vector ${index === 0 ? 'is-a' : 'is-b'}`} x1={middle} y1={middle}
                x2={toPlane(offset[0])} y2={fromPlane(offset[1])} />
              <circle className={index === 0 ? 'sc-mark-a' : 'sc-mark-b'} cx={toPlane(offset[0])} cy={fromPlane(offset[1])} r="5" />
              <text x={toPlane(offset[0]) + 8} y={fromPlane(offset[1]) - 8} style={{ fontSize: 11 }}>{index === 0 ? 'A' : 'B'}</text>
            </g>)}
            <circle className="sc-query" cx={middle} cy={middle} r="5" />
            <text x={middle + 8} y={middle + 18} style={{ fontSize: 11 }}>Q</text>
            <text x={size - 14} y={middle - 6} textAnchor="end" style={{ fontSize: 10.5 }}>scaled bill difference →</text>
            <text x={middle + 6} y="20" style={{ fontSize: 10.5 }}>↑ scaled mass difference</text>
          </svg>
          <p>
            Both axes are in scaled units, so an equal-distance set really is a circle here. The dashed circles have radii{' '}
            {round(Math.sqrt(outcome.scored[0].total), 4)} and {round(Math.sqrt(outcome.scored[1].total), 4)}: the square roots of the two totals.
          </p>
        </div>
        <div className="sc-panel">
          <h4>Physical measurements, in their own units</h4>
          <Plot caption="" width={300} height={230} domain={billSpan} range={massSpan}
            ticks={[billSpan[0], (billSpan[0] + billSpan[1]) / 2, billSpan[1]]}
            describe={`Bill length against body mass in physical units. Q at ${active.query[0]} millimetres and ${round(active.query[1])} grams, A at ${active.a[0]} and ${round(active.a[1])}, B at ${active.b[0]} and ${round(active.b[1])}. The two axes have different units, so no distance circle is drawn on this plot.`}>
            {(scaleX, scaleY) => <>
              <circle className="sc-mark-a" cx={scaleX(active.a[0])} cy={scaleY(active.a[1])} r="5" />
              <text x={scaleX(active.a[0]) + 8} y={scaleY(active.a[1]) - 6} style={{ fontSize: 11 }}>A</text>
              <circle className="sc-mark-b" cx={scaleX(active.b[0])} cy={scaleY(active.b[1])} r="5" />
              <text x={scaleX(active.b[0]) + 8} y={scaleY(active.b[1]) - 6} style={{ fontSize: 11 }}>B</text>
              <circle className="sc-query" cx={scaleX(active.query[0])} cy={scaleY(active.query[1])} r="5" />
              <text x={scaleX(active.query[0]) + 8} y={scaleY(active.query[1]) + 16} style={{ fontSize: 11 }}>Q</text>
            </>}
          </Plot>
          <p>Millimetres across, grams up. The axes carry different units, so a round circle here would assert a comparison the units do not support.</p>
        </div>
      </div>
      <p className="sc-readout">
        A is <strong>{round(outcome.scored[0].total, 6)}</strong> and B is <strong>{round(outcome.scored[1].total, 6)}</strong> in squared scaled
        units, so the rule selects <strong>{outcome.winner === 'tie' ? 'neither: they tie' : outcome.winner}</strong>. Multiplying both divisors by
        the same positive number multiplies every total by the same factor, which cannot change the ranking.
      </p>
    </>}
  </Investigation>;
}

/* ------------------------------------------------------------------ I2 */

const donorStart = {
  donors: [[1, 10, 100], [3, null, 300], [null, 14, 500]],
  query: [2, 12, null],
  target: 2,
  neighbours: 2,
};
const donorNames = ['D1', 'D2', 'D3'];
const featureLetters = ['a', 'b', 'c'];

/** §4 · I2. Which cells are allowed to contribute to a missing value. */
export function DonorLab() {
  const state = useInvestigationWithHistory(donorStart);
  const [predictedDonors, setPredictedDonors] = useState([]);
  const [predictedValue, setPredictedValue] = useState(null);
  const [noData, setNoData] = useState(false);
  const [reason, setReason] = useState('');
  const [result, setResult] = useState(null);
  const groupId = useId();
  const active = state.active;
  const compute = inputs => knnImpute({
    donors: inputs.donors, donorNames, query: inputs.query, target: inputs.target, neighbours: inputs.neighbours,
  });
  const outcome = compute(active);
  const draftKey = JSON.stringify(state.draft);
  const retire = () => { setResult(null); };
  const edit = update => { state.edit(update); retire(); };
  const setCell = (donor, column, value) => {
    const donors = state.draft.donors.map((row, index) => (index === donor ? row.map((cell, position) => (position === column ? value : cell)) : row));
    edit({ donors });
  };
  const setTarget = column => {
    // Selecting another target makes that query cell absent; we never invent a
    // value and then use it as an observed distance coordinate.
    edit({ target: column, query: state.draft.query.map((cell, index) => (index === column ? null : cell)) });
  };
  const commit = () => {
    const answer = compute(state.draft);
    state.advance();
    setResult({
      key: draftKey, answer,
      donors: [...predictedDonors].sort(),
      value: noData ? null : predictedValue,
      noData, reason,
    });
  };
  const selectedNames = outcome.selected.map(row => row.name);
  const donorsMatch = result && JSON.stringify(result.donors) === JSON.stringify(result.answer.selected.map(row => row.name).sort());
  const valueMatch = result && (result.noData
    ? result.answer.estimate === null
    : result.value !== null && result.answer.estimate !== null && withinTolerance(result.value, result.answer.estimate, 0.00005));

  return <Investigation
    title="Investigation 2 — Who is allowed to donate?"
    question={`Column ${featureLetters[active.target]} of the query is absent. Predict which donors will be used and what value they will supply, then apply the table and read the overlap arithmetic.`}
    note="Columns a, b and c are constructed numeric coordinates with no claimed physical unit. A donor with a gap elsewhere can still donate; a donor without the target measurement cannot."
    onReset={() => { state.reset(); setPredictedDonors([]); setPredictedValue(null); setNoData(false); setReason(''); setResult(null); }}>

    <Table caption="The source table. Edit any cell, or clear it to make it absent."
      headings={['row', ...featureLetters]}
      rows={[
        ['query', ...state.draft.query.map((cell, column) => (column === state.draft.target
          ? <span key={column} className="sc-struck">absent (target)</span>
          : <NumberField key={column} label={`query ${featureLetters[column]}`} value={cell} min={limits.donorCell.minimum}
            max={limits.donorCell.maximum} step="1" decimals={4} allowMissing
            onChange={value => edit({ query: state.draft.query.map((item, index) => (index === column ? value : item)) })} />))],
        ...state.draft.donors.map((row, donor) => [donorNames[donor], ...row.map((cell, column) => (
          <NumberField key={column} label={`${donorNames[donor]} ${featureLetters[column]}`} value={cell}
            min={limits.donorCell.minimum} max={limits.donorCell.maximum} step="1" decimals={4} allowMissing
            onChange={value => setCell(donor, column, value)} />
        ))]),
      ]} />
    <p className="sc-caption">An empty field means <strong>not recorded</strong>, which is a different state from a measured 0.</p>
    <div className="sc-controls">
      <SelectField label="Target column to estimate" value={String(state.draft.target)}
        onChange={value => setTarget(Number(value))}
        options={featureLetters.map((letter, index) => [String(index), `column ${letter}`])} />
      <NumberField label="Neighbours k" value={state.draft.neighbours} min={limits.neighbours.minimum} max={limits.neighbours.maximum}
        step="1" decimals={0} onChange={value => edit({ neighbours: value })} />
    </div>

    <div className="sc-prediction">
      <fieldset>
        <legend>Record a prediction first.</legend>
        <p>Which donors will supply the estimate, and what will it be? Round your estimate to four decimal places; answers within 0.00005 are accepted.</p>
        <div className="sc-choices">
          {donorNames.map(name => (
            <label className="sc-choice" key={name}>
              <input type="checkbox" name={groupId} value={name} checked={predictedDonors.includes(name)} disabled={Boolean(result)}
                onChange={event => setPredictedDonors(current => (event.target.checked
                  ? [...current, name] : current.filter(item => item !== name)))} />
              <span>{name} donates</span>
            </label>
          ))}
          <label className="sc-choice">
            <input type="checkbox" checked={noData} disabled={Boolean(result)} onChange={event => setNoData(event.target.checked)} />
            <span>No eligible data</span>
          </label>
        </div>
      </fieldset>
      {!noData && <NumberField label="Predicted estimate" value={predictedValue === null ? null : predictedValue}
        min={limits.donorCell.minimum} max={limits.donorCell.maximum} step="1" decimals={4} allowMissing
        onChange={value => setPredictedValue(value)} />}
      <Reason value={reason} onChange={setReason} disabled={Boolean(result)} />
      {state.pending && <p className="sc-pending" role="status">Inputs changed; record a new prediction.</p>}
      <div className="sc-buttons">
        <button type="button" className="is-primary" disabled={Boolean(result) || (!noData && predictedValue === null && !predictedDonors.length)}
          onClick={commit}>Check prediction</button>
        <button type="button" disabled={Boolean(result)}
          onClick={() => { state.advance(); setResult({ key: draftKey, answer: compute(state.draft), donors: null, value: null, noData: false, reason }); }}>
          Calculate without recording a prediction
        </button>
      </div>
      {result?.reason && <p className="sc-caption">Your reason, kept as you wrote it: “{result.reason}”</p>}
      {result && result.donors !== null && <p className={`sc-verdict ${donorsMatch && valueMatch ? '' : 'is-miss'}`} role="status">
        <span className="sc-verdict-mark" aria-hidden="true">{donorsMatch && valueMatch ? '=' : '≠'}</span>
        You recorded {result.noData ? 'no eligible data' : `${result.donors.length ? result.donors.join(' and ') : 'no donors'} supplying ${round(result.value, 6)}`}.
        The calculation uses {result.answer.selected.length
          ? [...result.answer.selected.map(row => row.name)].sort().join(' and ') : 'no eligible donor'} and
        gives {result.answer.estimate === null ? 'no estimate at all' : round(result.answer.estimate, 6)}.
      </p>}
      {result && result.donors === null && <p className="sc-verdict is-plain" role="status">
        <span className="sc-verdict-mark" aria-hidden="true">·</span>
        Calculated without a recorded prediction: {result.answer.estimate === null ? 'no estimate' : round(result.answer.estimate, 6)}.
      </p>}
    </div>

    {result && result.key === draftKey && <>
      <Table caption={`Overlap arithmetic with m = 3 features. Crossed coordinates are unavailable in that comparison.`}
        headings={['donor', 'shared', 'q', 'm ÷ q', 'raw sum', 'adjusted distance', `donates ${featureLetters[active.target]}?`]}
        rowClass={index => (selectedNames.includes(outcome.rows[index].name) ? 'is-selected' : (outcome.rows[index].eligible ? 'is-donor' : 'is-excluded'))}
        rows={outcome.rows.map(row => [
          row.name,
          featureLetters.map((letter, column) => (row.shared.some(item => item.index === column)
            ? <span key={letter}>{letter} </span>
            : <span key={letter} className="sc-struck">{letter} </span>)),
          String(row.q),
          row.factor === null ? 'undefined' : fraction(row.factor),
          row.q === 0 ? '—' : round(row.raw, 6),
          row.squared === null ? 'no overlap: undefined' : round(row.squared, 6),
          row.hasTarget ? `yes (${round(row.targetValue, 4)})` : 'no',
        ])} />
      <p className="sc-readout">
        {outcome.mode === 'neighbours' && <>
          The eligible donors are ranked by adjusted squared distance, and equal distances keep source-table order. The nearest{' '}
          {Math.min(active.neighbours, outcome.selected.length)} supply{' '}
          {outcome.selected.map(row => round(row.targetValue, 4)).join(' and ')}, whose uniform average is{' '}
          <strong>{round(outcome.estimate, 6)}</strong>.
        </>}
        {outcome.mode === 'fallback-mean' && <>
          No donor shares an observed coordinate with the query, so no overlap distance is defined. The fallback is the mean of the observed values
          in column {featureLetters[active.target]}: <strong>{round(outcome.estimate, 6)}</strong>. That is a labelled fallback, not a neighbour.
        </>}
        {outcome.mode === 'no-target-column' && <>
          Every donor is missing column {featureLetters[active.target]}, so this investigation reports that it <strong>cannot estimate</strong> rather
          than quietly inventing a zero. An implementation may instead keep an empty feature; that is a different declared policy.
        </>}
      </p>
      <p className="sc-caption">
        The spread of three donors is not an uncertainty statement. An imputed cell is a supplied model input, not a discovered measurement.
      </p>
    </>}
  </Investigation>;
}

/* ------------------------------------------------------------------ I3 */

const heldOut = rows.filter(row => row[7] === 1);
const recordStart = source => {
  const row = rows[source];
  return { source, numeric: numericOf(row), category: sexOf(row), coordinate: 0 };
};

/** §6 · I3. One held-out record through the frozen fitted bundle. */
export function PipelineLab() {
  const state = useInvestigationWithHistory(recordStart(309));
  const [predicted, setPredicted] = useState(null);
  const [categoryChoice, setCategoryChoice] = useState('');
  const [reason, setReason] = useState('');
  const [result, setResult] = useState(null);
  const active = state.active;
  const observed = rows[active.source];
  const edited = JSON.stringify(numericOf(observed)) !== JSON.stringify(active.numeric) || sexOf(observed) !== active.category;
  const draftEdited = JSON.stringify(numericOf(rows[state.draft.source])) !== JSON.stringify(state.draft.numeric)
    || sexOf(rows[state.draft.source]) !== state.draft.category;
  const compute = inputs => transformRecord(preparation, { numeric: inputs.numeric, category: inputs.category });
  const outcome = compute(active);
  const draftKey = JSON.stringify(state.draft);
  const isCategoryBlock = state.draft.coordinate === 4;
  const units = ['mm', 'mm', 'mm', 'g'];
  const shortNames = ['bill length', 'bill depth', 'flipper length', 'body mass'];
  const edit = update => { state.edit(update); setResult(null); };
  const commit = graded => {
    const answer = compute(state.draft);
    state.advance();
    setResult({
      key: draftKey, answer, graded,
      coordinate: state.draft.coordinate,
      predicted: graded ? predicted : null,
      categoryChoice: graded ? categoryChoice : '',
      reason,
    });
  };
  const categoryVectorText = vector => `[${vector.join(', ')}]`;
  const answerCategory = categoryVectorText(outcome.category.vector);
  const matched = result && (result.coordinate === 4
    ? result.categoryChoice === categoryVectorText(result.answer.category.vector)
    : result.predicted !== null && withinTolerance(result.predicted, result.answer.coordinates[result.coordinate], 0.005));

  return <Investigation
    title="Investigation 3 — Follow a record through the fitted pipeline"
    question="Choose any of the 86 held-out records, optionally edit an independent copy of it, select one output coordinate, and predict its value before applying."
    note={`The fitted medians, centres, scales and vocabulary below were fitted on the ${split.training} training rows and never change here, however you edit the record. That is the whole point of the boundary.`}
    onReset={() => { state.reset(); setPredicted(null); setCategoryChoice(''); setReason(''); setResult(null); }}>

    <div className="sc-controls">
      <SelectField label="Held-out record (zero-based source row)" value={String(state.draft.source)}
        onChange={value => { edit({ ...recordStart(Number(value)), coordinate: state.draft.coordinate }); }}
        options={heldOut.map(row => [String(row[0]), `row ${row[0]}`])} />
      {[0, 1, 2, 3].map(index => (
        <NumberField key={index} label={`${shortNames[index]} (${units[index]})`} value={state.draft.numeric[index]}
          min={recordBounds[index][0]} max={recordBounds[index][1]} step={index === 3 ? '25' : '0.1'} decimals={2} allowMissing
          onChange={value => edit({ numeric: state.draft.numeric.map((cell, position) => (position === index ? value : cell)) })} />
      ))}
      <SelectField label="Recorded sex" value={state.draft.category === null ? '__absent' : state.draft.category}
        onChange={value => edit({ category: value === '__absent' ? null : value })}
        options={[['female', 'female'], ['male', 'male'], ['__absent', 'absent (not recorded)'], ['juvenile', 'juvenile — a value the fit never saw']]} />
      <SelectField label="Output coordinate to predict" value={String(state.draft.coordinate)}
        onChange={value => { edit({ coordinate: Number(value) }); }}
        options={[...featureNames.slice(0, 4).map((name, index) => [String(index), name]), ['4', 'the three sex coordinates']]} />
    </div>
    <p className="sc-caption">
      An empty measurement field means <strong>not recorded</strong>. Clearing one shows what the fitted median supplies in its place.{' '}
      {draftEdited
        ? <span className="sc-badge is-edited">edited copy — not a measured specimen</span>
        : <span className="sc-badge is-observed">observed source row {state.draft.source}</span>}
    </p>

    <div className="sc-prediction">
      <fieldset>
        <legend>Record a prediction first.</legend>
        <p>
          What will coordinate <strong>{isCategoryBlock ? 'sex_female, sex_male, sex_not_recorded' : featureNames[state.draft.coordinate]}</strong>{' '}
          be for this record? Numeric answers within 0.005 are accepted.
        </p>
        {isCategoryBlock
          ? <div className="sc-choices">
            {['[1, 0, 0]', '[0, 1, 0]', '[0, 0, 1]', '[0, 0, 0]'].map(option => (
              <label className="sc-choice" key={option}>
                <input type="radio" name="sc-i3-category" value={option} checked={categoryChoice === option} disabled={Boolean(result)}
                  onChange={() => setCategoryChoice(option)} />
                <span>{option}</span>
              </label>
            ))}
          </div>
          : <NumberField label="Predicted coordinate value" value={predicted} min={-predictionLimit} max={predictionLimit} step="0.1" decimals={4} allowMissing
            onChange={setPredicted} />}
      </fieldset>
      <Reason value={reason} onChange={setReason} disabled={Boolean(result)} />
      {state.pending && <p className="sc-pending" role="status">The record or the selected coordinate changed; record a new prediction.</p>}
      <div className="sc-buttons">
        <button type="button" className="is-primary" disabled={Boolean(result) || (isCategoryBlock ? categoryChoice === '' : predicted === null)}
          onClick={() => commit(true)}>Check prediction</button>
        <button type="button" disabled={Boolean(result)} onClick={() => commit(false)}>Calculate without recording a prediction</button>
      </div>
      {result?.reason && <p className="sc-caption">Your reason, kept as you wrote it: “{result.reason}”</p>}
      {result && result.graded && <p className={`sc-verdict ${matched ? '' : 'is-miss'}`} role="status">
        <span className="sc-verdict-mark" aria-hidden="true">{matched ? '=' : '≠'}</span>
        {matched ? 'Your prediction matches. ' : ''}
        {result.coordinate === 4
          ? `You recorded ${result.categoryChoice}; the calculation gives ${categoryVectorText(result.answer.category.vector)}.`
          : `You recorded ${round(result.predicted, 6)}; the calculation gives ${round(result.answer.coordinates[result.coordinate], 10)}, a deviation of ${round(result.answer.coordinates[result.coordinate] - result.predicted, 6)} in units of that output coordinate.${matched ? ' That is within the 0.005 tolerance this investigation grades on.' : ''}`}
      </p>}
      {result && !result.graded && <p className="sc-verdict is-plain" role="status">
        <span className="sc-verdict-mark" aria-hidden="true">·</span>
        Calculated without a recorded prediction: {result.coordinate === 4
          ? categoryVectorText(result.answer.category.vector)
          : round(result.answer.coordinates[result.coordinate], 10)}.
      </p>}
    </div>

    {result && result.key === draftKey && <>
      <Table caption="The numeric branch, step by step, with the frozen training statistics it uses"
        headings={['column', 'as supplied', 'after imputation', '− fitted centre', '÷ fitted scale', 'output']}
        rowClass={index => (index === active.coordinate ? 'is-selected' : undefined)}
        rows={outcome.numeric.map((cell, index) => [
          `${shortNames[index]} (${units[index]})`,
          cell.wasMissing ? 'absent' : measured(cell.original, units[index]),
          cell.wasMissing ? `${round(cell.filled, 4)} (training median)` : round(cell.filled, 4),
          `${round(cell.filled, 4)} − ${round(cell.center, 6)} = ${round(cell.centered, 6)}`,
          `÷ ${round(cell.scale, 6)}`,
          fixed(cell.output, 10),
        ])} />
      <p className="sc-readout">
        The categorical branch received <strong>{active.category === null ? 'an absent value' : `“${active.category}”`}</strong>, which is{' '}
        {outcome.category.state === 'known' ? 'a fitted category' : outcome.category.state === 'missing'
          ? 'not recorded, so the fitted not_recorded category absorbs it'
          : 'not in the fitted vocabulary, so the declared policy represents it as zeros across the block'}, giving{' '}
        <strong>{answerCategory}</strong> over {preparation.vocabulary.categories.join(', ')}.
      </p>
      <div className="sc-coordinates">
        {outcome.coordinates.map((value, index) => (
          <div className={`sc-coordinate${index < 4 && outcome.numeric[index].wasMissing ? ' is-imputed' : ''}`} key={featureNames[index]}>
            <span>{featureNames[index]}</span>
            <span><strong>{index < 4 ? fixed(value, 10) : String(value)}</strong></span>
          </div>
        ))}
      </div>
      <p className="sc-caption">
        {edited
          ? 'This is an edited copy. The observed file is unchanged, and an edited record is not a measured specimen, so the fixed comparison counts in Figure 4c are not recalculated for it.'
          : `These are the coordinates the observed source row ${active.source} actually produced.`}
        {' '}Species is the answer, not an input: it never enters these seven coordinates.
      </p>
    </>}
  </Investigation>;
}

/* ------------------------------------------------------------------ I4 */

const encodingStart = {
  categories: ['A', 'A', 'B', 'B', 'C', 'C'],
  target: [1, 1, 0, 0, 1, 0],
  folds: [0, 1, 0, 1, 0, 1],
  smoothing: 2,
  inspect: 0,
};

/** §8 · I4. Which target can reach which encoded row. */
export function TargetEncodingLab() {
  const state = useInvestigationWithHistory(encodingStart);
  const [predicted, setPredicted] = useState(null);
  const [reason, setReason] = useState('');
  const [result, setResult] = useState(null);
  const active = state.active;
  const valid = inputs => [0, 1].every(fold => inputs.folds.some(value => value === fold) && inputs.folds.some(value => value !== fold));
  const compute = inputs => crossFitEncoding({
    categories: inputs.categories, target: inputs.target, folds: inputs.folds, smoothing: inputs.smoothing,
  });
  const workable = valid(active);
  const outcome = workable ? compute(active) : null;
  const draftKey = JSON.stringify(state.draft);
  const edit = update => { state.edit(update); setResult(null); };
  const inspected = outcome?.rows[active.inspect];
  const commit = graded => {
    if (!valid(state.draft)) return;
    const answer = compute(state.draft);
    state.advance();
    setResult({ key: draftKey, answer, graded, inspect: state.draft.inspect, predicted: graded ? predicted : null, reason });
  };
  const matched = result && result.predicted !== null
    && withinTolerance(result.predicted, result.answer.rows[result.inspect].value, 0.005);
  const place = index => 30 + index * 26;

  return <Investigation
    title="Investigation 4 — Trace which target can affect which encoded row"
    question="Six outer-training rows, two internal folds and a smoothing constant. Inspect one row, predict its cross-fitted encoded value, then expose the donor graph that produced it."
    note="The whole six-row table is the outer training set; no external evaluation rows are present. Each internal fold takes its sums, its counts and its prior from the other fold alone."
    onReset={() => { state.reset(); setPredicted(null); setReason(''); setResult(null); }}>

    <Table caption="Edit any category, target or fold. Both folds must stay non-empty."
      headings={['row', 'category', 'target', 'internal fold']}
      rowClass={index => (index === state.draft.inspect ? 'is-selected' : undefined)}
      rows={state.draft.categories.map((category, index) => [
        String(index),
        <SelectField key="c" label={`row ${index} category`} value={category}
          onChange={value => edit({ categories: state.draft.categories.map((item, position) => (position === index ? value : item)) })}
          options={['A', 'B', 'C', 'D'].map(name => [name, name])} />,
        <SelectField key="t" label={`row ${index} target`} value={String(state.draft.target[index])}
          onChange={value => edit({ target: state.draft.target.map((item, position) => (position === index ? Number(value) : item)) })}
          options={[['0', '0'], ['1', '1']]} />,
        <SelectField key="f" label={`row ${index} fold`} value={String(state.draft.folds[index])}
          onChange={value => edit({ folds: state.draft.folds.map((item, position) => (position === index ? Number(value) : item)) })}
          options={[['0', 'fold 0'], ['1', 'fold 1']]} />,
      ])} />
    <div className="sc-controls">
      <NumberField label="Smoothing α" value={state.draft.smoothing} min={limits.smoothing.minimum} max={limits.smoothing.maximum}
        step="0.5" decimals={2} onChange={value => edit({ smoothing: value })} />
      <SelectField label="Inspect row" value={String(state.draft.inspect)}
        onChange={value => edit({ inspect: Number(value) })}
        options={state.draft.categories.map((category, index) => [String(index), `row ${index} (${category})`])} />
    </div>
    {!valid(state.draft) && <p className="sc-pending" role="status">
      Both internal folds need at least one row. Move a row back before applying.
    </p>}

    <div className="sc-prediction">
      <fieldset>
        <legend>Record a prediction first.</legend>
        <p>What cross-fitted value will row {state.draft.inspect} receive? Answers within 0.005 are accepted.</p>
        <NumberField label="Predicted encoded value" value={predicted} min={-1} max={2} step="0.05" decimals={4} allowMissing
          onChange={setPredicted} />
      </fieldset>
      <Reason value={reason} onChange={setReason} disabled={Boolean(result)} />
      {state.pending && <p className="sc-pending" role="status">A table cell changed; record a new prediction, even if the result turns out to be unchanged.</p>}
      <div className="sc-buttons">
        <button type="button" className="is-primary" disabled={Boolean(result) || predicted === null || !valid(state.draft)}
          onClick={() => commit(true)}>Check prediction</button>
        <button type="button" disabled={Boolean(result) || !valid(state.draft)} onClick={() => commit(false)}>Calculate without recording a prediction</button>
      </div>
      {result?.reason && <p className="sc-caption">Your reason, kept as you wrote it: “{result.reason}”</p>}
      {result && result.graded && <p className={`sc-verdict ${matched ? '' : 'is-miss'}`} role="status">
        <span className="sc-verdict-mark" aria-hidden="true">{matched ? '=' : '≠'}</span>
        {matched ? 'Your prediction matches. ' : ''}You recorded {round(result.predicted, 6)}; the calculation gives{' '}
        {fraction(result.answer.rows[result.inspect].value)} = {round(result.answer.rows[result.inspect].value, 6)}.
      </p>}
      {result && !result.graded && <p className="sc-verdict is-plain" role="status">
        <span className="sc-verdict-mark" aria-hidden="true">·</span>
        Calculated without a recorded prediction: {fraction(result.answer.rows[result.inspect].value)}.
      </p>}
    </div>

    {result && result.key === draftKey && outcome && inspected && <>
      <div className="sc-panels">
        <div className="sc-panel">
          <h4>Donor edges into row {active.inspect}</h4>
          <svg viewBox="0 0 300 200" role="img" aria-label={`Row ${active.inspect} sits in fold ${inspected.fold}. Its donors are rows ${inspected.donors.join(', ')}, all from the other fold. Rows ${inspected.matching.length ? inspected.matching.join(', ') : 'none'} share its category and enter the numerator; all donors enter the fold prior. Row ${active.inspect} has no edge into its own encoding.`}>
            <FlowDefs />
            {/* Every donor feeds the prior; a same-category donor feeds the
                numerator as well. Drawing only one edge per donor made the
                prior look as if it came from the non-matching rows alone. */}
            {active.categories.map((category, index) => {
              const donor = inspected.donors.includes(index);
              return donor
                ? <path key={`prior-${index}`} className="sc-flow is-target" d={`M126,${place(index)} L206,140`} markerEnd="url(#sc-arrow-target)" />
                : null;
            })}
            {inspected.matching.map(index => (
              <path key={`sum-${index}`} className="sc-flow" d={`M126,${place(index)} L206,74`} markerEnd="url(#sc-arrow)" />
            ))}
            {active.categories.map((category, index) => (
              <g key={index}>
                <rect className={`sc-lane${index === active.inspect ? ' is-target' : (inspected.donors.includes(index) ? '' : ' is-muted')}`}
                  x="8" y={place(index) - 11} width="118" height="22" rx="4" />
                <text x="67" y={place(index) + 4} textAnchor="middle" style={{ fontSize: 10.5 }}>
                  {index} · {category} · y={active.target[index]} · f{active.folds[index]}
                </text>
              </g>
            ))}
            <rect className="sc-lane" x="206" y="60" width="86" height="28" rx="4" />
            <text x="249" y="72" textAnchor="middle" style={{ fontSize: 10.5 }}>same category</text>
            <text x="249" y="83" textAnchor="middle" style={{ fontSize: 10.5 }}>sum {inspected.sum} / n {inspected.count}</text>
            <rect className="sc-lane is-target" x="206" y="126" width="86" height="28" rx="4" />
            <text x="249" y="138" textAnchor="middle" style={{ fontSize: 10.5 }}>fold prior</text>
            <text x="249" y="149" textAnchor="middle" style={{ fontSize: 10.5 }}>{fraction(inspected.prior)}</text>
            <text x="8" y="14" style={{ fontSize: 10.5 }}>solid: numerator · dashed: prior (every donor)</text>
            <text x="8" y="196" style={{ fontSize: 10.5 }}>each chip reads row · category · target · fold</text>
          </svg>
          <p>
            Every donor row has a dashed edge into the fold prior and, if it shares the inspected row&apos;s category, a solid edge into the
            numerator as well. No edge leaves row {active.inspect} into its own value.
          </p>
        </div>
        <div className="sc-panel">
          <h4>The arithmetic for row {active.inspect}</h4>
          <Table caption={`Row ${active.inspect} is held out of fold ${inspected.fold}`}
            headings={['part', 'value']}
            rows={[
              ['donor rows (the other fold)', inspected.donors.join(', ')],
              ['same-category donors', inspected.matching.length ? inspected.matching.join(', ') : 'none'],
              ['their target sum', String(inspected.sum)],
              ['their count n_c', String(inspected.count)],
              ['fold prior μ', fraction(inspected.prior)],
              ['smoothing α', round(active.smoothing, 4)],
              ['numerator sum + αμ', fraction(inspected.numerator)],
              ['denominator n_c + α', fraction(inspected.denominator)],
              ['encoded value', `${fraction(inspected.value)} = ${round(inspected.value, 6)}`],
            ]} />
          {inspected.usedPriorFallback && <p>
            With α = 0 and no same-category donor the manuscript&apos;s formula would divide by zero. The declared policy here is the fold&apos;s own
            prior, which is also what an unseen category receives.
          </p>}
        </div>
      </div>
      <Table caption="All six cross-fitted values under the current table"
        headings={['row', 'category', 'target', 'fold', 'prior used', 'encoded value']}
        rowClass={index => (index === active.inspect ? 'is-selected' : undefined)}
        rows={outcome.rows.map(row => [
          String(row.row), row.category, String(row.target), String(row.fold), fraction(row.prior), fraction(row.value),
        ])} />
      <p className="sc-caption">
        Editing the inspected row&apos;s own target is a checked null for its own value. Editing one of its donors is a contrast — and a donor&apos;s
        target can move another category&apos;s encoding through the fold prior alone.
      </p>
    </>}
  </Investigation>;
}

/* ---------------------------------------------------------------- shared */

/** The draft/active/prediction contract, with a kept previous trial. This is a
 * thin wrapper so each lab above can add its own prediction shape. */
function useInvestigationWithHistory(initial) {
  const [draft, setDraft] = useState(initial);
  const [active, setActive] = useState(initial);
  const [choice, setChoice] = useState('');
  const [reason, setReason] = useState('');
  const [result, setResult] = useState(null);
  const [previous, setPrevious] = useState(null);
  return {
    draft, active, choice, setChoice, reason, setReason, result, previous,
    pending: JSON.stringify(draft) !== JSON.stringify(active),
    /** Promote the draft to the active inputs. A lab with its own prediction
     * shape calls this when it commits, so every displayed detail belongs to
     * the inputs the answer was computed from. */
    advance: () => setActive(draft),
    edit: update => {
      setDraft(current => ({ ...current, ...update }));
      setResult(current => { if (current) setPrevious(current); return null; });
      setChoice('');
    },
    check: (answerFor, label) => {
      setActive(draft);
      setPrevious(result);
      setResult({ key: JSON.stringify(draft), choice, reason, answer: answerFor(draft), graded: true, label: label?.(draft) });
    },
    explore: (answerFor, label) => {
      setActive(draft);
      setPrevious(result);
      setResult({ key: JSON.stringify(draft), choice: '', reason, answer: answerFor(draft), graded: false, label: label?.(draft) });
    },
    reset: () => { setDraft(initial); setActive(initial); setChoice(''); setReason(''); setResult(null); setPrevious(null); },
    load: inputs => { setDraft(inputs); setActive(inputs); setChoice(''); setReason(''); setResult(null); setPrevious(null); },
  };
}
