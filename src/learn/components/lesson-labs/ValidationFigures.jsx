import { useState } from 'react';
import {
  bootstrapDistinctShare, consecutiveFolds, expectedImprovement, fixedRuleAccuracyVariance, foldMeanVariance, foldPlan,
  hitProbability, meanPredictorRisk, outOfFoldCoverage, tpeAcquisition,
} from '../../data/validation-models';
import { NESTED_EXPERIMENT, RANDOM_COVERAGE_POINTS } from '../../data/validation-data';
import { Figure, Legend, Plot, RoleMark, Table, curve, fixed, round } from './ValidationShared.jsx';
import './validation-labs.css';

/** A line with a small solid head, so a connector's direction is visible
 * without depending on an SVG marker definition. */
function Arrow({ from, to, className = 'cv-flow', head = 5 }) {
  const [x1, y1] = from;
  const [x2, y2] = to;
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const tip = [x2, y2];
  const left = [x2 - head * Math.cos(angle - 0.42), y2 - head * Math.sin(angle - 0.42)];
  const right = [x2 - head * Math.cos(angle + 0.42), y2 - head * Math.sin(angle + 0.42)];
  return <g>
    <line className={className} x1={x1} y1={y1} x2={x2 - head * 0.7 * Math.cos(angle)} y2={y2 - head * 0.7 * Math.sin(angle)} />
    <polygon points={[tip, left, right].map(point => point.join(',')).join(' ')} fill="currentColor"
      style={{ color: className.includes('is-score') ? '#e7b94a' : className.includes('is-final') ? '#a8c2e0' : '#8eb9a5' }} />
  </g>;
}

/** A polyline through the integer values of an analytic function. Sampling a
 * continuous axis and rounding n produced a staircase that read as an artefact
 * rather than as the formula. */
function integerCurve(scaleX, scaleY, from, to, evaluate) {
  return Array.from({ length: to - from + 1 }, (_, index) => {
    const n = from + index;
    return `${scaleX(n).toFixed(2)},${scaleY(evaluate(n)).toFixed(2)}`;
  }).join(' ');
}

function Box({ x, y, width, height, label, kind = '', lines }) {
  const rows = lines ?? [label];
  return <g>
    <rect className={`cv-box ${kind}`} x={x} y={y} width={width} height={height} rx="3" />
    {rows.map((text, index) => (
      <text key={text} x={x + width / 2} y={y + height / 2 + 5 + (index - (rows.length - 1) / 2) * 15} textAnchor="middle">{text}</text>
    ))}
  </g>;
}

/* ================================================================== *
 * F1 · §1 — the fitting loop, the selection loop and what sits outside
 * ================================================================== */
export function LoopFigure() {
  return <Figure id="loops" caption="Figure 1 — Two loops, two kinds of learning. The inner loop learns parameters from training rows. The outer loop learns a setting from validation answers. The assessment rows sit outside both and send nothing back.">
    <svg viewBox="0 0 360 326" role="img" aria-label="A setting card and a bundle of training rows enter a box that fits the whole procedure, producing a fitted model. That fitted model and the held-out validation rows both feed a validation score, which returns along the left-hand side to the setting card: that return path is the selection loop. Below a separator labelled outside both loops, a box reading selected, refitted sends one arrow into a dashed box of protected assessment rows. No arrow leaves the protected box.">
      <Box x={30} y={8} width={140} height={26} label="setting: k = 5" />
      <Box x={182} y={8} width={170} height={26} label="training rows" kind="is-train" />
      <Arrow from={[100, 34]} to={[140, 52]} />
      <Arrow from={[267, 34]} to={[230, 52]} />
      <Box x={60} y={52} width={260} height={26} label="fit the whole procedure" />
      <Arrow from={[190, 78]} to={[190, 92]} />
      <Box x={90} y={92} width={200} height={26} label="fitted model" />
      <Arrow from={[190, 118]} to={[250, 132]} />
      <Box x={30} y={132} width={140} height={26} label="validation rows" kind="is-held" />
      <Arrow from={[170, 145]} to={[181, 145]} />
      <Box x={182} y={132} width={170} height={26} label="validation score" />
      <line className="cv-flow is-score" x1="267" y1="158" x2="267" y2="174" />
      <line className="cv-flow is-score" x1="267" y1="174" x2="18" y2="174" />
      <line className="cv-flow is-score" x1="18" y1="174" x2="18" y2="21" />
      <Arrow from={[18, 21]} to={[30, 21]} className="cv-flow is-score" />
      <text x="180" y="196" textAnchor="middle">selection loop returns the score</text>
      <line className="cv-grid" x1="8" y1="210" x2="352" y2="210" strokeDasharray="3 3" />
      <text x="180" y="228" textAnchor="middle">outside both loops</text>
      <Box x={92} y={234} width={196} height={26} label="selected, refitted" />
      <Arrow from={[190, 260]} to={[190, 276]} className="cv-flow is-final" />
      <Box x={50} y={276} width={280} height={30} kind="is-protected" lines={['protected assessment rows']} />
      <text x="180" y="320" textAnchor="middle">no arrow returns from here</text>
    </svg>
    <Legend kinds={['train', 'held', 'protected']} />
    <p>
      The inner box learns <strong>parameters</strong>: every median, scale, one-hot vocabulary and model coefficient, from training rows only. The
      outer path learns a <strong>setting</strong>: the score that comes back is what chooses <span style={{ whiteSpace: 'nowrap' }}>k = 5</span> over
      its rivals. The protected rows receive the selected, refitted procedure and send nothing back into either loop.
    </p>
    <p className="cv-caption">
      The proportions drawn here are not a recommendation. There is no universal 60/20/20 split; what matters is which rows were allowed to
      influence which decision.
    </p>
  </Figure>;
}

/* ================================================================== *
 * F2 · §3 — the same records, two different questions
 * ================================================================== */
const RECORDS = ['A day 1', 'A day 2', 'A day 3', 'B day 1', 'B day 2', 'B day 3'];
const person = id => id[0];

function RecordPanel({ title, heldOut, question, verdict }) {
  const plan = foldPlan({ rows: RECORDS, folds: [heldOut], groupOf: person, requirePartition: false });
  const fold = plan.folds[0];
  return <section className="cv-panel">
    <h4>{title}</h4>
    <p>{question}</p>
    <Table caption={`${title}: the role of every record`} headings={['record', 'person', 'role in this fit']}
      rows={RECORDS.map(id => [id, person(id), fold.validation.includes(id)
        ? <RoleMark kind="held" key={id} />
        : <RoleMark kind="train" key={id} />])} />
    <p>{plan.grouping.clean
      ? 'No person appears on both sides of this boundary, so the fit never saw this person before.'
      : `Both people appear on both sides here (${plan.grouping.spanning.map(item => item.group).join(' and ')}). That is what the question asks for.`}</p>
    <p><strong>{verdict}</strong></p>
  </section>;
}

export function SplitQuestionFigure() {
  const forward = foldPlan({ rows: [0, 1, 2, 3, 4, 5, 6, 7], folds: [[4, 5], [6, 7]], requirePartition: false });
  const kfold = foldPlan({ rows: [0, 1, 2, 3, 4, 5, 6, 7], folds: consecutiveFolds(8, 4) });
  const forwardRows = outOfFoldCoverage(forward);
  const kfoldRows = outOfFoldCoverage(kfold);
  const day = value => 40 + (value - 8) * 31;
  return <Figure id="split-question" caption="Figure 2 — Same records, different questions. Neither panel is the correct one in general; each answers the question written above it.">
    <div className="cv-panels">
      <RecordPanel title="Unseen-person question" heldOut={['B day 1', 'B day 2', 'B day 3']}
        question="Will this work for a person who was never in the training data?"
        verdict="Hold out every record of that person." />
      <RecordPanel title="Later-records question" heldOut={['A day 3', 'B day 3']}
        question="Will this predict tomorrow for people who already gave us a week of history?"
        verdict="Train on their earlier days and assess the later one." />
    </div>
    <p>
      Calling the second panel wrong would change the question. What would be wrong is using information unavailable at the actual prediction
      point, or offering one of these panels as evidence about the other deployment.
    </p>
    <h4>A seven-day target horizon moves the cutoff</h4>
    <svg viewBox="0 0 360 116" role="img" aria-label="A day axis from 8 to 16. A filled dot on day 9 marks a feature row. A dashed span runs from day 9 to day 16, the seven days over which its outcome is measured, ending in a hollow dot: the label is available only then. A solid vertical line on day 10 marks the moment a fit is made, and it falls inside that span, so the row has no known target yet.">
      <text x="180" y="22" textAnchor="middle">outcome measured over 7 days</text>
      <line className="cv-flow" x1={day(9)} y1="36" x2={day(16)} y2="36" strokeDasharray="4 3" />
      <line className="cv-flow" x1={day(9)} y1="31" x2={day(9)} y2="41" />
      <line className="cv-flow" x1={day(16)} y1="31" x2={day(16)} y2="41" />
      <line className="cv-flow is-final" x1={day(10)} y1="50" x2={day(10)} y2="76" />
      <text x={day(10) + 8} y="62">fit made here</text>
      <line className="cv-axis" x1="30" y1="80" x2="350" y2="80" />
      {[8, 9, 10, 11, 12, 13, 14, 15, 16].map(value => <g key={value}>
        <line className="cv-axis" x1={day(value)} y1="76" x2={day(value)} y2="84" />
        <text x={day(value)} y="100" textAnchor="middle">{value}</text>
      </g>)}
      <circle className="cv-mark" cx={day(9)} cy="80" r="5" />
      <circle className="cv-mark is-hollow" cx={day(16)} cy="80" r="5" />
    </svg>
    <p className="cv-caption">
      Day numbers run along the axis. Filled dot on day 9: the feature row is formed. Hollow dot on day 16: its label finally becomes
      available. The solid vertical line on day 10 is the moment a fit is made, and it falls inside the seven-day measurement span.
    </p>
    <p>
      A row formed on day 9 has no known target on day 10. Dropping rows by their feature timestamp alone is not enough: the feature date, the
      label-availability date and the prediction cutoff are three separate dates, and a gap between them is a consequence of how the label
      matures, not decorative spacing.
    </p>
    <h4>What a one-prediction-per-row table requires</h4>
    <div className="cv-panels">
      <section className="cv-panel">
        <h4>Forward plan: train 0–3 assess 4–5, then train 0–5 assess 6–7</h4>
        <Table caption="Held-out coverage under the forward plan" headings={['row', 'held out by']}
          rows={forwardRows.map(row => [row.row, row.usable
            ? <RoleMark key={row.row} kind="held">{row.status}</RoleMark>
            : <RoleMark key={row.row} kind="absent">{row.status}</RoleMark>])} />
        <p>Rows 0–3 are explicitly absent, not zero. This is a valid way to score folds; it cannot fill a once-per-row prediction table.</p>
      </section>
      <section className="cv-panel">
        <h4>Four-fold partition over the same eight rows</h4>
        <Table caption="Held-out coverage under the K-fold partition" headings={['row', 'held out by']}
          rows={kfoldRows.map(row => [row.row, <RoleMark key={row.row} kind="held">{row.status}</RoleMark>])} />
        <p>Every row is held out exactly once, which is the contract <code>cross_val_predict</code> needs.</p>
      </section>
    </div>
  </Figure>;
}

/* ================================================================== *
 * F3 · §4 — nested rooms, drawn with the real experiment's row IDs
 * ================================================================== */
export function NestedRoomsFigure() {
  const fold = NESTED_EXPERIMENT.folds[0];
  const sample = list => `${list.slice(0, 6).join(', ')}, … (${list.length} rows)`;
  const innerSizes = fold.innerSplits.map(split => split.validationRows.length);
  return <Figure id="nested-rooms" caption="Figure 3 — Nested rooms for information, using outer fold 1 of the real experiment. Original row IDs stay visible at both levels.">
    <ol className="cv-stages">
      <li>
        <h4>1. Split the development rows once</h4>
        <p>
          <RoleMark kind="protected" /> rows {sample(fold.testRows)} are closed off before anything is fitted or compared.{' '}
          <RoleMark kind="train" /> rows {sample(fold.trainingRows)} are all that the search may use.
        </p>
      </li>
      <li>
        <h4>2. Expand only the outer training room into inner folds</h4>
        <svg viewBox="0 0 360 96" role="img" aria-label={`The ${fold.trainingRows.length} outer training rows are divided into three inner validation folds of ${innerSizes.join(', ')} rows. The ${fold.testRows.length} protected rows sit outside and are not divided.`}>
          <Box x={8} y={8} width={230} height={24} label={`outer training · ${fold.trainingRows.length}`} kind="is-train" />
          <Box x={240} y={8} width={112} height={24} label={`protected ${fold.testRows.length}`} kind="is-protected" />
          {innerSizes.map((size, index) => <g key={index}>
            <Arrow from={[123, 32]} to={[52 + index * 78, 48]} />
            <Box x={16 + index * 78} y={48} width={72} height={24} label={`${size}`} kind="is-held" />
            <text x={52 + index * 78} y="88" textAnchor="middle">inner {index + 1}</text>
          </g>)}
        </svg>
        <p>
          Each inner fold assesses {innerSizes.join(', ')} of the outer training rows while the other two fit the complete pipeline. Six
          candidates × three inner folds = {NESTED_EXPERIMENT.folds[0].candidates.length * 3} candidate fits inside this one outer fold.
        </p>
        <ul className="cv-id-list">
          {fold.innerSplits.map((split, index) => (
            <li key={index}><strong>inner {index + 1}</strong> assesses rows {sample(split.validationRows)}</li>
          ))}
        </ul>
        <p className="cv-caption">
          Those are the same original row IDs as step 1, drawn only from the outer training list. Not one protected ID appears in any of the
          three.
        </p>
      </li>
      <li>
        <h4>3. Select, then refit on the whole outer training room</h4>
        <p>
          The inner comparison selects <strong>{fold.selected.k} neighbours with {fold.selected.scaler}</strong> at inner selection score{' '}
          {fixed(fold.selectionScore, 7)}. That candidate is then refitted from scratch on all {fold.trainingRows.length} outer training rows —
          a nineteenth fit, not a reused inner one.
        </p>
      </li>
      <li>
        <h4>4. One arrow reaches the protected rows</h4>
        <p>
          The refitted object predicts the {fold.testRows.length} protected rows and gets {fold.correct} right. That result is an assessment of
          the whole select-and-refit procedure at this training size. The inner score {fixed(fold.selectionScore, 7)} chose the setting and is
          not a second assessment of it.
        </p>
      </li>
      <li>
        <h4>5. Close the room and start again</h4>
        <p>
          The next outer fold builds a fresh object from its own rows. The three outer folds select different settings; those are not three
          competing candidates to vote among, because each was chosen from a different training sample.
        </p>
      </li>
    </ol>
  </Figure>;
}

/** §6 — a read-only explorer over the retained real result. Nothing here is
 * editable: these are recorded measurements, not a slider. */
export function RealExperimentExplorer() {
  const [index, setIndex] = useState(0);
  const finalSearch = NESTED_EXPERIMENT.finalSelected;
  const isFinal = index === NESTED_EXPERIMENT.folds.length;
  const fold = isFinal ? null : NESTED_EXPERIMENT.folds[index];
  const candidates = isFinal ? finalSearch.candidates : fold.candidates;
  const bestIndex = isFinal ? finalSearch.bestIndex : fold.bestIndex;
  const best = candidates[bestIndex];
  const tied = candidates.filter(candidate => candidate.meanScore === best.meanScore);
  const buttons = [
    ...NESTED_EXPERIMENT.folds.map((item, position) => [position, `Outer fold ${position + 1}`]),
    [NESTED_EXPERIMENT.folds.length, 'Final search · all 344 rows'],
  ];
  return <Figure id="real-experiment" caption="The recorded experiment, fold by fold, and the final all-data search beside them. These are read-only measurements from the run whose program is printed above.">
    <div className="cv-buttons" role="group" aria-label="Choose an outer fold or the final search">
      {buttons.map(([position, label]) => (
        <button key={position} type="button" aria-pressed={position === index} onClick={() => setIndex(position)}>{label}</button>
      ))}
    </div>
    {isFinal && <>
      <div className="cv-readout">
        <dl>
          <dt>Rows supplied to this search</dt><dd>{NESTED_EXPERIMENT.assessedRows}</dd>
          <dt>Inner folds</dt><dd>{NESTED_EXPERIMENT.innerFolds}, seed {NESTED_EXPERIMENT.innerSeed}</dd>
          <dt>Protected rows left over</dt><dd>none</dd>
        </dl>
      </div>
      <Table id="final-candidates" caption="The final search over all 344 rows. Every candidate is scored on the same three inner folds; the selected row is marked."
        headings={['candidate', 'inner 1', 'inner 2', 'inner 3', 'inner mean', 'role']}
        numeric={[1, 2, 3, 4]}
        rowClass={position => (position === bestIndex ? 'is-selected' : undefined)}
        rows={candidates.map((candidate, position) => [
          `k=${candidate.k} · ${candidate.scaler.replace('Scaler', '')}`,
          fixed(candidate.foldScores[0], 6), fixed(candidate.foldScores[1], 6), fixed(candidate.foldScores[2], 6),
          fixed(candidate.meanScore, 6),
          position === bestIndex ? 'selected' : '—',
        ])} />
      <p>
        The six candidates collapse into three tied pairs, so the scaler never differs from its partner here:{' '}
        <strong>{finalSearch.tiedWith.join(' and ')}</strong> share the top inner mean {fixed(finalSearch.selectionScore, 7)}, and the declared
        rule takes the first of them. The scaler in the model you would deploy is settled by enumeration order, not by evidence.
      </p>
      <div className="cv-readout">
        <dl>
          <dt>Selection score (chose the setting)</dt><dd>{fixed(finalSearch.selectionScore, 7)}</dd>
          <dt>Protected result (assessed the procedure)</dt><dd>none — every row was used to choose</dd>
        </dl>
      </div>
      <p className="cv-caption">
        This search has no assessment row of its own, by construction. The three outer folds above are what assess the procedure that produced
        it; this table is selection evidence only.
      </p>
    </>}
    {!isFinal && <>
    <div className="cv-readout">
      <dl>
        <dt>Outer training rows</dt><dd>{fold.trainingRows.length}</dd>
        <dt>Protected rows</dt><dd>{fold.testRows.length}</dd>
        <dt>Inner validation sizes</dt><dd>{fold.innerSplits.map(split => split.validationRows.length).join(' · ')}</dd>
        <dt>First protected row IDs</dt><dd>{fold.testRows.slice(0, 8).join(', ')} …</dd>
      </dl>
    </div>
    <Table id="candidates" caption={`Outer fold ${index + 1}: all six candidates, scored only on inner validation rows. The selected row is marked.`}
      headings={['candidate', 'inner 1', 'inner 2', 'inner 3', 'inner mean', 'role']}
      numeric={[1, 2, 3, 4]}
      rowClass={position => (position === fold.bestIndex ? 'is-selected' : undefined)}
      rows={fold.candidates.map((candidate, position) => [
        `k=${candidate.k} · ${candidate.scaler.replace('Scaler', '')}`,
        fixed(candidate.foldScores[0], 6), fixed(candidate.foldScores[1], 6), fixed(candidate.foldScores[2], 6),
        fixed(candidate.meanScore, 6),
        position === fold.bestIndex ? 'selected' : '—',
      ])} />
    <p>
      {tied.length > 1
        ? `${tied.length} of the six candidates share the top inner mean ${fixed(best.meanScore, 7)} here. The declared rule takes the first of them in the enumeration, which is k=${best.k} with ${best.scaler}. Choosing among ties after seeing the outer result would be a different, unreproducible procedure.`
        : `The highest inner mean is ${fixed(best.meanScore, 7)}, reached only by k=${best.k} with ${best.scaler}.`}
    </p>
    <div className="cv-readout">
      <dl>
        <dt>Selection score (chose the setting)</dt><dd>{fixed(fold.selectionScore, 7)}</dd>
        <dt>Protected result (assessed the procedure)</dt><dd>{fold.correct} / {fold.testRows.length} = {fixed(fold.accuracy, 7)}</dd>
        <dt>Majority baseline on the same rows</dt><dd>{fold.baselineCorrect} / {fold.testRows.length} (always “{fold.baselineLabel}”)</dd>
        <dt>Protected rows predicted wrongly</dt><dd>{fold.missed.map(row => `#${row.row} ${row.truth}→${row.prediction}`).join('; ') || 'none'}</dd>
      </dl>
    </div>
    <p className="cv-caption">
      The two scores above answer different questions and are never averaged together. Row IDs are zero-based positions in the original file.
    </p>
    </>}
  </Figure>;
}

/* ================================================================== *
 * F4 · §5 — coverage seen from one coordinate
 * ================================================================== */
const GRID_VALUES = [0.15, 0.5, 0.85];

function CoveragePanel({ title, points, note, describe }) {
  // Inset the drawing area inside its frame, so a draw at 0 or 1 sits inside
  // the square rather than half on its own border.
  const place = value => 38 + 284 * value;
  const lift = value => 142 - 128 * value;
  return <section className="cv-panel">
    <h4>{title}</h4>
    <svg viewBox="0 0 360 206" role="img" aria-label={describe}>
      <rect className="cv-box" x="30" y="8" width="300" height="140" />
      {points.map(([x, y], index) => <circle key={index} className="cv-mark" cx={place(x)} cy={lift(y)} r="4" />)}
      <line className="cv-axis" x1="30" y1="168" x2="330" y2="168" />
      {points.map(([x], index) => <g key={`p${index}`}>
        <line className="cv-grid" x1={place(x)} y1="148" x2={place(x)} y2="168" strokeDasharray="2 3" />
        <line className="cv-stem" x1={place(x)} y1="162" x2={place(x)} y2="174" />
      </g>)}
      <text x="180" y="192" textAnchor="middle">projected onto the important coordinate</text>
    </svg>
    <p>{note}</p>
  </section>;
}

export function CoverageFigure() {
  const gridPoints = GRID_VALUES.flatMap(x => GRID_VALUES.map(y => [x, y]));
  const distinct = new Set(RANDOM_COVERAGE_POINTS.map(point => point[0].toFixed(12))).size;
  const trials = 60;
  return <Figure id="coverage" caption="Figure 4 — Coverage seen from one coordinate, and what a hit probability actually says.">
    <div className="cv-panels">
      <CoveragePanel title="Nine grid positions" points={gridPoints}
        describe="A unit square holding a three-by-three grid at coordinates 0.15, 0.5 and 0.85. Projected downward, the nine points fall on only three distinct positions along the horizontal coordinate."
        note={`Three values on each axis. Projected down, the nine positions collapse onto ${GRID_VALUES.length} distinct places along the coordinate that matters.`} />
      <CoveragePanel title="Nine disclosed random draws" points={RANDOM_COVERAGE_POINTS}
        describe={`A unit square holding nine uniform draws generated by NumPy default_rng seed 5, the first at 0.8050 and 0.8079. Projected downward they fall on ${distinct} distinct positions along the horizontal coordinate.`}
        note={`These are the nine actual draws from NumPy default_rng(5), listed below. Projected down they take ${distinct} distinct values.`} />
    </div>
    <Table id="draws" caption="The nine disclosed draws. These are generated search positions, not measured model accuracies."
      headings={['draw', 'coordinate 1', 'coordinate 2']} numeric={[1, 2]}
      rows={RANDOM_COVERAGE_POINTS.map((point, index) => [index + 1, fixed(point[0], 7), fixed(point[1], 7)])} />
    <p>
      Nothing in this picture measures a score. It compares where two search plans place their evaluations, which is why the lesson does not
      draw an accuracy surface underneath it.
    </p>
    <h4>Probability of entering a region with a stated sampling mass</h4>
    <Plot caption="Analytic curve 1 − (1 − p)ᵀ for two declared masses. The vertical axis is a probability of drawing inside the region, never a probability of being within 5% of the best score."
      height={200} domain={[0, 120]} range={[0, 1]} xTicks={[0, 30, 60, 90, 120]} yTicks={[0, 0.25, 0.5, 0.75, 1]}
      xLabel="number of independent draws T" yLabel="probability of at least one draw inside the region"
      describe={`Two rising curves. With mass p equal to 0.05 the probability after 60 draws is ${fixed(hitProbability(0.05, 60), 7)}; with mass 0.01 it is ${fixed(hitProbability(0.01, 60), 7)}. Both approach one slowly.`}>
      {(scaleX, scaleY) => <>
        <polyline className="cv-curve is-first" points={integerCurve(scaleX, scaleY, 0, 120, draws => hitProbability(0.05, draws))} />
        <polyline className="cv-curve is-second" points={integerCurve(scaleX, scaleY, 0, 120, draws => hitProbability(0.01, draws))} />
        <line className="cv-grid" x1={scaleX(trials)} x2={scaleX(trials)} y1={scaleY(0)} y2={scaleY(1)} strokeDasharray="4 3" />
        <circle className="cv-mark" cx={scaleX(trials)} cy={scaleY(hitProbability(0.05, trials))} r="4" />
        <circle className="cv-mark is-hollow" cx={scaleX(trials)} cy={scaleY(hitProbability(0.01, trials))} r="4" />
        <text x={scaleX(trials) + 6} y={scaleY(hitProbability(0.05, trials)) - 6}>p = .05</text>
        <text x={scaleX(trials) + 6} y={scaleY(hitProbability(0.01, trials)) + 14}>p = .01</text>
      </>}
    </Plot>
    <Table id="hits" caption="Exact values at the marked point. Solid mark: mass .05. Hollow mark: mass .01."
      headings={['sampling mass p', 'draws T', 'probability of at least one hit']} numeric={[1, 2]}
      rows={[[0.05, trials, fixed(hitProbability(0.05, trials), 7)], [0.01, trials, fixed(hitProbability(0.01, trials), 7)]]} />
    <p className="cv-caption">
      Both curves assume independent draws from the declared sampling distribution. A region holding 5% of the sampling probability is a
      different object from a score within 5% of an optimum.
    </p>
  </Figure>;
}

/* ================================================================== *
 * F5 · §7 — name the distribution being averaged
 * ================================================================== */
export function RiskFigure() {
  const sigmaSquared = 4;
  const full = meanPredictorRisk(sigmaSquared, 12);
  const threeFold = meanPredictorRisk(sigmaSquared, 8);
  return <Figure id="risk" caption="Figure 5 — Name the distribution being averaged. Both curves below are exact formulas for the mean predictor, not measured learning curves.">
    <div className="cv-panels">
      <section className="cv-panel">
        <h4>Hold the fitted model fixed</h4>
        <svg viewBox="0 0 360 130" role="img" aria-label="One fitted model at the top. Three arrows leave it to three separate new assessment sets drawn from the population. The model does not change between them.">
          <Box x={100} y={8} width={160} height={26} label="one fitted model" />
          {[0, 1, 2].map(index => <g key={index}>
            <Arrow from={[180, 34]} to={[70 + index * 110, 62]} />
            <Box x={22 + index * 110} y={62} width={96} height={24} label={`new set ${index + 1}`} kind="is-protected" />
          </g>)}
          <text x="180" y="112" textAnchor="middle">the model is the same in all three</text>
        </svg>
        <p>This averages over new observations only. It is a question about one fitted object.</p>
      </section>
      <section className="cv-panel">
        <h4>Redraw the training sample too</h4>
        <svg viewBox="0 0 360 130" role="img" aria-label="Three separate training samples each produce their own fitted model, and each model then meets its own new observation. Both the training sample and the new observation are redrawn.">
          {[0, 1, 2].map(index => <g key={index}>
            <Box x={22 + index * 110} y={8} width={96} height={24} label={`sample ${index + 1}`} kind="is-train" />
            <Arrow from={[70 + index * 110, 32]} to={[70 + index * 110, 50]} />
            <Box x={22 + index * 110} y={50} width={96} height={24} label={`model ${index + 1}`} />
            <Arrow from={[70 + index * 110, 74]} to={[70 + index * 110, 90]} />
            <Box x={22 + index * 110} y={90} width={96} height={24} label="new draw" kind="is-protected" />
          </g>)}
        </svg>
        <p>This averages over training samples as well. It is a question about a learning recipe at a stated training size.</p>
      </section>
    </div>
    <Plot caption="Mean predictor with σ² = 4: expected training loss 4(1 − 1/n) below the irreducible line, expected new loss 4(1 + 1/n) above it."
      height={210} domain={[2, 40]} range={[2, 6]} xTicks={[2, 8, 12, 20, 30, 40]} yTicks={[2, 3, 4, 5, 6]}
      xLabel="training size n" yLabel="expected squared loss, in squared target units"
      describe={`Two analytic curves around a flat irreducible line at 4. At a training size of 8 the expected new loss is ${fixed(threeFold.expectedNewLoss, 4)}; at 12 it is ${fixed(full.expectedNewLoss, 4)}. The expected training loss at 12 is ${fixed(full.expectedTrainingLoss, 4)}.`}>
      {(scaleX, scaleY) => <>
        <polyline className="cv-curve is-flat" points={integerCurve(scaleX, scaleY, 2, 40, () => sigmaSquared)} />
        <polyline className="cv-curve is-first" points={integerCurve(scaleX, scaleY, 2, 40, n => meanPredictorRisk(sigmaSquared, n).expectedNewLoss)} />
        <polyline className="cv-curve is-second" points={integerCurve(scaleX, scaleY, 2, 40, n => meanPredictorRisk(sigmaSquared, n).expectedTrainingLoss)} />
        <circle className="cv-mark" cx={scaleX(8)} cy={scaleY(threeFold.expectedNewLoss)} r="5" />
        <circle className="cv-mark is-hollow" cx={scaleX(12)} cy={scaleY(full.expectedNewLoss)} r="5" />
      </>}
    </Plot>
    <p className="cv-caption">
      The filled mark on the upper curve is n = 8, the training size a three-fold split reaches from twelve rows; the hollow mark beside it is
      n = 12. Their exact values are in the table below, so no label sits on top of a curve.
    </p>
    <Table id="risk-values" caption="Exact values. Solid curve: expected new loss. Dashed curve: expected training loss. Dotted line: the irreducible 4."
      headings={['training size', 'expected training loss', 'irreducible', 'expected new loss', 'optimism']} numeric={[1, 2, 3, 4]}
      rows={[8, 12].map(m => {
        const risk = meanPredictorRisk(sigmaSquared, m);
        return [m, fixed(risk.expectedTrainingLoss, 4), fixed(risk.irreducible, 4), fixed(risk.expectedNewLoss, 4), fixed(risk.optimism, 4)];
      })} />
    <p>
      A three-fold estimate on twelve rows targets performance at about eight training rows, so {fixed(threeFold.expectedNewLoss, 1)} against{' '}
      {fixed(full.expectedNewLoss, 4)} is a difference in <em>training size</em>, not evidence that a splitter leaked.
    </p>
    <h4>Where the variance actually comes from</h4>
    <svg viewBox="0 0 360 142" role="img" aria-label="A five-by-five grid of fold pairs. The five diagonal cells are marked V for the per-fold variances. The twenty off-diagonal cells are marked C for the covariance terms, which depend on the losses and not on the fraction of shared training rows.">
      {Array.from({ length: 5 }, (_, row) => Array.from({ length: 5 }, (_, column) => (
        <rect key={`${row}-${column}`} className={`cv-box ${row === column ? 'is-variance' : 'is-covariance'}`} x={100 + column * 26} y={10 + row * 26} width={24} height={24} />
      )))}
      {Array.from({ length: 5 }, (_, row) => Array.from({ length: 5 }, (_, column) => (
        <text key={`g${row}-${column}`} x={112 + column * 26} y={27 + row * 26} textAnchor="middle"
          className={`cv-covariance-label ${row === column ? 'is-variance' : 'is-covariance'}`}>{row === column ? 'V' : 'C'}</text>
      )))}
      <text x="90" y="24" textAnchor="end">fold 1</text>
      <text x="90" y="128" textAnchor="end">fold 5</text>
    </svg>
    <p className="cv-caption">Five variances (V) sit on the diagonal; the twenty covariance terms (C) surround them.</p>
    <Table id="fold-mean-variance" caption="What the simplifying assumption buys: the variance of a five-fold mean when every fold has variance τ² = 1 and every pair shares one correlation ρ. These are consequences of the assumption, not measurements of overlap."
      headings={['common correlation ρ', 'variance of the fold mean', 'compared with independent folds']} numeric={[1, 2]}
      rows={[0, 0.25, 0.5, 1].map(rho => [
        fixed(rho, 2),
        fixed(foldMeanVariance({ tauSquared: 1, k: 5, rho }), 4),
        `${fixed(foldMeanVariance({ tauSquared: 1, k: 5, rho }) / foldMeanVariance({ tauSquared: 1, k: 5, rho: 0 }), 2)}×`,
      ])} />
    <p>
      The off-diagonal terms are covariances between <strong>losses</strong>. They are not a function of the percentage of training rows two
      folds share. Under equal variances τ² and one common correlation ρ the average has variance τ²[1 + (K − 1)ρ]/K — but those are
      assumptions that explain the formula, not a measurement of overlap.
    </p>
    <h4>The counterexample: a rule that ignores its training rows</h4>
    <svg viewBox="0 0 360 118" role="img" aria-label="A dashed path from the training rows toward the rule is marked no influence, while a solid arrow runs from the held-out row into a box reading always predicts zero. Only the held-out observation decides whether that prediction is correct, so the leave-one-out accuracy has variance one over four n.">
      <Box x={8} y={10} width={150} height={26} label="training rows" kind="is-train" />
      <Box x={202} y={10} width={150} height={26} label="held-out row" kind="is-held" />
      <line className="cv-grid" x1="83" y1="36" x2="140" y2="64" strokeDasharray="4 4" />
      <text x="96" y="60">no influence</text>
      <Arrow from={[277, 36]} to={[240, 64]} />
      <Box x={100} y={64} width={160} height={26} label="always predicts 0" />
      <text x="180" y="106" textAnchor="middle">variance of the LOO accuracy = 1/(4n)</text>
    </svg>
    <p>
      The leave-one-out training sets overlap almost completely, yet the correctness indicators are independent Bernoulli(½) draws, so the
      average has variance {round(fixedRuleAccuracyVariance(1), 4)}/n — exactly {round(fixedRuleAccuracyVariance(20), 4)} at n = 20. Overlap
      alone therefore cannot determine loss correlation.
    </p>
    <p className="cv-caption">
      A bootstrap sample of n positions leaves a given row out with probability (1 − 1/n)ⁿ, so it contains about{' '}
      {round(100 * bootstrapDistinctShare(344).distinctShare, 1)}% distinct original rows at n = 344, approaching{' '}
      {round(100 * bootstrapDistinctShare(1000000).distinctShare, 1)}%. That is a different resampling scheme again, with its own targets.
    </p>
  </Figure>;
}

/* ================================================================== *
 * F6 · §8 — expected loss and expected improvement
 * ================================================================== */
const INCUMBENT = 0.2;
const CANDIDATE_A = [{ label: 'certain', loss: 0.18, probability: 1 }];
const CANDIDATE_B = [{ label: 'better half', loss: 0.05, probability: 0.5 }, { label: 'worse half', loss: 0.45, probability: 0.5 }];

export function ImprovementFigure() {
  const a = expectedImprovement(INCUMBENT, CANDIDATE_A);
  const b = expectedImprovement(INCUMBENT, CANDIDATE_B);
  const place = value => 44 + (value / 0.5) * 286;
  const mass = probability => 132 - probability * 90;
  return <Figure id="improvement" caption="Figure 6 — Expected loss and expected improvement are different summaries of the same believed distribution. Lower loss is better, so improvement runs leftward from the incumbent.">
    <svg viewBox="0 0 360 176" role="img" aria-label={`A loss axis from 0 to 0.5 with the incumbent at 0.20. Candidate A has one probability mass of 1 at loss 0.18. Candidate B has two masses of one half at 0.05 and 0.45. Only outcomes left of the incumbent improve on it. Expected improvement is ${fixed(a.expectedImprovement, 3)} for A and ${fixed(b.expectedImprovement, 3)} for B, while their mean losses are ${fixed(a.meanLoss, 2)} and ${fixed(b.meanLoss, 2)}.`}>
      <rect x={place(0)} y="30" width={place(INCUMBENT) - place(0)} height="102" fill="#182220" />
      <line className="cv-axis" x1={place(0)} y1="132" x2={place(0.5)} y2="132" />
      {[0, 0.1, 0.2, 0.3, 0.4, 0.5].map(value => <g key={value}>
        <line className="cv-axis" x1={place(value)} y1="132" x2={place(value)} y2="138" />
        <text x={place(value)} y="152" textAnchor="middle">{value.toFixed(1)}</text>
      </g>)}
      <text x="180" y="170" textAnchor="middle">believed loss · lower is better</text>
      <line className="cv-flow is-final" x1={place(INCUMBENT)} y1="30" x2={place(INCUMBENT)} y2="132" />
      <text x={place(INCUMBENT) + 5} y="44">incumbent .20</text>
      <line className="cv-stem" x1={place(0.18)} y1="132" x2={place(0.18)} y2={mass(1)} />
      <circle className="cv-mark" cx={place(0.18)} cy={mass(1)} r="4" />
      <text x={place(0.18) - 8} y={mass(1) - 8} textAnchor="end">A: p=1</text>
      {CANDIDATE_B.map(outcome => <g key={outcome.loss}>
        <line className="cv-stem" x1={place(outcome.loss)} y1="132" x2={place(outcome.loss)} y2={mass(0.5)} strokeDasharray="4 3" />
        <circle className="cv-mark is-hollow" cx={place(outcome.loss)} cy={mass(0.5)} r="4" />
        <text x={place(outcome.loss)} y={mass(0.5) - 10} textAnchor="middle">B: p=½</text>
      </g>)}
    </svg>
    <Legend kinds={[]} extra={<>
      <span>Solid stem and filled dot: candidate A. Dashed stem and hollow dot: candidate B. Stem height is the believed probability of that
        outcome. The shaded band left of the incumbent line is the improvement region: only outcomes inside it contribute.</span>
    </>} />
    <Table id="improvement-table" caption="The exact calculation. Only outcomes below the incumbent contribute."
      headings={['candidate', 'believed loss', 'probability', 'positive improvement', 'weighted contribution']} numeric={[1, 2, 3, 4]}
      rows={[
        ...a.rows.map(row => ['A', fixed(row.loss, 2), fixed(row.probability, 2), fixed(row.improvement, 3), fixed(row.contribution, 3)]),
        ...b.rows.map(row => ['B', fixed(row.loss, 2), fixed(row.probability, 2), fixed(row.improvement, 3), fixed(row.contribution, 3)]),
      ]} />
    <div className="cv-readout">
      <dl>
        <dt>Mean believed loss, A</dt><dd>{fixed(a.meanLoss, 3)}</dd>
        <dt>Mean believed loss, B</dt><dd>{fixed(b.meanLoss, 3)}</dd>
        <dt>Expected improvement, A</dt><dd>{fixed(a.expectedImprovement, 3)}</dd>
        <dt>Expected improvement, B</dt><dd>{fixed(b.expectedImprovement, 3)}</dd>
      </dl>
    </div>
    <p>
      B has the worse mean and the larger expected improvement. A large possible gain can justify an uncertain trial even when the average
      prediction is worse. These are constructed beliefs stated to make the acquisition rule concrete, not a fitted posterior: no smooth
      Gaussian was ever estimated here.
    </p>
    <h4>What the tree-structured estimator models instead</h4>
    <svg viewBox="0 0 360 138" role="img" aria-label="The observed trial history splits into a better-loss group and the remaining group. A density l of lambda is fitted to the better group and a density g of lambda to the rest. The acquisition value depends on their ratio l over g, not on one Gaussian process fitted to the loss.">
      <Box x={80} y={8} width={200} height={26} label="observed trial history" />
      <Arrow from={[150, 34]} to={[92, 56]} />
      <Arrow from={[210, 34]} to={[268, 56]} />
      <Box x={12} y={56} width={160} height={26} label="better-loss group" kind="is-train" />
      <Box x={188} y={56} width={160} height={26} label="the rest" kind="is-held" />
      <text x="92" y="96" textAnchor="middle">density l(λ)</text>
      <text x="268" y="96" textAnchor="middle">density g(λ)</text>
      <Arrow from={[92, 104]} to={[160, 120]} className="cv-flow is-score" />
      <Arrow from={[268, 104]} to={[200, 120]} className="cv-flow is-score" />
      <text x="180" y="134" textAnchor="middle">seek a high l/g ratio</text>
    </svg>
    <Table id="tpe-ratio" caption="The acquisition value the original construction is proportional to, at γ = 0.25. Only the ratio l/g enters it; these rows are the formula evaluated, not a fitted density."
      headings={['l(λ) / g(λ)', 'acquisition value']} numeric={[1]}
      rows={[0.25, 1, 4, 16].map(ratio => [
        fixed(ratio, 2),
        fixed(tpeAcquisition({ gamma: 0.25, betterDensity: ratio, otherDensity: 1 }), 4),
      ])} />
    <p className="cv-caption">
      The densities are symbolic. No density was fitted for this diagram, the value is bounded above by 1/γ = 4, and the original construction
      is not the same object as one Gaussian process over the loss.
    </p>
  </Figure>;
}
