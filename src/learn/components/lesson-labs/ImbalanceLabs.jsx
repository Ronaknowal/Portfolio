import { useState } from 'react';
import {
  actionRisks, averagePrecisionOf, averagePrecisionOfScores, costOf, countsAt, fixtures,
  inverseWeightedOptimum, limits, precisionRecallPoints, queueAt, queueComparison,
  smoteConstruction, thresholdLadder, thresholdSweep, weightedLossCurve, weightedOptimum,
} from '../../data/imbalance-models';
import { inspectionRecords, methods, roles, study, tuningRecords } from '../../data/imbalance-data';
import {
  CountStrip, Investigation, NumberField, Plot, LiveResult, Select, SliderField, Table,
  asFraction, curve, polyline, round, steps, Undefined, useInvestigation,
} from './ImbalanceShared.jsx';
import './imbalance-labs.css';

const letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'];

/** Put an in-plot annotation in whichever corner of the drawing box is furthest
 * from every drawn line.
 *
 * A fixed position cannot work here, because the shape of both plots changes
 * with the learner's own inputs: with a false-alarm cost of 1 against a
 * missed-positive cost of 12, the selecting line runs along the bottom of the
 * box for its whole width, so "just above the axis" is the worst place for a
 * label, while with the costs reversed it is the best. Each obstacle is a
 * function from screen x to screen y; the corner whose label span stays
 * furthest from all of them wins.
 */
const plotBox = (width, height, padding) => ({
  left: padding.left, right: width - padding.right, top: padding.top, bottom: height - padding.bottom,
});
function clearCorner(box, obstacles, labelWidth = 92) {
  const candidates = [
    { x: box.left + 6, y: box.top + 14, anchor: 'start' },
    { x: box.right - 6, y: box.top + 14, anchor: 'end' },
    { x: box.left + 6, y: box.bottom - 8, anchor: 'start' },
    { x: box.right - 6, y: box.bottom - 8, anchor: 'end' },
  ];
  let best = candidates[0];
  let bestClearance = -Infinity;
  for (const candidate of candidates) {
    const far = candidate.anchor === 'start' ? candidate.x + labelWidth : candidate.x - labelWidth;
    let clearance = Infinity;
    for (const obstacle of obstacles) {
      for (let step = 0; step <= 8; step += 1) {
        const probe = candidate.x + (far - candidate.x) * step / 8;
        const value = obstacle(probe);
        if (Number.isFinite(value)) clearance = Math.min(clearance, Math.abs(value - candidate.y));
      }
    }
    if (clearance > bestClearance) { bestClearance = clearance; best = candidate; }
  }
  return { ...best, clearance: bestClearance };
}
/** A straight line through two screen points, as a function of screen x. */
const screenLine = ([x1, y1], [x2, y2]) => x =>
  (Math.abs(x2 - x1) < 1e-9 ? y1 : y1 + (y2 - y1) * (x - x1) / (x2 - x1));

/* ============================================================== §2 · I1 */

const queueBaseline = {
  records: fixtures.queue,
  start: fixtures.queueStartThreshold,
  target: fixtures.queueTargetThreshold,
};
const queueSetups = [
  { key: 'default', label: 'The counterexample: A .9/0, B .8/1, C .7/1', ...queueBaseline },
  {
    key: 'corrected', label: 'Correct A’s label to 1 — every record is now positive',
    records: fixtures.queue.map(row => (row.id === 'A' ? { ...row, truth: 1 } : row)),
    start: 0.7, target: 0.9,
  },
  {
    key: 'tied', label: 'Setup for the tie null: tie B and C at .8, and gate at .8',
    records: [
      { id: 'A', score: 0.9, truth: 0 }, { id: 'B', score: 0.8, truth: 1 }, { id: 'C', score: 0.8, truth: 1 },
    ],
    start: 0.7, target: 0.8,
  },
  {
    key: 'reordered', label: 'Null: the same tied records, displayed in the other order',
    records: [
      { id: 'A', score: 0.9, truth: 0 }, { id: 'C', score: 0.8, truth: 1 }, { id: 'B', score: 0.8, truth: 1 },
    ],
    start: 0.7, target: 0.8,
  },
  {
    key: 'practice', label: 'Practice 2: scores .95, .8, .8, .4 with truths 0, 1, 0, 1',
    records: fixtures.tiedQueue, start: 0.4, target: 0.8,
  },
  {
    key: 'ap', label: 'Average precision: descending truths 1, 0, 1, 0',
    records: fixtures.averagePrecisionQueue, start: 0.6, target: 0.9,
  },
  {
    key: 'empty', label: 'Above every score: select nothing',
    records: fixtures.queue, start: 0.7, target: 1,
  },
];

function BinnedRail({ point, revealed, order }) {
  const bin = id => (['tp', 'fp', 'fn', 'tn'].find(key => point.bins[key].includes(id)) ?? '');
  const names = { tp: 'selected, actually positive', fp: 'selected, actually negative', fn: 'not selected, actually positive', tn: 'not selected, actually negative' };
  /* Before the reveal the cards keep their stable record order. Listing the
     selected ones first would encode the score-versus-gate partition on a rail
     whose every card still says "awaiting the gate". */
  return <div className="imb-rail">
    {(revealed ? point.selectedIds.concat(point.unselectedIds) : order).map(id => {
      const key = bin(id);
      return <span key={id} className={`imb-card${revealed ? ` is-${key}` : ''}${point.selectedIds.includes(id) && revealed ? ' is-selected' : ''}`}>
        <b>{id}</b>
        {revealed
          ? <span className="imb-bin">{key.toUpperCase()} — {names[key]}</span>
          : <span className="imb-bin">awaiting the gate</span>}
      </span>;
    })}
  </div>;
}

export function ScoreQueueLab() {
  const state = useInvestigation(queueBaseline);
  const draft = state.draft;
  const target = queueAt(draft.records, draft.target);
  const applied = queueAt(state.active.records, state.active.target);
  const ladder = thresholdLadder(state.active.records);
  const ap = averagePrecisionOf(state.active.records);
  const revealed = Boolean(state.result);
  // The recall sentence and comparison must describe the same applied start/target gates.
  const comparison = revealed ? queueComparison(state.active.records, state.active.start, state.active.target) : null;
  const calculateInputs = proposed => {
    const result = queueComparison(proposed.records, proposed.start, proposed.target);
    // Precision at the new gate can be genuinely undefined — that is the state
    // the "above every score" setup exists to reach. It is passed through as
    // null rather than as NaN, and the verdict says there is no number to
    // compare a guess against instead of printing a dash and a tolerance.
    return { outcome: result.precision, value: result.after.precision };
  };
  const editRecord = (index, update) => state.edit({
    records: draft.records.map((row, position) => (position === index ? { ...row, ...update } : row)),
  });
  const prPoints = precisionRecallPoints(state.active.records);
  return <Investigation
    title="Move the gate through actual scored records"
    question={`A fixed ranking of named records, each with a score and a known truth. The gate selects every record whose score is at least the threshold, so it always takes a tied group whole. Starting from the gate at ${round(draft.start, 2)}, observe what happens to precision when only scores of at least ${round(draft.target, 2)} are selected and watch the counts update.`}
    note="The selected records and metrics update as you move the controls. Raising the gate can only shrink the selected set, so recall cannot rise; precision has no such guarantee, because the records that leave may be positive or negative."
    onReset={state.reset}>

    <fieldset className="imb-row-group" style={{ '--imb-row-columns': 2 }}>
      <legend>The records. Edit any score or truth; add or remove rows.</legend>
      {draft.records.map((row, index) => <div key={row.id} style={{ display: 'contents' }}>
        <NumberField label={`${row.id} — score`} value={row.score} min={limits.score.minimum} max={limits.score.maximum}
          step="0.01" decimals={2} onChange={score => editRecord(index, { score })} />
        <Select label={`${row.id} — actual truth`} value={String(row.truth)}
          options={[['0', '0 — actually negative'], ['1', '1 — actually positive']]}
          onChange={value => editRecord(index, { truth: Number(value) })} />
      </div>)}
    </fieldset>
    <div className="imb-row-actions">
      <button type="button" disabled={draft.records.length >= limits.maximumRecords}
        onClick={() => state.edit({
          records: [...draft.records, {
            id: letters[draft.records.length], score: 0.5, truth: 0,
          }],
        })}>Add a record</button>
      <button type="button" disabled={draft.records.length <= limits.minimumRecords}
        onClick={() => state.edit({ records: draft.records.slice(0, -1) })}>Remove the last record</button>
      <span className="imb-caption">
        Between {limits.minimumRecords} and {limits.maximumRecords} records; scores in [{limits.score.minimum}, {limits.score.maximum}].
      </span>
    </div>
    <div className="imb-controls">
      <SliderField label="Gate you are starting from" value={draft.start} min={0} max={1} step={0.01} decimals={2}
        onChange={start => state.edit({ start })} />
      <SliderField label="Gate you are moving to" value={draft.target} min={0} max={1} step={0.01} decimals={2}
        onChange={value => state.edit({ target: value })} />
    </div>
    <div className="imb-buttons">
      {queueSetups.map(setup => <button key={setup.key} type="button"
        onClick={() => state.suggest({ records: setup.records, start: setup.start, target: setup.target })}>
        {setup.label}
      </button>)}
    </div>
    <p className="imb-state-strip">
      <span>applied gate: <b>≥ {round(state.active.target, 2)}</b></span>
      <span className={state.pending ? 'is-draft' : undefined}>draft gate: <b>≥ {round(draft.target, 2)}</b></span>
      <span>records: <b>{draft.records.map(row => `${row.id}(${round(row.score, 2)}, ${row.truth})`).join(' ')}</b></span>
    </p>
    <BinnedRail point={target} revealed={revealed} order={draft.records.map(row => row.id)} />

    <LiveResult
      
      
      state={state} calculateInputs={calculateInputs}
      
      
      describe={comparison
        ? `Recall moved ${comparison.recall}, from ${comparison.before.recall === null ? 'undefined' : round(comparison.before.recall, 6)} to ${comparison.after.recall === null ? 'undefined' : round(comparison.after.recall, 6)}.`
        : undefined}
       />

    {revealed && <>
      <CountStrip counts={applied} />
      <p className="imb-caption">
        Selected: {applied.selectedIds.length === 0 ? 'nothing at all' : applied.selectedIds.join(', ')}.
        {' '}Not selected: {applied.unselectedIds.length === 0 ? 'nothing' : applied.unselectedIds.join(', ')}.
        {applied.precision === null
          ? <> Precision here is <Undefined because="TP + FP is zero" />, not zero: a ratio with an empty denominator has no value to report.</>
          : <> Precision is {applied.tp}/{applied.alerts}{asFraction(applied.precision) ? ` = ${asFraction(applied.precision)}` : ''}.</>}
      </p>
      <Table caption="Every operating point these records actually have: each distinct score, plus the policy above all of them"
        headings={['gate', 'selected IDs', 'TP', 'FP', 'FN', 'TN', 'precision', 'recall']}
        rows={ladder.map(point => [
          point.isNoAlert ? 'above every score' : `≥ ${round(point.threshold, 2)}`,
          point.selectedIds.join(' ') || '—',
          String(point.tp), String(point.fp), String(point.fn), String(point.tn),
          point.precision === null ? 'undefined (TP+FP = 0)' : (asFraction(point.precision) ?? round(point.precision, 6)),
          point.recall === null ? 'undefined (TP+FN = 0)' : (asFraction(point.recall) ?? round(point.recall, 6)),
        ])}
        rowClass={index => (ladder[index].threshold === state.active.target ? 'is-chosen' : undefined)}
        footnote="There is no operating point between two observed scores, so the chart below draws steps and never a smooth line." />
      {prPoints.length > 1 && <Plot caption="Precision and recall at each distinct score, drawn as steps"
        describe={`A step chart. ${prPoints.map(point => `At the gate ${round(point.threshold, 2)}, recall is ${point.recall === null ? 'undefined' : round(point.recall, 4)} and precision is ${point.precision === null ? 'undefined' : round(point.precision, 4)}.`).join(' ')}`}
        domain={[0, 1]} range={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} valueTicks={[0, 0.5, 1]} height={200}
        formatTick={value => round(value, 2)}>
        {(scaleX, scaleY) => <>
          <polyline className="imb-step" points={steps(
            [...prPoints].reverse().filter(point => point.precision !== null).map(point => [point.threshold, point.precision]),
            scaleX, scaleY)} />
          {prPoints.some(point => point.recall !== null) && <polyline className="imb-step is-recall" points={steps(
            [...prPoints].reverse().filter(point => point.recall !== null).map(point => [point.threshold, point.recall]), scaleX, scaleY)} />}
          {prPoints.map(point => <g key={point.threshold}>
            {point.precision !== null && <circle className="imb-mark" cx={scaleX(point.threshold)} cy={scaleY(point.precision)} r="3.5" />}
            {point.recall !== null && <circle className="imb-mark is-hollow" cx={scaleX(point.threshold)} cy={scaleY(point.recall)} r="3.5" />}
          </g>)}
          <text className="imb-small imb-muted" x={48} y={14}>gate on the horizontal axis</text>
        </>}
      </Plot>}
      {prPoints.every(point => point.recall === null) && <p className="imb-caption">There are no positive labels, so recall is undefined and has no line or markers.</p>}
      <p className="imb-legend">
        <span><svg viewBox="0 0 34 12" aria-hidden="true"><line className="imb-step" x1="1" x2="33" y1="6" y2="6" /></svg> precision, filled marks</span>
        <span><svg viewBox="0 0 34 12" aria-hidden="true"><line className="imb-step is-recall" x1="1" x2="33" y1="6" y2="6" /></svg> recall, hollow marks</span>
      </p>
      {ap.value === null
        ? <p className="imb-caption">
          Average precision is <Undefined because={ap.undefinedBecause} /> for these records. A library may report a
          number here by a declared convention; this view reports the undefined state instead.
        </p>
        : <Table caption="Average precision: a recall increment at each distinct score group, weighted by the precision there"
          headings={['score group', 'records in the group', 'recall after', 'recall increment', 'precision', 'contribution']}
          rows={ap.steps.map(entry => [
            round(entry.threshold, 2), entry.group.join(' '), round(entry.recall, 6),
            round(entry.increment, 6), round(entry.precision, 6), round(entry.contribution, 6),
          ])}
          footnote={`Average precision = ${asFraction(ap.value) ?? round(ap.value, 6)}. Tied records form one group; no order is invented between them, and no label is consulted to break a tie.`} />}
    </>}
  </Investigation>;
}

/* ============================================================== §3 · I2 */

const costBaseline = {
  posterior: fixtures.casePosterior,
  costFP: fixtures.costs.costFP,
  costFN: fixtures.costs.costFN,
};
const costSetups = [
  { key: 'default', label: `Declared setup: p = ${fixtures.casePosterior}, costs ${fixtures.costs.costFP} and ${fixtures.costs.costFN}`, ...costBaseline },
  { key: 'contrast', label: `Contrast: lower the posterior to ${fixtures.contrastPosterior}`, posterior: fixtures.contrastPosterior, costFP: 1, costFN: 12 },
  { key: 'null', label: 'Null: multiply both costs by 3, holding p = .1', posterior: 0.1, costFP: 3, costFN: 36 },
  { key: 'practice', label: 'Practice 3: costs 2 and 7 at p = .2', posterior: 0.2, costFP: 2, costFN: 7 },
  { key: 'equal', label: 'Equal costs at a rare p = .01', posterior: 0.01, costFP: 4, costFN: 4 },
  { key: 'nofp', label: 'Corner: a false alarm costs nothing', posterior: 0.3, costFP: 0, costFN: 5 },
  { key: 'nofn', label: 'Corner: a missed positive costs nothing', posterior: 0.3, costFP: 5, costFN: 0 },
  { key: 'both', label: 'Corner: both costs zero', posterior: 0.3, costFP: 0, costFN: 0 },
];

export function CostCrossingLab() {
  const state = useInvestigation(costBaseline);
  const draft = state.draft;
  const applied = actionRisks(state.active.posterior, state.active.costFP, state.active.costFN);
  const revealed = Boolean(state.result);
  const calculateInputs = proposed => {
    const result = actionRisks(proposed.posterior, proposed.costFP, proposed.costFN);
    return { outcome: result.tie ? 'tie' : result.action, value: result.selectRisk };
  };
  const ceiling = Math.max(applied.lines.ceiling, 1e-9);
  return <Investigation
    title="Two action costs cross"
    question="A case with a stated posterior, a stated cost for a false alarm and a stated cost for a missed positive. Correct decisions cost nothing. Decide which action has the lower expected cost before the two lines are drawn."
    note="These are hypothetical units per mistake, not laboratory prices and not medical guidance. The declared convention is to select when the two expected costs are exactly equal; at that point either action has the same expected cost."
    onReset={state.reset}>

    <div className="imb-controls is-wide">
      <SliderField label="Posterior p for this case" value={draft.posterior} min={0} max={1} step={0.01} decimals={2}
        onChange={posterior => state.edit({ posterior })} />
      <SliderField label="Cost of a false alarm, c_FP" value={draft.costFP} min={limits.cost.minimum} max={limits.cost.maximum}
        step={0.1} decimals={1} suffix={`${round(draft.costFP, 1)} units`} onChange={costFP => state.edit({ costFP })} />
      <SliderField label="Cost of a missed positive, c_FN" value={draft.costFN} min={limits.cost.minimum} max={limits.cost.maximum}
        step={0.1} decimals={1} suffix={`${round(draft.costFN, 1)} units`} onChange={costFN => state.edit({ costFN })} />
    </div>
    <div className="imb-buttons">
      {costSetups.map(setup => <button key={setup.key} type="button" onClick={() => state.suggest({
        posterior: setup.posterior, costFP: setup.costFP, costFN: setup.costFN,
      })}>{setup.label}</button>)}
    </div>
    <p className="imb-state-strip">
      <span>applied: p <b>{round(state.active.posterior, 2)}</b>, c_FP <b>{round(state.active.costFP, 1)}</b>, c_FN <b>{round(state.active.costFN, 1)}</b></span>
      <span className={state.pending ? 'is-draft' : undefined}>draft: p <b>{round(draft.posterior, 2)}</b>, c_FP <b>{round(draft.costFP, 1)}</b>, c_FN <b>{round(draft.costFN, 1)}</b></span>
    </p>

    <LiveResult
      
      
      state={state} calculateInputs={calculateInputs}
      
      
      describe={revealed
        ? `Selecting costs ${round(applied.selectRisk, 6)}; skipping costs ${round(applied.skipRisk, 6)}.`
        : undefined}
       />

    {revealed && <>
      <p className="imb-counts">
        <span>R(select) = (1 − p)·c_FP = <b>{round(applied.selectRisk, 6)}</b></span>
        <span>R(skip) = p·c_FN = <b>{round(applied.skipRisk, 6)}</b></span>
        {applied.cutoff === null
          ? <span className="is-undefined">cutoff <b>undefined</b> — both costs are zero</span>
          : <span>cutoff c_FP/(c_FP+c_FN) = <b>{asFraction(applied.cutoff, 60) ?? round(applied.cutoff, 6)}</b> ≈ {round(applied.cutoff, 6)}</span>}
        <span>action <b>{applied.action}</b>{applied.tie ? ' (by the equal-cost convention)' : ''}</span>
      </p>
      {applied.undefinedBecause && <p className="imb-note">{applied.undefinedBecause}.</p>}
      <Plot caption={`Both expected costs against the posterior, on one shared vertical scale up to ${round(ceiling, 2)} units`}
        describe={`Two lines on a posterior axis from 0 to 1. The selecting line falls from ${round(applied.costFP, 2)} at p = 0 to 0 at p = 1. The skipping line rises from 0 at p = 0 to ${round(applied.costFN, 2)} at p = 1. ${applied.cutoff === null ? 'With both costs zero the two lines coincide along the axis and there is no crossing.' : `They cross at p = ${round(applied.cutoff, 6)}.`} The current case sits at p = ${round(applied.posterior, 2)}, where selecting costs ${round(applied.selectRisk, 6)} and skipping costs ${round(applied.skipRisk, 6)}.`}
        domain={[0, 1]} range={[0, ceiling]} ticks={[0, 0.25, 0.5, 0.75, 1]}
        valueTicks={[0, ceiling / 2, ceiling]} height={200} formatValue={value => round(value, 2)}>
        {(scaleX, scaleY) => <>
          <polyline className="imb-curve is-select" points={polyline(applied.lines.select, scaleX, scaleY)} />
          <polyline className="imb-curve is-skip" points={polyline(applied.lines.skip, scaleX, scaleY)} />
          {applied.cutoff !== null && <g>
            <line className="imb-grid" x1={scaleX(applied.cutoff)} x2={scaleX(applied.cutoff)}
              y1={scaleY(0)} y2={scaleY(ceiling)} />
            <circle className="imb-mark is-hollow" cx={scaleX(applied.cutoff)}
              cy={scaleY(applied.costFP * applied.costFN / (applied.costFP + applied.costFN))} r="5" />
            {/* Just above the axis at the crossing, where neither line runs:
                both risks equal c_FP·c_FN/(c_FP+c_FN) there, which is well above
                zero whenever a crossing exists at all. A label placed on the
                lines themselves was travelled along by the other one. */}
            {(() => {
              const spot = clearCorner(plotBox(340, 200, { left: 48, right: 14, top: 16, bottom: 34 }), [
                screenLine([scaleX(0), scaleY(applied.costFP)], [scaleX(1), scaleY(0)]),
                screenLine([scaleX(0), scaleY(0)], [scaleX(1), scaleY(applied.costFN)]),
              ]);
              return <text className="imb-halo" x={spot.x} y={spot.y} textAnchor={spot.anchor}>
                cross at {round(applied.cutoff, 4)}
              </text>;
            })()}
          </g>}
          {/* The two case markers carry no text: both risks are printed exactly
              in the strip above, and any label placed at the case posterior sits
              where the two lines are closest to each other. */}
          <circle className="imb-mark" cx={scaleX(applied.posterior)} cy={scaleY(applied.selectRisk)} r="4.5" />
          <circle className="imb-mark is-hollow" cx={scaleX(applied.posterior)} cy={scaleY(applied.skipRisk)} r="4.5" />
        </>}
      </Plot>
      <p className="imb-legend">
        <span><svg viewBox="0 0 34 12" aria-hidden="true"><line className="imb-curve is-select" x1="1" x2="33" y1="6" y2="6" /></svg> selecting, filled mark</span>
        <span><svg viewBox="0 0 34 12" aria-hidden="true"><line className="imb-curve is-skip" x1="1" x2="33" y1="6" y2="6" /></svg> skipping, hollow mark</span>
      </p>
      <p className="imb-caption">
        Multiplying both costs by the same positive factor stretches this vertical axis and leaves the crossing exactly
        where it is. The axis maximum is printed in the caption for that reason: the shape of the picture is not the
        claim, the numbers are.
      </p>
    </>}
  </Investigation>;
}

/* ============================================================== §4 · I3 */

const weightedBaseline = {
  mode: 'forward',
  probability: fixtures.weighted.probability,
  weightedScore: 0.5,
  positiveWeight: fixtures.weighted.positiveWeight,
  negativeWeight: fixtures.weighted.negativeWeight,
};
const weightedSetups = [
  { key: 'default', label: 'Declared setup: p = .1 with weights 9 and 1', ...weightedBaseline },
  { key: 'equal', label: 'Contrast: the same p with equal weights 1 and 1', ...weightedBaseline, positiveWeight: 1, negativeWeight: 1 },
  { key: 'doubled', label: 'Null: double both weights to 18 and 2', ...weightedBaseline, positiveWeight: 18, negativeWeight: 2 },
  { key: 'practice', label: 'Practice 4: p = .2 with weights 4 and 1', ...weightedBaseline, probability: 0.2, positiveWeight: 4, negativeWeight: 1 },
  { key: 'inverse', label: 'Inverse mode: recover p from a weighted score of .5', ...weightedBaseline, mode: 'inverse' },
];

export function WeightedScoreLab() {
  const state = useInvestigation(weightedBaseline);
  const draft = state.draft;
  const forward = state.active.mode === 'forward';
  const populationFor = settings => settings.mode === 'forward' ? settings.probability
    : inverseWeightedOptimum(settings.weightedScore, settings.positiveWeight, settings.negativeWeight);
  const recovered = populationFor(state.active);
  const applied = weightedOptimum(recovered, state.active.positiveWeight, state.active.negativeWeight);
  const appliedCurve = weightedLossCurve(recovered, state.active.positiveWeight, state.active.negativeWeight);
  const revealed = Boolean(state.result);
  const previousOptimum = revealed
    ? weightedOptimum(populationFor(state.previous), state.previous.positiveWeight, state.previous.negativeWeight)
    : null;
  const calculateInputs = proposed => {
    if (proposed.mode === 'inverse') {
      const value = inverseWeightedOptimum(proposed.weightedScore, proposed.positiveWeight, proposed.negativeWeight);
      return { outcome: value < 0.5 - 1e-12 ? 'below' : value > 0.5 + 1e-12 ? 'above' : 'equal', value };
    }
    const result = weightedOptimum(proposed.probability, proposed.positiveWeight, proposed.negativeWeight);
    return {
      outcome: result.optimum < 0.5 - 1e-12 ? 'below' : result.optimum > 0.5 + 1e-12 ? 'above' : 'equal',
      value: result.optimum,
    };
  };
  const domain = [0, 1];
  const lossValues = appliedCurve.points.map(([, value]) => value);
  // Keep the analytic minimum on the drawn curve, even near a boundary where
  // a uniform grid can step over the entire low-loss region.
  const drawingScores = [...new Set([0, 1, applied.optimum, applied.optimum / 2,
    (1 + applied.optimum) / 2, ...appliedCurve.points.map(([q]) => q)])].sort((a, b) => a - b);
  const ceiling = Math.min(Math.max(...lossValues), applied.lossAtOptimum + Math.max(2, 4 * applied.lossAtOptimum));
  return <Investigation
    title="A score of one half can mean another probability"
    question={forward
      ? 'A population probability and two class weights. Before the loss curve appears, say where the weighted-loss optimum will sit relative to one half — and, if you like, name it exactly.'
      : 'A weighted-loss optimum and the two weights that produced it. Recover the original population probability before the mapping is shown.'}
    note="This is the unnormalised population expected loss, which is a different object from the finite-sample normalised objective of figure 3. Nothing here claims a classifier has been calibrated: the result follows from the two weighted probability masses and nothing else."
    onReset={state.reset}>

    <div className="imb-controls is-wide">
      <Select label="Mode" value={draft.mode}
        options={[['forward', 'Forward — find the weighted optimum q*'], ['inverse', 'Inverse — recover p from q*']]}
        onChange={mode => state.edit({ mode })} />
      {draft.mode === 'forward'
        ? <SliderField label="Population probability p" value={draft.probability}
          min={limits.probability.minimum} max={limits.probability.maximum} step={0.01} decimals={2}
          onChange={probability => state.edit({ probability })} />
        : <SliderField label="Weighted-loss optimum q*" value={draft.weightedScore}
          min={limits.weightedScore.minimum} max={limits.weightedScore.maximum} step={0.01} decimals={2}
          onChange={weightedScore => state.edit({ weightedScore })} />}
      <SliderField label="Positive class weight w₊" value={draft.positiveWeight}
        min={limits.weight.minimum} max={limits.weight.maximum} step={0.1} decimals={1}
        onChange={positiveWeight => state.edit({ positiveWeight })} />
      <SliderField label="Negative class weight w₋" value={draft.negativeWeight}
        min={limits.weight.minimum} max={limits.weight.maximum} step={0.1} decimals={1}
        onChange={negativeWeight => state.edit({ negativeWeight })} />
    </div>
    <div className="imb-buttons">
      {weightedSetups.map(setup => <button key={setup.key} type="button" onClick={() => state.suggest({
        mode: setup.mode, probability: setup.probability, weightedScore: setup.weightedScore,
        positiveWeight: setup.positiveWeight, negativeWeight: setup.negativeWeight,
      })}>{setup.label}</button>)}
    </div>
    <p className="imb-state-strip">
      <span>applied: {state.active.mode}, {state.active.mode === 'forward' ? <>p <b>{round(state.active.probability, 2)}</b></> : <>q* <b>{round(state.active.weightedScore, 2)}</b></>}, w₊ <b>{round(state.active.positiveWeight, 1)}</b>, w₋ <b>{round(state.active.negativeWeight, 1)}</b></span>
      <span className={state.pending ? 'is-draft' : undefined}>draft: {draft.mode}, {draft.mode === 'forward' ? <>p <b>{round(draft.probability, 2)}</b></> : <>q* <b>{round(draft.weightedScore, 2)}</b></>}, w₊ <b>{round(draft.positiveWeight, 1)}</b>, w₋ <b>{round(draft.negativeWeight, 1)}</b></span>
    </p>

    <LiveResult
      
      
      state={state} calculateInputs={calculateInputs}
      
      
      
       />

    {revealed && <>
      <div className="imb-panels is-pair">
        <div className="imb-panel">
          <h4>The two nodes, and the map between them</h4>
          {forward ? <>
            <p>
              Original population probability <strong>p = {round(applied.probability, 6)}</strong>. The two weighted
              masses are w₊p = {round(applied.massPositive, 6)} and w₋(1−p) = {round(applied.massNegative, 6)}.
            </p>
            <p>
              Weighted-loss optimum <strong>q* = {round(applied.optimum, 6)}</strong>
              {asFraction(applied.optimum) ? ` = ${asFraction(applied.optimum)}` : ''}. Inverting returns
              {' '}p = {round(inverseWeightedOptimum(
                applied.optimum,
                applied.positiveWeight, applied.negativeWeight), 6)}.
            </p>
            <p>
              Thresholding this ideal weighted score at one half is the same decision as thresholding p at
              {' '}w₋/(w₊+w₋) = {round(applied.equivalentProbabilityCutoff, 6)}. It is not a claim that the event has
              probability q* in the original population.
            </p>
          </> : <>
            <p>
              Weighted-loss optimum <strong>q* = {round(state.active.weightedScore, 6)}</strong>, with weights
              {' '}{round(state.active.positiveWeight, 1)} and {round(state.active.negativeWeight, 1)}.
            </p>
            <p>
              Recovered population probability <strong>p = w₋q*/[w₊(1−q*) + w₋q*] = {round(recovered, 6)}</strong>
              {asFraction(recovered) ? ` = ${asFraction(recovered)}` : ''}.
            </p>
            <p>
              Only the ratio of the two weights enters. The same q* with both weights doubled recovers the same p.
            </p>
          </>}
          {previousOptimum && <p className="imb-caption">
            Previous applied setting: p {round(previousOptimum.probability, 4)}, weights
            {' '}{round(previousOptimum.positiveWeight, 1)} and {round(previousOptimum.negativeWeight, 1)}, optimum
            {' '}{round(previousOptimum.optimum, 6)}, loss at that optimum {round(previousOptimum.lossAtOptimum, 6)}.
            The current loss at its optimum is {round(applied.lossAtOptimum, 6)}. Comparing those two printed numbers
            is the reliable way to see a doubling of both weights: it raises the curve without moving its minimiser.
          </p>}
        </div>
        <div className="imb-panel">
          <Plot caption="Unnormalised expected weighted loss against the score q"
            describe={`A loss curve over q from ${round(domain[0], 3)} to ${round(domain[1], 3)}, with the divergent endpoint limits clipped at the displayed loss ceiling. Its minimum is at q = ${round(applied.optimum, 6)}, where the loss is ${round(applied.lossAtOptimum, 6)}. The loss rises without bound towards both ends.`}
            domain={domain} range={[0, ceiling]} ticks={[0, 0.25, 0.5, 0.75, 1]}
            valueTicks={[0, ceiling / 2, ceiling]} height={200} formatValue={value => round(value, 2)}>
            {(scaleX, scaleY) => <>
              <polyline className="imb-curve is-loss" points={polyline(
                drawingScores.map(q => [q, Math.min(applied.loss(q), ceiling)]), scaleX, scaleY)} />
              <line className="imb-grid" x1={scaleX(applied.optimum)} x2={scaleX(applied.optimum)}
                y1={scaleY(0)} y2={scaleY(ceiling)} />
              <circle className="imb-mark" cx={scaleX(applied.optimum)} cy={scaleY(Math.min(applied.lossAtOptimum, ceiling))} r="4.5" />
              {/* Just above the axis at the optimum, which is the one place the
                  curve is furthest away; the curve is clipped at the ceiling and
                  runs along the top of the box, so a label there is crossed. */}
              {(() => {
                const toQ = screenX => domain[0] + (domain[1] - domain[0])
                  * (screenX - scaleX(domain[0])) / (scaleX(domain[1]) - scaleX(domain[0]));
                const spot = clearCorner(plotBox(340, 200, { left: 48, right: 14, top: 16, bottom: 34 }), [
                  screenX => scaleY(Math.min(applied.loss(Math.min(Math.max(toQ(screenX), domain[0]), domain[1])), ceiling)),
                ]);
                return <text className="imb-halo" x={spot.x} y={spot.y} textAnchor={spot.anchor}>
                  q* = {round(applied.optimum, 4)}
                </text>;
              })()}
            </>}
          </Plot>
          <p>
            Both ends of this curve rise without bound, so the drawing is clipped at {round(ceiling, 2)} loss units;
            the clipping is a drawing limit, not a property of the loss.
            The exact minimiser comes from the formula, not from reading the drawn grid: the uniform grid's lowest
            point is {round(appliedCurve.sampledMinimum, 6)} and the formula gives {round(appliedCurve.exactMinimum, 6)}, which is also included explicitly in the drawn curve.
            The analytic derivative there is {round(applied.derivativeAtOptimum, 9)} and the second derivative is
            {' '}{round(applied.secondDerivativeAtOptimum, 6)}, which is positive — an interior minimum, not a clipped
            endpoint.
          </p>
        </div>
      </div>
    </>}
  </Investigation>;
}

/* ============================================================== §5 · I4 */

const cloudBaseline = {
  points: fixtures.cloud,
  ...fixtures.cloudSetup,
  question: 'collision',
};
const cloudSetups = [
  { key: 'default', label: 'Declared setup: does the generated point land on a majority point?', ...cloudBaseline },
  {
    key: 'majority', label: 'Exact null: move only the majority point M to (1, 1.5)',
    ...cloudBaseline, question: 'majorityMove',
    points: fixtures.cloud.map(point => (point.id === 'M' ? { ...point, ...fixtures.cloudNullMajority } : point)),
  },
  {
    key: 'scale', label: `Geometric contrast: change the y divisor to ${fixtures.cloudContrastScaleY}`,
    ...cloudBaseline, question: 'neighbour', scaleY: fixtures.cloudContrastScaleY,
  },
  { key: 'endpoint', label: 'Both endpoints: set the fraction to 0, then to 1', ...cloudBaseline, fraction: 0 },
  {
    // Three minority points, as the practice text and its solution both state:
    // the fold then offers at most TWO other neighbours, so k = 5 is refused
    // with "Keep it between 1 and 2". C is placed at distance 5 from the
    // anchor, further than B's √20, so B remains the nearest and the generated
    // point is still exactly (2, 2.5).
    key: 'practice', label: 'Practice 5: three minority points, anchor (1, 2), neighbour (5, 4), fraction .25',
    question: 'collision', scaleX: 1, scaleY: 1, anchorId: 'A', k: 1, neighbourRank: 0, fraction: 0.25,
    points: [
      { id: 'A', cls: 'minority', x: 1, y: 2 },
      { id: 'B', cls: 'minority', x: 5, y: 4 },
      { id: 'C', cls: 'minority', x: 5, y: -1 },
      { id: 'M', cls: 'majority', x: 3, y: 3 },
    ],
  },
];

function Scatter({ construction, revealed }) {
  const width = 340;
  const height = 240;
  const all = [...construction.minority, ...construction.majority];
  const xs = all.map(point => point.x).concat(revealed ? [construction.generated.x] : []);
  const ys = all.map(point => point.y).concat(revealed ? [construction.generated.y] : []);
  const pad = 0.6;
  const spanX = [Math.min(...xs) - pad, Math.max(...xs) + pad];
  const spanY = [Math.min(...ys) - pad, Math.max(...ys) + pad];
  // Equal physical scale on both axes for the default geometry: the metric
  // divisors change the neighbour calculation, never the drawing's aspect.
  const extent = Math.max(spanX[1] - spanX[0], spanY[1] - spanY[0]);
  const originX = (spanX[0] + spanX[1]) / 2 - extent / 2;
  const originY = (spanY[0] + spanY[1]) / 2 - extent / 2;
  const place = value => 44 + (width - 58) * (value - originX) / extent;
  const lift = value => height - 34 - (height - 50) * (value - originY) / extent;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map(fraction => originX + extent * fraction);
  /* Point labels are offset PERPENDICULAR to the drawn segment, on the side the
     point already lies, and never along it. A fixed right-hand offset puts the
     label of any point sitting on the segment — the default cloud's majority
     point M is exactly on it — directly under the line, which is a collision no
     backplate fixes. Each label clears the line by more than its own half
     diagonal, so a point on the line gets the full offset. */
  const ax = place(construction.anchor.x);
  const ay = lift(construction.anchor.y);
  const bx = place(construction.neighbour.x);
  const by = lift(construction.neighbour.y);
  const run = bx - ax;
  const rise = by - ay;
  const span = Math.hypot(run, rise);
  const unit = span < 1e-9 ? { x: 0, y: -1 } : { x: -rise / span, y: run / span };
  const offsetFor = (x, y, distance = 18) => {
    const side = (x - ax) * unit.x + (y - ay) * unit.y;
    const sign = Math.abs(side) < 1e-9 ? 1 : Math.sign(side);
    return { dx: unit.x * distance * sign, dy: unit.y * distance * sign };
  };
  return <svg viewBox={`0 0 ${width} ${height}`} role="img"
    aria-label={`A scatter in the original feature units. ${construction.minority.map(point => `Minority ${point.id} at (${round(point.x, 2)}, ${round(point.y, 2)})`).join('; ')}. ${construction.majority.map(point => `Majority ${point.id} at (${round(point.x, 2)}, ${round(point.y, 2)})`).join('; ')}. ${revealed ? `The chosen segment runs from ${construction.anchor.id} to ${construction.neighbour.id}, and the generated point sits at (${round(construction.generated.x, 4)}, ${round(construction.generated.y, 4)})${construction.collidesWithMajority ? `, exactly on majority point ${construction.collisionIds.filter(id => id !== construction.anchor.id).join(' and ')}` : ''}.` : 'The generated point is available once the inputs form a valid construction.'}`}>
    <title>The minority points, the majority points and the chosen segment</title>
    {ticks.map((value, index) => <g key={value}>
      <line className="imb-grid" x1={place(value)} x2={place(value)} y1={12} y2={height - 34} />
      <text className="imb-small" x={place(value)} y={height - 18} textAnchor="middle">{round(value, 1)}</text>
      {/* The lowest vertical tick sits in the corner where the horizontal tick
          row begins, and the two labels are the same number; drawing both puts
          one on top of the other. The horizontal row keeps it. */}
      {index > 0 && <text className="imb-small" x={40} y={lift(value) + 4} textAnchor="end">{round(value, 1)}</text>}
    </g>)}
    <line className="imb-axis" x1={44} x2={width - 14} y1={height - 34} y2={height - 34} />
    <line className="imb-axis" x1={44} x2={44} y1={12} y2={height - 34} />
    {revealed && <line className="imb-curve is-segment" x1={place(construction.anchor.x)} y1={lift(construction.anchor.y)}
      x2={place(construction.neighbour.x)} y2={lift(construction.neighbour.y)} />}
    {construction.majority.map(point => {
      const x = place(point.x);
      const y = lift(point.y);
      const { dx, dy } = offsetFor(x, y);
      return <g key={point.id}>
        <rect className="imb-mark is-majority" x={x - 5} y={y - 5} width={10} height={10} />
        <text className="imb-halo imb-small" x={x + dx} y={y + dy + 4} textAnchor="middle">{point.id}</text>
      </g>;
    })}
    {construction.minority.map(point => {
      const x = place(point.x);
      const y = lift(point.y);
      const { dx, dy } = offsetFor(x, y);
      return <g key={point.id}>
        <circle className={`imb-mark${point.id === construction.anchor.id && revealed ? ' is-anchor' : ' is-minority'}`}
          cx={x} cy={y} r="5" />
        <text className="imb-halo imb-small" x={x + dx} y={y + dy + 4} textAnchor="middle">{point.id}</text>
      </g>;
    })}
    {revealed && (() => {
      /* Concentric distinct shapes plus a leader, so a generated point sitting
         exactly on an observed one is two visible identities, not one dot. The
         leader runs perpendicular to the segment for the same reason the point
         labels do. */
      const x = place(construction.generated.x);
      const y = lift(construction.generated.y);
      /* The generated point's label goes on the OPPOSITE side of the segment
         from the observed points' labels. When G lands exactly on an observed
         point — which is the whole point of the default setup — sharing a side
         puts the two labels on each other however far apart they are offset. */
      const away = offsetFor(x, y, 38);
      const dx = -away.dx;
      const dy = -away.dy;
      /* Clamped into the inner plot rectangle: left of 76 runs into the vertical
         tick labels, below height − 44 runs into the horizontal tick row, and
         either would put this label on top of an axis number. */
      const labelX = Math.min(Math.max(x + dx, 76), width - 40);
      const labelY = Math.min(Math.max(y + dy + 4, 22), height - 44);
      return <g>
        <circle className="imb-mark is-generated" cx={x} cy={y} r="11" />
        <line className="imb-axis" x1={x + (labelX - x) * 0.32} y1={y + (labelY - 4 - y) * 0.32}
          x2={x + (labelX - x) * 0.78} y2={y + (labelY - 4 - y) * 0.78} />
        <text className="imb-halo imb-small imb-strong" x={labelX} y={labelY} textAnchor="middle">
          G ({round(construction.generated.x, 3)}, {round(construction.generated.y, 3)})
        </text>
      </g>;
    })()}
  </svg>;
}

export function SmoteGeometryLab() {
  const state = useInvestigation(cloudBaseline);
  const draft = state.draft;
  const safeDraft = repairCloud(draft);
  // An edit can reach a cloud the construction legitimately refuses — every
  // point reclassified as majority, for instance. The refusal is explained in
  // place and the controls stay usable; it never becomes a blank page.
  const build = inputs => {
    try {
      return { construction: smoteConstruction(repairCloud(inputs)), problem: null };
    } catch (error) {
      return { construction: null, problem: error.message };
    }
  };
  const appliedBuild = build(state.active);
  const previewBuild = build(draft);
  const applied = appliedBuild.construction;
  const preview = previewBuild.construction;
  const blocked = previewBuild.problem ?? appliedBuild.problem;
  const revealed = Boolean(state.result) && Boolean(applied);
  const calculateInputs = (proposed, current) => {
    const after = smoteConstruction(repairCloud(proposed));
    const before = smoteConstruction(repairCloud(current));
    if (proposed.question === 'majorityMove') {
      const moved = Math.abs(after.generated.x - before.generated.x) > limits.tolerance
        || Math.abs(after.generated.y - before.generated.y) > limits.tolerance;
      return { outcome: moved ? 'moves' : 'unchanged', value: after.generated.x };
    }
    if (proposed.question === 'neighbour') {
      return { outcome: after.neighbour.id === before.neighbour.id ? 'unchanged' : 'moves', value: after.generated.x };
    }
    return { outcome: after.collidesWithMajority ? 'moves' : 'unchanged', value: after.generated.x };
  };
  const questionText = {
    collision: 'Will the generated point land exactly on top of an existing majority point?',
    majorityMove: 'After all the draft edits, including the required majority-point move, will the generated coordinates change?',
    neighbour: 'After all the draft edits, including a scale-divisor change, will the chosen neighbour change?',
  };
  const optionsFor = {
    collision: [['moves', 'Yes, it collides with a majority point'], ['unchanged', 'No majority-point collision']],
    majorityMove: [['moves', 'Yes, the generated point moves'], ['unchanged', 'No, it stays exactly where it was']],
    neighbour: [['moves', 'Yes, a different neighbour is chosen'], ['unchanged', 'No, the same neighbour is chosen']],
  };
  const editPoint = (index, update) => state.edit({
    points: draft.points.map((point, position) => (position === index ? { ...point, ...update } : point)),
  });
  return <Investigation
    title="Move the endpoints and inspect what the rule ignores"
    question="A small labelled cloud in original feature units, a declared distance metric, an anchor, a neighbour count and one interpolation fraction. Say what will happen before the construction is drawn."
    note="Ordinary SMOTE searches neighbours within the minority class only, and uses one scalar fraction for the whole vector. The scale divisors below are declared constants held fixed, not a scaler refitted from the edited points; refitting one silently would destroy the null this investigation depends on."
    onReset={state.reset}>

    <fieldset className="imb-row-group" style={{ '--imb-row-columns': 3 }}>
      <legend>The points. Coordinates are in original feature units.</legend>
      {draft.points.map((point, index) => <div key={point.id} style={{ display: 'contents' }}>
        <NumberField label={`${point.id} — x`} value={point.x} min={limits.coordinate.minimum} max={limits.coordinate.maximum}
          step="0.1" decimals={2} onChange={x => editPoint(index, { x })} />
        <NumberField label={`${point.id} — y`} value={point.y} min={limits.coordinate.minimum} max={limits.coordinate.maximum}
          step="0.1" decimals={2} onChange={y => editPoint(index, { y })} />
        <Select label={`${point.id} — class`} value={point.cls}
          options={[['minority', 'minority'], ['majority', 'majority']]}
          onChange={cls => editPoint(index, { cls })} />
      </div>)}
    </fieldset>
    <div className="imb-row-actions">
      <button type="button" disabled={draft.points.length >= limits.maximumPoints}
        onClick={() => state.edit({
          points: [...draft.points, { id: letters[draft.points.length], cls: 'minority', x: 1, y: 1 }],
        })}>Add a point</button>
      <button type="button" disabled={draft.points.length <= limits.minimumMinority}
        onClick={() => state.edit({ points: draft.points.slice(0, -1) })}>Remove the last point</button>
      <span className="imb-caption">
        At most {limits.maximumPoints} points, at least {limits.minimumMinority} of them minority. Coordinates in
        [{limits.coordinate.minimum}, {limits.coordinate.maximum}].
      </span>
    </div>
    <div className="imb-controls is-wide">
      <Select label="Anchor (a minority point)" value={safeDraft.anchorId}
        options={draft.points.filter(point => point.cls === 'minority').map(point => [point.id, point.id])}
        onChange={anchorId => state.edit({ anchorId })} />
      <NumberField label="k — neighbours considered" value={safeDraft.k} min={1}
        max={Math.max(1, draft.points.filter(point => point.cls === 'minority').length - 1)} step="1" decimals={0}
        hint={`At most one less than the number of minority points, which is ${draft.points.filter(point => point.cls === 'minority').length}.`}
        onChange={k => state.edit({ k })} />
      <NumberField label="Which of those k neighbours (rank, from 1)" value={safeDraft.neighbourRank + 1}
        min={1} max={safeDraft.k} step="1" decimals={0}
        onChange={value => state.edit({ neighbourRank: value - 1 })} />
      <SliderField label="Interpolation fraction u" value={safeDraft.fraction} min={0} max={1} step={0.05} decimals={2}
        onChange={fraction => state.edit({ fraction })} />
      <NumberField label="x scale divisor (metric only)" value={safeDraft.scaleX}
        min={limits.scaleDivisor.minimum} max={limits.scaleDivisor.maximum} step="0.1" decimals={2}
        onChange={scaleX => state.edit({ scaleX })} />
      <NumberField label="y scale divisor (metric only)" value={safeDraft.scaleY}
        min={limits.scaleDivisor.minimum} max={limits.scaleDivisor.maximum} step="0.1" decimals={2}
        onChange={scaleY => state.edit({ scaleY })} />
      <Select label="Question to answer" value={draft.question}
        options={[['collision', 'Does the generated point collide?'], ['majorityMove', 'Does moving a majority point change it?'], ['neighbour', 'Does the metric change the chosen neighbour?']]}
        onChange={question => state.edit({ question })} />
    </div>
    <div className="imb-buttons">
      {cloudSetups.map(setup => <button key={setup.key} type="button" onClick={() => state.suggest({
        points: setup.points, anchorId: setup.anchorId, k: setup.k, neighbourRank: setup.neighbourRank,
        fraction: setup.fraction, scaleX: setup.scaleX, scaleY: setup.scaleY, question: setup.question,
      })}>{setup.label}</button>)}
    </div>
    <p className="imb-state-strip">
      <span>applied: anchor <b>{applied ? applied.anchor.id : '—'}</b>, k <b>{applied ? applied.k : '—'}</b>, u <b>{applied ? round(applied.fraction, 2) : '—'}</b>, divisors <b>{applied ? `${round(applied.scaleX, 2)} / ${round(applied.scaleY, 2)}` : '—'}</b></span>
      <span className={state.pending ? 'is-draft' : undefined}>draft: anchor <b>{safeDraft.anchorId ?? '—'}</b>, k <b>{safeDraft.k}</b>, u <b>{round(safeDraft.fraction, 2)}</b>, divisors <b>{round(safeDraft.scaleX, 2)} / {round(safeDraft.scaleY, 2)}</b></span>
    </p>

    {blocked && <p className="imb-note" role="status">
      This cloud cannot support the construction: {blocked} Ordinary SMOTE needs at least two minority points and a
      neighbour count below their number, so the refusal is the algorithm&rsquo;s own contract rather than an interface
      limitation. Your other edits are kept; change a class or a count and the construction returns.
    </p>}

    {(revealed ? applied : preview) && <Scatter construction={revealed ? applied : preview} revealed={revealed} />}

    {!blocked && <LiveResult
      
      
      state={state} calculateInputs={calculateInputs}
      
      
      
      
      
       />}

    {revealed && <>
      <Table caption="The minority distance row from the anchor, under the declared divisors, with the anchor itself excluded"
        headings={['minority point', 'coordinates', 'scaled distance from the anchor', 'rank', 'inside the chosen k?']}
        rows={applied.distances.map(entry => [
          entry.id, `(${round(entry.x, 2)}, ${round(entry.y, 2)})`, round(entry.distance, 6), String(entry.rank),
          entry.rank <= applied.k ? 'yes' : 'no',
        ])}
        rowClass={index => (applied.distances[index].id === applied.neighbour.id ? 'is-chosen' : undefined)}
        footnote={`Distance is √((Δx/${round(applied.scaleX, 2)})² + (Δy/${round(applied.scaleY, 2)})²). Ties break on distance and then on identifier order.${applied.boundaryTie ? ' A tie straddles the k boundary here, so which points are even eligible depends on that rule.' : ''}${applied.tiedWithChosen.length > 0 ? ` The chosen neighbour is tied with ${applied.tiedWithChosen.join(', ')}.` : ''}`} />
      <p className="imb-caption">
        G = {applied.anchor.id} + u({applied.neighbour.id} − {applied.anchor.id}) =
        {' '}({round(applied.anchor.x, 2)}, {round(applied.anchor.y, 2)}) + {round(applied.fraction, 2)}·
        ({round(applied.neighbour.x - applied.anchor.x, 2)}, {round(applied.neighbour.y - applied.anchor.y, 2)}) =
        {' '}<strong>({round(applied.generated.x, 6)}, {round(applied.generated.y, 6)})</strong>.
        {' '}The same scalar u multiplies both coordinates; independent fractions per coordinate would generate a
        different shape entirely.
        {applied.zeroLength && ' The two endpoints coincide, so the segment has zero length and every fraction generates that same location.'}
      </p>
      <p className={applied.collidesWithMajority ? 'imb-note' : 'imb-caption'}>
        {applied.collidesWithMajority
          ? `The generated point sits exactly on ${applied.collisionIds.filter(id => id !== applied.anchor.id && id !== applied.neighbour.id).join(', ')}, an observed majority point. The interpolation consulted ${applied.consulted.join(' and ')} and never looked at ${applied.ignored.join(' or ')}. Two positive endpoints do not establish that the segment between them is positive.`
          : `${applied.collisionIds.length ? `The generated point coincides with minority point(s) ${applied.collisionIds.join(', ')}, but no majority point.` : 'The generated point does not coincide with an observed point.'} The construction still consulted only ${applied.consulted.join(' and ')}; the majority points ${applied.ignored.join(', ')} played no part in it either way.`}
      </p>
    </>}

    <div className="imb-panels is-pair">
      <div className="imb-panel">
        <h4>Why a one-hot vector cannot be interpolated</h4>
        <p>
          Two observations encode a three-level category as (1, 0, 0) and (0, 1, 0). Halfway between them is
          (.5, .5, 0). That is not a third category; it is not any category. SMOTENC handles mixed data and SMOTEN
          all-categorical data with different rules, and neither guarantees that a combined feature pattern is
          physically realisable.
        </p>
      </div>
      <div className="imb-panel">
        <h4>Why a complete label row must stay complete</h4>
        <p>
          Feature 0 was observed with labels (A = 1, B = 0); feature 2 with (A = 1, B = 1). Interpolating for A places
          a synthetic row at feature 1 and assigns A = 1. What is its B? The interpolation supplies no answer. Setting
          B = 0, copying one endpoint, taking the union or marking B missing are four different assumptions, and
          nothing in the geometry chooses among them.
        </p>
      </div>
    </div>
  </Investigation>;
}

/** Keep the anchor, k and rank valid after an edit removes or reclassifies a
 * point, rather than letting the model throw at render time. The repaired
 * values are what the state strip shows, so the learner sees the change. */
function repairCloud(inputs) {
  const minority = inputs.points.filter(point => point.cls === 'minority');
  const anchorId = minority.some(point => point.id === inputs.anchorId)
    ? inputs.anchorId : (minority[0]?.id ?? inputs.anchorId);
  const available = Math.max(1, minority.length - 1);
  const k = Math.min(Math.max(1, Math.round(inputs.k)), available);
  const neighbourRank = Math.min(Math.max(0, Math.round(inputs.neighbourRank)), k - 1);
  return { ...inputs, anchorId, k, neighbourRank };
}

/* ============================================================== §7 · I5 */

const tuningBaseline = {
  method: 'original',
  trial: 0.5,
  costFP: study.costFalsePositive,
  costFN: study.costFalseNegative,
  exploratory: false,
  
  question: 'cost',
};
const tuningSetups = [
  { key: 'default', label: 'Lower the trial gate from .5 to .15 on the original model', ...tuningBaseline, trial: 0.15 },
  { key: 'smote', label: 'Real contrast: the same gate on the SMOTE model', ...tuningBaseline, method: 'smote', trial: 0.15 },
  { key: 'double', label: 'Exact null: double both costs — does the selected gate move?', ...tuningBaseline, costFP: 2, costFN: 24, question: 'gate' },
  { key: 'doublecost', label: 'The same doubling, asked about the total cost instead', ...tuningBaseline, costFP: 2, costFN: 24, question: 'cost' },
  { key: 'cheapmiss', label: 'A cheaper miss: FN cost 3 — does the selected gate move?', ...tuningBaseline, costFN: 3, question: 'gate' },
  { key: 'dearmiss', label: 'A dearer miss: FN cost 48 — does the selected gate move?', ...tuningBaseline, costFN: 48, question: 'gate' },
  { key: 'noalert', label: 'The no-alert policy: gate above every score', ...tuningBaseline, trial: 1 },
];
const tuningQuestions = {
  cost: {
    options: [['decrease', 'It falls'], ['unchanged', 'Exactly unchanged'], ['increase', 'It rises']],
    numeric: { label: 'Optional: the new total cost', name: 'the new total cost', tolerance: 1e-6, digits: 6 },
  },
  gate: {
    options: [['unchanged', 'The selected gate does not move'], ['moves', 'The selected gate moves']],
    numeric: { label: 'Optional: the newly selected gate', name: 'the selected gate', tolerance: 1e-9, digits: 9 },
  },
};

export function TuningQueueLab() {
  const state = useInvestigation(tuningBaseline);
  const [showTruth, setShowTruth] = useState(true);
  const draft = state.draft;
  const active = state.active;
  const method = methods.find(entry => entry.name === active.method) ?? methods[0];
  const revealed = Boolean(state.result);
  const trialCounts = countsAt(tuningRecords.labels, method.tuningScores, active.trial);
  const trialCost = costOf(trialCounts, active.costFP, active.costFN);
  const sweep = thresholdSweep(tuningRecords.labels, method.tuningScores, active.costFP, active.costFN);
  const declaredSweep = thresholdSweep(tuningRecords.labels, method.tuningScores,
    study.costFalsePositive, study.costFalseNegative);
  const calculateInputs = (proposed, current) => {
    const nextMethod = methods.find(entry => entry.name === proposed.method) ?? methods[0];
    const previousMethod = methods.find(entry => entry.name === current.method) ?? methods[0];
    if (proposed.question === 'gate') {
      // The cost-optimal gate over the discrete candidate set. Scaling both
      // costs scales every candidate's cost by the same factor, so the argmin
      // cannot move: this is a genuine invariance of the cost-optimal gate.
      const afterGate = thresholdSweep(tuningRecords.labels, nextMethod.tuningScores,
        proposed.costFP, proposed.costFN).chosen;
      const beforeGate = thresholdSweep(tuningRecords.labels, previousMethod.tuningScores,
        current.costFP, current.costFN).chosen;
      const same = afterGate.threshold === beforeGate.threshold;
      return { outcome: same ? 'unchanged' : 'moves', value: afterGate.threshold };
    }
    const after = countsAt(tuningRecords.labels, nextMethod.tuningScores, proposed.trial);
    const before = countsAt(tuningRecords.labels, previousMethod.tuningScores, current.trial);
    const afterCost = costOf(after, proposed.costFP, proposed.costFN);
    const beforeCost = costOf(before, current.costFP, current.costFN);
    const difference = afterCost - beforeCost;
    return {
      outcome: Math.abs(difference) < 1e-12 ? 'unchanged' : difference < 0 ? 'decrease' : 'increase',
      value: afterCost,
    };
  };
  const selected = tuningRecords.sourceIds
    .map((id, index) => ({
      id, protein: tuningRecords.proteinIds[index], label: tuningRecords.labels[index],
      score: method.tuningScores[index],
    }))
    .sort((a, b) => b.score - a.score);
  const inspectionCounts = countsAt(inspectionRecords.labels, method.inspectionScores, method.chosenThreshold);
  return <Investigation
    title="Change an actual tuning policy and inspect the selected records"
    question={`The ${roles.tuning.records} tuning proteins, with the saved scores of one fitted procedure. A threshold is a development decision made on these records. Two quantities can be inspected here and they behave differently: the total declared cost at a trial gate, and the gate the cost sweep selects. Choose which quantity to inspect, and compare its live behavior.`}
    role={{
      kind: active.exploratory ? 'exploratory' : 'locked',
      text: active.exploratory
        ? 'Exploratory mode. You are choosing thresholds while looking at inspection outcomes. Anything you conclude here is exploratory: the inspection partition can no longer be described as untouched for that choice.'
        : `Role: tuning. These ${roles.tuning.records} proteins are the ones the protocol allows a threshold to be chosen on. The locked inspection result below was fixed before it was read, and the ${roles.reserve.records} reserved proteins are never scored anywhere in this lesson.`,
    }}
    note="Nothing here refits a classifier or chooses a model setting. The five score sets were fitted differently, so switching procedure changes the whole ranking — that is not one threshold moving."
    onReset={() => { state.reset(); setShowTruth(true); }}>

    <div className="imb-controls is-wide">
      <Select label="Fitted procedure whose saved scores you are thresholding" value={draft.method}
        options={methods.map(entry => [entry.name, entry.label])}
        onChange={value => state.edit({ method: value })} />
      <SliderField label="Trial gate on the tuning scores" value={draft.trial} min={0} max={1} step={0.01} decimals={2}
        onChange={trial => state.edit({ trial })} />
      <SliderField label="Hypothetical cost of a false alarm" value={draft.costFP}
        min={limits.cost.minimum} max={limits.cost.maximum} step={0.1} decimals={1}
        onChange={costFP => state.edit({ costFP })} />
      <SliderField label="Hypothetical cost of a missed positive" value={draft.costFN}
        min={limits.cost.minimum} max={limits.cost.maximum} step={0.1} decimals={1}
        onChange={costFN => state.edit({ costFN })} />
      <Select label="Quantity to inspect" value={draft.question}
        options={[['cost', 'The total declared cost at the trial gate'], ['gate', 'The gate the cost sweep selects']]}
        hint="Scaling both costs changes one of these and not the other; that is the distinction the null is about."
        onChange={question => state.edit({ question })} />
    </div>
    <div className="imb-buttons">
      {tuningSetups.map(setup => <button key={setup.key} type="button" onClick={() => state.suggest({
        method: setup.method, trial: setup.trial, costFP: setup.costFP, costFN: setup.costFN,
        question: setup.question, exploratory: draft.exploratory,
      })}>{setup.label}</button>)}
      <button type="button" className={draft.exploratory ? 'is-selected' : undefined}
        onClick={() => state.edit({ exploratory: !draft.exploratory })}>
        {draft.exploratory ? 'Leave exploratory mode' : 'Enter exploratory mode'}
      </button>
    </div>
    <p className="imb-state-strip">
      <span>applied: <b>{method.label}</b>, gate <b>≥ {round(active.trial, 2)}</b>, costs <b>{round(active.costFP, 1)} / {round(active.costFN, 1)}</b></span>
      <span className={state.pending ? 'is-draft' : undefined}>draft: <b>{(methods.find(entry => entry.name === draft.method) ?? methods[0]).label}</b>, gate <b>≥ {round(draft.trial, 2)}</b>, costs <b>{round(draft.costFP, 1)} / {round(draft.costFN, 1)}</b></span>
    </p>

    <LiveResult
      
      
      state={state} calculateInputs={calculateInputs}
      
      
      describe={revealed
        ? (active.question === 'gate'
          ? `The sweep selects ${sweep.chosen.isNoAlert ? 'the no-alert policy' : round(sweep.chosen.threshold, 9)} under these costs, at a total of ${round(sweep.chosen.cost, 2)} units. Scaling both costs by a common factor scales every candidate's cost by that factor, so the candidate that minimises it cannot change.`
          : `At the applied gate the counts are TP ${trialCounts.tp}, FP ${trialCounts.fp}, FN ${trialCounts.fn}, TN ${trialCounts.tn}, for a total of ${round(trialCost, 2)} units.`)
        : undefined}
      
       />

    {revealed && <>
      <CountStrip counts={trialCounts} />
      <p className="imb-caption">
        Cost = {round(active.costFP, 1)}×{trialCounts.fp} false alarms + {round(active.costFN, 1)}×{trialCounts.fn}
        {' '}missed positives = <strong>{round(trialCost, 2)}</strong> units. Detected positives: {trialCounts.tp} of
        {' '}{trialCounts.positives}.
      </p>
      <div className="imb-row-actions">
        <button type="button" className={showTruth ? 'is-selected' : undefined} onClick={() => setShowTruth(!showTruth)}>
          {showTruth ? 'Hide the actual labels in the queue' : 'Show the actual labels in the queue'}
        </button>
        <span className="imb-caption">
          Labels are shown alongside the selected records. You can hide them to reduce visual detail;
          this display choice changes nothing about the analysed set.
        </span>
      </div>
      <Table caption={`The ${roles.tuning.records} tuning proteins in saved score order, with the applied gate marking each as selected or not`}
        headings={['source ID', 'protein', 'saved score', 'selected at the gate', showTruth ? 'actual ME2?' : 'actual label']}
        rows={selected.map(entry => [
          String(entry.id), entry.protein, round(entry.score, 6),
          entry.score >= active.trial ? 'selected' : '—',
          showTruth ? (entry.label === 1 ? 'yes' : 'no') : 'hidden',
        ])}
        rowClass={index => (selected[index].score >= active.trial
          ? (showTruth && selected[index].label === 1 ? 'is-positive' : 'is-selected')
          : undefined)}
        scroll
        footnote="A search or a sort of this display would not change the analysed set; every record above is inside it." />
      <Table caption={`Displayed candidates: near-optimal gates plus a regular sample, in sweep order`}
        headings={['candidate gate', 'TP', 'FP', 'FN', 'TN', 'total cost']}
        rows={sweep.candidates
          .filter((candidate, index) => index === 0 || candidate.cost <= sweep.chosen.cost + 6 * active.costFN || index % 13 === 0)
          .map(candidate => [
            candidate.isNoAlert ? 'above every score (no alert)' : round(candidate.threshold, 6),
            String(candidate.counts.tp), String(candidate.counts.fp), String(candidate.counts.fn),
            String(candidate.counts.tn), round(candidate.cost, 2),
          ])}
        scroll
        footnote={`Of the ${sweep.candidates.length} candidates, the minimum cost is ${round(sweep.chosen.cost, 2)} at a gate of ${sweep.chosen.isNoAlert ? 'above every score' : round(sweep.chosen.threshold, 6)}. ${sweep.tiedCandidates.length > 1 ? `${sweep.tiedCandidates.length} candidates reach that same cost — ${sweep.tiedCandidates.map(candidate => (candidate.isNoAlert ? 'no alert' : round(candidate.threshold, 6))).join(' and ')} — and because candidates are traversed descending while a strict comparison keeps the first, the highest of them is the one selected.` : 'No other candidate reaches that cost.'} The rows shown are the near-optimal ones plus a regular sample of the rest; the sweep itself uses every candidate.`} />
      <p className="imb-caption">
        Under the study's declared costs of {study.costFalsePositive} and {study.costFalseNegative}, this procedure's
        selected gate is <strong>{round(declaredSweep.chosen.threshold, 6)}</strong>. Under the costs you applied it
        is {sweep.chosen.isNoAlert ? 'the no-alert policy' : round(sweep.chosen.threshold, 6)}. Both are shown to six
        places; the decision itself uses the saved score exactly, never this rounded display. A selected gate can
        stay put over a range of costs, because the candidate set is discrete — an unmoving marker there is
        information, not a broken control.
      </p>
      <p className="imb-caption">
        Moving this gate leaves the ranking alone. Average precision on these tuning records is
        {' '}{round(averagePrecisionOfScores(tuningRecords.labels, method.tuningScores), 6)} at every gate, because it
        summarises the order and not the cutoff. Switching procedure <em>does</em> change it, because those scores came
        from a different fit.
      </p>
      <div className={`imb-role ${active.exploratory ? 'is-exploratory' : 'is-locked'}`}>
        <strong>{active.exploratory ? 'Exploratory inspection view.' : 'Locked inspection result.'}</strong>{' '}
        {method.label} was frozen with its threshold {round(method.chosenThreshold, 6)}, chosen on the tuning records
        under the declared costs, and then applied once to the {roles.inspection.records} inspection proteins: TP
        {' '}{inspectionCounts.tp}, FP {inspectionCounts.fp}, FN {inspectionCounts.fn}, TN {inspectionCounts.tn}, for a
        realised cost of {costOf(inspectionCounts, study.costFalsePositive, study.costFalseNegative)} units.
        {active.exploratory
          ? ' Because you are in exploratory mode, treat this as an exploration of the inspection partition rather than as an untouched assessment; a newly chosen procedure would need a fresh protocol before being reported as independently assessed.'
          : ' That outcome was fixed before it was read, and nothing you do above changes it.'}
        {' '}The {roles.reserve.records} reserved proteins receive no prediction or score here or anywhere else.
      </div>
    </>}
  </Investigation>;
}

