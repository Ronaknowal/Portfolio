import { useState } from 'react';
import {
  consecutiveFolds, enumerateSelection, foldPlan, futureAccuracy, nearestNeighborRun, nestedTrace,
  scoreCandidates, successiveHalving,
} from '../../data/validation-models';
import {
  Investigation, NumberField, Prediction, RetiredNotice, RoleMark, SelectField, Table, fixed, round, useInvestigation,
} from './ValidationShared.jsx';
import './validation-labs.css';

const BINARY = [[0, 'label 0'], [1, 'label 1']];

/* ================================================================== *
 * I1 · §2 — build the held-out predictions
 * ================================================================== */
const FOLD_PRESETS = {
  base: {
    label: 'Baseline: three consecutive folds',
    inputs: { x: [0, 1, 2, 3, 4, 5, 6], y: [0, 0, 0, 1, 1, 1, 1], fold: [0, 0, 0, 1, 1, 2, 2], inspect: 0 },
    note: 'The constructed order from the lesson: folds 1, 2 and 3 hold out rows 0–2, 3–4 and 5–6.',
  },
  contrast: {
    label: 'Contrast: training row 3 relabelled',
    inputs: { x: [0, 1, 2, 3, 4, 5, 6], y: [0, 0, 0, 0, 1, 1, 1], fold: [0, 0, 0, 1, 1, 2, 2], inspect: 0 },
    note: 'Row 3 trains fit 1 and is the nearest neighbour of row 0, so its label reaches row 0’s prediction.',
  },
  ownLabel: {
    label: 'Null: row 0’s own label changed',
    inputs: { x: [0, 1, 2, 3, 4, 5, 6], y: [1, 0, 0, 1, 1, 1, 1], fold: [0, 0, 0, 1, 1, 2, 2], inspect: 0 },
    note: 'Row 0 is held out by fit 1, so its own label cannot reach the model predicting it. Only its correctness can move.',
  },
  shifted: {
    label: 'Null: every x shifted by +10',
    inputs: { x: [10, 11, 12, 13, 14, 15, 16], y: [0, 0, 0, 1, 1, 1, 1], fold: [0, 0, 0, 1, 1, 2, 2], inspect: 0 },
    note: 'A common translation changes every distance by nothing at all, because only differences matter.',
  },
  interleaved: {
    label: 'Alternative assignment: interleaved folds',
    inputs: { x: [0, 1, 2, 3, 4, 5, 6], y: [0, 0, 0, 1, 1, 1, 1], fold: [0, 1, 2, 0, 1, 2, 0], inspect: 0 },
    note: 'A different partition of the same rows. Every row is still assessed exactly once.',
  },
};

function buildFoldState(inputs) {
  const points = inputs.x.map((value, id) => ({ id, x: value, y: inputs.y[id] }));
  const folds = [0, 1, 2].map(index => points.filter(point => inputs.fold[point.id] === index).map(point => point.id));
  const plan = foldPlan({ rows: points.map(point => point.id), folds, labelOf: id => inputs.y[id] });
  return { points, plan, run: nearestNeighborRun(points, plan) };
}

export function FoldBuilderLab() {
  const state = useInvestigation(FOLD_PRESETS.base.inputs);
  const [presetKey, setPresetKey] = useState('base');
  const draft = state.draft;
  let draftError = null;
  try { buildFoldState(draft); } catch (error) { draftError = error.message; }
  const applied = state.result ? buildFoldState(state.result.inputs) : null;
  const inspectedFoldIndex = draft.fold[draft.inspect];
  const preview = (() => {
    try { return buildFoldState(draft); } catch { return null; }
  })();
  const previewFold = preview?.plan.folds[inspectedFoldIndex];
  const span = (() => {
    const values = draft.x;
    const low = Math.min(...values);
    const high = Math.max(...values);
    const pad = Math.max(0.5, (high - low) * 0.08);
    return [low - pad, high + pad];
  })();
  const place = value => 26 + 308 * (value - span[0]) / (span[1] - span[0]);
  const answerFor = inputs => ({ label: String(buildFoldState(inputs).run.folds
    .flatMap(fold => fold.rows).find(row => row.row === inputs.inspect).prediction) });
  const inspectedResult = applied?.run.folds.flatMap(fold => fold.rows).find(row => row.row === state.result.inputs.inspect);

  return <Investigation id="folds" title="Build the held-out predictions"
    question="Seven constructed rows, three validation folds and a one-nearest-neighbour rule. Choose a row to inspect, predict the label the fit that assesses it will give it, then look at which training rows that fit was actually allowed to use."
    note="Ties in distance resolve to the smallest source row ID. A held-out row's own label never reaches the fit that predicts it; it only decides whether that prediction is counted correct."
    onReset={() => { state.reset(); setPresetKey('base'); }}>
    <div className="cv-buttons" role="group" aria-label="Declared setups">
      {Object.entries(FOLD_PRESETS).map(([key, preset]) => (
        <button key={key} type="button" aria-pressed={key === presetKey}
          onClick={() => { state.load(preset.inputs); setPresetKey(key); }}>{preset.label}</button>
      ))}
    </div>
    <p className="cv-caption">{FOLD_PRESETS[presetKey]?.note ?? 'A custom fold plan. Every distance, neighbour and summary below is recomputed from these seven rows.'}</p>

    <h4>The seven rows</h4>
    <div className="cv-rows">
      {draft.x.map((value, id) => (
        <div className="cv-row" key={id}>
          <span className="cv-row-id">row {id}</span>
          <NumberField label={`feature x of row ${id}`} shortLabel="feature x" value={value} min={-20} max={30} step="0.5" decimals={2}
            onChange={next => { setPresetKey('custom'); state.edit({ x: draft.x.map((old, index) => (index === id ? next : old)) }); }} />
          <SelectField label={`label y of row ${id}`} shortLabel="label y" value={draft.y[id]} options={BINARY}
            onChange={next => { setPresetKey('custom'); state.edit({ y: draft.y.map((old, index) => (index === id ? Number(next) : old)) }); }} />
          <SelectField label={`validation fold of row ${id}`} shortLabel="fold" value={draft.fold[id]} options={[[0, 'fold 1'], [1, 'fold 2'], [2, 'fold 3']]}
            onChange={next => { setPresetKey('custom'); state.edit({ fold: draft.fold.map((old, index) => (index === id ? Number(next) : old)) }); }} />
        </div>
      ))}
    </div>
    {draftError && <p className="cv-field-error" role="alert" data-cv-error>{draftError}</p>}

    <div className="cv-controls">
      <SelectField label="Row to inspect" value={draft.inspect} range="0 to 6"
        options={draft.x.map((_, id) => [id, `row ${id} (fold ${draft.fold[id] + 1})`])}
        onChange={next => state.edit({ inspect: Number(next) })} />
    </div>

    <h4>Which fit may use which row</h4>
    <div className="cv-matrix" role="region" aria-label="Role matrix: source rows down the side, the three fits across the top" tabIndex={0}>
      <table>
        <caption className="cv-caption">Every column is one complete refit. A row is held out by exactly one of them.</caption>
        <thead><tr><th scope="col">source row</th>{[0, 1, 2].map(index => <th key={index} scope="col">fit {index + 1}</th>)}</tr></thead>
        <tbody>
          {draft.x.map((_, id) => <tr key={id}>
            <th scope="row">{id}</th>
            {[0, 1, 2].map(index => <td key={index}>
              {draft.fold[id] === index ? <RoleMark kind="held">held out</RoleMark> : <RoleMark kind="train">trains</RoleMark>}
            </td>)}
          </tr>)}
        </tbody>
      </table>
    </div>

    {previewFold && <>
      <h4>Fit {inspectedFoldIndex + 1} on one axis</h4>
      <svg viewBox="0 0 360 116" role="img" aria-label={`Fit ${inspectedFoldIndex + 1} trains on rows ${previewFold.training.join(', ')} and holds out rows ${previewFold.validation.join(', ')}. Training rows are drawn as filled dots above the axis and held-out rows as hollow squares below it, each labelled with its source ID.`}>
        <line className="cv-axis" x1="20" y1="62" x2="344" y2="62" />
        {previewFold.training.map(id => <g key={`t${id}`}>
          <circle className="cv-mark is-train" cx={place(draft.x[id])} cy="48" r="5" />
          <text x={place(draft.x[id])} y="38" textAnchor="middle">{id}</text>
          <text x={place(draft.x[id])} y="18" textAnchor="middle">y{draft.y[id]}</text>
        </g>)}
        {previewFold.validation.map(id => <g key={`v${id}`}>
          <rect className="cv-mark is-held" x={place(draft.x[id]) - 5} y="74" width="10" height="10" />
          <text x={place(draft.x[id])} y="104" textAnchor="middle">{id}</text>
        </g>)}
        {inspectedResult && state.result.inputs.fold[state.result.inputs.inspect] === inspectedFoldIndex && (
          <line className="cv-stem" x1={place(state.result.inputs.x[state.result.inputs.inspect])} y1="74"
            x2={place(state.result.inputs.x[inspectedResult.neighbor])} y2="49" />
        )}
      </svg>
      <p className="cv-caption">
        Filled dots above the axis are training rows, with their labels; hollow squares below it are the rows this fit assesses. Fit{' '}
        {inspectedFoldIndex + 1} trains on {previewFold.training.length} rows and assesses {previewFold.validation.length}.
      </p>
    </>}

    <Prediction
      questions={[{
        key: 'label',
        short: 'Predicted label',
        prompt: `Fit ${inspectedFoldIndex + 1} assesses row ${draft.inspect}. Which label will it predict for that row?`,
        options: BINARY.map(([value, text]) => [value, text]),
      }]}
      state={state} answerFor={answerFor} applyLabel="Apply the fold plan"
      disabled={Boolean(draftError)} disabledReason={draftError ? 'Fix the fold assignment before recording a prediction.' : undefined}
      describe={(answer, inputs) => {
        const built = buildFoldState(inputs);
        const row = built.run.folds.flatMap(fold => fold.rows).find(item => item.row === inputs.inspect);
        return `Row ${inputs.inspect} sits at x = ${round(inputs.x[inputs.inspect])}. Its nearest eligible training row is ${row.neighbor} at x = ${round(row.neighborX)}, a distance of ${round(row.distance)}${row.tied.length ? `, tied with row${row.tied.length > 2 ? 's' : ''} ${row.tied.slice(1).join(' and ')} and resolved to the smallest source ID` : ''}. That neighbour carries label ${row.prediction}, so the prediction is ${row.prediction}; the true label ${row.truth} decides only whether it counts as correct.`;
      }} />
    <RetiredNotice state={state} />

    {applied && inspectedResult && <>
      <h4>What fit {state.result.inputs.fold[state.result.inputs.inspect] + 1} could see</h4>
      <Table id="contributions" caption={`Every eligible training row for row ${state.result.inputs.inspect}, with its distance. The held-out row itself is absent by construction.`}
        headings={['training row', 'x', 'label', 'distance to the inspected row', 'role']} numeric={[1, 3]}
        rowClass={index => (inspectedResult.contributions[index].row === inspectedResult.neighbor ? 'is-selected' : undefined)}
        rows={inspectedResult.contributions.map(item => [item.row, round(item.x), item.label, round(item.distance),
          item.row === inspectedResult.neighbor ? 'nearest' : '—'])} />

      <h4>The complete held-out prediction table</h4>
      <Table id="predictions" caption="One row per assessed observation. Every row is assessed exactly once across the three fits."
        headings={['source row', 'assessed by', 'nearest training row', 'prediction', 'true label', 'counted']}
        rows={applied.run.folds.flatMap(fold => fold.rows.map(row => [
          row.row, `fit ${fold.index + 1}`, row.neighbor, row.prediction, row.truth, row.correct ? 'correct' : 'incorrect',
        ]))} />

      <div className="cv-readout">
        <dl>
          {applied.run.folds.map(fold => <span key={fold.index} style={{ display: 'contents' }}>
            <dt>Fit {fold.index + 1}</dt>
            <dd>{fold.correct} / {fold.assessed} = {round(fold.accuracy, 6)}</dd>
          </span>)}
          <dt>Unweighted fold mean</dt><dd>{round(applied.run.foldMean, 6)}</dd>
          <dt>Pooled row accuracy</dt><dd>{applied.run.totalCorrect} / {applied.run.totalAssessed} = {round(applied.run.pooled, 10)}</dd>
        </dl>
      </div>
      <p className="cv-caption">
        The two summaries weight differently: each fold carries one third in the first, each assessed row one{' '}
        {applied.run.totalAssessed}th in the second.{' '}
        {(() => {
          const sizes = applied.run.folds.map(fold => fold.assessed);
          const equalSizes = sizes.every(size => size === sizes[0]);
          if (applied.run.foldMean !== applied.run.pooled) return 'They differ here because the folds are unequal in size and their accuracies are not all the same.';
          if (equalSizes) return 'They agree here because the folds are equal in size, which makes the two weightings identical.';
          return 'They agree here despite the unequal fold sizes, because every fold reaches the same accuracy; equal weightings are not the only way they can coincide.';
        })()}{' '}
        The size-weighted fold mean, {round(applied.run.weightedFoldMean, 10)}, is the pooled value by construction.
      </p>
    </>}
    <p className="cv-limits">
      The consecutive baseline is deliberately ordered by class to make the mechanics visible. It is not a recommended split: ordering by
      class creates very different training problems in each fold.
    </p>
  </Investigation>;
}

/* ================================================================== *
 * I2 · §4 — win the familiar cases
 * ================================================================== */
const CONSTANT_CANDIDATES = [[0, 0, 0, 0], [1, 1, 1, 1]];
const SELECTION_PRESETS = {
  base: { label: 'Baseline: two constant rules', inputs: { labels: [1, 1, 1, 0], patterns: CONSTANT_CANDIDATES } },
  duplicate: { label: 'Null: add a duplicate rule', inputs: { labels: [1, 1, 1, 0], patterns: [...CONSTANT_CANDIDATES, [0, 0, 0, 0]] } },
  everyPattern: {
    label: 'All sixteen prediction patterns',
    inputs: { labels: [1, 1, 1, 0], patterns: Array.from({ length: 16 }, (_, index) => [3, 2, 1, 0].map(bit => (index >> bit) & 1)) },
  },
  alternating: { label: 'Fixture: labels 0,1,0,1', inputs: { labels: [0, 1, 0, 1], patterns: CONSTANT_CANDIDATES } },
  matching: {
    label: 'Fixture plus a matching rule',
    inputs: { labels: [0, 1, 0, 1], patterns: [...CONSTANT_CANDIDATES, [0, 1, 0, 1]] },
  },
};
const candidateName = index => `candidate ${String.fromCharCode(65 + index)}`;

export function SelectionLab() {
  const state = useInvestigation(SELECTION_PRESETS.base.inputs);
  const [presetKey, setPresetKey] = useState('base');
  const draft = state.draft;
  const candidatesOf = inputs => inputs.patterns.map((pattern, index) => ({ id: candidateName(index), pattern }));
  const answerFor = inputs => ({
    best: String(scoreCandidates(inputs.labels, candidatesOf(inputs)).selectedValidationAccuracy),
  });
  const applied = state.result ? {
    scored: scoreCandidates(state.result.inputs.labels, candidatesOf(state.result.inputs)),
    enumeration: enumerateSelection(candidatesOf(state.result.inputs)),
  } : null;
  const setPattern = (candidate, position, value) => {
    setPresetKey('custom');
    state.edit({ patterns: draft.patterns.map((pattern, index) => (index === candidate ? pattern.map((old, spot) => (spot === position ? value : old)) : pattern)) });
  };
  return <Investigation id="selection" title="Win the familiar cases"
    question="Four validation cases whose labels are independent fair coin flips, and a list of fixed prediction rules. Predict the best validation accuracy the list will reach, then compare it with what those rules can be expected to do on fresh labels."
    note="The generating model is declared before the result: each validation label, and each future label, is an independent fair bit that the features carry no information about. Every rule below is therefore expected to be right half the time on fresh labels, however it was chosen."
    onReset={() => { state.reset(); setPresetKey('base'); }}>
    <div className="cv-buttons" role="group" aria-label="Declared setups">
      {Object.entries(SELECTION_PRESETS).map(([key, preset]) => (
        <button key={key} type="button" aria-pressed={key === presetKey}
          onClick={() => { state.load(preset.inputs); setPresetKey(key); }}>{preset.label}</button>
      ))}
    </div>

    <h4>The four validation labels</h4>
    <div className="cv-controls">
      {draft.labels.map((value, position) => (
        <SelectField key={position} label={`validation label, case ${position + 1}`} value={value} options={BINARY}
          onChange={next => { setPresetKey('custom'); state.edit({ labels: draft.labels.map((old, index) => (index === position ? Number(next) : old)) }); }} />
      ))}
    </div>

    <h4>The candidate rules ({draft.patterns.length} of at most 16)</h4>
    <div className="cv-rows">
      {draft.patterns.map((pattern, candidate) => (
        <div className="cv-row" key={candidate} style={{ gridTemplateColumns: 'minmax(6rem, auto) repeat(4, minmax(0, 1fr)) auto' }}>
          <span className="cv-row-id">{candidateName(candidate)}</span>
          {pattern.map((value, position) => (
            <button key={position} type="button" aria-pressed={value === 1}
              aria-label={`${candidateName(candidate)}, case ${position + 1}, prediction ${value}`}
              onClick={() => setPattern(candidate, position, value === 1 ? 0 : 1)}>{value}</button>
          ))}
          <button type="button" disabled={draft.patterns.length <= 1}
            aria-label={`remove ${candidateName(candidate)}`}
            onClick={() => { setPresetKey('custom'); state.edit({ patterns: draft.patterns.filter((_, index) => index !== candidate) }); }}>remove</button>
        </div>
      ))}
    </div>
    <div className="cv-buttons">
      <button type="button" disabled={draft.patterns.length >= 16}
        onClick={() => { setPresetKey('custom'); state.edit({ patterns: [...draft.patterns, [0, 0, 0, 0]] }); }}>Add a candidate rule</button>
    </div>
    <p className="cv-caption">
      Each button is one rule's prediction at one fixed feature position; press it to flip that prediction between 0 and 1. At least one rule
      must remain, and at most sixteen are allowed, because sixteen distinct four-bit patterns exhaust the space.
    </p>

    <Prediction
      questions={[{
        key: 'best',
        short: 'Best validation accuracy',
        prompt: `With these ${draft.patterns.length} rules and these four labels, what is the best validation accuracy any rule reaches?`,
        options: [[0, '0'], [0.25, '0.25 (1 of 4)'], [0.5, '0.5 (2 of 4)'], [0.75, '0.75 (3 of 4)'], [1, '1 (4 of 4)']],
      }]}
      state={state} answerFor={answerFor} applyLabel="Apply and score the rules"
      describe={(answer, inputs) => {
        const scored = scoreCandidates(inputs.labels, candidatesOf(inputs));
        return `${scored.winner.id} matches ${scored.winner.correct} of the four labels, so the selected validation accuracy is ${round(scored.selectedValidationAccuracy)}. Its expected accuracy on fresh fair labels is ${round(futureAccuracy(scored.winner.pattern))}: selecting it used the validation labels, and it learned nothing about future ones.${scored.tieBrokenBy ? ` ${scored.tiedWith.length} rules tie at that score, resolved by the ${scored.tieBrokenBy}.` : ''}`;
      }} />
    <RetiredNotice state={state} />

    {applied && <>
      <Table id="candidate-scores" caption="Every rule against the applied validation labels, with the positions it matched. The last column is each rule's expected accuracy on fresh fair labels under the declared generating model."
        headings={['rule', 'predictions', 'matched cases', 'validation accuracy', 'on fresh labels']}
        numeric={[3, 4]}
        rowClass={index => (index === applied.scored.winner.index ? 'is-selected' : undefined)}
        rows={applied.scored.rows.map(row => [
          // The marker rides in the first cell rather than a sixth column: an
          // extra column pushed the fresh-label figure past the visible width.
          row.index === applied.scored.winner.index ? `${row.id} · selected` : row.id,
          row.pattern.join(' '),
          row.matches.map((matched, position) => (matched ? position + 1 : null)).filter(Boolean).join(', ') || 'none',
          `${row.correct} / 4 = ${round(row.validationAccuracy)}`,
          round(futureAccuracy(row.pattern)),
        ])} />
      <div className="cv-readout">
        <dl>
          <dt>Selected rule</dt><dd>{applied.scored.winner.id}</dd>
          <dt>Selected validation accuracy</dt><dd>{round(applied.scored.selectedValidationAccuracy)}</dd>
          <dt>Expected accuracy on fresh fair labels</dt><dd>{round(futureAccuracy(applied.scored.winner.pattern))}</dd>
        </dl>
      </div>
      <h4>All sixteen equally likely label patterns</h4>
      <Table id="enumeration" caption="Exhaustive enumeration for the applied rule list. No sampling and no simulation: each pattern has probability exactly 1/16."
        headings={['number of ones', 'patterns', 'probability', 'best correct count reached']} numeric={[1, 2, 3]}
        rows={applied.enumeration.byOnesCount.map(row => [
          row.ones, row.patterns, `${row.patterns}/16`, row.bestCorrect.join(' or '),
        ])} />
      <details>
        <summary>Inspect all 16 label patterns and reconstruct the average</summary>
        <Table id="enumeration-detail" caption="Each row has probability 1/16. Sum the best correct counts and divide by 16 × 4 to recover the average selected accuracy."
          headings={['labels', 'selected rule', 'best correct count', 'selected accuracy', 'probability']} numeric={[2, 3, 4]}
          rows={applied.enumeration.rows.map(row => [row.labels.join(' '), row.winner, row.bestCorrect, `${row.bestCorrect}/4`, '1/16'])} />
      </details>
      <div className="cv-readout">
        <dl>
          <dt>Patterns enumerated</dt><dd>{applied.enumeration.patternCount}</dd>
          <dt>Average selected validation accuracy</dt><dd>{round(applied.enumeration.meanSelectedAccuracy, 6)}</dd>
          <dt>Expected accuracy on fresh fair labels</dt><dd>{round(applied.enumeration.futureAccuracy)}</dd>
        </dl>
      </div>
      <p className="cv-caption">
        The average selected score is what a report would print if this whole exercise were repeated on new validation labels. The gap between
        it and {round(applied.enumeration.futureAccuracy)} is selection, not learning: adding a rule that happens to match raises the first
        number and never the second. Adding a duplicate of an existing rule raises neither.
      </p>
    </>}
    <p className="cv-limits">
      This is a constructed environment with an analytically known answer. It shows that the mechanism exists and how it works; it is not a
      claim that every real hyperparameter search carries this much optimism, nor that each candidate is an independent hypothesis test.
    </p>
  </Investigation>;
}

/* ================================================================== *
 * I3 · §6 — open one nested fold
 * ================================================================== */
const NESTED_BASE_X = Array.from({ length: 16 }, (_, index) => index);
const NESTED_BASE_Y = NESTED_BASE_X.map(value => (value >= 8 ? 1 : 0));
const relabel = (id, value) => NESTED_BASE_Y.map((old, index) => (index === id ? value : old));
const NESTED_PRESETS = {
  base: {
    label: 'Baseline',
    inputs: { x: NESTED_BASE_X, y: NESTED_BASE_Y, outer: 0, target: 'k', row: 8 },
    note: 'x runs 0 to 15 with labels 0 below 8 and 1 from 8 upward. Outer fold 1 assesses even row IDs; outer fold 2 assesses odd ones.',
  },
  selection: {
    label: 'Contrast: row 3 relabelled',
    inputs: { x: NESTED_BASE_X, y: relabel(3, 1), outer: 0, target: 'k', row: 8 },
    note: 'Row 3 is an outer training row in fold 1, so its label is a legitimate input to that fold’s selection.',
  },
  protectedNull: {
    label: 'Null: row 2 relabelled',
    inputs: { x: NESTED_BASE_X, y: relabel(2, 1), outer: 0, target: 'k', row: 8 },
    note: 'Row 2 is protected in fold 1. Its label can change whether a prediction is counted correct, and nothing else.',
  },
  roleSwap: {
    label: 'Role swap: row 7 relabelled',
    inputs: { x: NESTED_BASE_X, y: relabel(7, 1), outer: 0, target: 'row', row: 8 },
    note: 'Row 7 trains outer fold 1 and is protected in outer fold 2. Inspect both folds with this input applied.',
  },
};

export function NestedLab() {
  const state = useInvestigation(NESTED_PRESETS.base.inputs);
  const [presetKey, setPresetKey] = useState('base');
  const draft = state.draft;
  let draftError = null;
  try { nestedTrace({ x: draft.x, y: draft.y }); } catch (error) { draftError = error.message; }
  const previewFold = (() => {
    try { return nestedTrace({ x: draft.x, y: draft.y })[draft.outer]; } catch { return null; }
  })();
  // A protected row belongs to one outer fold, so switching folds moves the
  // target. Resolve it the same way here and in the prompt, rather than letting
  // a stale ID reach the trace.
  const protectedRowOf = inputs => (inputs.row % 2 === inputs.outer ? inputs.row : inputs.outer);
  const answerFor = inputs => {
    const fold = nestedTrace({ x: inputs.x, y: inputs.y })[inputs.outer];
    return inputs.target === 'k'
      ? { outcome: String(fold.selectedK) }
      : { outcome: String(fold.refit.find(row => row.row === protectedRowOf(inputs)).prediction) };
  };
  const applied = state.result ? nestedTrace({ x: state.result.inputs.x, y: state.result.inputs.y })[state.result.inputs.outer] : null;
  const protectedRows = previewFold ? previewFold.test : [];
  const targetRow = protectedRows.includes(draft.row) ? draft.row : protectedRows[0];

  return <Investigation id="nested" title="Open one nested fold"
    question="Sixteen constructed rows, two outer folds by row-ID parity, two inner folds by alternating position, and two candidate neighbour counts. Predict either which count the inner comparison selects, or the label a protected row receives, then watch every fit that produced it."
    note="Each phase is a separate step: split, compare candidates inside the outer training rows only, select, refit on all outer training rows, then predict the protected rows. Equal inner means select the smaller neighbour count."
    onReset={() => { state.reset(); setPresetKey('base'); }}>
    <div className="cv-buttons" role="group" aria-label="Declared setups">
      {Object.entries(NESTED_PRESETS).map(([key, preset]) => (
        <button key={key} type="button" aria-pressed={key === presetKey}
          onClick={() => { state.load(preset.inputs); setPresetKey(key); }}>{preset.label}</button>
      ))}
    </div>
    <p className="cv-caption">{NESTED_PRESETS[presetKey]?.note ?? 'A custom fixture. Every distance and vote below is recomputed from these values.'}</p>

    <div className="cv-controls">
      <SelectField label="Outer fold to open" value={draft.outer} range="fold 1 or 2"
        options={[[0, 'outer fold 1 · assesses even row IDs'], [1, 'outer fold 2 · assesses odd row IDs']]}
        onChange={next => {
          const outer = Number(next);
          state.edit({ outer, row: draft.row % 2 === outer ? draft.row : outer });
        }} />
      <SelectField label="What to predict" value={draft.target} range="two outputs"
        options={[['k', 'the selected neighbour count'], ['row', 'a protected row’s predicted label']]}
        onChange={next => state.edit({ target: next })} />
      {draft.target === 'row' && (
        <SelectField label="Protected row to predict" value={targetRow} range={`${protectedRows.length} rows`}
          options={protectedRows.map(id => [id, `row ${id}`])}
          onChange={next => state.edit({ row: Number(next) })} />
      )}
    </div>

    <details>
      <summary>Edit the sixteen rows</summary>
      <div className="cv-rows">
        {draft.x.map((value, id) => (
          <div className="cv-row" key={id}>
            <span className="cv-row-id">row {id}</span>
            <NumberField label={`feature x of row ${id}`} shortLabel="feature x" value={value} min={-20} max={40} step="0.5" decimals={2}
              onChange={next => { setPresetKey('custom'); state.edit({ x: draft.x.map((old, index) => (index === id ? next : old)) }); }} />
            <SelectField label={`label y of row ${id}`} shortLabel="label y" value={draft.y[id]} options={BINARY}
              onChange={next => { setPresetKey('custom'); state.edit({ y: draft.y.map((old, index) => (index === id ? Number(next) : old)) }); }} />
            <span className="cv-row-id">{id % 2 === draft.outer
              ? <RoleMark kind="protected">protected here</RoleMark>
              : <RoleMark kind="train">trains here</RoleMark>}</span>
          </div>
        ))}
      </div>
    </details>
    {draftError && <p className="cv-field-error" role="alert" data-cv-error>{draftError}</p>}

    {previewFold && <>
      <h4>Step 1 · Close the room</h4>
      <p className="cv-caption">
        Outer fold {draft.outer + 1} protects rows {previewFold.test.join(', ')} and may use rows {previewFold.train.join(', ')}. The inner
        folds partition only the second list: inner 1 assesses {previewFold.inner[0].validation.join(', ')} and inner 2 assesses{' '}
        {previewFold.inner[1].validation.join(', ')}.
      </p>
    </>}

    <Prediction
      questions={[{
        key: 'outcome',
        short: draft.target === 'k' ? 'Selected neighbour count' : `Prediction for row ${targetRow}`,
        prompt: draft.target === 'k'
          ? `Which neighbour count will outer fold ${draft.outer + 1}'s inner comparison select?`
          : `What label will outer fold ${draft.outer + 1}'s refitted model give protected row ${targetRow}?`,
        options: draft.target === 'k' ? [[1, '1 neighbour'], [3, '3 neighbours']] : BINARY,
      }]}
      state={state} answerFor={answerFor} applyLabel="Apply and run every fit"
      disabled={Boolean(draftError)} disabledReason={draftError ?? undefined}
      describe={(answer, inputs) => {
        const fold = nestedTrace({ x: inputs.x, y: inputs.y })[inputs.outer];
        const means = fold.candidates.map(candidate => `k=${candidate.k} averages ${round(candidate.mean, 4)}`).join(' and ');
        const target = protectedRowOf(inputs);
        return inputs.target === 'k'
          ? `${means}. ${fold.tie ? 'They tie, so the declared rule takes the smaller count' : 'The higher mean wins'}: ${fold.selectedK}. That mean is selection evidence produced by outer training rows only.`
          : `The inner comparison selected k = ${fold.selectedK} (${means}), and that candidate was refitted on all ${fold.train.length} outer training rows. Row ${target} takes the majority label of its ${fold.selectedK} nearest eligible training row${fold.selectedK > 1 ? 's' : ''}, which gives ${fold.refit.find(row => row.row === target).prediction}. Its own label never entered that fit.`;
      }} />
    <RetiredNotice state={state} />

    {applied && <>
      <h4>Step 2 · Compare candidates inside the outer training rows</h4>
      {applied.candidates.map(candidate => (
        <Table key={candidate.k} id={`inner-k${candidate.k}`}
          caption={`Candidate k = ${candidate.k}: every inner held-out prediction. Inner fold scores ${candidate.innerScores.map(score => round(score, 4)).join(' and ')}, mean ${round(candidate.mean, 6)}.`}
          headings={['inner fold', 'row', 'neighbours used', 'prediction', 'true label', 'counted']}
          rows={candidate.innerDetail.flatMap((detail, index) => detail.rows.map(row => [
            `inner ${index + 1}`, row.row, row.neighbors.map(item => item.id).join(', '), row.prediction, row.truth,
            row.correct ? 'correct' : 'incorrect',
          ]))} />
      ))}
      <h4>Step 3 · Select</h4>
      <div className="cv-readout">
        <dl>
          {applied.candidates.map(candidate => <span key={candidate.k} style={{ display: 'contents' }}>
            <dt>k = {candidate.k} inner mean</dt><dd>{round(candidate.mean, 6)}</dd>
          </span>)}
          <dt>Selected neighbour count</dt><dd>{applied.selectedK}{applied.tie ? ' (tie, smaller count)' : ''}</dd>
          <dt>Selection score</dt><dd>{round(applied.selectionScore, 6)}</dd>
        </dl>
      </div>
      <h4>Step 4 · Refit on the whole outer training room</h4>
      <p className="cv-caption">
        The selected candidate is fitted again from scratch on all {applied.train.length} outer training rows: rows{' '}
        {applied.train.join(', ')}. No inner fit is reused.
      </p>
      <h4>Step 5 · Predict the protected rows</h4>
      <Table id="protected" caption={`The refitted model meets rows it has never seen. Correct on ${applied.assessedCorrect} of ${applied.assessedCount}.`}
        headings={['protected row', 'neighbours used', 'prediction', 'true label', 'counted']}
        rowClass={() => 'is-protected'}
        rows={applied.refit.map(row => [
          row.row, row.neighbors.map(item => item.id).join(', '), row.prediction, row.truth, row.correct ? 'correct' : 'incorrect',
        ])} />
      <div className="cv-readout">
        <dl>
          <dt>Selection score (chose k)</dt><dd>{round(applied.selectionScore, 6)}</dd>
          <dt>Protected result (assessed the procedure)</dt><dd>{applied.assessedCorrect} / {applied.assessedCount} = {round(applied.assessedAccuracy, 6)}</dd>
        </dl>
      </div>
      <p className="cv-caption">
        These two numbers stay in separate rows on purpose. The first was produced by rows that were allowed to choose; the second by rows
        that were not.
      </p>
    </>}
    <p className="cv-limits">
      A constructed fixture, chosen so every one of its fits can be traced by hand. The real experiment elsewhere on this page uses three outer
      folds, three inner folds and six candidates, and its results are recorded separately.
    </p>
  </Investigation>;
}

/* ================================================================== *
 * I4 · §8 — allocate a finite training budget
 * ================================================================== */
const BUDGETS = [10, 30, 90];
const HALVING_PRESETS = {
  base: {
    label: 'Baseline: decide at budget 10',
    inputs: {
      losses: [[0.3, 0.25, 0.24], [0.4, 0.2, 0.1], [0.35, 0.28, 0.27]],
      factor: 3, firstStage: 0,
    },
    note: 'Three candidates. A starts best, B improves fastest, C stays in the middle.',
  },
  later: {
    label: 'Decide at budget 30 instead',
    inputs: { losses: [[0.3, 0.25, 0.24], [0.4, 0.2, 0.1], [0.35, 0.28, 0.27]], factor: 3, firstStage: 1 },
    note: 'The same trajectories, compared one stage later. Everything is paid for, and the ranking is different.',
  },
  lateValueNull: {
    label: 'Null: change a value nobody paid for',
    inputs: { losses: [[0.3, 0.25, 0.24], [0.4, 0.2, 0.02], [0.35, 0.28, 0.27]], factor: 3, firstStage: 0 },
    note: 'B’s budget-90 loss drops to .02. The schedule eliminated B at budget 10, so its decision cannot change.',
  },
  firstValueContrast: {
    label: 'Contrast: change a value it did pay for',
    inputs: { losses: [[0.3, 0.25, 0.24], [0.2, 0.2, 0.1], [0.35, 0.28, 0.27]], factor: 3, firstStage: 0 },
    note: 'B’s first observed loss drops to .20, which the comparison at budget 10 actually reads.',
  },
  nine: {
    label: 'Nine candidates, factor 3',
    inputs: {
      losses: Array.from({ length: 9 }, (_, index) => [50 - index, 40 - index, 30 - index].map(value => value / 100)),
      factor: 3, firstStage: 0,
    },
    note: 'The resource preset from the lesson: stage counts 9, 3, 1.',
  },
};
const HALVING_NAMES = index => String.fromCharCode(65 + index);

export function HalvingLab() {
  const state = useInvestigation(HALVING_PRESETS.base.inputs);
  const [presetKey, setPresetKey] = useState('base');
  const [hindsight, setHindsight] = useState(false);
  const draft = state.draft;
  const build = inputs => successiveHalving({
    candidates: inputs.losses.map((losses, index) => ({ id: HALVING_NAMES(index), losses })),
    budgets: BUDGETS, factor: inputs.factor, firstStage: inputs.firstStage,
  });
  let draftError = null;
  try { build(draft); } catch (error) { draftError = error.message; }
  const answerFor = inputs => {
    const run = build(inputs);
    return { survivor: run.survivor.id, matches: run.hindsight.matchesSurvivor ? 'yes' : 'no' };
  };
  const applied = state.result ? build(state.result.inputs) : null;
  const edit = (candidate, stage, value) => {
    setPresetKey('custom');
    setHindsight(false);
    state.edit({ losses: draft.losses.map((losses, index) => (index === candidate ? losses.map((old, spot) => (spot === stage ? value : old)) : losses)) });
  };
  const place = budget => 44 + 286 * (budget - BUDGETS[0]) / (BUDGETS[BUDGETS.length - 1] - BUDGETS[0]);
  // Zero stays on the axis, and the top is the largest authored loss rounded up
  // to a tenth, so the trajectories use the height instead of hugging one line.
  const maxLoss = Math.max(Math.ceil(Math.max(...draft.losses.flat()) * 10) / 10, 0.1);
  const lift = loss => 150 - 120 * loss / maxLoss;

  return <Investigation id="halving" title="Allocate a finite training budget"
    question="Three or more candidates, three increasing budgets and a survival factor. Predict which candidate survives the schedule, and whether that survivor is also the best at the full budget."
    note="Budget is an abstract cumulative training resource — rows, epochs or steps — and never seconds. You author these trajectories, so you can see all of them; the schedule reports only what it paid to observe."
    onReset={() => { state.reset(); setPresetKey('base'); setHindsight(false); }}>
    <div className="cv-buttons" role="group" aria-label="Declared setups">
      {Object.entries(HALVING_PRESETS).map(([key, preset]) => (
        <button key={key} type="button" aria-pressed={key === presetKey}
          onClick={() => { state.load(preset.inputs); setPresetKey(key); setHindsight(false); }}>{preset.label}</button>
      ))}
    </div>
    <p className="cv-caption">{HALVING_PRESETS[presetKey]?.note ?? 'A custom schedule. Losses may be nonmonotone; they must be finite and nonnegative.'}</p>

    <div className="cv-controls">
      <SelectField label="First comparison at budget" value={draft.firstStage} range="10 or 30"
        options={[[0, '10'], [1, '30']]}
        onChange={next => { setPresetKey('custom'); setHindsight(false); state.edit({ firstStage: Number(next) }); }} />
      <SelectField label="Survival factor" value={draft.factor} range="2 or 3"
        options={[[2, 'keep 1 in 2'], [3, 'keep 1 in 3']]}
        onChange={next => { setPresetKey('custom'); setHindsight(false); state.edit({ factor: Number(next) }); }} />
    </div>

    <h4>Candidate loss trajectories</h4>
    <div className="cv-rows">
      {draft.losses.map((losses, candidate) => (
        <div className="cv-row" key={candidate}>
          <span className="cv-row-id">candidate {HALVING_NAMES(candidate)}</span>
          {losses.map((loss, stage) => (
            <NumberField key={stage} label={`candidate ${HALVING_NAMES(candidate)} loss at budget ${BUDGETS[stage]}`}
              shortLabel={`loss at ${BUDGETS[stage]}`} value={loss} min={0} max={5} step="0.01" decimals={2}
              onChange={next => edit(candidate, stage, next)} />
          ))}
        </div>
      ))}
    </div>
    <div className="cv-buttons">
      <button type="button" disabled={draft.losses.length >= 9}
        onClick={() => { setPresetKey('custom'); setHindsight(false); state.edit({ losses: [...draft.losses, [0.4, 0.35, 0.3]] }); }}>Add a candidate</button>
      <button type="button" disabled={draft.losses.length <= 2}
        onClick={() => { setPresetKey('custom'); setHindsight(false); state.edit({ losses: draft.losses.slice(0, -1) }); }}>Remove the last candidate</button>
    </div>
    {draftError && <p className="cv-field-error" role="alert" data-cv-error>{draftError}</p>}

    <Prediction
      questions={[
        {
          key: 'survivor', short: 'Survivor',
          prompt: 'Which candidate is selected among those assessed at the largest budget? Ties select the earliest candidate.',
          options: draft.losses.map((_, index) => [HALVING_NAMES(index), `candidate ${HALVING_NAMES(index)}`]),
        },
        {
          key: 'matches', short: 'Matches the full-budget best',
          legend: 'And the second half of the prediction.',
          prompt: 'Will that survivor also have the lowest loss at budget 90?',
          options: [['yes', 'yes'], ['no', 'no']],
        },
      ]}
      state={state} answerFor={answerFor} applyLabel="Run the schedule"
      disabled={Boolean(draftError)} disabledReason={draftError ?? undefined}
      describe={(answer, inputs) => {
        const run = build(inputs);
        const first = run.stages[0];
        return `At budget ${first.budget} the schedule observed ${first.observed.map(row => `${row.id} ${round(row.loss, 3)}`).join(', ')} and kept ${first.keep} of ${first.assessed}. ${run.stages.at(-1).assessed} candidate(s) reached budget ${BUDGETS[BUDGETS.length - 1]}; candidate ${run.survivor.id} was selected among them. ${run.hindsight.matchesSurvivor ? 'It attains the lowest loss at the full budget, possibly tied with another candidate.' : 'It is not the lowest loss at the full budget. Which candidate is, and by how much, is information the schedule never paid for: Reveal hindsight names it.'}`;
      }} />
    <RetiredNotice state={state} />

    {applied && <>
      <h4>What the schedule paid to observe</h4>
      {applied.stages.map(stage => (
        <Table key={stage.stage} id={`stage-${stage.stage}`}
          caption={`Budget ${stage.budget}: ${stage.assessed} candidate${stage.assessed > 1 ? 's' : ''} assessed, ${stage.keep} retained. Nominal cost ${stage.nominalStageCost} units.`}
          headings={['rank', 'candidate', 'observed loss', 'outcome']} numeric={[0, 2]}
          rows={stage.ranked.map((row, position) => [
            position + 1, `candidate ${row.id}`, round(row.loss, 3),
            stage.stage === BUDGETS.length - 1
              ? row.index === applied.survivor.index ? 'selected' : 'assessed at final budget'
              : stage.survivors.includes(row.index) ? 'continues' : 'eliminated',
          ])} />
      ))}
      <svg viewBox="0 0 360 186" role="img" aria-label={`Loss against training budget. Each candidate is drawn only as far as the schedule paid to observe it. ${applied.stages.at(-1).assessed} candidates reach budget ${BUDGETS[BUDGETS.length - 1]}; candidate ${applied.survivor.id} is selected among them. Earlier eliminated candidates stop at their last assessed budget.`}>
        <line className="cv-axis" x1="34" y1="150" x2="344" y2="150" />
        <line className="cv-axis" x1="34" y1="18" x2="34" y2="150" />
        {BUDGETS.map(budget => <g key={budget}>
          <line className="cv-grid" x1={place(budget)} y1="18" x2={place(budget)} y2="150" />
          <text x={place(budget)} y="166" textAnchor="middle">{budget}</text>
        </g>)}
        <text x="190" y="182" textAnchor="middle">training budget · abstract units</text>
        {[0, 0.5, 1].map(share => <g key={share}>
          <line className="cv-grid" x1="34" y1={lift(maxLoss * share)} x2="344" y2={lift(maxLoss * share)} />
          <text x="30" y={lift(maxLoss * share) + 5} textAnchor="end">{round(maxLoss * share, 3)}</text>
        </g>)}
        <text x="8" y="14">loss</text>
        {state.result.inputs.losses.map((losses, index) => {
          const upTo = applied.observedUpTo[index];
          if (upTo < applied.firstStage) return null;
          const points = [];
          for (let stage = applied.firstStage; stage <= upTo; stage += 1) points.push(`${place(BUDGETS[stage])},${lift(losses[stage])}`);
          const survivor = applied.survivor.index === index;
          return <g key={index}>
            <title>{`Candidate ${HALVING_NAMES(index)}: last assessed at budget ${BUDGETS[upTo]}, loss ${round(losses[upTo], 3)}${survivor ? ', selected' : ''}`}</title>
            <polyline className={`cv-curve ${survivor ? 'is-first' : 'is-second'}`} points={points.join(' ')} fill="none" />
            {points.map((point, position) => {
              const [cx, cy] = point.split(',').map(Number);
              return survivor
                ? <circle key={position} className="cv-mark" cx={cx} cy={cy} r="4" />
                : <rect key={position} className="cv-mark is-hollow" x={cx - 4} y={cy - 4} width="8" height="8" />;
            })}
          </g>;
        })}
      </svg>
      <p className="cv-caption">
        Solid line with filled dots: the selected candidate. Dashed lines with hollow squares: other candidates, drawn through their last
        assessed budget. Several may reach the final budget; an earlier elimination ends a line because later losses were not observed.
      </p>
      <Table id="schedule-endpoints" caption="Curve endpoints by candidate. Nearby or coincident endpoints share space in the plot; their identities and exact values stay readable here."
        headings={['candidate', 'last assessed budget', 'last observed loss', 'selected']} numeric={[1, 2]}
        rows={state.result.inputs.losses.map((losses, index) => [HALVING_NAMES(index), BUDGETS[applied.observedUpTo[index]], round(losses[applied.observedUpTo[index]], 3), index === applied.survivor.index ? 'yes' : 'no'])} />
      <div className="cv-readout">
        <dl>
          <dt>Survivor</dt><dd>candidate {applied.survivor.id}</dd>
          <dt>Nominal resource, retraining from scratch</dt><dd>{applied.cost.nominal} units</dd>
          <dt>Giving every candidate the largest budget</dt><dd>{applied.cost.fullAllocation} units</dd>
          <dt>Incremental resource, if training genuinely resumes</dt><dd>{applied.cost.resumable} units</dd>
        </dl>
      </div>
      <p className="cv-caption">
        The resumable figure assumes a real saved-state continuation from the previous budget. Do not assume a library supports that merely
        because its schedule has increasing budgets. A three-fold evaluation would repeat the corresponding fits, and refitting the winner
        still needs a budget of its own.
      </p>
      <div className="cv-buttons">
        <button type="button" aria-pressed={hindsight} onClick={() => setHindsight(current => !current)} data-cv-hindsight-toggle>
          {hindsight ? 'Hide hindsight' : 'Reveal hindsight'}
        </button>
      </div>
      {hindsight && <div className="cv-readout" data-cv-hindsight>
        <dl>
          <dt>Lowest loss at budget {BUDGETS[BUDGETS.length - 1]}</dt><dd>candidate {applied.hindsight.bestId} at {round(applied.hindsight.bestLoss, 3)}</dd>
          <dt>Survivor's loss at that budget</dt><dd>{round(applied.hindsight.survivorFinalLoss, 3)}</dd>
          <dt>Hindsight regret</dt><dd>{round(applied.hindsight.regret, 3)}</dd>
        </dl>
        <p className="cv-caption">
          This comparison was not available when the schedule made its decision. It is stated here as hindsight in a constructed environment,
          never as a measured property of any real search.
        </p>
      </div>}
    </>}
    <p className="cv-limits">
      The risk this exposes is early ranking, not nonmonotone curves: in the baseline every trajectory improves at every stage and the
      elimination is still wrong. No speedup over any real framework is claimed here.
    </p>
  </Investigation>;
}
