import { useState } from 'react';
import {
  degreeComparison, finiteExperiment, fixtures, limits, predictionSpread, spreadComparison,
} from '../../data/bias-variance-models';
import {
  Field, Investigation, NumberField, Plot, Prediction, Table, Waterfall,
  round, signed, useInvestigation,
} from './BiasVarianceShared.jsx';
import './bias-variance-labs.css';

/* ============================================================ §1 · I1 */

/** Suggested setups. Each fills the draft only: the applied state stays where it
 * is until Apply, so the direction being predicted always has something to be a
 * direction from. */
const spreadSetups = [
  { key: 'baseline', label: 'Baseline: predictions 8, 10, 12', predictions: [8, 10, 12], trueMean: 10, noiseSpread: 1 },
  { key: 'inward', label: 'Move the dots inward: 9, 10, 11', predictions: [9, 10, 11], trueMean: 10, noiseSpread: 1 },
  { key: 'outward', label: 'Move them outward: 6, 10, 14', predictions: [6, 10, 14], trueMean: 10, noiseSpread: 1 },
  { key: 'collapsed', label: 'Same average, no spread: 10, 10, 10', predictions: [10, 10, 10], trueMean: 10, noiseSpread: 1 },
  { key: 'shifted', label: 'Same spread, shifted up: 9, 11, 13', predictions: [9, 11, 13], trueMean: 10, noiseSpread: 1 },
  { key: 'constant', label: 'Procedure B: 9, 9, 9', predictions: [9, 9, 9], trueMean: 10, noiseSpread: 1 },
  { key: 'permuted', label: 'Null: the same dots reordered, 12, 10, 8', predictions: [12, 10, 8], trueMean: 10, noiseSpread: 1 },
  { key: 'translated', label: 'Null: move everything up 2', predictions: [10, 12, 14], trueMean: 12, noiseSpread: 1 },
];
const spreadBaseline = { predictions: [8, 10, 12], trueMean: 10, noiseSpread: 1 };
const dotNames = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6'];

function Ruler({ title, describe, domain, rows, marks, bracket, height }) {
  const width = 320;
  // Wide enough for a row label that carries its own value, so no number has to
  // float beside a dot where a reference line would run straight through it.
  const left = 72;
  const right = 306;
  // A ruler that carries a bracket beneath its axis needs room for two extra
  // rows: the bracket itself and its printed value.
  const axisY = height - (bracket ? 58 : 42);
  const place = value => left + (right - left) * (value - domain[0]) / (domain[1] - domain[0]);
  const ticks = Array.from({ length: 5 }, (_, index) => domain[0] + (domain[1] - domain[0]) * index / 4);
  return <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={describe}>
    <title>{title}</title>
    {ticks.map(value => <g key={value}>
      <line className="bv-grid" x1={place(value)} x2={place(value)} y1={16} y2={axisY} />
      <text x={place(value)} y={axisY + 16} textAnchor="middle">{round(value, 2)}</text>
    </g>)}
    <line className="bv-axis" x1={left} x2={right} y1={axisY} y2={axisY} />
    {rows.map(row => <g key={row.name}>
      <text x={4} y={row.y + 4}>{row.name} {round(row.value, 2)}</text>
      <line className="bv-grid" x1={left} x2={right} y1={row.y} y2={row.y} />
      <circle className={row.className} cx={place(row.value)} cy={row.y} r="5" />
    </g>)}
    {marks.map(mark => <g key={mark.name}>
      <line className={mark.className} x1={place(mark.value)} x2={place(mark.value)} y1={16} y2={axisY} />
      {/* The baseline leaves room for the glyph's full ascent: any higher and
          the symbol's top edge leaves the viewBox once the drawing is scaled. */}
      <text x={place(mark.value) + mark.side * 4} y={12} textAnchor={mark.side < 0 ? 'end' : 'start'}>{mark.symbol}</text>
    </g>)}
    {bracket && <g>
      <line className="bv-axis" x1={place(bracket.from)} x2={place(bracket.to)} y1={axisY + 24} y2={axisY + 24} />
      <line className="bv-axis" x1={place(bracket.from)} x2={place(bracket.from)} y1={axisY + 19} y2={axisY + 29} />
      <line className="bv-axis" x1={place(bracket.to)} x2={place(bracket.to)} y1={axisY + 19} y2={axisY + 29} />
      <text x={(place(bracket.from) + place(bracket.to)) / 2} y={axisY + 40} textAnchor="middle">{bracket.label}</text>
    </g>}
  </svg>;
}

/** Move the predictions; watch which term moves. */
export function SpreadLab() {
  const state = useInvestigation(spreadBaseline);
  const draft = state.draft;
  const shown = predictionSpread(draft.predictions, draft.trueMean, draft.noiseSpread);
  const applied = predictionSpread(state.active.predictions, state.active.trueMean, state.active.noiseSpread);
  const comparison = state.result
    ? spreadComparison(state.previous, state.active)
    : null;
  const answerFor = (proposed, current) => {
    const result = spreadComparison(current, proposed);
    return { outcome: result.outcome, value: result.after.total };
  };
  const editPrediction = (index, value) => state.edit({
    predictions: draft.predictions.map((old, position) => (position === index ? value : old)),
  });
  const domain = [
    Math.min(shown.domain[0], applied.domain[0]),
    Math.max(shown.domain[1], applied.domain[1]),
  ];
  const changedTerms = comparison
    ? ['squaredBias', 'variance', 'noiseVariance'].filter(term => comparison.changed[term])
    : [];
  const termNames = { squaredBias: 'squared bias', variance: 'prediction variance', noiseVariance: 'target noise' };
  return <Investigation
    title="Move one prediction; watch which term moves"
    question="Three equally likely fitted predictions at one input, a known target mean, and fresh outcomes at that same input. Change the dots, then record whether total expected squared error will fall, stay exactly where it is, or rise — before the calculation appears."
    note="The two rulers are different kinds of spread. The upper one holds predictions from different training samples; the lower one holds outcomes of the target at this one input. An outcome dot is never a fitted model."
    onReset={state.reset}>

    <div className="bv-controls">
      {draft.predictions.map((value, index) => (
        <NumberField key={dotNames[index]} label={`Prediction ${dotNames[index]}`} value={value}
          min={limits.prediction.minimum} max={limits.prediction.maximum} step="0.25" decimals={2}
          onChange={next => editPrediction(index, next)} />
      ))}
      <NumberField label="True target mean f" value={draft.trueMean}
        min={limits.trueMean.minimum} max={limits.trueMean.maximum} step="0.25" decimals={2}
        onChange={trueMean => state.edit({ trueMean })} />
      <NumberField label="Fresh-outcome spread σ" value={draft.noiseSpread}
        min={limits.noiseSpread.minimum} max={limits.noiseSpread.maximum} step="0.25" decimals={2}
        onChange={noiseSpread => state.edit({ noiseSpread })} />
    </div>
    <div className="bv-buttons">
      {spreadSetups.map(setup => (
        <button key={setup.key} type="button" onClick={() => state.suggest({
          predictions: setup.predictions, trueMean: setup.trueMean, noiseSpread: setup.noiseSpread,
        })}>{setup.label}</button>
      ))}
    </div>
    <p className="bv-state-strip">
      <span>applied: <b>[{state.active.predictions.map(value => round(value, 2)).join(', ')}]</b>, f = <b>{round(state.active.trueMean, 2)}</b>, σ = <b>{round(state.active.noiseSpread, 2)}</b></span>
      <span className={state.pending ? 'is-draft' : undefined}>draft: <b>[{draft.predictions.map(value => round(value, 2)).join(', ')}]</b>, f = <b>{round(draft.trueMean, 2)}</b>, σ = <b>{round(draft.noiseSpread, 2)}</b></span>
    </p>
    <p className="bv-caption">
      Ranges: every prediction and the true mean from {round(limits.prediction.minimum)} to {round(limits.prediction.maximum)},
      the spread from {round(limits.noiseSpread.minimum)} to {round(limits.noiseSpread.maximum)}. Steps of 0.25.
      A suggested setup fills the fields only; it never selects the outcome, and it does not apply itself.
    </p>

    <div className="bv-panels is-pair">
      <div className="bv-panel">
        <h4>Predictions after different training samples</h4>
        <Ruler
          title="Prediction ruler"
          describe={`Three prediction rows on one scale from ${round(domain[0], 2)} to ${round(domain[1], 2)}. ${draft.predictions.map((value, index) => `${dotNames[index]} sits at ${round(value, 2)}`).join('; ')}. Their average m is ${round(shown.averagePrediction, 6)} and the true target mean f is ${round(draft.trueMean, 2)}. The bracket beneath the axis measures the signed offset m minus f, which is ${round(shown.signedOffset, 6)}.`}
          domain={domain} height={164}
          rows={draft.predictions.map((value, index) => ({
            name: dotNames[index], value, y: 26 + index * 22, className: 'bv-mark is-world',
          }))}
          /* The truth is drawn first and the dashed mean over it, so that when the
             two coincide — which is the default and most of the suggested setups —
             the gold shows through the gaps instead of one line hiding the other. */
          marks={[
            { name: 'truth', value: draft.trueMean, symbol: 'f', side: 1, className: 'bv-curve is-truth' },
            { name: 'mean', value: shown.averagePrediction, symbol: 'm', side: -1, className: 'bv-curve is-average' },
          ]}
          bracket={{ from: draft.trueMean, to: shown.averagePrediction, label: `m − f = ${signed(shown.signedOffset, 4)}` }} />
        <p>
          Dashed green line <strong>m</strong>: the average of the three predictions, {round(shown.averagePrediction, 6)}.
          Solid gold line <strong>f</strong>: the true target mean, {round(draft.trueMean, 2)}.
          {shown.signedOffset === 0
            ? ' Here the two coincide, which is why you can see gold through the gaps in the green dashes: the average prediction is exactly right, the bracket has zero length, and the squared bias is exactly zero. The dots are still spread, and that spread is a separate term.'
            : ' The bracket between them is the signed distance; squaring it gives the bias term, so a negative offset and a positive one of the same size cost the same.'}
        </p>
      </div>
      <div className="bv-panel">
        <h4>Fresh outcomes at the same input</h4>
        <Ruler
          title="Outcome ruler"
          describe={`Two equally likely fresh outcomes on the same scale: ${shown.outcomes.map(value => round(value, 2)).join(' and ')}, constructed as the true mean ${round(draft.trueMean, 2)} plus or minus ${round(draft.noiseSpread, 2)}. Their variance is ${round(shown.noiseVariance, 6)}. These are outcomes of the target, not fitted models.`}
          domain={domain} height={116}
          rows={shown.outcomes.map((value, index) => ({
            name: index === 0 ? 'y⁻' : 'y⁺', value, y: 26 + index * 22, className: 'bv-observation',
          }))}
          marks={[{ name: 'truth', value: draft.trueMean, symbol: 'f', side: 1, className: 'bv-curve is-truth' }]} />
        <p>
          Two equally likely outcomes at f ± σ, so their mean is f and their variance is σ² = {round(shown.noiseVariance, 6)}.
          Nothing was fitted to produce them. Increasing σ cannot change the prediction dots above.
        </p>
      </div>
    </div>

    <Prediction
      prompt={`Applying the draft would change the three predictions to [${draft.predictions.map(value => round(value, 2)).join(', ')}], f to ${round(draft.trueMean, 2)} and σ to ${round(draft.noiseSpread, 2)}. Against the applied state, what happens to total expected squared error? A tie means a difference no larger than 10⁻¹⁰ × max(1, |applied total|).`}
      options={[['decrease', 'It falls'], ['unchanged', 'Unchanged within tolerance'], ['increase', 'It rises']]}
      state={state} answerFor={answerFor}
      numeric={{ label: 'Optional: the new total expected squared error', name: 'the new total', tolerance: 1e-4, digits: 9 }}
      committed={() => (comparison
        ? `[${comparison.before.predictions.map(value => round(value, 2)).join(', ')}], f = ${round(comparison.before.trueMean, 2)}, σ = ${round(comparison.before.noiseSpread, 2)} → [${comparison.after.predictions.map(value => round(value, 2)).join(', ')}], f = ${round(comparison.after.trueMean, 2)}, σ = ${round(comparison.after.noiseSpread, 2)}`
        : '')}
      describe={comparison
        ? `Total moved from ${round(comparison.before.total, 9)} to ${round(comparison.after.total, 9)}, a change of ${signed(comparison.difference, 9)}.`
        : undefined} />

    {state.result && comparison && <>
      <Waterfall
        caption={`The applied state: squared bias + prediction variance + target noise = total expected squared error.`}
        describe={`An additive bar. Squared bias is ${round(applied.squaredBias, 9)}, prediction variance ${round(applied.variance, 9)} and target noise ${round(applied.noiseVariance, 9)}; they add to ${round(applied.total, 9)}.`}
        parts={[
          { name: 'squared bias (m − f)²', value: applied.squaredBias, className: 'is-bias' },
          { name: 'prediction variance', value: applied.variance, className: 'is-variance' },
          { name: 'target noise σ²', value: applied.noiseVariance, className: 'is-noise' },
        ]}
        totalLabel="total expected squared error" digits={9} />

      <Table caption="What moved between the previous applied state and this one. Every term is nonnegative; only the total is graded."
        headings={['term', 'before', 'now', 'change']}
        rows={[
          ['average prediction m', round(comparison.before.averagePrediction, 9), round(comparison.after.averagePrediction, 9), signed(comparison.after.averagePrediction - comparison.before.averagePrediction, 9)],
          ['signed offset m − f', signed(comparison.before.signedOffset, 9), signed(comparison.after.signedOffset, 9), signed(comparison.after.signedOffset - comparison.before.signedOffset, 9)],
          ['squared bias', round(comparison.before.squaredBias, 9), round(comparison.after.squaredBias, 9), signed(comparison.after.squaredBias - comparison.before.squaredBias, 9)],
          ['prediction variance', round(comparison.before.variance, 9), round(comparison.after.variance, 9), signed(comparison.after.variance - comparison.before.variance, 9)],
          ['target noise σ²', round(comparison.before.noiseVariance, 9), round(comparison.after.noiseVariance, 9), signed(comparison.after.noiseVariance - comparison.before.noiseVariance, 9)],
          ['total', round(comparison.before.total, 9), round(comparison.after.total, 9), signed(comparison.difference, 9)],
        ]}
        rowClass={index => {
          const term = { 2: 'squaredBias', 3: 'variance', 4: 'noiseVariance' }[index];
          return term && changedTerms.includes(term) ? 'is-selected' : undefined;
        }} />

      <p className="bv-readout" aria-live="polite">
        {changedTerms.length === 0
          ? 'Squared bias, prediction variance and target noise are unchanged within the comparison tolerance. The average prediction itself can still move with the target mean, leaving their offset unchanged.'
          : `Changed operand${changedTerms.length === 1 ? '' : 's'}: ${changedTerms.map(term => termNames[term]).join(', ')}.`}
        {' '}Squared bias is {round(applied.squaredBias, 9)}, prediction variance {round(applied.variance, 9)} and target
        noise {round(applied.noiseVariance, 9)}, totalling {round(applied.total, 9)}.
        {applied.agrees
          ? ` Averaging all ${applied.pairs.length} equally weighted prediction/outcome pairs gives the same ${round(applied.pairAverage, 9)}, which is what the decomposition claims.`
          : ' The two numerical routes disagree beyond the tolerance; inspect the calculation before interpreting it.'}
        {comparison.changed.variance && !comparison.changed.squaredBias && !comparison.changed.noiseVariance
          && ' Squared bias and target noise did not move beyond the tolerance, so the change comes from prediction variance.'}
        {comparison.changed.squaredBias && !comparison.changed.variance && !comparison.changed.noiseVariance
          && ' Prediction variance and target noise did not move beyond the tolerance, so the change comes from squared bias.'}
      </p>

      <details>
        <summary>Enumerate every prediction/outcome pair</summary>
        <Table caption={`All ${applied.pairs.length} equally weighted pairs. Each has probability ${round(applied.pairs[0].probability, 6)}; their average squared error is the total above.`}
          headings={['prediction', 'fresh outcome', 'probability', 'squared error']}
          rows={applied.pairs.map(pair => [
            `${dotNames[pair.predictionIndex]} = ${round(pair.prediction, 2)}`,
            round(pair.outcome, 2), round(pair.probability, 6), round(pair.squaredError, 9),
          ])} />
        <p className="bv-caption">
          This is the same number by a different route: average the squared error over every equally likely
          (prediction, outcome) pair, rather than adding the three terms. Two setups can have the same average prediction
          and different totals, which is the transfer task in the prose.
        </p>
      </details>
    </>}
  </Investigation>;
}

/* ============================================================ §3 · I2 */

const designs = {
  three: { label: 'three inputs at −1, 0, 1', trainX: fixtures.threeInputs },
  five: { label: 'five inputs at −1, −0.5, 0, 0.5, 1', trainX: fixtures.fiveInputs },
};
const degreeNames = ['constant', 'line', 'quadratic'];
const worldsBaseline = { curvature: 1, sigma: 0.5, probe: 0.5, design: 'three', reference: 1, candidate: 2 };
const worldsPresets = [
  { key: 'default', label: 'Baseline: curvature 1, line against quadratic at x = 0.5', ...worldsBaseline },
  { key: 'flat', label: 'Flat truth: curvature 0', ...worldsBaseline, curvature: 0 },
  { key: 'coincide', label: 'Null: probe 0, constant against line', ...worldsBaseline, probe: 0, reference: 0, candidate: 1 },
  { key: 'noiseless', label: 'Null: no noise and no curvature', ...worldsBaseline, curvature: 0, sigma: 0 },
  { key: 'louder', label: 'Louder noise: σ = 1', ...worldsBaseline, sigma: 1 },
  { key: 'fiveInputs', label: 'Five training inputs, constant against quadratic', ...worldsBaseline, design: 'five', reference: 0, candidate: 2 },
  { key: 'negative', label: 'Negative curvature: c = −1', ...worldsBaseline, curvature: -1 },
  { key: 'edge', label: 'Edge probe: x = −1.5', ...worldsBaseline, probe: -1.5 },
];

function WorldsPanel({ role, record, range, selected, probe }) {
  const domain = [-1.5, 1.5];
  return <div className="bv-panel">
    <h4>{role}: degree {record.degree}, the {degreeNames[record.degree]}</h4>
    <Plot width={300} height={210} domain={domain} range={range}
      padding={{ left: 40, right: 12, top: 14, bottom: 32 }}
      ticks={[-1.5, -0.75, 0, 0.75, 1.5]}
      describe={`${record.worldCount} fitted ${degreeNames[record.degree]} curves over inputs from −1.5 to 1.5, drawn from every equally likely training dataset. The true mean curve and the average fitted curve are drawn solid and dashed. At the probe ${round(record.probe, 4)} the true mean is ${round(record.probeTruth, 6)}, the average prediction is ${round(record.meanPrediction, 6)}, the prediction variance is ${round(record.variance, 9)} and the expected squared error is ${round(record.expectedError, 9)}. World ${selected + 1} is highlighted; its prediction at the probe is ${round(record.probePredictions[selected], 6)}.`}>
      {(scaleX, scaleY) => <>
        {record.curves.map((values, index) => (
          <polyline key={index} className={`bv-curve ${index === selected ? 'is-selected-world' : 'is-world'}`}
            points={record.grid.map((x, position) => `${scaleX(x).toFixed(2)},${scaleY(values[position]).toFixed(2)}`).join(' ')} />
        ))}
        <polyline className="bv-curve is-truth"
          points={record.grid.map((x, position) => `${scaleX(x).toFixed(2)},${scaleY(record.truthCurve[position]).toFixed(2)}`).join(' ')} />
        <polyline className="bv-curve is-average"
          points={record.grid.map((x, position) => `${scaleX(x).toFixed(2)},${scaleY(record.averageCurve[position]).toFixed(2)}`).join(' ')} />
        <line className="bv-stem is-probe" x1={scaleX(probe)} x2={scaleX(probe)} y1={scaleY(range[1])} y2={scaleY(range[0])} />
        {record.probePredictions.map((value, index) => (
          <circle key={index} className={index === selected ? 'bv-observation' : 'bv-mark is-world'}
            cx={scaleX(probe)} cy={scaleY(value)} r={index === selected ? 4 : 2.4} />
        ))}
        <circle className="bv-mark is-hollow" cx={scaleX(probe)} cy={scaleY(record.probeTruth)} r="6" />
        <rect className="bv-mark" x={scaleX(probe) - 4} y={scaleY(record.meanPrediction) - 4} width="8" height="8" />
        {record.trainX.map(value => (
          <line key={value} className="bv-grid" x1={scaleX(value)} x2={scaleX(value)} y1={scaleY(range[0])} y2={scaleY(range[0]) - 8} />
        ))}
      </>}
    </Plot>
    <p>
      Faint lines: one fitted curve per training world, with world {selected + 1} — the one chosen in the selector below —
      picked out in pale white. Solid gold: the true mean 1 + x + {round(record.curvature, 2)}x². Dashed green: the
      average fitted curve.
      {record.gridDecomposition.every(entry => entry.squaredBias <= 1e-12)
        ? ' Those two run together everywhere here, so the gold shows through the gaps in the green dashes: this fit is unbiased at every input, and all that is left is spread.'
        : ' Where the green leaves the gold, the average fitted curve misses the truth, and that gap squared is the bias term.'}
      {' '}The dotted vertical is the probe; the hollow circle on it is the true mean there and the filled square is the
      average prediction. Ticks along the bottom axis mark the fixed training inputs.
    </p>
  </div>;
}

/** Enumerate every possible tiny training set. */
export function WorldsLab() {
  const state = useInvestigation(worldsBaseline);
  const [selected, setSelected] = useState(0);
  const draft = state.draft;
  const active = state.active;
  const settings = {
    trainX: designs[active.design].trainX,
    curvature: active.curvature, sigma: active.sigma, probe: active.probe,
  };
  const reference = finiteExperiment({ ...settings, degree: active.reference });
  const candidate = finiteExperiment({ ...settings, degree: active.candidate });
  const comparison = degreeComparison(settings, active.reference, active.candidate);
  const range = [
    Math.min(reference.valueRange[0], candidate.valueRange[0]),
    Math.max(reference.valueRange[1], candidate.valueRange[1]),
  ];
  const world = Math.min(selected, reference.worldCount - 1);
  const answerFor = proposed => {
    const result = degreeComparison({
      trainX: designs[proposed.design].trainX,
      curvature: proposed.curvature, sigma: proposed.sigma, probe: proposed.probe,
    }, proposed.reference, proposed.candidate);
    return { outcome: result.outcome, value: result.candidate.expectedError };
  };
  const bothDesigns = Object.entries(designs).flatMap(([key, design]) => [0, 1, 2].map(degree => {
    const record = finiteExperiment({
      trainX: design.trainX, curvature: active.curvature, sigma: active.sigma,
      probe: active.probe, degree, grid: [active.probe],
    });
    return { design: key, label: design.label, degree, record };
  }));
  const describeRow = record => [
    round(record.meanPrediction, 9), round(record.squaredBias, 9), round(record.variance, 9),
    round(record.noiseVariance, 9), round(record.expectedError, 9),
  ];
  return <Investigation
    title="Enumerate every possible tiny training set"
    question="Fixed training inputs, a true curve you choose, and outcomes that are the truth plus or minus σ at each input. Every combination of signs is one equally likely training dataset, and all of them are fitted. Record whether the candidate degree will have lower, identical or higher expected squared error at the probe before applying."
    note="This experiment varies the training outcomes while holding the design points fixed. It is not a bootstrap, not a resample of rows and not a simulation of random input locations."
    onReset={() => { state.reset(); setSelected(0); }}>

    <div className="bv-buttons">
      {worldsPresets.map(preset => (
        <button key={preset.key} type="button" onClick={() => {
          state.load({
            curvature: preset.curvature, sigma: preset.sigma, probe: preset.probe,
            design: preset.design, reference: preset.reference, candidate: preset.candidate,
          });
          setSelected(0);
        }}>{preset.label}</button>
      ))}
    </div>
    <div className="bv-controls is-wide">
      <NumberField label="True curvature c" value={draft.curvature}
        min={limits.curvature.minimum} max={limits.curvature.maximum} step="0.05" decimals={2}
        onChange={curvature => state.edit({ curvature })} />
      <NumberField label="Training noise σ" value={draft.sigma}
        min={limits.sigma.minimum} max={limits.sigma.maximum} step="0.05" decimals={2}
        onChange={sigma => state.edit({ sigma })} />
      <NumberField label="Probe input x" value={draft.probe}
        min={limits.probe.minimum} max={limits.probe.maximum} step="0.05" decimals={2}
        onChange={probe => state.edit({ probe })} />
      <Field label="Training design">
        <select value={draft.design} onChange={event => { state.edit({ design: event.target.value }); setSelected(0); }}>
          {Object.entries(designs).map(([key, design]) => (
            <option key={key} value={key}>{design.label} · {2 ** design.trainX.length} worlds</option>
          ))}
        </select>
      </Field>
      <Field label="Reference degree">
        <select value={draft.reference} onChange={event => state.edit({ reference: Number(event.target.value) })}>
          {[0, 1, 2].map(degree => <option key={degree} value={degree}>{degree} · {degreeNames[degree]}</option>)}
        </select>
      </Field>
      <Field label="Candidate degree">
        <select value={draft.candidate} onChange={event => state.edit({ candidate: Number(event.target.value) })}>
          {[0, 1, 2].map(degree => <option key={degree} value={degree}>{degree} · {degreeNames[degree]}</option>)}
        </select>
      </Field>
    </div>
    <p className="bv-caption">
      Ranges: curvature from {round(limits.curvature.minimum)} to {round(limits.curvature.maximum)}, noise from
      {' '}{round(limits.sigma.minimum)} to {round(limits.sigma.maximum)}, probe from {round(limits.probe.minimum)} to
      {' '}{round(limits.probe.maximum)}, all in steps of 0.05; degrees 0 to 2.
      Switching to five inputs adds two particular measurement locations at −0.5 and 0.5 and changes the design; it is not
      an extra independent sample from a population, and it changes which fits are possible as well as how many worlds
      there are.
    </p>
    <p className="bv-state-strip">
      <span>applied: c = <b>{round(active.curvature, 2)}</b>, σ = <b>{round(active.sigma, 2)}</b>, x = <b>{round(active.probe, 2)}</b>, <b>{designs[active.design].label}</b>, reference <b>{degreeNames[active.reference]}</b>, candidate <b>{degreeNames[active.candidate]}</b></span>
      <span className={state.pending ? 'is-draft' : undefined}>draft: c = <b>{round(draft.curvature, 2)}</b>, σ = <b>{round(draft.sigma, 2)}</b>, x = <b>{round(draft.probe, 2)}</b>, <b>{designs[draft.design].label}</b>, reference <b>{degreeNames[draft.reference]}</b>, candidate <b>{degreeNames[draft.candidate]}</b></span>
    </p>

    <Prediction
      prompt={`With curvature ${round(draft.curvature, 2)}, noise ${round(draft.sigma, 2)} and ${2 ** designs[draft.design].trainX.length} equally likely training worlds, how will the candidate ${degreeNames[draft.candidate]}'s expected squared error at x = ${round(draft.probe, 2)} compare with the reference ${degreeNames[draft.reference]}'s? A tie means a difference no larger than 10⁻¹⁰ × max(1, |reference error|).`}
      options={[['lower', 'Candidate lower'], ['same', 'Equal within tolerance'], ['higher', 'Candidate higher']]}
      state={state} answerFor={answerFor}
      numeric={{ label: 'Optional: the candidate’s expected squared error', name: 'the candidate’s expected error', tolerance: 1e-4, digits: 9 }}
      committed={() => `c = ${round(active.curvature, 2)}, σ = ${round(active.sigma, 2)}, x = ${round(active.probe, 2)}, ${designs[active.design].label}, ${degreeNames[active.reference]} against ${degreeNames[active.candidate]}`}
      describe={`Reference ${round(comparison.reference.expectedError, 9)} against candidate ${round(comparison.candidate.expectedError, 9)}, a difference of ${signed(comparison.difference, 9)}.`} />

    {state.result && <>
      <div className="bv-panels is-pair">
        <WorldsPanel role="Reference" record={reference} range={range} selected={world} probe={active.probe} />
        <WorldsPanel role="Candidate" record={candidate} range={range} selected={world} probe={active.probe} />
      </div>
      <p className="bv-caption">
        Both panels share one vertical scale, {round(range[0], 3)} to {round(range[1], 3)}, so the two bundles are
        comparable. Nothing is clipped: the range follows the actual fitted curves.
        There are {reference.worldCount} training worlds, each with probability {round(1 / reference.worldCount, 6)}.
      </p>

      <div className="bv-controls">
        <Field label="Inspect one training world">
          <select value={world} onChange={event => setSelected(Number(event.target.value))}>
            {reference.worlds.map(entry => (
              <option key={entry.index} value={entry.index}>
                world {entry.index + 1}: targets {entry.targets.map(value => round(value, 2)).join(', ')}
              </option>
            ))}
          </select>
        </Field>
      </div>
      <p className="bv-caption">
        World {world + 1} has observed targets {reference.worlds[world].targets.map(value => round(value, 4)).join(', ')} at
        inputs {reference.trainX.map(value => round(value, 2)).join(', ')}. Its fitted curve is drawn pale in both panels and
        its probe prediction is marked. Selecting a world is inspection only: the average and the error terms are taken over
        all {reference.worldCount} of them and do not change with this control.
      </p>

      <Table caption={`Expected squared error at x = ${round(active.probe, 2)}, where the true mean is ${round(reference.probeTruth, 6)}`}
        headings={['fit', 'mean prediction', 'squared bias', 'prediction variance', 'target noise σ²', 'expected error']}
        rows={[
          [`reference · ${degreeNames[active.reference]}`, ...describeRow(reference)],
          [`candidate · ${degreeNames[active.candidate]}`, ...describeRow(candidate)],
        ]}
        rowClass={index => (index === 1 ? 'is-selected' : undefined)} />

      <p className="bv-readout" aria-live="polite">
        {comparison.outcome === 'same'
          ? `The two expected errors agree within ${round(limits.tolerance * Math.max(1, Math.abs(reference.expectedError)), 12)}, the stated scaled tolerance.`
          : `The candidate's expected error is ${comparison.outcome} by ${round(Math.abs(comparison.difference), 9)}.`}
        {comparison.identicalPredictions
          ? ' Every training world gives predictions within 10⁻¹² for these two degrees at this probe.'
          : ` Squared bias ${comparison.movedBias ? 'moved' : 'did not move'} and prediction variance ${comparison.movedVariance ? 'moved' : 'did not move'}; both terms have to be read together.`}
        {' '}The identity is exact here: the largest gap between the averaged squared error and bias² + variance is
        {' '}{round(Math.max(reference.identityResidual, candidate.identityResidual), 12)}.
        {active.sigma === 0 && ' With σ = 0 every world has the same targets, so there is one fitted curve and no prediction variance at all.'}
      </p>

      <Table caption={`Both designs at these settings, at x = ${round(active.probe, 2)}. Adding the two extra fixed inputs changes the design, so a pointwise error can rise even while its variance falls.`}
        headings={['design', 'fit', 'mean prediction', 'squared bias', 'prediction variance', 'expected error']}
        rows={bothDesigns.map(entry => [
          entry.design === active.design ? `${entry.label} (applied)` : entry.label,
          degreeNames[entry.degree],
          round(entry.record.meanPrediction, 6), round(entry.record.squaredBias, 9),
          round(entry.record.variance, 9), round(entry.record.expectedError, 9),
        ])}
        rowClass={index => (bothDesigns[index].design === active.design
          && bothDesigns[index].degree === active.candidate ? 'is-selected' : undefined)} />

      <details>
        <summary>Interpolation weights: the probe prediction as a fixed combination of the training outcomes</summary>
        <Table caption={`Each fit predicts at x = ${round(active.probe, 2)} by weighting the training outcomes. The weights depend on the design and the degree, never on the observed values, and they always sum to one.`}
          headings={['fit', ...reference.trainX.map(value => `weight at x = ${round(value, 2)}`), 'sum', 'σ² × Σ w²']}
          rows={[reference, candidate].map((record, index) => [
            `${index === 0 ? 'reference' : 'candidate'} · ${degreeNames[record.degree]}`,
            ...record.weights.map(value => round(value, 6)),
            round(record.weightSum, 9), round(record.varianceFromWeights, 9),
          ])} />
        <p className="bv-caption">
          A negative weight is normal for polynomial interpolation: the candidate has {candidate.negativeWeights} of them.
          It is also why raising one observed target can push the prediction at another input downwards. Multiplying the
          squared weights by σ² reproduces the prediction variance in the table above, by a route that never enumerates a
          single world.
        </p>
      </details>

      <details>
        <summary>Every training world: its outcomes and its prediction at the probe</summary>
        <Table caption={`All ${reference.worldCount} equally likely datasets, each with probability ${round(1 / reference.worldCount, 6)}.`}
          headings={['world', 'signs', 'observed targets', `reference at x = ${round(active.probe, 2)}`, `candidate at x = ${round(active.probe, 2)}`]}
          rows={reference.worlds.map((entry, index) => [
            `world ${index + 1}`,
            entry.signs.map(value => (value > 0 ? '+' : '−')).join(' '),
            entry.targets.map(value => round(value, 4)).join(', '),
            round(entry.probePrediction, 6),
            round(candidate.worlds[index].probePrediction, 6),
          ])} scroll />
      </details>
    </>}
  </Investigation>;
}
