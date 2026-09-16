import { useState } from 'react';
import {
  coordinateFit, dropoutEnumeration, limits, modelPrediction, nonlinearInset,
  scalarObjective, scalarSlopes, scalarSolution, termsTouchedBy,
} from '../../data/regularization-models';
import {
  featureNames, inferenceFixture, rawFeatureLabels, rawFeatureRanges, rawFeatureUnits, ridgeModel,
} from '../../data/regularization-data';
import {
  Field, Investigation, NumberField, Plot, Prediction, SignedBars, Table, coefficientText, curve, fixed, round, signed,
  useInvestigation,
} from './RegularizationShared.jsx';
import './regularization-labs.css';

const familyRatio = { ridge: 0, 'elastic net (ρ = 0.5)': 0.5, lasso: 1 };

/* ============================================================ §2 · I1 */

const thresholdPresets = {
  base: { label: 'Baseline: z = 0.4, λ = 1, lasso', z: 0.4, strength: 1, ratio: 1 },
  crossed: { label: 'Contrast: z = 1.4', z: 1.4, strength: 1, ratio: 1 },
  stillZero: { label: 'Null: z = −0.6', z: -0.6, strength: 1, ratio: 1 },
  noPenalty: { label: 'Null: λ = 0', z: 0.4, strength: 0, ratio: 1 },
  family: { label: 'Family comparison at z = 3', z: 3, strength: 1, ratio: 1 },
};

/** Where does a coefficient disappear? */
export function ThresholdLab() {
  const state = useInvestigation(thresholdPresets.base);
  const active = state.active;
  const solution = scalarSolution(active.z, active.strength, active.ratio);
  const slopes = scalarSlopes(solution.coefficient, active.z, active.strength, active.ratio);
  const zeroSlopes = scalarSlopes(0, active.z, active.strength, active.ratio);
  const answerFor = inputs => {
    const answer = scalarSolution(inputs.z, inputs.strength, inputs.ratio);
    return {
      outcome: answer.coefficient === 0 ? 'zero' : answer.coefficient > 0 ? 'positive' : 'negative',
      value: answer.coefficient,
    };
  };
  // Whole-number bounds, so the tick labels under the plot are readable numbers.
  const span = [Math.floor(Math.min(-1, active.z - 1.5)), Math.ceil(Math.max(1, active.z + 1.5))];
  const objective = w => scalarObjective(w, active.z, active.strength, active.ratio).total;
  const peak = Math.max(objective(span[0]), objective(span[1]));
  const zSpan = [-4, 4];
  const mapped = z => scalarSolution(z, active.strength, active.ratio).coefficient;
  const families = Object.entries(familyRatio).map(([name, ratio]) => ({
    name, ratio, solution: scalarSolution(active.z, active.strength, ratio),
  }));
  return <Investigation
    title="Where does a coefficient disappear?"
    question="One normalized coefficient, one data preference z, one strength λ and one mixing fraction ρ. Record whether the fitted coefficient will be negative, exactly zero or positive before solving."
    note="The threshold interval is where the data preference is not strong enough to buy a nonzero coefficient. Inside it the answer is exactly zero, not a small number rounded for display."
    onReset={state.reset}>
    <div className="rg-controls">
      <NumberField label="Data preference z" value={state.draft.z} min={limits.preference.minimum} max={limits.preference.maximum}
        step="0.1" decimals={4} onChange={z => state.edit({ z })} />
      <NumberField label="Strength λ" value={state.draft.strength} min={limits.strength.minimum} max={limits.strength.maximum}
        step="0.1" decimals={4} onChange={strength => state.edit({ strength })} />
      <NumberField label="Mixing fraction ρ" value={state.draft.ratio} min={limits.ratio.minimum} max={limits.ratio.maximum}
        step="0.1" decimals={4} onChange={ratio => state.edit({ ratio })} />
    </div>
    <div className="rg-buttons">
      {Object.entries(thresholdPresets).map(([key, preset]) => (
        <button key={key} type="button" onClick={() => state.load(preset)}>{preset.label}</button>
      ))}
    </div>
    <p className="rg-caption">
      Ranges: z from {round(limits.preference.minimum)} to {round(limits.preference.maximum)}, λ from {round(limits.strength.minimum)} to
      {' '}{round(limits.strength.maximum)}, ρ from {round(limits.ratio.minimum)} to {round(limits.ratio.maximum)}.
      The threshold is λρ = {round(state.draft.strength * state.draft.ratio, 6)} and the denominator is
      1 + λ(1 − ρ) = {round(1 + state.draft.strength * (1 - state.draft.ratio), 6)} for the values in the fields now.
      A preset fills in inputs only; it never preselects the outcome.
    </p>
    <Prediction
      prompt={`With z = ${round(state.draft.z, 4)}, λ = ${round(state.draft.strength, 4)} and ρ = ${round(state.draft.ratio, 4)}, what is the fitted coefficient?`}
      options={[['negative', 'Negative'], ['zero', 'Exactly zero'], ['positive', 'Positive']]}
      state={state} answerFor={answerFor}
      numeric={{ label: 'Optional: the coefficient itself', name: 'the coefficient', tolerance: 1e-4, digits: 9 }}
      describe={`S(${round(active.z, 4)}, ${round(solution.threshold, 6)}) = ${round(solution.thresholded, 9)}, divided by ${round(solution.denominator, 6)}, gives ${coefficientText(solution.coefficient, 9)}.`} />

    {state.result && <>
      <div className="rg-panels is-pair">
        <div className="rg-panel">
          <h4>The objective along the coefficient axis</h4>
          <Plot caption="" width={300} height={170} domain={span} range={[0, peak]}
            padding={{ left: 42, right: 12, top: 14, bottom: 30 }}
            describe={`The total objective against the coefficient w. Its minimum is at ${round(solution.coefficient, 9)} with value ${round(scalarObjective(solution.coefficient, active.z, active.strength, active.ratio).total, 9)}. The slope just left of that point is ${round(slopes.fromLeft, 6)} and just right of it ${round(slopes.fromRight, 6)}.`}>
            {(scaleX, scaleY) => <>
              <polyline className="rg-curve is-data" points={curve(scaleX, scaleY, span, w => scalarObjective(w, active.z, active.strength, active.ratio).data)} />
              <polyline className="rg-curve is-penalty" points={curve(scaleX, scaleY, span, w => scalarObjective(w, active.z, active.strength, active.ratio).penalty)} />
              <polyline className="rg-curve is-total" points={curve(scaleX, scaleY, span, objective)} />
              <circle className="rg-mark" cx={scaleX(solution.coefficient)} cy={scaleY(objective(solution.coefficient))} r="4.5" />
              <text x={scaleX(solution.coefficient)} y={scaleY(objective(solution.coefficient)) - 16} textAnchor="middle">w* = {round(solution.coefficient, 4)}</text>
            </>}
          </Plot>
          <p>Gold: the total. Dashed green: the data term. Dotted blue: the penalty. The corner at w = 0 exists when λρ &gt; 0; without that absolute-value penalty the curve is smooth.</p>
        </div>
        <div className="rg-panel">
          <h4>The data preference, and the coefficient it produces</h4>
          <Plot caption="" width={300} height={170} domain={zSpan} range={[-4, 4]}
            padding={{ left: 42, right: 12, top: 14, bottom: 30 }}
            valueTicks={[-4, 0, 4]}
            describe={`The map from the data preference z to the fitted coefficient. Between minus ${round(solution.threshold, 6)} and ${round(solution.threshold, 6)} the output is exactly zero; outside it the output is the soft-thresholded value divided by ${round(solution.denominator, 6)}. The current z is ${round(active.z, 4)} and its output is ${round(solution.coefficient, 9)}.`}>
            {(scaleX, scaleY) => <>
              {solution.threshold > 0 && <rect x={scaleX(-solution.threshold)} y={scaleY(3)}
                width={Math.max(0, scaleX(solution.threshold) - scaleX(-solution.threshold))} height={scaleY(-3) - scaleY(3)}
                fill="#1d2119" stroke="#9a8347" strokeDasharray="3 3" />}
              <line className="rg-grid" x1={scaleX(zSpan[0])} x2={scaleX(zSpan[1])} y1={scaleY(0)} y2={scaleY(0)} />
              <polyline className="rg-curve is-total" points={curve(scaleX, scaleY, zSpan, mapped, 401)} />
              <circle className="rg-mark" cx={scaleX(active.z)} cy={scaleY(solution.coefficient)} r="4.5" />
              <text x={scaleX(0)} y={scaleY(3) + 14} textAnchor="middle">threshold band</text>
              <text x={scaleX(zSpan[1])} y={scaleY(0) + 26} textAnchor="end">z</text>
            </>}
          </Plot>
          <p>A flat output over a whole interval of data preferences is the mechanism behind sparsity: the data term keeps changing and the answer does not.</p>
        </div>
      </div>

      <p className="rg-readout" aria-live="polite">
        Threshold λρ = {round(solution.threshold, 6)}, interval [{round(-solution.threshold, 6)}, {round(solution.threshold, 6)}];
        denominator 1 + λ(1 − ρ) = {round(solution.denominator, 6)}. The coefficient is {coefficientText(solution.coefficient, 9)}.
        {solution.coefficient === 0
          ? ` At w = 0 the smooth part of the derivative is ${round(zeroSlopes.smooth, 6)}, which lies inside [−${round(solution.threshold, 6)}, ${round(solution.threshold, 6)}], so zero is the minimum.`
          : ` At the minimum the derivative from the left is ${round(slopes.fromLeft, 6)} and from the right ${round(slopes.fromRight, 6)}.`}
        {solution.atEndpoint && ' This z sits on a threshold endpoint: the coefficient is zero and the subgradient condition holds with equality. The strictly convex quadratic makes zero the unique minimizer of this scalar problem.'}
        {active.strength === 0 && ' With λ = 0 the answer is z for every ρ, so changing ρ alone does nothing here.'}
      </p>

      <Table caption="The same z and λ under three families. These are minima of three different objectives, and the smallest of them is not a way to choose a family."
        headings={['family', 'ρ', 'threshold λρ', 'denominator', 'coefficient']}
        rows={families.map(entry => [entry.name, entry.ratio, round(entry.solution.threshold, 6),
          round(entry.solution.denominator, 6), coefficientText(entry.solution.coefficient, 9)])} />
    </>}
  </Investigation>;
}

/* ============================================================ §3 · I2 */

const coordinatePresets = {
  standard: { label: 'Four constructed rows', rows: [[1, 1, 3.4], [1, -1, 2.6], [-1, 1, -2.6], [-1, -1, -3.4]], strength: 1, ratio: 1, reversed: false },
  changed: { label: 'Contrast: row 0 target 3.4 → 7.4', rows: [[1, 1, 7.4], [1, -1, 2.6], [-1, 1, -2.6], [-1, -1, -3.4]], strength: 1, ratio: 1, reversed: false },
  shifted: { label: 'Null: add 7 to every target', rows: [[1, 1, 10.4], [1, -1, 9.6], [-1, 1, 4.4], [-1, -1, 3.6]], strength: 1, ratio: 1, reversed: false },
  reordered: { label: 'Null: the same rows in a different order', rows: [[-1, -1, -3.4], [1, 1, 3.4], [-1, 1, -2.6], [1, -1, 2.6]], strength: 1, ratio: 1, reversed: false },
  duplicate: { label: 'Two rows of duplicate sensors', rows: [[-1, -1, -2], [1, 1, 2]], strength: 1, ratio: 1, reversed: false },
  constant: { label: 'A constant second feature', rows: [[1, 2, 3], [1, 2, 3], [-1, 2, -3], [-1, 2, -3]], strength: 1, ratio: 1, reversed: false },
};
const rowNames = ['row 0', 'row 1', 'row 2', 'row 3'];

/** Fit the residual, then inspect the prediction. */
export function CoordinateLab() {
  const [output, setOutput] = useState('coefficient1');
  const [row, setRow] = useState(0);
  const [sweep, setSweep] = useState(1);
  const state = useInvestigation(coordinatePresets.standard);
  const active = state.active;
  const design = active.rows.map(entry => [entry[0], entry[1]]);
  const targets = active.rows.map(entry => entry[2]);
  const order = active.reversed ? [1, 0] : [0, 1];
  const fit = coordinateFit(design, targets, active.strength, active.ratio, { order });
  const selectedRow = Math.min(row, active.rows.length - 1);
  const shownSweep = Math.min(sweep, fit.history.length);
  const record = fit.history[shownSweep - 1];
  const answerFor = inputs => {
    const rows = inputs.rows;
    const result = coordinateFit(rows.map(entry => [entry[0], entry[1]]), rows.map(entry => entry[2]),
      inputs.strength, inputs.ratio, { order: inputs.reversed ? [1, 0] : [0, 1] });
    const index = Math.min(row, rows.length - 1);
    if (output === 'row') {
      const value = result.fitted[index];
      const target = rows[index][2];
      return {
        outcome: Math.abs(value - target) <= 1e-9 ? 'equal' : value > target ? 'above' : 'below',
        value,
      };
    }
    const column = output === 'coefficient1' ? 0 : 1;
    const weight = result.weights[column];
    return { outcome: weight === 0 ? 'zero' : weight > 0 ? 'positive' : 'negative', value: weight };
  };
  const options = output === 'row'
    ? [['above', 'Above that row’s target'], ['equal', 'Exactly at it'], ['below', 'Below it']]
    : [['negative', 'Negative'], ['zero', 'Exactly zero'], ['positive', 'Positive']];
  const edit = (index, column, value) => state.edit({
    rows: state.draft.rows.map((entry, position) => (position === index
      ? entry.map((old, spot) => (spot === column ? value : old)) : entry)),
  });
  return <Investigation
    title="Fit the residual, then inspect the prediction"
    question="Edit the actual rows, the targets, the strength, the mixing fraction and the coordinate order. Choose which output you are predicting, record the prediction, then step through the partial residuals that produce it."
    note="Each coordinate update uses the current values of the other coefficients. The stopping rule is the optimality condition itself, not a small step."
    onReset={() => { state.reset(); setOutput('coefficient1'); setRow(0); setSweep(1); }}>
    <div className="rg-buttons">
      {Object.entries(coordinatePresets).map(([key, preset]) => (
        <button key={key} type="button" onClick={() => { state.load(preset); setSweep(1); setRow(0); }}>{preset.label}</button>
      ))}
    </div>
    <div className="rg-controls is-wide">
      {state.draft.rows.map((entry, index) => <fieldset key={index} className="rg-row-group">
        <legend>{rowNames[index]}</legend>
        <NumberField label={`${rowNames[index]} x₁`} value={entry[0]} min={limits.entry.minimum} max={limits.entry.maximum} step="0.2" decimals={4}
          onChange={value => edit(index, 0, value)} />
        <NumberField label={`${rowNames[index]} x₂`} value={entry[1]} min={limits.entry.minimum} max={limits.entry.maximum} step="0.2" decimals={4}
          onChange={value => edit(index, 1, value)} />
        <NumberField label={`${rowNames[index]} target`} value={entry[2]} min={limits.entry.minimum} max={limits.entry.maximum} step="0.2" decimals={4}
          onChange={value => edit(index, 2, value)} />
      </fieldset>)}
    </div>
    <p className="rg-caption">
      Ranges: every feature value and target from {round(limits.entry.minimum)} to {round(limits.entry.maximum)}, λ from
      {' '}{round(limits.strength.minimum)} to {round(limits.strength.maximum)}, ρ from {round(limits.ratio.minimum)} to {round(limits.ratio.maximum)}.
    </p>
    <div className="rg-controls">
      <NumberField label="Strength λ" value={state.draft.strength} min={limits.strength.minimum} max={limits.strength.maximum}
        step="0.1" decimals={4} onChange={strength => state.edit({ strength })} />
      <NumberField label="Mixing fraction ρ" value={state.draft.ratio} min={limits.ratio.minimum} max={limits.ratio.maximum}
        step="0.1" decimals={4} onChange={ratio => state.edit({ ratio })} />
      <Field label="Coordinate order">
        <select value={state.draft.reversed ? 'reversed' : 'forward'}
          onChange={event => state.edit({ reversed: event.target.value === 'reversed' })}>
          <option value="forward">feature 1, then feature 2</option>
          <option value="reversed">feature 2, then feature 1</option>
        </select>
      </Field>
      <Field label="Output to predict">
        <select value={output} onChange={event => { setOutput(event.target.value); state.edit({}); }}>
          <option value="coefficient1">sign of coefficient 1</option>
          <option value="coefficient2">sign of coefficient 2</option>
          <option value="row">fitted value of a chosen row</option>
        </select>
      </Field>
      {output === 'row' && <Field label="Which row">
        <select value={selectedRow} onChange={event => { setRow(Number(event.target.value)); state.edit({}); }}>
          {state.draft.rows.map((entry, index) => <option key={index} value={index}>{rowNames[index]}</option>)}
        </select>
      </Field>}
    </div>
    <Prediction
      prompt={output === 'row'
        ? `Where will the fitted value of ${rowNames[selectedRow]} sit relative to its target ${round(state.draft.rows[selectedRow][2], 4)}?`
        : `What will coefficient ${output === 'coefficient1' ? 1 : 2} be at λ = ${round(state.draft.strength, 4)} and ρ = ${round(state.draft.ratio, 4)}?`}
      options={options} state={state} answerFor={answerFor}
      numeric={output === 'row'
        ? { label: 'The fitted value, to two decimals', name: 'the fitted value', tolerance: 0.05, digits: 6, required: true }
        : { label: 'Optional: the coefficient itself', name: 'the coefficient', tolerance: 1e-4, digits: 9 }}
      applyLabel="Apply and check"
      describe={`The recorded coordinate order was ${order.map(column => `feature ${column + 1}`).join(', then ')}.`} />

    {state.result && <>
      <div className="rg-stage">
        <h4>1. Centre the rows on their own training means</h4>
        <Table caption="Original values, the training means subtracted from them, and the centred target"
          headings={['row', 'x₁', 'x₂', 'target', 'centred x₁', 'centred x₂', 'centred target']}
          rows={active.rows.map((entry, index) => [
            rowNames[index], round(entry[0], 6), round(entry[1], 6), round(entry[2], 6),
            round(entry[0] - fit.featureMeans[0], 6), round(entry[1] - fit.featureMeans[1], 6),
            round(entry[2] - fit.targetMean, 6),
          ])} />
        <p className="rg-caption">
          Feature means {fit.featureMeans.map(value => round(value, 6)).join(' and ')}; target mean {round(fit.targetMean, 6)}.
          The intercept is recovered afterwards as ȳ − x̄ᵀw and is never penalised.
        </p>
      </div>

      <div className="rg-stage">
        <h4>2. One coordinate at a time, inside sweep {shownSweep} of {fit.history.length}</h4>
        <div className="rg-controls">
          <Field label="Inspect sweep">
            <select value={shownSweep} onChange={event => setSweep(Number(event.target.value))}>
              {fit.history.map(entry => <option key={entry.sweep} value={entry.sweep}>sweep {entry.sweep}</option>)}
            </select>
          </Field>
        </div>
        <Table caption="Each coordinate's partial residual association, curvature, threshold, divisor and resulting coefficient. A constant column has no data curvature: a positive penalty still minimizes at zero; only an unpenalized zero-curvature coordinate is flat and takes a declared zero."
          headings={['visit', 'feature', 'association c', 'curvature a', 'threshold λρ', 'divisor a + λ(1 − ρ)', 'coefficient']}
          rows={record.steps.map((step, index) => [
            index + 1, `feature ${step.column + 1}`, round(step.association, 9), round(step.curvature, 9),
            round(step.threshold, 6), round(step.denominator, 9),
            step.flat ? 'declared 0 · flat' : coefficientText(step.weight, 9),
          ])} />
        <Table caption="The residual after every coordinate in this sweep, one column per row"
          headings={['after visiting', ...active.rows.map((entry, index) => rowNames[index])]}
          rows={record.steps.map((step, index) => [`feature ${step.column + 1}`, ...step.residual.map(value => round(value, 9))])} />
      </div>

      <div className="rg-stage">
        <h4>3. The fit, and what it predicts</h4>
        <Table caption="Coefficients, the recovered intercept and each row's fitted value"
          headings={['quantity', 'value']}
          rows={[
            ['coefficient 1', coefficientText(fit.weights[0], 9)],
            ['coefficient 2', coefficientText(fit.weights[1], 9)],
            ['intercept', round(fit.intercept, 9)],
            ['sweeps used', `${fit.sweeps} of ${fit.maxSweeps}`],
            ['largest optimality violation', round(fit.kktResidual, 12)],
            ['objective', round(fit.objective, 9)],
            ['data term', round(fit.data, 9)],
            ['penalty', round(fit.penalty, 9)],
            ['mean squared error of the fit', round(fit.meanSquaredError, 9)],
            ['Pure-lasso λ threshold for all coefficients to be zero', round(fit.lambdaMax, 9)],
          ]} />
        <Table caption="Row by row"
          headings={['row', 'target', 'fitted value', 'residual']}
          rows={active.rows.map((entry, index) => [
            rowNames[index], round(entry[2], 6), round(fit.fitted[index], 9), signed(entry[2] - fit.fitted[index], 9),
          ])}
          rowClass={index => (output === 'row' && index === selectedRow ? 'is-selected' : undefined)} />
        <p className="rg-readout" aria-live="polite">
          {fit.converged
            ? `Converged: the largest optimality violation is ${round(fit.kktResidual, 12)}, at or below the tolerance ${fit.tolerance}, after ${fit.sweeps} sweep${fit.sweeps === 1 ? '' : 's'}.`
            : `Not converged: the sweep cap of ${fit.maxSweeps} was reached with a largest optimality violation of ${round(fit.kktResidual, 12)}. The values above are the current iterate and are inspectable, but they are not an exact optimum.`}
          {' '}The objective is not a mean squared error: it adds the penalty, and a smaller objective under a different penalty means nothing across families.
        </p>
      </div>

      {fit.history.length > 1 && <Plot caption="Objective by sweep, from the actual computed history"
        width={340} height={170} domain={[1, fit.history.length]}
        range={[Math.min(...fit.history.map(entry => entry.objective)) - 0.05, Math.max(...fit.history.map(entry => entry.objective)) + 0.05]}
        ticks={fit.history.filter(entry => entry.sweep % Math.max(1, Math.ceil(fit.history.length / 5)) === 0 || entry.sweep === 1).map(entry => entry.sweep)}
        describe={`The objective after each sweep: ${fit.history.map(entry => `sweep ${entry.sweep} gives ${round(entry.objective, 9)}`).join('; ')}.`}>
        {(scaleX, scaleY) => <>
          <polyline className="rg-curve is-total" points={fit.history.map(entry => `${scaleX(entry.sweep)},${scaleY(entry.objective)}`).join(' ')} />
          {fit.history.map(entry => <circle key={entry.sweep} className="rg-mark" cx={scaleX(entry.sweep)} cy={scaleY(entry.objective)} r="3" />)}
        </>}
      </Plot>}
      {fit.history.length === 1 && <p className="rg-caption">
        One sweep was enough here, so there is no history to plot: the optimality condition already held after the first pass.
      </p>}
    </>}
  </Investigation>;
}

/* ============================================================ §5 · I3 */

const step = [50, 0.1, 0.0001, 0.1, 0.0001];
const decimals = [4, 4, 6, 4, 9];

/** Trace one airfoil prediction through its terms. */
export function AirfoilTraceLab() {
  const start = { raw: inferenceFixture.rawFeatures.slice(), target: inferenceFixture.observedTarget };
  const state = useInvestigation(start);
  const active = state.active;
  const trace = modelPrediction(ridgeModel, active.raw);
  const baseline = modelPrediction(ridgeModel, inferenceFixture.rawFeatures);
  const changed = active.raw.map((value, index) => value !== inferenceFixture.rawFeatures[index]);
  const touched = new Set(changed.flatMap((isChanged, index) => (isChanged ? termsTouchedBy(index) : [])));
  const answerFor = inputs => {
    const value = modelPrediction(ridgeModel, inputs.raw).prediction;
    const difference = value - inferenceFixture.basePrediction;
    return {
      outcome: Math.abs(difference) <= 1e-9 ? 'same' : difference > 0 ? 'up' : 'down',
      value,
    };
  };
  // The browser recomputes the saved model rather than reading a stored answer,
  // so an unchanged row differs from the recorded value by floating-point dust.
  // Anything below the agreement tolerance is reported as the exact zero it is.
  const agreement = 1e-9;
  const drift = value => (Math.abs(value) <= agreement ? 0 : value);
  const change = drift(trace.prediction - inferenceFixture.basePrediction);
  const changeText = digits => (change === 0 ? 'no change' : `${signed(change, digits)} dB`);
  const ranked = trace.contributions
    .map((value, index) => ({ name: featureNames[index], value, index }))
    .sort((left, right) => Math.abs(right.value) - Math.abs(left.value));
  return <Investigation
    title="Trace one airfoil prediction through its terms"
    question={`Development row ${inferenceFixture.rowId} under the saved final ridge model at λ = 0.001. Edit the five physical measurements, record whether the predicted sound-pressure level will rise, fall or stay the same, then read the twenty polynomial terms and their signed contributions.`}
    note="No fitting happens here. The coefficients, means and scales are read from the saved model; changing the reference target cannot change the prediction."
    onReset={state.reset}>
    <div className="rg-controls is-wide">
      {rawFeatureLabels.map((label, index) => (
        <NumberField key={label} label={`${label} (${rawFeatureUnits[index]})`} value={state.draft.raw[index]}
          min={limits.raw[index].minimum} max={limits.raw[index].maximum} step={String(step[index])} decimals={decimals[index]}
          onChange={value => state.edit({ raw: state.draft.raw.map((old, position) => (position === index ? value : old)) })} />
      ))}
      <NumberField label="Reference target (dB), a comparison value only" value={state.draft.target}
        min={80} max={160} step="0.1" decimals={3} onChange={target => state.edit({ target })} />
    </div>
    <div className="rg-buttons">
      <button type="button" onClick={() => state.load(start)}>Recorded measurements</button>
      <button type="button" onClick={() => state.load({ ...state.draft, raw: inferenceFixture.changedFeatures.slice() })}>
        Contrast: frequency 1,250 → 1,750 Hz
      </button>
      <button type="button" onClick={() => state.load({ ...state.draft, raw: inferenceFixture.rawFeatures.slice(), target: 130 })}>
        Null: change only the reference target
      </button>
    </div>
    <p className="rg-caption">
      Observed development ranges: {rawFeatureLabels.map((label, index) => `${label} ${rawFeatureRanges[index][0]} to ${rawFeatureRanges[index][1]} ${rawFeatureUnits[index]}`).join('; ')}.
      An edited combination can be hypothetical even inside the observed ranges: treat a changed row as a model scenario, not a guaranteed
      feasible new experiment and not a causal effect. This deterministic trace carries no interval.
    </p>
    <Prediction
      prompt={`Against the recorded prediction of ${fixed(inferenceFixture.basePrediction, 6)} dB, what will these measurements give?`}
      options={[['up', 'A higher predicted level'], ['same', 'Exactly the same'], ['down', 'A lower predicted level']]}
      state={state} answerFor={answerFor}
      numeric={{ label: 'Optional: the predicted level in dB', name: 'the prediction', tolerance: 0.05, digits: 6 }}
      applyLabel="Apply and check"
      describe={`The prediction is ${fixed(trace.prediction, 9)} dB, ${change === 0 ? 'unchanged from' : `a change of ${signed(change, 6)} dB from`} the recorded row.`} />

    {state.result && <>
      <div className="rg-stage">
        <h4>1. Five physical measurements</h4>
        <Table caption="The edited row against the recorded one"
          headings={['measurement', 'unit', 'recorded', 'now', 'observed range']}
          rows={rawFeatureLabels.map((label, index) => [
            label, rawFeatureUnits[index], round(inferenceFixture.rawFeatures[index], 7), round(active.raw[index], 7),
            `${rawFeatureRanges[index][0]} to ${rawFeatureRanges[index][1]}`,
          ])}
          rowClass={index => (changed[index] ? 'is-selected' : undefined)} />
      </div>
      <div className="rg-stage">
        <h4>2. Twenty polynomial terms, in the model's own order</h4>
        <p className="rg-caption">
          {touched.size === 0
            ? 'Nothing is edited, so no term is highlighted.'
            : `Changing ${rawFeatureLabels.filter((label, index) => changed[index]).join(' and ')} moves ${touched.size} of the twenty terms together: its own value, its square and every product it takes part in.`}
        </p>
        <div className="rg-chips">
          {featureNames.map((name, index) => (
            <span key={name} className={`rg-chip${touched.has(index) ? ' is-touched' : ''}`}>{name}</span>
          ))}
        </div>
      </div>
      <div className="rg-stage">
        <h4>3. Signed contributions, all in decibels</h4>
        <SignedBars caption="Each term's standardized value times its saved coefficient. The raw terms carry compound units; every contribution here is in target decibels."
          describe={`Twenty signed contributions in decibels: ${ranked.slice(0, 5).map(item => `${item.name} contributes ${round(item.value, 6)}`).join('; ')}, and fifteen more.`}
          items={featureNames.map((name, index) => ({ name, value: trace.contributions[index] }))}
          unit=" dB" digits={6} highlight={index => touched.has(index)} />
        <p className="rg-readout" aria-live="polite">
          Intercept {fixed(trace.intercept, 9)} dB plus the twenty contributions {signed(trace.total, 9)} dB
          gives <strong>{fixed(trace.prediction, 9)} dB</strong>.
          The recorded row predicted {fixed(inferenceFixture.basePrediction, 9)} dB, so this is {changeText(9)}.
          Against the reference target {round(active.target, 3)} dB the residual is
          {' '}{signed(active.target - trace.prediction, 6)} dB — a comparison value, never a feature.
        </p>
      </div>
      <details>
        <summary>Inspect every term: raw value, saved mean and scale, standardized value, coefficient and contribution</summary>
        <Table caption="The full twenty-term trace, exactly as the saved model computes it"
          headings={['term', 'raw value', 'saved mean', 'saved scale', 'standardized', 'coefficient', 'contribution (dB)']}
          rows={featureNames.map((name, index) => [
            name, round(trace.terms[index], 6), round(ridgeModel.scaleMean[index], 6), round(ridgeModel.scaleScale[index], 6),
            round(trace.standardized[index], 6), round(ridgeModel.coefficients[index], 6), round(trace.contributions[index], 6),
          ])}
          rowClass={index => (touched.has(index) ? 'is-selected' : undefined)} scroll />
      </details>
      <p className="rg-caption">
        The twenty terms are not twenty independent dials: they are determined by the five measurements, which is why this investigation
        edits the measurements and not the scaled terms. This page recomputes the saved model here rather than reading a stored answer,
        and agrees with the recorded {fixed(inferenceFixture.basePrediction, 9)} dB to better than 10⁻⁹ dB; a difference inside that
        tolerance is reported as zero. Recomputed baseline {fixed(baseline.prediction, 9)} dB.
      </p>
    </>}
  </Investigation>;
}

/* ============================================================ §6 · I4 */

const dropoutPresets = {
  base: { label: 'Baseline: x = (2, 1), w = (1, −1), y = 1, q = 0.5', x: [2, 1], w: [1, -1], y: 1, keep: 0.5 },
  deterministic: { label: 'Contrast: q = 1', x: [2, 1], w: [1, -1], y: 1, keep: 1 },
  practice: { label: 'Practice 5: x = (1, 2), w = (2, 0), y = 1, q = 0.75', x: [1, 2], w: [2, 0], y: 1, keep: 0.75 },
  shiftedTarget: { label: 'Null: change the target to 3', x: [2, 1], w: [1, -1], y: 3, keep: 0.5 },
};

/** Enumerate the masks. */
export function DropoutLab() {
  const state = useInvestigation(dropoutPresets.base);
  const active = state.active;
  const enumeration = dropoutEnumeration(active.x, active.w, active.y, active.keep);
  const inset = nonlinearInset();
  const answerFor = inputs => {
    const result = dropoutEnumeration(inputs.x, inputs.w, inputs.y, inputs.keep);
    return {
      outcome: result.analyticExtra === 0 ? 'equal' : 'higher',
      value: result.expectedPrediction,
    };
  };
  // Two masks can give the same output; draw one stem per distinct value with
  // the combined probability, rather than stacking identical labels.
  const outputs = [...new Map(enumeration.branches.map(branch => [
    branch.prediction,
    {
      value: branch.prediction,
      masks: enumeration.branches.filter(other => other.prediction === branch.prediction).map(other => other.mask),
      probability: enumeration.branches
        .filter(other => other.prediction === branch.prediction)
        .reduce((sum, other) => sum + other.probability, 0),
    },
  ])).values()];
  const lowOutput = Math.min(...outputs.map(entry => entry.value), 0);
  const highOutput = Math.max(...outputs.map(entry => entry.value), 0);
  const place = value => highOutput === lowOutput ? 150 : 44 + 212 * (value - lowOutput) / (highOutput - lowOutput);
  return <Investigation
    title="Enumerate the masks"
    question="Two contributions, one target and one keep probability. Predict the mean noisy output and whether the expected noisy half-squared loss will exceed the clean one, then open the exact mask tree."
    note="Every branch below is enumerated with its actual probability. There is no seed, no sampling and no displayed realisation: this is the exact expectation."
    onReset={state.reset}>
    <div className="rg-controls">
      <NumberField label="Input x₁" value={state.draft.x[0]} min={limits.contribution.minimum} max={limits.contribution.maximum}
        step="0.5" decimals={4} onChange={value => state.edit({ x: [value, state.draft.x[1]] })} />
      <NumberField label="Input x₂" value={state.draft.x[1]} min={limits.contribution.minimum} max={limits.contribution.maximum}
        step="0.5" decimals={4} onChange={value => state.edit({ x: [state.draft.x[0], value] })} />
      <NumberField label="Coefficient w₁" value={state.draft.w[0]} min={limits.contribution.minimum} max={limits.contribution.maximum}
        step="0.5" decimals={4} onChange={value => state.edit({ w: [value, state.draft.w[1]] })} />
      <NumberField label="Coefficient w₂" value={state.draft.w[1]} min={limits.contribution.minimum} max={limits.contribution.maximum}
        step="0.5" decimals={4} onChange={value => state.edit({ w: [state.draft.w[0], value] })} />
      <NumberField label="Target y" value={state.draft.y} min={limits.target.minimum} max={limits.target.maximum}
        step="0.5" decimals={4} onChange={y => state.edit({ y })} />
      <NumberField label="Keep probability q" value={state.draft.keep} min={limits.keep.minimum} max={limits.keep.maximum}
        step="0.05" decimals={4} onChange={keep => state.edit({ keep })} />
    </div>
    <div className="rg-buttons">
      {Object.entries(dropoutPresets).map(([key, preset]) => (
        <button key={key} type="button" onClick={() => state.load(preset)}>{preset.label}</button>
      ))}
    </div>
    <p className="rg-caption">
      Ranges: each input and coefficient from {round(limits.contribution.minimum)} to {round(limits.contribution.maximum)}, the target
      from {round(limits.target.minimum)} to {round(limits.target.maximum)}, and the keep probability from
      {' '}{round(limits.keep.minimum)} to {round(limits.keep.maximum)}. A keep probability of zero is refused: every activation would
      be dropped and the rescaling would divide by zero.
    </p>
    <Prediction
      prompt={`With q = ${round(state.draft.keep, 4)}, how does the expected noisy half-squared loss compare with the clean one?`}
      options={[['higher', 'Higher than the clean loss'], ['equal', 'Exactly equal'], ['lower', 'Lower than the clean loss']]}
      state={state} answerFor={answerFor}
      numeric={{ label: 'The mean noisy output', name: 'the mean output', tolerance: 1e-6, digits: 9, required: true }}
      describe={`The clean prediction is ${round(enumeration.cleanPrediction, 9)} and the mean noisy prediction is ${round(enumeration.expectedPrediction, 9)}; the clean half-loss is ${round(enumeration.cleanLoss, 9)} and the expected noisy half-loss ${round(enumeration.expectedLoss, 9)}. The analytic excess is ${round(enumeration.analyticExtra, 9)}; subtracting two nearly equal floating-point losses may round this away.`} />

    {state.result && <>
      <div className="rg-stage">
        <h4>The four masks, with their actual probabilities</h4>
        <svg viewBox="0 0 340 270" role="img"
          aria-label={`A probability tree with four branches. ${enumeration.branches.map(branch => `mask ${branch.mask.join(' and ')} has probability ${round(branch.probability, 6)}, output ${round(branch.prediction, 6)} and half-loss ${round(branch.halfSquaredLoss, 6)}`).join('; ')}. Each branch carries a bar of its own probability, so the four are not drawn as equal shares; a branch drawn with a dashed outline has probability zero and cannot occur.`}>
          <rect className="rg-lane" x="6" y="122" width="70" height="26" rx="3" />
          <text x="41" y="139" textAnchor="middle">q = {round(active.keep, 4)}</text>
          {enumeration.branches.map((branch, index) => {
            const top = 12 + index * 64;
            const centre = top + 25;
            return <g key={branch.mask.join('')}>
              <path className="rg-flow" d={`M78,135 C106,135 118,${centre} 146,${centre}`} />
              <rect className="rg-lane" x="148" y={top} width="186" height="50" rx="3"
                strokeDasharray={branch.possible ? undefined : '4 3'} />
              <text x="154" y={top + 15}>({branch.mask.join(', ')})</text>
              <text x="154" y={top + 32}>→ {round(branch.prediction, 4)}</text>
              <text x="328" y={top + 15} textAnchor="end">{round(branch.probability, 4)}</text>
              <rect x="154" y={top + 41} width={Math.max(0, 174 * branch.probability).toFixed(1)} height="4" fill="#8eb9a5" />
            </g>;
          })}
        </svg>
        <p className="rg-caption">
          The green strip inside each branch is that branch's own probability, drawn to one shared scale; the number beside the curve is
          the same value, printed at the right of each branch. A branch with a dashed outline and no strip has probability zero and cannot occur under this keep
          probability, so it is listed rather than deleted. Nothing here is sampled.
        </p>
      </div>
      <Table caption="Every branch: the rescaled inputs, the output and what it contributes to the expectation"
        headings={['mask', 'probability', 'rescaled inputs', 'noisy output', 'half-squared loss', 'probability × loss']}
        rows={enumeration.branches.map(branch => [
          `(${branch.mask.join(', ')})`, round(branch.probability, 9),
          branch.scaled.map(value => round(value, 6)).join(', '),
          round(branch.prediction, 9), round(branch.halfSquaredLoss, 9),
          round(branch.probability * branch.halfSquaredLoss, 9),
        ])}
        rowClass={index => (enumeration.branches[index].possible ? undefined : 'is-zero')} />

      <div className="rg-panels is-pair">
        <div className="rg-panel">
          <h4>The discrete distribution of the output</h4>
          <svg viewBox="0 0 300 140" role="img" aria-label={`The distinct outputs and their combined probabilities: ${outputs.map(entry => `${round(entry.value, 6)} with probability ${round(entry.probability, 6)} from mask${entry.masks.length > 1 ? 's' : ''} ${entry.masks.map(mask => mask.join(' and ')).join(', ')}`).join('; ')}. The mean is ${round(enumeration.expectedPrediction, 6)} and the clean prediction is ${round(enumeration.cleanPrediction, 6)}.`}>
            <line className="rg-axis" x1="30" x2="296" y1="104" y2="104" />
            {outputs.map((entry, index) => (
              <g key={entry.masks.map(mask => mask.join('')).join('-')}>
                <line className="rg-stem is-penalty" x1={place(entry.value)} x2={place(entry.value)}
                  y1="104" y2={104 - 76 * entry.probability} />
                <circle className={entry.probability > 0 ? 'rg-mark' : 'rg-mark is-hollow'} cx={place(entry.value)} cy={104 - 76 * entry.probability} r="3.5" />
                <text x={place(entry.value)} y={index % 2 === 0 ? 120 : 134} textAnchor="middle">{String.fromCharCode(65 + index)}</text>
              </g>
            ))}
            <line className="rg-stem is-data" x1={place(enumeration.expectedPrediction)} x2={place(enumeration.expectedPrediction)} y1="104" y2="20" />
            <text x="150" y="16" textAnchor="middle">mean {round(enumeration.expectedPrediction, 4)}</text>
          </svg>
          <p>Stem height is the combined probability. The mean output equals the clean prediction; the spread around it is what the loss responds to.</p>
          <Table caption="Output labels and combined probabilities" headings={['label', 'output', 'probability']} rows={outputs.map((entry, index) => [String.fromCharCode(65 + index), round(entry.value, 9), round(entry.probability, 9)])} />
        </div>
        <div className="rg-panel">
          <h4>Two independent routes to the same extra loss</h4>
          <Table caption="Summed over branches, and from the closed formula"
            headings={['quantity', 'value']}
            rows={[
              ['clean prediction wᵀx', round(enumeration.cleanPrediction, 9)],
              ['mean noisy prediction', round(enumeration.expectedPrediction, 9)],
              ['clean half-squared loss', round(enumeration.cleanLoss, 9)],
              ['expected noisy half-squared loss', round(enumeration.expectedLoss, 9)],
              ['difference, from the enumeration', round(enumeration.extraLoss, 9)],
              ['(1 − q)/(2q) × Σ (wⱼxⱼ)²', round(enumeration.analyticExtra, 9)],
            ]} />
          <p>
            {enumeration.agrees
              ? 'The two routes agree within the floating-point check tolerance; the identity is exact in real arithmetic. Subtracting nearly equal losses can lose relative accuracy, so use the analytic excess for very small effects.'
              : 'These two floating-point routes disagree beyond the check tolerance; inspect numerical accuracy before drawing a conclusion.'}
            {active.keep === 1 && ' At q = 1 the zero-probability branches remain listed as impossible; nothing is divided by zero.'}
            {(active.w[0] === 0 || active.w[1] === 0) && ' A zero coefficient makes its mask irrelevant: that branch changes no output.'}
          </p>
        </div>
      </div>
      <p className="rg-readout" aria-live="polite">
        Mean output {round(enumeration.expectedPrediction, 9)}; expected noisy half-loss {round(enumeration.expectedLoss, 9)} against a
        clean {round(enumeration.cleanLoss, 9)}. The analytic excess {round(enumeration.analyticExtra, 9)} depends on x, w and q and <em>not</em> on
        the target, so changing y alone moves both losses and leaves their difference where it was.
      </p>
      <details>
        <summary>Why a mean-preserving mask is not exact model averaging through a nonlinearity</summary>
        <p className="rg-caption">
          Let a value be {inset.values.join(' or ')} with equal probability and pass it through f(u) = max(0, u − {inset.shift}).
          The mean of f is {round(inset.meanOutput, 6)}, while f of the mean input {round(inset.meanInput, 6)} is {round(inset.outputOfMean, 6)}.
          Ordinary dropout-off inference is therefore not generally an exact average over every masked nonlinear network. This inset is
          fixed; it takes no input of its own.
        </p>
      </details>
    </>}
  </Investigation>;
}
