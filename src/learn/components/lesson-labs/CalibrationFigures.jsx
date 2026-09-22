import { useId, useState } from 'react';
import { BarRows, Field, KindTag, Pill, PlotFrame, Select, Table, fixed, ratio, round } from './CalibrationShared.jsx';
import {
  apsSets, barGeometry, blockGeometry, classSets, conditioningFork, conformalThreshold, cqrInterval,
  entropyNats, fitPav, fixtures, intervalGeometry, leadingClass, mosaicGeometry, mosaicMarginal,
  normalizedInterval, orderStatisticCoverageMean, positivePredictiveValue, railGeometry,
  reliabilityFromRecord, reliabilityGeometry, resolutionComparison, scatterGeometry, softmaxAt,
} from '../../data/calibration-models.js';
import { calibrationData } from '../../data/calibration-data.js';

/** Inline figures for the Calibration & Conformal Prediction lesson.
 *
 * Each figure draws from `calibration-models.js` or from the generated
 * `calibration-data.js`, never from a literal copied out of the prose, so the
 * verifiers check the same numbers and the same geometry a reader sees.
 *
 * Interactive figures expose their current result immediately. Controls act
 * on the mathematical model; they never ask learners to guess an answer.
 *
 * Every figure that shows a measured quantity prints which KIND of quantity it
 * is. A count on eighty reserved rows and an exact constructed population rate
 * look identical on a page; on this topic, mistaking one for the other is the
 * whole failure mode.
 */

const classification = calibrationData.classification;
const regression = calibrationData.regression;

/* ------------------------------------- F1 · one forecast, two questions (§1) */

export function ConditioningForkFigure() {
  const [chosen, setChosen] = useState(fixtures.forecastGroups);
  const updateGroup = (index, key, value) => setChosen(previous => previous.map((group, position) => (
    position === index ? { ...group, [key]: value } : group
  )));
  const matches = preset => chosen.every((group, index) => (
    group.forecast === preset[index].forecast && group.positiveRate === preset[index].positiveRate
  ));
  const fork = conditioningFork(chosen);
  const geometry = reliabilityGeometry({
    kind: 'population',
    bins: fork.classOnePoints.map((point, index) => ({
      index, lower: 0, upper: 1, count: 1, positive: 0,
      meanP: point.x, fractionPositive: point.y, gap: point.y - point.x, members: [],
    })),
  });
  const confidenceGeometry = reliabilityGeometry({
    kind: 'population',
    bins: fork.confidencePoints.map((point, index) => ({
      index, lower: 0, upper: 1, count: 1, positive: 0,
      meanP: point.x, fractionPositive: point.y, gap: point.y - point.x, members: [],
    })),
  });
  return <div className="cal-figure-block" data-calibration-figure="conditioning-fork">
    <p className="cal-caption"><strong>Figure 1.</strong> The same two groups of forecasts, read two ways. On the
      left, the forecast for class 1 against how often class 1 actually happens. On the right, the confidence in
      whichever class was selected against how often that selection was right. They are different conditioning
      questions. Move the controls to see which errors each reading can expose or hide.</p>
    <div className="cal-presets">
      <button type="button" aria-pressed={matches(fixtures.forecastGroups)}
        onClick={() => setChosen(fixtures.forecastGroups)}>Reset: forecasts .2 and .8, rates .3 and .9</button>
      <button type="button" aria-pressed={matches(fixtures.practiceForecastGroups)}
        onClick={() => setChosen(fixtures.practiceForecastGroups)}>Forecasts .1 and .9, rates .2 and 1</button>
      <button type="button" onClick={() => setChosen(previous => previous.map(group => ({
        ...group, forecast: group.positiveRate,
      })))}>Match forecasts to the group rates</button>
    </div>
    <p className="cal-caption">Each group stays half the population. Forecasts and exact constructed rates
      range from 0 to 1; drag a slider or focus it and use the arrow keys (steps of .01).
      At a forecast of .5 the model selects class 1. Equal forecasts or confidences are pooled,
      so two groups can become one plotted point.</p>
    <div className="cal-fork-controls">
      {chosen.map((group, index) => <fieldset key={index}>
        <legend>Group {index === 0 ? 'A' : 'B'} · 50% of the population</legend>
        {[['forecast', 'Forecast for class 1'], ['positiveRate', 'Actual class-1 rate']].map(([key, label]) => (
          <Field key={key} label={`${label} · group ${index === 0 ? 'A' : 'B'}`} value={fixed(group[key], 2)}>
            <input type="range" min="0" max="1" step="0.01" value={group[key]}
              onChange={event => updateGroup(index, key, Number(event.target.value))} />
          </Field>
        ))}
      </fieldset>)}
    </div>
    <div className="cal-figure-row">
      <PlotFrame geometry={geometry} caption="Class-1 probability against the class-1 rate"
        xLabel="forecast for class 1" yLabel="rate of class 1"
        describe={`${fork.classOnePoints.length} points on a square plot with a dashed diagonal: `
          + fork.classOnePoints.map(point => `${round(point.x, 3)} against ${round(point.y, 3)}`).join(', ')
          + '. A point on the diagonal has matching forecast and class-1 rate.'}>
        <line className="cal-diagonal" x1={geometry.diagonal.x1} y1={geometry.diagonal.y1}
          x2={geometry.diagonal.x2} y2={geometry.diagonal.y2} />
        {geometry.points.map(point => <g key={point.index}>
          <line className="cal-gap-line" x1={point.x} y1={point.diagonalY} x2={point.x} y2={point.y} />
          <circle className={`cal-dot ${point.gap > 0 ? 'is-above' : point.gap < 0 ? 'is-below' : ''}`}
            cx={point.x} cy={point.y} r={5} />
        </g>)}
      </PlotFrame>
      <PlotFrame geometry={confidenceGeometry} caption="Top confidence against being right"
        xLabel="confidence in the selected class" yLabel="rate of being right"
        describe={`${fork.confidencePoints.length} point or points: `
          + fork.confidencePoints.map(point => `${round(point.x, 3)} against ${round(point.y, 3)}`).join(', ')
          + '. A point on the diagonal has matching confidence and rate of being right.'}>
        <line className="cal-diagonal" x1={confidenceGeometry.diagonal.x1} y1={confidenceGeometry.diagonal.y1}
          x2={confidenceGeometry.diagonal.x2} y2={confidenceGeometry.diagonal.y2} />
        {confidenceGeometry.points.map(point => <g key={point.index}>
          <circle className={`cal-dot ${Math.abs(point.gap) < 1e-15 ? '' : point.gap > 0 ? 'is-above' : 'is-below'}`}
            cx={point.x} cy={point.y} r={5} />
        </g>)}
      </PlotFrame>
    </div>
    <Table caption="Both readings of the same two groups, written out"
      headings={['group', 'share', 'forecast for class 1', 'rate of class 1', 'class it selects',
        'confidence in that class', 'rate that selection is right']}
      rows={fork.rows.map(row => [
        row.name, round(row.share, 3), round(row.forecast, 3), round(row.positiveRate, 3),
        String(row.predictedClass), round(row.confidence, 3), round(row.correctness, 3),
      ])}
      kind="population"
      footnote={`Averaged over the population, confidence is ${round(fork.meanConfidence, 6)} and the rate of being `
        + `right is ${round(fork.meanCorrectness, 6)}. `
        + (fork.confidenceLooksCalibrated
          ? 'Every confidence-conditioned point is on the diagonal. '
          : 'At least one confidence-conditioned point is off the diagonal; agreement of overall averages alone is insufficient. ')
        + (fork.classOneLooksCalibrated
          ? 'Each distinct class-1 forecast also matches its pooled class-1 rate.'
          : 'At least one class-1 forecast differs from its pooled class-1 rate. '
            + (fork.confidenceLooksCalibrated
              ? 'Pooling by confidence hides errors that remain visible when conditioning on the class-1 forecast.'
              : 'Both readings now expose a discrepancy.'))}
    />
  </div>;
}

/* --------------------------- F2 · calibration can discard information (§1) */

export function ResolutionFigure() {
  const model = resolutionComparison(fixtures.resolution);
  /* Two scales, not one.
   *
   * On a single shared length the cost pair runs to 16 and the Brier pair to
   * .16, so the difference between .16 and .15 — the whole point of the second
   * comparison — was a hundredth of a bar and invisible. The caption said as
   * much, which is the sign a figure has failed rather than an excuse for it.
   * Each pair is now scaled within itself, and the units are named on each. */
  const costBars = barGeometry([
    { name: 'coarse score .2', value: model.coarseCost },
    { name: 'the two group rates', value: model.fullCost },
  ]);
  const brierBars = barGeometry([
    { name: 'coarse score .2', value: model.coarseBrier },
    { name: 'the two group rates', value: model.fullBrier },
  ]);
  return <div className="cal-figure-block">
    <p className="cal-caption"><strong>Figure 2.</strong> Two production lines with true fault rates{' '}
      {model.rows.map(row => round(row.rate, 3)).join(' and ')}. A model that reports{' '}
      {round(model.coarseForecast, 3)} for both is calibrated with respect to its own score. So is a model that
      reports each line's own rate. They differ in resolution, and the difference is worth money.</p>
    <Table caption="What each forecast leads you to do, and what it costs"
      headings={['line', 'share', 'true fault rate', 'coarse forecast', 'action', 'expected cost',
        'own-rate forecast', 'action', 'expected cost']}
      rows={model.rows.map((row, index) => [
        `line ${index + 1}`, round(row.share, 3), round(row.rate, 3),
        round(row.coarseForecast, 3), row.coarseAction, round(row.coarseCost, 4),
        round(row.fullForecast, 3), row.fullAction, round(row.fullCost, 4),
      ])}
      kind="population"
      footnote={`Releasing a faulty item costs ${fixtures.resolution.releaseCost} and quarantining anything costs `
        + `${fixtures.resolution.quarantineCost}, so release is worth it exactly while the probability stays at or `
        + `below ${round(model.decideThreshold, 4)}. The coarse score is below that for both lines, so it releases `
        + `everything.`} />
    <div className="cal-figure-row">
      <BarRows caption="Expected cost, in cost units" rows={costBars} unit="units"
        describe={`Two bars: the coarse score costs ${round(model.coarseCost, 4)} and the two group rates cost `
          + `${round(model.fullCost, 4)}.`} />
      <BarRows caption="Expected Brier loss, in squared probability" rows={brierBars}
        describe={`Two bars: the coarse score scores ${round(model.coarseBrier, 4)} and the two group rates `
          + `${round(model.fullBrier, 4)}.`} />
    </div>
    <KindTag kind="population" />
    <p className="cal-caption">
      Each pair is drawn on its own scale, because a cost measured in tens and a squared probability error
      measured in hundredths share no useful axis. Both forecasts are calibrated. The better proper score and the
      lower cost reflect extra usable information, not a repaired calibration.
    </p>
  </div>;
}

/* ---------------------- F3 · a sigmoid and a pooled staircase, side by side (§3) */

export function MapComparisonFigure() {
  const [fixture, setFixture] = useState('eight');
  const scores = fixture === 'eight' ? fixtures.pavScores : fixtures.tiedScores;
  const labels = fixture === 'eight' ? fixtures.pavLabels : fixtures.tiedLabels;
  const fit = fitPav(scores, labels);
  const geometry = blockGeometry(fit);
  const record = calibrationData.constructedRecord.sigmoid;
  return <div className="cal-figure-block">
    <p className="cal-caption"><strong>Figure 3.</strong> The same observations under two assumptions. The blocks
      are the monotone fit: each one is a group of scores forced to share a probability, drawn at the height of
      that probability. The dashed curve is the two-parameter sigmoid, which cannot bend to a local contradiction
      and instead spreads it over the whole range.</p>
    <div className="cal-presets">
      <button type="button" className={fixture === 'eight' ? 'is-selected' : undefined}
        onClick={() => setFixture('eight')}>Eight distinct scores</button>
      <button type="button" className={fixture === 'tied' ? 'is-selected' : undefined}
        onClick={() => setFixture('tied')}>Four scores, two of them equal</button>
    </div>
    <PlotFrame geometry={{ ...geometry, ticks: null }} caption="Pooled blocks, with the raw group means marked"
      xLabel="observations in score order" yLabel="probability"
      describe={`Rectangles at probability heights: ${geometry.blocks.map(block =>
        `scores ${round(fit.knots[block.start], 3)} to ${round(fit.knots[block.end], 3)} at `
        + `${round(block.mean, 4)} over ${block.weight} observations`).join('; ')}. `
        + `Dots mark each score's own raw rate before pooling.`}>
      {[0, 0.25, 0.5, 0.75, 1].map(value => {
        const y = geometry.scales.y(value);
        return <g key={value}>
          <line className="cal-gridline" x1={geometry.box.left} y1={y}
            x2={geometry.box.width - geometry.box.right} y2={y} />
          <text className="cal-small" x={geometry.box.left - 6} y={y + 3} textAnchor="end">{value}</text>
        </g>;
      })}
      {geometry.blocks.map(block => <rect key={`${block.start}-${block.end}`}
        className={`cal-block${block.spans > 1 ? ' is-merged' : ''}`}
        x={block.x1} y={block.y} width={Math.max(block.x2 - block.x1, 2)}
        height={Math.max(geometry.baseline - block.y, 1)} />)}
      {geometry.knots.map(knot => <g key={knot.index}>
        <circle className="cal-raw-mark" cx={knot.x} cy={knot.rawY} r={3} />
        <text className="cal-small" x={knot.x} y={geometry.baseline + 13} textAnchor="middle">
          {round(knot.knot, 2)}
        </text>
      </g>)}
      {fixture === 'eight' && <path className="cal-sigmoid-line"
        d={geometry.knots.map((knot, index) =>
          `${index === 0 ? 'M' : 'L'}${knot.x.toFixed(2)},${geometry.scales.y(record.probabilities[index]).toFixed(2)}`)
          .join(' ')} />}
    </PlotFrame>
    <Table caption={fixture === 'eight'
      ? 'The eight observations, their raw rates, the monotone fit and the fitted sigmoid'
      : 'Four observations of which two share a score'}
      headings={fixture === 'eight'
        ? ['score', 'observations', 'positive', 'raw rate', 'monotone fit', 'sigmoid']
        : ['score', 'observations', 'positive', 'raw rate', 'monotone fit']}
      rows={fit.knots.map((knot, index) => {
        const row = [
          round(knot, 3), String(fit.weights[index]), String(fit.totals[index]),
          round(fit.totals[index] / fit.weights[index], 6), round(fit.fitted[index], 6),
        ];
        return fixture === 'eight' ? [...row, round(record.probabilities[index], 6)] : row;
      })}
      kind="calibration"
      footnote={fixture === 'eight'
        ? `The sigmoid's slope and offset are a ≈ ${round(record.a, 9)} and b ≈ ${round(record.b, 9)}, fitted by `
          + `${record.solver}. A block spanning several scores is drawn wider than one score; its height is the `
          + 'pooled probability, and its width is how many scores it covers.'
        : `The two observations at score ${round(fit.knots[0], 3)} become one block BEFORE any merging, so they can `
          + `never receive different fitted values. Pooling that block with the next gives `
          + `${round(fit.fitted[0], 6)} — the combined counts, `
          + `${fit.blocks[0].total} positive out of ${fit.blocks[0].weight}. Averaging the two block probabilities `
          + `instead would give ${round((0.5 + 0) / 2, 6)}, which is the wrong answer.`} />
  </div>;
}

/* ----------------------- F4 · temperature moves confidence, not the winner (§3) */

export function TemperatureFigure() {
  const [threshold, setThreshold] = useState(0.7);
  const rows = fixtures.temperatures.map(temperature => {
    const probabilities = softmaxAt(fixtures.temperatureLogits, temperature);
    return {
      temperature, probabilities,
      leading: leadingClass(probabilities),
      entropy: entropyNats(probabilities),
      passesThreshold: probabilities[0] >= threshold,
    };
  });
  const width = 300;
  const barLeft = 52;
  const barWidth = width - barLeft - 12;
  const height = 24 + rows.length * 34;
  const titleId = useId();
  return <div className="cal-figure-block">
    <p className="cal-caption"><strong>Figure 4.</strong> Three logits{' '}
      {fixtures.temperatureLogits.join(', ')} under four temperatures. Dividing every logit by the same positive
      number cannot reorder them, so class A wins in all four rows. What it does change is how much of the mass A
      keeps — and therefore whether A clears a probability threshold.</p>
    <figure className="cal-figure">
      <svg className="cal-plot" viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={titleId}
        style={{ maxWidth: `${width}px` }}>
        <title id={titleId}>
          {`Stacked probability bars at temperatures ${fixtures.temperatures.join(', ')}: `
            + rows.map(row => `at T = ${row.temperature}, A ${round(row.probabilities[0], 4)}, `
              + `B ${round(row.probabilities[1], 4)}, C ${round(row.probabilities[2], 4)}`).join('; ')}
        </title>
        {rows.map((row, index) => {
          const y = 14 + index * 34;
          let cursor = barLeft;
          return <g key={row.temperature}>
            <text className="cal-small" x={barLeft - 6} y={y + 11} textAnchor="end">T = {row.temperature}</text>
            {row.probabilities.map((probability, position) => {
              const segment = probability * barWidth;
              const x = cursor;
              cursor += segment;
              return <g key={position}>
                <rect className={`cal-prob-bar${position === row.leading.index ? ' is-leading' : ''}`}
                  x={x} y={y} width={Math.max(segment, 0)} height={16} />
                {/* Ink chosen for the fill it sits on. Dark on the gold winning
                    segment, near-white on the green ones. The first pass fixed
                    only the leading label by eye and left the other three at
                    3.01:1 — right size, right content, inside their own bars,
                    and unreadable. Contrast is now measured, not judged. */}
                {segment > 34 && <text
                  className={`cal-small ${position === row.leading.index ? 'is-on-fill' : 'is-on-bar'}`}
                  x={x + segment / 2} y={y + 12} textAnchor="middle">
                  {['A', 'B', 'C'][position]} {round(probability, 3)}
                </text>}
              </g>;
            })}
            <text className="cal-small" x={barLeft} y={y + 28}>
              entropy {round(row.entropy, 4)} nats · A {row.passesThreshold ? 'clears' : 'misses'} the threshold
            </text>
          </g>;
        })}
        {/* One segment per bar rather than one line down the whole figure.
            Drawn full height it ran straight through each row's entropy
            caption — "entropy 0.1069 nats · A clears the threshold" with a
            1.6px gold rule through the middle of it. Nothing reported that
            because the browser verifier's curve sampler queried
            `path, polyline, polygon` and this is a <line>; the guard audit
            found the gap by counting what the check had to look at (two
            shapes in the whole lesson) and widening the selector.

            The segments mark the same x on every bar, so the drawn rule is
            still exactly the applied rule: A's probability against the
            threshold, on each of the four temperatures. */}
        {rows.map((row, index) => {
          const y = 14 + index * 34;
          const x = barLeft + threshold * barWidth;
          /* Dark ink, not the gold used elsewhere. Confining the rule to the
             bars put it on the gold leading segment, where gold on gold is not
             a line at all — full height, it was legible only because it also
             crossed the dark ground between rows. Since it now never leaves a
             filled bar, and both bar fills are light enough for dark ink, one
             thin dark rule reads everywhere it goes. A casing under the gold
             worked too, but at the width needed it swallowed a digit of the
             segment label it crosses. */
          return <line key={`threshold-${row.temperature}`} className="cal-threshold-on-bar"
            x1={x} y1={y} x2={x} y2={y + 16} />;
        })}
      </svg>
    </figure>
    <div className="cal-controls">
      <Select label="Decision threshold on class A's probability" value={String(threshold)}
        options={[['0.5', '.5'], ['0.6', '.6'], ['0.7', '.7'], ['0.8', '.8'], ['0.9', '.9']]}
        onChange={value => setThreshold(Number(value))}
        hint="The vertical line. It is a decision rule, not part of the model." />
    </div>
    <Table caption="Exact probabilities at each temperature"
      headings={['temperature', 'class A', 'class B', 'class C', 'winner', 'entropy in nats',
        `A at or above ${threshold}?`]}
      rows={rows.map(row => [
        round(row.temperature, 3), round(row.probabilities[0], 6), round(row.probabilities[1], 6),
        round(row.probabilities[2], 6), ['A', 'B', 'C'][row.leading.index], round(row.entropy, 6),
        row.passesThreshold ? 'yes' : 'no',
      ])}
      rowClass={index => (rows[index].passesThreshold ? 'is-leading' : undefined)}
      kind="calibration"
      footnote={'The winner column never changes; the last column does. A claim about top-1 accuracy is not a '
        + 'claim about every downstream action.'} />
  </div>;
}

/* ------------------------------- F5 · who is allowed to see each label (§4) */

const ROLE_LANES = [
  {
    key: 'train', title: '1 · base fit',
    what: 'learn the scaler and the classifier',
    freeze: 'the model and its preprocessing are frozen here',
  },
  {
    key: 'probability_calibration', title: '2 · probability calibration',
    what: 'fit a map from the frozen model’s score to a probability',
    freeze: 'the probability map is frozen here',
  },
  {
    key: 'conformal', title: '3 · conformal calibration',
    what: 'choose the rank threshold using the now-frozen probability procedure',
    freeze: 'the threshold is frozen here',
  },
  {
    key: 'test', title: '4 · assessment',
    what: 'measure probability quality, coverage, set sizes and slices',
    freeze: 'nothing downstream may use these outcomes',
  },
];

export function LabelRolesFigure() {
  const total = Object.values(classification.roleSizes).reduce((sum, value) => sum + value, 0);
  /* Reflowing HTML, not SVG coordinates.
   *
   * This was four stacked SVG lanes with the sentence "80 rows — fit a map from
   * the frozen model's score to a probability" set as SVG text, which ran a
   * hundred pixels past the right edge of its own viewBox and was trimmed. The
   * only thing here that encodes a quantity is the width of each lane's bar,
   * which a CSS percentage expresses exactly; everything else is prose and
   * belongs where prose can wrap. */
  return <div className="cal-figure-block">
    <p className="cal-caption"><strong>Figure 5.</strong> The {total} banknote rows split into four roles, in
      order. Each step down the list hands on the object that has just been frozen, not the data: what passes to
      the next role is a fitted thing that can no longer change, which is exactly what makes the next role's
      calculation honest.</p>
    <ol className="cal-lanes">
      {ROLE_LANES.map((lane, index) => {
        const rows = classification.roleSizes[lane.key];
        return <li key={lane.key}>
          <span className="cal-lane-head">
            <b>{lane.title}</b>
            <span>{rows} of {total} rows</span>
          </span>
          <span className="cal-lane-track" role="img"
            aria-label={`${lane.title} holds ${rows} of ${total} rows`}>
            <span className="cal-lane-fill" style={{ width: `${(100 * rows) / total}%` }} />
          </span>
          <span className="cal-lane-what">{lane.what}</span>
          {index < ROLE_LANES.length - 1
            ? <span className="cal-lane-freeze">↓ {lane.freeze}</span>
            : <span className="cal-lane-freeze is-final">■ {lane.freeze}</span>}
        </li>;
      })}
    </ol>
    <Table caption="The four roles, their sizes and what each one is allowed to see"
      headings={['role', 'rows', 'what it does', 'what it hands on']}
      rows={ROLE_LANES.map(lane => [
        lane.title, String(classification.roleSizes[lane.key]), lane.what, lane.freeze,
      ])}
      footnote={'The third role is separate from the second because probability calibration is itself training. '
        + 'Reusing its outcomes for an ordinary split-conformal threshold would score them differently from a '
        + 'fresh example, which is precisely what the rank argument forbids. Holding out only the original '
        + 'classifier’s training data is not enough.'} />
    <Table caption="Two ways to use cross-validation instead of a frozen model, which are not the same procedure"
      headings={['setting', 'what fits the calibrator', 'what predicts afterwards', 'what it does not undo']}
      rows={[
        ['ensemble = False', 'out-of-fold scores from every fold',
          'one base estimator refitted on all the supplied training data',
          'hyperparameter selection that already used these outcomes'],
        ['ensemble = True', 'each fold’s own held-out scores',
          'the fold-specific calibrated models, averaged',
          'the same selection, and it is a different inference procedure from the frozen single model'],
      ]}
      footnote={'In both cases the scaler, the text vocabulary and any learned feature selection must live INSIDE '
        + 'the estimator cloned for each fold. Fitting a vocabulary before the folds are built leaks the fold’s '
        + 'own rows into its calibrator.'} />
  </div>;
}

/* ------------------------------ F6 · scores, the rail, and three sets (§5) */

export function ScoreSetFigure() {
  const threshold = conformalThreshold(fixtures.calibrationScores, fixtures.defaultAlpha);
  const geometry = railGeometry(fixtures.calibrationScores, threshold, { domain: [0, 1] });
  const smallThreshold = conformalThreshold(fixtures.calibrationScores, 0.05);
  return <div className="cal-figure-block">
    <p className="cal-caption"><strong>Figure 6.</strong> The nine calibration scores of section 5 on one axis,
      with the eighth smallest marked. A candidate's classes turn into scores, and every score at or below the
      mark is kept. Equality is kept too, which is why the second candidate holds two classes rather than one.</p>
    <PlotFrame geometry={{ ...geometry, ticks: null }} axes="x"
      caption={`Nine calibration scores; the mark is rank ${threshold.k} of ${threshold.n}`}
      xLabel="nonconformity score" yLabel=""
      describe={`Nine marks at ${[...fixtures.calibrationScores].sort((a, b) => a - b).join(', ')}, with a `
        + `vertical line at the eighth smallest, ${round(threshold.q, 3)}.`}>
      <line className="cal-rail" x1={geometry.axis.x1} y1={geometry.axis.y}
        x2={geometry.axis.x2} y2={geometry.axis.y} />
      {[0, 0.25, 0.5, 0.75, 1].map(value => <text key={value} className="cal-small"
        x={geometry.scales.x(value)} y={geometry.axis.y + 15} textAnchor="middle">{value}</text>)}
      {geometry.marks.map(mark => <g key={`${mark.rank}-${mark.score}`}>
        <circle className={`cal-rail-mark${mark.selected ? ' is-selected' : ''}${mark.belowThreshold ? '' : ' is-above'}`}
          cx={mark.x} cy={geometry.axis.y} r={mark.selected ? 5.5 : 4} />
        <text className="cal-small" x={mark.labelX} y={geometry.axis.y - 10} textAnchor="middle">{mark.rank}</text>
      </g>)}
      {geometry.markerX !== null && <line className="cal-threshold"
        x1={geometry.markerX} y1={geometry.box.top} x2={geometry.markerX} y2={geometry.axis.y + 6} />}
    </PlotFrame>
    <Table caption="Three candidates under the same threshold"
      headings={['candidate', 'probabilities', 'scores 1 − p', 'set', 'size']}
      rows={fixtures.candidateVectors.map(vector => {
        const sets = classSets(vector.probabilities, threshold.q);
        return [
          vector.name,
          vector.probabilities.map(value => round(value, 2)).join(', '),
          sets.scores.map(value => round(value, 2)).join(', '),
          sets.size === 0 ? 'empty' : `{${sets.included.map((keep, index) =>
            (keep ? fixtures.classNames[index] : null)).filter(Boolean).join(', ')}}`,
          String(sets.size),
        ];
      })}
      kind="calibration"
      footnote={`k = ceil((${threshold.n} + 1) × (1 − ${fixtures.defaultAlpha})) = ${threshold.k}, so the threshold `
        + `is the ${threshold.k}th smallest score, ${round(threshold.q, 6)}. An empty set is a legitimate result of `
        + 'this score, not an error: it happens when no class reaches the learned threshold. A singleton is not a '
        + 'label with an 80% chance of being right — the theorem is about a different probability.'} />
    <div className="cal-chips">
      <Pill tone="warn" mark="∞">
        At 95% coverage the rank is {smallThreshold.k}, beyond all {smallThreshold.n} cards, so the threshold is
        infinity and every class is returned
      </Pill>
    </div>
    <p className="cal-caption">
      Infinity is not a score just past the last mark, so no line is drawn for it: it is the whole label space.
      Clipping the rank down to the largest available card would be a different procedure, and it would discard
      exactly the protection a 95% target was asking for.
    </p>
  </div>;
}

/* ------------------------ F7 · the residual, the scale and the interval (§6) */

export function ResidualGeometryFigure() {
  const [mode, setMode] = useState('normalized');
  const normalizedScores = fixtures.residuals.map((residual, index) => residual / fixtures.localScales[index]);
  const absolute = conformalThreshold(fixtures.residuals, fixtures.intervalAlpha);
  const normalized = conformalThreshold(normalizedScores, fixtures.intervalAlpha);
  const cqrThreshold = conformalThreshold(fixtures.cqrScores, fixtures.cqrAlpha);
  const cqr = cqrInterval(fixtures.cqrBase.lower, fixtures.cqrBase.upper, cqrThreshold.q);
  const rows = mode === 'cqr'
    ? [{
      name: 'initial', lower: fixtures.cqrBase.lower, upper: fixtures.cqrBase.upper,
      y: (fixtures.cqrBase.lower + fixtures.cqrBase.upper) / 2,
    }, {
      name: 'adjusted', lower: cqr.lower, upper: cqr.upper,
      y: (cqr.lower + cqr.upper) / 2,
    }]
    : fixtures.queries.map(query => {
      const interval = mode === 'normalized'
        ? normalizedInterval(query.centre, query.localScale, normalized.q)
        : normalizedInterval(query.centre, 1, absolute.q);
      return { name: query.name, lower: interval.lower, upper: interval.upper, y: query.centre };
    });
  /* `left` makes room for the row name INSIDE the drawing. Anchored at the
     end just outside the bars, "easy case" started nine units left of zero
     and the viewBox trimmed it. */
  const geometry = intervalGeometry(rows, { height: 120, box: { left: 62 } });
  return <div className="cal-figure-block">
    <p className="cal-caption"><strong>Figure 7.</strong> The same calibration evidence, three score functions.
      The absolute residual gives every query the same width. Dividing by a local scale makes the threshold
      dimensionless and multiplies it back at each query, so an easy case is not handed a width it does not need.
      Conformalising a quantile interval can move the endpoints inwards.</p>
    <div className="cal-presets">
      {[['absolute', 'absolute residual'], ['normalized', 'residual over a local scale'],
        ['cqr', 'conformalized quantile regression']].map(([key, label]) => <button key={key} type="button"
          className={mode === key ? 'is-selected' : undefined} onClick={() => setMode(key)}>{label}</button>)}
    </div>
    <figure className="cal-figure">
      <figcaption>
        {mode === 'cqr' ? 'The initial interval and the conformalised one, to scale'
          : 'Each query’s interval, to scale, in response units'}
      </figcaption>
      <svg className="cal-plot" viewBox={`0 0 ${geometry.box.width} ${geometry.box.height}`} role="img"
        style={{ maxWidth: `${geometry.box.width}px` }}
        aria-label={geometry.rows.map(row =>
          `${row.name} from ${round(row.lower, 3)} to ${round(row.upper, 3)}, width ${round(row.width, 3)}`)
          .join('; ')}>
        {geometry.rows.map(row => <g key={row.name}>
          <line className="cal-interval" strokeWidth={3}
            x1={row.x1} y1={row.y} x2={row.x2} y2={row.y} />
          <line className="cal-cap" x1={row.x1} y1={row.y - 6} x2={row.x1} y2={row.y + 6} />
          <line className="cal-cap" x1={row.x2} y1={row.y - 6} x2={row.x2} y2={row.y + 6} />
          <circle className="cal-target" cx={row.target} cy={row.y} r={3.5} />
          <text className="cal-small" x={4} y={row.y + 3}>{row.name}</text>
        </g>)}
      </svg>
      <figcaption className="cal-caption">
        Horizontal axis in response units. The dot is the point prediction and the bar is the interval.
      </figcaption>
    </figure>
    <Table caption={mode === 'cqr' ? 'The signed CQR adjustment' : 'Thresholds and the widths they produce'}
      headings={mode === 'cqr'
        ? ['quantity', 'value']
        : ['query', 'point prediction', 'local scale', 'threshold', 'half-width', 'interval', 'width']}
      rows={mode === 'cqr'
        ? [
          ['calibration scores', fixtures.cqrScores.join(', ')],
          ['alpha', round(fixtures.cqrAlpha, 3)],
          ['rank k', String(cqrThreshold.k)],
          ['threshold q', round(cqrThreshold.q, 6)],
          ['initial interval', `[${fixtures.cqrBase.lower}, ${fixtures.cqrBase.upper}]`],
          ['conformalised interval', cqr.empty ? 'empty' : `[${round(cqr.lower, 6)}, ${round(cqr.upper, 6)}]`],
          ['width', cqr.empty ? 'no width' : round(cqr.width, 6)],
        ]
        : fixtures.queries.map(query => {
          const threshold = mode === 'normalized' ? normalized : absolute;
          const interval = mode === 'normalized'
            ? normalizedInterval(query.centre, query.localScale, threshold.q)
            : normalizedInterval(query.centre, 1, threshold.q);
          return [
            query.name, round(query.centre, 3),
            mode === 'normalized' ? round(query.localScale, 3) : '1 (not used)',
            round(threshold.q, 6), round(interval.halfWidth, 6),
            `[${round(interval.lower, 6)}, ${round(interval.upper, 6)}]`, round(interval.width, 6),
          ];
        })}
      kind="calibration"
      footnote={mode === 'cqr'
        ? 'Every calibration score here is negative, which means every response fell strictly inside its initial '
          + 'interval. A negative threshold therefore pulls both endpoints inwards. If they crossed, the sublevel '
          + 'set would be empty — a legitimate outcome, not a negative width.'
        : mode === 'normalized'
          ? `The normalised scores are ${normalizedScores.map(score => round(score, 3)).join(', ')} and are `
            + `dimensionless; the threshold ${round(normalized.q, 3)} carries no units, and the half-width it `
            + 'produces does, because it is multiplied by a local scale measured in response units.'
          : `The absolute threshold ${round(absolute.q, 3)} is in response units and is added to every point `
            + 'prediction unchanged, so both queries receive the same width whether or not one of them is harder.'} />
  </div>;
}

/* --------------------- F8 · the measured banknote comparison (§7) */

export function MeasuredClassificationFigure() {
  const [method, setMethod] = useState('sigmoid');
  const row = classification.methods[method];
  const report = reliabilityFromRecord(row.reliability, classification.binEdges);
  const geometry = reliabilityGeometry(report);
  const sets = row.testProbabilities.map(probability => classSets([1 - probability, probability], row.q));
  const calibration = [...row.calibrationScores].sort((a, b) => a - b);
  const railGeo = railGeometry(calibration, { finite: true, k: row.rank, n: calibration.length, q: row.q },
    { domain: [0, Math.max(1, ...calibration)] });
  return <div className="cal-figure-block">
    <p className="cal-caption"><strong>Figure 8.</strong> One fixed split of the banknote data, five declared
      procedures. The left plot is the reliability of the probabilities on the {classification.testLabels.length} assessment rows. The
      rail beneath is the eighty CONFORMAL calibration scores with the selected rank marked — a different eighty
      rows, and the reason the threshold is a learned object rather than a property of the world.</p>
    <div className="cal-presets">
      {classification.methodOrder.map(key => <button key={key} type="button"
        className={method === key ? 'is-selected' : undefined}
        onClick={() => setMethod(key)}>{classification.methods[key].label}</button>)}
    </div>
    <div className="cal-figure-row">
      <div>
        <PlotFrame geometry={geometry} caption={`${row.label}: reliability on the assessment rows`}
          xLabel="mean predicted probability of class 1" yLabel="observed fraction of class 1"
          describe={`A square plot with a dashed diagonal and ${geometry.points.length} dots: `
            + geometry.points.map(point =>
              `bin ${point.index + 1} at ${round(point.meanP, 4)} against ${round(point.fractionPositive, 4)} over `
              + `${point.count} rows`).join('; ') + '.'}>
          <line className="cal-diagonal" x1={geometry.diagonal.x1} y1={geometry.diagonal.y1}
            x2={geometry.diagonal.x2} y2={geometry.diagonal.y2} />
          {geometry.points.map(point => <g key={point.index}>
            <line className="cal-gap-line" x1={point.x} y1={point.diagonalY} x2={point.x} y2={point.y} />
            <circle className={`cal-dot ${Math.abs(point.gap) < 1e-15 ? '' : point.gap > 0 ? 'is-above' : 'is-below'}`}
              cx={point.x} cy={point.y} r={4} />
          </g>)}
          {/* The baseline comes from the geometry, which places the band clear
              of the tick-label row; computing it here is what put the bars on
              top of the labels. */}
          {geometry.rail.map(bar => <g key={bar.index}>
            <rect className={`cal-rail-bar${bar.empty ? ' is-empty' : ''}`}
              x={bar.x1 + 1} y={geometry.railBaseline - bar.markerHeight}
              width={Math.max(bar.x2 - bar.x1 - 2, 1)} height={bar.markerHeight} />
            <text className="cal-small" x={bar.countLabelX} y={geometry.countLabelY} textAnchor="middle">
              {bar.count}
            </text>
          </g>)}
        </PlotFrame>
        <KindTag kind="assessment"
          extra={`${report.count} reserved rows. Binned ECE on this partition is ${round(report.ece, 6)}; that is a `
            + 'summary of this binning as much as of these probabilities.'} />
      </div>
      <div>
        <PlotFrame geometry={{ ...railGeo, ticks: null }} axes="x"
          caption={`The ${calibration.length} conformal calibration scores, with rank ${row.rank} marked`}
          xLabel="nonconformity score on the calibration rows" yLabel=""
          describe={`${calibration.length} marks between ${round(calibration[0], 4)} and `
            + `${round(calibration[calibration.length - 1], 4)}, with a line at the `
            + `${row.rank}th smallest, ${round(row.q, 6)}.`}>
          <line className="cal-rail" x1={railGeo.axis.x1} y1={railGeo.axis.y}
            x2={railGeo.axis.x2} y2={railGeo.axis.y} />
          {railGeo.marks.map(mark => <circle key={`${mark.rank}-${mark.score}`}
            className={`cal-rail-mark${mark.selected ? ' is-selected' : ''}`}
            cx={mark.x} cy={railGeo.axis.y} r={mark.selected ? 5 : 2.5} />)}
          {railGeo.markerX !== null && <line className="cal-threshold"
            x1={railGeo.markerX} y1={railGeo.box.top} x2={railGeo.markerX} y2={railGeo.axis.y + 6} />}
          {[0, 0.25, 0.5, 0.75, 1].map(fraction => {
            const [low, high] = railGeo.scales.x.domain;
            const value = low + fraction * (high - low);
            return <text key={fraction} className="cal-small" x={railGeo.scales.x(value)}
              y={railGeo.axis.y + 15} textAnchor="middle">{Number(value.toFixed(3))}</text>;
          })}
        </PlotFrame>
        <KindTag kind="calibration"
          extra={`k = ceil(81 × 0.9) = ${row.rank}, so the threshold is the ${row.rank}th smallest of these eighty `
            + `scores: ${round(row.q, 9)}. Another eighty rows would give another threshold.`} />
      </div>
    </div>
    <Table caption="All five declared procedures on the same fixed split"
      headings={['procedure', 'correct', 'Brier', 'log loss', 'AUC', 'covered', 'mean size',
        'empty', 'one', 'both']}
      rows={classification.methodOrder.map(key => {
        const entry = classification.methods[key];
        return [
          entry.shortLabel, `${entry.correct}/80`, round(entry.brier, 6), round(entry.logLoss, 6),
          round(entry.auc, 4), `${entry.covered}/80`, fixed(entry.meanSetSize, 4),
          String(entry.sizeCounts['0']), String(entry.sizeCounts['1']), String(entry.sizeCounts['2']),
        ];
      })}
      rowClass={index => (classification.methodOrder[index] === classification.primaryPredeclaredMethod
        ? 'is-primary' : undefined)}
      kind="assessment"
      footnote={`The highlighted row is the procedure declared primary before any of these numbers was seen. `
        + `Coverage here is ${classification.methods.sigmoid.covered} of 80 for that procedure, below the 90% the `
        + 'rank was chosen for. That is one realised split and a finite assessment sample; it is not a refutation '
        + 'of a marginal theorem, and the 76 of 80 on the row below it is not proof of one either.'} />
    <Table caption={`${row.label}: what the eighty assessment rows received`}
      headings={['set size', 'rows', 'what that means']}
      rows={[
        ['0', String(row.sizeCounts['0']), 'no class reached the threshold — a legitimate empty set'],
        ['1', String(row.sizeCounts['1']), 'one class — not a label with a 90% chance of being right'],
        ['2', String(row.sizeCounts['2']), 'both classes — the set carries no class distinction at all'],
      ]}
      footnote={`Coverage by true class: `
        + Object.entries(row.classCoverage).map(([label, entry]) =>
          `class ${label} ${entry.covered} of ${entry.n}`).join(', ')
        + `. Marginal coverage of ${row.covered} of 80 averages over both; it does not promise either of them.`
        + ` Set membership here is recomputed in your browser from the stored probabilities and the stored `
        + `threshold under the same weak comparison the offline program used, and reproduces its `
        + `${sets.filter(entry => entry.size === 1).length} singletons exactly.`} />
  </div>;
}

/* ----------------------- F9 · the measured acoustic intervals (§7) */

export function MeasuredIntervalsFigure() {
  const [method, setMethod] = useState('ridge_absolute');
  const entry = regression.methods[method];
  const bounds = intervalsFor(method);
  const order = regression.testY
    .map((value, index) => index)
    .sort((a, b) => regression.testY[a] - regression.testY[b]);
  const width = 320;
  const left = 46;
  const right = 10;
  const rowHeight = 4;
  /* Twenty-eight units of clearance under the last row, not eight: a missed
     target on the bottom row is drawn at a larger radius and reached down
     into the tick-label row. */
  const height = 12 + order.length * rowHeight + 28;
  const lowest = Math.min(...bounds.lower, ...regression.testY);
  const highest = Math.max(...bounds.upper, ...regression.testY);
  const map = value => left + ((value - lowest) / (highest - lowest)) * (width - left - right);
  const titleId = useId();
  const widthPoints = scatterGeometry(
    regression.testFrequencyHz.map((frequency, index) => ({
      x: frequency, y: bounds.upper[index] - bounds.lower[index], index,
      covered: bounds.lower[index] <= regression.testY[index] && regression.testY[index] <= bounds.upper[index],
    })),
    { xLog: true, box: { width: 320, height: 200 }, yDomain: [0, Math.max(...bounds.upper.map((value, index) => value - bounds.lower[index])) * 1.1] },
  );
  const missed = order.filter(index =>
    !(bounds.lower[index] <= regression.testY[index] && regression.testY[index] <= bounds.upper[index]));
  return <div className="cal-figure-block">
    <p className="cal-caption"><strong>Figure 9.</strong> Every one of the {regression.testY.length} reserved
      airfoil rows, drawn to scale in decibels. Rows are sorted by their observed sound pressure so the bands can
      be read; that order is for viewing only and is not a time trajectory. A filled mark outside its band is a
      miss, and there are {missed.length} of them here.</p>
    <div className="cal-presets">
      {regression.methodOrder.map(key => <button key={key} type="button"
        className={method === key ? 'is-selected' : undefined}
        onClick={() => setMethod(key)}>{regression.methods[key].label}</button>)}
    </div>
    <figure className="cal-figure">
      <svg className="cal-plot" viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={titleId}
        style={{ maxWidth: `${width}px` }}>
        <title id={titleId}>
          {`${order.length} horizontal interval segments in decibels for ${entry.label}, sorted by observed sound `
            + `pressure. ${entry.covered} of ${entry.n} observed values fall inside their interval; `
            + `${entry.n - entry.covered} fall outside. Mean width ${round(entry.meanWidth, 4)} dB.`}
        </title>
        {order.map((index, position) => {
          const y = 12 + position * rowHeight;
          const covered = bounds.lower[index] <= regression.testY[index]
            && regression.testY[index] <= bounds.upper[index];
          return <g key={regression.testIds[index]}>
            <line className={`cal-interval${covered ? '' : ' is-missed'}`}
              x1={map(bounds.lower[index])} y1={y} x2={map(bounds.upper[index])} y2={y} />
            <circle className={`cal-target${covered ? '' : ' is-missed'}`}
              cx={map(regression.testY[index])} cy={y} r={covered ? 1.1 : 2} />
          </g>;
        })}
        {/* Only ticks the drawing actually spans. Hard-coded, the 105 dB label
            sat at x = −5.6 for the procedure whose intervals start higher, and
            the viewBox cut it in half. */}
        {[105, 110, 115, 120, 125, 130, 135, 140]
          .filter(value => value >= lowest && value <= highest)
          .map(value => <text key={value} className="cal-small"
            x={map(value)} y={height - 14} textAnchor="middle">{value}</text>)}
        <text className="cal-small" x={(left + width - right) / 2} y={height - 3} textAnchor="middle">
          scaled sound pressure level in dB
        </text>
      </svg>
    </figure>
    <PlotFrame geometry={widthPoints} caption={`${entry.label}: interval width against frequency`}
      xLabel="frequency in Hz, logarithmic" yLabel="interval width in dB"
      describe={`A scatter of ${widthPoints.points.length} points on a base-ten logarithmic frequency axis from `
        + `${Math.min(...regression.testFrequencyHz)} to ${Math.max(...regression.testFrequencyHz)} Hz. `
        + (entry.widthQuantiles[0] === entry.widthQuantiles[4]
          ? `Every width is ${round(entry.meanWidth, 4)} dB, so the points form one horizontal line.`
          : `Widths range from ${round(entry.widthQuantiles[0], 4)} to ${round(entry.widthQuantiles[4], 4)} dB.`)}>
      {/* Only ticks the data actually spans. A 20 kHz label was drawn for a
          domain whose largest observed frequency is lower, so the log scale
          extrapolated it past the right edge of the drawing. */}
      {[200, 500, 1000, 2000, 5000, 10000, 20000]
        .filter(value => value >= widthPoints.xDomain[0] && value <= widthPoints.xDomain[1])
        .map(value => <g key={value}>
        <line className="cal-gridline" x1={widthPoints.scales.x(value)} y1={widthPoints.box.top}
          x2={widthPoints.scales.x(value)} y2={widthPoints.box.height - widthPoints.box.bottom} />
        {/* Sixteen, not thirteen: at thirteen the leftmost frequency label and
            the y-axis zero met at the origin corner and overlapped. */}
        <text className="cal-small" x={widthPoints.scales.x(value)}
          y={widthPoints.box.height - widthPoints.box.bottom + 16} textAnchor="middle">
          {value >= 1000 ? `${value / 1000}k` : value}
        </text>
      </g>)}
      <line className="cal-threshold" x1={widthPoints.scales.x(regression.frequencySplitHz)}
        y1={widthPoints.box.top} x2={widthPoints.scales.x(regression.frequencySplitHz)}
        y2={widthPoints.box.height - widthPoints.box.bottom} />
      {widthPoints.points.map(point => <circle key={point.index}
        className={`cal-target${point.covered ? '' : ' is-missed'}`}
        cx={point.cx} cy={point.cy} r={point.covered ? 1.6 : 2.6} />)}
      {[0, 5, 10, 15, 20].filter(value => value <= widthPoints.yDomain[1]).map(value => <text key={value}
        className="cal-small" x={widthPoints.box.left - 6} y={widthPoints.scales.y(value) + 3} textAnchor="end">
        {value}
      </text>)}
    </PlotFrame>
    <p className="cal-caption">
      The frequency axis is base-ten logarithmic and the vertical line marks {regression.frequencySplitHz} Hz.
      On a linear axis every row below that line would collapse into the first tenth of the drawing, and the slice
      whose coverage actually differs would be unreadable.
    </p>
    <Table caption="All four declared interval procedures on the same fixed split"
      headings={['procedure', 'covered', 'mean dB', 'min dB', 'max dB', 'empty',
        'below 2 kHz', 'from 2 kHz']}
      rows={regression.methodOrder.map(key => {
        const method_ = regression.methods[key];
        return [
          method_.shortLabel, `${method_.covered}/120`, round(method_.meanWidth, 6),
          round(method_.widthQuantiles[0], 4), round(method_.widthQuantiles[4], 4),
          String(method_.emptyCount),
          `${method_.frequencyGroups.below_2000_hz.covered} / ${method_.frequencyGroups.below_2000_hz.n}`,
          `${method_.frequencyGroups.at_least_2000_hz.covered} / ${method_.frequencyGroups.at_least_2000_hz.n}`,
        ];
      })}
      kind="assessment"
      footnote={`Point error is a different question: ridge has a mean absolute error of `
        + `${round(regression.pointMae, 4)} dB against ${round(regression.constantMae, 4)} for the training-mean `
        + 'baseline. Conformalising the quantile intervals raises this sample’s coverage over the unadjusted '
        + 'ones and stays narrower on average than the ridge intervals, while covering fewer rows than they do. '
        + 'Shorter is not automatically better if the misses are the ones the task needed.'} />
    <p className="cal-caption">
      The two frequency slices were declared before the run. For ridge they are{' '}
      {regression.methods.ridge_absolute.frequencyGroups.below_2000_hz.covered} of{' '}
      {regression.methods.ridge_absolute.frequencyGroups.below_2000_hz.n} below 2 kHz and{' '}
      {regression.methods.ridge_absolute.frequencyGroups.at_least_2000_hz.covered} of{' '}
      {regression.methods.ridge_absolute.frequencyGroups.at_least_2000_hz.n} above, against{' '}
      {regression.methods.ridge_absolute.covered} of {regression.methods.ridge_absolute.n} overall. The same
      overall number hides quite different local behaviour, and marginal coverage never promised otherwise.
    </p>
  </div>;
}

/** Endpoints for one interval procedure, rebuilt in the browser from the stored
 *  predictions and thresholds rather than shipped twice. The data verifier
 *  checks that this reconstruction reproduces the offline endpoints exactly. */
function intervalsFor(method) {
  if (method === 'constant') {
    return {
      lower: regression.testY.map(() => regression.trainingMean - regression.qConstant),
      upper: regression.testY.map(() => regression.trainingMean + regression.qConstant),
    };
  }
  if (method === 'ridge_absolute') {
    return {
      lower: regression.pointPredictions.map(value => value - regression.qAbsolute),
      upper: regression.pointPredictions.map(value => value + regression.qAbsolute),
    };
  }
  if (method === 'raw_quantiles') {
    return { lower: regression.rawLower.slice(), upper: regression.rawUpper.slice() };
  }
  return {
    lower: regression.rawLower.map(value => value - regression.qCqr),
    upper: regression.rawUpper.map(value => value + regression.qCqr),
  };
}

/* ------------------------------ F10 · three probabilities, kept apart (§8) */

export function ProbabilityLayersFigure() {
  const beta = orderStatisticCoverageMean(classification.rank, 80);
  return <div className="cal-figure-block">
    <p className="cal-caption"><strong>Figure 10.</strong> Three quantities that a single percentage sign hides.
      The top row averages over calibration samples and future examples; the middle row holds the calibration
      sample fixed; the bottom row counts observed outcomes. Only the first is the marginal theorem's claim.</p>
    <Table caption="What each of the three is averaging over"
      headings={['quantity', 'what is averaged or held fixed', 'what it establishes', 'kind']}
      rows={[
        ['marginal conformal coverage',
          'the calibration sample and the next example, together',
          'the stated lower bound, under the score and sampling assumptions',
          'a population statement'],
        ['coverage given one calibration sample',
          'fresh examples, with this fitted threshold held fixed',
          'a random population property that varies from one calibration sample to the next',
          'a random variable'],
        ['observed assessment coverage',
          'nothing — it is a finite count of recorded outcomes',
          'an estimate, or a measurement of this fixed corpus, with its denominator',
          'a count'],
      ]}
      footnote={'A reported 92 of 100 at a .9 target is an observation, not a proof. A reported 87 of 100 is a '
        + 'reason to examine sampling, data roles, dependence, score versions and shift — not by itself a '
        + 'mathematical refutation.'} />
    <div className="cal-figure-row">
      <div>
        <KindTag kind="population"
          extra={`Under iid continuous scores with no ties and a model fitted independently of the calibration `
            + `sample, the coverage a fixed rank-${beta.alpha} threshold achieves on ${80} scores follows a `
            + `Beta(${beta.alpha}, ${beta.beta}) distribution, whose mean is ${round(beta.mean, 6)}. That is where `
            + 'the variation between calibration samples comes from.'} />
      </div>
      <div>
        <KindTag kind="assessment"
          extra={'A finite assessment set adds binomial variation ON TOP of that, conditional on the fixed '
            + 'threshold. Marginally the indicators share a random threshold and are not independent trials with '
            + 'parameter exactly 1 − alpha. Neither statement describes tied isotonic scores or our '
            + 'finite-corpus sampling scheme, which is why the isotonic threshold of exactly zero is outside '
            + 'both of them.'} />
      </div>
    </div>
  </div>;
}

/* ------------------------- F11 · marginal coverage and a group it fails (§8) */

export function GroupMosaicFigure() {
  const model = mosaicMarginal(fixtures.mosaic);
  const geometry = mosaicGeometry(model, { width: 280, height: 110 });
  const titleId = useId();
  const shift = fixtures.labelShift.prevalences.map(prevalence => ({
    prevalence,
    ppv: positivePredictiveValue(fixtures.labelShift.sensitivity, fixtures.labelShift.falsePositiveRate, prevalence),
  }));
  return <div className="cal-figure-block">
    <p className="cal-caption"><strong>Figure 11.</strong> An exact constructed population. Column width is each
      group's share, filled height is that group's coverage, so the filled area IS the marginal coverage. The
      marginal number is {round(model.marginal, 3)} and half of group B's answers are missed.</p>
    <figure className="cal-figure">
      <svg className="cal-plot" viewBox={`0 0 ${geometry.width + 20} ${geometry.height + 34}`} role="img"
        aria-labelledby={titleId} style={{ maxWidth: `${geometry.width + 20}px` }}>
        <title id={titleId}>
          {`Two columns. ${geometry.groups.map(group =>
            `${group.name} occupies ${round(100 * group.share, 1)} per cent of the width and is filled to `
            + `${round(100 * group.coverage, 1)} per cent of the height`).join('; ')}. `
            + `The filled area is ${round(geometry.coveredArea, 4)} of the whole.`}
        </title>
        {geometry.groups.map(group => <g key={group.name}>
          {group.missedHeight > 0 && <rect className="cal-area-missed" x={group.x + 10} y={group.missedY + 6}
            width={Math.max(group.width - 2, 1)} height={group.missedHeight} />}
          <rect className="cal-area-covered" x={group.x + 10} y={group.coveredY + 6}
            width={Math.max(group.width - 2, 1)} height={group.coveredHeight} />
          <text className="cal-small" x={group.x + 10 + group.width / 2} y={geometry.height + 20}
            textAnchor="middle">{group.name}</text>

        </g>)}
      </svg>
      <figcaption className="cal-caption">
        {geometry.groups.map(group =>
          `${group.name}: ${round(100 * group.share, 0)}% of the population, covered `
          + `${round(100 * group.coverage, 0)}%`).join(' · ')}. Filled area is the marginal coverage,
        {' '}{round(geometry.coveredArea, 3)}.
      </figcaption>
    </figure>
    <Table caption="The weighted calculation, written out"
      headings={['group', 'share of the population', 'coverage within the group', 'contribution to the marginal']}
      rows={[
        ...model.groups.map(group => [
          group.name, round(group.share, 3), round(group.coverage, 3), round(group.contribution, 3),
        ]),
        ['marginal coverage', '1', '—', round(model.marginal, 3)],
      ]}
      kind="population"
      footnote={`A group-conditional or Mondrian construction computes a separate rank inside each predeclared `
        + 'group. The groups and the score must be fixed independently of those groups’ own conformal '
        + 'outcomes, and a small group can need infinity to support a demanding target. A guarantee for several '
        + 'named groups is not a guarantee conditional on every possible input, nor on a slice chosen after '
        + 'inspecting the failures.'} />
    <Table caption="The same class-conditional test behaviour in two populations"
      headings={['prevalence of the event', 'sensitivity', 'false-positive rate',
        'probability the event happened, given a positive result']}
      rows={shift.map(entry => [
        round(entry.prevalence, 3), round(fixtures.labelShift.sensitivity, 3),
        round(fixtures.labelShift.falsePositiveRate, 3),
        ratio(entry.ppv.value, entry.ppv.because),
      ])}
      kind="population"
      footnote={'Nothing about the test changed between the two rows. What changed is the population it is applied '
        + 'to, and with it the meaning of a positive result. A conformal threshold depends on the sampling '
        + 'structure in the same way.'} />
  </div>;
}

/* --------------------------- F12 · the adaptive score sorts the labels (§9) */

export function ApsFigure() {
  const [cutoff, setCutoff] = useState(0.8);
  const sets = apsSets(fixtures.apsProbabilities, cutoff);
  const bars = barGeometry(sets.steps.map(step => ({
    name: `${['A', 'B', 'C'][step.classIndex]} — rank ${step.rank + 1}`, value: step.cumulative,
  })), { maximum: 1 });
  return <div className="cal-figure-block">
    <p className="cal-caption"><strong>Figure 12.</strong> The adaptive score of a class is the cumulative
      probability mass up to it, after sorting classes by descending probability. A class's score therefore
      depends on the competition among labels, not only on its own probability.</p>
    <BarRows caption={`Cumulative mass for probabilities ${fixtures.apsProbabilities.join(', ')}`}
      rows={bars} highlight={0}
      describe={`Three cumulative bars: ${sets.steps.map(step =>
        `${['A', 'B', 'C'][step.classIndex]} reaches ${round(step.cumulative, 3)}`).join(', ')}.`} />
    <div className="cal-controls">
      <Select label="Threshold on the cumulative score" value={String(cutoff)}
        options={[['0.5', '.5'], ['0.7', '.7'], ['0.8', '.8'], ['0.9', '.9'], ['1', '1']]}
        onChange={value => setCutoff(Number(value))} />
    </div>
    <Table caption={`At a threshold of ${cutoff}: the direct sublevel set, and the boundary-expanded variant`}
      headings={['class', 'probability', 'cumulative score', 'direct sublevel set', 'boundary-expanded set']}
      rows={['A', 'B', 'C'].map((name, index) => [
        name, round(fixtures.apsProbabilities[index], 3), round(sets.scores[index], 3),
        sets.direct[index] ? 'in' : 'out', sets.expanded[index] ? 'in' : 'out',
      ])}
      kind="calibration"
      footnote={`The direct set holds ${sets.directSize} class${sets.directSize === 1 ? '' : 'es'}; the expanded `
        + `variant holds ${sets.expandedSize}. ${sets.strictSuperset
          ? 'The expanded set is a strict superset here: it adds the next class whose inclusion would cross the '
            + 'cutoff, which guarantees a non-empty result and is a more conservative construction.'
          : 'They coincide here.'} These are two different procedures. A result labelled only "APS", without its `
        + 'tie and boundary rules, cannot be reproduced; randomized variants need matching conventions at '
        + 'calibration and at prediction time.'} />
  </div>;
}
