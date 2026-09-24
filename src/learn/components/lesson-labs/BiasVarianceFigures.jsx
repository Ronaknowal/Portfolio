import { useState } from 'react';
import {
  averageVariance, brierDecomposition, doubleDescentBranch, doubleDescentNoise, doubleDescentRisk,
  doubleDescentSignalNorm,
  firstCrossing, fixedDesignOptimism, fixtures, hiddenSettingVariance, learningCurveSeries,
  newLocationRisk, optimismDesign, projectionSmoother, restrictionEffect, smootherOptimism,
  thresholdedError, thresholdedErrorAt, trajectorySeries, validationCurveSeries,
} from '../../data/bias-variance-models';
import {
  boostingTrajectory, printedRounds, procedures, provenance, trainSizes, validationCurveRecord,
} from '../../data/bias-variance-data';
import { Field, Plot, Table, fixed, polyline, round } from './BiasVarianceShared.jsx';
import './bias-variance-labs.css';

/* Every SVG string below is kept short deliberately: at the 12px monospace this
   lesson uses, roughly forty characters is the widest label a 320-unit viewBox
   can hold without running past its own frame. Anything longer is prose. */

const series = Object.fromEntries(procedures.map(record => [record.model, learningCurveSeries(record)]));
const order = ['mean', 'ridge', 'tree_leaf1', 'tree_leaf20'];
const stroke = { mean: '#a08a5a', ridge: '#91aecf', tree_leaf1: '#e7b94a', tree_leaf20: '#8eb9a5' };
// Four patterns, not two: Ridge and the leaf-1 tree are the pair whose
// crossover is this figure's headline claim, so they must not be separated
// by colour alone.
const dash = { mean: '2 4', ridge: '9 3 2 3', tree_leaf1: '', tree_leaf20: '6 3' };

/** A short line sample, so a legend names the pattern as well as the colour. */
function LinePattern({ colour, pattern, width = 34 }) {
  return <svg viewBox={`0 0 ${width} 10`} aria-hidden="true">
    <line x1="1" x2={width - 1} y1="5" y2="5" stroke={colour} strokeWidth="2.4" strokeDasharray={pattern || undefined} />
  </svg>;
}

/* ------------------------------------------------------- §2 · F1 */

const deviationParts = [
  { symbol: 'Y − f(x)', name: 'fresh target deviation', note: 'mean zero, and independent of the fitted prediction given x' },
  { symbol: 'f(x) − f̄(x)', name: 'fixed offset', note: 'a constant once the procedure and the input are fixed' },
  { symbol: 'f̄(x) − f̂_D(x)', name: 'training-draw deviation', note: 'mean zero when averaged over repeated training draws' },
];

/** The error splits into centred deviations. */
export function DeviationFigure() {
  const hidden = hiddenSettingVariance();
  const positions = [
    { key: 'prediction', x: 52, label: 'f̂(x)' },
    { key: 'average', x: 132, label: 'f̄(x)' },
    { key: 'truth', x: 212, label: 'f(x)' },
    { key: 'outcome', x: 292, label: 'Y' },
  ];
  const lanes = [
    { title: 'observe X only', value: hidden.givenInputOnly },
    { title: 'observe X and Z', value: hidden.givenInputAndSetting },
  ];
  const laneScale = 120 / hidden.givenInputOnly;
  return <figure className="bv-figure">
    <figcaption>
      <strong>Figure 1 — The error splits into centred deviations.</strong> A schematic number line: the four positions
      are placed for legibility, not measured. Reading right to left, one fresh error is the sum of three named
      differences, and squaring that sum is where the three-term identity comes from.
    </figcaption>
    <svg viewBox="0 0 340 146" role="img" aria-label="A schematic number line with four marked positions: one fitted prediction f-hat of x, the average prediction f-bar of x, the population mean f of x, and one fresh outcome Y. Three labelled spans join neighbouring positions: f-bar minus f-hat is the training-draw deviation, f of x minus f-bar is the fixed offset, and Y minus f of x is the fresh target deviation. Their sum is Y minus f-hat, the whole error. Positions are schematic and no distance is measured.">
      <text x="24" y="16">one fresh error, split</text>
      <line className="bv-axis" x1="30" x2="316" y1="62" y2="62" />
      {positions.map(position => <g key={position.key}>
        <line className="bv-grid" x1={position.x} x2={position.x} y1="36" y2="74" />
        <circle className={position.key === 'outcome' ? 'bv-observation' : 'bv-mark'} cx={position.x} cy="62" r="4.5" />
        <text x={position.x} y="30" textAnchor="middle">{position.label}</text>
      </g>)}
      {[[0, 1, 'training-draw'], [1, 2, 'fixed offset'], [2, 3, 'fresh target']].map(([from, to, label], index) => {
        const y = 92 + index * 18;
        return <g key={label}>
          <line className="bv-flow" x1={positions[from].x} x2={positions[to].x} y1={y} y2={y} />
          {/* End ticks drop below the span: the label sits above it, and a tick
              rising into the text would cross the glyphs. */}
          <line className="bv-flow" x1={positions[from].x} x2={positions[from].x} y1={y} y2={y + 5} />
          <line className="bv-flow" x1={positions[to].x} x2={positions[to].x} y1={y} y2={y + 5} />
          <text x={(positions[from].x + positions[to].x) / 2} y={y - 5} textAnchor="middle">{label}</text>
        </g>;
      })}
    </svg>
    <p className="bv-caption">
      Positions left to right: one fitted prediction, the average prediction over training draws, the population mean at
      this input, and one fresh outcome. The three bracketed spans are the three deviations. Nothing here is a measured
      distance; this diagram encodes the algebra.
    </p>
    <Table caption="The three deviations, and what makes each one behave as it does"
      headings={['deviation', 'name', 'why it behaves this way']}
      rows={deviationParts.map(part => [part.symbol, part.name, part.note])} />
    <div className="bv-panel">
      <h4>Squaring the sum: three squares and three cross terms</h4>
      <Table caption="Every product in the expansion. The diagonal supplies the three terms of the identity; the off-diagonal products have expectation zero under the stated centring and independence, so they are named rather than quietly dropped."
        headings={['×', 'Y − f', 'f − f̄', 'f̄ − f̂']}
        rows={[
          ['Y − f', 'σ²(x): target noise', 'zero: a mean-zero factor times a constant', 'zero: independent, both mean zero'],
          ['f − f̄', 'zero: the mirror of the cell above', 'squared bias', 'zero: a constant times a mean-zero deviation'],
          ['f̄ − f̂', 'zero: independent, both mean zero', 'zero: the mirror of the cell above', 'prediction variance'],
        ]} />
      <p>
        Each distinct mixed product appears in two off-diagonal cells, so six cross terms vanish and three squares remain. The zeros are
        consequences of two stated conditions: both the fresh target deviation and the training-draw deviation have mean
        zero, and the fresh target is independent of the trained prediction given the input. Change either condition and
        the cross terms come back.
      </p>
    </div>
    <div className="bv-panel">
      <h4>“Irreducible” is relative to what you measure</h4>
      <svg viewBox="0 0 340 116" role="img" aria-label={`Two information lanes drawn as bars on one shared scale. Observing the input X alone leaves the hidden setting Z plus the independent noise, a remaining variance of ${hidden.givenInputOnly}. Observing X and Z leaves the independent noise only, a remaining variance of ${hidden.givenInputAndSetting}. The difference, ${hidden.explainedByMeasuringSetting}, is the part measuring Z explains.`}>
        {lanes.map((lane, index) => {
          const y = 10 + index * 42;
          return <g key={lane.title}>
            <rect className="bv-lane" x="6" y={y} width="112" height="28" rx="3" />
            <text x="12" y={y + 18}>{lane.title}</text>
            <path className="bv-flow" d={`M118,${y + 14} L128,${y + 14}`} />
            <rect className="bv-lane is-fold" x="130" y={y} width={(lane.value * laneScale).toFixed(1)} height="28" rx="3" />
            <text x={136 + lane.value * laneScale} y={y + 18}>{lane.value}</text>
          </g>;
        })}
        <text x="6" y="108">bar length = remaining variance</text>
      </svg>
      <p>
        With Y = X + Z + ε, a hidden setting Z equally likely −1 or +1 and independent noise of variance
        {' '}{hidden.noiseVariance}, the best mean predictor given X alone leaves {hidden.givenInputOnly}. Measuring Z
        first leaves {hidden.givenInputAndSetting}. The difference {hidden.explainedByMeasuringSetting} is the variance of
        the conditional mean — the second term in the law of total variance — and it is exactly what observing Z can
        explain here. The new feature did not refute the old floor; it defined a better-informed problem.
      </p>
    </div>
  </figure>;
}

/* ------------------------------------------------------- §§4–5 · F2 */

/** Fit size is a count inside each fold. */
export function SplitFigure() {
  const rowScale = 300 / provenance.rows;
  const developmentWidth = provenance.developmentRows * rowScale;
  const reservedWidth = provenance.reservedRows * rowScale;
  const subsetScale = 144 / provenance.foldTrainRows;
  return <figure className="bv-figure">
    <figcaption>
      <strong>Figure 2 — A requested size is a count of fitted rows inside each fold.</strong> Bars are proportional to
      actual row counts and every count is printed beside them. Requesting 900 means 900 fitted rows in
      {' '}<em>each</em> fold, not 900 rows shared between five folds.
    </figcaption>
    <div className="bv-panel">
      <h4>1. The supplied collection, split once</h4>
      <svg viewBox="0 0 340 122" role="img" aria-label={`A proportional bar of ${provenance.rows} retained rows splits into ${provenance.developmentRows} development rows and ${provenance.reservedRows} reserved rows. The development bar then divides into ${provenance.folds} equal folds, each contributing ${provenance.foldValidationRows} validation rows. An arrow leaving the reserved block is crossed out: no fitted prediction and no score comes from it.`}>
        <rect className="bv-lane" x="18" y="10" width="300" height="22" rx="3" />
        <text x="22" y="25">{provenance.rows.toLocaleString('en-US')} retained rows</text>
        <rect className="bv-lane is-fold" x="18" y="44" width={developmentWidth.toFixed(1)} height="22" rx="3" />
        <text x="22" y="59">{provenance.developmentRows.toLocaleString('en-US')} development</text>
        <rect className="bv-lane is-reserved" x={(18 + developmentWidth).toFixed(1)} y="44" width={reservedWidth.toFixed(1)} height="22" rx="3" />
        {Array.from({ length: provenance.folds }, (_, index) => (
          <g key={index}>
            <rect className={index === 0 ? 'bv-lane is-fold' : 'bv-lane'}
              x={(18 + index * developmentWidth / provenance.folds).toFixed(1)} y="76"
              width={(developmentWidth / provenance.folds - 3).toFixed(1)} height="20" rx="2" />
            <text x={(22 + index * developmentWidth / provenance.folds).toFixed(1)} y="90">f{index + 1}</text>
          </g>
        ))}
        <text x={(18 + developmentWidth + reservedWidth / 2).toFixed(1)} y="90" textAnchor="middle">{provenance.reservedRows}</text>
        <line className="bv-flow is-blocked" x1={(20 + developmentWidth).toFixed(1)} x2="312" y1="59" y2="59" />
        <line className="bv-zero-mark" x1="296" x2="308" y1="53" y2="65" />
        <line className="bv-zero-mark" x1="296" x2="308" y1="65" y2="53" />
        <text x="18" y="116">reserved: no fit, no score</text>
      </svg>
      <p>
        {provenance.rows.toLocaleString('en-US')} rows split with seed {provenance.splitSeed} into
        {' '}{provenance.developmentRows.toLocaleString('en-US')} development rows and {provenance.reservedRows} reserved
        rows, then five shuffled folds with seed {provenance.foldSeed} inside development. Each fold contributes
        {' '}{provenance.foldValidationRows} validation rows and leaves {provenance.foldTrainRows} available for fitting.
        The crossed arrow marks the boundary this lesson keeps: nothing is fitted to, predicted for or scored on the
        reserved block.
      </p>
    </div>
    <div className="bv-panel">
      <h4>2. Inside one fold: nested fitting subsets, one unchanging evaluator</h4>
      <svg viewBox="0 0 340 184" role="img" aria-label={`Inside fold 1, ${provenance.foldTrainRows} training rows are available and ${provenance.foldValidationRows} are held out. Five nested bars of ${trainSizes.join(', ')} rows each reach their own freshly fitted model, and every one of those models is scored against the same ${provenance.foldValidationRows} validation rows.`}>
        <rect className="bv-lane is-fold" x="32" y="8" width="144" height="20" rx="3" />
        <text x="36" y="22">fold 1: {provenance.foldTrainRows} available</text>
        {trainSizes.map((size, index) => {
          const width = size * subsetScale;
          const y = 38 + index * 24;
          return <g key={size}>
            {/* The count sits in a gutter to the left of its bar. Printed at the
                bar's right edge it would land on the flow curve leaving it. */}
            <text x="26" y={y + 13} textAnchor="end">{size}</text>
            <rect className="bv-lane" x="32" y={y} width={width.toFixed(1)} height="18" rx="2" />
            <path className="bv-flow" d={`M${(32 + width).toFixed(1)},${y + 9} C214,${y + 9} 214,150 236,150`} />
          </g>;
        })}
        <rect className="bv-lane is-reserved" x="238" y="136" width="94" height="28" rx="3" />
        <text x="242" y="154">{provenance.foldValidationRows} held out</text>
        <text x="18" y="178">one evaluator scores all five fits</text>
      </svg>
      <p>
        Each bar is a fitting subset of {trainSizes.join(', ')} rows, and each reaches a model fitted from scratch. A
        larger subset contains the smaller ones, because the sizes are prefixes of one permutation of that fold&rsquo;s
        training rows. The held-out {provenance.foldValidationRows} rows never change with size, so a comparison across
        sizes cannot quietly become a comparison of different evaluation populations. Preprocessing lives inside each
        fit: the Ridge procedure learns its standardization from the fitted subset alone.
      </p>
    </div>
    <Table caption="Exact counts, so no bar has to be measured"
      headings={['quantity', 'rows']}
      rows={[
        ['retained in the supplied file', provenance.rows.toLocaleString('en-US')],
        ['development pool', provenance.developmentRows.toLocaleString('en-US')],
        ['reserved, unused by this lesson', String(provenance.reservedRows)],
        ['validation rows in each fold', String(provenance.foldValidationRows)],
        ['available training rows in each fold', String(provenance.foldTrainRows)],
        ['requested fitted sizes', trainSizes.join(', ')],
        ['fits performed', `${order.length} × ${trainSizes.length} × ${provenance.folds} = ${order.length * trainSizes.length * provenance.folds}`],
      ]} />
  </figure>;
}

/* ------------------------------------------------------- §5 · F3 */

const sizeDomain = [30, 930];
// All panels share a scale covering individual folds as well as their means.
const maximumFoldError = Math.max(...Object.values(series).flatMap(record =>
  [...record.training, ...record.validation].flatMap(entry => entry.folds)));
const errorRange = [0, Math.max(50, Math.ceil(maximumFoldError * 1.03 / 10) * 10)];
const errorTicks = [0, errorRange[1] / 2, errorRange[1]];

function CurvePanel({ record, showFolds, baselineMean }) {
  return <div className="bv-panel">
    <h4>{record.label}</h4>
    <Plot width={300} height={190} domain={sizeDomain} range={errorRange}
      padding={{ left: 40, right: 12, top: 14, bottom: 32 }}
      ticks={[60, 240, 480, 900]} valueTicks={errorTicks}
      formatTick={value => String(value)} formatValue={value => String(value)}
      describe={`${record.label}: mean squared error against fitted rows per fold, on axes shared with the other three panels. Validation means are ${record.validationMeans.map((value, index) => `${value.toFixed(4)} at ${record.sizes[index]} rows`).join(', ')}. Training means are ${record.trainingMeans.map((value, index) => `${value.toFixed(4)} at ${record.sizes[index]} rows`).join(', ')}.`}>
      {(scaleX, scaleY) => <>
        <line className="bv-curve is-baseline" x1={scaleX(sizeDomain[0])} x2={scaleX(sizeDomain[1])}
          y1={scaleY(baselineMean)} y2={scaleY(baselineMean)} />
        {showFolds && record.validation.flatMap(entry => entry.folds.map((value, index) => (
          <circle key={`v${entry.size}-${index}`} cx={scaleX(entry.size)} cy={scaleY(value)} r="1.8" fill="#6d7d75" />
        )))}
        {showFolds && record.training.flatMap(entry => entry.folds.map((value, index) => (
          <circle key={`t${entry.size}-${index}`} cx={scaleX(entry.size)} cy={scaleY(value)} r="1.8" fill="#4f6273" />
        )))}
        <polyline className="bv-curve is-training"
          points={polyline(record.sizes.map((size, index) => [size, record.trainingMeans[index]]), scaleX, scaleY)} />
        <polyline className="bv-curve is-validation"
          points={polyline(record.sizes.map((size, index) => [size, record.validationMeans[index]]), scaleX, scaleY)} />
        {record.sizes.map((size, index) => <g key={size}>
          <rect x={scaleX(size) - 2.8} y={scaleY(record.trainingMeans[index]) - 2.8} width="5.6" height="5.6" fill="#91aecf" />
          <circle className="bv-mark" cx={scaleX(size)} cy={scaleY(record.validationMeans[index])} r="2.8" />
        </g>)}
      </>}
    </Plot>
    <p>
      Validation {record.validationMeans[0].toFixed(4)} at {record.sizes[0]} fitted rows down to
      {' '}{record.validationMeans.at(-1).toFixed(4)} at {record.sizes.at(-1)}.
      {record.trainingExactlyZero
        ? ' Its training error is exactly zero at every inspected size, so the dashed line lies along the zero axis and its square markers straddle it. The axis starts at zero: no negative error is being drawn.'
        : ` Training ${record.trainingMeans[0].toFixed(4)} to ${record.trainingMeans.at(-1).toFixed(4)}, so the gap at the largest size is ${record.finalGap.toFixed(4)}.`}
    </p>
  </div>;
}

/** Real learning curves, and a crossover. */
export function LearningCurveFigure() {
  const [showFolds, setShowFolds] = useState(true);
  const baselineMean = series.mean.validationMeans.at(-1);
  const crossing = firstCrossing(series.tree_leaf1, series.ridge);
  const restriction = restrictionEffect(series.tree_leaf20, series.tree_leaf1);
  return <figure className="bv-figure">
    <figcaption>
      <strong>Figure 3 — Measured learning curves, with an actual crossover.</strong> Five fitted sizes, five folds and
      four prespecified procedures on the airfoil development pool. These are the recorded values: no curve is drawn
      between fitted sizes as though it had been measured there, and the fold spread is a descriptive diagnostic rather
      than a confidence band.
    </figcaption>
    <div className="bv-panel">
      <h4>All four validation means on one axis</h4>
      <Plot width={340} height={210} domain={sizeDomain} range={errorRange}
        padding={{ left: 42, right: 14, top: 16, bottom: 34 }}
        ticks={[60, 240, 480, 900]} valueTicks={errorTicks}
        formatTick={value => String(value)} formatValue={value => String(value)}
        describe={`Mean validation MSE in squared decibels against fitted rows per fold, for four procedures. ${order.map(key => `${series[key].label}: ${series[key].validationMeans.map((value, index) => `${value.toFixed(4)} at ${trainSizes[index]}`).join(', ')}`).join('. ')}. The unrestricted tree starts worst and ends best, first passing Ridge at ${crossing} fitted rows.`}>
        {(scaleX, scaleY) => <>
          {order.map(key => (
            <polyline key={key} className="bv-curve" fill="none" stroke={stroke[key]} strokeDasharray={dash[key] || undefined}
              style={{ strokeWidth: 2.2 }}
              points={polyline(trainSizes.map((size, index) => [size, series[key].validationMeans[index]]), scaleX, scaleY)} />
          ))}
          {order.flatMap(key => trainSizes.map((size, index) => (
            <circle key={`${key}-${size}`} cx={scaleX(size)} cy={scaleY(series[key].validationMeans[index])} r="2.6" fill={stroke[key]} />
          )))}
          <line className="bv-stem is-probe" x1={scaleX(crossing)} x2={scaleX(crossing)} y1={scaleY(errorRange[1])} y2={scaleY(errorRange[0])} />
          <text className="bv-halo" x={scaleX(crossing) + 6} y="24">tree better from {crossing}</text>
        </>}
      </Plot>
      <div className="bv-legend">
        {order.map(key => <span key={key}><LinePattern colour={stroke[key]} pattern={dash[key]} /> {series[key].label}</span>)}
      </div>
      <p>
        At {trainSizes[0]} fitted rows the unrestricted tree is worse than Ridge
        ({series.tree_leaf1.validationMeans[0].toFixed(4)} against {series.ridge.validationMeans[0].toFixed(4)}); at
        {' '}{trainSizes.at(-1)} it is much better ({series.tree_leaf1.validationMeans.at(-1).toFixed(4)} against
        {' '}{series.ridge.validationMeans.at(-1).toFixed(4)}). The dotted line is the training-mean baseline, which stays
        flat: more rows cannot help a procedure that ignores its inputs.
      </p>
      <p>
        The segments between fitted sizes are joins, not measurements, so the place where two lines appear to meet is not
        itself a measured crossing. The marked size is the first <em>inspected</em> one at which the tree's mean is the
        lower of the two: at {trainSizes[1]} fitted rows Ridge was still ahead
        ({series.ridge.validationMeans[1].toFixed(4)} against {series.tree_leaf1.validationMeans[1].toFixed(4)}), and at
        {' '}{crossing} it was not.
      </p>
    </div>
    <div className="bv-legend">
      <span><LinePattern colour="#e7b94a" pattern="" /> validation mean, round markers</span>
      <span><LinePattern colour="#91aecf" pattern="4 3" /> training mean, square markers</span>
      <span><LinePattern colour="#a08a5a" pattern="2 4" /> the baseline level, {baselineMean.toFixed(4)}</span>
      <span>small dots: the five individual folds</span>
    </div>
    <div className="bv-buttons">
      <button type="button" onClick={() => setShowFolds(value => !value)}>
        {showFolds ? 'Hide the individual fold values' : 'Show the individual fold values'}
      </button>
      <span>inspection only; nothing here is graded</span>
    </div>
    <div className="bv-panels is-pair">
      {order.map(key => <CurvePanel key={key} record={series[key]} showFolds={showFolds} baselineMean={baselineMean} />)}
    </div>
    {/* The validation means are tabulated in the prose immediately above this
        figure; this table supplies the other line each panel draws. */}
    <Table caption="Recorded mean training MSE in squared decibels, the dashed line in each panel"
      headings={['fitted rows per fold', ...order.map(key => series[key].label)]}
      rows={trainSizes.map((size, index) => [String(size), ...order.map(key => series[key].trainingMeans[index].toFixed(4))])} />
    <p>
      Requiring twenty items per leaf helps the tree only at {restriction.helps.join(', ')} fitted rows and hurts at
      {' '}{restriction.hurts.join(', ')}. At the largest size Ridge has training and validation MSE
      {' '}{series.ridge.trainingMeans.at(-1).toFixed(4)} and {series.ridge.validationMeans.at(-1).toFixed(4)}, while the
      restricted tree has {series.tree_leaf20.trainingMeans.at(-1).toFixed(4)} and
      {' '}{series.tree_leaf20.validationMeans.at(-1).toFixed(4)}: the smaller gap belongs to the worse predictor.
    </p>
    <details>
      <summary>Every recorded fold value</summary>
      <Table caption="Five folds per fitted size, training and validation, in squared decibels. These folds share training rows, so a standard deviation across them is not a confidence interval for a difference."
        headings={['procedure', 'fitted rows', 'training folds', 'training mean', 'validation folds', 'validation mean']}
        rows={order.flatMap(key => series[key].sizes.map((size, index) => [
          series[key].label, String(size),
          series[key].training[index].folds.map(value => value.toFixed(3)).join(', '),
          series[key].trainingMeans[index].toFixed(4),
          series[key].validation[index].folds.map(value => value.toFixed(3)).join(', '),
          series[key].validationMeans[index].toFixed(4),
        ]))} scroll />
    </details>
    <p className="bv-caption">
      Source: the {provenance.name} collection, {provenance.rows.toLocaleString('en-US')} rows, licensed
      {' '}<a href={provenance.licenseUrl}>{provenance.license}</a>; this page serves
      {' '}<a href={provenance.file} download>its own unchanged copy</a> of the data file and its
      {' '}<a href={provenance.attribution}>attribution</a>. This is a row-level diagnostic of the supplied collection
      under one declared development protocol, and the {provenance.reservedRows} reserved rows received no prediction and
      no score.
    </p>
  </figure>;
}

/* ------------------------------------------------------- §5 · F4 */

/** Restriction and training time have different axes. */
export function AxesFigure() {
  const leaf = validationCurveSeries(validationCurveRecord);
  const trace = trajectorySeries(boostingTrajectory);
  const [inspected, setInspected] = useState(trace.bestRound);
  const settingDomain = [-0.4, leaf.settings.length - 0.6];
  return <figure className="bv-figure">
    <figcaption>
      <strong>Figure 4 — A changed setting and a longer training run are different horizontal axes.</strong> Two charts,
      each labelled with its own split. They are not two views of one experiment, and neither of them is a learning
      curve.
    </figcaption>
    <div className="bv-panels is-pair">
      <div className="bv-panel">
        <h4>Validation curve: one restriction, at {validationCurveRecord.fitRowsPerFold} fitted rows per fold</h4>
        <Plot width={300} height={196} domain={settingDomain} range={[0, 25]}
          padding={{ left: 40, right: 14, top: 16, bottom: 34 }}
          ticks={leaf.settings.map((setting, index) => index)} valueTicks={[0, 12.5, 25]}
          formatTick={value => String(leaf.settings[Math.round(value)])}
          formatValue={value => String(value)}
          describe={`Mean squared error against minimum leaf size on an evenly spaced categorical axis with the settings ${leaf.settings.join(', ')}. Validation means are ${leaf.validationMeans.map(value => value.toFixed(4)).join(', ')} and training means are ${leaf.trainingMeans.map(value => value.toFixed(4)).join(', ')}. Both rise as the restriction tightens, so the lowest validation value is at the least restricted setting, ${leaf.bestSetting}.`}>
          {(scaleX, scaleY) => <>
            <polyline className="bv-curve is-training"
              points={polyline(leaf.trainingMeans.map((value, index) => [index, value]), scaleX, scaleY)} />
            <polyline className="bv-curve is-validation"
              points={polyline(leaf.validationMeans.map((value, index) => [index, value]), scaleX, scaleY)} />
            {leaf.validationMeans.map((value, index) => <g key={index}>
              <rect x={scaleX(index) - 2.8} y={scaleY(leaf.trainingMeans[index]) - 2.8} width="5.6" height="5.6" fill="#91aecf" />
              <circle className="bv-mark" cx={scaleX(index)} cy={scaleY(value)} r="3" />
            </g>)}
          </>}
        </Plot>
        <div className="bv-legend">
          <span><LinePattern colour="#e7b94a" pattern="" /> validation mean</span>
          <span><LinePattern colour="#91aecf" pattern="4 3" /> training mean</span>
        </div>
        <p>
          Horizontal axis: minimum leaf size, evenly spaced because these six values are a chosen grid rather than a
          measured scale. {leaf.restrictionImproves
            ? 'Some restricted setting does improve on the unrestricted one here.'
            : `No inspected restriction improves the score: the best of the six is the least restricted, ${leaf.bestSetting}.`}
          {' '}Its value {leaf.validationMeans[0].toFixed(4)} differs from the learning curve&rsquo;s
          {' '}{series.tree_leaf1.validationMeans.at(-1).toFixed(4)} because this run fits
          {' '}{validationCurveRecord.fitRowsPerFold} rows per fold rather than {trainSizes.at(-1)}.
        </p>
      </div>
      <div className="bv-panel">
        <h4>Training trajectory: one fit, {trace.lastRound} rounds, on its own {boostingTrajectory.fitRows}/{boostingTrajectory.monitorRows} split</h4>
        <Plot width={300} height={196} domain={[1, trace.lastRound]} range={[0, 50]}
          padding={{ left: 40, right: 20, top: 16, bottom: 34 }}
          ticks={[1, 30, 60, 90, 120]} valueTicks={[0, 25, 50]}
          formatTick={value => String(Math.round(value))} formatValue={value => String(value)}
          describe={`Mean squared error against boosting round from 1 to ${trace.lastRound}, for the fitted rows and for the separate monitoring rows. Round 1 gives ${trace.train[0].toFixed(4)} and ${trace.monitor[0].toFixed(4)}; round ${trace.lastRound} gives ${trace.train.at(-1).toFixed(4)} and ${trace.monitor.at(-1).toFixed(4)}. The lowest monitoring value over all ${trace.lastRound} recorded rounds is ${trace.bestMonitor.toFixed(4)} at round ${trace.bestRound}. The monitoring trace rises again at ${trace.risingRounds.length} individual rounds, the largest single rise being ${trace.largestRise.toFixed(4)}.`}>
          {(scaleX, scaleY) => <>
            <polyline className="bv-curve is-training"
              points={polyline(trace.rounds.map((number, index) => [number, trace.train[index]]), scaleX, scaleY)} />
            <polyline className="bv-curve is-validation"
              points={polyline(trace.rounds.map((number, index) => [number, trace.monitor[index]]), scaleX, scaleY)} />
            {printedRounds.map(number => <g key={number}>
              <rect x={scaleX(number) - 2.6} y={scaleY(trace.train[number - 1]) - 2.6} width="5.2" height="5.2" fill="#91aecf" />
              <circle className="bv-mark" cx={scaleX(number)} cy={scaleY(trace.monitor[number - 1])} r="2.8" />
            </g>)}
            <line className="bv-stem is-probe" x1={scaleX(inspected)} x2={scaleX(inspected)} y1={scaleY(50)} y2={scaleY(0)} />
            <circle className="bv-mark is-hollow" cx={scaleX(inspected)} cy={scaleY(trace.monitor[inspected - 1])} r="5" />
            <text className="bv-halo" x={scaleX(trace.bestRound) - 4} y="26" textAnchor="end">lowest observed</text>
          </>}
        </Plot>
        <p>
          The monitoring error is still falling when the run stops: its lowest value over all {trace.lastRound} recorded
          rounds is {trace.bestMonitor.toFixed(4)} at round {trace.bestRound}, which is the last one. No stopping point
          followed by sustained deterioration was observed in this range, so none is drawn.
        </p>
        <p>
          The trace is not perfectly monotone. It rises again at {trace.risingRounds.length} individual rounds
          ({trace.risingRounds.join(', ')}), but by at most {trace.largestRise.toFixed(4)} squared decibels — under a
          thousandth of this vertical axis, so those rises are far too small to see at this scale rather than being
          smoothed away. Read round {trace.risingRounds[6] - 1} and then round {trace.risingRounds[6]} in the selector
          below to check one: {trace.monitor[trace.risingRounds[6] - 2].toFixed(4)} becomes
          {' '}{trace.monitor[trace.risingRounds[6] - 1].toFixed(4)}. A rise you cannot see is still not an overfitting
          turn; what would be one is a sustained climb, and there is none here.
        </p>
      </div>
    </div>
    <div className="bv-controls">
      <Field label="Inspect one recorded round">
        <select value={inspected} onChange={event => setInspected(Number(event.target.value))}>
          {trace.rounds.map(number => <option key={number} value={number}>round {number}</option>)}
        </select>
      </Field>
    </div>
    <p className="bv-readout" aria-live="polite">
      Round {inspected}: training MSE {trace.train[inspected - 1].toFixed(4)}, monitoring MSE
      {' '}{trace.monitor[inspected - 1].toFixed(4)}, a gap of
      {' '}{(trace.monitor[inspected - 1] - trace.train[inspected - 1]).toFixed(4)}. This is an explanatory trace through
      recorded values, not a second investigation.
    </p>
    <div className="bv-panels is-pair">
      <Table caption="Validation curve: mean squared error by minimum leaf size, five folds of 960 fitted rows"
        headings={['minimum leaf size', 'training mean', 'validation mean']}
        rows={leaf.settings.map((setting, index) => [
          String(setting), leaf.trainingMeans[index].toFixed(4), leaf.validationMeans[index].toFixed(4),
        ])} />
      <Table caption={`Boosting trajectory: the rounds the program prints, out of ${trace.lastRound}`}
        headings={['round', 'training MSE', 'monitoring MSE']}
        rows={printedRounds.map(number => [
          String(number), trace.train[number - 1].toFixed(4), trace.monitor[number - 1].toFixed(4),
        ])} />
    </div>
    <p className="bv-caption">
      The two charts use different evaluation partitions: the validation curve reuses the five development folds, while
      the trajectory uses its own {boostingTrajectory.fitRows}/{boostingTrajectory.monitorRows} split of the same
      development pool, with seeds {provenance.stageSplitSeed} and {provenance.stageSeed}. Their numbers are not a
      ranking against each other.
    </p>
  </figure>;
}

/* ------------------------------------------------------- §7 · F5 */

const optimism = fixedDesignOptimism(6, 2, fixtures.optimismNoiseVariance);
const design = optimismDesign();
const hat = projectionSmoother(design);
const correctMean = fixtures.optimismInputs.map(value => 1 + 2 * value);
const curvedMean = fixtures.optimismInputs.map(value => 1 + 2 * value + value * value);
const correctAccount = smootherOptimism(hat.matrix, correctMean, fixtures.optimismNoiseVariance);
const curvedAccount = smootherOptimism(hat.matrix, curvedMean, fixtures.optimismNoiseVariance);

/** Same inputs, new outcomes. */
export function SameInputsFigure() {
  const barScale = 28;
  const locations = [0, 1.5, 2.5, 4].map(value => ({
    value, ...newLocationRisk(design, [1, value], fixtures.optimismNoiseVariance),
    inDesign: fixtures.optimismInputs.includes(value),
  }));
  return <figure className="bv-figure">
    <figcaption>
      <strong>Figure 5 — Same inputs, new outcomes.</strong> The design matrix is the same on both sides; only the
      outcomes change. On the left the fit is scored against the very noise it used; on the right the noise in the
      outcomes is independent of the fit. Every number is exact for n&nbsp;=&nbsp;{optimism.n},
      p&nbsp;=&nbsp;{optimism.p} and σ²&nbsp;=&nbsp;{optimism.noiseVariance}.
    </figcaption>
    <div className="bv-panels is-pair">
      <div className="bv-panel">
        <h4>Two lanes, one shared X</h4>
        <svg viewBox="0 0 320 168" role="img" aria-label={`Two mirrored lanes with the same fixed design X. On the left, training noise epsilon enters the outcomes, the fit produces H y and the residual is the identity minus H applied to epsilon, giving expected training mean squared error ${optimism.trainingMse.toFixed(6)}. On the right, independent new noise epsilon prime enters fresh outcomes at the same inputs while the prediction still depends on the original epsilon, so the error is epsilon prime minus H epsilon and the expected mean squared error is ${optimism.newOutcomeMse.toFixed(6)}.`}>
          {[
            { x: 6, heading: 'same outcomes', noise: 'ε', error: '(I − H)ε', value: optimism.trainingMse },
            { x: 168, heading: 'fresh outcomes', noise: 'ε′', error: 'ε′ − Hε', value: optimism.newOutcomeMse },
          ].map(lane => <g key={lane.x}>
            <text x={lane.x + 4} y="14">{lane.heading}</text>
            <rect className="bv-lane is-fold" x={lane.x} y="22" width="146" height="24" rx="3" />
            <text x={lane.x + 6} y="38">X: fixed, shared</text>
            <path className="bv-flow" d={`M${lane.x + 73},48 L${lane.x + 73},62`} />
            <rect className="bv-lane" x={lane.x} y="64" width="146" height="24" rx="3" />
            <text x={lane.x + 6} y="80">noise {lane.noise} in y</text>
            <path className="bv-flow" d={`M${lane.x + 73},90 L${lane.x + 73},104`} />
            <rect className="bv-lane" x={lane.x} y="106" width="146" height="24" rx="3" />
            <text x={lane.x + 6} y="122">error {lane.error}</text>
            <text x={lane.x + 6} y="150">MSE {lane.value.toFixed(6)}</text>
          </g>)}
          <line className="bv-grid" x1="160" x2="160" y1="6" y2="160" />
        </svg>
        <p>
          The prediction on the right was still built from the original training noise ε; only the outcomes it is scored
          against are new. That single change turns σ²(1 − p/n) into σ²(1 + p/n).
        </p>
      </div>
      <div className="bv-panel">
        <h4>Three quantities from the same fitted model</h4>
        <svg viewBox="0 0 300 160" role="img" aria-label={`Three bars on one shared scale: expected training mean squared error ${optimism.trainingMse.toFixed(6)}, expected fresh-outcome mean squared error at the same inputs ${optimism.newOutcomeMse.toFixed(6)}, and average fitted-prediction variance ${optimism.predictionVariance.toFixed(6)}. The gap between the first two is ${optimism.gap.toFixed(6)}, which is exactly twice the third.`}>
          {[
            { name: 'training', value: optimism.trainingMse, fill: '#91aecf' },
            { name: 'fresh outcomes', value: optimism.newOutcomeMse, fill: '#e7b94a' },
            { name: 'fit variance', value: optimism.predictionVariance, fill: '#8eb9a5' },
          ].map((bar, index) => <g key={bar.name}>
            <text x="4" y={20 + index * 40}>{bar.name}</text>
            <rect x="4" y={26 + index * 40} width={(bar.value * barScale).toFixed(1)} height="16" fill={bar.fill} />
            <text x={(bar.value * barScale + 10).toFixed(1)} y={39 + index * 40}>{bar.value.toFixed(6)}</text>
          </g>)}
          <line className="bv-axis" x1={(optimism.trainingMse * barScale + 4).toFixed(1)}
            x2={(optimism.newOutcomeMse * barScale + 4).toFixed(1)} y1="134" y2="134" />
          <text x="4" y="148">gap {optimism.gap.toFixed(6)} = 2 × fit variance</text>
        </svg>
        <p>
          One shared scale, so the lengths are comparable. The gap between the first two bars is
          {' '}{optimism.gap.toFixed(6)}, exactly twice the third. A train/validation gap is therefore not a measurement
          of prediction variance, even in this most favourable setting.
        </p>
      </div>
    </div>
    <div className="bv-panels is-pair">
      <div className="bv-panel">
        <h4>Why: the fit is a projection</h4>
        <svg viewBox="0 0 300 140" role="img" aria-label="A qualitative sketch. A horizontal line represents the column space of X. A segment runs from a point on that line up to the observed outcome vector y; the foot of that segment is the fitted vector H y, and the joining segment, the residual, meets the column space at a right angle, marked with a small square.">
          <line className="bv-axis" x1="24" x2="276" y1="104" y2="104" />
          <text x="24" y="122">column space of X, p = {optimism.p}</text>
          <line className="bv-flow" x1="60" y1="104" x2="186" y2="34" />
          <line className="bv-stem is-probe" x1="186" y1="34" x2="186" y2="104" />
          <rect x="178" y="96" width="8" height="8" fill="none" stroke="#4a5650" />
          <circle className="bv-observation" cx="186" cy="34" r="4" />
          <circle className="bv-mark" cx="186" cy="104" r="4" />
          <text x="194" y="32">y observed</text>
          <text x="194" y="100">ŷ = Hy</text>
          {/* Beside the vertical segment it names, not across the diagonal. */}
          <text x="194" y="70">(I − H)y</text>
        </svg>
        <p>
          A qualitative sketch, not a plotted dataset: no coordinate of ε is drawn. H projects the outcomes onto the space
          the fit can reach, so the residual is what is left orthogonal to it. That projection has rank {hat.rank}, which
          is why the expected training error loses exactly p/n of the noise.
        </p>
      </div>
      <div className="bv-panel">
        <h4>The same numbers from the general smoother formula</h4>
        <Table caption="For a fixed linear smoother ŷ = Sy the difference between fresh and training error is 2σ²tr(S)/n. Least squares is the case S = H, where tr(H) equals p exactly, and the approximation term is zero to every displayed decimal because this mean lies in the column space the fit can reach."
          headings={['quantity', 'value']}
          rows={[
            ['trace of S = H', round(correctAccount.trace, 9)],
            ['trace of SᵀS', round(correctAccount.traceSquared, 9)],
            ['approximation ‖(I − S)f‖²/n', fixed(correctAccount.approximation, 9)],
            ['training error', round(correctAccount.training, 9)],
            ['fresh-outcome error', round(correctAccount.fresh, 9)],
            ['their difference', round(correctAccount.difference, 9)],
            ['2σ²tr(S)/n', round(correctAccount.expectedDifference, 9)],
          ]} />
        <p>
          With a correctly specified mean these reproduce {optimism.trainingMse.toFixed(6)} and
          {' '}{optimism.newOutcomeMse.toFixed(6)} exactly. If the truth instead carries a curvature term this design
          cannot represent, the approximation term becomes {curvedAccount.approximation.toFixed(6)} and <em>both</em>
          {' '}errors gain it — {curvedAccount.training.toFixed(6)} and {curvedAccount.fresh.toFixed(6)} — while the gap
          stays at {curvedAccount.difference.toFixed(6)}. High training error can contain approximation failure.
        </p>
      </div>
    </div>
    <Table caption={`A new input is not the same-X average. The training inputs are ${fixtures.optimismInputs.join(', ')} with an intercept, so the average fitted-prediction variance over them is exactly pσ²/n = ${optimism.predictionVariance.toFixed(6)}.`}
      headings={['new input x⋆', 'inside the fitted design?', 'leverage x⋆ᵀ(XᵀX)⁻¹x⋆', 'prediction variance', 'expected squared error']}
      rows={locations.map(entry => [
        round(entry.value, 2), entry.inDesign ? 'yes' : 'no',
        round(entry.leverage, 6), round(entry.predictionVariance, 6), round(entry.expectedSquaredError, 6),
      ])} />
    <p>
      The same fitted model answers a new question differently depending on where it is asked. At the centre of this
      design the prediction variance is {locations[0].predictionVariance.toFixed(6)}; at x⋆ = {locations[3].value}, well
      outside it, the variance is {locations[3].predictionVariance.toFixed(6)}. Averaging prediction variance (σ² times leverage) over the six training
      rows gives {optimism.predictionVariance.toFixed(6)}, so the tidy 1 + p/n statement is an average over
      {' '}<em>those</em> rows, not a universal out-of-sample formula.
    </p>
  </figure>;
}

/* ------------------------------------------------------- §8 · F6 */

/** Loss-specific behaviour: an exact identity beside a direct decision count. */
export function LossFigure() {
  const brierA = brierDecomposition(fixtures.brierA.eta, fixtures.brierA.probabilities);
  const brierB = brierDecomposition(fixtures.brierB.eta, fixtures.brierB.probabilities);
  const zeroOneA = thresholdedError(fixtures.brierA.eta, fixtures.brierA.probabilities).error;
  const zeroOneB = thresholdedError(fixtures.brierB.eta, fixtures.brierB.probabilities).error;
  const averaging = averageVariance(fixtures.averaging.v, fixtures.averaging.rho, fixtures.averaging.B);
  const eta = 0.8;
  const rates = [0, 0.25, 0.75, 1];
  return <figure className="bv-figure">
    <figcaption>
      <strong>Figure 6 — Two losses, two different accounts of the same pair of procedures.</strong> At one input with
      P(Y = 1) = {fixtures.brierA.eta}, procedure A always outputs {fixtures.brierA.probabilities[0]} while B outputs
      {' '}{fixtures.brierB.probabilities.join(' or ')} with equal chance across training datasets. Both average to
      {' '}{round(brierB.averageProbability, 6)}. The squared probability loss decomposes exactly; the thresholded decision does
      not use that decomposition at all.
    </figcaption>
    <Table caption="The same two procedures under two losses, at threshold 0.5. Mean p̂ is the average predicted probability across training datasets; every entry is exact arithmetic on the stated distribution."
      headings={['procedure', 'mean p̂', 'squared bias', 'variance', 'noise η(1 − η)', 'Brier loss', 'zero-one error']}
      rows={[
        [`A: always ${fixtures.brierA.probabilities[0]}`, round(brierA.averageProbability, 6), round(brierA.squaredBias, 6),
          round(brierA.variance, 6), round(brierA.noise, 6), round(brierA.total, 6), round(zeroOneA, 6)],
        [`B: ${fixtures.brierB.probabilities.join(' or ')}`, round(brierB.averageProbability, 6), round(brierB.squaredBias, 6),
          round(brierB.variance, 6), round(brierB.noise, 6), round(brierB.total, 6), round(zeroOneB, 6)],
      ]} />
    <div className="bv-panels is-pair">
      <div className="bv-panel">
        <h4>Thresholded error is a straight line in the class-1 rate</h4>
        <Plot width={300} height={196} domain={[0, 1]} range={[0, 1]}
          padding={{ left: 42, right: 14, top: 18, bottom: 50 }}
          ticks={[0, 0.25, 0.5, 0.75, 1]} valueTicks={[0, 0.5, 1]}
          describe={`Expected zero-one error against the probability q that a procedure predicts class 1, at eta 0.8. It falls in a straight line from 0.8 at q equals 0 to 0.2 at q equals 1, passing through ${rates.map(rate => `${thresholdedErrorAt(eta, rate).toFixed(2)} at q equals ${rate}`).join(', ')}. More variation in the predicted class helps when it replaces the wrong constant decision and hurts when it replaces the right one.`}>
          {(scaleX, scaleY) => <>
            <polyline className="bv-curve is-validation"
              points={polyline([[0, thresholdedErrorAt(eta, 0)], [1, thresholdedErrorAt(eta, 1)]], scaleX, scaleY)} />
            {rates.map(rate => <g key={rate}>
              <circle className="bv-mark" cx={scaleX(rate)} cy={scaleY(thresholdedErrorAt(eta, rate))} r="3.4" />
              {/* The end labels are anchored inward so neither runs past the frame,
                  and the last one sits below its point: anchored inward above it,
                  it lands on the descending line arriving at that point. */}
              <text className="bv-halo" x={scaleX(rate)}
                y={scaleY(thresholdedErrorAt(eta, rate)) + (rate === 1 ? 17 : -9)}
                textAnchor={rate === 1 ? 'end' : rate === 0 ? 'start' : 'middle'}>
                {thresholdedErrorAt(eta, rate).toFixed(2)}
              </text>
            </g>)}
            <text x={scaleX(0.5)} y="188" textAnchor="middle">q, the class-1 rate</text>
          </>}
        </Plot>
        <p>
          At η = {eta} the expected zero-one error is {eta} − {(2 * eta - 1).toFixed(1)}q. Moving q from 0 to 0.25 lowers
          it from {thresholdedErrorAt(eta, 0).toFixed(2)} to {thresholdedErrorAt(eta, 0.25).toFixed(2)}, while moving from
          1 to 0.75 raises it from {thresholdedErrorAt(eta, 1).toFixed(2)} to {thresholdedErrorAt(eta, 0.75).toFixed(2)}.
          The same added variation helps or hurts depending on which decisions it replaces, which is why the squared-loss
          identity does not transfer to accuracy.
        </p>
      </div>
      <div className="bv-panel">
        <h4>Averaging cannot remove shared variation</h4>
        <Table caption={`Equal variance v = ${averaging.v} and pairwise correlation ρ = ${averaging.rho}: the equal-weight average of B of them has variance v(ρ + (1 − ρ)/B).`}
          headings={['quantity', 'value']}
          rows={[
            ['B individual variances', round(averaging.individualTerms, 6)],
            ['B(B − 1) pair covariances', round(averaging.pairTerms, 6)],
            [`variance of the average, B = ${averaging.B}`, round(averaging.variance, 6)],
            ['if they were independent, v/B', round(averaging.independentVariance, 6)],
            ['if they were perfectly correlated', round(averaging.perfectlyCorrelated, 6)],
          ]} />
        <p>
          Expand the variance of the sum — {averaging.B} individual variances and {averaging.B * (averaging.B - 1)} pair
          covariances — then divide by B². The answer is {averaging.variance}, not {averaging.independentVariance}. And
          if the constituent mean predictions stay the same, averaging leaves their common bias exactly where it was.
        </p>
      </div>
    </div>
  </figure>;
}

/* ------------------------------------------------------- §9 · F7 */

// The branches are drawn as far as the last two ratios the table gives, 0.99 and
// 1.01, where the approximation is 4.01 and 4.04. Ending them there puts the
// drawn peak at the tabulated height instead of a fifth of it, and leaves the
// only missing point — γ = 1 exactly — as the single point it actually is.
const ddBreak = [0.99, 1.01];
const ddBelow = doubleDescentBranch(0.1, ddBreak[0], 160);
const ddAbove = doubleDescentBranch(ddBreak[1], 3, 160);
const ddRange = [0.03, 6];

/** Theoretical double descent. */
export function DoubleDescentFigure() {
  const table = fixtures.doubleDescentRatios.map(ratio => ({ ratio, ...doubleDescentRisk(ratio) }));
  const insetPoints = doubleDescentBranch(0.1, 0.95, 120);
  return <figure className="bv-figure">
    <figcaption>
      <strong>Figure 7 — A theoretical risk curve that is not U-shaped.</strong> These are asymptotic calculations from
      the stated formula for the minimum-norm ridgeless fit, with isotropic Gaussian inputs, signal norm
      {' '}{doubleDescentSignalNorm} and target-noise variance {doubleDescentNoise}. They are not a measured dataset, a timing experiment or a finite simulation, and no neural network appears in them.
    </figcaption>
    <div className="bv-panel">
      <h4>Expected fresh-target MSE against γ = n/p, logarithmic vertical axis</h4>
      <Plot width={340} height={224} domain={[0.1, 3]} range={ddRange} valueScale="log"
        padding={{ left: 46, right: 16, top: 20, bottom: 38 }}
        ticks={[0.1, 0.5, 1, 1.5, 2, 2.5, 3]} valueTicks={[0.04, 0.1, 0.4, 1, 4]}
        formatTick={value => String(Number(value.toFixed(2)))} formatValue={value => String(value)}
        describe={`Expected fresh-target mean squared error against the ratio gamma of training rows to coordinates, from 0.1 to 3, on a logarithmic vertical axis from 0.03 to 6. The curve is drawn as two separate branches. Below gamma equals one it falls from ${doubleDescentRisk(0.1).riskApprox.toFixed(6)} to a local low of ${doubleDescentRisk(0.8).riskApprox.toFixed(2)} near gamma 0.8 and then climbs steeply towards the boundary. Above gamma equals one it falls from a very large value back to ${doubleDescentRisk(3).riskApprox.toFixed(2)}. The two branches are never joined, because gamma equals one is a singular boundary where this approximation is undefined. A dashed horizontal line marks the target-noise floor ${doubleDescentNoise}, and an upward arrow on the singular boundary indicates divergence to show that the plotted ends are not a finite maximum.`}>
        {(scaleX, scaleY) => <>
          <line className="bv-curve is-baseline" x1={scaleX(0.1)} x2={scaleX(3)} y1={scaleY(doubleDescentNoise)} y2={scaleY(doubleDescentNoise)} />
          {/* The asymptote is named at the left end, where the risk curve is an
              order of magnitude above it. At the right end the curve descends
              onto this line and would run straight through the words. */}
          <text className="bv-halo" x={scaleX(0.115)} y={scaleY(doubleDescentNoise) - 6}>noise {doubleDescentNoise}</text>
          <polyline className="bv-curve is-validation"
            points={polyline(ddBelow.map(point => [point.ratio, Math.min(point.riskApprox, ddRange[1])]), scaleX, scaleY)} />
          <polyline className="bv-curve is-validation"
            points={polyline(ddAbove.map(point => [point.ratio, Math.min(point.riskApprox, ddRange[1])]), scaleX, scaleY)} />
          {/* The single undefined point, drawn as one dashed line that arrows off
              the top: the approximation grows without bound from either side, and
              the two branch ends sit just below the arrow's tail so it reads as
              their continuation rather than as a floating annotation. */}
          <line className="bv-stem is-probe" x1={scaleX(1)} x2={scaleX(1)} y1={scaleY(4.9)} y2={scaleY(ddRange[0])} />
          <polygon className="bv-flow" style={{ fill: '#8eb9a5' }}
            points={`${scaleX(1)},${scaleY(5.9)} ${scaleX(1) - 5},${scaleY(4.8)} ${scaleX(1) + 5},${scaleY(4.8)}`} />
          <text x={scaleX(1)} y="14" textAnchor="middle">γ = 1, undefined</text>
          {fixtures.doubleDescentRatios.filter(ratio => ratio !== 0.99 && ratio !== 1.01).map(ratio => (
            <circle key={ratio} className="bv-mark" cx={scaleX(ratio)} cy={scaleY(doubleDescentRisk(ratio).riskApprox)} r="3" />
          ))}
        </>}
      </Plot>
      <p>
        The vertical axis is logarithmic, with labelled values {[0.04, 0.1, 0.4, 1, 4].join(', ')}; that transform is what
        keeps the peak and both low branches readable in one picture. Each branch is drawn out to the last ratio the
        table gives on its side — γ = {ddBreak[0]} and {ddBreak[1]}, where the expression is
        {' '}{doubleDescentRisk(ddBreak[0]).riskApprox.toFixed(2)} and
        {' '}{doubleDescentRisk(ddBreak[1]).riskApprox.toFixed(2)} — and stops there, still climbing. The two are never
        joined, because <strong>exactly one point is missing: γ = 1</strong>, where the dashed line stands and the arrow
        leaves the top of the frame.
      </p>
      <p>
        That is the whole of the undefined set. The expression has a perfectly ordinary value on either side of it,
        arbitrarily close to it: {doubleDescentRisk(0.999).riskApprox.toFixed(2)} at γ = 0.999 and
        {' '}{doubleDescentRisk(1.001).riskApprox.toFixed(2)} at γ = 1.001, growing without bound as γ approaches 1 from
        either direction. The gap between the two branch ends is that single missing point drawn at the width this axis
        gives it, not an interval on which the formula fails.
      </p>
    </div>
    <div className="bv-panel">
      <h4>The initial fall, and the rise before the boundary, on a linear axis</h4>
      <Plot width={300} height={176} domain={[0.1, 0.95]} range={[0, 1]}
        padding={{ left: 42, right: 14, top: 18, bottom: 34 }}
        ticks={[0.1, 0.3, 0.5, 0.7, 0.9]} valueTicks={[0, 0.5, 1]}
        formatTick={value => String(Number(value.toFixed(2)))} formatValue={value => String(value)}
        describe={`The same lower branch on a linear vertical axis from 0 to 1, over gamma from 0.1 to 0.95. It falls from ${doubleDescentRisk(0.1).riskApprox.toFixed(4)} to its lowest plotted value ${doubleDescentRisk(0.8).riskApprox.toFixed(4)} near gamma 0.8, then rises again to ${doubleDescentRisk(0.95).riskApprox.toFixed(4)} at gamma 0.95.`}>
        {(scaleX, scaleY) => <>
          <line className="bv-curve is-baseline" x1={scaleX(0.1)} x2={scaleX(0.95)} y1={scaleY(doubleDescentNoise)} y2={scaleY(doubleDescentNoise)} />
          <polyline className="bv-curve is-validation"
            points={polyline(insetPoints.map(point => [point.ratio, Math.min(point.riskApprox, 1)]), scaleX, scaleY)} />
          {[0.1, 0.5, 0.8, 0.9].map(ratio => <g key={ratio}>
            <circle className="bv-mark" cx={scaleX(ratio)} cy={scaleY(doubleDescentRisk(ratio).riskApprox)} r="3" />
            {/* Three of these four values sit on a vertical gridline, so each
                carries a backplate; the leftmost is also anchored away from the
                vertical axis, whose own tick labels sit immediately beside it. */}
            <text className="bv-halo"
              x={scaleX(ratio) + (ratio === 0.1 ? 6 : ratio === 0.9 ? -8 : 0)}
              y={scaleY(doubleDescentRisk(ratio).riskApprox) + (ratio === 0.9 ? -18 : ratio === 0.8 ? -13 : -9)}
              textAnchor={ratio === 0.1 ? 'start' : ratio === 0.9 ? 'end' : 'middle'}>
              {doubleDescentRisk(ratio).riskApprox.toFixed(2)}
            </text>
          </g>)}
        </>}
      </Plot>
      <p>
        Adding rows helps down to about γ = 0.8, then hurts as the boundary approaches. Below the boundary the variance
        term carries both the randomness of the observed input subspace and amplified target noise; close to γ = 1, small
        singular values make the fit especially sensitive to that noise.
      </p>
    </div>
    <Table caption="The nine stated ratios, from the piecewise expression plus the target noise"
      headings={['γ = n/p', 'squared bias', 'variance term', 'excess risk', 'fresh-target MSE']}
      rows={table.map(entry => [
        fixed(entry.ratio, 2), round(entry.biasSquared, 6), round(entry.varianceApprox, 6),
        round(entry.excess, 6), fixed(entry.riskApprox, 6),
      ])} />
    <p>
      Below the boundary the excess risk is (1 − γ) + {doubleDescentNoise}γ/(1 − γ); above it,
      {' '}{doubleDescentNoise}/(γ − 1). Add {doubleDescentNoise} for fresh-target noise in both cases. The identity of
      section 2 remains true throughout: it constrains how the three terms add, not how they move as the ratio changes.
    </p>
  </figure>;
}
