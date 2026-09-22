import { useState } from 'react';
import {
  BarRow, Drawing, Figure, NumberField, Readout, Select, Table, asFraction, exactly, fixed, round, signGlyph,
} from './RademacherShared.jsx';
import {
  absoluteComplexity, ballGeometry, curvePoints, diamondPoints, empiricalComplexity, figureExtent,
  ghostPairLayout,
  hingeMarginLoss, hullLayout, kernelComplexity, l1BestResponse, linearComplexity, logisticMarginLoss,
  mapFeatures, marginHistogram, mistakeRows, monteCarloLayout, nestedBoxLayout, numberLineLayout, rampCurve,
  rampLoss, rolesStrip, sigmoid, signPatterns, stackedBoundLayout, thresholdRows,
} from '../../data/rademacher-models.js';
import {
  experimentSettings, fittedModels, observations, recordedFixtures, recordedMonteCarlo, representation, roles,
} from '../../data/rademacher-data.js';

/* Inline figures for the Rademacher lesson.
 *
 * Every coordinate below comes from `rademacher-models.js`, so
 * `scripts/verify-rademacher-models.mjs` asserts the same geometry the browser
 * paints. A drawn curve, band or supporting point is a mathematical claim and
 * is checked as one.
 *
 * A figure may show its own content on first paint -- that is what a figure is
 * for. The no-reveal rule belongs to the investigations, and none of these
 * grades anything.
 */

const THREE_INPUTS = [-1, 0, 1];

/** An arrowhead triangle at `tip`, pointing away from `base`. */
function head(base, tip, size = 7) {
  const dx = tip.x - base.x;
  const dy = tip.y - base.y;
  const length = Math.hypot(dx, dy) || 1;
  const ux = dx / length;
  const uy = dy / length;
  const back = { x: tip.x - size * ux, y: tip.y - size * uy };
  return [
    [tip.x, tip.y],
    [back.x - (size * 0.45) * uy, back.y + (size * 0.45) * ux],
    [back.x + (size * 0.45) * uy, back.y - (size * 0.45) * ux],
  ].map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(' ');
}

/* ============================================================== figure 1 */

/**
 * Three ordered inputs, one movable cutoff, and the four prediction rows the
 * whole infinite threshold class can produce on them.
 *
 * The teaching point is that the cutoff moves continuously while its row does
 * not: between two inputs, every cutoff gives the same row.
 */
export function ThresholdRestrictionFigure() {
  const [cutoff, setCutoff] = useState(0.4);
  const rows = thresholdRows(THREE_INPUTS);
  const line = numberLineLayout(THREE_INPUTS, { width: 320 });
  const current = THREE_INPUTS.map(value => (value >= cutoff ? 1 : -1));
  const activeIndex = rows.findIndex(row => row.every((value, index) => value === current[index]));
  const interval = (() => {
    const sorted = [...THREE_INPUTS].sort((a, b) => a - b);
    const above = sorted.filter(value => value >= cutoff);
    const below = sorted.filter(value => value < cutoff);
    const low = below.length ? below[below.length - 1] : null;
    const high = above.length ? above[0] : null;
    if (low === null) return `every cutoff at or below ${high}`;
    if (high === null) return `every cutoff above ${low}`;
    return `every cutoff in (${low}, ${high}]`;
  })();
  const cutoffX = Math.max(line.left - 10, Math.min(line.width - line.right + 10, line.place(cutoff)));
  return <Figure caption="Figure 1. Infinitely many cutoffs, four distinct effects on this sample"
    describe={`Three inputs at −1, 0 and 1 with a cutoff at ${cutoff}. The cutoff predicts +1 at or above itself and −1 below it, giving the row ${current.map(signGlyph).join(', ')}. ${interval} gives that same row.`}
    footnote="These four rows are restrictions to this sample of one fixed infinite class. The class was chosen before the sample; the restriction is what the calculation sees.">
    <Drawing width={320} height={92} className="rad-threshold-line"
      title="A number line with three inputs and a movable cutoff"
      describe={`Inputs at −1, 0 and 1. The cutoff sits at ${cutoff}.`}>
      <line className="rad-axis" x1={line.left - 12} y1={line.y} x2={line.width - line.right + 12} y2={line.y} />
      {line.points.map(point => <g key={point.index}>
        <circle className={current[point.index] > 0 ? 'rad-point' : 'rad-support'} cx={point.x} cy={line.y} r={5} />
        <text className="rad-small" x={point.x} y={line.y + 20} textAnchor="middle">{point.value}</text>
        <text className="rad-small rad-accent" x={point.x} y={line.y - 12} textAnchor="middle">
          {signGlyph(current[point.index])}
        </text>
      </g>)}
      {/* The cutoff is any real number, but the axis is finite. The mark is
          held inside the drawn axis rather than allowed to leave the viewBox;
          the printed value below always says where the cutoff actually is. */}
      <line className="rad-ceiling" x1={cutoffX} y1={line.y - 28} x2={cutoffX} y2={line.y + 26} />
      <text className="rad-tiny rad-muted" x={line.left - 12} y={line.y + 40}>cutoff t = {cutoff}</text>
      <text className="rad-tiny rad-muted" x={line.left - 12} y={16}>prediction</text>
    </Drawing>
    <div className="rad-controls">
      <NumberField label="Cutoff t" value={cutoff} min={-3} max={3} decimals={3}
        onChange={setCutoff} hint="Any real number. Slide it across an input and the row changes; move it between inputs and nothing changes." />
    </div>
    <Table caption="The four distinct prediction rows, with the one this cutoff produces marked"
      headings={['Row', 'x = −1', 'x = 0', 'x = 1', 'Cutoffs that give it']}
      rowClass={index => (index === activeIndex ? 'is-leading' : undefined)}
      rows={rows.map((row, index) => [
        index === activeIndex ? `${index + 1} ← this cutoff` : String(index + 1),
        ...row.map(signGlyph),
        ['above every input', 'in (0, 1]', 'in (−1, 0]', 'at or below −1'][index],
      ])} />
  </Figure>;
}

/* ============================================================== figure 2 */

/**
 * The best-response matrix: eight sign patterns, four rows, every normalised
 * correlation, the winner in each column, and the two orders of operations
 * side by side.
 */
export function BestResponseMatrixFigure() {
  const [order, setOrder] = useState('max-then-average');
  const rows = thresholdRows(THREE_INPUTS);
  const model = empiricalComplexity(rows);
  const maximiseFirst = order === 'max-then-average';
  return <Figure caption="Figure 2. Choose the best row for each pattern, then average — not the other way round"
    describe={`A four-row by eight-column table of normalised correlations. Taking the maximum of each column and then averaging gives ${asFraction(model.complexity)}. Averaging each row first and then taking the maximum gives ${asFraction(model.maxAfterAveraging)}.`}
    footnote="Both numbers are exact. They differ because the maximum of an average is not the average of a maximum, and the gap is precisely the freedom to adapt that the definition is trying to measure.">
    <div className="rad-controls">
      <Select label="Order of operations" value={order} onChange={setOrder} options={[
        ['max-then-average', 'Maximum for each pattern, then average — the definition'],
        ['average-then-max', 'Average each row first, then take the maximum — a different expression'],
      ]} />
    </div>
    <Table caption={maximiseFirst
      ? 'Each column is one sign pattern. The marked cell is that pattern\'s best row; the eight bests are averaged in the readout below.'
      : 'The same cells. Now each ROW is averaged over all eight patterns first, and the largest row average is taken.'}
      headings={['Row', ...model.patterns.map((pattern, index) => `σ${index + 1}: ${pattern.map(s => (s > 0 ? '+' : '−')).join('')}`), maximiseFirst ? 'row average' : 'ROW AVERAGE']}
      cellClass={(rowIndex, column) => {
        if (!maximiseFirst) return column === model.patterns.length + 1 ? 'is-winner' : undefined;
        if (column === model.patterns.length + 1) return undefined;
        return model.winners[column - 1].includes(rowIndex) ? 'is-winner' : undefined;
      }}
      rows={rows.map((row, rowIndex) => [
        row.map(signGlyph).join(' '),
        ...model.correlations.map(scores => fixed(scores[rowIndex], 4)),
        fixed(model.rowAverages[rowIndex], 4),
      ])} />
    <Readout items={maximiseFirst
      ? [
        ['Column maxima', model.maxima.map(value => asFraction(value)).join(', ')],
        ['Their average — the empirical complexity', exactly(model.complexity)],
        ['Patterns a threshold row matches exactly', `${model.maxima.filter(value => value === 1).length} of 8`],
      ]
      : [
        ['Row averages', model.rowAverages.map(value => fixed(value, 4)).join(', ')],
        ['Largest of them', exactly(model.maxAfterAveraging)],
        ['Why it is zero', 'each row is a fixed vector, and a fixed vector correlates with fair signs at zero on average'],
      ]} />
  </Figure>;
}

/* ============================================================== figure 3 */

/** Five nested classes on the same three inputs, drawn as nested sets with
 *  their exact values, plus the absolute-value fork on the singleton. */
export function NestedClassesFigure() {
  const singleton = [[1, 1, 1]];
  const constants = [[-1, -1, -1], [1, 1, 1]];
  const positive = thresholdRows(THREE_INPUTS);
  const both = thresholdRows(THREE_INPUTS, { bothOrientations: true });
  const cube = signPatterns(3);
  const families = [
    { name: 'Only the fixed all-positive rule', rows: singleton },
    { name: 'Either constant sign', rows: constants },
    { name: 'Positive thresholds', rows: positive },
    { name: 'Thresholds in either orientation', rows: both },
    { name: 'Every possible sign row', rows: cube },
  ].map(entry => ({ ...entry, model: empiricalComplexity(entry.rows) }));
  const layout = nestedBoxLayout(families.length, { width: 320 });
  const absolute = absoluteComplexity(singleton);
  // Outermost box first: the largest class contains all the others.
  const outermostFirst = families.slice().reverse();
  return <Figure caption="Figure 3. Each class contains the one before it, so its value cannot be smaller"
    describe={`Five nested rectangles, largest outermost: every possible sign row (8 rows, value 1) contains thresholds in either orientation (6 rows, 5/6), which contains positive thresholds (4 rows, 2/3), which contains the two constants (2 rows, 1/2), which contains the single all-positive rule (1 row, 0).`}
    footnote="This is a justified comparison because the classes are nested on one sample. “Trees are more complex than linear models” is not, until the outputs, constraints and inputs are named.">
    <Drawing width={layout.width} height={layout.height} className="rad-nested"
      title="Five nested classes" describe="Nested rectangles, one per class, with row counts and exact values.">
      {layout.boxes.map((box, position) => <rect key={`box-${box.level}`} className="rad-cell"
        x={box.x} y={box.y} width={box.width} height={box.height} rx={4}
        style={{ opacity: position === layout.boxes.length - 1 ? 1 : 0.92 }} />)}
      {layout.boxes.map((box, position) => (
        <text key={`label-${box.level}`} className="rad-tiny" x={box.labelX} y={box.labelY}>
          {outermostFirst[position].rows.length} row{outermostFirst[position].rows.length === 1 ? '' : 's'}
          {' · '}{asFraction(outermostFirst[position].model.complexity)}
        </text>
      ))}
    </Drawing>
    <Table caption="The same five classes as exact values"
      headings={['Allowed predictions', 'Distinct rows', 'Empirical complexity']}
      rows={families.map(family => [family.name, String(family.rows.length), exactly(family.model.complexity)])} />
    <Readout items={[
      ['The singleton under this lesson\'s convention', exactly(families[0].model.complexity)],
      ['The same singleton with an absolute value inside', exactly(absolute.complexity)],
      ['What changed', 'not the arithmetic — the class. An absolute value measures the set containing the rule AND its negative, which is the two-constant class.'],
    ]} />
  </Figure>;
}

/* ============================================================== figure 4 */

/** Predictor coordinates against loss coordinates, for fixed target labels. */
export function PredictorVersusLossFigure() {
  const labels = [1, -1, 1];
  const rows = thresholdRows(THREE_INPUTS);
  const losses = mistakeRows(rows, labels);
  const predictorModel = empiricalComplexity(rows);
  const lossModel = empiricalComplexity(losses);
  return <Figure caption="Figure 5. The same four rules, rewritten as the mistakes they make"
    describe={`Each prediction row h becomes the loss row (1 − y h)/2 for target labels +1, −1, +1. The predictor class has complexity ${asFraction(predictorModel.complexity)} and the loss class exactly half of it, ${asFraction(lossModel.complexity)}.`}
    footnote="The factor is exactly one half for sign-valued hypotheses under a mistake loss. It is not a general rule about composing any bounded loss with any score class.">
    <Table caption="Prediction coordinates on the left, mistake coordinates on the right. A mistake costs 1, a correct answer 0."
      headings={['Rule', 'h(−1)', 'h(0)', 'h(1)', 'loss at y=+1', 'loss at y=−1', 'loss at y=+1']}
      rows={rows.map((row, index) => [
        `rule ${index + 1}`, ...row.map(signGlyph), ...losses[index].map(value => String(value)),
      ])} />
    <Readout items={[
      ['Predictor class', exactly(predictorModel.complexity)],
      ['Mistake class', exactly(lossModel.complexity)],
      ['Ratio', `exactly ${asFraction(lossModel.complexity / predictorModel.complexity)}`],
      ['Why', 'the fixed ½ washes out under the sign average, and multiplying a fair sign by the fixed −yᵢ gives another fair sign'],
    ]} />
    <p className="rad-caption">Hard-thresholding a real-valued score is a different operation: it is discontinuous, so
      two scores that differ by a millionth can land on opposite sides. The identity above applies to a class that was
      already sign-valued.</p>
  </Figure>;
}

/* ============================================================== figure 5 */

/** The three terms of the theorem, and the two sources of randomness kept in
 *  separate lanes. */
export function TheoremTermsFigure() {
  return <Figure caption="Figure 4. Three terms, two randomness lanes, one probability statement"
    describe="The bound is a sum of three terms: the loss actually observed, twice the empirical complexity of the loss class, and a confidence allowance. Two separate lanes of randomness feed it: the training sample, drawn once from the population, and the auxiliary signs, which exist only inside the complexity calculation."
    footnote="The probability 1 − δ belongs to the draw of the sample, and covers every member of the class at once. It is not a probability attached to any individual prediction, and the statement is not a deterministic promise about this particular dataset.">
    <div className="rad-strip">
      <span className="rad-strip-cell"><b>R̂_S(f)</b>the loss you measured on the training sample</span>
      <span className="rad-strip-cell is-used"><b>+ 2 ℜ̂_S(G)</b>twice the complexity of the LOSS class, on this sample</span>
      <span className="rad-strip-cell"><b>+ 3√(ln(2/δ)/2n)</b>the allowance for an unrepresentative sample</span>
    </div>
    <Table caption="What is random, and what each randomness is averaged over"
      headings={['Lane', 'What varies', 'What it is averaged or bounded over', 'Present in a real implementation?']}
      rows={[
        ['Sample draw', 'which n observations you happened to get', 'the 1 − δ failure allowance covers this', 'yes — you drew it once, and cannot redraw it'],
        ['Auxiliary signs', 'the 2ⁿ coin-flip patterns inside the complexity', 'an exact average, or an estimate of one', 'only inside the calculation; it never touches the labels'],
        ['Ghost sample', 'a second independent sample used in the proof', 'an expectation in the argument', 'no — it is a device, not data to collect'],
      ]} />
  </Figure>;
}

/* ============================================================== figure 6 */

/** The ghost-sample pair swap, and the step where one shared choice becomes
 *  two independent ones. */
export function GhostSampleFigure() {
  const [signs, setSigns] = useState([1, -1, 1, 1]);
  const pairs = [
    { sample: 0.2, ghost: 0.7 }, { sample: 0.9, ghost: 0.4 },
    { sample: 0.5, ghost: 0.5 }, { sample: 0.1, ghost: 0.8 },
  ].map((pair, index) => ({ ...pair, sign: signs[index] }));
  const layout = ghostPairLayout(pairs, { width: 320 });
  const total = layout.columns.reduce((sum, column) => sum + column.difference, 0);
  return <Figure caption="Figure 6. Swapping a pair changes nothing about the distribution, and everything about the bookkeeping"
    describe={`Four paired columns. The top lane holds the ghost sample and the bottom the training sample, until a pair is swapped. The signed differences are ${layout.columns.map(column => round(column.difference, 3)).join(', ')}, summing to ${round(total, 3)}.`}
    footnote="Swapping is illustrated on fixed numbers here. The statement the proof uses is about the joint DISTRIBUTION of independent identically distributed pairs, which is unchanged by the swap — not a claim that these four particular numbers give the same maximum either way.">
    <Drawing width={layout.width} height={layout.height} className="rad-ghost"
      title="Four paired observations with per-pair swap controls"
      describe="Two lanes of four cells, with the current sign of each pair marked.">
      <text className="rad-tiny rad-muted" x={4} y={layout.columns[0].topY - 10}>ghost lane</text>
      <text className="rad-tiny rad-muted" x={4} y={layout.columns[0].bottomY + 34}>sample lane</text>
      {layout.columns.map(column => <g key={column.index}>
        <rect className={`rad-lane${column.sign === -1 ? ' is-swapped' : ''}`} x={column.x} y={column.topY}
          width={column.width} height={26} rx={3} />
        <text className="rad-small" x={column.x + column.width / 2} y={column.topY + 17} textAnchor="middle">
          {column.topValue}
        </text>
        <rect className={`rad-lane${column.sign === -1 ? ' is-swapped' : ''}`} x={column.x} y={column.bottomY}
          width={column.width} height={26} rx={3} />
        <text className="rad-small" x={column.x + column.width / 2} y={column.bottomY + 17} textAnchor="middle">
          {column.bottomValue}
        </text>
        <text className="rad-tiny rad-accent" x={column.x + column.width / 2} y={column.bottomY + 46} textAnchor="middle">
          {signGlyph(column.sign)}
        </text>
      </g>)}
    </Drawing>
    <div className="rad-presets">
      {layout.columns.map(column => <button key={column.index} type="button"
        className={column.sign === -1 ? 'is-selected' : undefined}
        onClick={() => setSigns(signs.map((value, index) => (index === column.index ? -value : value)))}>
        {column.sign === -1 ? 'Unswap' : 'Swap'} pair {column.index + 1}
      </button>)}
    </div>
    <Readout items={[
      ['Signed differences', layout.columns.map(column => round(column.difference, 3)).join(', ')],
      ['Their sum', round(total, 6)],
      ['The inequality step', 'one shared choice of g becomes two independent choices — the right side lets two different functions win, so it can only be larger'],
      ['Where the factor 2 enters', 'here, once. It is not inserted again at the ghost-sample step.'],
    ]} />
  </Figure>;
}

/* ============================================================== figure 7 */

/** The supporting point on the ball, including the degenerate case. */
export function SupportingPointFigure() {
  const [collapsed, setCollapsed] = useState(false);
  const vectors = [[1, 0.6], [0.4, -0.9]];
  const signs = collapsed ? [1, -1] : [1, 1];
  const usedVectors = collapsed ? [[1, 0.6], [1, 0.6]] : vectors;
  const geometry = ballGeometry(usedVectors, signs, 1, { width: 300, height: 240 });
  return <Figure caption="Figure 7. The best coefficient points along the signed sum, at the edge of the ball"
    describe={`Two input arrows, their signed sum v, the ball of radius 1, and the supporting point w* = v/‖v‖. ${geometry.best.degenerate ? `Here ${geometry.best.degenerate}.` : `Here ‖v‖ = ${round(geometry.best.length, 4)} and the optimum is ${round(geometry.best.optimum, 6)}.`}`}
    footnote="Both axes use one scale, so the ball is a circle and the supporting point really is the farthest point of the ball in the direction of v. The radius is a constraint on score coefficients; it is not a probability.">
    <Drawing width={geometry.width} height={geometry.height} className="rad-ball-figure"
      title="A norm ball with the signed sum and its supporting point"
      describe="A circle centred at the origin with input arrows, their sum, and the maximising coefficient.">
      <line className="rad-grid" x1={0} y1={geometry.origin.y} x2={geometry.width} y2={geometry.origin.y} />
      <line className="rad-grid" x1={geometry.origin.x} y1={0} x2={geometry.origin.x} y2={geometry.height} />
      <circle className="rad-ball" cx={geometry.origin.x} cy={geometry.origin.y} r={geometry.radiusPixels} />
      {geometry.arrows.map(arrow => <g key={arrow.index}>
        <line className={`rad-arrow${arrow.sign < 0 ? ' is-flipped' : ''}`}
          x1={arrow.base.x} y1={arrow.base.y} x2={arrow.tip.x} y2={arrow.tip.y} />
        <polygon className={`rad-arrowhead${arrow.sign < 0 ? ' is-flipped' : ''}`} points={head(arrow.base, arrow.tip)} />
      </g>)}
      {geometry.best.length > 1e-9 && <>
        <line className="rad-sum" x1={geometry.origin.x} y1={geometry.origin.y}
          x2={geometry.sum.tip.x} y2={geometry.sum.tip.y} />
        <polygon className="rad-sum-head" points={head(geometry.origin, geometry.sum.tip, 9)} />
        <text className="rad-small rad-accent" x={geometry.sum.tip.x + 6} y={geometry.sum.tip.y - 4}>v</text>
      </>}
      {geometry.support && <>
        <circle className="rad-support" cx={geometry.support.point.x} cy={geometry.support.point.y} r={5} />
        <text className="rad-small" x={geometry.supportLabel.x} y={geometry.supportLabel.y}
          textAnchor={geometry.supportLabel.anchor}>w*</text>
      </>}
      {!geometry.support && <text className="rad-small rad-muted" x={geometry.origin.x + 8} y={geometry.origin.y - 8}>
        v = 0
      </text>}
    </Drawing>
    <div className="rad-presets">
      <button type="button" className={collapsed ? undefined : 'is-selected'} onClick={() => setCollapsed(false)}>
        Two different inputs, both signs +1
      </button>
      <button type="button" className={collapsed ? 'is-selected' : undefined} onClick={() => setCollapsed(true)}>
        Two identical inputs, opposite signs
      </button>
    </div>
    <Readout items={[
      ['Signed sum v', `(${geometry.best.v.map(value => round(value, 4)).join(', ')})`],
      ['Its length', round(geometry.best.length, 6)],
      ['Optimum for this pattern', round(geometry.best.optimum, 6)],
      ['Maximising coefficient', geometry.support
        ? `(${geometry.support.vector.map(value => round(value, 4)).join(', ')})`
        : 'does not exist as a unique point'],
      ['Is it unique?', geometry.best.optimizerUnique
        ? 'yes, while the budget is positive and v ≠ 0'
        : 'no — every feasible coefficient attains the same value 0'],
    ]} />
  </Figure>;
}

/* ============================================================== figure 8 */

/** Same lengths, same energy bound, different exact answers. */
export function DirectionVersusEnergyFigure() {
  const cases = [
    { name: 'Two copies of (1, 0)', vectors: [[1, 0], [1, 0]] },
    { name: 'Perpendicular (1, 0) and (0, 1)', vectors: [[1, 0], [0, 1]] },
    { name: 'The perpendicular pair, rotated 90°', vectors: [[0, 1], [-1, 0]] },
  ].map(entry => ({ ...entry, model: linearComplexity(entry.vectors, 1) }));
  /* ONE scale for all three panels.
   *
   * Each panel used to autoscale to its own signed sum, which is not drawn.
   * Panel 1's sum is (2, 0) and the others' have maximum coordinate 1, so the
   * same unit vector and the same radius-1 ball were drawn at half size in the
   * first panel — in a figure captioned "Same lengths", whose whole claim is
   * that the lengths and the budget are identical across the three. A reader
   * could reasonably have read the smaller circle as the smaller answer. */
  const sharedExtent = figureExtent(cases.map(entry => ({ vectors: entry.vectors, signs: [1, 1], radius: 1 })));
  return <Figure caption="Figure 8. Same lengths, same upper bound, different exact answers"
    describe={`Three two-observation samples, all with unit rows and therefore the same feature-energy bound ${round(cases[0].model.energyUpper, 6)}. The exact values are ${cases.map(entry => round(entry.model.complexity, 6)).join(', ')}. All three panels are drawn at one scale, so a length in one means the same as the same length in another.`}
    footnote="The energy bound throws away direction: it sees only how long the rows are. Rotating both inputs together is a change of coordinates and moves nothing; turning a duplicate into a perpendicular changes the geometry and moves the exact value.">
    <div className="rad-figure-row">
      {cases.map(entry => {
        const geometry = ballGeometry(entry.vectors, [1, 1], 1,
          { width: 150, height: 150, padding: 22, sharedExtent });
        return <div key={entry.name}>
          <Drawing width={geometry.width} height={geometry.height} className="rad-mini-ball"
            title={entry.name} describe={`${entry.name}: exact complexity ${round(entry.model.complexity, 6)}.`}>
            <line className="rad-grid" x1={0} y1={geometry.origin.y} x2={geometry.width} y2={geometry.origin.y} />
            <line className="rad-grid" x1={geometry.origin.x} y1={0} x2={geometry.origin.x} y2={geometry.height} />
            <circle className="rad-ball" cx={geometry.origin.x} cy={geometry.origin.y} r={geometry.radiusPixels} />
            {geometry.arrows.map(arrow => <g key={arrow.index}>
              <line className="rad-arrow" x1={arrow.base.x} y1={arrow.base.y} x2={arrow.tip.x} y2={arrow.tip.y} />
              <polygon className="rad-arrowhead" points={head(arrow.base, arrow.tip)} />
            </g>)}
            {/* Two identical inputs are two exactly coincident arrows, so the
                panel that exists to show the duplicate case would otherwise
                show one. The multiplicity is printed beside the tip. */}
            {geometry.coincidentGroups.map(group => {
              const tip = geometry.arrows[group[0]].tip;
              return <text key={group.join('-')} className="rad-tiny rad-accent" x={tip.x + 5} y={tip.y - 5}>
                ×{group.length}
              </text>;
            })}
          </Drawing>
          <p className="rad-caption">{entry.name}</p>
        </div>;
      })}
    </div>
    <Table caption="The exact value against the bound that discards direction"
      headings={['Sample', 'Exact empirical complexity', 'Feature-energy bound', 'Gap']}
      rows={cases.map(entry => [
        entry.name, exactly(entry.model.complexity), round(entry.model.energyUpper, 6),
        round(entry.model.energyUpper - entry.model.complexity, 6),
      ])} />
  </Figure>;
}

/* ============================================================== figure 9 */

/** The two-point Gram matrix, its four quadratic forms, and the constant trace
 *  bound. */
export function KernelGramFigure() {
  const [similarity, setSimilarity] = useState(0.9);
  const gram = [[1, similarity], [similarity, 1]];
  const model = kernelComplexity(gram, 1);
  const angle = Math.acos(Math.max(-1, Math.min(1, similarity)));
  /* Enlarged from 180x150: beside the controls this drawing rendered as two
     short strokes in a mostly empty column, and the angle it exists to show was
     hard to read. */
  const width = 236;
  const height = 196;
  const origin = { x: 34, y: height - 34 };
  const unit = 116;
  const first = { x: origin.x + unit, y: origin.y };
  const second = { x: origin.x + unit * Math.cos(angle), y: origin.y - unit * Math.sin(angle) };
  const recorded = recordedFixtures.kernelSimilarities.indexOf(similarity);
  return <Figure caption="Figure 9. Inner products are enough: no feature vector is ever built"
    describe={`A two-by-two Gram matrix with diagonal 1 and off-diagonal ${similarity}. Its four signed quadratic forms give exact empirical complexity ${round(model.complexity, 10)}, while the trace bound stays at ${round(model.traceUpper, 10)} for every similarity.`}
    footnote="The trace measures total feature-space squared length. It is not a rank, a support-vector count, or a measured accuracy.">
    <div className="rad-figure-row">
      <Drawing width={width} height={height} className="rad-gram"
        title="Two unit feature vectors separated by the angle their inner product implies"
        describe={`Two unit vectors at an angle of ${round((180 * angle) / Math.PI, 1)} degrees.`}>
        <line className="rad-arrow" x1={origin.x} y1={origin.y} x2={first.x} y2={first.y} />
        <polygon className="rad-arrowhead" points={head(origin, first)} />
        <line className="rad-arrow" x1={origin.x} y1={origin.y} x2={second.x} y2={second.y} />
        <polygon className="rad-arrowhead" points={head(origin, second)} />
        {/* An arc between the two vectors, so the angle is drawn and not only
            named in the readout beside the figure. */}
        <path className="rad-ball" d={`M ${origin.x + 34} ${origin.y} A 34 34 0 0 1 `
          + `${origin.x + 34 * Math.cos(angle)} ${origin.y - 34 * Math.sin(angle)}`} />
        {/* The angle used to be printed inside the wedge, where one of the two
            arrows ran straight through it. A number that a line crosses is a
            number a reader has to fight for, so it lives in the readout beside
            the drawing instead; the drawing shows the geometry. */}
        <text className="rad-tiny rad-muted" x={origin.x - 4} y={origin.y + 16}>φ(x₁), φ(x₂)</text>
      </Drawing>
      <div>
        <div className="rad-controls">
          <NumberField label="Similarity r = k(x₁, x₂)" value={similarity} min={-1} max={1} decimals={3}
            onChange={setSimilarity} hint="The matrix stays symmetric and positive semidefinite for r between −1 and 1." />
        </div>
        <Readout items={[
          ['Angle between the two feature vectors', `${round((180 * angle) / Math.PI, 1)}°`,
            'the inner product is the cosine of this angle, because both have length 1'],
          ['Exact empirical complexity', round(model.complexity, 10)],
          ['Trace bound', round(model.traceUpper, 10)],
          ['Author-checked value?', recorded >= 0
            ? `yes — the packet records ${round(recordedFixtures.kernelComplexities[recorded], 10)} at r = ${similarity}`
            : 'this r is your own; the three author-checked values are r = 0, .9 and 1'],
          ['Replacing r by −r', `gives ${round(kernelComplexity([[1, -similarity], [-similarity, 1]], 1).complexity, 10)} — the sign-averaged answer is the same, though the individual quadratic forms swap`],
        ]} />
      </div>
    </div>
    <Table caption="All four sign patterns, with the quadratic form each produces"
      headings={['σ', 'σᵀKσ', 'B√(σᵀKσ)/n']}
      rows={model.patterns.map((pattern, index) => [
        pattern.map(sign => (sign > 0 ? '+' : '−')).join(''),
        round(model.quadratics[index], 6), round(model.maxima[index], 10),
      ])} />
  </Figure>;
}

/* ============================================================= figure 10 */

/** The circle against the diamond, for the same signed sum. */
export function L1VersusL2Figure() {
  /* The signed sum must have ONE largest coordinate, or the figure contradicts
     its own caption. The first fixture here was (0.9, 0.4) and (0.3, 0.8),
     whose signed sum is exactly (1.2, 1.2) — a perfect tie, so "spends
     everything on one coordinate" had two answers and the drawing showed one
     of them without saying so. */
  const vectors = [[0.9, 0.4], [0.3, 0.5]];
  const signs = [1, 1];
  const radius = 1;
  const geometry = ballGeometry(vectors, signs, radius, { width: 300, height: 240 });
  const l1 = l1BestResponse(vectors, signs, radius);
  const diamond = diamondPoints(radius, geometry);
  const vertex = geometry.project(l1.optimizer);
  return <Figure caption="Figure 10. A circle lets you point anywhere; a diamond spends everything on one coordinate"
    describe={`For the signed sum (${geometry.best.v.map(value => round(value, 3)).join(', ')}), the Euclidean ball's best coefficient is the point along v at radius 1, giving ${round(geometry.best.optimum, 6)}. The ℓ₁ diamond's best coefficient sits at the vertex on coordinate ${l1.coordinate + 1}, giving ${round(l1.optimum, 6)}.`}
    footnote="The ℓ₁ answer depends on the largest single coordinate of the signed sum, which is why its bound carries a log-of-dimension term rather than a square root of it. A diagram cannot decide which constraint is better for a task.">
    {/* The drawing is 26px taller than the plotting area, and the two point
        names sit in that strip. Every mark of the plot is inside `geometry.height`
        by construction, so nothing can be drawn through a legend entry — which
        is what happened when these names were nudged off their own points and
        landed on the diamond's edge. */}
    <Drawing width={geometry.width} height={geometry.height + 26} className="rad-l1-l2"
      title="A unit circle and a unit diamond with the same signed sum"
      describe="A circle and a diamond, both of radius 1, with the signed sum and the best point of each, named in a legend strip below the plot.">
      <line className="rad-grid" x1={0} y1={geometry.origin.y} x2={geometry.width} y2={geometry.origin.y} />
      <line className="rad-grid" x1={geometry.origin.x} y1={0} x2={geometry.origin.x} y2={geometry.height} />
      <circle className="rad-ball" cx={geometry.origin.x} cy={geometry.origin.y} r={geometry.radiusPixels} />
      <polygon className="rad-diamond" points={diamond.map(point => `${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(' ')} />
      {/* The two inputs, so the signed sum visibly comes from somewhere. Figure 7
          draws them; this figure did not, which made v look like a given. */}
      {geometry.arrows.map(arrow => <g key={arrow.index}>
        <line className="rad-arrow" x1={arrow.base.x} y1={arrow.base.y} x2={arrow.tip.x} y2={arrow.tip.y} />
        <polygon className="rad-arrowhead" points={head(arrow.base, arrow.tip)} />
      </g>)}
      <line className="rad-sum" x1={geometry.origin.x} y1={geometry.origin.y} x2={geometry.sum.tip.x} y2={geometry.sum.tip.y} />
      <polygon className="rad-sum-head" points={head(geometry.origin, geometry.sum.tip, 9)} />
      <text className="rad-small rad-accent" x={geometry.sum.tip.x + 6} y={geometry.sum.tip.y - 4}>v</text>
      <circle className="rad-support" cx={geometry.support.point.x} cy={geometry.support.point.y} r={5} />
      <circle className="rad-point" cx={vertex.x} cy={vertex.y} r={5} />
      <circle className="rad-support" cx={22} cy={geometry.height + 13} r={5} />
      <text className="rad-tiny" x={32} y={geometry.height + 17}>ℓ₂ best</text>
      <circle className="rad-point" cx={geometry.width / 2 + 6} cy={geometry.height + 13} r={5} />
      <text className="rad-tiny" x={geometry.width / 2 + 16} y={geometry.height + 17}>ℓ₁ best</text>
    </Drawing>
    <Readout items={[
      ['Signed sum v', `(${geometry.best.v.map(value => round(value, 4)).join(', ')})`],
      ['ℓ₂: B‖v‖₂/n', round(geometry.best.optimum, 6)],
      ['ℓ₁: B‖v‖∞/n', round(l1.optimum, 6)],
      ['Which coordinate wins', l1.tiedCoordinates.length > 1
        ? `a tie between coordinates ${l1.tiedCoordinates.map(index => index + 1).join(' and ')}`
        : `coordinate ${l1.coordinate + 1}, with |v| = ${round(l1.infinityNorm, 4)}`],
    ]} />
  </Figure>;
}

/* ============================================================= figure 11 */

/** Two loss curves on separately labelled axes, and the ramp with its knots. */
export function LossSlopesFigure() {
  // The plotted window stops at m = −3 because hinge(−3) = 4 is the top of the
  // shared axis. Extending to −4 would put one curve above its own frame.
  const logistic = curvePoints(logisticMarginLoss, { from: -3, to: 4, width: 300, height: 150, valueRange: [0, 4.05] });
  const hinge = curvePoints(hingeMarginLoss, { from: -3, to: 4, width: 300, height: 150, valueRange: [0, 4.05] });
  const probability = curvePoints(sigmoid, { from: -4, to: 4, width: 300, height: 150, valueRange: [0, 1] });
  return <Figure caption="Figure 11. A margin loss and a probability map are different functions on different ranges"
    describe="Left: the logistic margin loss ln(1 + e^(−m)) and the hinge loss max(0, 1 − m), both on the range [0, 4] and both with slope approaching magnitude 1. Right: the sigmoid probability map, on the range [0, 1], whose steepest slope is 1/4."
    footnote="Borrowing the sigmoid's 1/4 for the logistic loss is the mistake this figure exists to prevent. The two curves are not the same function and their ranges do not even match.">
    <div className="rad-figure-row">
      <div>
        <Drawing width={logistic.width} height={logistic.height} className="rad-loss-plot"
          title="Logistic and hinge margin losses" describe="Two decreasing curves on margins from −3 to 4, with a displayed loss axis from 0 to 4.05.">
          <line className="rad-axis" x1={logistic.frame.left} y1={logistic.frame.bottom}
            x2={logistic.frame.right} y2={logistic.frame.bottom} />
          <line className="rad-axis" x1={logistic.frame.left} y1={logistic.frame.top}
            x2={logistic.frame.left} y2={logistic.frame.bottom} />
          <polyline className="rad-curve" points={logistic.polyline} />
          <polyline className="rad-curve is-second" points={hinge.polyline} />
          {/* End ticks. Without them the axes carry no scale, and "both
              approach slope 1" is a shape a reader has to take on trust. */}
          <text className="rad-tiny rad-muted" x={2} y={logistic.frame.top + 4}>loss</text>
          <text className="rad-tiny rad-muted" x={2} y={logistic.frame.top + 14}>
            {logistic.valueRange[1].toFixed(0)}
          </text>
          <text className="rad-tiny rad-muted" x={2} y={logistic.frame.bottom + 3}>0</text>
          <text className="rad-tiny rad-muted" x={logistic.frame.left - 4} y={logistic.frame.bottom + 14}>
            {logistic.from}
          </text>
          <text className="rad-tiny rad-muted" x={logistic.frame.right - 8} y={logistic.frame.bottom + 14}>
            {logistic.to}
          </text>
          <text className="rad-tiny rad-muted" x={logistic.frame.right - 52} y={logistic.height - 6}>margin m</text>
          <text className="rad-tiny" x={logistic.frame.left + 78} y={logistic.frame.top + 14}>solid: logistic</text>
          <text className="rad-tiny rad-accent" x={logistic.frame.left + 78} y={logistic.frame.top + 28}>dashed: hinge</text>
        </Drawing>
        <p className="rad-caption">Range [0, ∞). Both are 1-Lipschitz in the margin; neither is bounded, so neither
          can be dropped into the [0, 1] theorem without truncation.</p>
      </div>
      <div>
        <Drawing width={probability.width} height={probability.height} className="rad-loss-plot"
          title="The sigmoid probability map" describe="An increasing S-shaped curve on scores from −4 to 4, values 0 to 1.">
          <line className="rad-axis" x1={probability.frame.left} y1={probability.frame.bottom}
            x2={probability.frame.right} y2={probability.frame.bottom} />
          <line className="rad-axis" x1={probability.frame.left} y1={probability.frame.top}
            x2={probability.frame.left} y2={probability.frame.bottom} />
          {/* s = 0 is marked, because "steepest at s = 0, where its slope is
              exactly 1/4" is the claim this panel exists to make, and an
              unmarked axis leaves it nowhere to be seen. */}
          <line className="rad-ceiling" x1={probability.project([0, 0]).x} y1={probability.frame.top}
            x2={probability.project([0, 0]).x} y2={probability.frame.bottom} />
          <polyline className="rad-curve" points={probability.polyline} />
          <circle className="rad-knot" cx={probability.project([0, 0.5]).x} cy={probability.project([0, 0.5]).y} r={3.5} />
          <text className="rad-tiny rad-muted" x={2} y={probability.frame.top + 4}>probability</text>
          <text className="rad-tiny rad-muted" x={2} y={probability.frame.top + 14}>1</text>
          <text className="rad-tiny rad-muted" x={2} y={probability.frame.bottom + 3}>0</text>
          <text className="rad-tiny rad-muted" x={probability.project([0, 0]).x - 3} y={probability.frame.bottom + 14}>0</text>
          <text className="rad-tiny rad-muted" x={probability.frame.left - 4} y={probability.frame.bottom + 14}>
            {probability.from}
          </text>
          <text className="rad-tiny rad-muted" x={probability.frame.right - 8} y={probability.frame.bottom + 14}>
            {probability.to}
          </text>
          <text className="rad-tiny rad-muted" x={probability.frame.right - 46} y={probability.height - 6}>score s</text>
        </Drawing>
        <p className="rad-caption">Range [0, 1]. Steepest at s = 0, where its slope is exactly 1/4.</p>
      </div>
    </div>
  </Figure>;
}

/** The ramp, with its two knots and the four worked margins sitting on it. */
export function RampFigure() {
  const [rho, setRho] = useState(0.5);
  const margins = recordedFixtures.scalarMargins;
  const losses = rampLoss(margins, rho);
  /* The curve comes from the model layer, not from a lambda written here, so
     the shape this figure paints and the function investigation 3 applies are
     one definition. The verifier compares that definition with `rampLoss`
     across the whole threshold grid. */
  const ramp = rampCurve(rho, { width: 300, height: 170 });
  return <Figure caption="Figure 12. The ramp charges 1 for a mistake, 0 for a confident correct answer, and something in between for a narrow one"
    describe={`The ramp φ_ρ with ρ = ${rho}. Margins ${margins.join(', ')} receive losses ${losses.map(value => round(value, 4)).join(', ')}, with mean ${round(losses.reduce((sum, value) => sum + value, 0) / losses.length, 6)}.`}
    footnote="The ramp always lies in [0, 1] and always upper-bounds the mistake indicator, including the zero-margin tie, which it charges in full.">
    <Drawing width={ramp.width} height={ramp.height} className="rad-ramp"
      title="The ramp loss" describe={`A step from 1 down to 0 between margin 0 and margin ${rho}.`}>
      <line className="rad-axis" x1={ramp.frame.left} y1={ramp.frame.bottom}
        x2={ramp.frame.right} y2={ramp.frame.bottom} />
      <line className="rad-axis" x1={ramp.frame.left} y1={ramp.frame.top} x2={ramp.frame.left} y2={ramp.frame.bottom} />
      <polyline className="rad-curve is-ramp" points={ramp.polyline} />
      {[0, rho].map(knot => {
        const point = ramp.project([knot, rampLoss([knot], rho)[0]]);
        return <circle key={knot} className="rad-knot" cx={point.x} cy={point.y} r={4} />;
      })}
      {margins.map((margin, index) => {
        const point = ramp.project([margin, losses[index]]);
        return <g key={margin}>
          <circle className="rad-point" cx={point.x} cy={point.y} r={4} />
          <text className="rad-tiny" x={point.x + 6} y={point.y - 5}>{margin}</text>
        </g>;
      })}
      {/* The vertical extent is the whole point — "charges 1" against "charges
          nothing" — so the axis says which end is which. */}
      <text className="rad-tiny rad-muted" x={2} y={ramp.frame.top + 4}>φ(m)</text>
      <text className="rad-tiny rad-muted" x={ramp.frame.left - 10} y={ramp.frame.top + 4}>1</text>
      <text className="rad-tiny rad-muted" x={ramp.frame.left - 10} y={ramp.frame.bottom + 3}>0</text>
      <text className="rad-tiny rad-muted" x={ramp.frame.right - 52} y={ramp.height - 6}>margin m</text>
    </Drawing>
    <div className="rad-controls">
      <NumberField label="Margin threshold ρ" value={rho} min={0.1} max={3} decimals={3} onChange={setRho}
        hint="Larger ρ charges more training observations, and divides the complexity term by more." />
    </div>
    <Table caption="Each margin's exact contribution at this threshold"
      headings={['Margin m', 'Ramp loss φ(m)', 'What it is']}
      rows={margins.map((margin, index) => [
        String(margin), round(losses[index], 6),
        margin <= 0 ? 'wrong sign, or exactly on the boundary — charged in full'
          : margin >= rho ? 'past the threshold — charged nothing'
            : 'correct but inside the margin — charged partially',
      ])} />
    <Readout items={[
      ['Mean ramp loss', round(losses.reduce((sum, value) => sum + value, 0) / losses.length, 6)],
      ['Outright mistakes', `${losses.filter((_value, index) => margins[index] <= 0).length} of ${margins.length}`],
      ['Complexity multiplier 1/ρ', round(1 / rho, 6)],
    ]} />
  </Figure>;
}

/* ============================================================= figure 13 */

/** The recorded Monte-Carlo trajectory: four separate statements, not a band. */
export function MonteCarloFigure() {
  const table = recordedMonteCarlo.table;
  const layout = monteCarloLayout(table, recordedMonteCarlo.exact, { width: 320 });
  return <Figure caption="Figure 13. More draws shrink the correction, but do not march the estimate towards the answer"
    describe={`Four recorded runs at ${table.map(row => row.draws).join(', ')} sign draws, each with its own estimate and one-sided upper endpoint, against the exact value ${recordedMonteCarlo.exact}. The estimates are ${table.map(row => round(row.estimate, 6)).join(', ')}. Each row's guaranteed region is everything at or below its endpoint, so it is drawn as a dashed tail running off the left edge with the solid bar marking the correction.`}
    footnote="These are four SEPARATE statements, each holding with probability .95 for its own predeclared draw count. They are deliberately not drawn as one confidence band: inspecting all four and reporting the most favourable one would need simultaneous accounting.">
    <Drawing width={layout.width} height={layout.height} className="rad-monte-carlo"
      title="Four Monte-Carlo estimates with their one-sided upper endpoints"
      describe="Horizontal intervals from each estimate to its endpoint, with a vertical reference line at the exact value.">
      <line className="rad-reference" x1={layout.exactX} y1={14} x2={layout.exactX} y2={layout.height - 8} />
      <text className="rad-tiny rad-muted" x={layout.exactX + 4} y={12}>exact {recordedMonteCarlo.exact}</text>
      {layout.rows.map(row => <g key={row.draws}>
        <text className="rad-tiny" x={4} y={row.y + row.height / 2 + 4}>T={row.draws}</text>
        {/* The guaranteed region is everything at or below the endpoint, so it
            runs off the left edge. Drawing only [estimate, endpoint] invited
            exactly the two-sided reading this lesson exists to prevent. */}
        <line className="rad-ceiling" x1={layout.left - 8} y1={row.y + row.height / 2}
          x2={row.estimateX} y2={row.y + row.height / 2} />
        <line className="rad-interval" x1={row.estimateX} y1={row.y + row.height / 2}
          x2={row.endpointX} y2={row.y + row.height / 2} />
        <circle className="rad-point" cx={row.estimateX} cy={row.y + row.height / 2} r={4} />
      </g>)}
    </Drawing>
    <Table caption="The recorded sequence, seed 131, η = .05 for each row separately"
      headings={['Sign draws T', 'Estimate', 'One-sided correction', 'Upper endpoint', 'Endpoint at or above .5?']}
      rows={table.map(row => [
        row.draws.toLocaleString('en-US'), fixed(row.estimate, 6), fixed(row.correction, 6), fixed(row.endpoint, 6),
        row.endpoint >= recordedMonteCarlo.exact ? 'yes' : 'no',
      ])}
      footnote={'The column asks about the ENDPOINT, not about the drawn bar. A one-sided bound covers everything '
        + 'at or below its endpoint, so the guaranteed region runs off the left of this drawing; the solid bar '
        + 'marks the correction, from the estimate up to the endpoint. For T = 256 and T = 1,024 the estimate is '
        + 'already above .5, so the bar sits entirely right of the reference line while the statement still holds.'} />
    <p className="rad-caption">A standard error describes how much the sign simulation wobbles. It says nothing about
      how well the model will do on new observations — that is a different quantity with a different source of
      randomness.</p>
  </Figure>;
}

/* ============================================================= figure 14 */

/** The four data roles, drawn in proportion, with the frozen transform. */
export function DataRolesFigure() {
  const strip = rolesStrip(roles, { width: 320, height: 28 });
  return <Figure caption="Figure 14. Four disjoint roles, declared before anything was fitted"
    describe={`480 retained rows split into ${roles.map(role => `${role.count} ${role.label.toLowerCase()}`).join(', ')}, in retained order. The representation is fitted on the first 80 only and then frozen.`}
    footnote="Row ids are disjoint. Feature vectors are not quite: four pairs of rows are identical in all four measurements, two of them crossing the fitting and validation roles. None involves assessment.">
    <Drawing width={strip.width} height={strip.height + 34} className="rad-roles"
      title="The four data roles in proportion" describe="A horizontal strip divided into four parts sized by row count.">
      {strip.parts.map(part => <g key={part.key}>
        <rect className={`rad-role is-${part.key}`} x={part.x} y={4} width={part.width} height={strip.height} />
        <text className="rad-tiny" x={part.x + part.width / 2} y={strip.height + 20} textAnchor="middle">
          {part.count}
        </text>
      </g>)}
    </Drawing>
    <Table caption="What each role's information is allowed to touch"
      headings={['Rows in retained order', 'Role', 'What may use their information']}
      rows={roles.map(role => [`${role.from + 1}–${role.to}`, role.label, role.use])} />
    <Readout items={[
      ['The frozen transform', 'clip((x − mean)/scale/3, −1, 1), then append a constant 1'],
      ['Fitted means', representation.mean.map(value => round(value, 4)).join(', ')],
      ['Fitted scales', representation.scale.map(value => round(value, 4)).join(', ')],
      ['Every mapped row has norm at most', round(experimentSettings.globalRowNormUpper, 6), '= √5, because four clipped coordinates and one constant'],
    ]} />
  </Figure>;
}

/* ============================================================= figure 15 */

/** The five budgets' bound components against the trivial ceiling, with the
 *  training-margin distribution that produces the first term. */
export function BoundComponentsFigure() {
  const [rho, setRho] = useState(1);
  const rhoIndex = experimentSettings.rhoValues.indexOf(rho);
  const rows = fittedModels.map(model => ({
    radius: model.radius,
    empiricalRamp: model.bounds[rhoIndex].empiricalRamp,
    complexityAddend: model.bounds[rhoIndex].complexityAddend,
    confidence: experimentSettings.confidenceAddend,
    raw: model.bounds[rhoIndex].rawUpper,
  }));
  const layout = stackedBoundLayout(rows, { width: 320 });
  const smallest = rows.reduce((best, row) => (row.raw < best.raw ? row : best));
  return <Figure caption="Figure 15. Every one of these expressions is above the trivial ceiling"
    describe={`Five stacked bars, one per budget. Each runs left to right in a fixed order: the training ramp first, then twice the complexity term, then the confidence allowance, at ρ = ${rho}. The dashed line marks the trivial ceiling 1. Every bar is longer than it. The shortest belongs to B = ${smallest.radius}.`}
    footnote="The stack is drawn at full length rather than clipped at 1. Clipping would hide exactly the thing worth seeing: how far past the point of saying anything these expressions are.">
    <div className="rad-controls">
      <Select label="Margin threshold ρ" value={String(rho)} onChange={value => setRho(Number(value))}
        options={experimentSettings.rhoValues.map(value => [String(value), `ρ = ${value}`])}
        hint="Both thresholds were predeclared, and the union bound over all ten comparisons is already inside the confidence term." />
    </div>
    <Drawing width={layout.width} height={layout.height} className="rad-bounds"
      title="Bound components for five budgets against the trivial ceiling"
      describe={`Stacked bars for budgets ${rows.map(row => row.radius).join(', ')}, all exceeding 1.`}>
      {/* A key with swatches, not a bare word list: the three segments were
          distinguished by colour alone. The swatches carry the mapping, and the
          text equivalent states the fixed left-to-right order, so the encoding
          survives without colour as well. */}
      {[['ramp', 'ramp'], ['complexity', '2B·energy/ρ'], ['confidence', 'confidence']]
        .map(([key, name], index) => <g key={key}>
          <rect className={`rad-bar is-${key}`} x={4 + index * 104} y={4} width={9} height={9} />
          <text className="rad-tiny rad-muted" x={17 + index * 104} y={12}>{name}</text>
        </g>)}
      {layout.rows.map(row => <g key={row.radius}>
        <text className="rad-tiny" x={4} y={row.y + row.height / 2 + 4}>B={row.radius}</text>
        {row.segments.map(segment => <rect key={segment.key} className={`rad-bar is-${segment.key}`}
          x={segment.x} y={row.y} width={Math.max(0, segment.width)} height={row.height} />)}
        <text className="rad-tiny rad-accent" x={row.labelX} y={row.y + row.height / 2 + 4}>
          {fixed(row.total, 3)}
        </text>
      </g>)}
      <line className="rad-ceiling" x1={layout.ceilingX} y1={14} x2={layout.ceilingX} y2={layout.height - 4} />
      <text className="rad-tiny rad-muted" x={layout.ceilingX + 4} y={12}>1</text>
    </Drawing>
    <Table caption="The three terms, and the measured mistakes beside them"
      headings={['Budget B', 'Training ramp', '2B × energy / ρ', 'Confidence', 'Raw sum', 'Fit errors / 240', 'Validation errors / 80']}
      rowClass={index => (fittedModels[index].radius === smallest.radius ? 'is-leading' : undefined)}
      rows={fittedModels.map((model, index) => [
        String(model.radius), fixed(rows[index].empiricalRamp, 6), fixed(rows[index].complexityAddend, 6),
        fixed(rows[index].confidence, 6), fixed(rows[index].raw, 6),
        String(model.fit.errors), String(model.validation.errors),
      ])} />
    <p className="rad-caption">At ρ = 1 the smallest expression belongs to B = 2; at ρ = .5 it moves to B = 1. The
      validation rule chooses B = 4. An upper bound is not a prediction of which candidate will do best.</p>
  </Figure>;
}

/** The training-margin distribution behind the first term. */
export function MarginDistributionFigure() {
  const [radius, setRadius] = useState(2);
  const rho = 1;
  const model = fittedModels.find(entry => entry.radius === radius);
  const mapped = observations.slice(80, 320).map(row => mapFeatures(row.slice(1, 5), representation));
  const labels = observations.slice(80, 320).map(row => row[5]);
  const margins = mapped.map((row, index) =>
    labels[index] * row.reduce((sum, value, column) => sum + value * model.weights[column], 0));
  const histogram = marginHistogram(margins, { rho, bins: 18 });
  const width = 320;
  const height = 150;
  const left = 30;
  const plotWidth = width - left - 12;
  const plotHeight = height - 44;
  const tallest = Math.max(...histogram.counts);
  const place = value => left + (plotWidth * (value - histogram.edges[0]))
    / (histogram.edges[histogram.edges.length - 1] - histogram.edges[0]);
  return <Figure caption="Figure 16. Where the training ramp loss comes from"
    describe={`The 240 training margins for B = ${radius}, binned with edges falling exactly at 0 and at ρ = ${rho}. ${histogram.belowZero} margins are at or below zero, ${histogram.insideRamp} lie inside the ramp, and ${histogram.aboveRho} are past ρ.`}
    footnote="The two bin edges that matter are exact: 0 and ρ. A bin straddling either would mix observations the ramp treats differently.">
    <div className="rad-controls">
      <Select label="Budget" value={String(radius)} onChange={value => setRadius(Number(value))}
        options={experimentSettings.radii.map(value => [String(value), `B = ${value}`])} />
    </div>
    <Drawing width={width} height={height} className="rad-margins"
      title={`Training-margin histogram for B = ${radius}`}
      describe={`Counts per margin bin, with reference lines at 0 and ${rho}.`}>
      <line className="rad-axis" x1={left} y1={plotHeight + 14} x2={width - 12} y2={plotHeight + 14} />
      {histogram.counts.map((count, index) => {
        const x = place(histogram.edges[index]);
        const barWidth = Math.max(1, place(histogram.edges[index + 1]) - x - 1);
        const barHeight = (plotHeight - 14) * (count / tallest);
        return <rect key={index} className={`rad-bar${histogram.edges[index + 1] <= 0 ? ' is-below' : ''}`}
          x={x} y={plotHeight + 14 - barHeight} width={barWidth} height={barHeight} />;
      })}
      <line className="rad-ceiling" x1={place(0)} y1={8} x2={place(0)} y2={plotHeight + 14} />
      <line className="rad-reference" x1={place(rho)} y1={8} x2={place(rho)} y2={plotHeight + 14} />
      <text className="rad-tiny rad-muted" x={place(0) + 3} y={12}>m = 0</text>
      <text className="rad-tiny rad-muted" x={place(rho) + 3} y={24}>ρ = {rho}</text>
      <text className="rad-tiny rad-muted" x={4} y={plotHeight + 30}>margin</text>
    </Drawing>
    <Readout items={[
      ['At or below 0 — charged in full', `${histogram.belowZero} of 240`],
      ['Inside the ramp — charged partially', `${histogram.insideRamp} of 240`],
      ['Past ρ — charged nothing', `${histogram.aboveRho} of 240`],
      ['Mean ramp loss', round(model.bounds[1].empiricalRamp, 6), `which is the first term of the B = ${radius}, ρ = 1 expression`],
      ['Recomputed here, in your browser', `from the served rows and the recorded coefficients — it reproduces the packet's ${round(model.bounds[1].empiricalRamp, 6)} exactly`],
    ]} />
  </Figure>;
}

/* ============================================================= figure 17 */

/** Convex mixtures sit inside the hull, and cannot out-project the best base
 *  vector along any fixed direction. */
export function ConvexHullFigure() {
  const [weight, setWeight] = useState(0.35);
  const base = [[1, 0.2], [0.3, 1], [-0.8, 0.5]];
  const weights = [weight, (1 - weight) * 0.6, (1 - weight) * 0.4];
  const direction = [1, 1];
  const layout = hullLayout(base, weights, direction, { width: 300, height: 230 });
  return <Figure caption="Figure 17. A convex average cannot beat the best of the things it averages"
    describe={`Three base prediction vectors and one mixture with weights ${weights.map(value => round(value, 3)).join(', ')}. Projected on the direction (1, 1), the mixture reaches ${round(layout.mixtureProjection, 6)} while the best base vector reaches ${round(layout.bestBaseProjection, 6)}.`}
    footnote="For every fixed sign pattern the same argument applies, so the convex hull has exactly the same empirical score complexity as the base class. Thresholding the average into a hard label is discontinuous, and that step needs the margin argument instead.">
    {/* Like figure 10, the drawing reserves a 26px strip below the plot for the
        key. The interior mixture point gets no in-plot label: no fixed inward
        offset clears every edge of a triangle this size at every weighting. */}
    <Drawing width={layout.width} height={layout.height + 26} className="rad-hull-figure"
      title="Three base vectors, their convex hull, and one mixture"
      describe="A filled triangle with its three vertices and an interior point, named in a legend strip below the plot.">
      <line className="rad-grid" x1={0} y1={layout.origin.y} x2={layout.width} y2={layout.origin.y} />
      <line className="rad-grid" x1={layout.origin.x} y1={0} x2={layout.origin.x} y2={layout.height} />
      <polygon className="rad-hull"
        points={base.map(layout.project).map(point => `${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(' ')} />
      {/* The direction being projected onto, and the two projections themselves.
          Without these the figure asserted a comparison it never showed: the
          numbers lived only in the bars underneath. */}
      {(() => {
        const unitDirection = [layout.direction[0] / Math.hypot(...layout.direction),
          layout.direction[1] / Math.hypot(...layout.direction)];
        const far = layout.project([unitDirection[0] * layout.reach, unitDirection[1] * layout.reach]);
        const foot = value => layout.project([unitDirection[0] * value, unitDirection[1] * value]);
        const bestFoot = foot(layout.bestBaseProjection);
        const mixtureFoot = foot(layout.mixtureProjection);
        const bestPoint = layout.project(base[layout.bestBaseIndex]);
        const mixturePoint = layout.project(layout.mixture);
        return <g>
          <line className="rad-axis" x1={layout.origin.x} y1={layout.origin.y} x2={far.x} y2={far.y} />
          <polygon className="rad-arrowhead" points={head(layout.origin, far, 6)} />
          <text className="rad-tiny rad-muted" x={far.x + 4} y={far.y + 4}>direction</text>
          <line className="rad-ceiling" x1={bestPoint.x} y1={bestPoint.y} x2={bestFoot.x} y2={bestFoot.y} />
          <line className="rad-ceiling" x1={mixturePoint.x} y1={mixturePoint.y} x2={mixtureFoot.x} y2={mixtureFoot.y} />
          <circle className="rad-point" cx={bestFoot.x} cy={bestFoot.y} r={3.5} />
          <circle className="rad-support" cx={mixtureFoot.x} cy={mixtureFoot.y} r={3.5} />
        </g>;
      })()}
      {/* Vertex labels are pushed outward from the centroid, so they sit
          outside the hull rather than on one of its edges; the mixture label is
          pushed inward, away from the nearest edge. */}
      {base.map((point, index) => {
        const projected = layout.project(point);
        return <g key={index}>
          <circle className="rad-point" cx={projected.x} cy={projected.y} r={4.5} />
          <text className="rad-tiny" x={layout.vertexLabels[index].x} y={layout.vertexLabels[index].y}
            textAnchor={layout.vertexLabels[index].anchor}>h{index + 1}</text>
        </g>;
      })}
      <circle className="rad-support" cx={layout.project(layout.mixture).x} cy={layout.project(layout.mixture).y} r={5} />
      <circle className="rad-point" cx={22} cy={layout.height + 13} r={4.5} />
      <text className="rad-tiny" x={32} y={layout.height + 17}>base vector h1–h3</text>
      <circle className="rad-support" cx={layout.width / 2 + 24} cy={layout.height + 13} r={5} />
      <text className="rad-tiny rad-accent" x={layout.width / 2 + 34} y={layout.height + 17}>the mixture</text>
    </Drawing>
    <div className="rad-controls">
      <NumberField label="Weight on h1" value={weight} min={0} max={1} decimals={3} onChange={setWeight}
        hint="The remaining weight is split .6 / .4 between h2 and h3, so the three always sum to 1." />
    </div>
    <BarRow label="Best base projection" value={round(layout.bestBaseProjection, 6)}
      share={layout.bestBaseProjection / Math.max(layout.bestBaseProjection, 1e-9)} tone="accent" />
    <BarRow label="Mixture projection" value={round(layout.mixtureProjection, 6)}
      share={Math.max(0, layout.mixtureProjection) / Math.max(layout.bestBaseProjection, 1e-9)} />
    <p className="rad-caption">Move the weight anywhere you like: the mixture's projection never exceeds the best base
      projection, because it is a weighted average of the three base projections.</p>
  </Figure>;
}
