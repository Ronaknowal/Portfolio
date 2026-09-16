import {
  balancedWeights, classBands, costOf, countsAt, fixtures, focalMass, populationFlow, rankedQueue, weightedStep,
} from '../../data/imbalance-models';
import {
  inspectionRecords, methods, provenance, roles, study,
} from '../../data/imbalance-data';
import { Plot, Table, curve, fixed, round, signed } from './ImbalanceShared.jsx';
import './imbalance-labs.css';

/* Every proportion, length, coordinate and total drawn below comes out of
   imbalance-models.js or imbalance-data.js. Nothing here chooses a shape. */

/* ================================================================ figure 1 */

const model = classBands(fixtures.modelCounts);
const baseline = classBands(fixtures.baselineCounts);

/** Two class bands, each normalised within its own class, beside the exact
 * four-cell table. Separate normalisation is the whole point: it keeps six
 * missed positives visible without suggesting the classes are the same size. */
function Bands({ label, bands, describe }) {
  const width = 340;
  const rowHeight = 30;
  const left = 4;
  const right = 336;
  const span = right - left;
  return <svg viewBox={`0 0 ${width} ${bands.length * (rowHeight + 32) + 12}`} role="img" aria-label={describe}>
    <title>{label}</title>
    {bands.map((band, index) => {
      const top = index * (rowHeight + 32) + 20;
      let offset = left;
      return <g key={band.label}>
        <text x={left} y={top - 6} className="imb-small">
          {band.label}: {band.size} of {model.table.total} cases
        </text>
        {band.parts.map(part => {
          const partWidth = band.size === 0 ? 0 : span * part.withinClass;
          const x = offset;
          offset += partWidth;
          return <g key={part.key}>
            <rect className={`imb-band is-${part.key}`} x={x} y={top} width={Math.max(partWidth, 0)} height={rowHeight} />
            {partWidth > 46 && <text className="imb-halo imb-small" x={x + partWidth / 2} y={top + 19} textAnchor="middle">
              {part.name} {part.count}
            </text>}
          </g>;
        })}
        {/* The band outline must be a class, not a fill="none" attribute: a
            presentation attribute loses to the stylesheet, so `.imb-lane`'s
            opaque fill painted straight over the bands and their labels. */}
        <rect className="imb-lane is-outline" x={left} y={top} width={span} height={rowHeight} />
        {/* An overflow label is anchored to the side its own segment sits on.
            Always printing it at the right put "false alarm 18" some 300 px
            away from the six-unit sliver at the left edge that it names. */}
        {band.parts.filter(part => span * part.withinClass <= 46).map((part, slot) => {
          const before = band.parts.slice(0, band.parts.indexOf(part))
            .reduce((total, earlier) => total + span * earlier.withinClass, 0);
          const nearLeft = before + span * part.withinClass / 2 < span / 2;
          return <text key={part.key} className="imb-small imb-muted"
            x={nearLeft ? left : right} y={top + rowHeight + 14 + slot * 13}
            textAnchor={nearLeft ? 'start' : 'end'}>
            {part.name} {part.count} ({round(100 * part.withinClass, 1)}% of this class)
          </text>;
        })}
      </g>;
    })}
  </svg>;
}

export function CaseFlowFigure() {
  const table = model.table;
  const base = baseline.table;
  return <figure className="imb-figure">
    <figcaption>
      <strong>Figure 1 — Follow the cases, not just a percentage.</strong> One thousand cases, twenty of them actually
      positive. Each class band is normalised <em>within its own class</em>, so the six missed positives stay visible;
      the population totals beside them keep the real proportions. The two bands are not on a shared scale.
    </figcaption>
    <div className="imb-panels is-stacked">
      <div className="imb-panel">
        <h4>Where the {table.total.toLocaleString('en-US')} cases go under the actual model</h4>
        <Bands label="Class bands for the actual model" bands={model.bands}
          describe={`Two bands, each normalised within its own class. Among ${table.positives} actual positives, ${table.tp} are detected and ${table.fn} missed. Among ${table.negatives} actual negatives, ${table.fp} trigger an alert and ${table.tn} do not.`} />
        <p>
          The positive band magnifies its parts {round(table.total / table.positives, 0)} times relative to the page:
          the missed six are {round(100 * model.bands[0].parts[1].withinClass, 0)}% of their class and
          {' '}{round(100 * model.bands[0].parts[1].ofPopulation, 1)}% of all cases. That is the reason for two scales
          rather than one.
        </p>
      </div>
      <div className="imb-panel">
        <h4>The exact four cells</h4>
        <Table caption="Actual class by predicted class, for the model that detects fourteen"
          headings={['actual class', 'predicted negative', 'predicted positive', 'class total']}
          rows={[
            ['negative', `TN = ${table.tn}`, `FP = ${table.fp}`, String(table.negatives)],
            ['positive', `FN = ${table.fn}`, `TP = ${table.tp}`, String(table.positives)],
            ['column total', String(table.cleared), String(table.alerts), String(table.total)],
          ]}
          footnote={`The alert column holds ${table.alerts} cases, so precision is ${table.tp}/${table.alerts} = ${round(table.precision, 4)}. No cost appears here: costs are defined in section 3.`} />
      </div>
      <div className="imb-panel">
        <h4>Two rules, side by side</h4>
        <Table caption="The always-negative baseline against the model, with accuracy, recall and counts in separate columns"
          headings={['rule', 'accuracy', 'recall', 'detected', 'missed', 'false alarms', 'errors']}
          rows={[
            ['always negative', `${round(100 * base.accuracy, 1)}%`, `${round(100 * base.recall, 1)}%`,
              String(base.tp), String(base.fn), String(base.fp), String(base.fn + base.fp)],
            ['the actual model', `${round(100 * table.accuracy, 1)}%`, `${round(100 * table.recall, 1)}%`,
              String(table.tp), String(table.fn), String(table.fp), String(table.fn + table.fp)],
          ]}
          footnote={`The baseline is the more accurate of the two and detects nothing; its precision is undefined, because it selects no case at all. Accuracy and recall are separate columns because they answer separate questions.`} />
      </div>
    </div>
  </figure>;
}

/* ================================================================ figure 2 */

const flowA = populationFlow(fixtures.flowA);
const flowB = populationFlow(fixtures.flowB);

/** One class-origin flow diagram: two source boxes, four flow curves, one alert
 * tray. No arrowheads are drawn anywhere in this lesson, so nothing visible on
 * the page speaks of arrows; direction is carried by left-to-right reading. */
function Flow({ flow, label, describe }) {
  const width = 340;
  const height = 210;
  const sourceX = 8;
  const trayX = 214;
  const boxWidth = 116;
  const print = value => (Number.isInteger(value) ? String(value) : round(value, 1));
  const rows = [
    { key: 'tp', from: 34, to: 44, text: `detected ${print(flow.tp)}`, className: 'imb-flow' },
    { key: 'fn', from: 34, to: 150, text: `missed ${print(flow.fn)}`, className: 'imb-flow is-cleared' },
    { key: 'fp', from: 142, to: 78, text: `false alarm ${print(flow.fp)}`, className: 'imb-flow is-alarm' },
    { key: 'tn', from: 142, to: 184, text: `cleared ${print(flow.tn)}`, className: 'imb-flow is-cleared' },
  ];
  return <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={describe}>
    <title>{label}</title>
    <rect className="imb-lane" x={sourceX} y={16} width={boxWidth} height={42} rx="3" />
    <text className="imb-halo imb-small" x={sourceX + 6} y={34}>positives</text>
    <text className="imb-halo imb-small imb-strong" x={sourceX + 6} y={50}>{print(flow.positives)}</text>
    <rect className="imb-lane" x={sourceX} y={124} width={boxWidth} height={42} rx="3" />
    <text className="imb-halo imb-small" x={sourceX + 6} y={142}>negatives</text>
    <text className="imb-halo imb-small imb-strong" x={sourceX + 6} y={158}>{print(flow.negatives)}</text>
    {rows.map(row => <g key={row.key}>
      <path className={row.className}
        d={`M ${sourceX + boxWidth} ${row.from} C ${sourceX + boxWidth + 34} ${row.from}, ${trayX - 34} ${row.to}, ${trayX} ${row.to}`} />
      <text className="imb-halo imb-small" x={trayX + 6} y={row.to + 4}>{row.text}</text>
    </g>)}
    <rect className="imb-lane is-fitting" x={trayX - 4} y={30} width={2} height={60} />
    <text className="imb-small imb-muted" x={sourceX} y={192}>
      alert tray: {print(flow.alerts)} cases
    </text>
    <text className="imb-small imb-strong" x={sourceX} y={206}>
      precision {round(flow.precision, 6)}
    </text>
  </svg>;
}

export function PrevalenceFigure() {
  const rocWidth = 340;
  // Tall enough for the plot, one row of tick labels, and two caption lines
  // beneath them: a caption sharing the tick row lands on top of the numbers.
  const rocHeight = 178;
  const rocAxisY = rocHeight - 58;
  // A false-positive rate of .01 would be a pixel from the axis at full scale,
  // so the marker is drawn on a zoomed axis whose range is stated in the label.
  const zoom = 0.05;
  const place = value => 48 + (rocWidth - 62) * (value / zoom);
  const lift = value => rocAxisY - (rocAxisY - 22) * value;
  return <figure className="imb-figure">
    <figcaption>
      <strong>Figure 2 — One detector, two populations.</strong> The true-positive rate stays at {flowA.tpr} and the
      false-positive rate at {flowA.fpr} in both panels. Only the mix of the {flowA.population.toLocaleString('en-US')}
      {' '}cases changes. These are <em>expected counts under the declared rates</em>, not a retrained classifier and not
      a measured prevalence-shift experiment; the second panel's counts are deliberately left fractional.
    </figcaption>
    <div className="imb-panels is-pair">
      <div className="imb-panel">
        <h4>Prevalence {flowA.prevalence}</h4>
        <Flow flow={flowA} label="Population flow at one percent prevalence"
          describe={`Of ${flowA.population} cases, ${flowA.positives} are positive: ${flowA.tp} detected and ${flowA.fn} missed. Of ${flowA.negatives} negatives, ${flowA.fp} raise a false alarm and ${flowA.tn} are cleared. The alert tray holds ${flowA.alerts} cases and precision is ${round(flowA.precision, 6)}.`} />
        <p>Precision is {flowA.tp}/{flowA.alerts} = {round(flowA.precision, 6)}. Every count here is a whole number.</p>
      </div>
      <div className="imb-panel">
        <h4>Prevalence {flowB.prevalence}</h4>
        <Flow flow={flowB} label="Population flow at one tenth of one percent prevalence"
          describe={`Of ${flowB.population} cases, ${flowB.positives} are positive: ${flowB.tp} detected and ${flowB.fn} missed. Of ${flowB.negatives} negatives, an expected ${round(flowB.fp, 1)} raise a false alarm and an expected ${round(flowB.tn, 1)} are cleared. The alert tray holds an expected ${round(flowB.alerts, 1)} cases and precision is ${round(flowB.precision, 6)}.`} />
        <p>
          Precision is {flowB.tp}/{round(flowB.alerts, 1)} = {round(flowB.precision, 6)}. Rounding the expected
          {' '}{round(flowB.fp, 1)} up to 100 first and then dividing would give {round(8 / 108, 6)} — a different
          number, which is why the fractional expectation is kept.
        </p>
      </div>
    </div>
    <div className="imb-panel">
      <h4>The ROC operating point does not move</h4>
      <svg viewBox={`0 0 ${rocWidth} ${rocHeight}`} role="img"
        aria-label={`A zoomed ROC axis running from a false-positive rate of 0 to ${zoom}. One marker sits at false-positive rate ${flowA.fpr} and true-positive rate ${flowA.tpr}, and it is the same marker for both populations.`}>
        <title>The shared ROC operating point</title>
        {[0, 0.0125, 0.025, 0.0375, 0.05].map(value => <g key={value}>
          <line className="imb-grid" x1={place(value)} x2={place(value)} y1={16} y2={rocAxisY} />
          <text className="imb-small" x={place(value)} y={rocAxisY + 16} textAnchor="middle">{value}</text>
        </g>)}
        {[0, 0.5, 1].map(value => (
          <text key={value} className="imb-small" x={42} y={lift(value) + 4} textAnchor="end">{value}</text>
        ))}
        <line className="imb-axis" x1={48} x2={rocWidth - 14} y1={rocAxisY} y2={rocAxisY} />
        <line className="imb-axis" x1={48} x2={48} y1={16} y2={rocAxisY} />
        <circle className="imb-mark" cx={place(flowA.fpr)} cy={lift(flowA.tpr)} r="5" />
        <circle className="imb-mark is-hollow" cx={place(flowB.fpr)} cy={lift(flowB.tpr)} r="9" />
        <text className="imb-halo imb-small" x={place(flowA.fpr) + 13} y={lift(flowA.tpr) + 4}>
          both populations: FPR {flowA.fpr}, TPR {flowA.tpr}
        </text>
        <text className="imb-small imb-muted" x={2} y={rocAxisY + 34}>
          horizontal: false-positive rate, zoomed to 0–{zoom}
        </text>
        <text className="imb-small imb-muted" x={2} y={rocAxisY + 46}>
          vertical: true-positive rate
        </text>
      </svg>
      <p>
        The same marker serves both panels, drawn once filled and once as a ring around it so the coincidence is
        visible rather than implied. The fixed-conditionals assumption is what makes that legitimate: within each
        class, the score distribution is declared unchanged.
      </p>
    </div>
  </figure>;
}

/* ================================================================ figure 3 */

const step = weightedStep({ rows: fixtures.stepRows, rate: fixtures.stepRate, penalty: fixtures.stepPenalty });
const balanced = balancedWeights([study.fittingNegatives, study.fittingPositives]);

export function WeightedStepFigure() {
  return <figure className="imb-figure">
    <figcaption>
      <strong>Figure 3 — A weighted update has two sums.</strong> Two rows, one gradient step of {fixtures.stepRate}
      {' '}from zero parameters. The plotted vertical coordinate is a <em>predicted score</em>, not an observed event
      frequency, and one step is an illustration rather than a converged fit.
    </figcaption>
    <Table caption="Each row's residual, weight and contribution to the two gradient sums"
      headings={['row', 'feature x', 'label y', 'weight a', 'score q', 'residual q − y', 'intercept term a(q−y)', 'coefficient term a(q−y)x']}
      rows={step.rows.map(row => [
        row.id, round(row.x, 2), String(row.y), round(row.weight, 2), round(row.score, 6),
        signed(row.residual, 6), signed(row.interceptContribution, 6), signed(row.coefficientContribution, 6),
      ])}
      footnote={`Total weight A = ${round(step.totalWeight, 2)}. The objective divides by A, so multiplying every weight by the same positive factor leaves both gradients exactly where they are.`} />
    <div className="imb-panels is-pair">
      <div className="imb-panel">
        <h4>The two sums</h4>
        <p>
          Intercept: ({signed(step.rows[0].interceptContribution, 2)} {signed(step.rows[1].interceptContribution, 2)})
          {' '}/ {round(step.totalWeight, 0)} = {signed(step.interceptGradient, 6)}.
        </p>
        <p>
          Coefficient: ({signed(step.rows[0].coefficientContribution, 2)} {signed(step.rows[1].coefficientContribution, 2)})
          {' '}/ {round(step.totalWeight, 0)} = {signed(step.coefficientGradient, 6)}. The intercept is unpenalised, and
          at zero parameters the penalty term contributes {round(step.penaltyContribution, 6)}.
        </p>
        <p>
          A step of {fixtures.stepRate} gives intercept {round(step.interceptAfter, 6)} and coefficient
          {' '}{round(step.coefficientAfter, 6)}. The two rows then score {round(step.scoresAfter[0], 6)} and
          {' '}{round(step.scoresAfter[1], 6)}.
        </p>
      </div>
      <div className="imb-panel">
        <Plot caption="Predicted score against the feature, before and after one step"
          describe={`A horizontal axis from 0 to 2 and a vertical predicted score from 0 to 1. The dashed line before the step is flat at 0.5 across the whole interval. The solid line after the step rises from ${round(step.after(0), 6)} at x = 0 to ${round(step.after(2), 6)} at x = 2.`}
          domain={[0, 2]} range={[0, 1]} ticks={[0, 0.5, 1, 1.5, 2]} valueTicks={[0, 0.5, 1]} height={190}>
          {(scaleX, scaleY) => <>
            <polyline className="imb-curve is-before" points={curve(scaleX, scaleY, [0, 2], step.before)} />
            <polyline className="imb-curve is-after" points={curve(scaleX, scaleY, [0, 2], step.after)} />
            {step.rows.map((row, index) => <g key={row.id}>
              <circle className="imb-mark" cx={scaleX(row.x)} cy={scaleY(step.scoresAfter[index])} r="4" />
              <text className="imb-halo imb-small" x={scaleX(row.x) + (index === 0 ? 8 : -8)}
                y={scaleY(step.scoresAfter[index]) - 8} textAnchor={index === 0 ? 'start' : 'end'}>
                {row.id} {round(step.scoresAfter[index], 4)}
              </text>
            </g>)}
          </>}
        </Plot>
        <p className="imb-legend">
          <span><svg viewBox="0 0 34 12" aria-hidden="true"><line className="imb-curve is-before" x1="1" x2="33" y1="6" y2="6" /></svg> before, flat at 0.5</span>
          <span><svg viewBox="0 0 34 12" aria-hidden="true"><line className="imb-curve is-after" x1="1" x2="33" y1="6" y2="6" /></svg> after one step</span>
        </p>
      </div>
    </div>
    <p>
      A weight of three on the second row is not a third observation. Three separate rows of weight one give a
      different normalised gradient here, and neither arrangement adds information about an unseen positive subtype.
      The balanced convention on this lesson's own fitting rows gives weights {round(balanced.weights[1], 6)} for the
      {' '}{study.fittingPositives} positives and {round(balanced.weights[0], 6)} for the {study.fittingNegatives}
      {' '}negatives, so each class carries a total weight of {round(balanced.totalPerClass[0], 0)}.
    </p>
  </figure>;
}

/* ================================================================ figure 4 */

export function PipelineFigure() {
  const width = 340;
  const height = 300;
  const lanes = [
    { key: 'fitting', label: 'fitting', role: roles.fitting, y: 74, className: 'imb-lane is-fitting' },
    { key: 'tuning', label: 'tuning', role: roles.tuning, y: 140, className: 'imb-lane is-tuning' },
    { key: 'inspection', label: 'inspection', role: roles.inspection, y: 196, className: 'imb-lane is-inspection' },
    { key: 'reserve', label: 'reserved', role: roles.reserve, y: 252, className: 'imb-lane is-reserve' },
  ];
  return <figure className="imb-figure">
    <figcaption>
      <strong>Figure 4 — A fork in the pipeline, not a transform applied everywhere.</strong> The fitting branch learns
      the scaler, resamples and fits the classifier. The assessment branches only transform with the saved scaler and
      predict. Read the connectors left to right: <em>no connector carries a sampler into an assessment branch</em>,
      and <em>no connector leaves the reserved proteins into a transform or predict step</em>. The split itself does
      reach the reserve — that is how the reserve is created — and it goes no further.
    </figcaption>
    <div className="imb-panel">
      <svg viewBox={`0 0 ${width} ${height}`} role="img"
        aria-label={`A role diagram. ${provenance.sourceRows} source rows become ${provenance.studyRows} distinct proteins after exact-duplicate repair, then split into four roles. Fitting, ${roles.fitting.records} proteins with ${roles.fitting.positives} positives, runs fit scaler, then resample, then fit model. Tuning, ${roles.tuning.records} proteins with ${roles.tuning.positives} positives, runs transform with the fitted scaler, then predict, then choose a threshold. Inspection, ${roles.inspection.records} proteins with ${roles.inspection.positives} positives, runs transform, then predict with the locked threshold, then report. Reserved, ${roles.reserve.records} proteins with ${roles.reserve.positives} positives, receives the split like every other role and has no connector onward into any transform or predict step.`}>
        <title>Which role may influence which step</title>
        <rect className="imb-lane" x={6} y={8} width={328} height={44} rx="3" />
        <text className="imb-halo imb-small" x={14} y={26}>
          {provenance.sourceRows.toLocaleString('en-US')} source rows → {provenance.studyRows.toLocaleString('en-US')} distinct proteins
        </text>
        <text className="imb-halo imb-small imb-muted" x={14} y={42}>
          identity and group constraints settled first
        </text>
        {lanes.map(lane => <g key={lane.key}>
          {/* Docked at the top edge of each role box rather than run into its
              middle: the split does create every role, including the reserve,
              and the drawing should show it arriving rather than passing through. */}
          <line className="imb-flow" x1={26} x2={26} y1={52} y2={lane.y} />
          <rect className={lane.className} x={26} y={lane.y} width={104} height={40} rx="3" />
          <text className="imb-halo imb-small imb-strong" x={32} y={lane.y + 17}>{lane.label}</text>
          <text className="imb-halo imb-small" x={32} y={lane.y + 32}>
            {lane.role.records} / {lane.role.positives}+
          </text>
        </g>)}
        {/* The fitting branch: three steps, in order. */}
        {['fit scaler', 'resample', 'fit model'].map((label, index) => (
          <g key={label}>
            {/* Drawn along the top edge of the box row, not through its middle:
                a step label is wider than its own box at phone type sizes, so a
                mid-height connector lands inside the previous label. */}
            <line className="imb-flow" x1={130 + index * 68} x2={140 + index * 68} y1={83} y2={83} />
            <rect className="imb-lane is-fitting" x={140 + index * 68} y={78} width={62} height={32} rx="3" />
            <text className="imb-halo imb-small" x={144 + index * 68} y={98}>{label}</text>
          </g>
        ))}
        {/* The assessment branches: transform and predict only. */}
        {[
          { y: 160, steps: ['transform', 'predict', 'choose t'] },
          { y: 216, steps: ['transform', 'predict', 'report'] },
        ].map(branch => <g key={branch.y}>
          {branch.steps.map((label, index) => <g key={label}>
            <line className="imb-flow" x1={130 + index * 68} x2={140 + index * 68} y1={branch.y - 11} y2={branch.y - 11} />
            <rect className="imb-lane" x={140 + index * 68} y={branch.y - 16} width={62} height={32} rx="3" />
            <text className="imb-halo imb-small" x={144 + index * 68} y={branch.y + 4}>{label}</text>
          </g>)}
        </g>)}
        {/* The saved scaler and the fitted model travel forward only. Both
            connectors run down the left gutter at x = 134, clear of every step
            box and of the caption beside them: a connector routed across the
            rows travelled the length of its own label. */}
        <path className="imb-flow" d="M 134 110 C 134 124, 134 130, 134 144" />
        <path className="imb-flow" d="M 134 176 C 134 188, 134 192, 134 200" />
        <text className="imb-halo imb-small imb-muted" x={150} y={128}>saved scaler + model</text>
        <text className="imb-halo imb-small imb-muted" x={150} y={192}>same scaler + model</text>
        <text className="imb-halo imb-small imb-muted" x={136} y={276}>nothing leaves the reserve</text>
      </svg>
    </div>
    <Table caption="What each role is allowed to influence in this study"
      headings={['role', 'distinct proteins', 'ME2 positives', 'what it may influence']}
      rows={Object.entries(roles).map(([name, entry]) => [
        name, String(entry.records), String(entry.positives), entry.influences,
      ])}
      footnote={`The four roles are disjoint and together cover all ${provenance.studyRows.toLocaleString('en-US')} retained proteins and all ${provenance.positives} positives. The scaler is fitted on the ${roles.fitting.records} original fitting rows for every method; resampling changes only the classifier's training matrix afterwards.`} />
    <p>
      An <code>imblearn.pipeline.Pipeline</code> is a convenient way to express this, and a correct manual fold loop is
      another. Neither is the only valid expression, and neither can repair duplicated entities split across
      partitions. The optional three-fold demonstration later uses the {roles.fitting.records} fitting proteins alone
      and is a separate protocol, not the source of the outcome table.
    </p>
  </figure>;
}

/* ================================================================ figure 5 */

export function OutcomesFigure() {
  const rows = methods.map(method => {
    const atHalf = countsAt(inspectionRecords.labels, method.inspectionScores, 0.5);
    const tuned = countsAt(inspectionRecords.labels, method.inspectionScores, method.chosenThreshold);
    return {
      method,
      atHalf,
      tuned,
      halfCost: costOf(atHalf, study.costFalsePositive, study.costFalseNegative),
      tunedCost: costOf(tuned, study.costFalsePositive, study.costFalseNegative),
      queue: rankedQueue(inspectionRecords.sourceIds, inspectionRecords.labels, method.inspectionScores, 10),
    };
  });
  const worst = Math.max(study.baselineCost, ...rows.map(row => Math.max(row.halfCost, row.tunedCost)));
  const barWidth = 340;
  const barLeft = 96;
  // A fixed right-hand column holds every value label, so a long bar can never
  // push its own label off the viewBox and no label has to sit on top of a bar.
  const labelColumn = 92;
  const scale = value => (barWidth - barLeft - labelColumn) * value / worst;
  const bestAp = Math.max(...rows.map(row => row.method.averagePrecision));
  return <figure className="imb-figure">
    <figcaption>
      <strong>Figure 5 — Each improvement has a named objective.</strong> Observed counts on the
      {' '}{roles.inspection.records} inspection proteins, first at the default {0.5} and then at each procedure's
      separately selected tuning threshold. The cost axis is in hypothetical error-cost units of
      {' '}{study.costFalsePositive} per false alarm and {study.costFalseNegative} per missed positive; it is not a
      percentage. The ranking panel below is a different question with a different unit, and the two are never put on
      one axis.
    </figcaption>
    <div className="imb-panel">
      <h4>Realised cost at 0.5 and at the selected threshold</h4>
      <svg viewBox={`0 0 ${barWidth} ${rows.length * 46 + 76}`} role="img"
        aria-label={rows.map(row => `${row.method.label}: at 0.5, TP ${row.atHalf.tp}, FP ${row.atHalf.fp}, FN ${row.atHalf.fn}, TN ${row.atHalf.tn}, cost ${row.halfCost}. At its selected threshold ${round(row.method.chosenThreshold, 6)}, TP ${row.tuned.tp}, FP ${row.tuned.fp}, FN ${row.tuned.fn}, TN ${row.tuned.tn}, cost ${row.tunedCost}.`).join(' ') + ` The always-negative baseline costs ${study.baselineCost}.`}>
        <title>Paired cost bars, starting at zero</title>
        {rows.map((row, index) => {
          const top = index * 46 + 10;
          return <g key={row.method.name}>
            <text className="imb-small imb-strong" x={2} y={top + 12}>{row.method.label}</text>
            {[
              { key: 'half', y: top + 16, cost: row.halfCost, counts: row.atHalf, label: 'at 0.5' },
              { key: 'tuned', y: top + 30, cost: row.tunedCost, counts: row.tuned, label: 'tuned' },
            ].map(bar => <g key={bar.key}>
              <text className="imb-small imb-muted" x={2} y={bar.y + 9}>{bar.label}</text>
              <rect className="imb-bar is-cost-fp" x={barLeft} y={bar.y} width={Math.max(scale(bar.counts.fp * study.costFalsePositive), 0)} height={11} />
              <rect className="imb-bar is-cost-fn" x={barLeft + scale(bar.counts.fp * study.costFalsePositive)} y={bar.y}
                width={Math.max(scale(bar.counts.fn * study.costFalseNegative), 0)} height={11} />
              <text className="imb-halo imb-small" x={barWidth - labelColumn + 4} y={bar.y + 9}>
                {bar.cost} = {bar.counts.fp} + {study.costFalseNegative}×{bar.counts.fn}
              </text>
            </g>)}
          </g>;
        })}
        {/* Both vertical rules start below the first group heading: begun at the
            very top they ran through the procedure names, which overhang the
            zero line by design so the names can be read at full width. */}
        <line className="imb-axis" x1={barLeft} x2={barLeft} y1={24} y2={rows.length * 46 + 14} />
        <line className="imb-axis" x1={barLeft + scale(study.baselineCost)} x2={barLeft + scale(study.baselineCost)}
          y1={24} y2={rows.length * 46 + 14} strokeDasharray="4 3" />
        <text className="imb-small imb-muted" x={2} y={rows.length * 46 + 28}>0 — every bar starts at zero.</text>
        <text className="imb-small imb-muted" x={2} y={rows.length * 46 + 40}>
          Dashed line: the always-negative baseline, cost {study.baselineCost}.
        </text>
        <text className="imb-small imb-muted" x={2} y={rows.length * 46 + 52}>Left segment: false alarms.</text>
        <text className="imb-small imb-muted" x={2} y={rows.length * 46 + 64}>
          Right segment: missed positives, each worth {study.costFalseNegative}.
        </text>
      </svg>
    </div>
    <Table caption="The observed inspection outcome for each declared procedure"
      headings={['procedure', 'selected threshold', 'TP / FP / FN / TN at 0.5', 'TP / FP / FN / TN tuned', 'realised cost', 'average precision', 'positives in top ten']}
      rows={rows.map(row => [
        row.method.label,
        round(row.method.chosenThreshold, 6),
        `${row.atHalf.tp} / ${row.atHalf.fp} / ${row.atHalf.fn} / ${row.atHalf.tn}`,
        `${row.tuned.tp} / ${row.tuned.fp} / ${row.tuned.fn} / ${row.tuned.tn}`,
        String(row.tunedCost),
        fixed(row.method.averagePrecision, 6),
        String(row.queue.positives),
      ])}
      rowClass={index => (rows[index].tunedCost === Math.min(...rows.map(row => row.tunedCost)) ? 'is-chosen' : undefined)}
      footnote={`Every row totals ${roles.inspection.records} with ${roles.inspection.positives} positives. The always-negative baseline is ${round(100 * study.baselineAccuracy, 1)}% accurate, has recall 0, cost ${study.baselineCost}, and undefined precision because it selects nothing.`} />
    <div className="imb-panel">
      <h4>A different question: the ranking, and the first ten proteins reviewed</h4>
      <Table caption="Average precision and the actual top-ten source identities, in saved rank order"
        headings={['procedure', 'average precision', 'ROC-AUC', 'Brier', 'positives in top ten', 'top-ten source IDs in rank order']}
        rows={rows.map(row => [
          row.method.label,
          fixed(row.method.averagePrecision, 6),
          fixed(row.method.rocAuc, 6),
          fixed(row.method.brierScore, 6),
          String(row.queue.positives),
          row.queue.top.map(entry => `${entry.id}${entry.label === 1 ? '•' : ''}`).join(' '),
        ])}
        rowClass={index => (rows[index].method.averagePrecision === bestAp ? 'is-chosen' : undefined)}
        footnote={'A dot marks an actual ME2 protein. Ranks come from the saved scores alone; no tie is reordered by a label. Brier is a mean squared probability error, which is not the same thing as a calibration measurement.'} />
      <p>
        SMOTE reaches the lowest realised cost, {Math.min(...rows.map(row => row.tunedCost))}. Random oversampling
        reaches the highest average precision, {round(bestAp, 6)}. The original fit, random oversampling and random
        undersampling each find two positives in their top ten. Three questions, three different winners — and no
        smooth precision–recall curve is drawn from these five summaries, because five summaries do not contain one.
      </p>
    </div>
  </figure>;
}

/* ================================================================ figure 6 */

const focal = focalMass(fixtures.focal);

export function LossMassFigure() {
  const width = 340;
  const height = 160;
  const left = 96;
  const barTop = 28;
  // The two totals differ by two orders of magnitude, so each panel is drawn on
  // its own stated scale rather than on one axis that hides the small bars.
  const panel = (rows, total, className, label) => (
    <svg viewBox={`0 0 ${width} ${height}`} role="img"
      aria-label={`${label}. ${rows.map(row => `${row.count} ${row.name} examples at ${round(row.perExample, 6)} each total ${round(row.total, 6)}`).join('; ')}. The two totals together are ${round(total, 6)}.`}>
      <title>{label}</title>
      {rows.map((row, index) => {
        const top = barTop + index * 46;
        const length = Math.max((width - left - 18) * row.total / total, 1);
        // A bar that nearly fills the track leaves no room for a label beside
        // it, so the value moves inside the bar rather than off the viewBox.
        const inside = length > 62;
        return <g key={row.name}>
          <text className="imb-small imb-strong" x={2} y={top + 12}>{row.name}</text>
          <text className="imb-small imb-muted" x={2} y={top + 26}>{row.count.toLocaleString('en-US')} × {round(row.perExample, 4)}</text>
          <rect className={`imb-bar ${className}`} x={left} y={top} width={length} height={18} />
          <text className="imb-halo imb-small" x={inside ? left + length - 5 : left + length + 5} y={top + 14}
            textAnchor={inside ? 'end' : 'start'}>{round(row.total, 4)}</text>
        </g>;
      })}
      <line className="imb-axis" x1={left} x2={left} y1={20} y2={height - 34} />
      <text className="imb-small imb-muted" x={2} y={height - 18}>0 — every bar starts at zero.</text>
      <text className="imb-small imb-muted" x={2} y={height - 6}>Shares of this panel's total {round(total, 4)}.</text>
    </svg>
  );
  const ceRows = focal.rows.map(row => ({
    name: row.name, count: row.count, perExample: row.crossEntropy, total: row.crossEntropyTotal,
  }));
  const focalRows = focal.rows.map(row => ({
    name: row.name, count: row.count, perExample: row.focal, total: row.focalTotal,
  }));
  return <figure className="imb-figure">
    <figcaption>
      <strong>Figure 6 — Loss mass from many easy cases.</strong> A constructed loss calculation on
      {' '}{focal.rows[0].count.toLocaleString('en-US')} easy examples at p<sub>t</sub> = {focal.rows[0].pt} and
      {' '}{focal.rows[1].count} difficult ones at p<sub>t</sub> = {focal.rows[1].pt}, with γ = {focal.gamma} and
      α<sub>t</sub> = {focal.alpha}. The axis says <em>loss</em>, not gradient. The two panels carry their own scales,
      each stated, because one shared axis would make the difficult bar invisible under cross-entropy.
    </figcaption>
    <Table caption="Count times per-example loss equals total loss, under each objective. CE is cross-entropy; the modulator is (1 − p_t) raised to gamma."
      headings={['group', 'count', 'p_t', 'CE each', 'CE total', 'modulator', 'focal each', 'focal total']}
      rows={focal.rows.map(row => [
        row.name, row.count.toLocaleString('en-US'), round(row.pt, 2), round(row.crossEntropy, 6),
        round(row.crossEntropyTotal, 6), round(row.modulator, 4), round(row.focal, 6), round(row.focalTotal, 6),
      ])}
      footnote={`Under cross-entropy the easy group carries ${round(100 * focal.shares.crossEntropy[0], 1)}% of the loss mass; under focal loss it carries ${round(100 * focal.shares.focal[0], 1)}%. The balancing comes from current prediction difficulty, not from a class label.`} />
    <div className="imb-panels is-pair">
      <div className="imb-panel">
        <h4>Cross-entropy, total {round(focal.crossEntropyTotal, 4)}</h4>
        {panel(ceRows, focal.crossEntropyTotal, 'is-ce', 'Loss mass under cross-entropy')}
      </div>
      <div className="imb-panel">
        <h4>Focal loss, total {round(focal.focalTotal, 4)}</h4>
        {panel(focalRows, focal.focalTotal, 'is-focal', 'Loss mass under focal loss')}
      </div>
    </div>
    <p>
      No gradient axis appears here, and that is deliberate. Differentiating the focal objective also differentiates
      the factor (1 − p<sub>t</sub>)<sup>γ</sup>, so the derivative keeps a term that a &ldquo;cross-entropy gradient
      times {round(focal.rows[0].modulator, 2)}&rdquo; claim would drop. At γ = 0 every modulator is exactly 1 and this
      is weighted cross-entropy again.
    </p>
  </figure>;
}
