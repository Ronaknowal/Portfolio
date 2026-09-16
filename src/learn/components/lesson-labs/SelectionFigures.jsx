import { useState } from 'react';
import {
  barScale, duplicateColumns, explainPolynomial, explainTreeInstance, impurityImportance,
  maskMatrix, sizePoints, treeDecision, waterfall, xorWorld,
} from '../../data/selection-models';
import {
  candidates, explainedCases, featureLabels, fourFieldModel, majorityBaseline, permutationRecords,
  provenance, selectedModel, split, backgroundIds, backgroundRows,
} from '../../data/selection-data';
import { Folded, Table, TreeDiagram, Waterfall, fixed, round, settled, signed, unsigned } from './SelectionShared.jsx';
import { ScrollRegion } from './SelectionShared.jsx';
import './selection-labs.css';

const SMALL_LABELS = fourFieldModel.labels;
const SHORT_LABELS = ['alcohol', 'malic', 'flavan.', 'proline'];
const TREE = fourFieldModel.tree;

/* ------------------------------------------------------------- §1 · F1 */

/** `short` is the label inside the diagram, kept inside the lane's own width;
 * `title` is the full question, which lives in the text table below it. */
const ROUTES = [
  {
    key: 'information',
    short: 'Information in the data',
    title: 'What does this measurement reveal on its own?',
    stages: ['count the observed pairs', 'entropy difference, in bits'],
    fits: false,
    fixed: 'the observed distribution and the variable definitions',
    starting: 'mutual information or another univariate statistic',
  },
  {
    key: 'subset',
    short: 'Remove it and refit',
    title: 'Could we train a useful model without it?',
    stages: ['choose candidate subsets', 'fit the procedure', 'assess it'],
    fits: true,
    fixed: 'the training and assessment procedure; the model is refitted',
    starting: 'a subset comparison or a removal-and-refit experiment',
  },
  {
    key: 'permutation',
    short: 'Reliance of a fixed model',
    title: 'How much does this fitted model rely on it?',
    stages: ['keep the model fixed', 'shuffle one column with donors', 'score again'],
    fits: false,
    fixed: 'the model, the assessment rows and the performance metric',
    starting: 'permutation importance',
  },
  {
    key: 'attribution',
    short: 'Allocation of one prediction',
    title: 'How is this one prediction allocated?',
    stages: ['one instance', 'a declared reference population', 'allocate the output'],
    fits: false,
    fixed: 'the model, the instance, the output scale and the missing-feature rule',
    starting: 'a Shapley or SHAP attribution',
  },
  {
    key: 'causal',
    short: 'A causal question',
    title: 'Would changing the real quantity change the outcome?',
    stages: ['state a causal problem', 'evidence beyond these tools'],
    fits: false,
    fixed: 'a stated causal problem',
    starting: 'causal assumptions and evidence these diagnostics do not supply',
  },
];

/** F1 — one dataset, different questions. */
export function QuestionRoutesFigure() {
  // Each lane is sized from its own contents: a title line, one line per stage,
  // and a badge line of its own. Nothing shares a line with anything else, which
  // is what keeps a long label from landing on top of its neighbour.
  const laneHeight = route => 18 + route.stages.length * 15 + 16;
  const tops = [];
  let running = 8;
  ROUTES.forEach(route => { tops.push(running); running += laneHeight(route) + 8; });
  const height = running;
  const middle = height / 2;
  return <figure className="fs-figure">
    <figcaption>
      <strong>Figure 1 — One dataset, different questions.</strong> The same measurements enter five different experiments.
      What separates them is which objects are held fixed and which are allowed to change, not which one produces a prettier
      ranking. This is an information-flow diagram; the five routes are not interchangeable bars on one chart.
    </figcaption>
    <div className="fs-svg-frame">
    <ScrollRegion><svg viewBox={`0 0 340 ${height}`} role="img"
      aria-label={`A route map. One card of thirteen measurements and a cultivar label feeds five routes. ${ROUTES.map(route => `${route.title} goes through ${route.stages.join(', then ')}, and ${route.fits ? 'refits a model' : 'fits nothing'}`).join('. ')}. The causal route is drawn separately because it needs assumptions and evidence these diagnostics do not supply.`}>
      <rect className="fs-lane" x="4" y={middle - 30} width="72" height="60" rx="3" />
      <text className="is-small" x="40" y={middle - 12} textAnchor="middle">thirteen</text>
      <text className="is-small" x="40" y={middle + 2} textAnchor="middle">measurements</text>
      <text className="is-small" x="40" y={middle + 16} textAnchor="middle">+ a label</text>
      {ROUTES.map((route, index) => {
        const y = tops[index];
        const box = laneHeight(route);
        const causal = route.key === 'causal';
        return <g key={route.key}>
          <path className={causal ? 'fs-flow is-causal' : 'fs-flow'}
            d={`M78,${middle} C100,${middle} 104,${y + box / 2} 122,${y + box / 2}`} />
          <rect className={causal ? 'fs-lane is-closed' : route.fits ? 'fs-lane is-refit' : 'fs-lane is-fixed'}
            x="124" y={y} width="212" height={box} rx="3" />
          <text className="is-small" x="132" y={y + 13}>{route.short}</text>
          {route.stages.map((stage, position) => (
            <text key={stage} className="is-small" x="138" y={y + 28 + position * 15}
              style={{ fill: causal ? '#d8a9a0' : '#b5bfbb' }}>· {stage}</text>
          ))}
          <text className="is-small" x="330" y={y + box - 5} textAnchor="end"
            style={{ fill: route.fits ? '#e7b94a' : causal ? '#cf9191' : '#8eb9a5' }}>
            {causal ? 'outside these tools' : route.fits ? 'refits a model' : 'fits nothing'}
          </text>
        </g>;
      })}
    </svg></ScrollRegion>
    </div>
    <p className="fs-caption">
      Gold outline: the route refits a model. Green outline: the route reads a fitted model or the data without fitting anything.
      Red dashed: the causal question, drawn apart because none of the other four answers it. The labels in the diagram are short
      names; each route&rsquo;s full question is the first column of the table below. A measurement badged “selected” by any of
      these routes has not been shown to be causally necessary.
    </p>
    <Table caption="The same five routes as text: what each question holds fixed, and where it starts" wrap
      headings={['question', 'what stays fixed', 'suitable starting point', 'does it refit?']}
      rows={ROUTES.map(route => [route.title, route.fixed, route.starting, route.fits ? 'yes, the model is refitted' : 'no'])} />
    <p>
      The practical question this figure answers is what must change to answer a request to stop paying for an assay. That is the
      subset route: remove the measurement and refit, then assess the reduced recipe. A permutation score on the model you already
      have answers a different question, and a large association can come from a measurement taken after the outcome was known.
      Establish availability and the unit of prediction before ranking anything.
    </p>
  </figure>;
}

/* ------------------------------------------------------------- §2 · F2 */

/** F2 — the XOR square and its projections. */
export function XorSquareFigure() {
  const world = xorWorld();
  const duplicate = duplicateColumns();
  const size = 150;
  const place = value => 46 + value * size;
  // The plot starts below the legend line, so a corner marker at B = 1 cannot
  // collide with the legend text above it.
  const lift = value => 40 + (1 - value) * size;
  return <figure className="fs-figure">
    <figcaption>
      <strong>Figure 2 — The XOR square and its projections.</strong> Four equally likely states, coloured and labelled by their
      target. Each one-dimensional projection combines two states with opposite targets and leaves a half/half split; the joint
      square separates all four. Nothing here was trained: these are exact information quantities over a declared world.
    </figcaption>
    <div className="fs-panels is-pair">
      <div className="fs-panel">
        <h4>The square, and what each axis keeps</h4>
        <ScrollRegion><svg viewBox="0 0 250 246" role="img"
          aria-label={`Four points at A and B equal to 0,0 target 0; 0,1 target 1; 1,0 target 1; 1,1 target 0. Projecting onto A, each value of A carries one target 0 and one target 1, so the information is ${world.projections[0].information} bits. The same holds for B. The joint pair carries ${world.jointInformation} bit.`}>
          <line className="fs-axis" x1={place(0)} x2={place(1)} y1={lift(0)} y2={lift(0)} />
          <line className="fs-axis" x1={place(0)} x2={place(0)} y1={lift(1)} y2={lift(0)} />
          {world.points.map(point => (
            <g key={point.index}>
              {point.target === 0
                ? <circle className="fs-target-0" cx={place(point.a)} cy={lift(point.b)} r="9" />
                : <rect className="fs-target-1" x={place(point.a) - 8} y={lift(point.b) - 8} width="16" height="16" />}
              <text className="is-small" x={place(point.a)} y={lift(point.b) + 4} textAnchor="middle"
                style={{ fill: '#10150f' }}>{point.target}</text>
            </g>
          ))}
          {[0, 1].map(value => <g key={`a${value}`}>
            <text className="is-small" x={place(value)} y={lift(0) + 18} textAnchor="middle">A = {value}</text>
            <text className="is-small" x={place(0) - 14} y={lift(value) + 4} textAnchor="end">B = {value}</text>
          </g>)}
          {world.projections[0].values.map(entry => (
            <g key={`pa${entry.value}`}>
              <text className="is-small" x={place(entry.value)} y={lift(0) + 34} textAnchor="middle">
                {entry.counts[0]} of target 0
              </text>
              <text className="is-small" x={place(entry.value)} y={lift(0) + 46} textAnchor="middle">
                {entry.counts[1]} of target 1
              </text>
            </g>
          ))}
          <text className="is-small" x="6" y="14">circle = target 0 · square = target 1</text>
        </svg></ScrollRegion>
        <p>
          Projecting onto A leaves one of each target at every value of A, and the same holds for B. The counts under each
          axis value are the actual projected states, not a fitted decision boundary: nothing here was trained.
        </p>
      </div>
      <div className="fs-panel">
        <h4>Three exact quantities</h4>
        <Table caption="Computed from the declared four-state distribution, in bits"
          headings={['quantity', 'bits']}
          rows={[
            ['I(A; Y)', round(world.projections[0].information, 6)],
            ['I(B; Y)', round(world.projections[1].information, 6)],
            ['I((A, B); Y)', round(world.jointInformation, 6)],
          ]} />
        <p>
          A ranking that discarded every zero-information individual input would discard both essential parts of this mechanism.
          That is a limitation of the <em>question</em> a univariate filter asks, not a proof that every criterion computed
          without a model ignores interactions: a filter is free to evaluate joint or conditional information.
        </p>
      </div>
    </div>
    <div className="fs-panel">
      <h4>The opposite problem: two exact copies</h4>
      <Table caption="Two identical measurements each reveal the same single bit"
        headings={['quantity', 'bits']}
        rows={[
          ['I(A; Y) for the first copy', round(duplicate.first, 6)],
          ['I(B; Y) for the second copy', round(duplicate.second, 6)],
          ['their sum, if you added the two rankings', round(duplicate.sum, 6)],
          ['I((A, B); Y), what the pair actually reveals', round(duplicate.joint, 6)],
        ]}
        rowClass={index => (index === 2 ? 'is-zero' : undefined)} />
      <p>
        Summing individual scores double counts the shared bit. This table sits beside the XOR square on purpose, and only after
        it: “correlated” and “jointly useful” are different questions, and neither one is settled by the other. Slightly
        correlated measurements are subtler still — they can carry complementary signal or independent measurement noise, so a
        correlation threshold alone does not prove one is safe to discard.
      </p>
    </div>
  </figure>;
}

/* ------------------------------------------------------------- §5 · F3 */

/** F3 — two arrival orders, one allocation. */
export function ArrivalOrderFigure() {
  const [step, setStep] = useState(2);
  const explanation = explainPolynomial({ instance: [2, 3], background: [[0, 0]], gamma: 1 });
  const chart = waterfall(explanation.baseline, explanation.phi);
  const names = ['A', 'B'];
  return <figure className="fs-figure">
    <figcaption>
      <strong>Figure 3 — Two arrival orders, one allocation.</strong> f(a, b) = a + b + ab, the instance (2, 3) and the declared
      reference (0, 0). Each order is walked with its actual intermediate input and output; averaging the matching increments
      gives the allocation. Nothing here implies that 5 is the physical effect of manipulating A.
    </figcaption>
    <div className="fs-controls">
      <label className="fs-field"><span>Show the paths up to</span>
        <select value={step} onChange={event => setStep(Number(event.target.value))}>
          <option value={0}>the empty coalition only</option>
          <option value={1}>the first arrival</option>
          <option value={2}>both arrivals</option>
        </select>
      </label>
    </div>
    <p className="fs-caption">
      The step control is optional: every value is in the table below at all times, and no animation is needed to read the result.
    </p>
    <div className="fs-panels is-pair">
      {explanation.paths.paths.map(path => (
        <div className="fs-panel" key={path.order.join('')}>
          <h4>{path.order.map(player => names[player]).join(' arrives, then ')} arrives</h4>
          <ScrollRegion><svg viewBox="0 0 300 130" role="img"
            aria-label={`${path.order.map(player => names[player]).join(' then ')}: starting from the reference input with value ${round(explanation.values[0], 6)}, ${path.steps.map(stepEntry => `adding ${names[stepEntry.player]} moves the evaluated input to value ${round(stepEntry.to, 6)}, an increment of ${round(stepEntry.increment, 6)}`).join('; ')}.`}>
            {[{ from: null, to: path.steps[0].before }, ...path.steps.map(entry => ({ from: entry.before, to: entry.after, entry }))]
              .map((node, index) => {
                const x = 14 + index * 96;
                const visible = index <= step;
                const inputs = explanation.masks[node.to].rows[0];
                return <g key={index} opacity={visible ? 1 : 0.25}>
                  {index > 0 && <path className="fs-flow" d={`M${x - 14},44 L${x - 2},44`} />}
                  <rect className={`fs-lane${index === 2 ? ' is-refit' : ''}`} x={x} y="24" width="86" height="42" rx="3" />
                  <text className="is-small" x={x + 43} y="40" textAnchor="middle">({inputs.map(value => round(value, 4)).join(', ')})</text>
                  <text className="is-small" x={x + 43} y="56" textAnchor="middle">v = {round(explanation.values[node.to], 4)}</text>
                  {node.entry && <text className="is-small" x={x + 43} y="86" textAnchor="middle"
                    style={{ fill: '#8eb9a5' }}>
                    {names[node.entry.player]} adds {signed(node.entry.increment, 4)}
                  </text>}
                </g>;
              })}
            <text className="is-small" x="14" y="16">evaluated input, then v(S)</text>
            <text className="is-small" x="14" y="118">both orders end at v = {round(explanation.prediction, 4)}</text>
          </svg></ScrollRegion>
          <p>
            Each box is the input actually evaluated at that point and the coalition value it produces. The increment under an
            arrow is what the arriving feature added.
          </p>
        </div>
      ))}
    </div>
    <div className="fs-panels is-pair">
      <div className="fs-panel">
        <h4>The complete coalition table</h4>
        <Table caption="Bitmask order: none, A, B, then A and B"
          headings={['retained', 'evaluated input', 'v(S)']}
          rows={explanation.masks.map(entry => [
            entry.mask === 0 ? 'none' : entry.mask === 1 ? 'A' : entry.mask === 2 ? 'B' : 'A and B',
            `(${entry.rows[0].map(value => round(value, 4)).join(', ')})`,
            round(entry.value, 6),
          ])} />
      </div>
      <div className="fs-panel">
        <h4>Averaging the two orders</h4>
        <Waterfall model={chart} names={['φ_A', 'φ_B']} baselineLabel="baseline v(∅)" totalLabel="reconstructed"
          unit="output units" digits={6}
          describe={`A waterfall starting at ${round(chart.baseline, 6)}, adding ${round(chart.segments[0].value, 6)} for A and ${round(chart.segments[1].value, 6)} for B, ending at ${round(chart.reconstruction, 6)}.`} />
        <p>
          A receives (2 + 8) ÷ 2 = {round(explanation.phi[0], 6)} and B receives (3 + 9) ÷ 2 = {round(explanation.phi[1], 6)}.
          Each takes its standalone contribution plus half of the interaction, which is six here. They sum to
          {' '}{round(settled(chart.reconstruction), 6)}, the difference from the reference. The model is still nonlinear; the
          additive accounting belongs to this instance and this reference.
        </p>
      </div>
    </div>
  </figure>;
}

/* ------------------------------------------------------------- §6 · F4 */

/** F4 — selection changes a set, not just a score. */
export function SelectionProcedureFigure() {
  const [inspected, setInspected] = useState(6);
  const points = sizePoints(candidates);
  const row = candidates.find(entry => entry.k === inspected);
  const masks = maskMatrix(row.folds, featureLabels.length);
  const importance = impurityImportance(TREE, 4);
  const permutationScale = barScale(permutationRecords.flatMap(entry => [entry.mean, ...entry.drops]));
  const impurityScale = barScale(importance.normalized);
  const plotLow = 0.78;
  const plotHigh = 0.87;
  const place = k => 46 + 250 * (k - 3) / 10;
  const lift = value => 126 - 100 * (value - plotLow) / (plotHigh - plotLow);
  return <figure className="fs-figure">
    <figcaption>
      <strong>Figure 4 — Selection changes a set, not just a score.</strong> The recorded study: three retained sizes, three
      inner folds, eleven fits in total. The size is one hyperparameter; each fitted selector still learns its own subset from
      its own rows, and those subsets are not identical.
    </figcaption>

    <div className="fs-panel">
      <h4>Where every row went, and which rows were never scored</h4>
      <ScrollRegion><svg viewBox="0 0 340 180" role="img"
        aria-label={`A boundary diagram. All ${split.total} source rows split into ${split.development} development rows and ${split.reserved} reserved rows with seed ${split.splitSeed}. The development rows split into ${split.fitting} fitting rows and ${split.inspection} inspection rows with seed ${split.innerSeed}. Three stratified folds with seed ${split.foldSeed} sit inside the fitting rows. The reserved rows receive no prediction and no score in this lesson.`}>
        <rect className="fs-lane" x="4" y="8" width="94" height="30" rx="3" />
        <text className="is-small" x="51" y="21" textAnchor="middle">all {split.total} rows</text>
        <text className="is-small" x="51" y="33" textAnchor="middle">source order kept</text>
        <path className="fs-flow" d="M98,23 C114,23 118,60 134,60" />
        <path className="fs-flow is-causal" d="M98,23 C114,23 118,150 134,150" />
        <rect className="fs-lane" x="136" y="44" width="96" height="30" rx="3" />
        <text className="is-small" x="184" y="57" textAnchor="middle">development</text>
        <text className="is-small" x="184" y="69" textAnchor="middle">{split.development} rows</text>
        <rect className="fs-lane is-closed" x="136" y="136" width="200" height="30" rx="3" />
        <text className="is-small" x="144" y="149" style={{ fill: '#d8a9a0' }}>reserved {split.reserved} rows —</text>
        <text className="is-small" x="144" y="161" style={{ fill: '#d8a9a0' }}>never predicted, never scored</text>
        <path className="fs-flow" d="M232,59 C246,59 250,26 264,26" />
        <path className="fs-flow" d="M232,59 C246,59 250,96 264,96" />
        <rect className="fs-lane is-refit" x="266" y="10" width="70" height="32" rx="3" />
        <text className="is-small" x="301" y="24" textAnchor="middle">fitting {split.fitting}</text>
        <text className="is-small" x="301" y="36" textAnchor="middle">3 inner folds</text>
        <rect className="fs-lane is-fixed" x="266" y="80" width="70" height="32" rx="3" />
        <text className="is-small" x="301" y="94" textAnchor="middle">inspection</text>
        <text className="is-small" x="301" y="106" textAnchor="middle">{split.inspection} rows</text>
        <text className="is-small" x="4" y="96">seeds {split.splitSeed}, {split.innerSeed}, {split.foldSeed}</text>
        <text className="is-small" x="4" y="110">stratified by cultivar</text>
      </svg></ScrollRegion>
      <p>
        The three inner folds occur entirely inside the {split.fitting} fitting rows: the selector and the tree are fitted
        separately in each of them. The {split.inspection} inspection rows did not choose the retained size, and the
        {' '}{split.reserved} reserved rows were never predicted. A row split is not an outer assessment of a new region,
        vineyard, laboratory or measurement process, and no grouping metadata here would support one.
      </p>
    </div>

    <div className="fs-panels is-pair">
      <div className="fs-panel">
        <h4>The three sizes that were actually evaluated</h4>
        <ScrollRegion><svg viewBox="0 0 340 168" role="img"
          aria-label={`Mean inner-fold accuracy against the retained size, for the three evaluated sizes only. ${candidates.map(entry => `size ${entry.k} gives ${entry.meanAccuracy}`).join('; ')}. The individual fold accuracies are also plotted. No size between the three was fitted.`}>
          <line className="fs-axis" x1="46" x2="308" y1="126" y2="126" />
          <line className="fs-axis" x1="46" x2="46" y1="14" y2="126" />
          {[0.78, 0.82, 0.86].map(value => <g key={value}>
            <line className="fs-grid" x1="46" x2="308" y1={lift(value)} y2={lift(value)} />
            <text className="is-small" x="42" y={lift(value) + 4} textAnchor="end">{value.toFixed(2)}</text>
          </g>)}
          {candidates.map(entry => <text key={entry.k} className="is-small" x={place(entry.k)} y="144" textAnchor="middle">k = {entry.k}</text>)}
          <polyline className="fs-curve is-faint"
            points={candidates.map(entry => `${place(entry.k)},${lift(entry.meanAccuracy)}`).join(' ')} />
          {candidates.flatMap(entry => entry.folds.map(fold => (
            <circle key={`${entry.k}-${fold.fold}`} cx={place(entry.k)} cy={lift(fold.accuracy)} r="2.4" fill="#5c6b64" />
          )))}
          {candidates.map(entry => (
            <g key={`m${entry.k}`}>
              <circle className={entry.k === selectedModel.k ? 'fs-mark' : 'fs-mark is-hollow'}
                cx={place(entry.k)} cy={lift(entry.meanAccuracy)} r="5" />
              {/* The first point sits against the axis, so its value label is
                  anchored to the right of it rather than centred on the ticks. */}
              <text className="is-small" x={place(entry.k) + (entry.k === points.lowSize ? 10 : 0)}
                y={lift(entry.meanAccuracy) - 10}
                textAnchor={entry.k === points.lowSize ? 'start' : 'middle'}>
                {round(entry.meanAccuracy, 4)}
              </text>
            </g>
          ))}
        </svg></ScrollRegion>
        <p>
          Vertical axis: mean inner-fold accuracy. Only k = {points.sizes.join(', ')} were fitted.
        </p>
        <p>
          The dashed line joins the three evaluated points and nothing else: no intermediate size was fitted, so there is no
          curve to read between them, no smoothing and no interval. Grey dots are the individual fold accuracies. The rule
          selects k = {points.chosen}, preferring the first and smaller size on an exact tie. The differences here are small and
          this experiment does not make them decisive.
        </p>
      </div>
      <div className="fs-panel">
        <h4>The fold results behind those means</h4>
        <Table caption="Correct counts in each inner validation fold, and the mean they average to" wrap
          headings={['retained size', 'fold 1', 'fold 2', 'fold 3', 'mean accuracy']}
          rows={candidates.map(entry => [
            entry.k, ...entry.folds.map(fold => `${fold.correct}/${fold.total}`), fixed(entry.meanAccuracy, 6),
          ])}
          rowClass={index => (candidates[index].k === selectedModel.k ? 'is-selected' : undefined)} />
      </div>
    </div>

    <div className="fs-panel">
      <h4>The retained sets, fold by fold</h4>
      <div className="fs-controls">
        <label className="fs-field"><span>Retained size to inspect</span>
          <select value={inspected} onChange={event => setInspected(Number(event.target.value))}>
            {candidates.map(entry => <option key={entry.k} value={entry.k}>k = {entry.k}</option>)}
          </select>
        </label>
      </div>
      <Table caption={`Which of the thirteen fields each fold's own selector kept at k = ${inspected}, beside the set the final refit kept at k = ${selectedModel.k}`}
        headings={['field', ...masks.map(entry => `fold ${entry.fold + 1}`), `final refit (k = ${selectedModel.k})`]}
        rows={featureLabels.map((label, column) => [
          label,
          ...masks.map(entry => (entry.membership[column] ? 'kept' : '—')),
          selectedModel.columns.includes(column) ? 'kept' : '—',
        ])}
        rowClass={column => (selectedModel.columns.includes(column) ? 'is-selected' : undefined)} />
      <p>
        At k = {inspected} the three folds kept {masks.map(entry => `{${entry.selected.map(column => featureLabels[column]).join(', ')}}`).join(', ')}.
        The final refit on all {split.fitting} fitting rows kept {selectedModel.labels.join(', ')}. The column count is one
        hyperparameter; the identities are learned separately by every selector, which is why subset stability and prediction
        quality are separate questions.
      </p>
    </div>

    <div className="fs-panel">
      <h4>What each model got right on the inspection rows</h4>
      <Table caption="Three separately declared predictors on the same 38 inspection rows. None of them was chosen by this score." wrap
        headings={['model', 'inputs', 'correct', 'accuracy']}
        rows={[
          [`MI-selected tree, k = ${selectedModel.k}`, selectedModel.labels.join(', '),
            `${selectedModel.correct}/${selectedModel.total}`, fixed(selectedModel.accuracy, 6)],
          ['predeclared four-field tree', fourFieldModel.labels.join(', '),
            `${fourFieldModel.correct}/${fourFieldModel.total}`, fixed(fourFieldModel.accuracy, 6)],
          [`training-majority baseline (cultivar ${majorityBaseline.class})`, 'none',
            `${majorityBaseline.correct}/${majorityBaseline.total}`, fixed(majorityBaseline.accuracy, 6)],
        ]} />
      <p>
        The selected model and the separately declared four-field model are both correct on {fourFieldModel.correct} of
        {' '}{fourFieldModel.total}. That demonstrates the protocol and a possible measurement reduction. It does not prove these
        are the best six assays, that the two models are equivalent, or that the small cross-validation difference is decisive.
      </p>
    </div>

    <div className="fs-panels is-pair">
      <div className="fs-panel">
        <h4>Permutation: mean accuracy decrease</h4>
        <ScrollRegion><svg viewBox="0 0 300 172" role="img"
          aria-label={`Accuracy decrease for the four fields of the predeclared tree, with all twenty donor repeats shown as dots. ${permutationRecords.map(entry => `${entry.label} has mean ${round(entry.mean, 6)} and population standard deviation ${round(entry.sd, 6)}`).join('; ')}.`}>
          <line className="fs-axis" x1="86" x2="292" y1="150" y2="150" />
          {permutationRecords.map((entry, index) => {
            const y = 20 + index * 32;
            const width = 200 * (entry.mean / Math.max(permutationScale.extent, 1e-9));
            return <g key={entry.name}>
              <text className="is-small" x="82" y={y + 14} textAnchor="end">{entry.label}</text>
              <rect className="fs-lane" x="86" y={y} width="206" height="20" rx="2" />
              {entry.mean === 0
                ? <text className="is-small" x="92" y={y + 14} style={{ fill: '#da9c86' }}>exactly 0</text>
                : <rect x="87" y={y + 2} width={Math.max(width, 1).toFixed(1)} height="16" fill="#8eb9a5" />}
              {entry.drops.map((drop, position) => (
                <circle key={position} cx={87 + 200 * (drop / Math.max(permutationScale.extent, 1e-9))}
                  cy={y + 10} r="1.8" fill="#0b0f10" fillOpacity="0.85" />
              ))}
            </g>;
          })}
          <text className="is-small" x="86" y="164">accuracy decrease →</text>
        </svg></ScrollRegion>
        <p>Each dot is one of the twenty donor orderings; the bar is their mean.</p>
        <Table caption="Exact means and population standard deviations over the twenty donor orderings"
          headings={['field', 'mean accuracy decrease', 'SD across 20 repeats']}
          rows={permutationRecords.map(entry => [entry.label, fixed(entry.mean, 6), fixed(entry.sd, 6)])} />
        <p>
          Accuracy decrease is a proportion: {fixed(permutationRecords[0].mean, 6)} is about
          {' '}{(100 * permutationRecords[0].mean).toFixed(1)} percentage points under this averaging scheme. The spread describes
          donor randomization at a fixed model and fixed assessment rows. It is not uncertainty from a new training sample, a
          different model, other assessment cases or a changed population.
        </p>
      </div>
      <div className="fs-panel">
        <h4>Impurity: a different quantity, on its own axis</h4>
        <ScrollRegion><svg viewBox="0 0 300 140" role="img"
          aria-label={`Normalized training-impurity decrease for the same four fields: ${SMALL_LABELS.map((label, index) => `${label} ${round(importance.normalized[index], 6)}`).join('; ')}. This axis is not the accuracy-decrease axis beside it.`}>
          <line className="fs-axis" x1="86" x2="292" y1="122" y2="122" />
          {SMALL_LABELS.map((label, index) => {
            const y = 14 + index * 26;
            const width = 200 * (importance.normalized[index] / Math.max(impurityScale.extent, 1e-9));
            return <g key={label}>
              <text className="is-small" x="82" y={y + 13} textAnchor="end">{label}</text>
              <rect className="fs-lane" x="86" y={y} width="206" height="18" rx="2" />
              {importance.normalized[index] === 0
                ? <text className="is-small" x="92" y={y + 13} style={{ fill: '#da9c86' }}>exactly 0</text>
                : <rect x="87" y={y + 2} width={Math.max(width, 1).toFixed(1)} height="14" fill="#91aecf" />}
            </g>;
          })}
          <text className="is-small" x="86" y="136">impurity decrease →</text>
        </svg></ScrollRegion>
        <p className="fs-caption">
          Normalized training-impurity decrease. This axis is <strong>not</strong> the accuracy-decrease axis beside it.
        </p>
        <Table caption="Recomputed from the saved tree's weighted impurity decreases and normalised as the estimator does"
          headings={['field', 'weighted decrease', 'normalized']}
          rows={SMALL_LABELS.map((label, index) => [label, fixed(importance.raw[index], 6), fixed(importance.normalized[index], 6)])} />
        <p>
          The two panels use two axes because they measure two things: one is a change in held-out accuracy when a column is
          disturbed, the other is how this fitted tree partitioned its training criterion. Normalising both to a common-looking
          scale would disguise that. Many candidate splits can favour chance reductions, so a high-cardinality measurement can
          collect excessive training importance.
        </p>
      </div>
    </div>
    <p>
      Malic acid is exactly zero in both panels. The saved tree never splits on it, so changing that coordinate cannot change
      its prediction function. That is a verified property of this fitted tree, and not a statement that malic acid has no
      association with cultivar or that no other model could use it.
    </p>
    <Folded summary="All eighty recorded accuracy decreases, twenty donor orderings per field">
      <Table caption="Each donor ordering's accuracy decrease, seeds 56 to 75" scroll
        headings={['repeat', ...permutationRecords.map(entry => entry.label)]}
        rows={Array.from({ length: split.donorRepeats }, (_, repeat) => [
          `seed ${split.donorSeedBase + repeat}`,
          ...permutationRecords.map(entry => fixed(entry.drops[repeat], 6)),
        ])} />
    </Folded>
    <p className="fs-caption">
      Source: {provenance.name}, {provenance.creator}, licensed {provenance.license}. {split.total} rows, {split.development}
      {' '}development and {split.reserved} reserved with seed {split.splitSeed}; {split.fitting} fitting and {split.inspection}
      {' '}inspection with seed {split.innerSeed}; three stratified folds with seed {split.foldSeed}.
      {' '}{split.fits.total} fits in total: {split.fits.selectionCv} inner candidates, {split.fits.selectedRefit} selected refit
      and {split.fits.fourFieldRefit} predeclared four-field refit. No reserved row was predicted or scored.
    </p>
  </figure>;
}

/* ------------------------------------------------------------- §6 · F5 */

/** F5 — from a hybrid row to a probability waterfall. */
export function TreeExplanationFigure() {
  const [mask, setMask] = useState(4);
  const [feature, setFeature] = useState(0);
  const explained = explainedCases[0];
  const explanation = explainTreeInstance({
    tree: TREE, instance: explained.input, background: backgroundRows,
    classIndex: fourFieldModel.classIndex, keepHybrids: true,
  });
  const chart = waterfall(explanation.baseline, explanation.phi);
  const tally = explanation.leafTally[mask];
  const retained = SMALL_LABELS.filter((_, index) => (mask & (1 << index)) !== 0);
  const values = explainedCases.map(entry => entry.phi[feature]);
  const extent = Math.max(...explainedCases.flatMap(entry => entry.phi.map(Math.abs)), 1e-9);
  // Leave room for the outermost tick labels inside the viewBox: a point at the
  // extreme sits at 290, and its label is centred there.
  const place = value => 160 + 130 * value / extent;
  return <figure className="fs-figure">
    <figcaption>
      <strong>Figure 5 — From a hybrid row to a probability waterfall.</strong> The saved four-field tree, one coalition's
      hundred actual hybrid rows, the leaves they reach, and the class-1 probability they average to. The output is a leaf
      proportion in probability units: not a log-odds value and not a quality score.
    </figcaption>

    <div className="fs-panels is-pair">
      <div className="fs-panel">
        <h4>The saved tree, with source row {explained.sourceId}'s route</h4>
        <TreeDiagram tree={TREE} shortLabels={SHORT_LABELS} controls={fourFieldModel.limits}
          decision={explanation.decision} classIndex={fourFieldModel.classIndex}
          describe={`The saved nine-node tree from the predeclared four-field fit. Node 0 splits alcohol at ${TREE.threshold[0]}; its left child splits flavanoids at ${TREE.threshold[1]}; the right subtree splits flavanoids at ${TREE.threshold[4]} and then proline at ${TREE.threshold[6]}. Source row ${explained.sourceId} takes the highlighted route to leaf ${explanation.decision.leaf}, whose class-1 probability is ${round(explanation.prediction, 6)}.`} />
        <p>
          Gold boxes are the route this sample takes. A drawn threshold is the largest value the investigation's controls can
          hold that still takes the left branch, so the drawn rule and the applied rule agree at every enterable value; the
          stored thresholds are longer and are given exactly in the table beside this one. Solid edges are taken, dashed are not.
        </p>
      </div>
    </div>

    <div className="fs-panel">
      <h4>Every node, exactly</h4>
      <Table caption="The saved tree at full precision, with the fitted row counts behind each leaf" scroll
        headings={['node', 'split field', 'exact threshold', 'left', 'right', 'fitted rows', 'class distribution 1 / 2 / 3']}
        rows={TREE.childrenLeft.map((left, node) => [
          node,
          left === -1 ? 'leaf' : SMALL_LABELS[TREE.feature[node]],
          left === -1 ? '—' : String(TREE.threshold[node]),
          left === -1 ? '—' : TREE.childrenLeft[node],
          left === -1 ? '—' : TREE.childrenRight[node],
          TREE.samples[node],
          TREE.value[node].map(value => round(value, 4)).join(' / '),
        ])}
        rowClass={node => (explanation.decision.path.some(entry => entry.node === node) || node === explanation.decision.leaf ? 'is-selected' : undefined)} />
    </div>

    <div className="fs-panel">
      <h4>One coalition, its hundred hybrid rows and the average they produce</h4>
      <div className="fs-controls">
        <label className="fs-field"><span>Coalition to inspect</span>
          <select value={mask} onChange={event => setMask(Number(event.target.value))}>
            {explanation.masks.map(entry => (
              <option key={entry.mask} value={entry.mask}>
                retain {entry.mask === 0 ? 'nothing' : SMALL_LABELS.filter((_, index) => (entry.mask & (1 << index)) !== 0).join(' + ')}
              </option>
            ))}
          </select>
        </label>
      </div>
      <Table caption={`The first eight of ${backgroundRows.length} hybrid rows: retained fields take source row ${explained.sourceId}'s values, the rest come from the same donor row`}
        headings={['donor', ...SMALL_LABELS, 'leaf', 'class-1 probability']}
        rows={explanation.masks[mask].rows.slice(0, 8).map((row, index) => {
          const decision = treeDecision(TREE, row);
          return [
            `source row ${backgroundIds[index]}`,
            ...row.map((value, column) => ((mask & (1 << column)) !== 0 ? `${round(value, 4)} ←` : round(value, 4))),
            decision.leaf, round(decision.probabilities[fourFieldModel.classIndex], 4),
          ];
        })} />
      <Table caption={`Where all ${backgroundRows.length} hybrid rows land`}
        headings={['leaf', 'hybrid rows', 'its class-1 probability', 'contribution to v(S)']}
        rows={tally.leaves.map(entry => [
          entry.leaf, entry.count, round(entry.probability, 4),
          round(entry.count * entry.probability / backgroundRows.length, 6),
        ])} />
      <p>
        v(retain {retained.length === 0 ? 'nothing' : retained.join(' + ')}) = <strong>{round(tally.value, 6)}</strong>. The
        alcohol-only coalition has value {round(explanation.values[1], 4)}; flavanoids-only {round(explanation.values[4], 4)};
        proline-only {round(explanation.values[8], 4)}; flavanoids and proline together {round(explanation.values[12], 4)}. Every
        coalition that retains this sample's alcohol has value {round(explanation.values[1], 4)} in the saved tree.
      </p>
    </div>

    <div className="fs-panels is-pair">
      <div className="fs-panel">
        <h4>The waterfall over all 100 fitting rows</h4>
        <Waterfall model={chart} names={SMALL_LABELS} baselineLabel="background mean"
          totalLabel="reconstructed" unit="probability units" digits={6}
          describe={`A waterfall from the background class-1 mean ${round(chart.baseline, 4)} through ${SMALL_LABELS.map((label, index) => `${label} ${round(chart.segments[index].value, 4)}`).join(', ')} to ${round(settled(chart.reconstruction), 4)}.`} />
        <p>
          A negative contribution can be larger in magnitude than the baseline, because positive contributions offset part of it.
          An individual attribution is not a probability and need not lie between zero and one; the reconstructed output does.
          Reconstruction error here is {round(Math.abs(explanation.efficiencyError), 12)}.
        </p>
      </div>
      <div className="fs-panel">
        <h4>The twelve explained cases, one point each</h4>
        <div className="fs-controls">
          <label className="fs-field"><span>Field</span>
            <select value={feature} onChange={event => setFeature(Number(event.target.value))}>
              {SMALL_LABELS.map((label, index) => <option key={label} value={index}>{label}</option>)}
            </select>
          </label>
        </div>
        <ScrollRegion><svg viewBox="0 0 316 132" role="img"
          aria-label={`Twelve points, one per explained inspection row, showing the ${SMALL_LABELS[feature]} attribution in probability units. ${explainedCases.map(entry => `source row ${entry.sourceId} contributes ${round(entry.phi[feature], 6)}`).join('; ')}. Vertical placement is decorative jitter only.`}>
          <line className="fs-axis" x1="24" x2="296" y1="96" y2="96" />
          <line className="fs-grid" x1={place(0)} x2={place(0)} y1="14" y2="96" />
          {[-extent, 0, extent].map(value => (
            <text key={value} className="is-small" x={place(value)} y="112" textAnchor="middle">{round(value, 3)}</text>
          ))}
          {explainedCases.map((entry, index) => (
            <g key={entry.sourceId} tabIndex={0}>
              <title>{`source row ${entry.sourceId}, cultivar ${entry.actualClass}: ${SMALL_LABELS[feature]} contributes ${round(entry.phi[feature], 6)} probability units`}</title>
              <circle className={entry.phi[feature] === 0 ? 'fs-mark is-hollow' : 'fs-mark'}
                cx={place(entry.phi[feature])} cy={30 + (index % 6) * 10} r="3.6" />
            </g>
          ))}
          <text className="is-small" x="24" y="128">n = 12 · probability units</text>
        </svg></ScrollRegion>
        <p className="fs-caption">
          One point per explained inspection row. The vertical spread is decorative spacing so that coincident points stay
          countable; it encodes nothing.
        </p>
        <Table caption={`Every explained case's ${SMALL_LABELS[feature]} measurement and its attribution`} scroll wrap
          headings={['source row', 'cultivar', `${SMALL_LABELS[feature]} value`, 'attribution', 'class-1 probability']}
          rows={explainedCases.map(entry => [
            entry.sourceId, entry.actualClass, round(entry.input[feature], 4),
            entry.phi[feature] === 0 ? 'exactly 0' : signed(entry.phi[feature], 6), round(entry.prediction, 4),
          ])} />
        <p>
          These are twelve actual explained rows, not a hundred invented dots. The spread across them measures variation between
          cases. It is not a confidence interval over new training fits, and not the spread across permutation repeats in the
          figure above: those are three different distributions.
          {values.every(value => value === 0)
            ? ` Every attribution for ${SMALL_LABELS[feature]} is exactly zero across all twelve, because the saved tree never splits on it.`
            : ''}
        </p>
      </div>
    </div>
  </figure>;
}
