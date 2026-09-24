import { useId, useState } from 'react';
import {
  Dag, Distribution, Table, fixed, round,
} from './BayesNetShared.jsx';
import {
  alarmNetwork, classPosterior, counterfactualPair, declaredFrontdoorParameters, eliminationRun, fixtures,
  frontdoorModel, inducedWidth, jointProbability, worldFactors,
} from '../../data/bayesnet-models.js';
import { protocol, scores, trainingModels, validationSpecimens } from '../../data/bayesnet-data.js';

/** Inline figures for the Bayesian-networks lesson.
 *
 * Each figure draws from `bayesnet-models.js`, never from a literal copied out
 * of the prose, so the verifier checks the same numbers and the same geometry
 * the reader sees. Figures carry no graded prediction: the four investigations
 * own that contract, and a figure that quietly answered one of their questions
 * would break it.
 */

const stateLabel = state => (state === 1 ? '1' : '0');

/* ------------------------------------------------- F1 · assemble one world */

export function WorldAssemblyFigure() {
  const [world, setWorld] = useState(fixtures.assembledWorld);
  const factors = worldFactors(alarmNetwork, world);
  const product = jointProbability(alarmNetwork, world);
  const selected = alarmNetwork.nodes.map(node => `${node}=${world[node]}`).join(', ');
  return <div className="bn-figure">
    <p className="bn-caption"><strong>Figure 1.</strong> One world is a product of five local choices, not a sum
      of five confidence scores. Change a state and watch which table row each node reads.</p>
    <div className="bn-figure-row">
      <Dag edges={fixtures.alarmEdges} labels={alarmNetwork.labels} width={250} height={210}
        observed={[]} endpoints={[]}
        describe={'The alarm network: burglary and earthquake both point into the alarm, and the alarm points '
          + 'into John calls and Mary calls.'}
        caption="The five-node alarm network" />
      <div>
        <div className="bn-controls">
          {alarmNetwork.nodes.map(node => <label key={node} className="bn-field">
            <span>{node} — {alarmNetwork.labels[node]}</span>
            <select value={world[node]} onChange={event => setWorld({ ...world, [node]: Number(event.target.value) })}>
              <option value={0}>0</option>
              <option value={1}>1</option>
            </select>
          </label>)}
        </div>
        <button type="button" onClick={() => setWorld(fixtures.assembledWorld)}>Back to the worked world</button>
      </div>
    </div>
    <div className="bn-strip">
      {factors.map(factor => <span key={factor.node} className="bn-strip-cell is-used">
        {factor.node} = {stateLabel(factor.state)}
        {factor.parents.length
          ? <span className="bn-bin"> given {factor.parents.map((parent, index) =>
            `${parent}=${factor.parentStates[index]}`).join(', ')}</span>
          : <span className="bn-bin"> (root)</span>}
        <b>{round(factor.value, 6)}</b>
      </span>)}
      <span className="bn-strip-cell">product<b>{round(product, 10)}</b></span>
    </div>
    <p className="bn-caption">
      Selected world {selected}. The five highlighted entries multiply to {round(product, 10)}.
      Each node's own table row is chosen by its parents' states, and the entry used is the chance of the state
      shown when that state is 1, and one minus it when the state is 0. This is a single joint cell,
      not a posterior: it is the probability of that exact combination, before any evidence is imposed.
    </p>
  </div>;
}

/* ------------------------------------------------- F2 · the factor workbench */

export function FactorWorkbenchFigure() {
  const run = eliminationRun(alarmNetwork, fixtures.eliminationEvidence, fixtures.eliminationOrder);
  const [step, setStep] = useState(run.steps.length);
  const shown = run.steps.slice(0, step);
  const current = shown.at(-1) ?? null;
  return <div className="bn-figure">
    <p className="bn-caption"><strong>Figure 2.</strong> Variable elimination on the two-call query. Each step
      collects the factors that mention one variable, multiplies them, sums that variable out, and puts the
      smaller table back. Axis unions and sums are arithmetic here; no arrow in this figure means a causal effect.</p>
    <div className="bn-buttons">
      <button type="button" onClick={() => setStep(0)} disabled={step === 0}>Start</button>
      <button type="button" onClick={() => setStep(value => (value > 0 ? value - 1 : 0))} disabled={step === 0}>Back</button>
      <button type="button" className="is-primary" onClick={() => setStep(value => (value < run.steps.length ? value + 1 : value))}
        disabled={step === run.steps.length}>Sum out the next variable</button>
      <span>{step} of {run.steps.length} eliminations applied, in the order {run.order.join(', ')}.</span>
    </div>
    {current && <Table
      caption={`After summing out ${current.variable}: the product ranged over ${current.productScope.join(', ')} `
        + `(${current.productCells} cells) and the result ranges over `
        + `${current.resultScope.join(', ') || 'no variables'} (${current.cells.length} cells).`}
      headings={[...current.resultScope.map(name => `${name}`), 'factor value']}
      rows={current.cells.map(cell => [
        ...current.resultScope.map(name => stateLabel(cell.states[name])),
        fixed(cell.value, 6),
      ])}
      footnote={'These are the cells of the factor this step just produced. Factors the step did not touch are '
        + 'still waiting alongside it — here the burglary prior — so these numbers are not yet the masses below, '
        + 'and multiplying that prior in is what turns them into those masses.'} />}
    {!current && <p className="bn-caption">
      Nothing has been eliminated yet. The starting factors are the five conditional probability tables, with the
      two observed callers already fixed at 1: {run.steps.length} eliminations remain.
    </p>}
    <Distribution caption="Unnormalised burglary mass after every elimination, then the normalised posterior"
      labels={['B = 0', 'B = 1']} values={run.mass.map(value => value / (run.mass[0] + run.mass[1]))}
      highlight={1}
      describe="Two bars: no burglary takes about seventy-two percent of the mass and burglary about twenty-eight." />
    <p className="bn-caption">
      The two unnormalised masses are {round(run.mass[0], 12)} and {round(run.mass[1], 12)}; their sum
      {' '}{round(run.evidenceProbability, 12)} is the probability of the evidence itself. Dividing gives
      {' '}{round(run.posterior, 12)}, the same number section 2 obtained by adding all 32 worlds. The two routes
      agree because elimination only reorders the same sums and products; it never drops a dependency.
    </p>
  </div>;
}

/* --------------------------------------------- F3 · what an order costs */

/** The chain, its elimination positions, and every fill edge the order actually
 * creates. The arcs are drawn from `inducedWidth(...).fillEdges`, not from a
 * remembered picture: an order that couples two more nodes late in the sweep
 * must show that second arc rather than quietly omit it. */
function UndirectedChain({ order, fills, width = 300, baseline = 66 }) {
  const titleId = useId();
  const descriptionId = useId();
  const nodes = ['A', 'B', 'C', 'D'];
  const radius = 15;
  const positions = Object.fromEntries(nodes.map((node, index) => (
    [node, { x: 40 + index * ((width - 80) / 3), y: baseline }])));
  const arcHeight = span => 16 + 12 * span;
  const height = baseline + 34;
  return <svg viewBox={`0 0 ${width} ${height}`} role="img" style={{ maxWidth: `${width}px` }}
    aria-labelledby={titleId} aria-describedby={descriptionId}>
    <title id={titleId}>{`The chain A–B–C–D, eliminated in the order ${order.join(', ')}`}</title>
    <desc id={descriptionId}>{`An undirected chain A–B–C–D eliminated in the order ${order.join(', ')}`
      + `${fills.length ? `, with fill edges ${fills.map(pair => pair.join('–')).join(' and ')}` : ' and no fill edge'}.`}</desc>
    {[['A', 'B'], ['B', 'C'], ['C', 'D']].map(([from, to]) => <line key={`${from}${to}`} className="bn-axis"
      x1={positions[from].x + radius} y1={positions[from].y} x2={positions[to].x - radius} y2={positions[to].y} />)}
    {fills.map(([from, to]) => {
      const span = nodes.indexOf(to) - nodes.indexOf(from);
      const lift = arcHeight(span);
      return <path key={`${from}${to}`} className="bn-fill-arc"
        d={`M ${positions[from].x} ${positions[from].y - radius} `
          + `Q ${(positions[from].x + positions[to].x) / 2} ${positions[from].y - radius - lift} `
          + `${positions[to].x} ${positions[to].y - radius}`} />;
    })}
    {nodes.map(node => <g key={node} className="bn-node">
      <circle cx={positions[node].x} cy={positions[node].y} r={radius} />
      <text x={positions[node].x} y={positions[node].y + 5} textAnchor="middle" className="bn-node-name">{node}</text>
      <text x={positions[node].x} y={positions[node].y + radius + 15} textAnchor="middle" className="bn-small">
        step {order.indexOf(node) + 1}
      </text>
    </g>)}
  </svg>;
}

export function EliminationGeometryFigure() {
  const chainNetwork = {
    nodes: ['A', 'B', 'C', 'D'],
    parents: { A: [], B: ['A'], C: ['B'], D: ['C'] },
    chance: { A: { '': 0.5 }, B: { 0: 0.5, 1: 0.5 }, C: { 0: 0.5, 1: 0.5 }, D: { 0: 0.5, 1: 0.5 } },
  };
  const endpoints = inducedWidth(chainNetwork, fixtures.chainEndpointOrder);
  const middle = inducedWidth(chainNetwork, fixtures.chainMiddleOrder);
  return <div className="bn-figure">
    <p className="bn-caption"><strong>Figure 3.</strong> The same four-variable chain, eliminated two ways. The
      small number under each node is its position in that order. Both orders give the same marginal; they do not
      cost the same. This is an exact operation and storage count for binary variables, not a measured runtime.</p>
    <div className="bn-figure-row">
      <div>
        <p className="bn-caption">Endpoints first: {fixtures.chainEndpointOrder.join(', ')}</p>
        <UndirectedChain order={fixtures.chainEndpointOrder} fills={endpoints.fillEdges} />
        <p className="bn-caption">Largest intermediate scope {endpoints.width + 1} variables,
          {' '}{endpoints.largestBinaryFactorCells} binary cells.
          {' '}{endpoints.fillEdges.length ? `Fill edges: ${endpoints.fillEdges.map(pair => pair.join('–')).join(', ')}.`
            : 'No fill edge is created at all.'}</p>
      </div>
      <div>
        <p className="bn-caption">Middle first: {fixtures.chainMiddleOrder.join(', ')}</p>
        <UndirectedChain order={fixtures.chainMiddleOrder} fills={middle.fillEdges} />
        <p className="bn-caption">Largest intermediate scope {middle.width + 1} variables,
          {' '}{middle.largestBinaryFactorCells} binary cells. Fill
          edges: {middle.fillEdges.map(pair => pair.join('–')).join(', ')}.</p>
      </div>
    </div>
    <p className="bn-caption">
      The dashed arcs are the whole structural difference. Eliminating an interior node couples its remaining
      neighbours, and that new coupling is what the next step has to carry: removing B joins A and C, and then
      removing C joins A and D in turn. Doubling the cell count on four variables is a small price; the same
      mechanism at width 30 is what makes a single intermediate factor take 16 GiB.
    </p>
  </div>;
}

/* ------------------------------- F4 · two fitted models on the same specimens */

/** Recompute both models' answers for every validation specimen.
 *
 * The averages of the log-loss column here must reproduce the recorded global
 * scores, which were computed natively. That agreement is the point of the
 * figure: a reader can see the two numbers the table reports being assembled
 * from the 36 rows beneath them.
 */
export function measuredContrast() {
  const medians = trainingModels.treeAugmented.medians;
  return validationSpecimens.map(specimen => {
    const bits = specimen.features.map((value, index) => (value > medians[index] ? 1 : 0));
    const naive = classPosterior(trainingModels.naiveBayes, bits, [0, 1, 2, 3]);
    const tree = classPosterior(trainingModels.treeAugmented, bits, [0, 1, 2, 3]);
    return {
      id: specimen.id, cultivar: specimen.cultivar,
      naiveTrue: naive.posterior[specimen.cultivar],
      treeTrue: tree.posterior[specimen.cultivar],
      naiveLoss: -Math.log(naive.posterior[specimen.cultivar]),
      treeLoss: -Math.log(tree.posterior[specimen.cultivar]),
      naiveCorrect: naive.leading === specimen.cultivar,
      treeCorrect: tree.leading === specimen.cultivar,
    };
  });
}

export function MeasuredContrastFigure() {
  const rows = measuredContrast();
  const [sorted, setSorted] = useState(false);
  const mean = key => rows.reduce((sum, row) => sum + row[key], 0) / rows.length;
  const ordered = sorted
    ? [...rows].sort((left, right) => (right.naiveLoss - right.treeLoss) - (left.naiveLoss - left.treeLoss))
    : rows;
  const disagreements = rows.filter(row => row.naiveCorrect !== row.treeCorrect);
  const naiveWrong = rows.filter(row => !row.naiveCorrect);
  const treeWrong = rows.filter(row => !row.treeCorrect);
  return <div className="bn-figure">
    <p className="bn-caption"><strong>Figure 4.</strong> Every validation specimen, with the probability each
      fitted model assigned to that specimen's <em>actual</em> cultivar and the log-loss contribution that
      follows. These are real observations and a real measured comparison, on one small split.</p>
    <div className="bn-buttons">
      <button type="button" className={sorted ? 'is-selected' : undefined} onClick={() => setSorted(value => !value)}>
        {sorted ? 'Back to specimen order' : 'Sort by how much the two models disagree'}
      </button>
      <span>All {rows.length} rows stay on screen in either order; nothing is filtered out.</span>
    </div>
    <Table
      caption={'Validation specimens: probability assigned to the true cultivar, and its log-loss contribution. '
        + 'A lower probability costs more.'}
      headings={['specimen', 'true cultivar', 'NB p(true)', 'TAN p(true)', 'NB −log p', 'TAN −log p', 'who is right']}
      rowClass={index => (ordered[index].naiveCorrect !== ordered[index].treeCorrect ? 'is-leading' : undefined)}
      rows={ordered.map(row => [
        String(row.id), String(row.cultivar),
        fixed(row.naiveTrue, 6), fixed(row.treeTrue, 6),
        fixed(row.naiveLoss, 6), fixed(row.treeLoss, 6),
        row.naiveCorrect && row.treeCorrect ? 'both' : row.naiveCorrect ? 'NB only' : row.treeCorrect ? 'TAN only' : 'neither',
      ])}
      footnote={`Column means: NB ${round(mean('naiveLoss'), 6)} and TAN ${round(mean('treeLoss'), 6)}. `
        + `Those are exactly the validation log losses ${round(scores.naiveBayesValidation.logLoss, 6)} and `
        + `${round(scores.treeAugmentedValidation.logLoss, 6)} reported in the table above, reassembled from the rows.`} />
    <p className="bn-caption">
      {naiveWrong.length === treeWrong.length
        ? `Both models misclassify ${naiveWrong.length} of ${rows.length} specimens, so accuracy cannot separate them.`
        : `NB misclassifies ${naiveWrong.length} specimens and TAN ${treeWrong.length}.`}
      {disagreements.length
        ? ` They disagree about which class leads on ${disagreements.length} specimen${disagreements.length === 1 ? '' : 's'}: `
          + `${disagreements.map(row => row.id).join(', ')}. Those rows are highlighted.`
        : ' They never disagree about which class leads, so every difference between them is a difference of confidence alone.'}
      {' '}What separates the two totals is confidence on the rows they both get right, and how badly each is hurt
      by the rows it gets wrong. Accuracy reads only the largest entry; log loss reads the whole probability vector.
    </p>
  </div>;
}

/* ------------------------------------------- F5 · the two frontdoor averages */

export function FrontdoorFigure() {
  const model = frontdoorModel();
  const rows = model.observedRiskGivenMediatorAndTreatment;
  return <div className="bn-figure">
    <p className="bn-caption"><strong>Figure 5.</strong> The frontdoor calculation as two nested averages. The
      inner tray corrects each mediator response for its association with the upstream treatment; the outer tray
      combines those two corrected responses using the mediator distribution the intervention induces. Every
      number here comes from the fully specified constructed model, not from data.</p>
    <Table caption="The declared generative model behind the frontdoor example"
      headings={['mechanism', 'conditioning values', 'probability of 1']}
      rows={[
        ['U', 'none', String(declaredFrontdoorParameters.latentPrior)],
        ...declaredFrontdoorParameters.chanceX.map((value, u) => ['X', `U = ${u}`, String(value)]),
        ...declaredFrontdoorParameters.chanceM.map((value, x) => ['M', `X = ${x}`, String(value)]),
        ...declaredFrontdoorParameters.chanceY.flatMap((row, m) => row.map((value, u) =>
          ['Y', `M = ${m}, U = ${u}`, String(value)])),
      ]}
      footnote="All four variables are binary; each probability of 0 is the complement of the listed value. U is hidden from the observed-data formula, but its distribution and effects are specified here so we can independently check that formula by intervening on the complete model." />
    <div className="bn-figure-row">
      <Dag edges={fixtures.frontdoorEdges} width={250} height={220}
        labels={{ U: 'unobserved common cause', X: 'treatment', M: 'mediator', Y: 'outcome' }}
        endpoints={['X', 'Y']} dimmedEdges={[['U', 'X'], ['U', 'Y']]}
        describe={'U points into X and into Y with dashed arrows because U is never observed. X points into M, '
          + 'and M points into Y. There is no direct arrow from X to Y.'}
        caption="U is unobserved; the dashed arrows are the pair no adjustment set can reach" />
      <div>
        <Table caption="Inner tray: the observed failure probability for each mediator state and treatment, then its average over the treatment distribution"
          headings={['mediator M', 'X = 0', 'X = 1', 'average over P(X)']}
          rows={[0, 1].map(m => [
            `M = ${m}`, fixed(rows[m][0], 6), fixed(rows[m][1], 6), fixed(model.inner[m], 6),
          ])}
          footnote={`The treatment distribution is ${round(model.marginX[0], 6)} and ${round(model.marginX[1], 6)}, `
            + 'so the inner average is an even mixture of the two columns here.'} />
      </div>
    </div>
    {/* The outer tray sits below the row rather than inside it. Its point is
        that the last two columns agree, and in a half-width column the second
        of them falls outside the scroll box — the claim would be one the reader
        has to scroll sideways to check. */}
    <Table caption="Outer tray: the mediator distribution each intervention induces, applied to those two inner responses. The last two columns are the same quantity by two different routes."
      headings={['intervention', 'P(M=0 | do)', 'P(M=1 | do)', 'frontdoor formula', 'truncated factorisation']}
      rows={model.queries.map(query => [
        `do(X = ${query.treatment})`,
        fixed(query.mediatorGiven[0], 6), fixed(query.mediatorGiven[1], 6),
        fixed(query.frontdoor, 6), fixed(query.truncated, 6),
      ])} />
    <p className="bn-caption">
      The two right-hand columns agree to the last digit, and they were computed by genuinely different routes: the
      frontdoor formula uses only the observed variables X, M and Y, while the truncated factorisation uses the
      complete model including the hidden U. That agreement is what identification means here. Conditioning on the
      treatment instead would answer a different question: the plain observational probabilities are
      {' '}{round(model.queries[0].observational, 6)} and {round(model.queries[1].observational, 6)}, a gap of
      {' '}{round(model.observationalDifference, 6)} against the causal {round(model.frontdoorDifference, 6)}.
    </p>
  </div>;
}

/* --------------------------------- F6 · the same units across two worlds */

export function CounterfactualFigure() {
  const { models } = counterfactualPair();
  return <div className="bn-figure">
    <p className="bn-caption"><strong>Figure 6.</strong> Two models, two units each, with the unit identity kept on
      the row. Read down a column to get a population average; read across a row to get one unit's pair of
      outcomes. The columns agree and the rows do not.</p>
    <div className="bn-figure-row">
      {/* Four columns, not six. The two panels sit side by side, so each gets
          about half the figure's width; at five columns the second outcome
          column fell outside the scroll box, and that column is the whole
          point of the figure. The two units carry equal weight, which the
          caption states once rather than repeating down a column. */}
      {models.map(model => <div key={model.name}>
        <Table caption={`Model ${model.name}: ${model.assignment}. Both units carry weight one half.`}
          headings={['unit U', 'Y if X = 0', 'Y if X = 1', 'changes?']}
          rows={[
            ...model.rows.map(row => [
              String(row.unit), String(row.outcomeUnderZero), String(row.outcomeUnderOne),
              row.outcomeUnderZero === row.outcomeUnderOne ? 'no' : 'yes',
            ]),
            ['average', fixed(model.averageUnderZero, 1), fixed(model.averageUnderOne, 1), '—'],
          ]} />
      </div>)}
    </div>
    <p className="bn-caption">
      Both column averages are one half in both models, so every observational and every population-interventional
      statement about X and Y is identical. In model A no unit's outcome moves when X is changed; in model B every
      unit's does. The conditional probability tables never recorded which of those two worlds we are in, because
      they only ever describe one column at a time.
    </p>
  </div>;
}

/* ------------------------------------------- F7 · a Markov equivalence family */

export function EquivalenceFigure() {
  const family = fixtures.equivalenceClass;
  const collider = fixtures.collider;
  return <div className="bn-figure">
    <p className="bn-caption"><strong>Figure 7.</strong> Three orientations of the same three-node skeleton, and
      the one that is different. The first three imply exactly the same conditional independence; observational
      independence information alone cannot choose between them.</p>
    <div className="bn-figure-row">
      {family.map(member => <Dag key={member.name} edges={member.edges} width={190}
        endpoints={['X', 'Y']} caption={member.name} legend={false}
        describe={`A three-node graph, ${member.name}, joining X, Y and Z.`} />)}
    </div>
    <p className="bn-caption">
      In all four drawings arrowheads give direction, a dashed square badge marks the two query endpoints X and Y,
      and no variable is observed. Only the arrowheads differ.
    </p>
    <p className="bn-caption">
      All three have the skeleton X–Z–Y and no unshielded collider, and all three say X and Y are independent once
      Z is given. Drawn separately below is the fourth orientation, which has the same skeleton but an unshielded
      collider at Z. It belongs to a different equivalence class and says the opposite: X and Y are independent
      with nothing observed, and dependence becomes possible once Z is.
    </p>
    <Dag edges={collider.edges} width={190} endpoints={['X', 'Y']} legend={false}
      caption="collider — a separate class"
      describe="A three-node graph in which X and Y both point into Z." />
    <p className="bn-caption">
      A missing or spurious adjacency is a different kind of error and is not excused by equivalence: these four
      graphs all share the skeleton, and a graph with an extra or absent edge is not a member of either class.
      The protocol behind this lesson's fitted models is fixed at {protocol.trainSize} training specimens and does
      not attempt structure discovery beyond the declared tree.
    </p>
  </div>;
}
