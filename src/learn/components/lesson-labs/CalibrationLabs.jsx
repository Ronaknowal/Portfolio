import { useState } from 'react';
import {
  BarRows, Investigation, KindTag, NumberField, OutcomeChoice, Pill, PlotFrame, LiveResult, Select, Table,
  ratio, round, useInvestigation,
} from './CalibrationShared.jsx';
import {
  binIndexOf, blockGeometry, brierLoss, changeDirection, classSets, conformalThreshold, controlSteps,
  cqrInterval, fitSigmoid, fixtures, gradedText, isotonicPredict, limits, normalizedInterval, pavStages,
  railGeometry, rankRotation, reliability, reliabilityGeometry, rocAuc, thresholdDirection,
} from '../../data/calibration-models.js';



const TOLERANCE_NOTE = 'unchanged means within 10⁻¹²';

/* ==================================== I1 · forecast cards and the vanishing ECE */

const cardInitial = {
  cards: fixtures.cards.map(card => ({ ...card })),
  internalEdges: [0.5],
  target: 'ece',
};

const edgesOf = draft => [0, ...draft.internalEdges, 1];

const reportOf = draft => reliability(
  draft.cards.map(card => card.forecast), draft.cards.map(card => card.outcome), edgesOf(draft));

/** The three quantities this lab can grade, each with its own null case. */
const cardQuantity = (draft, key) => {
  const forecasts = draft.cards.map(card => card.forecast);
  const outcomes = draft.cards.map(card => card.outcome);
  if (key === 'ece') return { value: reportOf(draft).ece, because: null };
  if (key === 'brier') return { value: brierLoss(forecasts, outcomes), because: null };
  const auc = rocAuc(forecasts, outcomes);
  return { value: auc.value, because: auc.because };
};

const QUANTITY_LABELS = {
  ece: 'the binned ECE',
  brier: 'the Brier loss',
  auc: 'the ROC AUC',
};

export function ReliabilityLab() {
  const state = useInvestigation(cardInitial);
  const { draft, active } = state;
  const revealed = Boolean(state.result);
  
  const baseline = revealed ? state.previous : active;
  const baselineReport = reportOf(baseline);
  const draftReport = reportOf(draft);
  const baselineGeometry = reliabilityGeometry(baselineReport);
  const draftGeometry = reliabilityGeometry(draftReport);
  const [edgeText, setEdgeText] = useState('0.5');
  const [edgeError, setEdgeError] = useState(null);

  const applyEdges = text => {
    setEdgeText(text);
    const parts = text.split(',').map(part => part.trim()).filter(Boolean);
    const values = parts.map(Number);
    if (values.some(value => !Number.isFinite(value))) {
      setEdgeError('Every boundary must be a number. Separate them with commas.'); return;
    }
    if (values.some(value => value <= 0 || value >= 1)) {
      setEdgeError('Internal boundaries lie strictly between 0 and 1; the outer two are always 0 and 1.'); return;
    }
    for (let index = 1; index < values.length; index += 1) {
      if (values[index] <= values[index - 1]) {
        setEdgeError('Boundaries must strictly increase, and no two may coincide.'); return;
      }
    }
    if (values.length + 1 > limits.bins.maximum) {
      setEdgeError(`At most ${limits.bins.maximum} bins, so at most ${limits.bins.maximum - 1} internal boundaries.`);
      return;
    }
    setEdgeError(null);
    state.edit({ internalEdges: values });
  };

  const setCard = (index, update) => state.edit({
    cards: draft.cards.map((card, position) => (position === index ? { ...card, ...update } : card)),
  });

  
  const from = active;
  const presets = [
    ['Merge the two bins into one', { ...from, internalEdges: [] }],
    ['Original ten cards with forecasts .4 and .6', {
      ...from,
      cards: fixtures.cards.map((card, index) => ({
        ...card, forecast: fixtures.repairedForecasts[index],
      })),
      internalEdges: [0.5],
    }],
    ['Reverse the card order — an exact null', { ...from, cards: [...from.cards].reverse() }],
    ['Three bins at .25 and .75', { ...from, internalEdges: [0.25, 0.75] }],
    ['The boundary fixture: forecasts 0, .5 and 1', {
      ...from, cards: fixtures.boundaryCards.map(card => ({ ...card })), internalEdges: [0.5],
    }],
    ['Make every outcome 1 — AUC then has no value', {
      ...from, cards: from.cards.map(card => ({ ...card, outcome: 1 })),
    }],
    ['Back to the ten original cards', cardInitial],
  ];

  return <Investigation
    title="Investigation 1 — make an aggregate error disappear"
    question={'Ten forecasts with their outcomes. Choose the bins, or change a card, and inspect what happens to '
      + 'the summary as the visual updates. One of the changes below leaves every card exactly as it was.'}
    role={{
      kind: 'constructed',
      text: 'Ten constructed forecast cards. These are designed teaching values, not measured forecasts, and '
        + 'nothing here reads the banknote or airfoil experiments.',
    }}
    onReset={() => { state.reset(); setEdgeText('0.5'); setEdgeError(null); }}>

    <div className="cal-cards">
      {draft.cards.map((card, index) => {
        const bin = binIndexOf(edgesOf(draft), card.forecast);
        return <div key={card.id} className="cal-card">
          <span className="cal-card-name"><b>card {card.id}</b><span>bin {bin + 1}</span></span>
          <NumberField label="forecast for class 1" value={card.forecast} {...controlSteps.forecast}
            min={controlSteps.forecast.minimum} max={controlSteps.forecast.maximum}
            onChange={value => setCard(index, { forecast: value })} />
          <OutcomeChoice label="outcome" value={card.outcome}
            onChange={value => setCard(index, { outcome: value })} />
          <span className="cal-bin">
            falls in [{round(edgesOf(draft)[bin], 3)}, {round(edgesOf(draft)[bin + 1], 3)}
            {bin === edgesOf(draft).length - 2 ? ']' : ')'}
          </span>
        </div>;
      })}
    </div>
    <div className="cal-buttons">
      <button type="button" disabled={draft.cards.length >= limits.cards.maximum}
        onClick={() => state.edit({
          cards: [...draft.cards, {
            id: String.fromCharCode(65 + draft.cards.length), forecast: 0.5, outcome: 0,
          }],
        })}>Add a card</button>
      <button type="button" disabled={draft.cards.length <= limits.cards.minimum}
        onClick={() => state.edit({ cards: draft.cards.slice(0, -1) })}>Remove the last card</button>
      <span>Between {limits.cards.minimum} and {limits.cards.maximum} cards.</span>
    </div>

    <div className="cal-controls is-wide">
      <label className="cal-field">
        <span>Internal bin boundaries, separated by commas</span>
        <input type="text" value={edgeText} onChange={event => applyEdges(event.target.value)}
          aria-invalid={Boolean(edgeError)} />
        <span className="cal-caption">
          The outer boundaries are always 0 and 1. A boundary belongs to the bin on its right, and a forecast of
          exactly 1 stays in the last bin. Leave this empty for a single bin.
        </span>
        {edgeError && <span className="cal-field-error" role="alert">{edgeError} The bins are unchanged.</span>}
      </label>
      <Select label="Summary to inspect" value={draft.target}
        options={[['ece', QUANTITY_LABELS.ece], ['brier', QUANTITY_LABELS.brier], ['auc', QUANTITY_LABELS.auc]]}
        onChange={value => state.edit({ target: value })}
        hint="Switching this changes the inspected quantity and recomputes its comparison." />
    </div>

    <div className="cal-presets">
      {presets.map(([label, setup]) => <button key={label} type="button"
        onClick={() => { setEdgeText((setup.internalEdges ?? []).join(', ')); setEdgeError(null); state.suggest(setup); }}>
        {label}
      </button>)}
    </div>
    <p className="cal-caption">
      A suggested setup makes one change to the state that is currently <em>applied</em>, not to whatever is
      in the fields. That is what makes an exact null exactly null whenever you click it: the comparison measures
      the change the button names. The calculation updates directly.
    </p>

    {/* Keep the comparison baseline visible so the magnitude and direction of changes are meaningful. */}
    {/* Not a two-column row. A six-column bin table in half the lesson width
        overflowed its own scroll box by 47 px at 1366, so the plot and the
        table stack and the table gets the full width. */}
    <ReliabilityPlot geometry={baselineGeometry} report={baselineReport}
      caption="The state used as the comparison baseline" />
    <div>
      <Table caption="Bins of the state used as the comparison baseline"
          headings={['bin', 'cards', 'positive', 'mean forecast', 'observed fraction', 'observed − forecast']}
          rows={baselineReport.bins.map(bin => [
            `${bin.index + 1}: [${round(bin.lower, 3)}, ${round(bin.upper, 3)}${bin.index === baselineReport.bins.length - 1 ? ']' : ')'}`,
            String(bin.count),
            bin.count ? String(bin.positive) : '—',
            bin.count ? round(bin.meanP, 6) : ratio(null, 'no card falls in this bin'),
            bin.count ? round(bin.fractionPositive, 6) : ratio(null, 'no card falls in this bin'),
            bin.count ? round(bin.gap, 6) : '—',
          ])}
          kind={baselineReport.kind}
          footnote={'An empty bin has no observed fraction. That is not a fraction of zero, and it is why the row '
            + 'stays here while no dot is drawn for it.'} />
        <p className="cal-caption">
          Applied state: {QUANTITY_LABELS[draft.target]} is {' '}
          {ratio(cardQuantity(baseline, draft.target).value, cardQuantity(baseline, draft.target).because)}
          {' '}over {baselineReport.count} cards in {baselineReport.occupiedBins} occupied
        {baselineReport.emptyBins ? ` and ${baselineReport.emptyBins} empty` : ''} bins.
      </p>
    </div>

    <LiveResult
      state={state}
      
      
      
      
      
      
      
      calculateInputs={(next, current) => {
        const after = cardQuantity(next, next.target);
        
        const before = cardQuantity(current, next.target);
        const outcome = changeDirection(after.value, before.value);
        const afterReport = reportOf(next);
        const beforeReport = reportOf(current);
        return {
          outcome, value: after.value,
          explain: `${QUANTITY_LABELS[next.target]} went from ${gradedText(before.value)} to ${gradedText(after.value)}. `
            + `The applied state had ${beforeReport.occupiedBins} occupied bins and this one has `
            + `${afterReport.occupiedBins}. `
            + (after.because
              ? `${after.because[0].toUpperCase()}${after.because.slice(1)}: a quantity with no value is not zero.`
              : afterReport.occupiedBins === 1 && next.target === 'ece'
                ? 'With one occupied bin, ECE is the absolute difference between the overall mean forecast and '
                  + 'the overall positive fraction. It is zero only when those averages agree. Merging bins can '
                  + 'cancel opposing local discrepancies; it does not repair a model.'
                : next.target === 'ece'
                  ? 'Binned ECE is a summary of one chosen partition. A smaller value on a different partition is a '
                    + 'statement about the partition as much as about the forecasts.'
                  : next.target === 'brier'
                    ? 'Brier loss averages squared forecast errors. Moving only bin boundaries cannot change it.'
                    : 'AUC compares positive-negative score pairs, giving half credit to ties. Moving only bin '
                      + 'boundaries cannot change it.'),
        };
      }} />

    {revealed && <div className="cal-graded">
      <ReliabilityPlot geometry={draftGeometry} report={draftReport}
        caption="The state for the current inputs" />
      <div>
        <Table caption="Bins of the state for the current inputs"
            headings={['bin', 'cards', 'positive', 'mean forecast', 'observed fraction', 'observed − forecast']}
            rows={draftReport.bins.map(bin => [
              `${bin.index + 1}: [${round(bin.lower, 3)}, ${round(bin.upper, 3)}${bin.index === draftReport.bins.length - 1 ? ']' : ')'}`,
              String(bin.count),
              bin.count ? String(bin.positive) : '—',
              bin.count ? round(bin.meanP, 6) : ratio(null, 'no card falls in this bin'),
              bin.count ? round(bin.fractionPositive, 6) : ratio(null, 'no card falls in this bin'),
              bin.count ? round(bin.gap, 6) : '—',
            ])}
            kind={draftReport.kind} />
          <p className="cal-caption">
            Binned ECE {gradedText(draftReport.ece)}; Brier loss{' '}
            {gradedText(brierLoss(draft.cards.map(card => card.forecast), draft.cards.map(card => card.outcome)))};
            {' '}AUC {ratio(rocAuc(draft.cards.map(card => card.forecast),
              draft.cards.map(card => card.outcome)).value,
            'every observation carries the same outcome')}.
        </p>
      </div>
    </div>}
  </Investigation>;
}

/** The reliability scatter, its diagonal, its signed gaps and its count rail. */
function ReliabilityPlot({ geometry, report, caption }) {
  const { box } = geometry;
  /* The rail's baseline comes from the geometry, which places it clear of the
     tick-label row. Computing it here was how the bars came to cover the
     labels. */
  const railTop = geometry.railBaseline;
  return <div>
    <PlotFrame geometry={geometry} caption={caption}
      xLabel="mean forecast for class 1" yLabel="observed fraction of class 1"
      describe={`A square plot from 0 to 1 on both axes with a dashed diagonal. `
        + `${geometry.points.length} dots are drawn: `
        + `${geometry.points.map(point =>
          `bin ${point.index + 1} at mean forecast ${round(point.meanP, 4)} and observed fraction `
          + `${round(point.fractionPositive, 4)} over ${point.count} cards, `
          + `${Math.abs(point.gap) < 1e-15 ? 'on' : point.gap > 0 ? 'above' : 'below'} the diagonal`).join('; ')}.`
        + (geometry.emptyBins.length
          ? ` ${geometry.emptyBins.length} bin or bins are empty and have no dot at all.`
          : '')}>
      <line className="cal-diagonal" x1={geometry.diagonal.x1} y1={geometry.diagonal.y1}
        x2={geometry.diagonal.x2} y2={geometry.diagonal.y2} />
      {geometry.points.map(point => <g key={point.index}>
        <line className="cal-gap-line" x1={point.x} y1={point.diagonalY} x2={point.x} y2={point.y} />
        <circle className={`cal-dot ${Math.abs(point.gap) < 1e-15 ? '' : point.gap > 0 ? 'is-above' : 'is-below'}`}
          cx={point.x} cy={point.y} r={4.5} />
        <text className="cal-point-label" x={point.labelX} y={point.labelY} textAnchor="middle">
          {point.index + 1}
        </text>
      </g>)}
      {geometry.rail.map(bar => <g key={bar.index}>
        <rect className={`cal-rail-bar${bar.empty ? ' is-empty' : ''}`}
          x={bar.x1 + 1} y={railTop - bar.markerHeight}
          width={Math.max(bar.x2 - bar.x1 - 2, 1)}
          height={bar.markerHeight} />
        <text className="cal-small" x={bar.countLabelX} y={geometry.countLabelY} textAnchor="middle">
          {bar.count}
        </text>
      </g>)}
    </PlotFrame>
    <p className="cal-caption">
      A hollow ring sits above the diagonal — the outcome happened more often than the forecast said. A filled
      disc sits below it — the forecast was too high. The bar strip under the axis is how many cards each bin
      holds; a dashed outline is an empty bin, which has no dot because it has no observed fraction.
      Counts: {report.bins.map(bin => `bin ${bin.index + 1} holds ${bin.count}`).join(', ')}.
    </p>
  </div>;
}

/* ============================================ I2 · build the monotone map */

const monotoneInitial = {
  rows: fixtures.labPavScores.map((score, index) => ({
    id: `r${index + 1}`, score, label: fixtures.labPavLabels[index],
  })),
  applied: 0,
};

const stagesOf = draft => pavStages(draft.rows.map(row => row.score), draft.rows.map(row => row.label));

export function MonotoneLab() {
  const state = useInvestigation(monotoneInitial);
  const { draft } = state;
  const revealed = Boolean(state.result);
  const stages = stagesOf(draft);
  const applied = Math.min(draft.applied, stages.stages.length - 1);
  const blocks = stages.stages[applied];
  const nextMerge = stages.merges[applied] ?? null;
  const [showSigmoid, setShowSigmoid] = useState(false);

  /* Options are the adjacent pairs actually present in the current stage, plus
     the "nothing left to merge" answer. No pair is marked as violating: that is
     the question. */
  const pairOptions = blocks.slice(0, -1).map((block, index) => [
    `pair-${index}`,
    `blocks ${block.start + 1}–${block.end + 1} and ${blocks[index + 1].start + 1}–${blocks[index + 1].end + 1}`,
  ]);
  const options = [...pairOptions, ['none', 'no merge is required — the blocks already increase']];

  const setRow = (index, update) => state.edit({
    rows: draft.rows.map((row, position) => (position === index ? { ...row, ...update } : row)),
    applied: 0,
  });

  const presets = [
    ['Tied scores −2, −2, 0, 1 with labels 1, 0, 0, 1', {
      rows: fixtures.tiedScores.map((score, index) => ({
        id: `t${index + 1}`, score, label: fixtures.tiedLabels[index],
      })), applied: 0,
    }],
    ['The same tied scores with the score-0 label flipped to 1', {
      rows: fixtures.tiedScores.map((score, index) => ({
        id: `t${index + 1}`, score, label: fixtures.tiedChangedLabels[index],
      })), applied: 0,
    }],
    ['Practice 2: scores 1…6 with labels 0, 1, 0, 1, 0, 1', {
      rows: fixtures.practicePavScores.map((score, index) => ({
        id: `p${index + 1}`, score, label: fixtures.practicePavLabels[index],
      })), applied: 0,
    }],
    ['The eight scores worked through in section 3', {
      rows: fixtures.pavScores.map((score, index) => ({
        id: `s${index + 1}`, score, label: fixtures.pavLabels[index],
      })), applied: 0,
    }],
    /* Built from the applied state, not the draft: see investigation 1. */
    ['Reverse the row order — an exact null', { rows: [...state.active.rows].reverse(), applied: 0 }],
    ['Make every label 1 — a legitimate constant fit', {
      rows: state.active.rows.map(row => ({ ...row, label: 1 })), applied: 0,
    }],
    ['Back to the six original observations', monotoneInitial],
  ];

  const sigmoid = fitSigmoid(draft.rows.map(row => row.score), draft.rows.map(row => row.label));
  const geometry = blockGeometry({
    knots: stages.knots, fitted: stages.fitted, weights: stages.weights, totals: stages.totals,
    blocks: blocks.map(block => ({ ...block })),
  });

  return <Investigation
    title="Investigation 2 — pool the adjacent violators, one merge at a time"
    question={'Every block below shows how many observations it holds and how many of them are positive. Decide '
      + 'which adjacent pair the algorithm must merge next, and what their pooled probability will be, before '
      + 'applying it.'}
    role={{
      kind: 'constructed',
      text: 'Constructed scores and labels. The fitted probabilities are a calibration-sample quantity: they are '
        + 'what this sample says, not a measured population rate.',
    }}
    onReset={() => { state.reset(); setShowSigmoid(false); }}>

    <div className="cal-controls is-wide">
      {draft.rows.map((row, index) => <div key={row.id} className="cal-card">
        <span className="cal-card-name"><b>{row.id}</b></span>
        <NumberField label="score" value={row.score} {...controlSteps.score}
          min={controlSteps.score.minimum} max={controlSteps.score.maximum}
          onChange={value => setRow(index, { score: value })} />
        <OutcomeChoice label="label" value={row.label} onChange={value => setRow(index, { label: value })} />
      </div>)}
    </div>
    <div className="cal-buttons">
      <button type="button" disabled={draft.rows.length >= limits.pavRows.maximum}
        onClick={() => state.edit({
          rows: [...draft.rows, { id: `r${draft.rows.length + 1}`, score: 5, label: 1 }], applied: 0,
        })}>Add an observation</button>
      <button type="button" disabled={draft.rows.length <= limits.pavRows.minimum}
        onClick={() => state.edit({ rows: draft.rows.slice(0, -1), applied: 0 })}>Remove the last</button>
      <button type="button" disabled={applied === 0}
        onClick={() => state.edit({ applied: 0 })}>Start the merges again</button>
      <span>Any edit restarts the merges, because the blocks are built from the observations.</span>
    </div>

    <div className="cal-presets">
      {presets.map(([label, setup]) => <button key={label} type="button"
        onClick={() => state.suggest(setup)}>{label}</button>)}
    </div>
    <p className="cal-caption">
      A suggested setup makes one change to the state that is currently <em>applied</em>, not to whatever is
      in the fields. That is what makes an exact null exactly null whenever you click it: the comparison measures
      the change the button names. The calculation updates directly.
    </p>

    <p className="cal-caption">
      Equal scores are one block before any merging starts, so they can never receive different fitted values.
      {' '}{applied} of {stages.merges.length === 0 ? 'no' : stages.merges.length} required merges applied.
    </p>

    <Table caption={`Blocks after ${applied} merge${applied === 1 ? '' : 's'}`}
      headings={['block', 'scores it spans', 'observations', 'positive', 'block probability']}
      rows={blocks.map(block => [
        `${block.start + 1}–${block.end + 1}`,
        stages.knots.slice(block.start, block.end + 1).map(knot => round(knot, 3)).join(', '),
        String(block.weight), String(block.total), round(block.mean, 6),
      ])}
      kind="calibration"
      footnote={'A block probability is its positive count divided by its observation count. Two blocks pool to '
        + 'the combined counts, not to the average of these two numbers.'} />

    <BlockPlot geometry={geometry} knots={stages.knots} sigmoid={showSigmoid ? sigmoid : null}
      caption={`The blocks after ${applied} merge${applied === 1 ? '' : 's'}`} />

    <LiveResult
      state={state}
      
      
      
      
      
      calculateInputs={next => {
        const nextStages = stagesOf(next);
        const position = Math.min(next.applied, nextStages.stages.length - 1);
        const merge = nextStages.merges[position] ?? null;
        return {
          outcome: merge ? `pair-${merge.index}` : 'none',
          value: merge ? merge.merged.mean : null,
          explain: merge
            ? `Blocks ${merge.left.start + 1}–${merge.left.end + 1} at ${round(merge.left.mean, 6)} over `
              + `${merge.left.weight} observations and ${merge.right.start + 1}–${merge.right.end + 1} at `
              + `${round(merge.right.mean, 6)} over ${merge.right.weight} violate the required order, so they pool `
              + `to ${(merge.left.total + merge.right.total)}/${merge.left.weight + merge.right.weight} = `
              + `${gradedText(merge.merged.mean)}. Averaging the two block probabilities instead would give `
              + `${gradedText((merge.left.mean + merge.right.mean) / 2)}, which is a different number whenever the `
              + 'two blocks hold different numbers of observations.'
            : 'Every adjacent pair already increases, so the monotone fit is finished. A pair whose two '
              + 'probabilities are exactly equal is not a violation and is left alone.',
        };
      }} />

    {revealed && <div className="cal-graded">
      <div className="cal-buttons">
        <button type="button" className="is-primary"
          disabled={!stages.merges[applied]}
          onClick={() => state.edit({ applied: applied + 1 })}>
          Apply this merge and ask again
        </button>
        <button type="button" onClick={() => setShowSigmoid(value => !value)}>
          {showSigmoid ? 'Hide' : 'Overlay'} the two-parameter sigmoid fit
        </button>
      </div>
      {nextMerge
        ? <Table caption="The merge for the current inputs"
          headings={['side', 'blocks', 'observations', 'positive', 'probability']}
          rows={[
            ['left', `${nextMerge.left.start + 1}–${nextMerge.left.end + 1}`,
              String(nextMerge.left.weight), String(nextMerge.left.total), round(nextMerge.left.mean, 9)],
            ['right', `${nextMerge.right.start + 1}–${nextMerge.right.end + 1}`,
              String(nextMerge.right.weight), String(nextMerge.right.total), round(nextMerge.right.mean, 9)],
            ['pooled', `${nextMerge.merged.start + 1}–${nextMerge.merged.end + 1}`,
              String(nextMerge.merged.weight), String(nextMerge.merged.total), gradedText(nextMerge.merged.mean)],
          ]}
          kind="calibration" />
        : <div>
          <Table caption="The finished monotone map, at its knots"
            headings={['score', 'observations', 'positive', 'fitted probability']}
            rows={stages.knots.map((knot, index) => [
              round(knot, 6), String(stages.weights[index]), String(stages.totals[index]),
              round(stages.fitted[index], 9),
            ])}
            kind="calibration"
            footnote={stages.knots.length === 1
              ? `There is one distinct score, so every observation belongs to one block. Its fitted probability is `
                + `${round(stages.fitted[0], 6)}; endpoint clipping returns that same value at every query score.`
              : 'These values are defined AT the knots. Between them a library must choose: ours interpolates '
              + 'linearly and clips outside the observed range, so the result need not be a staircase. Halfway '
              + `between the first two knots the interpolated value is `
              + `${round(isotonicPredict(stages.knots, stages.fitted, (stages.knots[0] + stages.knots[1]) / 2), 6)}.`} />
          {showSigmoid && <p className="cal-caption">
            {sigmoid.converged
              ? `A two-parameter sigmoid fitted to the same observations gives a ≈ ${round(sigmoid.a, 6)} and `
                + `b ≈ ${round(sigmoid.b, 6)}, so its probability at the first knot is `
                + `${round(1 / (1 + Math.exp(-(sigmoid.a * stages.knots[0] + sigmoid.b))), 6)} against the monotone `
                + `fit's ${round(stages.fitted[0], 6)}. Two assumptions, two different answers on the same evidence.`
              : `No sigmoid is fitted here: ${sigmoid.because}.`}
          </p>}
        </div>}
    </div>}
  </Investigation>;
}

/** Blocks as rectangles, raw group means as marks, and the fitted step path. */
function BlockPlot({ geometry, knots, sigmoid, caption }) {
  const { box } = geometry;
  return <PlotFrame geometry={{ ...geometry, ticks: null }} caption={caption}
    xLabel="observations in score order" yLabel="probability"
    titleText={caption}
    describe={`Rectangles at probability heights: `
      + `${geometry.blocks.map(block =>
        `a block spanning scores ${round(knots[block.start], 3)} to ${round(knots[block.end], 3)} at probability `
        + `${round(block.mean, 4)} over ${block.weight} observations`).join('; ')}.`}>
    {[0, 0.25, 0.5, 0.75, 1].map(value => {
      const y = geometry.scales.y(value);
      return <g key={value}>
        <line className="cal-gridline" x1={box.left} y1={y} x2={box.width - box.right} y2={y} />
        <text className="cal-small" x={box.left - 6} y={y + 3} textAnchor="end">{value}</text>
      </g>;
    })}
    {geometry.blocks.map(block => <rect key={`${block.start}-${block.end}`} className="cal-block"
      x={block.x1} y={block.y} width={Math.max(block.x2 - block.x1, 2)}
      height={Math.max(geometry.baseline - block.y, 1)} />)}
    {geometry.knots.map(knot => <g key={knot.index}>
      <circle className="cal-raw-mark" cx={knot.x} cy={knot.rawY} r={3} />
      <text className="cal-small" x={knot.x} y={geometry.baseline + 13} textAnchor="middle">
        {round(knot.knot, 2)}
      </text>
    </g>)}
    {sigmoid?.converged && <path className="cal-sigmoid-line"
      d={geometry.knots.map((knot, index) => {
        const probability = 1 / (1 + Math.exp(-(sigmoid.a * knot.knot + sigmoid.b)));
        return `${index === 0 ? 'M' : 'L'}${knot.x.toFixed(2)},${geometry.scales.y(probability).toFixed(2)}`;
      }).join(' ')} />}
  </PlotFrame>;
}

/* ================================ I3 · reserve a rank for the unknown answer */

const rankInitial = {
  scores: fixtures.labCalibrationScores.slice(),
  alpha: fixtures.defaultAlpha,
  /* Opens on the candidate whose leading score lands exactly ON the threshold,
     so the first question a learner meets is the one the weak comparison
     settles. */
  candidate: 1,
  mode: 'set',
  rotationScores: fixtures.rotationScores.slice(),
};

const SIZE_OUTCOMES = [
  ['empty', 'no class passes — the set is empty'],
  ['singleton', 'exactly one class passes'],
  ['partial', 'two or more classes, but not every class'],
  ['all', 'every class passes'],
];

const sizeOutcome = (size, classes) => {
  if (size === 0) return 'empty';
  if (size === 1) return 'singleton';
  return size === classes ? 'all' : 'partial';
};

export function RankLab() {
  const state = useInvestigation(rankInitial);
  const { draft } = state;
  const revealed = Boolean(state.result);
  const threshold = conformalThreshold(draft.scores, draft.alpha);
  const candidate = fixtures.candidateVectors[draft.candidate];
  /* The domain is read off the CARDS, never off the threshold: a domain that
     stretched to accommodate q would move the axis the moment the answer
     changed, and that is a leak drawn in geometry rather than printed in text.
     A finite q is always at most the largest card, so this always contains it. */
  const geometry = railGeometry(draft.scores, threshold, {
    domain: [0, Math.max(1, ...draft.scores)],
  });
  const rotation = rankRotation(draft.rotationScores, draft.alpha);

  const setScore = (index, value) => state.edit({
    scores: draft.scores.map((score, position) => (position === index ? value : score)),
  });

  /* Built from the applied state, not the draft: see investigation 1. */
  const from = state.active;
  const presets = [
    ['The nine cards worked through in section 5', { ...from, scores: fixtures.calibrationScores.slice() }],
    ['Raise the eighth smallest card to .75', {
      ...from, scores: [...fixtures.calibrationScores.slice(0, 7), 0.75, 0.9],
    }],
    ['Lower the smallest card to 0 — an exact null', {
      ...from, scores: [0, ...[...from.scores].sort((a, b) => a - b).slice(1)],
    }],
    ['Reverse the cards — an exact null', { ...from, scores: [...from.scores].reverse() }],
    ['Ask for 95% coverage instead', { ...from, alpha: 0.05 }],
    ['Make every card .2', { ...from, scores: from.scores.map(() => 0.2) }],
    ['Back to the nine cards this lab opened with', rankInitial],
  ];

  return <Investigation
    title="Investigation 3 — reserve a rank for the answer you have not seen"
    question={draft.mode === 'set'
      ? 'Nine calibration scores and a coverage target. Work out the rank, then decide what the candidate below '
        + 'receives — a single class, several, every class, or nothing at all.'
      : 'Hide one of the ten cards as the future example, threshold on the other nine, and ask whether the hidden '
        + 'one is covered. Then do that for every card in turn.'}
    role={{
      kind: 'constructed',
      text: 'Constructed scores. The threshold is a calibration-sample quantity: it is an order statistic of these '
        + 'particular cards and would be a different number on another sample. Nothing here is a coverage '
        + 'measurement, and nothing here reads the banknote experiment.',
    }}
    onReset={state.reset}>

    <div className="cal-controls is-wide">
      <Select label="Comparison to inspect" value={draft.mode}
        options={[
          ['set', 'the prediction set a candidate receives'],
          ['rotation', 'how many held-out cards a full rotation covers'],
        ]}
        onChange={value => state.edit({ mode: value })}
        hint="Switching this changes the inspected quantity and recomputes its comparison." />
      <NumberField label="alpha — one minus the coverage target" value={draft.alpha} {...controlSteps.alpha}
        min={controlSteps.alpha.minimum} max={controlSteps.alpha.maximum}
        onChange={value => state.edit({ alpha: value })}
        hint="Typed, not a slider: the rank is a step function of alpha and some steps are one thousandth wide." />
      {draft.mode === 'set' && <Select label="Candidate to score" value={String(draft.candidate)}
        options={fixtures.candidateVectors.map((vector, index) => [String(index), vector.name])}
        onChange={value => state.edit({ candidate: Number(value) })} />}
    </div>

    {draft.mode === 'set' && <>
      <div className="cal-controls">
        {draft.scores.map((score, index) => <NumberField key={index} label={`calibration card ${index + 1}`}
          value={score} {...controlSteps.conformalScore}
          min={controlSteps.conformalScore.minimum} max={controlSteps.conformalScore.maximum}
          onChange={value => setScore(index, value)} />)}
      </div>
      <div className="cal-buttons">
        <button type="button" disabled={draft.scores.length >= limits.calibrationScores.maximum}
          onClick={() => state.edit({ scores: [...draft.scores, 0.5] })}>Add a card</button>
        <button type="button" disabled={draft.scores.length <= limits.calibrationScores.minimum}
          onClick={() => state.edit({ scores: draft.scores.slice(0, -1) })}>Remove the last card</button>
        <span>
          {draft.scores.length} calibration cards, so the rank denominator is {draft.scores.length + 1}: one place
          is held for the example you have not seen.
        </span>
      </div>
      <div className="cal-presets">
        {presets.map(([label, setup]) => <button key={label} type="button"
          onClick={() => state.suggest(setup)}>{label}</button>)}
      </div>
      <p className="cal-caption">
        A suggested setup makes one change to the state that is currently <em>applied</em>, not to whatever is in
        the fields. That is what makes an exact null exactly null whenever you click it: the comparison is the change the button names. The calculation updates directly.
      </p>

      <RankRail geometry={geometry} scores={draft.scores} revealed={revealed} threshold={threshold} />

      <Table caption={`Candidate: ${candidate.name}`}
        headings={['class', 'probability', 'score 1 − p']}
        rows={candidate.probabilities.map((probability, index) => [
          fixtures.classNames[index], round(probability, 6), round(1 - probability, 6),
        ])}
        kind="calibration"
        footnote={'The probabilities and their scores are inputs. Which of them pass is the question below, and a '
          + 'class passes when its score is at most the threshold — equality included.'} />
    </>}

    {draft.mode === 'rotation' && <>
      <div className="cal-controls">
        {draft.rotationScores.map((score, index) => <NumberField key={index}
          label={`card ${index + 1}`} value={score} {...controlSteps.conformalScore}
          min={controlSteps.conformalScore.minimum} max={controlSteps.conformalScore.maximum}
          onChange={value => state.edit({
            rotationScores: draft.rotationScores.map((old, position) => (position === index ? value : old)),
          })} />)}
      </div>
      {/* Built from the APPLIED state, like every other suggested setup in the
          lesson. These two were inline rather than entries in a `presets`
          array, and the static guard written to forbid a draft-built preset
          scanned only `const presets = [...]` blocks — so it could not see
          them, passed, and the record read its green result as proof the class
          was closed. The guard now enumerates every `state.suggest` site. */}
      <div className="cal-presets">
        <button type="button" onClick={() => state.suggest({
          ...state.active, rotationScores: fixtures.tiedRotationScores.slice(),
        })}>Make every card .2 — all ties</button>
        <button type="button" onClick={() => state.suggest({
          ...state.active, rotationScores: fixtures.rotationScores.slice(),
        })}>Back to ten distinct cards</button>
      </div>
      <p className="cal-caption">
        A suggested setup makes one change to the state that is currently <em>applied</em>, not to whatever is in
        the fields. The calculation updates directly.
      </p>
      <p className="cal-caption">
        Each card takes its turn as the hidden future example. The other {draft.rotationScores.length - 1} become
        the calibration set, and the question is whether the hidden card's score is at most the threshold they
        produce. Every placement is equally likely — that is what exchangeability means here — so this is an exact
        count over a fixed multiset, not a simulation and not a statement about a test sample.
      </p>
    </>}

    <LiveResult
      state={state}
      
      
      
      
      
      calculateInputs={next => {
        if (next.mode === 'rotation') {
          const result = rankRotation(next.rotationScores, next.alpha);
          const distinct = new Set(next.rotationScores).size;
          return {
            outcome: result.covered === result.total ? 'all' : 'some',
            value: result.covered,
            /* The rank equivalence holds for DISTINCT scores. Stated
               unconditionally it was false in exactly the case the "make every
               card .2" button produces: under full ties all ten are covered
               while only eight can have combined rank at most eight. The next
               clause explained the ties correctly, so the two sentences
               contradicted each other with nothing reconciling the first. */
            explain: `${result.covered} of ${result.total} hidden cards are covered. Each rotation uses rank `
              + `${result.targetRank} of ${result.total - 1} cards. `
              + (distinct === result.total
                ? `With no two scores equal, a hidden card is covered exactly when its combined rank among all `
                  + `${result.total} is at most ${result.targetRank}. `
                : `With ties present that rank test is only one direction: a card whose combined rank exceeds `
                  + `${result.targetRank} can still be covered, because equality passes the weak comparison. `)
              + (distinct === 1
                ? 'With every score equal, the weak comparison includes every card: coverage is 1 even though the '
                  + 'target was lower. Ties can only enlarge coverage, which is why the familiar upper limit needs '
                  + 'a no-ties condition.'
                : 'This is an exact count over these particular cards. It is not an estimate, and it is not a '
                  + 'statement about any population.'),
          };
        }
        const own = conformalThreshold(next.scores, next.alpha);
        const vector = fixtures.candidateVectors[next.candidate].probabilities;
        const sets = classSets(vector, own.q);
        return {
          outcome: sizeOutcome(sets.size, vector.length),
          value: own.k,
          explain: `k = ceil((${next.scores.length} + 1) × (1 − ${next.alpha})) = ${own.k}. `
            + (own.finite
              ? `That names the ${own.k}th smallest of the ${own.n} cards, which is ${round(own.q, 6)}. `
                + `The candidate's scores are ${vector.map(value => round(1 - value, 6)).join(', ')}, so `
                + `${sets.size === 0 ? 'none of them is at most the threshold and the set is empty'
                  : `${sets.included.map((keep, index) => (keep ? fixtures.classNames[index] : null))
                    .filter(Boolean).join(' and ')} pass`}. `
              : `Rank ${own.k} is beyond the ${own.n} cards available, so the threshold is infinity and every `
                + 'candidate answer is included. Clipping the rank to the largest card would be a different '
                + 'procedure and would quietly discard the protection you asked for. ')
            + 'The comparison is weak, so a score exactly equal to the threshold passes.',
        };
      }} />

    {revealed && <div className="cal-graded">
      {draft.mode === 'set'
        ? <>
          <Table caption="The rank and threshold for the current inputs"
            headings={['quantity', 'value', 'what it is']}
            rows={[
              ['calibration cards n', String(threshold.n), 'the scores you can see'],
              ['rank denominator n + 1', String(threshold.n + 1), 'one place held for the unseen example'],
              ['rank k', String(threshold.k), 'ceil((n + 1)(1 − alpha)), in exact decimal arithmetic'],
              ['threshold q', threshold.finite ? gradedText(threshold.q) : 'infinity',
                threshold.finite ? `the ${threshold.k}th smallest card` : 'the whole label space'],
            ]}
            kind="calibration" />
          <div className="cal-chips">
            {classSets(candidate.probabilities, threshold.q).included.map((keep, index) => <Pill key={index}
              tone={keep ? 'in' : 'out'} mark={keep ? '✓' : '×'}>
              {fixtures.classNames[index]}: score {round(1 - candidate.probabilities[index], 6)}
              {keep ? ' ≤ ' : ' > '}{threshold.finite ? round(threshold.q, 6) : '∞'}
            </Pill>)}
          </div>
          <p className="cal-caption">
            A singleton here is not a label with {round(100 * (1 - draft.alpha), 3)}% probability of being correct.
            The guarantee is about the sampling procedure and this fixed score function, averaged over calibration
            sets and the next example — not about any one displayed set.
          </p>
        </>
        : <Table caption="Every rotation for the current inputs"
          headings={['hidden card', 'threshold from the other nine', 'covered?']}
          rows={rotation.rows.map(row => [
            round(row.held, 6),
            Number.isFinite(row.q) ? round(row.q, 6) : 'infinity',
            row.covered ? 'covered' : 'missed',
          ])}
          rowClass={index => (rotation.rows[index].covered ? undefined : 'is-muted')}
          kind="population"
          footnote={`${rotation.covered} of ${rotation.total} covered. This is an exact enumeration of a fixed `
            + 'multiset, so it carries no sampling error at all — and equally it says nothing about any other set '
            + 'of scores.'} />}
    </div>}
  </Investigation>;
}


function RankRail({ geometry, scores, revealed, threshold }) {
  const { box } = geometry;
  return <div>
    <PlotFrame geometry={{ ...geometry, ticks: null }} caption="The calibration cards, sorted"
      xLabel="nonconformity score" yLabel="" axes="x"
      titleText="The calibration cards on one axis"
      describe={`${scores.length} marks on an axis from ${geometry.scales.x.domain.join(' to ')} at scores `
        + `${[...scores].sort((a, b) => a - b).map(score => round(score, 3)).join(', ')}.`
        + (revealed
          ? threshold.finite
            ? ` A vertical line marks the threshold at ${round(threshold.q, 4)}.`
            : ' The threshold is infinity, so no line is drawn: it is not a point on this axis.'
          : ' The threshold appears as the inputs change.')}>
      <line className="cal-rail" x1={geometry.axis.x1} y1={geometry.axis.y}
        x2={geometry.axis.x2} y2={geometry.axis.y} />
      {/* Ticks come from the scale's own declared domain, so a card typed above
          1 relabels the axis instead of being drawn off the end of it. */}
      {[0, 0.25, 0.5, 0.75, 1].map(fraction => {
        const [low, high] = geometry.scales.x.domain;
        const value = low + fraction * (high - low);
        return <text key={fraction} className="cal-small"
          x={geometry.scales.x(value)} y={geometry.axis.y + 15} textAnchor="middle">
          {Number(value.toFixed(3))}
        </text>;
      })}
      {geometry.marks.map(mark => <g key={`${mark.rank}-${mark.score}`}>
        <circle className={`cal-rail-mark${revealed && mark.selected ? ' is-selected' : ''}`}
          cx={mark.x} cy={geometry.axis.y} r={revealed && mark.selected ? 5.5 : 4} />
        <text className="cal-small" x={revealed ? mark.labelX : mark.x} y={geometry.axis.y - 10} textAnchor="middle">{mark.rank}</text>
      </g>)}
      {revealed && geometry.markerX !== null && <line className="cal-threshold"
        x1={geometry.markerX} y1={box.top} x2={geometry.markerX} y2={geometry.axis.y + 6} />}
    </PlotFrame>
    <p className="cal-caption">
      Each mark is one calibration card; the small number above it is its rank from the smallest. The rank the
      procedure selects, and the threshold it names, update directly with the calibration cards and alpha.
      {revealed && !threshold.finite
        ? ' Here the threshold is infinity, so there is no line to draw: infinity is not a point just past the '
          + 'last card, it is the whole label space.'
        : ''}
    </p>
  </div>;
}

/* ============================== I4 · from a residual scale to a physical interval */

const intervalInitial = {
  pairs: fixtures.residuals.map((residual, index) => ({
    id: `c${index + 1}`, residual, localScale: fixtures.localScales[index],
  })),
  queries: fixtures.queries.map((query, index) => ({ ...query, id: `q${index + 1}` })),
  alpha: fixtures.intervalAlpha,
  branch: 'normalized',
  query: 0,
  cqrLower: fixtures.cqrBase.lower,
  cqrUpper: fixtures.cqrBase.upper,
  cqrScores: fixtures.cqrScores.slice(),
  cqrAlpha: fixtures.cqrAlpha,
};

/** Did only the local scales move between two states?
 *
 * Used to explain a null by the reason that actually produced it: under the
 * absolute and CQR scores the local scales are never read, so changing them is
 * a null because the procedure ignores the input, not because an order
 * statistic happened to hold still.
 */
function scalesOnlyChanged(next, current) {
  const withoutScales = state => JSON.stringify({
    ...state,
    pairs: state.pairs.map(pair => ({ ...pair, localScale: null })),
    queries: state.queries.map(query => ({ ...query, localScale: null })),
  });
  const scalesOf = state => JSON.stringify([
    state.pairs.map(pair => pair.localScale), state.queries.map(query => query.localScale),
  ]);
  return withoutScales(next) === withoutScales(current) && scalesOf(next) !== scalesOf(current);
}

/** The width of the selected query under the selected branch, or null for an empty set. */
function selectedWidth(draft) {
  if (draft.branch === 'cqr') {
    const threshold = conformalThreshold(draft.cqrScores, draft.cqrAlpha);
    const interval = cqrInterval(draft.cqrLower, draft.cqrUpper, threshold.q);
    return { width: interval.empty ? null : interval.width, interval, threshold };
  }
  const scores = draft.branch === 'normalized'
    ? draft.pairs.map(pair => pair.residual / pair.localScale)
    : draft.pairs.map(pair => pair.residual);
  const threshold = conformalThreshold(scores, draft.alpha);
  const query = draft.queries[Math.min(draft.query, draft.queries.length - 1)];
  const interval = draft.branch === 'normalized'
    ? normalizedInterval(query.centre, query.localScale, threshold.q)
    : normalizedInterval(query.centre, 1, threshold.q);
  return { width: interval.width, interval, threshold, query };
}

export function IntervalLab() {
  const state = useInvestigation(intervalInitial);
  const { draft, active } = state;
  const revealed = Boolean(state.result);
  const baseline = revealed ? state.previous : active;
  const baselineResult = selectedWidth({ ...baseline, branch: draft.branch, query: draft.query });
  const draftResult = selectedWidth(draft);

  const setPair = (index, update) => state.edit({
    pairs: draft.pairs.map((pair, position) => (position === index ? { ...pair, ...update } : pair)),
  });

  
  const from = active;
  const presets = [
    ['Change the last two residuals to 12 and 16', {
      ...from,
      pairs: from.pairs.map((pair, index) => ({ ...pair,
        residual: index === from.pairs.length - 2 ? 12 : index === from.pairs.length - 1 ? 16 : pair.residual,
      })),
    }],
    /* Offered only while every doubled scale is still a value the control would
       accept. A preset must obey the same bounds as a manual edit; doubling a
       scale of 8 to 16 would leave a field the learner could not type back. */
    ['Double every calibration AND query scale — an exact null', {
      ...from,
      pairs: from.pairs.map(pair => ({ ...pair, localScale: pair.localScale * 2 })),
      queries: from.queries.map(query => ({ ...query, localScale: query.localScale * 2 })),
    }, [...from.pairs, ...from.queries]
      .every(entry => entry.localScale * 2 <= controlSteps.localScale.maximum)],
    ['Set the first query centre to 14', {
      ...from,
      queries: from.queries.map((query, index) => (index === 0 ? { ...query, centre: 14 } : query)),
    }],
    ['CQR scores −2, −1, 0, 1, 3 instead', { ...from, branch: 'cqr', cqrScores: fixtures.cqrAlternativeScores.slice() }],
    ['CQR base interval 10 to 12 with the original shrinking scores', {
      ...from, branch: 'cqr', cqrLower: 10, cqrUpper: 12,
      cqrScores: fixtures.cqrScores.slice(), cqrAlpha: fixtures.cqrAlpha,
    }],
    ['Back to the nine original pairs', intervalInitial],
  ];

  const branchScores = draft.branch === 'normalized'
    ? draft.pairs.map(pair => pair.residual / pair.localScale)
    : draft.pairs.map(pair => pair.residual);

  return <Investigation
    title="Investigation 4 — turn a residual scale into a physical interval"
    question={'Will changing the calibration evidence move the threshold, the physical width, or neither? One of '
      + 'the setups below changes every number on screen and leaves every interval exactly where it was.'}
    role={{
      kind: 'constructed',
      text: 'Constructed residuals and scales in generic response units. The local scale is a heuristic already '
        + 'fixed by separate fitting; editing it here defines a new constructed score, not a refitted real model. '
        + 'The measured decibel intervals are a separate view and are not touched by anything here.',
    }}
    onReset={state.reset}>

    <div className="cal-controls is-wide">
      <Select label="Which score function" value={draft.branch}
        options={[
          ['absolute', 'absolute residual — one width for everyone'],
          ['normalized', 'residual divided by a local scale'],
          ['cqr', 'conformalized quantile regression'],
        ]}
        onChange={value => state.edit({ branch: value })}
        hint="Changing the score function changes the procedure and recomputes its threshold." />
      {draft.branch !== 'cqr' && <NumberField label="alpha" value={draft.alpha} {...controlSteps.alpha}
        min={controlSteps.alpha.minimum} max={controlSteps.alpha.maximum}
        onChange={value => state.edit({ alpha: value })} />}
      {draft.branch === 'cqr' && <NumberField label="alpha" value={draft.cqrAlpha} {...controlSteps.alpha}
        min={controlSteps.alpha.minimum} max={controlSteps.alpha.maximum}
        onChange={value => state.edit({ cqrAlpha: value })} />}
      {draft.branch !== 'cqr' && <Select label="Query to inspect" value={String(draft.query)}
        options={draft.queries.map((query, index) => [String(index), `${query.name} — centre ${query.centre}`])}
        onChange={value => state.edit({ query: Number(value) })} />}
    </div>

    {draft.branch !== 'cqr' ? <>
      <div className="cal-controls">
        {draft.pairs.map((pair, index) => <div key={pair.id} className="cal-card">
          <span className="cal-card-name"><b>{pair.id}</b></span>
          <NumberField label="residual, response units" value={pair.residual} {...controlSteps.residual}
            min={controlSteps.residual.minimum} max={controlSteps.residual.maximum}
            onChange={value => setPair(index, { residual: value })} />
          <NumberField label="local scale, response units" value={pair.localScale} {...controlSteps.localScale}
            min={controlSteps.localScale.minimum} max={controlSteps.localScale.maximum}
            onChange={value => setPair(index, { localScale: value })}
            hint="Strictly positive. A scale of zero makes the normalised score undefined and is refused." />
          <span className="cal-bin">
            normalised score {round(pair.residual / pair.localScale, 6)}
          </span>
        </div>)}
      </div>
      <div className="cal-controls is-wide">
        {draft.queries.map((query, index) => <div key={query.id} className="cal-card">
          <span className="cal-card-name"><b>{query.name}</b></span>
          <NumberField label="point prediction" value={query.centre} {...controlSteps.queryCentre}
            min={controlSteps.queryCentre.minimum} max={controlSteps.queryCentre.maximum}
            onChange={value => state.edit({
              queries: draft.queries.map((old, position) =>
                (position === index ? { ...old, centre: value } : old)),
            })} />
          <NumberField label="local scale here" value={query.localScale} {...controlSteps.localScale}
            min={controlSteps.localScale.minimum} max={controlSteps.localScale.maximum}
            onChange={value => state.edit({
              queries: draft.queries.map((old, position) =>
                (position === index ? { ...old, localScale: value } : old)),
            })} />
        </div>)}
      </div>
      <div className="cal-buttons">
        <button type="button" disabled={draft.queries.length >= limits.queries.maximum}
          onClick={() => state.edit({
            queries: [...draft.queries, {
              id: `q${draft.queries.length + 1}`, name: `query ${draft.queries.length + 1}`,
              centre: 15, localScale: 2,
            }],
          })}>Add a query</button>
        <button type="button" disabled={draft.queries.length <= limits.queries.minimum}
          onClick={() => state.edit({
            queries: draft.queries.slice(0, -1),
            query: Math.min(draft.query, draft.queries.length - 2),
          })}>Remove the last query</button>
        <span>Between {limits.queries.minimum} and {limits.queries.maximum} queries.</span>
      </div>
    </> : <div className="cal-controls is-wide">
      <NumberField label="initial lower endpoint L(x)" value={draft.cqrLower} {...controlSteps.signedScore}
        min={controlSteps.signedScore.minimum} max={controlSteps.signedScore.maximum}
        onChange={value => state.edit({ cqrLower: value })} />
      <NumberField label="initial upper endpoint U(x)" value={draft.cqrUpper} {...controlSteps.signedScore}
        min={controlSteps.signedScore.minimum} max={controlSteps.signedScore.maximum}
        onChange={value => state.edit({ cqrUpper: value })} />
      {draft.cqrScores.map((score, index) => <NumberField key={index}
        label={`CQR calibration score ${index + 1}`} value={score} {...controlSteps.signedScore}
        min={controlSteps.signedScore.minimum} max={controlSteps.signedScore.maximum}
        onChange={value => state.edit({
          cqrScores: draft.cqrScores.map((old, position) => (position === index ? value : old)),
        })} />)}
    </div>}

    <div className="cal-presets">
      {presets.map(([label, setup, enabled = true]) => <button key={label} type="button"
        disabled={!enabled}
        title={enabled ? undefined : 'Every scale would leave the range these fields accept.'}
        onClick={() => state.suggest(setup)}>{label}</button>)}
    </div>
    <p className="cal-caption">
      A suggested setup makes one change to the state that is currently <em>applied</em>, not to whatever is in
      the fields. That is what makes “an exact null” true whenever you click it: the comparison is
      the change the button names. The calculation updates directly.
    </p>

    {draft.branch !== 'cqr' && <BarRows
      caption={`The ${draft.branch === 'normalized' ? 'normalised' : 'absolute'} calibration scores, sorted`}
      rows={[...branchScores].sort((a, b) => a - b).map((score, index) => ({
        name: `rank ${index + 1}`, value: score,
        share: score / Math.max(...branchScores, 1),
      }))}
      unit={draft.branch === 'normalized' ? '(dimensionless)' : 'units'}
      describe={`Sorted calibration scores: ${[...branchScores].sort((a, b) => a - b)
        .map(score => round(score, 3)).join(', ')}.`} />}

    <p className="cal-caption">
      The comparison baseline is {draft.branch === 'cqr' ? 'the conformalised interval' : 'the interval for the selected query'}
      {' '}in the applied state, whose width is{' '}
      {ratio(baselineResult.width, 'the shrunken endpoints cross, so the set is empty and has no width')}
      {draft.branch === 'cqr' ? '' : ' response units'}.
      {' '}The threshold and the new width are the answer to the question below.
    </p>
    <KindTag kind="calibration"
      extra={'The threshold is an order statistic of these calibration scores. A different calibration sample '
        + 'would give a different threshold, and the interval it produces would move with it.'} />

    <LiveResult
      state={state}
      
      
      
      
      
      
      
      calculateInputs={(next, current) => {
        const after = selectedWidth(next);
        /* Same rule as investigation 1: the applied INPUTS, read under the
           question now being asked. Comparing a normalised width with a CQR
           width because the branch changed between them would be comparing two
           procedures and calling it a change in one. */
        const before = selectedWidth({ ...current, branch: next.branch, query: next.query });
        const outcome = after.width === Infinity
          ? (before.width === Infinity ? 'unchanged' : 'unbounded')
          : after.width === null || before.width === null
            ? changeDirection(after.width, before.width)
            : thresholdDirection(after.width, before.width);
        const thresholdMoved = thresholdDirection(after.threshold.q, before.threshold.q);
        return {
          outcome, value: after.width,
          explain: `The threshold went from ${before.threshold.finite ? gradedText(before.threshold.q) : 'infinity'} `
            + `to ${after.threshold.finite ? gradedText(after.threshold.q) : 'infinity'} (${thresholdMoved}), and the `
            + `width from ${before.width === null ? 'no width — the set was empty' : gradedText(before.width)} to `
            + `${after.width === null ? 'no width — the set is empty' : gradedText(after.width)}. `
            + (after.width === Infinity
              ? 'The requested rank exceeds the available calibration scores, so every real response is included. '
                + 'The whole line has unbounded width; it is not an empty set.'
              : before.width === Infinity
                ? 'The rank is now available among the calibration scores, so the whole line becomes a finite interval.'
              : thresholdMoved !== 'unchanged' && outcome === 'unchanged' && next.branch === 'normalized'
              ? 'The threshold moved and the physical interval did not: the normalised score is dimensionless, so '
                + 'scaling every scale by the same factor divides the threshold by it and multiplies the local '
                + 'half-width back again. Calibration evidence changes the learned threshold; a change of units '
                + 'does not change what is covered.'
              : outcome === 'unchanged'
                /* Name the cause that actually applies. "The threshold is an
                   order statistic" was printed for every unchanged verdict,
                   including the branches where the local scales are not read
                   at all — so an edit that changed nothing for a trivial
                   reason was explained by a subtle one. The outcome was right
                   and the explanation plausible, which is the shape of defect
                   this whole investigation is about. */
                ? (next.branch !== 'normalized' && scalesOnlyChanged(next, current)
                  ? `The ${next.branch === 'cqr' ? 'conformalized quantile' : 'absolute residual'} score does not `
                    + 'read the local scales at all, so changing them cannot move anything here. This is a null '
                    + 'for a duller reason than the normalised branch’s: not an invariance, just an input '
                    + 'the procedure ignores.'
                  : 'The threshold is an order statistic, so many numerical changes leave it exactly where it '
                    + 'was — only a change that crosses the selected rank moves it.')
                : next.branch === 'normalized'
                  ? 'The half-width is the threshold multiplied by the local scale at this query, so both of them '
                    + 'matter and they matter in different ways.'
                  : next.branch === 'absolute'
                    ? 'The absolute-residual interval has width twice its threshold, in response units.'
                    : 'The CQR endpoints are L − q and U + q. A negative q shrinks the interval, and crossed '
                      + 'adjusted endpoints give an empty set.'),
        };
      }} />

    {revealed && <div className="cal-graded">
      {draft.branch === 'cqr'
        ? <Table caption="The conformalised interval for the current inputs"
          headings={['quantity', 'value']}
          rows={[
            ['rank k', String(draftResult.threshold.k)],
            ['threshold q', draftResult.threshold.finite ? gradedText(draftResult.threshold.q) : 'infinity'],
            ['initial interval', `[${round(draft.cqrLower, 6)}, ${round(draft.cqrUpper, 6)}]`],
            ['conformalised interval', draftResult.interval.empty
              ? 'empty — the shrunken endpoints cross'
              : `[${round(draftResult.interval.lower, 6)}, ${round(draftResult.interval.upper, 6)}]`],
            ['width', draftResult.width === null ? 'no width' : gradedText(draftResult.width)],
          ]}
          kind="calibration"
          footnote={'A negative threshold shrinks an over-wide initial interval; the score is negative exactly when '
            + 'the response fell inside its initial interval. If the shrunken endpoints cross, the sublevel set is '
            + 'empty. That is a legitimate outcome of this construction, not a negative width.'} />
        : <>
          <Table caption="Every query under the threshold for the current inputs"
            headings={['query', 'centre', 'local scale', 'half-width', 'interval', 'width']}
            rows={draft.queries.map(query => {
              const interval = draft.branch === 'normalized'
                ? normalizedInterval(query.centre, query.localScale, draftResult.threshold.q)
                : normalizedInterval(query.centre, 1, draftResult.threshold.q);
              return [
                query.name, round(query.centre, 6),
                draft.branch === 'normalized' ? round(query.localScale, 6) : '1 (not used)',
                interval.unbounded ? 'unbounded' : round(interval.halfWidth, 6),
                interval.unbounded ? 'the whole line' : `[${round(interval.lower, 6)}, ${round(interval.upper, 6)}]`,
                interval.unbounded ? 'unbounded' : gradedText(interval.width),
              ];
            })}
            kind="calibration"
            footnote={`Rank ${draftResult.threshold.k} of ${draftResult.threshold.n} calibration scores at alpha `
              + `${draft.alpha}, giving a threshold of `
              + `${draftResult.threshold.finite ? round(draftResult.threshold.q, 9) : 'infinity'}. Under the `
              + 'absolute score every width is the same; under the normalised score the widths differ because the '
              + 'local scales do.'} />
          <IntervalStrip queries={draft.queries} threshold={draftResult.threshold} branch={draft.branch} />
        </>}
    </div>}
  </Investigation>;
}

/** The query intervals as horizontal segments, with the shared threshold marked. */
function IntervalStrip({ queries, threshold, branch }) {
  if (!threshold.finite) {
    return <p className="cal-note">
      The threshold is infinity, so every interval is the whole line. There is nothing to draw at a finite scale,
      and drawing a very wide bar instead would suggest a finite width that does not exist.
    </p>;
  }
  const rows = queries.map(query => {
    const interval = branch === 'normalized'
      ? normalizedInterval(query.centre, query.localScale, threshold.q)
      : normalizedInterval(query.centre, 1, threshold.q);
    return { id: query.id, name: query.name, centre: query.centre, ...interval };
  });
  const lowest = Math.min(...rows.map(row => row.lower));
  const highest = Math.max(...rows.map(row => row.upper));
  const pad = (highest - lowest) * 0.08 || 1;
  const width = 300;
  /* Room for the row name INSIDE the box, left-aligned. Anchoring it at the end
     just outside the bars put "easy case" a few units left of zero, where the
     viewBox trimmed it. */
  const left = 76;
  const right = 12;
  const map = value => left + ((value - (lowest - pad)) / ((highest + pad) - (lowest - pad))) * (width - left - right);
  const height = 34 + rows.length * 26;
  return <figure className="cal-figure">
    <figcaption>Each query's interval, drawn to scale in response units</figcaption>
    <svg className="cal-plot" viewBox={`0 0 ${width} ${height}`} role="img"
      style={{ maxWidth: `${width}px` }}
      aria-label={`Horizontal interval segments: ${rows.map(row =>
        `${row.name} from ${round(row.lower, 3)} to ${round(row.upper, 3)}, width ${round(row.width, 3)}`).join('; ')}.`}>
      {rows.map((row, index) => {
        const y = 18 + index * 26;
        return <g key={row.id}>
          <text className="cal-small" x={4} y={y + 3}>{row.name}</text>
          <line className="cal-interval" strokeWidth={3}
            x1={map(row.lower)} y1={y} x2={map(row.upper)} y2={y} />
          <line className="cal-cap" x1={map(row.lower)} y1={y - 5} x2={map(row.lower)} y2={y + 5} />
          <line className="cal-cap" x1={map(row.upper)} y1={y - 5} x2={map(row.upper)} y2={y + 5} />
          <circle className="cal-target" cx={map(row.centre)} cy={y} r={3} />
        </g>;
      })}
    </svg>
    <p className="cal-caption">
      Horizontal axis in response units. The dot is the point prediction and the bar is the interval.
    </p>
  </figure>;
}
