import { useState } from 'react';
import {
  Diagram, Legend, Plate, PlotFrame, Series, Table, asInput, fixed,
} from './TimeSeriesShared.jsx';
import {
  BASELINE_DESCRIPTIONS, BASELINE_RULES, METHOD_LABELS, RULE_LABELS, arrivalGeometry, arrowStyleFor,
  baselineForecasts,
  developmentBarGeometry, errorSummary, fixtures, horizonBoundaryTable, horizonCurveGeometry,
  issueTimeGeometry, mislabelledTrace, recursiveTrace, seasonalNaiveCoincidence,
  staircaseGeometry, updatedTrace, weekdayName,
} from '../../data/timeseries-models.js';
import { timeSeriesData } from '../../data/timeseries-data.js';

/** Inline figures for the forecasting lesson.
 *
 * Every figure draws from `timeseries-models.js`, never from a literal copied
 * out of the prose, so the verifier checks the same numbers and the same
 * geometry the reader sees. A drawn split boundary is a mathematical claim
 * about what was knowable when; a boundary drawn even slightly wrong teaches
 * the opposite of this lesson, so none of them is placed by hand.
 *
 * No file outside TimeSeriesShared.jsx opens an `<svg>` tag. `Diagram` applies
 * the layout classes the stylesheet is scoped to, so a new figure cannot omit
 * them and put an unscoped rule back in front of KaTeX's radical SVGs.
 *
 * Figures carry no graded prediction. The three investigations own that
 * contract, and a figure that quietly answered one of their questions would
 * break it -- which is why figure 1 uses the first development week while
 * investigation 1 uses a constructed cycle, and why figure 5 shows the FINAL
 * period that investigation 3 never offers.
 */

/** An arrowhead drawn as an explicit polygon rather than through a `marker`
 *  reference. A marker id that does not resolve -- duplicated across two
 *  instances of the same figure, or a defs block that moved -- fails silently
 *  and leaves the line with no head at all. */
function Arrow({ x1, y1, x2, y2, kind = '', head = 4.2, inputKind, encoding }) {
  const direction = x2 >= x1 ? 1 : -1;
  const tip = x2;
  const shaftEnd = tip - direction * head;
  /* `data-input-kind` carries the arrow's own input to the browser verifier,
     which asserts that the drawn style is the one that input maps to. Without
     it, "the style matches the arrow" is a claim only a reader can check. */
  return <g>
    <line className={`ts-arrow${kind ? ` is-${kind}` : ''}`} data-input-kind={inputKind}
      data-encoding={encoding} x1={x1} y1={y1} x2={shaftEnd} y2={y2} />
    <polygon className={`ts-arrow-head${kind ? ` is-${kind}` : ''}`} data-input-kind={inputKind}
      points={`${tip},${y2} ${shaftEnd},${y2 - head * 0.72} ${shaftEnd},${y2 + head * 0.72}`} />
  </g>;
}

/** Hatching drawn as real line elements rather than a `url(#pattern)` fill: a
 *  pattern reference that does not resolve fails silently and leaves the band
 *  unpainted, which is how a figure's regions once became invisible while every
 *  offline assertion passed. */
function Hatch({ x, width, y, height, kind, step = 5, encoding }) {
  const marks = [];
  for (let offset = step / 2; offset < width; offset += step) marks.push(x + offset);
  return marks.map((position, index) => <line key={index} className={`ts-hatch-line is-${kind}`}
    data-encoding={encoding} x1={position} y1={y} x2={position} y2={y + height} />);
}

/* ================================ F1 · one target, two information boundaries */

export function IssueTimeFigure() {
  const geometry = issueTimeGeometry();
  const inset = geometry.inset;
  return <div className="ts-figure">
    <p className="ts-caption"><strong>Figure 1.</strong> One target day, two issue dates. Both lanes end on the
      same Saturday, {geometry.targetDate}, and the target square is identical in each. Only the shaded region
      differs: it is everything the forecaster had observed when that lane's forecast was issued. Same outcome,
      different issue dates, different available observations.</p>
    <p className="ts-role is-constructed" role="status">Real calendar dates from the first week of the
      experiment in section 5. No counts and no errors are attached here — this figure is about
      <em> when</em>, not about how well.</p>
    <Plate caption={`The arrow length is the horizon. Shaded cells were observed by the issue date; hatched cells had not happened yet at that moment.`}>
      <Diagram kind="line" width={geometry.width} height={geometry.height}
        title="Two fourteen-day calendar lanes ending on the same Saturday"
        describe={geometry.lanes.map(lane =>
          `${lane.label}: origin ${lane.originDate}, a ${lane.originWeekday}; horizon ${lane.horizon} days to `
          + `the target ${geometry.targetDate}; ${lane.known.toIndex + 1} of ${geometry.days.length} days `
          + 'observed at that moment').join('. ')}>
        {geometry.lanes.map(lane => <g key={lane.id}>
          {/* PAINT ORDER, corrected. The hatch used to be emitted FIRST, before
              the cell rects — and a `ts-cell` rect is drawn for the unobserved
              days too, with an opaque base fill. Every unobserved cell
              therefore painted over the hatching that cell existed to show:
              31 hatch lines in the DOM, every attribute correct, and a declared
              stroke that appeared nowhere on the canvas (measured 1.07:1
              against the fill it was supposed to mark).
              The bands and cells now go down first and the hatch goes on top,
              which is the order figure 4's identical helper already used. */}
          <rect className="ts-known-band" x={lane.known.x} y={lane.y}
            width={lane.known.width} height={geometry.rows.cellHeight} />
          {geometry.days.map(day => <rect key={day.index}
            data-encoding={day.isTarget ? 'target' : day.index === lane.originIndex ? 'origin'
              : day.index <= lane.originIndex ? 'known' : undefined}
            className={`ts-cell${day.isTarget ? ' is-target' : day.index === lane.originIndex ? ' is-origin'
              : day.index <= lane.originIndex ? ' is-known' : ''}`}
            x={day.x + geometry.cellGap / 2} y={lane.y}
            width={geometry.cellWidth - geometry.cellGap} height={geometry.rows.cellHeight} />)}
          <Hatch x={inset + (lane.originIndex + 1) * geometry.cellWidth} y={lane.y}
            width={(geometry.days.length - lane.originIndex - 1) * geometry.cellWidth}
            height={geometry.rows.cellHeight} kind="unknown" encoding="unknown" />
          {geometry.days.map(day => <text key={`letter-${day.index}`} className="ts-small"
            x={day.x + geometry.cellWidth / 2} y={lane.letterBaseline} textAnchor="middle">
            {day.weekday.slice(0, 1)}
          </text>)}
          <Arrow x1={lane.arrow.x1} y1={lane.arrow.y} x2={lane.arrow.x2} y2={lane.arrow.y} />
          {/* `h = 1`, not `horizon 1`: the one-cell arrow cannot be given
              horizontal clearance from a label several cells wide, so the label
              is kept narrow as well as dropped further below the arrow. The
              word "horizon" is in the caption and the table beneath. */}
          <text className="ts-small ts-strong" x={lane.arrow.labelX} y={lane.arrow.labelBaseline}
            textAnchor="middle">{`h = ${lane.horizon}`}</text>
        </g>)}
      </Diagram>
    </Plate>
    <Legend entries={[
      { key: 'known', text: 'observed by this lane’s issue date' },
      { key: 'unknown', text: 'not yet observed (hatched)' },
      { key: 'origin', text: 'the issue date itself' },
      { key: 'target', text: 'the target day, identical in both lanes' },
    ]} />
    {/* The target column was dropped: it held the same date in both rows, which
        is the figure's whole point and belongs in the caption rather than in a
        column that repeats itself and pushes the table past its own box. */}
    <Table caption={`The same request written out. Both lanes target ${geometry.targetDate}, a ${geometry.targetWeekday}.`}
      headings={['Lane', 'Issued', 'Horizon', 'Notation', 'Days observed']}
      wrap={[0, 3]}
      rows={geometry.lanes.map(lane => [
        lane.label,
        `${lane.originDate} · ${lane.originWeekday.slice(0, 3)}`,
        String(lane.horizon),
        lane.annotation,
        `${lane.known.toIndex + 1} of ${geometry.days.length}`,
      ])}
      footnote={'The vertical bar in the notation means “given information through this origin”. It is '
        + 'not division.'} />
    <p className="ts-caption"><strong>An observation can exist without being available.</strong> A count
      recorded on {geometry.arrival.eventDate} ({weekdayName(geometry.arrival.eventDate)}) that only arrives
      on {geometry.arrival.arrivalDate} is not a feature of a forecast issued
      on {geometry.arrival.decisionDate}: at that moment it has not turned
      up. {geometry.arrival.availableAtDecision
        ? 'Here it would already have arrived.'
        : 'Here it has not arrived, so the earlier event time does not help.'} The preceding lesson's
      event-time and availability-time contract is exactly this distinction; forecasting adds the third clock,
      the target's own date.</p>
  </div>;
}

/* ======================= F2 · a row has a past window and a future label */

export function ArrivalFigure() {
  const geometry = arrivalGeometry();
  return <div className="ts-figure">
    <p className="ts-caption"><strong>Figure 2.</strong> One training row under a two-day reporting delay.
      Issued at day {geometry.issueDay}, it predicts day {geometry.targetDay}, whose count arrives on
      day {geometry.labelArrival}. At the current cutoff, day {geometry.cutoff}, this looks like a complete row
      in a spreadsheet downloaded later and is an ineligible training row in the historical fit.</p>
    <p className="ts-role is-constructed" role="status">A constructed clock in whole days. Two decisions are
      shown independently: whether the row's historical features had arrived, and whether its label had.</p>
    <Plate caption={`Feature days ${geometry.features[0].index}–${geometry.latestFeatureDay} are the count days available at the issue time; each arrives ${geometry.delay} days after its own date. The label is a separate question.`}>
      <Diagram kind="line" width={geometry.width} height={geometry.height}
        title="A training row's feature window, its target and its label arrival, against a cutoff"
        describe={`Issued at day ${geometry.issueDay}. Features cover days ${geometry.features[0].index} to `
          + `${geometry.latestFeatureDay}, arriving days ${geometry.features[0].arrival} to `
          + `${geometry.latestFeatureDay + geometry.delay}. The target is day ${geometry.targetDay} and its `
          + `count arrives day ${geometry.labelArrival}. The cutoff is day ${geometry.cutoff}, so the `
          + `features are ${geometry.featuresLegitimate ? 'available' : 'not available'} and the label is `
          + `${geometry.labelAvailable ? 'available' : 'not available'}.`}>
        {/* 1 · the feature window */}
        {geometry.features.map(feature => <g key={`f${feature.index}`}>
          <rect className="ts-feature-mark" data-encoding="known" x={feature.x - 3.2}
            y={geometry.rows.featureY - 3.2} width={6.4} height={6.4} />
        </g>)}
        <text className="ts-small" x={geometry.features[0].x - 5} y={geometry.rows.featureLabelBaseline}
          textAnchor="start">features</text>
        {/* 2 · the target and its arrival */}
        <circle className="ts-label-mark" data-encoding="target" cx={geometry.marks.target.x}
          cy={geometry.rows.targetY} r={3.6} />
        <Arrow x1={geometry.marks.target.x} y1={geometry.rows.targetY}
          x2={geometry.marks.arrival.x} y2={geometry.rows.targetY} kind="invalid" encoding="invalid" />
        <circle className="ts-arrival-mark" cx={geometry.marks.arrival.x} cy={geometry.rows.targetY} r={3.6} />
        <text className="ts-small" x={geometry.marks.target.x - 6} y={geometry.rows.targetLabelBaseline}
          textAnchor="end">target</text>
        <text className="ts-small ts-invalid" x={geometry.marks.arrival.x + 6}
          y={geometry.rows.targetLabelBaseline} textAnchor="start">arrives</text>
        {/* 3 · the cutoff, spanning both rows so the comparison is visible */}
        <line className="ts-cutoff" x1={geometry.marks.cutoff.x} y1={geometry.rows.featureLabelBaseline - 8}
          x2={geometry.marks.cutoff.x} y2={geometry.rows.axisY} />
        <text className="ts-small ts-strong" x={geometry.marks.cutoff.x}
          y={geometry.rows.featureLabelBaseline - 11} textAnchor="middle">cutoff</text>
        {/* 4 · the issue day */}
        <line className="ts-boundary" data-encoding="origin" x1={geometry.marks.issue.x}
          y1={geometry.rows.featureY - 10} x2={geometry.marks.issue.x} y2={geometry.rows.axisY} />
        {/* 5 · one shared axis, below everything it indexes */}
        <line className="ts-axis" x1={geometry.inset} y1={geometry.rows.axisY}
          x2={geometry.width - geometry.inset} y2={geometry.rows.axisY} />
        {geometry.ticks.map((tick, index) => <g key={tick.value}>
          <line className="ts-tick" x1={tick.x} y1={geometry.rows.axisY} x2={tick.x} y2={geometry.rows.tickY} />
          <text className="ts-small" x={tick.x} y={geometry.rows.tickLabelBaseline}
            textAnchor={index === 0 ? 'start' : index === geometry.ticks.length - 1 ? 'end' : 'middle'}>
            {tick.value}
          </text>
        </g>)}
        {/* Its own band below the tick labels. End-anchored at the same x as
            the last tick, it needs the vertical separation the taller viewBox
            provides. */}
        <text className="ts-small ts-axis-title" x={geometry.width - geometry.inset}
          y={geometry.height - 4} textAnchor="end">day</text>
      </Diagram>
    </Plate>
    <Legend entries={[
      { key: 'known', text: 'feature day (square)' },
      { key: 'target', text: 'target day (circle)' },
      { key: 'invalid', text: 'the label’s journey to its arrival day' },
      { key: 'origin', text: 'the issue day' },
    ]} />
    <Table caption="Two decisions, taken separately"
      headings={['Question', 'Arithmetic', 'At this cutoff']}
      rows={[
        /* The relational operator is COMPUTED. It was hard-coded to ≤, so the
           label row rendered "13 ≤ 12" beside the answer "No" -- a false
           statement printed as the reason for a correct verdict. */
        ['Have the row’s features arrived?',
          `latest feature day ${geometry.latestFeatureDay} + delay ${geometry.delay} = `
          + `${geometry.latestFeatureDay + geometry.delay} `
          + `${geometry.featuresLegitimate ? '≤' : '>'} ${geometry.cutoff}`,
          geometry.featuresLegitimate ? 'Yes' : 'No'],
        ['Has the row’s label arrived?',
          `target ${geometry.targetDay} + delay ${geometry.delay} = ${geometry.labelArrival} `
          + `${geometry.labelAvailable ? '≤' : '>'} ${geometry.cutoff}`,
          geometry.labelAvailable ? 'Yes' : 'No'],
      ]}
      footnote={'The delay moves both boundaries. A pipeline that postpones only the label while still reading '
        + `day ${geometry.issueDay}'s own count as a feature is the same leak in a different costume.`} />
  </div>;
}

/* ================================== F3 · freeze the origin, or advance it */

const CONTINUATIONS = {
  original: { key: 'original', label: 'outcomes 12, 22, 12, 22', outcomes: fixtures.recursion.outcomes },
  changed: { key: 'changed', label: 'outcomes 22, 24, 26, 28', outcomes: fixtures.recursionChanged.outcomes },
};

export function ContinuationFigure() {
  const [which, setWhich] = useState('original');
  const outcomes = CONTINUATIONS[which].outcomes;
  const last = fixtures.recursion.lastObserved;
  const step = fixtures.recursion.step;
  const recursive = recursiveTrace({ lastObserved: last, step, horizon: outcomes.length });
  const updated = updatedTrace({ lastObserved: last, step, outcomes });
  const mislabelled = mislabelledTrace({ lastObserved: last, step, outcomes });
  const scores = {
    recursive: errorSummary(outcomes, recursive.predictions),
    updated: errorSummary(outcomes, updated.predictions),
  };
  const width = 300;
  const laneGap = 56;
  const left = 30;
  const cellWidth = (width - left - 12) / (outcomes.length + 1);
  const height = 3 * laneGap + 26;
  /* `kind` is the style for a LEGITIMATE arrow in that lane. The third lane
     marks only the arrows whose input postdates the origin it claims, which is
     every arrow but the first: the first reads the origin's own observation and
     is perfectly legal. The first draft passed `kind: 'invalid'` for that lane,
     so its legal first arrow was drawn red and dashed too -- the figure accused
     an arrow that is not at fault, which is the opposite of "mark those
     specific arrows invalid". */
  const lateInputs = mislabelled.rows.filter(row => row.illegalInput).length;
  /* NO BLANKET LANE STYLE. Each arrow's style comes from the input that arrow
     actually consumes, which the model layer has always computed as
     `row.inputKind` and which nothing used to read. Styling by lane drew lane
     1's first arrow as "input is this chain's own prediction" when its input is
     the observed 22 — a claim the chain cannot make about a prediction it had
     not yet produced — and lane 2's as "a newly observed outcome" when its
     input was already in hand at the origin. The first arrow is identical in
     all three lanes and is the fixed point that makes the comparison legible,
     so it is exactly the one that must not be mis-styled.
     `marksInvalid` is a separate axis: it overrides the style only for the
     arrows whose input postdates the origin the lane's claim names. */
  const lanes = [
    { key: 'recursive', title: 'fixed origin, recursive', trace: recursive, marksInvalid: false },
    { key: 'updated', title: 'advancing origin, updated', trace: updated, marksInvalid: false },
    { key: 'mislabelled', title: 'the same numbers, the first origin’s label', trace: mislabelled,
      marksInvalid: true },
  ];
  return <div className="ts-figure">
    <p className="ts-caption"><strong>Figure 3.</strong> One deliberately simple rule, next = last + {step},
      from a final observed count of {last}. The top chain feeds its own predictions forward behind a fixed
      cutoff. The middle chain receives a newly observed outcome at each advancing cutoff — a different, and
      legitimate, task. The bottom chain copies the middle chain's inputs while keeping the first chain's issue
      date; its marked arrows are the ones that make the claim wrong.</p>
    <p className="ts-role is-constructed" role="status">Exact constructed arithmetic on whole numbers. These are
      not rental counts and no model is fitted.</p>
    <div className="ts-buttons">
      <span>continuation</span>
      {Object.values(CONTINUATIONS).map(option => <button key={option.key} type="button"
        className={which === option.key ? 'is-selected' : undefined}
        onClick={() => setWhich(option.key)}>{option.label}</button>)}
    </div>
    <Plate caption={`Each lane starts from the same observed ${last}. Positions along a lane are horizons 1 to ${outcomes.length}, not dates.`}>
      <Diagram kind="line" width={width} height={height}
        title="Three forecast chains from the same last observation"
        describe={lanes.map(lane =>
          `${lane.title}: ${lane.trace.predictions.join(', ')}`).join('. ')
          + `. The outcomes are ${outcomes.join(', ')}.`}>
        {lanes.map((lane, order) => {
          const y = 18 + order * laneGap;
          return <g key={lane.key}>
            {/* x = 2, not 0: a start-anchored label has a small left side
                bearing, so at x = 0 its painted box began a pixel outside the
                viewBox at 320 px. */}
            <text className="ts-small ts-muted" x={2} y={y - 6}>{lane.title}</text>
            <circle className="ts-issue-mark" cx={left} cy={y + 8} r={3.4} />
            <text className="ts-small" x={left} y={y + 24} textAnchor="middle">{last}</text>
            {lane.trace.rows.map((row, index) => {
              const fromX = left + index * cellWidth;
              const toX = left + (index + 1) * cellWidth;
              const invalid = lane.marksInvalid && row.illegalInput;
              const style = invalid ? 'invalid' : arrowStyleFor(row.inputKind);
              return <g key={row.lead}>
                <Arrow x1={fromX + 5} y1={y + 8} x2={toX - 5} y2={y + 8} kind={style}
                  inputKind={row.inputKind} encoding={style} />
                <text className={`ts-small${invalid ? ' ts-invalid' : ''}`} x={toX} y={y + 12}
                  textAnchor="middle">{row.predicted}</text>
                <text className="ts-small ts-muted" x={toX} y={y + 24} textAnchor="middle">h{row.lead}</text>
              </g>;
            })}
          </g>;
        })}
      </Diagram>
    </Plate>
    <Legend entries={[
      { key: 'origin-observation', text: 'input is the origin’s own observed count' },
      { key: 'prediction-fed', text: 'input is this chain’s own prediction' },
      { key: 'observation-fed', text: 'input is a newly observed outcome' },
      { key: 'invalid', text: 'input the claimed issue date had not seen' },
    ]} />
    <p className="ts-caption">Every lane's first arrow carries the same thing — the origin's own observed
      count of {last} — so all three are drawn alike. What differs is what the SECOND arrow carries: the top
      chain feeds itself, the middle chain is handed a count that has since arrived, and the bottom chain does
      the same while claiming the top chain's issue date.</p>
    <Table caption={`Both chains scored against the same revealed outcomes, ${outcomes.join(', ')}`}
      headings={['Chain', 'Forecasts', 'MAE', 'Is the claim legitimate?']}
      wrap={[0, 3]}
      rowClass={index => (index === 2 ? 'is-invalid' : undefined)}
      rows={[
        ['Fixed origin, recursive', recursive.predictions.join(', '), fixed(scores.recursive.mae, 2),
          `Yes — it reads only day ${recursive.origin}`],
        ['Advancing origin, updated', updated.predictions.join(', '), fixed(scores.updated.mae, 2),
          'Yes, as a one-day forecast reissued each evening'],
        /* The count is of LATE INPUTS, not of audit violations. The audit also
           records a closure violation for the fit as a whole, so quoting its
           length here said "4 of its inputs" when three inputs are late. */
        ['Advancing numbers, fixed-origin claim', mislabelled.predictions.join(', '), fixed(scores.updated.mae, 2),
          `No — ${lateInputs} of its ${mislabelled.rows.length} inputs are dated after day ${mislabelled.origin}`],
      ]}
      footnote={which === 'original'
        ? 'Both scores are equal here. A protocol error does not have to improve a score to be a protocol '
          + 'error: what is wrong is the claim about when the forecast was issued, not the number.'
        : 'Now the updated chain scores better. The attractive number still belongs to the updated task, and '
          + 'reporting it as a four-day forecast issued once would be reporting a task nobody ran.'} />
    <p className="ts-caption">The third row's verdict is computed, not asserted: the audit walks every input the
      chain consumes and compares its date with the origin the claim names. Of
      its {mislabelled.rows.length} inputs, {lateInputs} are dated after day {mislabelled.origin} — the first
      one, which reads the origin's own count, is not at fault and is not marked. The recursive chain's audit
      finds {recursive.audit.violations.length} violations of any kind.</p>
  </div>;
}

/* ============================= F4 · a staircase of genuine rehearsals */

export function StaircaseFigure() {
  const [mode, setMode] = useState('expanding');
  const [horizon, setHorizon] = useState(1);
  const [step, setStep] = useState(0);
  const geometry = staircaseGeometry({ mode, horizon });
  const overview = geometry.overview;
  const detail = geometry.detail;
  const boundaries = horizonBoundaryTable({
    issueOrigin: fixtures.staircase.issueOrigins[step],
    firstTrainOrigin: fixtures.staircase.firstTrainOrigin,
    window: mode === 'sliding' ? fixtures.staircase.window : null,
  });
  const issued = overview.lanes.slice(0, step + 1).map(lane => lane.issueOrigin);
  return <div className="ts-figure">
    <p className="ts-caption"><strong>Figure 4.</strong> Three rehearsals of the real task, seven days apart,
      each issuing seven horizons. In <strong>expanding</strong> mode the left training edge stays fixed; in
      <strong> sliding</strong> mode it moves and the row count stops growing. The dashed boundary on each lane
      is that horizon's last eligible training origin, t − h — and it is in a different place for every
      horizon, which is why one overview cannot pretend horizon 1 and horizon 7 share a training matrix.</p>
    <p className="ts-role is-constructed" role="status">The schedule and its boundaries, drawn from the same
      index arithmetic the experiment runs. No model is fitted here; the measured results are in figure 5.</p>
    <div className="ts-buttons">
      <span>window</span>
      {[['expanding', 'expanding'], ['sliding', `sliding, last ${fixtures.staircase.window} rows`]]
        .map(([key, text]) => <button key={key} type="button"
          className={mode === key ? 'is-selected' : undefined} onClick={() => setMode(key)}>{text}</button>)}
    </div>
    <div className="ts-buttons">
      <span>inspect horizon</span>
      {[1, 4, 7].map(value => <button key={value} type="button"
        className={horizon === value ? 'is-selected' : undefined} onClick={() => setHorizon(value)}>
        h = {value}
      </button>)}
    </div>
    <div className="ts-buttons">
      <button type="button" onClick={() => setStep(0)} disabled={step === 0}>Back to the first origin</button>
      <button type="button" className="is-primary" disabled={step === overview.lanes.length - 1}
        onClick={() => setStep(value => Math.min(overview.lanes.length - 1, value + 1))}>
        Advance the origin
      </button>
      <span>{step + 1} of {overview.lanes.length} rehearsals issued.</span>
    </div>
    {/* PANEL A. Absolute day numbers, whole history. Its only job is the left
        edge — fixed in expanding mode, moving in sliding mode. No boundary and
        no targets are drawn at this scale: on a 365-day axis they occupy the
        last two percent of the width and drawing them there would claim a
        legibility this panel does not have. Panel B carries them. */}
    <Plate caption={`Panel A — the whole history each rehearsal trains on, in absolute day numbers. Watch the left edge: in ${mode} mode it ${mode === 'expanding' ? 'stays put while the bar grows to the right' : 'moves with the origin, so the bar keeps its length'}.`}>
      <Diagram kind="line" width={overview.width} height={overview.height}
        title={`The training region of each rehearsal in ${mode} mode`}
        describe={overview.lanes.map(lane =>
          `Origin ${lane.issueOrigin} trains on origins ${lane.trainStart} to ${lane.trainEnd}, `
          + `${lane.trainCount} rows of ${lane.eligibleCount} eligible`).join('. ')}>
        {overview.lanes.map((lane, order) => {
          const issued = order <= step;
          return <g key={lane.issueOrigin}>
            <rect className={`ts-train-band${mode === 'sliding' ? ' is-sliding' : ''}${issued ? '' : ' is-pending'}`}
              data-encoding={issued ? (mode === 'sliding' ? 'train-sliding' : 'train') : 'pending'}
              x={lane.train.x} y={lane.y} width={Math.max(lane.train.width, 1)}
              height={overview.rows.barHeight} />
            {issued && <circle className="ts-issue-mark" data-encoding="origin" cx={lane.issue.x}
              cy={lane.y + overview.rows.barHeight / 2} r={3} />}
            {/* The row count is NOT printed here. Placed just right of the
                drawing area it overflowed the viewBox; the caption below lists
                the counts in reflowing HTML, which is where a multi-line
                numeric readout belongs anyway. */}
            <text className={`ts-small${issued ? '' : ' ts-muted'}`} x={overview.inset - 3} y={lane.y + 8}
              textAnchor="end">{lane.issueOrigin}</text>
          </g>;
        })}
        <line className="ts-axis" x1={overview.inset} y1={overview.height - 20}
          x2={overview.width - overview.inset} y2={overview.height - 20} />
        {overview.ticks.map((tick, index) => <g key={tick.value}>
          <line className="ts-tick" x1={tick.x} y1={overview.height - 20} x2={tick.x} y2={overview.height - 16} />
          <text className="ts-small" x={tick.x} y={overview.height - 6}
            textAnchor={index === 0 ? 'start' : index === overview.ticks.length - 1 ? 'end' : 'middle'}>
            {tick.value}
          </text>
        </g>)}
      </Diagram>
    </Plate>
    <Legend entries={[
      { key: mode === 'sliding' ? 'train-sliding' : 'train', text: `${mode} training region` },
      { key: 'origin', text: 'the issue day' },
      { key: 'pending', text: 'a rehearsal not yet issued' },
    ]} />
    <p className="ts-caption">Rows used, in order: {overview.lanes.map(lane => lane.trainCount).join(', ')}.
      {mode === 'expanding'
        ? ' Expanding keeps one left edge and the count grows with every rehearsal.'
        : ` Sliding keeps the last ${geometry.window} rows, so the count stops growing while the eligible pool `
          + `(${overview.lanes.map(lane => lane.eligibleCount).join(', ')}) still does.`}</p>

    {/* PANEL B. The same rehearsals on an axis of days RELATIVE to each one's
        own origin, so all three share a scale and the boundary, the issue day
        and the seven targets are a readable distance apart. */}
    <Plate caption={`Panel B — the last ${detail.daysBefore} days and the next ${detail.daysAfter} of each rehearsal, measured from its own issue day. The dashed rule is that horizon's last eligible training origin, at −${horizon}; the seven marks after 0 are the target days.`}>
      <Diagram kind="line" width={detail.width} height={detail.height}
        title={`Each rehearsal's boundary, issue day and target days at horizon ${horizon}`}
        describe={detail.lanes.map(lane =>
          `Origin ${lane.issueOrigin}: training ends at day ${lane.boundary.day}, which is ${horizon} `
          + `day${horizon === 1 ? '' : 's'} before the issue day; the targets are days `
          + `${lane.targets[0].index} to ${lane.targets[lane.targets.length - 1].index}; the latest `
          + `observation any of its fits reads is day ${lane.maxObservationIndex}`).join('. ')}>
        {detail.lanes.map((lane, order) => {
          const issued = order <= step;
          return <g key={lane.issueOrigin}>
            <rect className={`ts-train-band${mode === 'sliding' ? ' is-sliding' : ''}${issued ? '' : ' is-pending'}`}
              data-encoding={issued ? (mode === 'sliding' ? 'train-sliding' : 'train') : 'pending'}
              x={lane.train.x} y={lane.y} width={Math.max(lane.train.width, 1)}
              height={detail.rows.barHeight} />
            {/* A ragged left edge, drawn as hatching, because the training data
                continues past this window and a flat edge would claim it began
                here. */}
            {lane.train.continuesLeft && <Hatch x={lane.train.x} width={7} y={lane.y}
              height={detail.rows.barHeight} kind="unknown" step={2.4} encoding="unknown" />}
            <line className="ts-boundary" data-encoding="coincident" x1={lane.boundary.x} y1={lane.y - 4}
              x2={lane.boundary.x} y2={lane.y + detail.rows.barHeight + 4} />
            {issued && <circle className="ts-issue-mark" data-encoding="origin" cx={lane.issue.x}
              cy={lane.y + detail.rows.barHeight / 2} r={3.2} />}
            {lane.targets.map(target => <rect key={target.offset}
              data-encoding={issued ? 'target' : 'pending'}
              className={`ts-target-mark${target.isInspected ? ' is-inspected' : ''}${issued ? '' : ' is-pending'}`}
              x={target.x - 2.2} y={lane.y - 1} width={4.4} height={detail.rows.targetHeight + 2} />)}
            <text className={`ts-small${issued ? '' : ' ts-muted'}`} x={detail.inset - 3} y={lane.y + 8}
              textAnchor="end">{lane.issueOrigin}</text>
          </g>;
        })}
        <line className="ts-axis" x1={detail.inset} y1={detail.height - detail.rows.axisOffset}
          x2={detail.width - detail.inset} y2={detail.height - detail.rows.axisOffset} />
        {detail.ticks.map((tick, index) => <g key={tick.value}>
          <line className="ts-tick" x1={tick.x} y1={detail.height - detail.rows.axisOffset}
            x2={tick.x} y2={detail.height - detail.rows.tickOffset} />
          <text className="ts-small" x={tick.x} y={detail.height - detail.rows.tickLabelOffset}
            textAnchor={index === 0 ? 'start' : index === detail.ticks.length - 1 ? 'end' : 'middle'}>
            {tick.value > 0 ? `+${tick.value}` : tick.value}
          </text>
        </g>)}
        {/* BELOW the tick labels, in its own band. Placed above them it landed
            on the third lane's target marks. */}
        <text className="ts-small ts-axis-title" x={detail.width - detail.inset}
          y={detail.height - detail.rows.titleOffset}
          textAnchor="end">days from this rehearsal&#8217;s own origin</text>
      </Diagram>
    </Plate>
    <Legend entries={[
      { key: mode === 'sliding' ? 'train-sliding' : 'train', text: 'eligible training rows' },
      { key: 'unknown', text: 'training continues past this window' },
      { key: 'origin', text: 'the issue day, at 0' },
      { key: 'target', text: 'the seven target days' },
      { key: 'coincident', text: `dashed: last eligible training origin, at −${horizon}` },
    ]} />
    <Table caption={`Every horizon's own boundary at issue origin ${fixtures.staircase.issueOrigins[step]}, in ${mode} mode`}
      headings={['Horizon', 'Last training origin', 'Rows used', 'Rows eligible', 'Target day', 'Latest day read']}
      rowClass={index => (index + 1 === horizon ? 'is-leading' : undefined)}
      rows={boundaries.map(row => [
        `h = ${row.horizon}`,
        String(row.lastTrainOrigin),
        String(row.trainCount),
        String(row.eligibleCount),
        String(row.targetIndex),
        String(row.maxObservationIndex),
      ])}
      footnote={'The last column is the claim this whole lesson rests on: across all seven fits the latest '
        + `observation any of them reads is day ${fixtures.staircase.issueOrigins[step]}, the issue origin `
        + 'itself, and never a day after it.'} />
    <p className="ts-caption"><strong>Forecasts already issued are not revised.</strong> Advancing the origin
      has issued forecasts at {issued.join(', ')}. Later observed values enter the next scheduled fit; they do
      not reach back and change a number already recorded at an earlier origin. What the final assessment locks
      is the update policy, not all future learning.</p>
    <p className="ts-caption"><strong>The other kind of assessment.</strong> A frozen-model assessment fits once,
      before the assessment period, and never refits: every lane above would keep the first lane's training
      region and only its issue day would move. That answers a different question — how one fitted model ages —
      and the two must not be reported under the same name.</p>
  </div>;
}

/* ====================== F5 · the measured comparison, at two resolutions */

export function ComparisonFigure() {
  const [view, setView] = useState('final');
  const development = timeSeriesData.development.map(row => ({ ...row, label: METHOD_LABELS[row.key] }));
  const bars = developmentBarGeometry({ rows: development });
  const finalSeries = timeSeriesData.final.map(entry => ({
    key: entry.key, label: METHOD_LABELS[entry.key], values: entry.maeByHorizon,
  }));
  const curves = horizonCurveGeometry({ series: finalSeries });
  const weekdays = timeSeriesData.protocol.targetWeekdays;
  /* A REAL history length, not a round number picked to illustrate. The first
     draft passed 400 and the page then read "copies history position 400 of
     400", which is a true statement about a history this experiment never
     has. The first final origin sees every day up to and including itself. */
  const finalHistoryLength = timeSeriesData.protocol.finalOriginIndices[0] + 1;
  const coincidence = seasonalNaiveCoincidence({
    historyLength: finalHistoryLength, period: 7, horizon: 7,
  });
  return <div className="ts-figure">
    <p className="ts-caption"><strong>Figure 5.</strong> The measured comparison at two resolutions. The
      development view selects a procedure from six candidates over {development[0].origins} origins; the final
      view assesses the selected procedure and the two predeclared baselines over
      {' '}{timeSeriesData.final[0].origins} later origins. These are different periods under the same weekly
      schedule, not repetitions of identical conditions.</p>
    <p className="ts-role is-measured" role="status">Measured results from the author experiment on the served
      dataset: {timeSeriesData.protocol.developmentRidgeFits} development fits
      and {timeSeriesData.protocol.finalRidgeFits} final fits. Development covers
      {' '}{timeSeriesData.protocol.developmentTargetRange.join(' to ')}; final
      covers {timeSeriesData.protocol.finalTargetRange.join(' to ')}.</p>
    <div className="ts-buttons">
      <span>view</span>
      {[['development', 'development: choose'], ['final', 'final: assess']].map(([key, text]) =>
        <button key={key} type="button" className={view === key ? 'is-selected' : undefined}
          onClick={() => setView(key)}>{text}</button>)}
    </div>

    {view === 'development' && <>
      <Plate caption={`Pooled MAE across ${development[0].forecasts} development forecasts, in rentals per day. Bar length runs from a true zero, so twice the bar is twice the error.`}>
        <Diagram kind="plot" width={bars.width} height={bars.height}
          title="Six candidate procedures compared on the development origins"
          describe={development.map(row => `${row.label}: MAE ${row.mae.toFixed(2)} rentals per day`).join('; ')}>
          {bars.rows.map(row => <g key={row.key}>
            <text className="ts-small" x={0} y={row.y + bars.barHeight - 3}>{row.label}</text>
            <rect className="ts-bar-track" x={bars.zeroX} y={row.y}
              width={bars.width - 6 - bars.zeroX} height={bars.barHeight} />
            <rect className={`ts-bar${row.selected ? ' is-selected' : ''}`} x={bars.zeroX} y={row.y}
              width={Math.max(row.barWidth, 1)} height={bars.barHeight} />
          </g>)}
          <line className="ts-axis" x1={bars.zeroX} y1={0} x2={bars.zeroX} y2={bars.height - 16} />
          <text className="ts-small" x={bars.zeroX} y={bars.height - 4} textAnchor="start">0</text>
          <text className="ts-small ts-axis-title" x={bars.width - 6} y={bars.height - 4} textAnchor="end">
            MAE, rentals/day
          </text>
        </Diagram>
      </Plate>
      <Table caption="Development results, pooled across every origin and horizon"
        headings={['Candidate', 'MAE, rentals/day', 'RMSE, rentals/day', 'Origins', 'Forecasts']}
        rowClass={index => (bars.selectedKeys.includes(development[index].key) ? 'is-leading' : undefined)}
        rows={development.map(row => [
          row.label, fixed(row.mae, 2), fixed(row.rmse, 2), String(row.origins), String(row.forecasts),
        ])}
        footnote={`Selected by the declared rule: lowest development MAE. That is `
          + `${METHOD_LABELS[timeSeriesData.selectedMethod]}. The final period plays no part in this choice.`} />
    </>}

    {view === 'final' && <>
      <PlotFrame geometry={curves} xLabel="horizon, days ahead" yLabel="MAE, rentals/day"
        xTickText={value => `${value}`}
        yTickText={value => value.toLocaleString('en-US')}
        title="Final MAE by horizon for the selected procedure and the two predeclared baselines"
        describe={finalSeries.map(entry =>
          `${entry.label}: ${entry.values.map(value => value.toFixed(2)).join(', ')}`).join('. ')
          + ` At horizon 7 the naive and seven-day seasonal rules are the same rule, so their errors are equal.`}
        caption={`Horizon 1 is ${weekdays[0]} and horizon 7 is ${weekdays[6]}, because every origin is a Saturday. Horizon and weekday are tied together here, so this is not a clean picture of difficulty against distance.`}>
        {curves.series.map(entry => <Series key={entry.key} points={entry.points}
          className={`is-${entry.key}`} encoding={entry.key}
          marker={entry.key === 'naive' ? 'square' : entry.key === 'seasonal' ? 'triangle' : 'dot'} />)}
        {/* A point two methods share is announced rather than left hidden under
            whichever mark happened to be painted last. */}
        {curves.coincidences.map(point => <circle key={point.horizon} className="ts-coincident"
          data-encoding="coincident" cx={curves.x(point.horizon)} cy={curves.y(point.value)} r={5.4} />)}
      </PlotFrame>
      <Legend entries={[
        ...finalSeries.map(entry => ({ key: entry.key, text: entry.label })),
        { key: 'coincident', text: 'a point two rules share' },
      ]} />
      <Table caption="Final MAE by horizon, rentals per day, with the winning rule at each"
        headings={['Horizon', 'Weekday', ...finalSeries.map(entry => METHOD_LABELS[entry.key]), 'Lowest']}
        wrap={[5]}
        rows={curves.winners.map(winner => [
          `h = ${winner.horizon}`,
          weekdays[winner.horizon - 1],
          ...finalSeries.map(entry => fixed(entry.values[winner.horizon - 1], 2)),
          /* EVERY method attaining the minimum, never the first one found. A
             tie reported as a single winner is exactly how a correct answer
             gets graded wrong. */
          winner.keys.map(key => METHOD_LABELS[key]).join(' and '),
        ])}
        footnote={'At horizon 7 the naive and seven-day seasonal rules are identical: a seven-day season at '
          + `horizon 7 copies history position ${coincidence.donor + 1} of ${coincidence.lastIndex + 1}, which `
          + 'is the origin’s own count, and that is exactly what the naive rule repeats. Their equal '
          + 'errors are required by the mechanism, not a coincidental tie.'} />
      <Table caption="Final results pooled across all origins and horizons"
        headings={['Method', 'MAE, rentals/day', 'RMSE, rentals/day', 'Forecasts']}
        rows={timeSeriesData.final.map(entry => [
          METHOD_LABELS[entry.key], fixed(entry.mae, 2), fixed(entry.rmse, 2), String(entry.forecasts),
        ])}
        footnote={'The aggregate winner does not win every horizon, and no confidence band is drawn: '
          + `${timeSeriesData.final[0].forecasts} forecasts from adjacent weeks of one system are not `
          + 'independent replicates.'} />
    </>}
  </div>;
}

/* ---- a small shared readout the prose uses in more than one place --------- */

export function BaselineSummary({ history, horizon, period, future }) {
  const rules = baselineForecasts({ history, horizon, period });
  const summaries = Object.fromEntries(BASELINE_RULES.map(rule => [rule, errorSummary(future, rules[rule])]));
  return <Table caption={`Four rules on the history ${history.join(', ')}, season length ${period}`}
    headings={['Rule', `Forecasts, horizons 1–${horizon}`, 'What it carries forward',
      future ? 'MAE' : 'Scored']}
    rows={BASELINE_RULES.map(rule => [
      RULE_LABELS[rule],
      rules[rule].map(value => asInput(Number(value.toFixed(4)))).join(', '),
      BASELINE_DESCRIPTIONS[rule],
      summaries[rule].mae === null ? 'no scored horizon' : fixed(summaries[rule].mae, 4),
    ])} />;
}
