import { useState } from 'react';
import {
  CellStrip, Diagram, Investigation, Legend, NumberField, Plate, LiveResult, Select, Table,
  asInput, fixed, rentals, round, useInvestigation,
} from './TimeSeriesShared.jsx';
import {
  BASELINE_DESCRIPTIONS, BASELINE_RULES, RULE_LABELS, baselineForecasts, compareScores, donorAnswer,
  eligibilityAnswer, eligibilityRows, errorSummary, fixtures, forecastRequest, historyChartGeometry,
  requestAnswer, seasonalDonorIndex, shortDay, splitterGap, weekdayName,
} from '../../data/timeseries-models.js';
import { timeSeriesData } from '../../data/timeseries-data.js';



const series = timeSeriesData.series;
const SOURCE_COUNTS = series.counts;
const SOURCE_DATES = series.dates;

/* ============================================ I1 · follow a seasonal dependency */

const TOY_MIN = 0;
const TOY_MAX = 100;
const TOY_INITIAL = {
  history: [...fixtures.toy.history],
  future: [...fixtures.toy.future],
  period: 2,
  horizonCount: 4,
  selectedHorizon: 3,
};

/** Adding a constant to every value leaves every absolute error unchanged --
 *  UNLESS the drift rule's clip at zero bites, which is exactly why the null is
 *  never stated without that qualification. */
function shifted(state, amount) {
  return {
    ...state,
    history: state.history.map(value => value + amount),
    future: state.future.map(value => value + amount),
  };
}

export function SeasonalDonorLab() {
  const state = useInvestigation(TOY_INITIAL);
  const draft = state.draft;
  const shown = state.result;
  const period = Math.min(draft.period, draft.history.length);
  const rules = baselineForecasts({ history: draft.history, horizon: draft.horizonCount, period });
  const outcomes = Array.from({ length: draft.horizonCount }, (_unused, index) =>
    (index < draft.future.length ? draft.future[index] : null));
  const summaries = Object.fromEntries(BASELINE_RULES.map(rule => [rule, errorSummary(outcomes, rules[rule])]));
  const selected = Math.min(draft.selectedHorizon, draft.horizonCount);
  const donorIndex = seasonalDonorIndex({
    historyLength: draft.history.length, period, horizon: selected,
  });
  
  const currentCalculation = shown?.calculation ?? null;
  const scored = outcomes.filter(value => value !== null).length;
  const clipping = rules.driftClipped.some(Boolean);
  const canShiftFive = [...draft.history, ...draft.future].every(value => value <= TOY_MAX - 5);

  const width = 300;
  const cellWidth = 28;
  const dividerX = 8 + draft.history.length * cellWidth + 6;
  const strip = { historyY: 26, forecastY: 64, cellHeight: 20 };
  const forecastCount = draft.horizonCount;
  const drawWidth = Math.max(width, dividerX + forecastCount * cellWidth + 12);

  return <Investigation title="Investigation 1 — which observation feeds this forecast?"
    onReset={state.reset}
    question={'Edit the six historical values, choose a season length and a horizon count, then say which '
      + 'history position supplies the seasonal forecast at the horizon you select — and what number it gives.'}
    role={{ kind: 'constructed', text: 'A constructed six-day operating cycle and a constructed continuation. '
      + 'These are not rental counts; the real series is investigation 3.' }}
    note={'The four rules occupy separate rows with their own dependency. Horizons past the supplied outcomes '
      + 'are unscored, not zero, and the denominator is printed beside every mean.'}>

    <div className="ts-controls is-tight">
      {draft.history.map((value, index) => <NumberField key={index}
        label={`history ${index + 1}`} value={value} min={TOY_MIN} max={TOY_MAX}
        onChange={next => state.edit({ history: draft.history.map((old, slot) => (slot === index ? next : old)) })} />)}
    </div>
    <div className="ts-controls is-tight">
      {draft.future.map((value, index) => <NumberField key={index}
        label={`outcome h${index + 1}`} value={value} min={TOY_MIN} max={TOY_MAX}
        onChange={next => state.edit({ future: draft.future.map((old, slot) => (slot === index ? next : old)) })} />)}
    </div>
    <div className="ts-controls">
      <NumberField label="season length m" value={draft.period} min={1} max={draft.history.length}
        hint={`An integer no longer than the ${draft.history.length} observed values.`}
        onChange={next => state.edit({ period: next })} />
      <NumberField label="horizons requested" value={draft.horizonCount} min={1} max={8}
        hint={`Only ${draft.future.length} outcomes exist, so beyond that the horizons are unscored.`}
        onChange={next => state.edit({
          horizonCount: next, selectedHorizon: Math.min(draft.selectedHorizon, next),
        })} />
      <NumberField label="horizon to inspect" value={selected} min={1} max={draft.horizonCount}
        onChange={next => state.edit({ selectedHorizon: next })} />
    </div>
    <div className="ts-presets">
      <button type="button"
        onClick={() => state.suggest({ ...TOY_INITIAL })}>The starting cycle</button>
      <button type="button"
        onClick={() => state.suggest({ ...draft, history: [10, 20, 10, 20, 18, 22] })}>
        Change the fifth value to 18
      </button>
      <button type="button" disabled={Boolean(shown) || !canShiftFive}
        onClick={() => state.suggest(shifted(draft, 5))}>Add 5 to every value</button>
      <button type="button"
        onClick={() => state.suggest({ ...draft, history: [100, 80, 60, 40, 20, 0], period: 2 })}>
        A falling history, where drift clips at zero
      </button>
      <button type="button"
        onClick={() => state.suggest({ ...draft, horizonCount: 8, selectedHorizon: 7 })}>
        Ask for eight horizons
      </button>
    </div>

    {!canShiftFive && <p className="ts-note">Adding 5 needs every history and outcome value to be at most 95,
      so all values stay within this investigation's 0–100 range without changing the size of the shift.</p>}

    <LiveResult state={state} 
      
      
      
      
      calculateInputs={inputs => donorAnswer({
        history: inputs.history,
        period: Math.min(inputs.period, inputs.history.length),
        horizonCount: inputs.horizonCount,
        selectedHorizon: Math.min(inputs.selectedHorizon, inputs.horizonCount),
        future: inputs.future,
      })} />

    {shown && <Plate caption={`History on the left of the origin divider, forecasts on the right. The highlighted cell is the observation the seasonal rule copies at horizon ${Math.min(shown.inputs.selectedHorizon, shown.inputs.horizonCount)}, and the connector is that copy.`}>
      <Diagram kind="line" width={drawWidth} height={104}
        title="The seasonal donor and the forecast cell it feeds"
        describe={`History ${shown.inputs.history.join(', ')}. The seasonal forecast at horizon `
          + `${Math.min(shown.inputs.selectedHorizon, shown.inputs.horizonCount)} copies position `
          + `${currentCalculation.donorPosition}, value ${currentCalculation.value}.`}>
        {shown.inputs.history.map((value, index) => <g key={`h${index}`}>
          <rect className={`ts-cell${index === currentCalculation.donorIndex ? ' is-origin' : ' is-known'}`}
            x={8 + index * cellWidth} y={strip.historyY} width={cellWidth - 4} height={strip.cellHeight} />
          <text className="ts-small" x={8 + index * cellWidth + (cellWidth - 4) / 2} y={strip.historyY + 14}
            textAnchor="middle">{asInput(value)}</text>
          <text className="ts-small ts-muted" x={8 + index * cellWidth + (cellWidth - 4) / 2}
            y={strip.historyY - 5} textAnchor="middle">{index + 1}</text>
        </g>)}
        {/* ABOVE the divider, not below it. Below, the label sat one unit past
            the foot of the viewBox and collided with the "h1" caption under the
            first forecast cell; the layout inspector reported both. */}
        <text className="ts-small ts-strong" x={dividerX - 3} y={strip.historyY - 14}
          textAnchor="middle">origin</text>
        <line className="ts-cutoff" x1={dividerX - 3} y1={strip.historyY - 11}
          x2={dividerX - 3} y2={strip.forecastY + strip.cellHeight + 6} />
        {currentCalculation.rules.seasonal.map((value, index) => {
          const lead = index + 1;
          const inspected = lead === Math.min(shown.inputs.selectedHorizon, shown.inputs.horizonCount);
          return <g key={`f${index}`}>
            <rect className={`ts-cell${inspected ? ' is-target' : ''}`}
              x={dividerX + index * cellWidth} y={strip.forecastY} width={cellWidth - 4} height={strip.cellHeight} />
            <text className="ts-small" x={dividerX + index * cellWidth + (cellWidth - 4) / 2}
              y={strip.forecastY + 14} textAnchor="middle">{asInput(value)}</text>
            <text className="ts-small ts-muted" x={dividerX + index * cellWidth + (cellWidth - 4) / 2}
              y={strip.forecastY + strip.cellHeight + 10} textAnchor="middle">h{lead}</text>
          </g>;
        })}
        {/* The connector: from the donor cell to the inspected forecast cell.
            Both endpoints come from the model, so the line cannot point at a
            cell the rule did not use. */}
        <line className="ts-arrow"
          x1={8 + currentCalculation.donorIndex * cellWidth + (cellWidth - 4) / 2}
          y1={strip.historyY + strip.cellHeight}
          x2={dividerX + (Math.min(shown.inputs.selectedHorizon, shown.inputs.horizonCount) - 1) * cellWidth
            + (cellWidth - 4) / 2}
          y2={strip.forecastY} />
      </Diagram>
    </Plate>}

    {}
    {!shown && <p className="ts-caption">The seasonal index
      is <strong>T − m + ((h − 1) mod m)</strong>, counted from zero over a history of T values with season
      length m. Change the horizon and inspect the donor and forecast together. The four
      rules, their forecasts and their errors appear once you have.</p>}

    {shown && <>
      <Table caption={`All four rules on this history, season length ${period}, ${draft.horizonCount} horizons`}
        headings={['Rule', 'Forecasts', 'Depends on', 'MAE', 'Scored horizons']}
        wrap={[2]}
        rowClass={index => (BASELINE_RULES[index] === 'seasonal' ? 'is-leading' : undefined)}
        rows={BASELINE_RULES.map(rule => [
          RULE_LABELS[rule],
          rules[rule].map(value => asInput(Number(value.toFixed(4)))).join(', '),
          BASELINE_DESCRIPTIONS[rule],
          summaries[rule].mae === null ? 'no scored horizon' : fixed(summaries[rule].mae, 6),
          `${summaries[rule].denominator} of ${draft.horizonCount}`,
        ])}
        footnote={scored < draft.horizonCount
          ? `${draft.horizonCount - scored} horizon${draft.horizonCount - scored === 1 ? '' : 's'} beyond the `
            + `${draft.future.length} supplied outcomes are unscored. They are not zero, and the denominator `
            + 'above says so.'
          : 'Every requested horizon has an outcome, so every rule is scored on the same denominator.'} />

      <CellStrip label="The seasonal donor for each horizon"
        cells={rules.seasonalDonors.map((index, lead) => ({
          value: asInput(draft.history[index]),
          label: `h${lead + 1} ← pos ${index + 1}`,
          highlight: lead + 1 === selected,
        }))} />

      <p className="ts-caption">With T = {draft.history.length} and m = {period},
        horizon {selected} lands on position {donorIndex + 1}. Past one full cycle the rule repeats
        the <em>observed</em> block again; it never reads a future outcome to extend
        itself. {clipping
          ? 'The drift rule is clipping at zero here, which is why the “add a constant to everything” null is '
            + 'stated with that qualification: under clipping the absolute errors do move.'
          : 'Adding the same constant to every history and outcome value would leave all four absolute errors '
            + 'unchanged here — but only because no drift prediction is being clipped at zero.'}</p>
    </>}
  </Investigation>;
}

/* ================================ I2 · admit a training row */

const ELIGIBILITY_INITIAL = {
  cutoff: 12,
  horizon: 3,
  delay: 2,
  inspected: 8,
  targetValue: 40,
};
const OFFERED_ORIGINS = fixtures.eligibility.origins;
const NONE_QUALIFY = 'none-qualify';

const describeSelection = chosen => (chosen.length === 0
  ? NONE_QUALIFY
  : [...chosen].sort((left, right) => left - right).join(','));
const readSelection = choice => (choice === '' || choice === NONE_QUALIFY
  ? []
  : choice.split(',').map(Number));

export function EligibilityLab() {
  const state = useInvestigation(ELIGIBILITY_INITIAL);
  const draft = state.draft;
  const shown = state.result;
  const rows = eligibilityRows({
    origins: OFFERED_ORIGINS, cutoff: draft.cutoff, horizon: draft.horizon, delay: draft.delay,
  });
  const inspectedRow = rows.find(row => row.origin === draft.inspected) ?? rows[0];
  const chosen = rows.filter(row => row.eligible).map(row => row.origin);
  const committedRows = shown
    ? eligibilityRows({
      origins: OFFERED_ORIGINS, cutoff: shown.inputs.cutoff, horizon: shown.inputs.horizon,
      delay: shown.inputs.delay,
    })
    : null;
  const splitter = splitterGap({ horizon: draft.horizon, delay: draft.delay });
  const axisDays = Math.max(21, ...rows.map(row => row.labelArrival), draft.cutoff) + 1;
  const width = 300;
  const inset = 16;
  const rowHeight = 13;
  const height = 26 + rows.length * rowHeight + 24;
  const scale = value => inset + (value / (axisDays - 1)) * (width - 2 * inset);



  return <Investigation title="Investigation 2 — which rows may this fit contain?"
    onReset={state.reset}
    question={'Move the cutoff, the horizon or the reporting delay, then tick the complete set of offered '
      + 'origins whose rows may enter a fit performed at that cutoff. Tick nothing and choose “none qualify” '
      + 'if that is your answer — an empty eligible set is a result, not a broken state.'}
    role={{ kind: 'constructed', text: 'A constructed clock in whole days. No model is fitted and no outcome '
      + 'is used: this activity is entirely about what had arrived by when.' }}
    note={`Offered origins are fixed at ${OFFERED_ORIGINS[0]}–${OFFERED_ORIGINS[OFFERED_ORIGINS.length - 1]}. `
      + 'It is acceptable for an edit to admit none of them.'}>

    <div className="ts-controls">
      <NumberField label="cutoff day" value={draft.cutoff} min={10} max={16}
        hint="The moment the fit is performed." onChange={next => state.edit({ cutoff: next })} />
      <NumberField label="horizon h" value={draft.horizon} min={1} max={4}
        hint="Days from a row's own origin to its target." onChange={next => state.edit({ horizon: next })} />
      <NumberField label="reporting delay d" value={draft.delay} min={0} max={3}
        hint="Days from a count's own date to its arrival." onChange={next => state.edit({ delay: next })} />
    </div>
    <div className="ts-controls">
      <Select label="inspect one row" value={String(draft.inspected)}
        options={OFFERED_ORIGINS.map(origin => [String(origin), `origin ${origin}`])}
        onChange={next => state.edit({ inspected: Number(next) })} />
      <NumberField label="that row's target count" value={draft.targetValue} min={0} max={200}
       
        hint="Changing the label's VALUE must not change whether the row is admitted."
        onChange={next => state.edit({ targetValue: next })} />
    </div>
    <div className="ts-presets">
      <button type="button"
        onClick={() => state.suggest({ ...ELIGIBILITY_INITIAL })}>Cutoff 12, h 3, delay 2</button>
      <button type="button"
        onClick={() => state.suggest({ ...draft, cutoff: 12, horizon: 1, delay: 0 })}>Cutoff 12, h 1, no delay</button>
      <button type="button"
        onClick={() => state.suggest({ ...draft, cutoff: 12, horizon: 3, delay: 0 })}>Cutoff 12, h 3, no delay</button>
      <button type="button"
        onClick={() => state.suggest({ ...draft, cutoff: 10, horizon: 3, delay: 2 })}>Cutoff 10, h 3, delay 2</button>
    </div>

    <LiveResult state={state} 
      
      /* The verdict names what the learner ticked, not the internal key. */
      
      
      
      
      
      calculateInputs={inputs => {
        const answer = eligibilityAnswer({
          origins: OFFERED_ORIGINS, cutoff: inputs.cutoff, horizon: inputs.horizon, delay: inputs.delay,
          selected: [],
        });
        const inspected = eligibilityRows({
          origins: [inputs.inspected], cutoff: inputs.cutoff, horizon: inputs.horizon, delay: inputs.delay,
        })[0];
        
        return {
          ...answer,
          outcome: describeSelection(answer.eligible),
          value: inspected.latestFeatureDay,
        };
      }}
      describe={'Eligibility is decided by dates alone. Changing the target count above moves no boundary.'} />

    {}
    {!shown && <p className="ts-caption">The rule is one inequality: a row at origin s may enter a fit
      performed at the cutoff when <strong>s + h + d ≤ cutoff</strong>. Apply it to each offered origin
      yourself and tick the ones that pass. The timeline, the per-row arithmetic and the inspected row's
      window appear once your set is recorded.</p>}

    {shown && <>
    <Plate caption={`Each offered origin, its feature window, its target day and the day that label arrives. The solid rule is the cutoff.`}>
      <Diagram kind="line" width={width} height={height}
        title="Every offered training row against the cutoff"
        describe={rows.map(row =>
          `Origin ${row.origin}: features ${row.features[0]} to ${row.latestFeatureDay}, target `
          + `${row.targetDay}, label arrives ${row.labelArrival}; `
          + `${row.eligible ? 'eligible' : 'not eligible'}`).join('. ')}>
        <text className="ts-small ts-muted" x={2} y={10}>row</text>
        {rows.map((row, index) => {
          const y = 22 + index * rowHeight;
          return <g key={row.origin}>
            <rect className={`ts-train-band${row.eligible ? '' : ' is-sliding'}`}
              x={scale(row.features[0])} y={y - 3.5}
              width={Math.max(scale(row.latestFeatureDay) - scale(row.features[0]), 1.5)} height={7} />
            <circle className="ts-label-mark" cx={scale(row.targetDay)} cy={y} r={2.8} />
            <line className={`ts-arrow${row.eligible ? '' : ' is-invalid'}`}
              x1={scale(row.targetDay)} y1={y} x2={scale(row.labelArrival)} y2={y} />
            <circle className={row.eligible ? 'ts-feature-mark' : 'ts-arrival-mark'}
              cx={scale(row.labelArrival)} cy={y} r={2.8} />
            <text className="ts-small ts-muted" x={inset - 3} y={y + 3} textAnchor="end">{row.origin}</text>
          </g>;
        })}
        <line className="ts-cutoff" x1={scale(draft.cutoff)} y1={12}
          x2={scale(draft.cutoff)} y2={22 + rows.length * rowHeight} />
        <text className="ts-small ts-strong" x={scale(draft.cutoff)} y={10} textAnchor="middle">cutoff</text>
        <line className="ts-axis" x1={inset} y1={height - 18} x2={width - inset} y2={height - 18} />
        {/* The final tick is added only when it clears the previous regular one
            by at least three days. Added unconditionally it produced "20" and
            "21" printed on top of each other whenever the axis ended one day
            after a multiple of four. */}
        {Array.from({ length: axisDays }, (_unused, day) => day)
          .filter(day => day % 4 === 0 || (day === axisDays - 1 && day % 4 >= 3))
          .map((day, index, list) => <g key={day}>
            <line className="ts-tick" x1={scale(day)} y1={height - 18} x2={scale(day)} y2={height - 14} />
            <text className="ts-small" x={scale(day)} y={height - 4}
              textAnchor={index === 0 ? 'start' : index === list.length - 1 ? 'end' : 'middle'}>{day}</text>
          </g>)}
      </Diagram>
    </Plate>
    <Legend entries={[
      { key: 'train', text: 'feature window of an eligible row' },
      { key: 'train-sliding', text: 'feature window of an ineligible row' },
      { key: 'target', text: 'target day' },
      { key: 'invalid', text: 'a label that has not arrived by the cutoff' },
    ]} />

    <Table caption={`Every offered row at cutoff ${draft.cutoff}, horizon ${draft.horizon}, delay ${draft.delay}`}
      headings={['Origin', 'Feature days', 'Those arrive', 'Target day', 'Label arrives', 's + h + d ≤ cutoff']}
      rowClass={index => (rows[index].eligible ? 'is-leading' : 'is-excluded')}
      rows={rows.map(row => [
        String(row.origin),
        `${row.features[0]}–${row.latestFeatureDay}`,
        `${row.featureArrivals[0]}–${row.featureArrivals[row.featureArrivals.length - 1]}`,
        String(row.targetDay),
        String(row.labelArrival),
        `${row.origin} + ${draft.horizon} + ${draft.delay} = ${row.labelArrival} `
          + `${row.eligible ? '≤' : '>'} ${draft.cutoff}`,
      ])}
      footnote={'The delay moves the feature window too. Raising d from 0 to 2 pushes every row’s latest '
        + 'available count two days earlier, as well as postponing its label.'} />

    <p className="ts-caption"><strong>The inspected row.</strong> At origin {inspectedRow.origin} the three
      latest available count days are {inspectedRow.features.join(', ')}, arriving
      on {inspectedRow.featureArrivals.join(', ')}. Its target is day {inspectedRow.targetDay} and that count
      arrives on day {inspectedRow.labelArrival}. Its label value is currently
      set to {asInput(draft.targetValue)} — change it and nothing in the table above moves, because admission
      is an information-time decision and not a favourable-label decision.</p>

    <p className="ts-caption"><strong>Why there is no mystery exclusion.</strong> The rule is one inequality,
      s + h + d ≤ cutoff. For a positive horizon that already implies s &lt; cutoff, so no separate
      “origin before the cutoff” condition is needed: with h = {draft.horizon} and d = {draft.delay} the
      admitted origins stop at {draft.cutoff - draft.horizon - draft.delay}.</p>
    </>}

    <p className="ts-caption"><strong>If you use an index-based splitter instead.</strong> With the first test
      origin at t and training ending at t − g − 1, enforcing the same inequality
      needs g ≥ h + d − 1. At h = {splitter.horizon} and d = {splitter.delay} that is
      g ≥ {splitter.minimumGap}, so the last training origin is t − {splitter.lastTrainOffset}. The minus one is
      that splitter's indexing convention and nothing more; the real experiment below compares dates directly
      rather than copying a gap out of a library example.</p>

    {shown && committedRows && <div className="ts-reveal">
      <dl>
        <dt>Committed inputs</dt>
        <dd>cutoff {shown.inputs.cutoff}, h {shown.inputs.horizon}, d {shown.inputs.delay}</dd>
        <dt>Eligible set</dt>
        <dd>{shown.calculation.eligible.length ? shown.calculation.eligible.join(', ') : 'empty'}</dd>
        <dt>You ticked</dt>
        <dd>{shown.calculation.chosen.length ? shown.calculation.chosen.join(', ') : 'none'}</dd>
        <dt>Missed</dt>
        <dd>{shown.calculation.missed.length ? shown.calculation.missed.join(', ') : 'none'}</dd>
        <dt>Wrongly included</dt>
        <dd>{shown.calculation.extra.length ? shown.calculation.extra.join(', ') : 'none'}</dd>
        <dt>Rows a fit could use</dt>
        <dd>{committedRows.filter(row => row.eligible).length} of {committedRows.length}</dd>
      </dl>
    </div>}
  </Investigation>;
}

/* ============================ I3 · issue a real forecast, then reveal it */

const DEVELOPMENT_ORIGINS = timeSeriesData.protocol.developmentOriginIndices;
const SHOWN_DAYS = 14;
const COUNT_MIN = 0;
const COUNT_MAX = 15000;
const REQUEST_INITIAL = {
  origin: 364,
  period: 7,
  historyOffset: 7,
  historyValue: SOURCE_COUNTS[364 - SHOWN_DAYS + 1 + 7],
  historyOn: false,
  futureHorizon: 1,
  futureValue: 0,
  futureOn: false,
};

/** Observed-count replacements may copy their visible source value. A future
 *  replacement is an explicit counterfactual, never a prefilled source outcome:
 *  an input value would reveal the outcome just as surely as an output table. */
function draftAtOrigin(origin, overrides = {}) {
  const historyOffset = overrides.historyOffset ?? REQUEST_INITIAL.historyOffset;
  const futureHorizon = overrides.futureHorizon ?? REQUEST_INITIAL.futureHorizon;
  return {
    ...REQUEST_INITIAL,
    origin,
    historyOffset,
    futureHorizon,
    historyValue: SOURCE_COUNTS[origin - SHOWN_DAYS + 1 + historyOffset],
    ...overrides,
  };
}

/** The series the request actually reads, with any counterfactual edits applied.
 *  Built fresh from the source each time, so an edit is never layered on top of
 *  a previous edit without the controls saying so. */
function effectiveSeries(inputs) {
  const counts = [...SOURCE_COUNTS];
  const historyIndex = inputs.origin - SHOWN_DAYS + 1 + inputs.historyOffset;
  if (inputs.historyOn) counts[historyIndex] = inputs.historyValue;
  if (inputs.futureOn) counts[inputs.origin + inputs.futureHorizon] = inputs.futureValue;
  return { counts, historyIndex, edited: inputs.historyOn || inputs.futureOn };
}

export function ForecastRequestLab() {
  const state = useInvestigation(REQUEST_INITIAL);
  const [showRidge, setShowRidge] = useState(false);
  const draft = state.draft;
  const shown = state.result;
  const live = effectiveSeries(draft);
  const chart = historyChartGeometry({
    dates: SOURCE_DATES, counts: live.counts, origin: draft.origin, period: draft.period, shown: SHOWN_DAYS,
  });
  const committed = shown ? effectiveSeries(shown.inputs) : null;
  const request = shown
    ? forecastRequest({
      counts: committed.counts, dates: SOURCE_DATES, origin: shown.inputs.origin,
      period: shown.inputs.period, horizon: 7,
    })
    : null;
  const ridge = timeSeriesData.ridgeEvidence.byOrigin[String(shown?.inputs.origin ?? draft.origin)];
  const originDate = SOURCE_DATES[draft.origin];

  return <Investigation title="Investigation 3 — compare a real forecast with its outcomes"
    onReset={() => { state.reset(); setShowRidge(false); }}
    question={'Pick a development origin and season length. Inspect the forecast donors, horizon-specific outputs and errors against the naive rule as you change the controls.'}
    role={{ kind: live.edited ? 'counterfactual' : 'measured',
      text: live.edited
        ? 'Counterfactual: at least one count has been replaced. Everything below is constructed arithmetic on '
          + 'an edited copy of a real series, not an observation and not a data correction.'
        : `Recorded daily rentals, ${timeSeriesData.provenance.name}, ${timeSeriesData.provenance.license}. `
          + 'Only development origins are offered; the final assessment period of figure 5 is not explorable '
          + 'here.' }}
    note={'Valid edits immediately update the forecast and its measured errors. A saved comparison '
      + 'baseline remains labelled with the inputs it used.'}>

    <div className="ts-controls is-wide">
      <Select label="forecast origin" value={String(draft.origin)}
        options={DEVELOPMENT_ORIGINS.map(origin =>
          [String(origin), `${SOURCE_DATES[origin]} · ${weekdayName(SOURCE_DATES[origin])} · index ${origin}`])}
        onChange={next => {
          const origin = Number(next);
          state.edit({
            origin,
            historyValue: SOURCE_COUNTS[origin - SHOWN_DAYS + 1 + draft.historyOffset],
          });
        }} />
      <Select label="season length" value={String(draft.period)}
        options={[['1', '1 day — this is the naive rule'], ['7', '7 days — one week'], ['14', '14 days — two weeks']]}
        onChange={next => state.edit({ period: Number(next) })} />
    </div>

    <div className="ts-controls is-wide">
      <Select label="replace an observed count" value={String(draft.historyOffset)}
        options={Array.from({ length: SHOWN_DAYS }, (_unused, offset) => {
          const index = draft.origin - SHOWN_DAYS + 1 + offset;
          return [String(offset), `${SOURCE_DATES[index]} (${rentals(SOURCE_COUNTS[index])})`];
        })}
        onChange={next => state.edit({
          historyOffset: Number(next),
          historyValue: SOURCE_COUNTS[draft.origin - SHOWN_DAYS + 1 + Number(next)],
        })} />
      <NumberField label="with this count" value={draft.historyValue} min={COUNT_MIN} max={COUNT_MAX}
        onChange={next => state.edit({ historyValue: next })} />
      <label className="ts-field">
        <span>apply the observed-count edit</span>
        <input type="checkbox" checked={draft.historyOn}
          onChange={() => state.edit({ historyOn: !draft.historyOn })} />
      </label>
    </div>

    <div className="ts-controls is-wide">
      <Select label="replace a future outcome" value={String(draft.futureHorizon)}
        options={Array.from({ length: 7 }, (_unused, index) => {
          const lead = index + 1;
          return [String(lead), `h${lead} · ${SOURCE_DATES[draft.origin + lead]}`];
        })}
        onChange={next => state.edit({
          futureHorizon: Number(next),
        })} />
      <NumberField label="with this count" value={draft.futureValue} min={COUNT_MIN} max={COUNT_MAX}
        hint="Your counterfactual replacement, initially zero; this is not the concealed source outcome."
        onChange={next => state.edit({ futureValue: next })} />
      <label className="ts-field">
        <span>apply the future-outcome edit</span>
        <input type="checkbox" checked={draft.futureOn}
          onChange={() => state.edit({ futureOn: !draft.futureOn })} />
      </label>
    </div>

    <div className="ts-presets">
      <button type="button"
        onClick={() => state.suggest(draftAtOrigin(364))}>The source series, 2011-12-31</button>
      <button type="button"
        onClick={() => state.suggest(draftAtOrigin(371))}>A week later, 2012-01-07</button>
      <button type="button"
        onClick={() => state.suggest(draftAtOrigin(476, { period: 14 }))}>
        2012-04-21 with a fortnight season
      </button>
      <button type="button"
        onClick={() => state.suggest(draftAtOrigin(364, {
          futureOn: true, futureHorizon: 1, futureValue: 3294,
        }))}>Set the first outcome to 3294</button>
      <button type="button"
        onClick={() => state.suggest(draftAtOrigin(364, {
          historyOn: true, historyOffset: 7, historyValue: 1754,
        }))}>
        Change the first donor day to 1754
      </button>
    </div>

    {}
    <Plate caption={shown
      ? `The ${SHOWN_DAYS} days up to the origin ${originDate}. Highlighted days are the ${shown.inputs.period}-day block the seasonal rule copied; the vertical rule is the cutoff. Nothing right of it is drawn.`
      : `The ${SHOWN_DAYS} days up to the origin ${originDate}. The vertical rule is the cutoff: nothing right of it is drawn, because nothing right of it has happened yet. Change the season length to follow the block copied into the forecast.`}>
      <Diagram kind="plot" width={chart.width} height={chart.height}
        title={`Recorded rentals for the ${SHOWN_DAYS} days ending ${originDate}`}
        describe={chart.points.map(point =>
          `${point.date}, ${point.weekday}: ${Math.round(point.count)} rentals`).join('; ')}>
        {chart.yTicks.map(tick => <g key={tick.value}>
          <line className="ts-grid" x1={chart.padding.left} y1={tick.y}
            x2={chart.width - chart.padding.right} y2={tick.y} />
          <text className="ts-small" x={chart.padding.left - 4} y={tick.y + 3} textAnchor="end">
            {tick.value.toLocaleString('en-US')}
          </text>
        </g>)}
        <polyline className="ts-series-line" style={{ stroke: '#d6dedb' }}
          points={chart.points.map(point => `${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(' ')} />
        {chart.points.map(point => {
          const marked = point.isDonor && Boolean(shown);
          return <circle key={point.index} className={marked ? 'ts-label-mark' : 'ts-day-dot'}
            cx={point.x} cy={point.y} r={marked ? 3.4 : 2.2} />;
        })}
        <line className="ts-cutoff" x1={chart.cutoff.x} y1={chart.padding.top - 6}
          x2={chart.cutoff.x} y2={chart.height - chart.padding.bottom} />
        <text className="ts-small ts-strong" x={chart.cutoff.x} y={chart.padding.top - 10}
          textAnchor="end">origin</text>
        <line className="ts-axis" x1={chart.padding.left} y1={chart.height - chart.padding.bottom}
          x2={chart.width - chart.padding.right} y2={chart.height - chart.padding.bottom} />
        <text className="ts-small" x={chart.padding.left} y={chart.height - 6} textAnchor="start">
          {shortDay(chart.points[0].date)}
        </text>
        <text className="ts-small" x={chart.width - chart.padding.right} y={chart.height - 6} textAnchor="end">
          {shortDay(originDate)}
        </text>
        <text className="ts-small ts-axis-title" x={2} y={10} textAnchor="start">rentals/day</text>
      </Diagram>
    </Plate>

    <LiveResult state={state} 
      
      
      
      
      
      calculateInputs={inputs => {
        const built = effectiveSeries(inputs);
        return requestAnswer({
          counts: built.counts, dates: SOURCE_DATES, origin: inputs.origin, period: inputs.period, horizon: 7,
        });
      }}
      describe={draft.period === 1
        ? 'A season length of one IS the naive rule, so the two are the same rule and must tie exactly.'
        : undefined} />

    {shown && request && <>
      <Table caption={`Issued at ${request.originDate} (${request.originWeekday}), season length ${shown.inputs.period}. The forecast, and the historical day each value was copied from.`}
        headings={['Horizon', 'Target date', 'Weekday', 'Forecast', 'Copied from', 'That day’s date']}
        rows={request.forecasts.seasonal.map((value, index) => [
          `h = ${index + 1}`,
          request.targetDates[index],
          request.targetWeekdays[index],
          rentals(value),
          `index ${request.donors[index].index}`,
          `${request.donors[index].date} (${request.donors[index].weekday})`,
        ])}
        footnote={`The naive rule forecasts ${rentals(request.forecasts.naive[0])} at every horizon: the `
          + 'origin’s own count, repeated.'} />

      {state.ready && <>
        {/* The naive FORECAST column was dropped and the forecast header
            shortened: the column held the same number on every row, it is
            stated in the sentence just above this table, and with it the table
            ran past its own box and clipped the last column at desktop. A table
            that has lost a column is not a table that scrolls. */}
        <Table caption={`What actually happened. One row per target day; “Season ${shown.inputs.period}” is that rule’s forecast, and the two error columns are the absolute misses it and the naive rule made.`}
          headings={['Horizon', 'Target date', 'Actual', `Season ${shown.inputs.period}`,
            'Its error', 'Naive error']}
          rows={request.summaries.seasonal.rows.map((row, index) => [
            `h = ${index + 1}`,
            request.targetDates[index],
            rentals(row.actual),
            rentals(row.predicted),
            fixed(row.absolute, 0),
            fixed(request.summaries.naive.rows[index].absolute, 0),
          ])}
          footnote={`Seasonal MAE ${fixed(request.summaries.seasonal.mae, 6)} against naive MAE `
            + `${fixed(request.summaries.naive.mae, 6)}, each over `
            + `${request.summaries.seasonal.denominator} scored horizons.`} />
        <div className="ts-reveal">
          <dl>
            <dt>Seasonal MAE</dt><dd>{fixed(request.summaries.seasonal.mae, 6)}</dd>
            <dt>Naive MAE</dt><dd>{fixed(request.summaries.naive.mae, 6)}</dd>
            <dt>Difference</dt>
            <dd>{fixed(request.summaries.seasonal.mae - request.summaries.naive.mae, 6)}</dd>
            <dt>Verdict</dt>
            <dd>{compareScores(request.summaries.seasonal.mae, request.summaries.naive.mae)}</dd>
          </dl>
        </div>
      </>}

      <div className="ts-buttons">
        <button type="button" className={showRidge ? 'is-selected' : undefined}
          onClick={() => setShowRidge(value => !value)}>
          {showRidge ? 'Hide the saved ridge evidence' : 'Show the saved ridge evidence'}
        </button>
      </div>
      {showRidge && <Table
        caption={committed.edited
          ? 'FROZEN ORIGINAL-SERIES EVIDENCE. These are the recorded expanding-ridge predictions for the '
            + 'unedited series at this origin. No model was refitted for your edit, and these numbers do not '
            + 'answer the counterfactual above.'
          : 'Saved expanding-ridge predictions recorded for this origin by the author experiment. No model is '
            + 'fitted in your browser.'}
        headings={['Horizon', 'Saved ridge forecast', `Season ${shown.inputs.period} forecast`]}
        rowClass={() => (committed.edited ? 'is-excluded' : undefined)}
        rows={(ridge ?? []).map((value, index) => [
          `h = ${index + 1}`, rentals(value), rentals(request.forecasts.seasonal[index]),
        ])} />}
    </>}

    <p className="ts-caption"><strong>Two edits that behave differently.</strong> Replacing a
      <em> future</em> outcome cannot change a forecast already issued from this origin — the forecast copies
      history, and the outcome is not in it — but it does change the error that outcome contributes. Replacing
      an <em>observed</em> count inside the copied block changes the forecast itself. Both are constructed
      explorations on a real series, not corrections to the data.</p>
  </Investigation>;
}
