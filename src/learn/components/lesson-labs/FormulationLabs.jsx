import { useState } from 'react';
import {
  Drawing, Figure, Investigation, Legend, NumberField, LiveResult, Readouts, Select, Table,
  asInput, fixed, probabilityText, round, useInvestigation,
} from './FormulationShared.jsx';
import {
  latestKnown, membershipChange, movementOf, policyOrder, rankByScore, recordKey, recordLabel,
  selectionMetrics, timelineFixture, timelineGeometry, translateValues,
  unchangedTolerance, withArrival,
} from '../../data/formulation-models.js';
import { formulationData } from '../../data/formulation-data.js';



const stateKey = state => JSON.stringify(state);

/* ============================================================ investigation 1 */

const TIMELINE_PRESETS = [
  /* Named for sensor A's delayed record, and it has to move THAT record
     whatever the entity selector is set to. `withArrival` matches on the
     QUERIED entity, so with the selector on sensor B this preset silently
     matched B's event-4 record -- whose arrival is already 4 -- and left A's
     at 8. The button carrying the destination note's whole point was a no-op
     one click away, and this investigation passes no `requireChange`, so
     nothing caught it. */
  ['Let sensor A’s delayed event arrive at time 4 instead',
    state => withArrival(state, { entity: 'sensor_A', event: 4, version: 1, available: 4 })],
  ['Move the cutoff and the age limit to 7',
    state => ({ ...state, cutoff: 7, maximumAge: 7 })],
  ['Move the cutoff and the age limit to 9',
    state => ({ ...state, cutoff: 9, maximumAge: 9 })],
  ['Tighten the age limit to 2',
    state => ({ ...state, maximumAge: 2 })],
  ['Change the other sensor’s values to 88',
    state => ({ ...state, records: state.records.map(record => record.entity === state.entity
      ? record : { ...record, value: 88 }) })],
  ['Add 5 to every value',
    state => translateValues(state, 5), state => state.records.some(record => record.value > 95)],
  ['Ask about the other sensor instead',
    state => ({ ...state, entity: state.entity === 'sensor_A' ? 'sensor_B' : 'sensor_A' })],
];

export function TimelineLab() {
  const state = useInvestigation(timelineFixture, stateKey);
  const draft = state.draft;
  const shown = state.result;

  const outOfOrder = draft.records.find(record => record.available < record.event);
  const problem = outOfOrder
    ? `${recordLabel(outOfOrder)} would arrive at ${outOfOrder.available}, before the event it measures `
      + `happened at ${outOfOrder.event}. A value cannot be available before it exists; move the arrival to `
      + `${outOfOrder.event} or later.`
    : null;

  
  const geometry = problem ? null : timelineGeometry({
    records: draft.records, entity: draft.entity, cutoff: draft.cutoff, maximumAge: draft.maximumAge,
  });
  const committedGeometry = shown && shown.calculation.lookup
    ? timelineGeometry({
      records: shown.inputs.records, entity: shown.inputs.entity,
      cutoff: shown.inputs.cutoff, maximumAge: shown.inputs.maximumAge,
    })
    : null;

  const calculateInputs = inputs => {
    const lookup = latestKnown(inputs);
    const newestEvent = inputs.records
      .filter(record => record.entity === inputs.entity && record.event <= inputs.cutoff)
      .reduce((best, record) => (best === null || record.event > best.event ? record : best), null);
    const newestIsSelected = lookup.selected !== null && newestEvent !== null
      && lookup.selected.record.event === newestEvent.event;
    return {
      outcome: lookup.selectedKey ?? 'none',
      value: lookup.value,
      lookup,
      newestEvent,
      explain: lookup.selected === null
        ? `Nothing qualifies at cutoff ${inputs.cutoff} with an age limit of ${inputs.maximumAge}: `
          + `${lookup.rejected.slice(0, 3).map(entry => `${recordLabel(entry.record)} ${entry.reasons[0]}`)
            .join('; ')}. `
          + 'A missing calibration is the answer here. Substituting a value that had not arrived would be '
          + 'the leak this section is about.'
        : `${recordLabel(lookup.selected.record)} is the latest eligible event, and among that event's `
          + `eligible versions the latest available one, so the value is ${fixed(lookup.value, 6)}. `
          + (newestEvent && !newestIsSelected
            ? `The newest event at or before the cutoff is ${recordLabel(newestEvent)}, and it was not `
              + `chosen: ${lookup.assessed.find(entry => entry.key === recordKey(newestEvent)).reasons[0]}. `
              + 'The newest event and the newest knowledge are different things.'
            : 'Here the newest event is also the newest knowledge, which is the easy case and the one that '
              + 'makes a backward join on event time look sufficient.'),
    };
  };

  const lookup = shown?.calculation.lookup ?? null;

  const options = [
    ...draft.records.map(record => [recordKey(record), recordLabel(record)]),
    ['none', 'no record qualifies'],
  ];

  return <Investigation title="Investigation 1 · move the arrival, keep the event"
    question={'Four calibration records, each with the time its event happened and the time its value reached '
      + 'the serving system. A prediction has to be made at the cutoff. Say which record it is entitled to '
      + 'use — then change an arrival, a cutoff, an age limit or a value and say it again.'}
    role={{ kind: 'constructed', text: 'A constructed history in abstract time units. Every number here is an '
      + 'integer and the selection is exact, so there is no tolerance and no estimate anywhere in this lab. '
      + 'No value is taken from the bank records.' }}
    note={'Reset returns the four records to their original arrivals and values, the cutoff to 5, the age '
      + 'limit to 5 and the question to sensor A.'}
    onReset={state.reset}>

    <div className="formulation-controls is-wide">
      <Select label="Which entity needs the calibration?" value={draft.entity}
        options={[['sensor_A', 'sensor A'], ['sensor_B', 'sensor B']]}
        onChange={value => state.edit({ entity: value })}
        hint="Matching the entity is the first of the five steps, and the easiest one to skip." />
      <NumberField label="prediction cutoff" value={draft.cutoff} min={0} max={12}
        onChange={value => state.edit({ cutoff: value })}
        hint="The moment the prediction has to be ready. Nothing that arrives later may be used." />
      <NumberField label="maximum event age" value={draft.maximumAge} min={0} max={12}
        onChange={value => state.edit({ maximumAge: value })}
        hint="A freshness limit on the EVENT time. It constrains how old a measurement may be; it does not make an unavailable one available." />
    </div>

    <p className="formulation-caption">Each record's arrival time and value can be edited. Event times and
      version numbers are fixed, so that “which version” always has a defined answer.</p>
    <div className="formulation-controls">
      {draft.records.map((record, index) => <NumberField key={`${recordKey(record)}-arrival`}
        label={`${recordLabel(record)} — arrives at`} value={record.available} min={0} max={12}
        onChange={value => state.edit({
          records: draft.records.map((entry, position) =>
            (position === index ? { ...entry, available: value } : entry)),
        })} />)}
    </div>
    <div className="formulation-controls">
      {draft.records.map((record, index) => <NumberField key={`${recordKey(record)}-value`}
        label={`${recordLabel(record)} — value`} value={record.value} min={-100} max={100}
        onChange={value => state.edit({
          records: draft.records.map((entry, position) =>
            (position === index ? { ...entry, value } : entry)),
        })} />)}
    </div>

    <div className="formulation-presets">
      {/* Built from the APPLIED state, not the draft. These buttons are
          comparisons -- "change the other sensor's value" is presented as an
          exact null -- and composed onto a half-typed draft they stop
          demonstrating the thing they are named after. */}
      {TIMELINE_PRESETS.map(([label, apply, disabled]) => <button key={label} type="button"
        disabled={disabled?.(state.active) ?? false}
        onClick={() => state.suggest(apply(state.active))}>{label}</button>)}
    </div>
    <p className="formulation-caption">Each setup above describes a change from the state currently applied, so
      choosing one replaces edits you have typed but not yet applied. That is what keeps “change the other
      sensor's value” an exact null however you reach it.</p>
    {problem && <p className="formulation-note" role="status">{problem}</p>}

    {geometry && <Figure caption={`The records as they stand: cutoff ${asInput(draft.cutoff)}, admissible event window `
      + `${asInput(draft.cutoff - draft.maximumAge)} to ${asInput(draft.cutoff)}, asking about `
      + `${draft.entity.replace('sensor_', 'sensor ')}. `
      /* The model flags a window reaching back past the ruler, and nothing used
         to read the flag: the caption said "-10 to 2" beside a band drawn from
         0 to 2, with nothing saying it had been truncated. A figure's extent is
         its claim. */
      + (geometry.ageWindow.startsBeforeTheRuler
        ? `The window starts before the ruler does, so the shaded band shows only its drawn part, `
          + `${asInput(geometry.ageWindow.drawnFrom)} to ${asInput(draft.cutoff)}. `
        : '')
      + 'Which records are eligible, and which one wins, is what the live calculation displays.'}>
      <Drawing kind="lanes" width={geometry.width} height={geometry.height} maxWidth={480}
        caption="Event times, arrival times and the prediction cutoff"
        describe={geometry.lanes.map(lane =>
          `${lane.label}: event at ${lane.record.event}, value available from ${lane.record.available}`
            + `${lane.isOtherEntity ? ', a different entity' : ''}`).join('; ')
          + `. The cutoff is at ${geometry.cutoff.value} and the admissible event window runs from `
          + `${geometry.ageWindow.from} to ${geometry.ageWindow.to}.`}>
        <rect className="form-age-window" x={geometry.ageWindow.x} y={geometry.topPadding - 8}
          width={Math.max(geometry.ageWindow.width, 0.6)}
          height={geometry.lanes.length * geometry.laneHeight + 8} />
        {geometry.lanes.map(lane => <g key={lane.key}
          className={`form-lane${lane.isOtherEntity ? ' is-other' : ''}`}>
          <text className="form-small" x={0} y={lane.markY + 3}>{lane.drawnLabel}</text>
          <line className="form-delay" x1={lane.eventX} y1={lane.markY} x2={lane.availableX} y2={lane.markY} />
          <circle className="form-event" cx={lane.eventX} cy={lane.markY} r={3.4} />
          <path className="form-arrival"
            d={`M ${lane.availableX} ${lane.markY - 5.2} L ${lane.availableX + 5.2} ${lane.markY} `
              + `L ${lane.availableX} ${lane.markY + 5.2} L ${lane.availableX - 5.2} ${lane.markY} Z`} />
        </g>)}
        <line className="form-cutoff" x1={geometry.cutoff.x} y1={geometry.cutoff.lineTopY}
          x2={geometry.cutoff.x} y2={geometry.axisY} />
        <text className="form-small form-axis-title" x={geometry.cutoff.labelX} y={geometry.cutoff.labelY}
          textAnchor="middle">cutoff</text>
        <line className="form-axis" x1={geometry.labelWidth} y1={geometry.axisY}
          x2={geometry.width - geometry.inset} y2={geometry.axisY} />
        {geometry.ticks.map((tick, index) => <g key={tick.value}>
          <line className="form-tick" x1={tick.x} y1={geometry.axisY} x2={tick.x} y2={geometry.tickY} />
          <text className="form-small" x={tick.x} y={geometry.tickLabelY}
            textAnchor={index === 0 ? 'start' : index === geometry.ticks.length - 1 ? 'end' : 'middle'}>
            {tick.value}
          </text>
        </g>)}
      </Drawing>
    </Figure>}
    {/* Gated with the drawing. A legend naming five marks, printed above a
        suppressed figure, legends nothing. */}
    {geometry && <Legend entries={[['event', 'event time'], ['arrival', 'available-at time'],
      ['delay', 'the gap between them'], ['age-window', 'admissible event ages'], ['cutoff', 'prediction cutoff']]} />}

    <Table caption="The history as recorded. Two versions of one event are two records; neither overwrites the other."
      headings={['Record', 'Event time', 'Available at', 'Version', 'Value']}
      rows={draft.records.map(record => [
        recordLabel(record), asInput(record.event), asInput(record.available),
        asInput(record.version), asInput(record.value),
      ])}
      footnote={'This table is the input. It says nothing about which record the cutoff admits, or which of '
        + 'the admitted ones wins — that is the question.'} />

    <LiveResult
      
      
      state={state}
      calculateInputs={inputs => (problem ? { outcome: null, value: NaN, lookup: null, explain: problem } : calculateInputs(inputs))}
      blocked={problem}
      
       />

    {lookup && <div className="formulation-reveal">
      <Table caption="Every record against the three conditions, with the reason each rejection gives"
        headings={['Record', 'Same entity?', 'Event at or before the cutoff?', 'Within the age limit?',
          'Arrived by the cutoff?', 'Verdict']}
        rowClass={index => (lookup.assessed[index].key === lookup.selectedKey ? 'is-leading'
          : lookup.assessed[index].eligible ? undefined : 'is-rejected')}
        rows={lookup.assessed.map(entry => [
          `${recordLabel(entry.record)} — value ${asInput(entry.record.value)}`,
          entry.entityMatch ? 'yes' : 'no',
          entry.notInFuture ? 'yes' : 'no',
          entry.withinAge ? 'yes' : 'no',
          entry.known ? 'yes' : 'no',
          entry.key === lookup.selectedKey ? 'selected'
            : entry.eligible ? 'eligible, but not the latest' : entry.reasons[0],
        ])}
        footnote={'Rejected records keep their values and their reason. They do not disappear: knowing that a '
          + 'value exists and may not be used is different from not knowing it exists.'} />
      <dl>
        <dt>Selected record</dt>
        <dd>{lookup.selected ? recordLabel(lookup.selected.record) : 'none'}</dd>
        <dt>Selected value</dt>
        <dd>{lookup.selected ? fixed(lookup.value, 6) : 'no calibration'}</dd>
        <dt>Eligible records</dt>
        <dd>{lookup.eligible.length} of {lookup.assessed.length}</dd>
      </dl>
      {committedGeometry && <Figure caption="The current valid state, with the eligible records marked and the selected one outlined.">
        <Drawing kind="lanes" width={committedGeometry.width} height={committedGeometry.height} maxWidth={480}
          caption="The same records, with eligibility and the selection shown"
          describe={`${lookup.eligible.length} records are eligible. `
            + (lookup.selected ? `${recordLabel(lookup.selected.record)} is selected.` : 'None is selected.')}>
          <rect className="form-age-window" x={committedGeometry.ageWindow.x}
            y={committedGeometry.topPadding - 8}
            width={Math.max(committedGeometry.ageWindow.width, 0.6)}
            height={committedGeometry.lanes.length * committedGeometry.laneHeight + 8} />
          {committedGeometry.lanes.map(lane => {
            const entry = lookup.assessed.find(item => item.key === lane.key);
            const selected = lane.key === lookup.selectedKey;
            return <g key={lane.key}
              className={`form-lane${entry.eligible ? ' is-eligible' : ' is-rejected'}${selected ? ' is-selected' : ''}`}>
              <text className="form-small" x={0} y={lane.markY + 3}>{lane.drawnLabel}</text>
              <line className="form-delay" x1={lane.eventX} y1={lane.markY} x2={lane.availableX} y2={lane.markY} />
              <circle className="form-event" cx={lane.eventX} cy={lane.markY} r={3.4} />
              <path className="form-arrival"
                d={`M ${lane.availableX} ${lane.markY - 5.2} L ${lane.availableX + 5.2} ${lane.markY} `
                  + `L ${lane.availableX} ${lane.markY + 5.2} L ${lane.availableX - 5.2} ${lane.markY} Z`} />
              {selected && <rect className="form-selected-outline" x={committedGeometry.labelWidth - 4}
                y={lane.y + 1} width={committedGeometry.width - committedGeometry.labelWidth} height={lane.markY - lane.y + 8} rx={3} />}
            </g>;
          })}
          <line className="form-cutoff" x1={committedGeometry.cutoff.x} y1={committedGeometry.cutoff.lineTopY}
            x2={committedGeometry.cutoff.x} y2={committedGeometry.axisY} />
          <text className="form-small form-axis-title" x={committedGeometry.cutoff.labelX}
            y={committedGeometry.cutoff.labelY} textAnchor="middle">cutoff</text>
          <line className="form-axis" x1={committedGeometry.labelWidth} y1={committedGeometry.axisY}
            x2={committedGeometry.width - committedGeometry.inset} y2={committedGeometry.axisY} />
          {committedGeometry.ticks.map((tick, index) => <g key={tick.value}>
            <line className="form-tick" x1={tick.x} y1={committedGeometry.axisY}
              x2={tick.x} y2={committedGeometry.tickY} />
            <text className="form-small" x={tick.x} y={committedGeometry.tickLabelY}
              textAnchor={index === 0 ? 'start' : index === committedGeometry.ticks.length - 1 ? 'end' : 'middle'}>
              {tick.value}
            </text>
          </g>)}
        </Drawing>
      </Figure>}
    </div>}
  </Investigation>;
}

/* ============================================================ investigation 2 */

const SOURCES = [
  ['candidate', 'the candidate recorded-feature model'],
  ['prior', 'the constant training-prior baseline'],
  ['duration', 'the model that includes final call duration'],
];

const validationIds = formulationData.validation.ids;
const validationTargets = formulationData.validation.targets;
const targetById = Object.fromEntries(validationIds.map((id, index) => [id, validationTargets[index]]));
const totalPositives = formulationData.partition.validationPositives;

/* The three base rankings, built once at module scope from the recorded
   per-case scores. The constant baseline's scores are all equal, so its
   ranking is entirely the tie rule -- which is the point. */
const priorScores = validationIds.map(() => formulationData.partition.trainPrior);
const baseOrders = {
  candidate: rankByScore({ ids: validationIds, scores: formulationData.validation.scores.candidate }),
  duration: rankByScore({ ids: validationIds, scores: formulationData.validation.scores.duration }),
  prior: rankByScore({ ids: validationIds, scores: priorScores }),
};
const scoresBySource = {
  candidate: Object.fromEntries(validationIds.map((id, index) =>
    [id, formulationData.validation.scores.candidate[index]])),
  duration: Object.fromEntries(validationIds.map((id, index) =>
    [id, formulationData.validation.scores.duration[index]])),
  prior: Object.fromEntries(validationIds.map(id => [id, formulationData.partition.trainPrior])),
};

const candidateProcedure = formulationData.procedures.find(procedure => procedure.id === 'candidate');

/** Identities with their recorded outcomes, bounded. */
const IDENTITIES_SHOWN = 10;
function describeIdentities(ids) {
  if (!ids.length) return 'nothing';
  const shown = ids.slice(0, IDENTITIES_SHOWN)
    .map(id => `${id} (${targetById[id] === 1 ? 'subscribed' : 'did not'})`).join(', ');
  return ids.length > IDENTITIES_SHOWN
    ? `${shown}, and ${ids.length - IDENTITIES_SHOWN} more`
    : shown;
}
const initialPolicy = { source: 'candidate', capacity: formulationData.partition.capacity, swaps: [], reversed: false };

export function CapacityLab() {
  const state = useInvestigation(initialPolicy, stateKey);
  const draft = state.draft;
  const shown = state.result;
  const [windowCentre, setWindowCentre] = useState(formulationData.partition.capacity);
  const [pendingOut, setPendingOut] = useState(null);
  const [pendingIn, setPendingIn] = useState(null);
  const clearPendingSwap = () => { setPendingOut(null); setPendingIn(null); };

  const edited = draft.swaps.length > 0 || draft.reversed;

  /* The APPLIED state's metrics — the baseline `calculateInputs` compares against.
     The caption below reads these, so the sentence that states the baseline
     and the grader that uses it cannot disagree. */
  const appliedMetrics = selectionMetrics({
    order: policyOrder({
      baseOrder: baseOrders[state.active.source], capacity: state.active.capacity,
      swaps: state.active.swaps, reversed: state.active.reversed,
    }),
    capacity: state.active.capacity,
    targetById,
    totalPositives,
  });
  const isFirstRound = stateKey(state.active) === stateKey(initialPolicy);
  const order = policyOrder({
    baseOrder: baseOrders[draft.source], capacity: draft.capacity, swaps: draft.swaps, reversed: draft.reversed,
  });
  const selectedSet = new Set(order.slice(0, draft.capacity));

  
  const WINDOW_ROWS = 12;
  const first = Math.max(0, Math.min(windowCentre - WINDOW_ROWS / 2, order.length - WINDOW_ROWS));
  const rows = order.slice(first, first + WINDOW_ROWS).map((id, offset) => ({
    id,
    rank: first + offset + 1,
    score: scoresBySource[draft.source][id],
    selected: selectedSet.has(id),
  }));

  const applySwap = () => {
    if (pendingOut === null || pendingIn === null) return;
    state.edit({ swaps: [...draft.swaps, [pendingOut, pendingIn]] });
    setPendingOut(null);
    setPendingIn(null);
  };

  const calculateInputs = (inputs, previous) => {
    const nowOrder = policyOrder({
      baseOrder: baseOrders[inputs.source], capacity: inputs.capacity,
      swaps: inputs.swaps, reversed: inputs.reversed,
    });
    const beforeOrder = policyOrder({
      baseOrder: baseOrders[previous.source], capacity: previous.capacity,
      swaps: previous.swaps, reversed: previous.reversed,
    });
    const now = selectionMetrics({ order: nowOrder, capacity: inputs.capacity, targetById, totalPositives });
    const before = selectionMetrics({
      order: beforeOrder, capacity: previous.capacity, targetById, totalPositives,
    });
    const movement = movementOf(before.precision, now.precision);
    const change = membershipChange(before.selected, now.selected);
    return {
      outcome: movement.outcome,
      value: now.precision,
      now,
      before,
      movement,
      change,
      explain: `The selected ${now.capacity} contain ${now.positivesFound} recorded subscriptions, so precision `
        + `is ${now.positivesFound}/${now.capacity} = ${fixed(now.precision, 6)} against `
        + `${before.positivesFound}/${before.capacity} = ${fixed(before.precision, 6)} before. Recall moved `
        + `from ${before.positivesFound}/${totalPositives} to ${now.positivesFound}/${totalPositives}. `
        + (movement.outcome === 'unchanged'
          ? `The two differ by ${fixed(Math.abs(movement.rawDifference), 12)}, inside the tolerance `
            + `${unchangedTolerance}, so this is reported as unchanged. ${change.entered.length === 0
              ? 'No identity entered or left the selected set, which is why no set metric could move.'
              : 'Identities changed and the count happened to land in the same place, which is not the same '
                + 'thing as nothing changing.'}`
          : `${change.entered.length} identit${change.entered.length === 1 ? 'y' : 'ies'} entered the selected `
            + `set and ${change.left.length} left.`),
    };
  };

  const result = shown?.calculation ?? null;

  return <Investigation title="Investigation 2 · a score becomes a limited action set"
    question={'The candidate model has ranked all 824 validation opportunities. Calling capacity decides how '
      + 'far down that list anyone actually goes. Change the capacity, or exchange one selected opportunity '
      + 'for one unselected one, and follow how the selected identities change precision while each '
      + 'observed outcome remains visible.'}
    role={{ kind: 'measured', text: 'Measured. The probabilities are this lesson\'s recorded validation '
      + 'outputs and the outcomes are what those customers actually did. They are DEVELOPMENT evidence: this '
      + 'exercise deliberately looks at answers that have already been seen, to watch which identities enter a '
      + 'metric. Choosing a better action set on them would not be evidence about anyone else.' }}
    note={'Outcomes are shown as you explore. This is development evidence, not a sealed test — the '
      + 'numbers are in this page\'s data module either way. Reset returns to the candidate ranking at '
      + 'capacity 50 with no edits.'}
    onReset={() => { state.reset(); setWindowCentre(formulationData.partition.capacity); setPendingOut(null); setPendingIn(null); }}>

    <div className="formulation-controls is-wide">
      <Select label="Rank by" value={draft.source} options={SOURCES}
        onChange={value => { state.edit({ source: value, swaps: [], reversed: false }); setPendingOut(null); setPendingIn(null); }}
        hint="Changing the score source clears any edited action set, because an exchange refers to positions in a particular ranking." />
      <NumberField label="calling capacity" value={draft.capacity} min={1} max={100} disabled={edited}
        onChange={value => { state.edit({ capacity: value }); setWindowCentre(value); clearPendingSwap(); }}
        hint={edited
          ? 'Disabled while the action set is edited: restore the model ranking first, so that an exchange and a capacity change are never applied to each other\'s state.'
          : 'How many of the ranked opportunities are actually called.'} />
    </div>

    <div className="formulation-presets">
      <button type="button" disabled={edited}
        onClick={() => { state.suggest({ ...state.active, capacity: 25, swaps: [], reversed: false }); setWindowCentre(25); clearPendingSwap(); }}>
        Halve the capacity to 25
      </button>
      <button type="button" disabled={edited}
        onClick={() => { state.suggest({ ...state.active, capacity: 100, swaps: [], reversed: false }); setWindowCentre(100); clearPendingSwap(); }}>
        Double it to 100
      </button>
      <button type="button"
        onClick={() => { state.suggest({ ...state.active, reversed: !state.active.reversed }); clearPendingSwap(); }}>
        {draft.reversed ? 'Undo the reversal of the selected block' : 'Reverse the order within the selected block'}
      </button>
      <button type="button"
        onClick={() => {
          state.suggest({
            source: 'candidate', capacity: 25,
            swaps: [[formulationData.policyFixtures.swap.removeId, formulationData.policyFixtures.swap.addId]],
            reversed: false,
          });
          setWindowCentre(25);
          clearPendingSwap();
        }}>
        Set capacity 25 AND load the recorded exchange (two changes at once)
      </button>
      <button type="button" disabled={!edited}
        onClick={() => { state.suggest({ ...state.active, swaps: [], reversed: false }); setPendingOut(null); setPendingIn(null); }}>
        Restore the model ranking
      </button>
    </div>
    {edited && <p className="formulation-note" role="status">
      {/* Built from the parts that are actually present. "0 exchanges and a
          reversal of the selected block" is what a template with no zero case
          produces, and it reads as a fault. */}
      The action set is edited: {[
        draft.swaps.length ? `${draft.swaps.length} exchange${draft.swaps.length === 1 ? '' : 's'}` : null,
        draft.reversed ? 'the selected block reversed' : null,
      ].filter(Boolean).join(', and ')}. This is a change of policy, not of the model: no probability, no
      outcome and no fitted coefficient has moved.
    </p>}
    {/* The availability contract travels with the ranking, every time. On a page
        about information arriving before it should, a control that silently
        switches to the model built on an impossible input would be teaching the
        opposite of the lesson. */}
    <p className={`formulation-role is-${draft.source === 'duration' ? 'exploratory' : 'measured'}`} role="status">
      {formulationData.procedures.find(procedure => procedure.id === draft.source).availability}
    </p>

    <div className="formulation-controls">
      <NumberField label="show the ranking around rank" value={windowCentre} min={1} max={order.length}
        onChange={setWindowCentre}
        hint="A view control. Moving the window shows different rows and changes the view without changing the calculation." />
    </div>

    <Table caption={`Ranks ${first + 1} to ${first + rows.length} of ${order.length}, with the selection `
      + `boundary at ${draft.capacity}`}
      headings={['Rank', 'Source row', 'Probability', 'In the selected set?', 'Outcome', 'Exchange']}
      rowClass={index => (rows[index].selected ? 'is-selected-row' : undefined)}
      rows={rows.map(row => [
        row.rank,
        row.id,
        probabilityText(row.score),
        row.selected ? 'selected' : 'not selected',
        result ? (targetById[row.id] === 1 ? 'subscribed' : 'did not subscribe') : 'waiting for a valid calculation',
        row.selected
          ? <button key="out" type="button" className={pendingOut === row.id ? 'is-selected' : undefined}
            onClick={() => setPendingOut(pendingOut === row.id ? null : row.id)}>
            {pendingOut === row.id ? `drop ${row.id} ✓` : `drop ${row.id}`}
          </button>
          : <button key="in" type="button" className={pendingIn === row.id ? 'is-selected' : undefined}
            onClick={() => setPendingIn(pendingIn === row.id ? null : row.id)}>
            {pendingIn === row.id ? `take ${row.id} ✓` : `take ${row.id}`}
          </button>,
      ])}
      footnote={'The outcome column is the only thing hidden. Ranks, identities and probabilities are all on '
        + 'screen while you explore, because those are what a scheduler would actually have.'} />

    <div className="formulation-buttons">
      <button type="button" disabled={pendingOut === null || pendingIn === null} onClick={applySwap}>
        {pendingOut === null || pendingIn === null
          ? 'Choose one selected and one unselected opportunity to exchange'
          : `Select ${pendingIn} instead of ${pendingOut}`}
      </button>
    </div>

    <Readouts cells={[
      ['ranking', SOURCES.find(entry => entry[0] === draft.source)[1].replace('the ', '')],
      ['capacity', asInput(draft.capacity)],
      ['selected', asInput(selectedSet.size)],
      ['positives in the whole evaluated set', asInput(totalPositives)],
    ]} />
    {}
    {!shown && <p className="formulation-caption">
      {isFirstRound
        ? <>Section 5 records the starting point for you: at capacity {appliedMetrics.capacity} the candidate's
          selected set contains {appliedMetrics.positivesFound} subscriptions, a precision
          of {round(appliedMetrics.precision, 6)}. Change a control and inspect the difference.</>
        : <>Your last applied state is the one the comparison uses: the {SOURCES.find(entry =>
          entry[0] === state.active.source)[1].replace('the ', '')} at capacity {appliedMetrics.capacity}
          {state.active.swaps.length || state.active.reversed ? ', with your edits to the action set,' : ''}
          {' '}selects {appliedMetrics.positivesFound} subscriptions, a precision
          of {round(appliedMetrics.precision, 6)}. Compare what your next change does to <em>that</em>.</>}
    </p>}

    <LiveResult
      
      
      state={state}
      calculateInputs={calculateInputs}
      
      
      
       />

    {result && <div className="formulation-reveal">
      <dl>
        <dt>Selected</dt>
        <dd>{result.now.capacity}</dd>
        <dt>Subscriptions among them</dt>
        <dd>{result.now.positivesFound}</dd>
        <dt>Precision</dt>
        <dd>{result.now.positivesFound}/{result.now.capacity} = {fixed(result.now.precision, 6)}</dd>
        <dt>Recall</dt>
        <dd>{result.now.positivesFound}/{totalPositives} = {fixed(result.now.recall, 6)}</dd>
        <dt>Not reached</dt>
        <dd>{result.now.missed} of the {totalPositives} subscriptions in the evaluated set</dd>
        {/* Capped. A capacity change can move twenty-five identities at once,
            and twenty-five parenthesised outcomes run on for six lines and stop
            being read. The first ten carry the point; the ranked table above
            carries the rest, in order, with their probabilities. */}
        <dt>Entered the selected set</dt>
        <dd>{describeIdentities(result.change.entered)}</dd>
        <dt>Left it</dt>
        <dd>{describeIdentities(result.change.left)}</dd>
      </dl>
      <p className="formulation-caption">
        Precision and recall have the same numerator and different denominators. Precision divides
        by {result.now.capacity}, the capacity you chose; recall divides by {totalPositives}, every
        subscription in the evaluated set, which no capacity can change. That is why one can rise while the
        other falls, and why neither on its own describes the decision.
      </p>
    </div>}
  </Investigation>;
}
