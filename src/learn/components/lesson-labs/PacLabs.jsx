import { useState } from 'react';
import {
  BitRow, Investigation, LabelChoice, NumberField, NumberLine, LiveResult, Select, Table,
  asInput, fixed, round, useInvestigation,
} from './PacShared.jsx';
import {
  finiteWorldInputs, finiteWorldProbability, finiteWorldRun, fixtures, intervalExperiment, intervalPatterns,
  manualIntervalRequest, movementOf, requestVerdict, seededUniformSample, stripGeometry,
  thresholdPatterns, unchangedTolerance, witnessPredictions,
} from '../../data/pac-models.js';



const sampleKey = state => JSON.stringify(state);

/* ============================================================ investigation 1 */

const FINITE_PRESETS = [
  ['Add input 2, the first positive', draft => ({ ...draft, sample: [...draft.sample, 2] })],
  ['Repeat an input you already have',
    draft => ({ ...draft, sample: [...draft.sample, draft.sample[0] ?? 0] }),
    draft => draft.sample.length === 0],
  ['Observe all four inputs', draft => ({ ...draft, sample: [0, 1, 2, 3] })],
  ['Change the target to 0000', draft => ({ ...draft, target: [0, 0, 0, 0] })],
];

export function FiniteWorldLab() {
  const state = useInvestigation({ ...fixtures.finiteWorld, probabilityN: 4 }, sampleKey);
  const draft = state.draft;
  const tooLong = draft.sample.length > 24;
  /* An empty observation sequence is allowed on purpose: "what does this learner
     return before it has seen anything" is a real question in this world, and
     the answer -- the all-zero rule -- is the reason the default matters. */
  const problem = tooLong ? 'This world takes at most 24 observations.' : null;

  const calculateInputs = inputs => {
    const run = finiteWorldRun(inputs);
    return {
      outcome: run.meetsTarget ? 'meets' : 'fails',
      value: run.risk,
      run,
      explain: `The first consistent rule is ${run.ruleText}. It differs from the target at `
        + `${run.wrong.length ? `input${run.wrong.length === 1 ? '' : 's'} ${run.wrong.join(', ')}` : 'no input'}, `
        + `so its population risk is ${run.wrong.length}/4 = ${fixed(run.risk, 6)}, and ε is ${asInput(run.epsilon)}. `
        + `Training error is ${fixed(run.empiricalRisk, 6)} either way: this setting is realizable, so a `
        + 'consistent rule always exists and fitting the sample perfectly is not the achievement.',
    };
  };

  const shown = state.result;
  const run = shown?.calculation.run ?? null;
  const probability = shown
    ? finiteWorldProbability({ target: shown.inputs.target, n: shown.inputs.probabilityN, epsilon: shown.inputs.epsilon })
    : null;

  return <Investigation title="Investigation 1 · which observations remove the uncertainty?"
    question={'Four inputs 0, 1, 2 and 3 are equally likely. All sixteen binary rules on them are candidates, '
      + 'fixed before anything is seen, and the learner returns the first consistent one in the order below — '
      + 'which means it predicts 0 wherever it has seen nothing. Choose what to observe, then say what that '
      + 'learner will cost.'}
    role={{ kind: 'constructed', text: 'A constructed world with a known target, so the population risk is an '
      + 'exact count and not an estimate. The learner never reads a label it has not observed.' }}
    note="Reset returns the target to 0011, the sample to 0 then 1, and ε to .25."
    onReset={state.reset}>

    <div className="pac-controls">
      {finiteWorldInputs.map(input => <LabelChoice key={input} label={`target label at input ${input}`}
        value={draft.target[input]}
        onChange={value => state.edit({
          target: draft.target.map((bit, index) => (index === input ? value : bit)),
        })} />)}
    </div>

    <div className="pac-controls">
      <NumberField label="ε, the population-error target" value={draft.epsilon} min={0.05} max={0.5}
        decimals={2} onChange={value => state.edit({ epsilon: value })}
        hint="Risk exactly equal to ε counts as meeting it: the comparison is ≤, not <." />
    </div>

    <p className="pac-caption">Observation sequence, in the order drawn:</p>
    <div className="pac-presets">
      {draft.sample.map((input, index) => <button key={`${input}-${index}`} type="button"
        onClick={() => state.edit({ sample: draft.sample.filter((_value, position) => position !== index) })}>
        input {input} ✕
      </button>)}
      {!draft.sample.length && <span className="pac-caption">nothing observed yet</span>}
    </div>
    <div className="pac-presets">
      {finiteWorldInputs.map(input => <button key={input} type="button" disabled={draft.sample.length >= 24}
        onClick={() => state.edit({ sample: [...draft.sample, input] })}>add an observation at {input}</button>)}
    </div>
    <div className="pac-presets">
      {FINITE_PRESETS.map(([label, apply, unavailable]) => <button key={label} type="button"
        disabled={Boolean(unavailable && unavailable(draft)) || apply(draft).sample.length > 24}
        onClick={() => state.suggest(apply(draft))}>{label}</button>)}
    </div>
    {problem && <p className="pac-note" role="status">{problem}</p>}

    <Table caption="The sixteen candidate rules, fixed before any observation, with their own error regions"
      headings={['Rule', 'Differs from the target at', 'Population risk']}
      rows={Array.from({ length: 16 }, (_unused, code) => {
        const rule = [0, 1, 2, 3].map(index => (code >> (3 - index)) & 1);
        const wrong = [0, 1, 2, 3].filter(index => rule[index] !== draft.target[index]);
        return [
          <BitRow key={code} bits={rule} highlight={wrong} />,
          wrong.length ? `inputs ${wrong.join(', ')}` : 'nowhere',
          fixed(wrong.length / 4, 2),
        ];
      })}
      footnote={'This table describes the class against the target you set. It says nothing about which rule '
        + 'your sample selects; that is the question.'} />

    <LiveResult
      
      
      state={state} calculateInputs={calculateInputs}
      
      blocked={problem}
       />

    {run && <div className="pac-reveal">
      <dl>
        <dt>Rule returned</dt>
        <dd><BitRow bits={run.rule} highlight={run.wrong} /> ({run.ruleText})</dd>
        <dt>Inputs observed</dt>
        <dd>{run.seen.length ? run.seen.join(', ') : 'none'}{run.repeatedObservations
          ? ` (${run.repeatedObservations} repeated observation${run.repeatedObservations === 1 ? '' : 's'} revealed no new label)`
          : ''}</dd>
        <dt>Inputs never seen</dt>
        <dd>{run.unseen.length ? `${run.unseen.join(', ')} — predicted 0 by default` : 'none'}</dd>
        <dt>Training error</dt>
        <dd>{fixed(run.empiricalRisk, 6)}</dd>
        <dt>Population risk</dt>
        <dd>{fixed(run.risk, 6)}</dd>
        <dt>Meets ε = {asInput(run.epsilon)}</dt>
        <dd>{run.meetsTarget ? 'yes' : 'no'}</dd>
      </dl>
    </div>}

    {probability && <>
      <p className="pac-caption">
        Now the other level of randomness. Holding the same target and ε, and drawing n observations
        independently and uniformly from the four inputs, the exact probability that this learner ends up above
        ε is computed below by tracking which inputs have been seen. It is exact rational arithmetic on a
        sixteen-state space, so there is no sampling noise in it at all.
      </p>
      <div className="pac-controls">
        <NumberField label="n, independent draws" value={draft.probabilityN} min={0} max={40} decimals={0}
          onChange={value => state.edit({ probabilityN: value })}
          hint="Changing n updates the repeated-sampling comparison above." />
      </div>
      <div className="pac-strip">
        <span className="pac-strip-cell is-used">exact failure probability<b>{probability.failureExact}</b></span>
        <span className="pac-strip-cell">as a decimal<b>{fixed(probability.failure, 9)}</b></span>
        <span className="pac-strip-cell">16 e<sup>−nε</sup>, raw<b>{fixed(probability.bound.raw, 6)}</b></span>
        <span className="pac-strip-cell">the same, clipped to a probability<b>{fixed(probability.bound.clipped, 6)}</b></span>
      </div>
      <p className="pac-caption">
        {probability.bound.vacuous
          ? 'At this n the generic finite-class bound is above 1, so it restricts nothing. That is not a wrong '
            + 'bound and not evidence that learning has failed — it is a valid statement carrying no information.'
          : `At this n the bound is below 1 and does restrict the failure probability. The exact value is `
            + `${probability.failureExact}, far below it: a general bound that holds for every target and every `
            + 'consistent learner has to be loose for this particular one.'}
      </p>
      <Table caption="Every reachable seen-set at this n, its exact probability, and what the learner then returns"
        headings={['Inputs seen', 'Probability', 'Rule returned', 'Population risk', 'Above ε?']}
        rowClass={index => (probability.states[index].fails ? 'is-failing' : undefined)}
        rows={probability.states.map(entry => [
          entry.seen.length ? entry.seen.join(', ') : 'none',
          entry.probability,
          entry.ruleText,
          entry.riskExact,
          entry.fails ? 'yes' : 'no',
        ])}
        footnote={'The probabilities are exact fractions and sum to exactly 1. The failing rows are the ones '
          + 'the failure probability adds up.'} />
    </>}
  </Investigation>;
}

/* ============================================================ investigation 2 */

const NAMES = ['A', 'B', 'C', 'D', 'E', 'F'];

const WITNESS_PRESETS = [
  ['Repair B to positive: 111', draft => ({
    ...draft, points: draft.points.map(point => (point.id === 'B' ? { ...point, label: 1 } : point)),
  })],
  ['Move all three left, keeping order: .1 / .4 / .9', draft => ({
    ...draft, points: draft.points.map((point, index) => ({ ...point, x: [0.1, 0.4, 0.9][index] ?? point.x })),
  })],
  ['Translate everything by +2', draft => ({
    ...draft, points: draft.points.map(point => ({ ...point, x: Number((point.x + 2).toFixed(3)) })),
  })],
  ['Move B past C, to .95', draft => ({
    ...draft, points: draft.points.map(point => (point.id === 'B' ? { ...point, x: 0.95 } : point)),
  })],
];

export function WitnessLab() {
  const state = useInvestigation(fixtures.request, sampleKey);
  const draft = state.draft;
  const shown = state.result;
  const coordinates = draft.points.map(point => point.x);
  const clash = coordinates.find((value, index) => coordinates.indexOf(value) !== index);
  const problem = clash !== undefined
    ? `Two points share the coordinate ${clash}. Separate them: this mode needs distinct positions, because `
      + 'two labels at one position is a different question from a class being unable to realize a pattern.'
    : null;

  const calculateInputs = inputs => {
    const verdict = requestVerdict(inputs);
    const count = inputs.family === 'interval'
      ? intervalPatterns(inputs.points.length).length
      : thresholdPatterns(inputs.points.length).length;
    return {
      outcome: verdict.feasible ? 'possible' : 'impossible',
      value: count,
      verdict,
      count,
      total: 2 ** inputs.points.length,
      explain: verdict.explain,
    };
  };

  const verdict = shown?.calculation.verdict ?? null;
  const predictions = verdict ? witnessPredictions(verdict.witness, shown.inputs.points) : null;

  return <Investigation title="Investigation 2 · build the witness, or name the obstruction"
    question={'Place the points, attach the labels you want, and choose the class. Before touching an '
      + 'endpoint, say whether any member of that class can produce those labels — and how many of the 2ⁿ '
      + 'labelings it could produce on these points altogether.'}
    role={{ kind: 'constructed', text: 'Exact enumeration on the points you set. A witness here is a real rule '
      + 'with its predictions shown point by point, not a search that gave up.' }}
    note="Reset returns to A, B, C at .2, .5, .8 with the labels 1, 0, 1 and the interval class."
    onReset={state.reset}>

    <div className="pac-controls is-wide">
      <Select label="Class" value={draft.family}
        options={[['interval', 'One closed interval, including the empty positive region'],
          ['threshold', 'Increasing threshold, h(x) = 1 when x ≥ a']]}
        onChange={value => state.edit({ family: value })} />
    </div>

    <div className="pac-controls">
      {draft.points.map((point, index) => <NumberField key={point.id} label={`${point.id} coordinate`}
        value={point.x} min={-5} max={5} decimals={3}
        onChange={value => state.edit({
          points: draft.points.map((entry, position) => (position === index ? { ...entry, x: value } : entry)),
        })} />)}
    </div>
    <div className="pac-controls">
      {draft.points.map((point, index) => <LabelChoice key={point.id} label={`${point.id} requested label`}
        value={point.label}
        onChange={value => state.edit({
          points: draft.points.map((entry, position) => (position === index ? { ...entry, label: value } : entry)),
        })} />)}
    </div>

    <div className="pac-presets">
      <button type="button" disabled={draft.points.length >= 6}
        onClick={() => state.edit({
          points: [...draft.points, {
            id: NAMES[draft.points.length],
            x: Math.max(...coordinates) <= 4.8
              ? Number((Math.max(...coordinates) + 0.2).toFixed(3))
              : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5].find(value => !coordinates.includes(value)),
            label: 0,
          }],
        })}>Add a point</button>
      <button type="button" disabled={draft.points.length <= 2}
        onClick={() => state.edit({ points: draft.points.slice(0, -1) })}>Remove the last point</button>
      {WITNESS_PRESETS.map(([label, apply]) => <button key={label} type="button"
        disabled={apply(draft).points.some(point => point.x < -5 || point.x > 5)}
        onClick={() => state.suggest(apply(draft))}>{label}</button>)}
    </div>
    {problem && <p className="pac-note" role="status">{problem}</p>}

    <Table caption="The request as it stands, sorted by coordinate — the order is what decides feasibility"
      headings={['Point', 'Coordinate', 'Requested label']}
      rows={[...draft.points].sort((left, right) => left.x - right.x).map(point => [
        point.id, asInput(point.x), point.label === 1 ? '1 · positive' : '0 · negative',
      ])}
      footnote={'Sorted here so the run of positives is visible. Identities stay attached: moving a point past '
        + 'another changes the sorted pattern without changing which label belongs to which name.'} />

    <LiveResult
      
      
      state={state}
      calculateInputs={inputs => (problem ? { outcome: null, value: NaN, verdict: null, explain: problem } : calculateInputs(inputs))}
      blocked={problem}
      
       />

    {verdict && <div className="pac-reveal">
      <dl>
        <dt>Sorted order</dt>
        <dd>{verdict.order.join(' < ')}</dd>
        <dt>Requested pattern</dt>
        <dd><BitRow bits={verdict.pattern} /></dd>
        <dt>Verdict</dt>
        <dd>{verdict.feasible ? 'realizable' : 'not realizable by this class'}</dd>
        <dt>Witness</dt>
        <dd>{verdict.feasible
          ? (verdict.witness.kind === 'threshold'
            ? `h(x) = 1 when x ≥ ${verdict.witness.threshold}`
            : verdict.witness.empty
              ? 'the empty positive region'
              : `the interval [${verdict.witness.interval[0]}, ${verdict.witness.interval[1]}]`)
          : 'none exists'}</dd>
        <dt>Realizable labelings</dt>
        <dd>{shown.calculation.count} of {shown.calculation.total}</dd>
      </dl>
      <p className="pac-caption">{verdict.explain}</p>
      {predictions && <Table caption="What the witness actually predicts, point by point"
        headings={['Point', 'Requested', 'Predicted by the witness']}
        rows={predictions.map((entry, index) => [
          entry.id,
          String([...shown.inputs.points].find(point => point.id === entry.id).label),
          String(entry.predicted),
        ])} />}
      <p className="pac-caption">
        One feasible request is not shattering. Shattering this set would mean every one of
        the {shown.calculation.total} labelings has a witness, and this class manages {shown.calculation.count} of them.
        Neither is a statement about the class's VC dimension, which quantifies over <em>all</em> point sets.
      </p>
    </div>}
  </Investigation>;
}

/* ============================================================ investigation 3 */

const INTERVAL_PRESETS = [
  ['Add two positives near the edges: .31 and .69', draft => ({
    ...draft, points: [...draft.points, 0.31, 0.69].sort((left, right) => left - right),
  })],
  ['Add two negatives far outside: .01 and .99', draft => ({
    ...draft, points: [...draft.points, 0.01, 0.99].sort((left, right) => left - right),
  })],
  ['Remove every positive observation', draft => ({
    ...draft, points: draft.points.filter(x => x < draft.target[0] || x > draft.target[1]),
  })],
  ['Move the target to [.25, .75]', draft => ({ ...draft, target: [0.25, 0.75] })],
];

export function BoundaryStripLab() {
  const initial = {
    ...fixtures.interval, mode: 'generated', labels: null, seed: 7, drawSize: 8,
  };
  const state = useInvestigation(initial, sampleKey);
  const draft = state.draft;
  const shown = state.result;
  const [seedDraft, setSeedDraft] = useState(7);
  const [sizeDraft, setSizeDraft] = useState(8);

  const targetBad = draft.target[0] > draft.target[1];
  const empty = draft.points.length === 0;
  const problem = targetBad ? 'The target needs its left endpoint at or below its right one.'
    : draft.points.length > 24 ? 'This investigation takes at most 24 observations.'
    : empty ? 'Add at least one observation.' : null;

  
  const applied = state.active;
  const comparisonBase = applied.mode === 'generated' ? applied : draft;
  const hasBaseline = shown?.calculation.experiment ? Boolean(shown.calculation.before) : applied.mode === 'generated';
  const appliedExperiment = applied.mode === 'generated' && applied.points.length && applied.target[0] <= applied.target[1]
    ? intervalExperiment({ points: applied.points, target: applied.target, epsilon: applied.epsilon })
    : null;

  const calculateInputs = (inputs, previous) => {
    if (inputs.mode === 'manual') {
      const request = manualIntervalRequest({ points: inputs.points, labels: inputs.labels });
      return {
        outcome: request.feasible ? 'possible' : 'impossible',
        value: NaN,
        request,
        explain: `${request.explain} The realizable guarantee of the section above does not apply to a request `
          + 'no member of the class can fit: it assumed the labels came from some interval.',
      };
    }
    const now = intervalExperiment({ points: inputs.points, target: inputs.target, epsilon: inputs.epsilon });
    const before = previous.mode === 'generated' && previous.points.length && previous.target[0] <= previous.target[1]
      ? intervalExperiment({ points: previous.points, target: previous.target, epsilon: previous.epsilon })
      : null;
    const movement = before ? movementOf(before.risk, now.risk) : null;
    return {
      outcome: movement ? movement.outcome : 'baseline',
      value: now.risk,
      experiment: now,
      before,
      movement,
      explain: `The tight fit is ${now.interval ? `[${now.interval[0]}, ${now.interval[1]}]` : 'the empty positive region'} `
        + `and its population risk is ${fixed(now.risk, 6)}`
        + (before ? `, against ${fixed(before.risk, 6)} before. ` : '. ')
        + (movement && movement.outcome === 'unchanged'
          ? `The two risks differ by ${fixed(Math.abs(movement.rawDifference), 12)}, inside the tolerance `
            + `${unchangedTolerance}, so this is reported as unchanged. Training error was 0 both times, and `
            + 'that is exactly why training error could not have told you.'
          : before ? 'Training error is 0 in both cases, so it could not have told you which way this went.'
            : 'This establishes a new generated-label baseline. The previous manual-label task had no population risk to compare.'),
    };
  };

  const experiment = shown?.calculation.experiment ?? null;
  const request = shown?.calculation.request ?? null;

  return <Investigation title="Investigation 3 · explore what one more observation can change"
    question={'The target interval generates every label, so the fit is always consistent with the sample and '
      + 'its training error is always 0. Propose a change to the sample or the target, say which way the '
      + "population risk will move and what it will become, then look."}
    role={{ kind: 'constructed', text: 'A constructed world: X is uniform on [0, 1] and the target is an '
      + 'interval, so a length is exactly a probability and the risk is computed rather than estimated.' }}
    note={'The drawing below shows the state currently APPLIED, whose fit section 8 has already worked '
      + 'through. Editing the sample or target updates the fit and population risk. Reset returns to the worked sample and target [.3, .7].'}
    onReset={() => { state.reset(); setSeedDraft(7); setSizeDraft(8); }}>

    <div className="pac-controls is-wide">
      <Select label="Where the labels come from" value={draft.mode}
        options={[['generated', 'Generated by the target — the realizable setting'],
          ['manual', 'Typed by hand — consistency becomes a question']]}
        onChange={value => state.edit({
          mode: value,
          labels: value === 'manual'
            ? draft.points.map(x => (x >= draft.target[0] && x <= draft.target[1] ? 1 : 0))
            : null,
        })} />
    </div>

    <div className="pac-controls">
      <NumberField label="target left endpoint" value={draft.target[0]} min={0} max={1} decimals={3}
        disabled={draft.mode === 'manual'}
        onChange={value => state.edit({ target: [value, draft.target[1]] })} />
      <NumberField label="target right endpoint" value={draft.target[1]} min={0} max={1} decimals={3}
        disabled={draft.mode === 'manual'}
        onChange={value => state.edit({ target: [draft.target[0], value] })} />
      <NumberField label="ε, the risk target" value={draft.epsilon} min={0.01} max={0.9} decimals={2}
        disabled={draft.mode === 'manual'}
        onChange={value => state.edit({ epsilon: value })} />
    </div>

    <p className="pac-caption">Observations, each in [0, 1]:</p>
    <div className="pac-controls">
      {draft.points.map((x, index) => <NumberField key={index} label={`observation ${index + 1}`} value={x}
        min={0} max={1} decimals={3}
        onChange={value => state.edit({
          points: draft.points.map((entry, position) => (position === index ? value : entry)),
        })} />)}
    </div>
    {draft.mode === 'manual' && <div className="pac-controls">
      {draft.points.map((x, index) => <LabelChoice key={index} label={`label you request at ${asInput(x)}`}
        value={draft.labels[index]}
        onChange={value => state.edit({
          labels: draft.labels.map((entry, position) => (position === index ? value : entry)),
        })} />)}
    </div>}

    <div className="pac-presets">
      <button type="button" disabled={draft.points.length >= 24}
        onClick={() => state.edit({
          points: [...draft.points, 0.5],
          labels: draft.mode === 'manual' ? [...draft.labels, 1] : null,
        })}>Add an observation at .5</button>
      <button type="button" disabled={draft.points.length <= 1}
        onClick={() => state.edit({
          points: draft.points.slice(0, -1),
          labels: draft.mode === 'manual' ? draft.labels.slice(0, -1) : null,
        })}>Remove the last observation</button>
      {/* Built from the APPLIED state, not the draft.
          These presets are comparisons: "add two negatives far outside" is the
          button section 8 and practice 8 both present as the demonstration that
          an exterior negative leaves the fit untouched. Composed onto a dirty
          draft it stopped demonstrating that -- an edited target plus the
          preset moved the risk, and the verdict correctly said "it falls",
          which is a correct answer to a question the learner did not think they
          had asked. Applying them to `state.active` discards pending edits so
          the button restores the null it promises. */}
      {draft.mode === 'generated' && INTERVAL_PRESETS.map(([label, apply]) => <button key={label} type="button"
        disabled={apply(comparisonBase).points.length === 0 || apply(comparisonBase).points.length > 24}
        onClick={() => state.suggest(apply(comparisonBase))}>
        {label.startsWith('Add two positives') && !(comparisonBase.target[0] <= .31 && comparisonBase.target[1] >= .69)
          ? 'Add observations at .31 and .69 (labels follow the current target)'
          : label.startsWith('Add two negatives') && !(comparisonBase.target[0] > .01 && comparisonBase.target[1] < .99)
            ? 'Add observations at .01 and .99 (labels follow the current target)' : label}
      </button>)}
      {draft.mode === 'manual' && <button type="button"
        onClick={() => state.suggest({
          ...draft, points: [0.2, 0.5, 0.8], labels: [1, 0, 1],
        })}>Request positives at .2 and .8 with a negative at .5</button>}
    </div>
    {draft.mode === 'generated' && <p className="pac-caption">
      Each of the four setups above describes a change from the state currently applied, so choosing one
      replaces any edits you have typed but not yet applied. Adding .01 and .99 is an exact null when both
      lie outside the applied target. If the target includes either point, its generated label changes too.
    </p>}

    {draft.mode === 'generated' && <div className="pac-controls">
      <NumberField label="RNG seed" value={seedDraft} min={0} max={999999} decimals={0}
        onChange={setSeedDraft} />
      <NumberField label="how many points to draw" value={sizeDraft} min={1} max={24} decimals={0}
        onChange={setSizeDraft} />
      <div className="pac-buttons">
        <button type="button"
          onClick={() => state.suggest({ ...draft, seed: seedDraft, drawSize: sizeDraft,
            points: seededUniformSample(seedDraft, sizeDraft) })}>
          Draw a fresh independent sample
        </button>
      </div>
    </div>}
    {draft.mode === 'generated' && <p className="pac-caption">
      The draw uses a mulberry32 generator in your browser, seeded by the number above and reproducible from
      it. It is <em>not</em> NumPy's generator: the retained seed-41 experiment in figure 9 cannot be
      reproduced here, and this control never claims to. A drawn sample is an input like any other — you still
      change it and inspect the recalculated result.
    </p>}
    {problem && <p className="pac-note" role="status">{problem}</p>}

    {appliedExperiment && <>
      <NumberLine geometry={stripGeometry({ experiment: appliedExperiment })} showStrips={appliedExperiment.strips.length > 0}
        caption={`The state currently applied: target [${asInput(applied.target[0])}, ${asInput(applied.target[1])}], `
          + `${applied.points.length} observation${applied.points.length === 1 ? '' : 's'}, fit `
          + (appliedExperiment.interval
            ? `[${appliedExperiment.interval[0]}, ${appliedExperiment.interval[1]}]`
            : 'the empty positive region')}
        describe={'A unit line carrying the target band, the fitted band, the disagreement strips and the '
          + 'observations as filled circles for positive and open squares for negative.'} />
      <div className="pac-strip">
        <span className="pac-strip-cell">applied training error<b>{fixed(appliedExperiment.empiricalRisk, 6)}</b></span>
        <span className="pac-strip-cell">applied population risk<b>{fixed(appliedExperiment.risk, 6)}</b></span>
      </div>
      {!appliedExperiment.strips.length && <p className="pac-caption">
        ε is at least as large as the target width. This tight learner cannot miss more than that width,
        so every sample meets the risk target; the two-strip sufficient event is unnecessary here.
      </p>}
    </>}
    {state.pending && <p className="pac-pending" role="status">
      The drawing above still shows the applied state. Your edits are held in the fields and will be used the
      moment you apply.
    </p>}

    <LiveResult
      
      
      state={state}
      calculateInputs={(inputs, previous) => (problem
        ? { outcome: null, value: NaN, explain: problem }
        : calculateInputs(inputs, previous))}
      blocked={problem}
      
      
      
      
       />

    {experiment && <div className="pac-reveal">
      <dl>
        <dt>Fit after the change</dt>
        <dd>{experiment.interval
          ? `[${experiment.interval[0]}, ${experiment.interval[1]}]`
          : 'the empty positive region'}</dd>
        <dt>Training error</dt>
        <dd>{fixed(experiment.empiricalRisk, 6)}</dd>
        <dt>Population risk</dt>
        <dd>{fixed(experiment.risk, 6)}</dd>
        <dt>Where it is wrong</dt>
        <dd>{experiment.segments.length
          ? experiment.segments.map(segment =>
            `${segment.kind === 'missed' ? 'missed' : 'invented'} [${round(segment.from, 6)}, ${round(segment.to, 6)}]`).join('; ')
          : 'nowhere — the fit and the target agree exactly'}</dd>
        <dt>Those lengths add to</dt>
        <dd>{fixed(experiment.segmentTotal, 6)}</dd>
        <dt>Meets ε = {asInput(experiment.epsilon)}</dt>
        <dd>{experiment.meetsTarget ? 'yes' : 'no'}</dd>
      </dl>
      <NumberLine geometry={stripGeometry({ experiment })} showStrips={experiment.strips.length > 0}
        caption="The current valid state after the change"
        describe={'The same unit line redrawn for the current valid inputs, with the fitted band and the '
          + 'disagreement strips in their new positions.'} />
    </div>}

    {request && <div className="pac-reveal">
      <dl>
        <dt>Requested labels</dt>
        <dd><BitRow bits={request.pattern} /></dd>
        <dt>Consistent interval</dt>
        <dd>{request.feasible
          ? (request.witness.empty ? 'the empty positive region' : `[${request.witness.interval[0]}, ${request.witness.interval[1]}]`)
          : 'none exists'}</dd>
      </dl>
      <p className="pac-caption">{request.explain}</p>
    </div>}
  </Investigation>;
}
