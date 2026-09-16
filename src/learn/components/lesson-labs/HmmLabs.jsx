import { useState } from 'react';
import {
  MISSING, beliefMoveOutcomes, countFlow, durationModel, emStep, enumeratePaths, envelopes, exitAfter,
  expectedCounts, fixtures, gradeBoundaryConstruction, gradeLegalPathConstruction,
  gradeSmoothedConstruction, hazardOutcome, infer, legalPathGuarantee, limits, pathJointOf,
  pathLegal, pathRankOutcome, recordingTotals, withStart,
} from '../../data/hmm-models.js';
import {
  configurations, decisionChanges, developmentSentences, fittedModels, majority,
  selectedConfigurationIndex, tieAudit, unknownDevelopmentTokens, vocabulary,
} from '../../data/hmm-data.js';
import { DurationBars } from './HmmFigures.jsx';
import {
  Construction, Investigation, NumberField, Prediction, ProbabilityRow, Select, SliderField, StateKey,
  Table, Trellis, TrellisTable, Undefined, exactly, fixed, round, useInvestigation,
} from './HmmShared.jsx';
import './hmm-labs.css';

const weather = fixtures.weather;
const activityOptions = weather.symbolNames
  .map((name, index) => [String(index), name])
  .concat([[String(MISSING), 'missing (the step is kept, the report is not)']]);
const moveDigits = limits.moveDigits;

/* ====================================================== §1 · one path's product */

const pathSetups = [
  { key: 'all-rainy', label: 'All Rainy', path: [0, 0, 0, 0] },
  { key: 'best', label: 'Sunny, Sunny, Sunny, Rainy', path: [1, 1, 1, 0] },
  { key: 'alternating', label: 'Alternating from Rainy', path: [0, 1, 0, 1] },
  { key: 'all-sunny', label: 'All Sunny', path: [1, 1, 1, 1] },
];

export function PathProductLab() {
  const state = useInvestigation({ path: [0, 0, 0, 0] });
  const enumerated = enumeratePaths(weather, fixtures.reports);
  const largest = enumerated.largest.joint;
  const jointOf = path => pathJointOf(weather, path, fixtures.reports);
  /* The rule is `pathRankOutcome` in the model layer, not a comparison written
     here, so the verifier exercises the rule the page actually applies over all
     sixteen paths of several models rather than at the two a designer happened
     to try. */
  const answerFor = draft => ({
    outcomes: { rank: pathRankOutcome(weather, draft.path, fixtures.reports) },
    value: jointOf(draft.path),
  });
  const applied = state.result ? jointOf(JSON.parse(state.result.key).path) : null;
  return <Investigation
    title="Follow one complete story through the model"
    question="Choose a hidden state for each of the four times. Before anything is calculated, commit the joint probability of that whole story with its four reports — start, emit, transition, emit, transition, emit, transition, emit."
    note="Sixteen stories are possible here. Multiplying follows one of them; adding over all of them is what the next section does without ever listing them."
    onReset={state.reset}>
    <StateKey names={weather.stateNames} prefix="hidden state" />
    <fieldset className="hmm-row-group" style={{ '--hmm-row-columns': 4 }}>
      <legend>The hidden state at each time. The reports beneath are fixed at Walk, Shop, Walk, Clean.</legend>
      {state.draft.path.map((value, time) => <Select key={time}
        label={`t = ${time}, report ${weather.symbolNames[fixtures.reports[time]]}`}
        value={String(value)}
        options={weather.stateNames.map((name, index) => [String(index), name])}
        onChange={next => state.edit({
          path: state.draft.path.map((entry, index) => (index === time ? Number(next) : entry)),
        })} />)}
    </fieldset>
    <div className="hmm-buttons">
      {pathSetups.map(setup => <button key={setup.key} type="button"
        onClick={() => state.suggest({ path: setup.path })}>{setup.label}</button>)}
    </div>
    <p className="hmm-state-strip">
      <span>applied path: <b>{JSON.parse(state.result?.key ?? JSON.stringify(state.active)).path.map(index => weather.stateNames[index]).join(' → ')}</b></span>
      <span className={state.pending ? 'is-draft' : undefined}>
        draft path: <b>{state.draft.path.map(index => weather.stateNames[index]).join(' → ')}</b>
      </span>
    </p>
    <Prediction
      groups={[{
        key: 'rank', short: 'this path against the best one',
        prompt: 'Among all sixteen complete stories, this one is:',
        options: [
          ['largest', 'the most probable of them'],
          ['smaller', 'less probable than the best one'],
          ['zero', 'impossible — probability exactly zero'],
        ],
      }]}
      state={state} answerFor={answerFor}
      numeric={{ label: 'Optional: this path’s joint probability with the four reports',
        name: 'the joint probability', tolerance: 1e-9, digits: 9 }}
      committed={shown => JSON.parse(shown.key).path.map(index => weather.stateNames[index]).join(' → ')}
      describe={state.result
        ? `The largest of the sixteen is ${round(largest, 9)}, and they add to ${round(enumerated.evidence, 9)} — the probability of the four reports.`
        : undefined} />
    {state.result && <>
      <Table caption="The eight factors of the applied story, in the order the formula multiplies them"
        headings={['step', 'factor', 'value', 'running product']}
        rows={(() => {
          const path = JSON.parse(state.result.key).path;
          const rows = [];
          let running = 1;
          path.forEach((stateIndex, time) => {
            if (time > 0) {
              const value = weather.transition[path[time - 1]][stateIndex];
              running *= value;
              rows.push([`t = ${time}`, `transition ${weather.stateNames[path[time - 1]]} → ${weather.stateNames[stateIndex]}`,
                round(value, 6), fixed(running, 9)]);
            } else {
              running *= weather.start[stateIndex];
              rows.push(['t = 0', `start in ${weather.stateNames[stateIndex]}`, round(weather.start[stateIndex], 6), fixed(running, 9)]);
            }
            const emission = weather.emission[stateIndex][fixtures.reports[time]];
            running *= emission;
            rows.push([`t = ${time}`, `${weather.stateNames[stateIndex]} emits ${weather.symbolNames[fixtures.reports[time]]}`,
              round(emission, 6), fixed(running, 9)]);
          });
          return rows;
        })()}
        footnote={`Eight factors: one start, four emissions and three transitions. Four reports contain three transitions, not four. The product is ${round(applied, 9)}.`} />
      <Table caption="All sixteen stories, largest first" scroll
        headings={['path', 'joint probability', 'share of the evidence']}
        rows={[...enumerated.paths].sort((left, right) => right.joint - left.joint).map(entry => [
          entry.path.map(index => weather.stateNames[index]).join(' → '),
          fixed(entry.joint, 9), fixed(entry.joint / enumerated.evidence, 6),
        ])}
        rowClass={index => {
          const sorted = [...enumerated.paths].sort((left, right) => right.joint - left.joint);
          const applying = JSON.parse(state.result.key).path;
          return sorted[index].path.every((value, time) => value === applying[time]) ? 'is-chosen' : undefined;
        }}
        footnote="A thousand time steps would give two to the thousandth power rows here. The trellis in the next section computes the same total without building this list. This table scrolls; every one of the sixteen is in it." />
    </>}
  </Investigation>;
}

/* ================================================ §4 · evidence arriving later */

const reportSetups = [
  { key: 'original', label: 'The recording as given: Walk, Shop, Walk, Clean', observations: [0, 1, 0, 2] },
  { key: 'corrected', label: 'Correct the final report to Walk', observations: [0, 1, 0, 0] },
  { key: 'missing', label: 'Keep the second step but lose its report', observations: [0, MISSING, 0, 2] },
  { key: 'deleted', label: 'Delete that step entirely — three reports', observations: [0, 0, 2] },
  { key: 'single', label: 'One report only, so smoothing and filtering must agree', observations: [0] },
];

export function EvidenceLaterLab() {
  const state = useInvestigation({ observations: [...fixtures.reports], query: 1, emissionRainy: null });
  const [freeEdit, setFreeEdit] = useState(false);
  const modelFor = draft => (draft.emissionRainy
    ? { ...weather, emission: [draft.emissionRainy, weather.emission[1]] }
    : weather);
  const answerFor = (draft, active) => {
    const moved = beliefMoveOutcomes(modelFor(active), active.observations,
      modelFor(draft), draft.observations, draft.query);
    return { ...moved, value: moved.after.smoothed };
  };
  const draftModel = modelFor(state.draft);
  const draftResult = infer(draftModel, state.draft.observations);
  const query = Math.min(state.draft.query, state.draft.observations.length - 1);
  const revealed = Boolean(state.result);

  /* The construction task is graded by running inference on what was submitted.
     Loading the setup that happens to solve it still has to be submitted and
     computed; nothing is unlocked by recognising a button. */
  const grade = draft => {
    const verdict = gradeSmoothedConstruction(
      { observations: draft.observations, modelEdited: draft.emissionRainy !== null },
      fixtures.reports, fixtures.smoothedTarget, 1);
    const original = infer(weather, fixtures.reports);
    return {
      solved: verdict.solved,
      conditions: [
        { met: !verdict.modelEdited,
          text: 'The model is unedited \u2014 this task is about changing evidence, not parameters.' },
        { met: verdict.prefixKept,
          text: `Only the final report differs from Walk, Shop, Walk (the first three are ${verdict.prefixKept ? 'unchanged' : 'not all unchanged'}).` },
        { met: verdict.held,
          text: verdict.filtered === null
            ? 'Time 1 filtering has no value for this sequence.'
            : `Time 1 filtering is ${fixed(verdict.filtered, moveDigits)}, ${verdict.held ? 'within' : 'outside'} ${limits.nullTolerance} of the original ${fixed(original.filtered[1][0], moveDigits)}.` },
        { met: verdict.smoothed !== null && verdict.smoothed < fixtures.smoothedTarget,
          text: verdict.smoothed === null
            ? 'Time 1 smoothing has no value for this sequence.'
            : `Time 1 smoothed Rainy is ${fixed(verdict.smoothed, 6)}, which is ${verdict.smoothed < fixtures.smoothedTarget ? 'below' : 'not below'} ${fixtures.smoothedTarget}.` },
      ],
    };
  };

  return <Investigation
    title="Change the future; watch which belief moves and which cannot"
    question={`Rainy at time ${query} has two answers: one conditioned on the reports up to time ${query}, one conditioned on all of them. Predict what each does when you change the recording — separately, because they are separate questions.`}
    note="Filtering at a time can only use reports up to that time, so editing anything later cannot touch it. Smoothing uses the whole recording, so editing anything at all can move it."
    onReset={() => { setFreeEdit(false); state.reset(); }}>
    <StateKey names={weather.stateNames} prefix="hidden state" />
    <fieldset className="hmm-row-group" style={{ '--hmm-row-columns': 2 }}>
      <legend>The reported activities. Edit any of them; add or remove a step.</legend>
      {state.draft.observations.map((value, time) => <Select key={time} label={`report at t = ${time}`}
        value={String(value)} options={activityOptions}
        onChange={next => state.edit({
          observations: state.draft.observations.map((entry, index) => (index === time ? Number(next) : entry)),
        })} />)}
    </fieldset>
    <div className="hmm-row-actions">
      <button type="button" disabled={state.draft.observations.length >= envelopes.toy.maximumTimeSteps}
        onClick={() => state.edit({ observations: [...state.draft.observations, 0] })}>Add a report</button>
      <button type="button" disabled={state.draft.observations.length <= 1}
        onClick={() => state.edit({ observations: state.draft.observations.slice(0, -1) })}>Remove the last report</button>
      <span className="hmm-caption">
        Between 1 and {envelopes.toy.maximumTimeSteps} time steps. A missing report keeps its step and its
        transition; deleting the step does not.
      </span>
    </div>
    <div className="hmm-controls">
      <NumberField label="Which time the two questions are asked about" value={state.draft.query}
        min={0} max={state.draft.observations.length - 1} step={1} decimals={0}
        onChange={value => state.edit({ query: value })} />
    </div>
    <div className="hmm-buttons">
      {reportSetups.map(setup => <button key={setup.key} type="button"
        onClick={() => state.suggest({ observations: [...setup.observations], query: Math.min(state.draft.query, setup.observations.length - 1), emissionRainy: state.draft.emissionRainy })}>
        {setup.label}
      </button>)}
    </div>
    <p className="hmm-state-strip">
      <span>applied: <b>{state.active.observations.map(value => (value === MISSING ? 'missing' : weather.symbolNames[value])).join(', ')}</b></span>
      <span className={state.pending ? 'is-draft' : undefined}>
        draft: <b>{state.draft.observations.map(value => (value === MISSING ? 'missing' : weather.symbolNames[value])).join(', ')}</b>
      </span>
      <span>query time: <b>{query}</b></span>
    </p>
    {state.draft.observations.length <= 6 && <Trellis model={draftModel} observations={state.draft.observations}
      mode="sum" queryTime={query} showValues={revealed}
      label="The forward trellis of the drafted recording"
      describe={revealed
        ? (draftResult.impossible
          ? 'This recording is impossible under the model, so it has no forward masses.'
          : draftResult.filtered.map((row, time) => `At time ${time}, filtered Rainy is ${round(row[0], 6)}.`).join(' '))
        : `The structure of the drafted recording: ${state.draft.observations.length} time steps with reports ${state.draft.observations.map(value => (value === MISSING ? 'missing' : weather.symbolNames[value])).join(', ')} and the queried time highlighted. Cell values appear once a prediction is recorded.`} />}

    <Prediction
      groups={[
        { key: 'filtering', short: `filtering at t = ${query}`,
          prompt: `Moving from the applied recording to the drafted one, P(Rainy at t = ${query} | reports up to t = ${query}):`,
          options: [['rises', 'rises'], ['falls', 'falls'], ['unchanged', 'does not move'], ['undefined', 'has no value at all']] },
        { key: 'smoothing', short: `smoothing at t = ${query}`,
          prompt: `And P(Rainy at t = ${query} | the whole recording):`,
          options: [['rises', 'rises'], ['falls', 'falls'], ['unchanged', 'does not move'], ['undefined', 'has no value at all']] },
      ]}
      state={state} answerFor={answerFor}
      numeric={{ label: 'Optional: the smoothed value after your edit', name: 'the smoothed probability',
        tolerance: 1e-6, digits: 6,
        undefinedNote: 'this recording has zero evidence under the model, so no state has a conditional probability' }}
      requireChange={(draft, active) => JSON.stringify(draft) !== JSON.stringify(active)}
      pendingHint="Both questions compare two recordings, so there has to be a second one. Edit a report, or load one of the setups above, before applying."
      sameQuestion={(left, right) => JSON.parse(left).query === JSON.parse(right).query}
      committed={shown => {
        const draft = JSON.parse(shown.key);
        return `${draft.observations.map(value => (value === MISSING ? 'missing' : weather.symbolNames[value])).join(', ')}, asked about t = ${Math.min(draft.query, draft.observations.length - 1)}`;
      }}
      describe={state.result
        ? `Filtering: ${state.result.answer.before.filtered === null ? 'no value' : fixed(state.result.answer.before.filtered, moveDigits)} → ${state.result.answer.after.filtered === null ? 'no value' : fixed(state.result.answer.after.filtered, moveDigits)}. Smoothing: ${state.result.answer.before.smoothed === null ? 'no value' : fixed(state.result.answer.before.smoothed, moveDigits)} → ${state.result.answer.after.smoothed === null ? 'no value' : fixed(state.result.answer.after.smoothed, moveDigits)}. Both are printed to ${moveDigits} places, so “does not move” is a claim you can read rather than take on trust.`
        : undefined}
      historyLabel="Compared with your previous edit" />

    {revealed && !draftResult.impossible && <TrellisTable model={modelFor(state.active)}
      observations={state.active.observations} mode="sum"
      caption="Every forward cell of the applied recording, which is where the exact values are read on a narrow screen" />}
    {revealed && !draftResult.impossible && <Table
      caption="Every filtered and smoothed row of the applied recording"
      headings={['time and report', ...weather.stateNames.map(name => `filtered ${name}`), ...weather.stateNames.map(name => `smoothed ${name}`)]}
      rows={infer(modelFor(state.active), state.active.observations).filtered.map((row, time) => [
        `${time} ${state.active.observations[time] === MISSING ? 'missing' : weather.symbolNames[state.active.observations[time]]}`,
        ...row.map(value => fixed(value, 6)),
        ...infer(modelFor(state.active), state.active.observations).smoothed[time].map(value => fixed(value, 6)),
      ])}
      rowClass={index => (index === Math.min(state.active.query, state.active.observations.length - 1) ? 'is-chosen' : undefined)}
      footnote="At the last time the two columns agree exactly, because there is no later report left to add." />}
    {revealed && draftResult.impossible && <p className="hmm-note">
      The applied recording has zero evidence under this model, so there is no filtered or smoothed row to print:
      the posterior is <Undefined because="the model gives this sequence no probability mass to condition on" />.
    </p>}

    <Construction attempts={state.attempts} onSubmit={() => state.submit(grade)} task={`Leaving the model alone and keeping Walk, Shop, Walk as the first three reports, change only the final report so that smoothed Rainy at time 1 falls below ${fixtures.smoothedTarget} while filtering at time 1 stays exactly where it was.`}
      grade={grade}
      disabledNote="Model editing is disabled while this task is open, because the task is about what evidence can do, not about what parameters can do." />

    <details>
      <summary>Free exploration: edit the model as well</summary>
      <p className="hmm-caption">
        Outside the construction task the Rainy emission row is editable. For the original Walk, Shop, Walk, Clean recording, replacing it with .2, .3, .5 is a useful
        contrast: the best path keeps its joint mass exactly, and its posterior share changes, because the evidence
        it is divided by changed.
      </p>
      <div className="hmm-buttons">
        <button type="button" className={freeEdit ? 'is-selected' : undefined}
          onClick={() => setFreeEdit(current => !current)}>{freeEdit ? 'Lock the model again' : 'Unlock the model'}</button>
        <button type="button" disabled={!freeEdit}
          onClick={() => state.edit({ emissionRainy: [...fixtures.changedRainyEmission] })}>
          Load the .2, .3, .5 Rainy row
        </button>
        <button type="button" disabled={!freeEdit || state.draft.emissionRainy === null}
          onClick={() => state.edit({ emissionRainy: null })}>Restore the original Rainy row</button>
      </div>
      {freeEdit && <ProbabilityRow label="Rainy emission row" names={weather.symbolNames}
        values={state.draft.emissionRainy ?? weather.emission[0]}
        onApply={row => state.edit({ emissionRainy: row })} />}
      {state.draft.emissionRainy && !draftResult.impossible && <Table
        caption="What a changed emission row does and does not move"
        headings={['quantity', 'original model', 'drafted model']}
        rows={(() => {
          const base = infer(weather, state.draft.observations);
          return [
            ['best path', base.path.map(index => weather.stateNames[index]).join(' → '),
              draftResult.path.map(index => weather.stateNames[index]).join(' → ')],
            ['its joint mass', fixed(base.pathJoint, 9), fixed(draftResult.pathJoint, 9)],
            ['evidence', fixed(base.evidence, 9), fixed(draftResult.evidence, 9)],
            ['its posterior share', fixed(base.pathPosterior, 9), fixed(draftResult.pathPosterior, 9)],
          ];
        })()}
        footnote="A fixed path keeps its joint mass when every start, transition and emission entry used along that path stays fixed. Here, the original winning path uses Rainy only for Clean: changing other entries of that row can change the evidence and hence its posterior share. Other edits can also change the winning path." />}
    </details>
  </Investigation>;
}

/* ============================================ §5 · repair a path in a graph */

const constrained = fixtures.constrained;
const constrainedReports = [0, 0];

export function LegalPathLab() {
  const state = useInvestigation({ path: [0, 0], start: [...constrained.start] });
  const modelFor = draft => withStart(constrained, draft.start);
  const answerFor = draft => {
    const model = modelFor(draft);
    const guarantee = legalPathGuarantee(model, constrainedReports);
    return {
      outcomes: { guarantee: guarantee.outcome },
      value: pathJointOf(model, draft.path, constrainedReports),
      modes: guarantee.modes, best: guarantee.best, modesJoint: guarantee.modesJoint,
    };
  };
  const draftModel = modelFor(state.draft);
  const draftResult = infer(draftModel, constrainedReports);
  const draftLegal = pathLegal(draftModel, state.draft.path);
  const grade = draft => {
    const model = modelFor(draft);
    const { solved, legal, joint } = gradeLegalPathConstruction(
      model, draft.path, constrainedReports, fixtures.legalPathTarget);
    return {
      solved,
      conditions: [
        { met: legal,
          text: legal
            ? `Every edge of ${draft.path.map(index => constrained.stateNames[index]).join(' → ')} exists in this model.`
            : `${draft.path.map(index => constrained.stateNames[index]).join(' → ')} uses an edge the model forbids, so it is not a path at all.` },
        { met: joint >= fixtures.legalPathTarget,
          text: `Its joint probability is ${exactly(joint, 6)}, which is ${joint >= fixtures.legalPathTarget ? 'at least' : 'below'} ${fixtures.legalPathTarget}.` },
      ],
    };
  };
  return <Investigation
    title="Connect the most probable cells, then check whether you have a path"
    question="Three states, two times, one symbol that every state emits with certainty. Choose a state at each time. Before anything is computed, predict whether joining the two highest-marginal cells is guaranteed to give the highest-probability legal path."
    note="Because the single symbol is certain in every state, the observations carry no information here: all of the model's content is in the initial probabilities and the four permitted edges. In the drawing a solid gold edge is the predecessor its destination stored, a dotted hairline is an edge the model forbids, and a gold outline marks a node on the decoded path."
    onReset={state.reset}>
    <StateKey names={constrained.stateNames} prefix="hidden state" />
    <fieldset className="hmm-row-group" style={{ '--hmm-row-columns': 2 }}>
      <legend>The state you are selecting at each time.</legend>
      {state.draft.path.map((value, time) => <Select key={time} label={`state at t = ${time}`} value={String(value)}
        options={constrained.stateNames.map((name, index) => [String(index), name])}
        onChange={next => state.edit({
          path: state.draft.path.map((entry, index) => (index === time ? Number(next) : entry)),
        })} />)}
    </fieldset>
    <ProbabilityRow label="Initial-state probabilities" names={constrained.stateNames}
      values={state.draft.start} onApply={row => state.edit({ start: row })} />
    <div className="hmm-buttons">
      <button type="button" onClick={() => state.suggest({ ...state.draft, start: [...constrained.start] })}>
        The declared prior .4, .35, .25
      </button>
      <button type="button" onClick={() => state.suggest({ ...state.draft, start: [...fixtures.changedConstrainedStart] })}>
        The changed prior .2, .55, .25
      </button>
      <button type="button" onClick={() => state.suggest({ ...state.draft, path: [0, 0] })}>Select A → A</button>
      <button type="button" onClick={() => state.suggest({ ...state.draft, path: [1, 0] })}>Select B → A</button>
    </div>
    <p className="hmm-state-strip">
      <span>selected: <b>{state.draft.path.map(index => constrained.stateNames[index]).join(' → ')}</b></span>
      <span>legal: <b>{draftLegal ? 'yes' : 'no — it uses a forbidden edge'}</b></span>
      <span>marginal modes: <b>{draftResult.marginalModes.map(index => constrained.stateNames[index]).join(' → ')}</b></span>
    </p>
    <Trellis model={draftModel} observations={constrainedReports} mode="max" showForbidden
      showValues={Boolean(state.result)} path={draftLegal ? state.draft.path : null}
      label="The constrained two-step graph with the drafted prior"
      describe={`Marginals are ${draftResult.smoothed[0].map(value => round(value, 4)).join(', ')} at the first time and ${draftResult.smoothed[1].map(value => round(value, 4)).join(', ')} at the second. Dotted hairlines mark forbidden transitions.${state.result ? '' : ' Path scores appear once a prediction is recorded.'}`} />

    <Prediction
      groups={[{
        key: 'guarantee', short: 'joining the two highest cells',
        prompt: 'For the prior now drafted, does joining the highest-marginal cell at each time give the highest-probability legal path?',
        options: [['yes', 'yes, it does here'], ['no', 'no — it does not, or is not even a path']],
      }]}
      state={state} answerFor={answerFor}
      numeric={{ label: 'Optional: the joint probability of the path you selected', name: 'that joint probability',
        tolerance: 1e-10, digits: 10 }}
      committed={shown => {
        const draft = JSON.parse(shown.key);
        return `${draft.path.map(index => constrained.stateNames[index]).join(' → ')}, prior ${draft.start.map(value => round(value, 4)).join(', ')}`;
      }}
      describe={state.result
        ? `The marginal modes are ${state.result.answer.modes.map(index => constrained.stateNames[index]).join(' → ')} with joint probability ${exactly(state.result.answer.modesJoint, 6)}; the highest-probability legal path is ${state.result.answer.best.path.map(index => constrained.stateNames[index]).join(' → ')} at ${round(state.result.answer.best.joint, 6)}.`
        : undefined} />

    {state.result && <Table caption="Every two-step selection, legal or not"
      headings={['selection', 'is it a path the model permits?', 'joint probability', 'expected correct positions']}
      rows={enumeratePaths(modelFor(state.active), constrainedReports).paths.map(entry => [
        entry.path.map(index => constrained.stateNames[index]).join(' → '),
        pathLegal(modelFor(state.active), entry.path) ? 'yes' : 'no — forbidden edge',
        exactly(entry.joint, 6),
        round(entry.path.reduce((total, value, time) => total + infer(modelFor(state.active), constrainedReports).smoothed[time][value], 0), 6),
      ])}
      rowClass={index => {
        const entries = enumeratePaths(modelFor(state.active), constrainedReports).paths;
        return entries[index].path.every((value, time) => value === state.active.path[time]) ? 'is-chosen' : undefined;
      }}
      footnote="The pointwise modes can win the last column while scoring exactly zero in the one before it. Better expected position count and validity as a joint path are separate properties, and a constrained decoder that maximised the last column over legal paths would be a third rule again." />}

    <Construction attempts={state.attempts} onSubmit={() => state.submit(grade)}
      task={`Select a pair of states that the model actually permits and whose joint probability is at least ${fixtures.legalPathTarget}. Editing the initial row is allowed; both conditions are computed from what you submit.`}
      grade={grade} />
  </Investigation>;
}

/* ================================================ §6 · boundaries and counts */

const boundarySetups = [
  { key: 'declared', label: 'Two two-step sessions', recordings: [[0, 1], [0, 2]] },
  { key: 'joined', label: 'One four-step recording', recordings: [[0, 1, 0, 2]] },
  { key: 'uneven', label: 'A one-step and a three-step session', recordings: [[0], [1, 0, 2]] },
  { key: 'duplicated', label: 'The two sessions, duplicated', recordings: [[0, 1], [0, 2], [0, 1], [0, 2]] },
];
const wholeOptions = count => Array.from({ length: count + 1 }, (unused, index) => [String(index), String(index)])
  .concat([['more', `more than ${count}`]]);

const declaredLengths = [2, 2];

export function BoundaryCountLab() {
  const state = useInvestigation({ recordings: [[0, 1, 0, 2]] });
  const bound = (value, maximum) => (value > maximum ? 'more' : String(value));
  const answerFor = draft => {
    const totals = recordingTotals(draft.recordings);
    return {
      outcomes: {
        starts: bound(totals.starts, 4), transitions: bound(totals.transitions, 6), emissions: bound(totals.emissions, 8),
      },
      value: expectedCounts(weather, draft.recordings).logLikelihood,
      totals,
    };
  };
  const grade = draft => {
    const verdict = gradeBoundaryConstruction(draft.recordings, declaredLengths);
    return {
      solved: verdict.solved,
      conditions: [
        { met: verdict.totalsMatch,
          text: `The event totals are ${verdict.totals.starts} starts, ${verdict.totals.transitions} transitions and ${verdict.totals.emissions} emissions; the declared recordings need ${verdict.expected.starts}, ${verdict.expected.transitions} and ${verdict.expected.emissions}.` },
        { met: verdict.structureMatches,
          text: `The session lengths are ${verdict.lengths.join(' and ')}; the declared recordings are ${declaredLengths.join(' and ')}. A one-and-three split reaches the same three totals and is still a different claim about what was recorded.` },
      ],
    };
  };
  const draftFlow = countFlow(weather, state.draft.recordings);
  const revealed = Boolean(state.result);
  const editReport = (recording, position, value) => state.edit({
    recordings: state.draft.recordings.map((row, index) => (index === recording
      ? row.map((entry, slot) => (slot === position ? value : entry)) : row)),
  });
  const totalReports = state.draft.recordings.reduce((total, row) => total + row.length, 0);
  return <Investigation
    title="Move the boundary and watch the event totals follow"
    question="Predict how many starts, within-recording transitions and emissions the E-step should count, before any fractional flow appears. Then move the boundary and predict again."
    note="Every recording contributes one start, one emission per observed report, and one transition per step inside it. A length-one recording contributes a start and an emission and no transition at all."
    onReset={state.reset}>
    {state.draft.recordings.map((recording, index) => <fieldset key={index} className="hmm-row-group"
      style={{ '--hmm-row-columns': 2 }}>
      <legend>Recording {index + 1}, {recording.length} step{recording.length === 1 ? '' : 's'}</legend>
      {recording.map((value, position) => <Select key={position} label={`t = ${position}`} value={String(value)}
        options={activityOptions} onChange={next => editReport(index, position, Number(next))} />)}
    </fieldset>)}
    <div className="hmm-row-actions">
      <button type="button" disabled={state.draft.recordings.length >= limits.maximumSequences}
        onClick={() => state.edit({ recordings: [...state.draft.recordings, [0]] })}>Add a recording</button>
      <button type="button" disabled={state.draft.recordings.length <= 1}
        onClick={() => state.edit({ recordings: state.draft.recordings.slice(0, -1) })}>Remove the last recording</button>
      <button type="button" disabled={totalReports >= envelopes.toy.maximumTimeSteps}
        onClick={() => state.edit({
          recordings: state.draft.recordings.map((row, index) => (index === state.draft.recordings.length - 1 ? [...row, 0] : row)),
        })}>Add a step to the last recording</button>
      <button type="button" disabled={state.draft.recordings.at(-1).length <= 1}
        onClick={() => state.edit({
          recordings: state.draft.recordings.map((row, index) => (index === state.draft.recordings.length - 1 ? row.slice(0, -1) : row)),
        })}>Remove a step from the last recording</button>
    </div>
    <div className="hmm-buttons">
      {boundarySetups.map(setup => <button key={setup.key} type="button"
        onClick={() => state.suggest({ recordings: setup.recordings.map(row => [...row]) })}>{setup.label}</button>)}
    </div>
    <p className="hmm-state-strip">
      <span>applied: <b>{state.active.recordings.map(row => `[${row.map(value => (value === MISSING ? 'missing' : weather.symbolNames[value])).join(' ')}]`).join(' ')}</b></span>
      <span className={state.pending ? 'is-draft' : undefined}>
        draft: <b>{state.draft.recordings.map(row => `[${row.map(value => (value === MISSING ? 'missing' : weather.symbolNames[value])).join(' ')}]`).join(' ')}</b>
      </span>
    </p>

    <Prediction
      groups={[
        { key: 'starts', short: 'expected starts',
          prompt: 'The expected start counts will add to:', options: wholeOptions(4) },
        { key: 'transitions', short: 'expected transitions',
          prompt: 'The expected transition counts will add to:', options: wholeOptions(6) },
        { key: 'emissions', short: 'expected emissions',
          prompt: 'The expected emission counts will add to:', options: wholeOptions(8) },
      ]}
      state={state} answerFor={answerFor}
      numeric={{ label: 'Optional: the training log likelihood of these recordings', name: 'the log likelihood',
        tolerance: 1e-6, digits: 6 }}
      committed={shown => JSON.parse(shown.key).recordings
        .map(row => `[${row.map(value => (value === MISSING ? 'missing' : weather.symbolNames[value])).join(' ')}]`).join(' ')}
      historyLabel="Compared with your previous boundary" />

    {revealed && <>
      {/* The long list of counts is the LAST column on purpose: every other cell
          holds a number and must not break, and the stylesheet lets only the
          last cell wrap. In the middle it ran off the end of its own scroll box
          and the emission row was cut mid-entry. */}
      <Table caption="Where the fractional mass went, and the whole numbers it adds to"
        headings={['event group', 'total', 'exact expected counts']}
        rows={[
          ['starts', fixed(draftFlow.totals.startMass, 6),
            draftFlow.start.map(entry => `${entry.name} ${fixed(entry.mass, 6)}`).join(', ')],
          ['transitions', fixed(draftFlow.totals.edgeMass, 6),
            draftFlow.edges.flat().map(entry => `${weather.stateNames[entry.origin]}→${weather.stateNames[entry.destination]} ${fixed(entry.mass, 6)}`).join(', ')],
          ['emissions', fixed(draftFlow.totals.symbolMass, 6),
            draftFlow.symbols.flat().filter(entry => entry.mass > 0).map(entry => `${weather.stateNames[entry.state]}/${weather.symbolNames[entry.symbol]} ${fixed(entry.mass, 6)}`).join(', ')],
        ]}
        footnote={`Individual counts may be fractional, while the totals count actual events: ${draftFlow.totals.starts} starts, ${draftFlow.totals.transitions} transitions, ${draftFlow.totals.emissions} emissions.`} />
      {(() => {
        const updated = emStep(weather, state.active.recordings);
        return <><Table caption="The M-step: updated start and transition rows"
          headings={['row', ...weather.stateNames, 'log likelihood before', 'after']}
          rows={[
            ['initial', ...updated.model.start.map(value => fixed(value, 6)),
              fixed(updated.logLikelihoodBefore, 6), fixed(updated.logLikelihoodAfter, 6)],
            ...updated.model.transition.map((row, index) => [
              `from ${weather.stateNames[index]}`, ...row.map(value => fixed(value, 6)), '', '',
            ]),
          ]}
          footnote="Duplicating the whole dataset doubles every expected count and leaves these updated rows identical — a useful exact null, and one you can reach with the duplication setup above." />
          <Table caption="The M-step: updated emission rows"
            headings={['hidden state', ...weather.symbolNames]}
            rows={updated.model.emission.map((row, index) => [
              weather.stateNames[index], ...row.map(value => fixed(value, 6)),
            ])}
            footnote="Normalize each state's expected symbol counts by that state's total observed emissions. If that total is zero, these observations cannot identify the row; this implementation retains its previous values." />
        </>;
      })()}
    </>}

    <Construction attempts={state.attempts} onSubmit={() => state.submit(grade)}
      task="Two independent two-step sessions were recorded, and the four reports arrived concatenated. Restore the boundary. Both the event totals and the session structure are checked, because a one-and-three split reaches the same totals and is a different claim about what happened."
      grade={grade} />
  </Investigation>;
}

/* ====================================================== §9 · dwell time */

/* Ascending, so a reader can follow the mean climbing down the column: the
   declared contrasts, practice 7's value, and the absorbing end. */
const durationSettings = [...new Set([...fixtures.durationContrasts, fixtures.durationPractice, 1])]
  .sort((left, right) => left - right);

export function DurationLab() {
  const state = useInvestigation({ stay: fixtures.durationDefault });
  const answerFor = draft => {
    const model = durationModel(draft.stay);
    return {
      outcomes: { hazard: hazardOutcome(draft.stay, 0, 10) },
      value: model.mean,
      model, afterOne: exitAfter(draft.stay, 0), afterTen: exitAfter(draft.stay, 10),
    };
  };
  const model = durationModel(state.draft.stay);
  return <Investigation
    title="A self-transition is a duration assumption you can look at"
    question="Choose a self-transition probability and inspect the duration bars. Predict how the chance of leaving on the next step compares between a state that has just been entered and one that has already lasted ten steps."
    note="The first eight bars need not add to one: the ninth bar holds every duration of nine or more, and that tail mass is part of the assumption rather than an inconvenience to normalise away."
    onReset={state.reset}>
    <div className="hmm-controls">
      <SliderField label="Self-transition probability a" value={state.draft.stay} min={0} max={1} step={0.01}
        decimals={2} onChange={value => state.edit({ stay: value })}
        hint="At a = 1 the state is absorbing: it never leaves, and there is no finite mean duration to report." />
    </div>
    <div className="hmm-buttons">
      {[0, 0.7, 0.8, 0.95, 1].map(value => <button key={value} type="button"
        onClick={() => state.suggest({ stay: value })}>a = {value}</button>)}
    </div>
    <p className="hmm-state-strip">
      <span>drafted a: <b>{round(state.draft.stay, 2)}</b></span>
      {/* The exit probability and the mean are both graded here, so neither is
          printed until the prediction is committed. The bars stay visible,
          because looking at the shape is the work the question asks for. */}
      <span>next-step exit probability: <b>{state.result ? round(model.exitProbability, 6) : 'recorded after your prediction'}</b></span>
      <span>mean duration: <b>{state.result ? (model.absorbing ? 'no finite mean' : round(model.mean, 6)) : 'recorded after your prediction'}</b></span>
    </p>
    <DurationBars stay={state.draft.stay} label={`Duration distribution at a = ${round(state.draft.stay, 2)}`}
      describe={`Bars for durations one to eight with probabilities ${model.probabilities.map(value => round(value, 4)).join(', ')}, and a final bar of ${round(model.tailMass, 4)} for every duration of nine or more.`} />

    <Prediction
      groups={[{
        key: 'hazard', short: 'the chance of leaving next',
        prompt: 'Compared with a state just entered, a state that has already lasted ten steps leaves on the next step with probability:',
        options: [['higher', 'higher'], ['lower', 'lower'], ['same', 'exactly the same'], ['undefined', 'undefined — that elapsed history is impossible']],
      }]}
      state={state} answerFor={answerFor}
      numeric={{ label: 'Optional: the mean duration', name: 'the mean duration', tolerance: 1e-6, digits: 6,
        undefinedNote: 'an absorbing state never leaves, so its duration has no finite mean' }}
      committed={shown => `a = ${round(JSON.parse(shown.key).stay, 2)}`}
      describe={state.result
        ? state.result.answer.afterTen === null
          ? 'At a = 0 the state leaves on its first transition. Surviving ten steps has probability zero, so the conditional chance of leaving after that impossible history is undefined.'
          : `Just after entry the exit probability is ${round(state.result.answer.afterOne, 6)}; after ten steps it is ${round(state.result.answer.afterTen, 6)}. That constant hazard, on histories with positive survival probability, is what makes this family geometric.`
        : undefined} />

    {state.result && <Table caption="The same assumption at the three declared settings, the value practice 7 asks for, and the absorbing end"
      headings={['self-transition a', 'mean duration', 'P(duration = 3)', 'next-step exit probability', 'mass beyond eight steps']}
      rows={durationSettings.map(stay => {
        const entry = durationModel(stay);
        return [
          round(stay, 2), entry.absorbing ? 'no finite mean' : round(entry.mean, 6),
          exactly(entry.probabilities[2], 6), round(entry.exitProbability, 6), exactly(entry.tailMass, 6),
        ];
      })}
      rowClass={index => (durationSettings[index] === state.active.stay ? 'is-chosen' : undefined)}
      footnote="An absorbing state is reported as having no finite mean rather than as a very large one: 1/(1−a) is not a number at a = 1, and drawing it as a tall geometric curve would be a different claim." />}
  </Investigation>;
}

/* =============================================== §8 · the real sentences */

const labelNames = ['Noun', 'Verb', 'Other'];
/* One decimal place, matching every mention of these strengths in the prose
   and in the results table. Rendered from the raw number, 1.0 printed as "1". */
const smoothingLabel = value => value.toFixed(1);
const smoothingChoices = fittedModels.map((fit, index) =>
  [String(index), `smoothing ${smoothingLabel(fit.smoothing)}`]);

function beliefAt(row, position) {
  return [row.beliefs[position * 3], row.beliefs[position * 3 + 1], row.beliefs[position * 3 + 2]];
}
const configurationFor = (fitIndex, method) =>
  configurations.findIndex(entry => entry.fitIndex === fitIndex && entry.method === method);

export function RealTaggingLab() {
  const state = useInvestigation({ sentence: 27, fitIndex: 1 });
  const [reveal, setReveal] = useState(false);
  /* Both start at sentence 0, token 0, which is neither a repair nor a break.
     Defaulting them to a correct answer would hand over half the task on first
     paint, which is exactly what a construction task must not do. */
  const [repairGuess, setRepairGuess] = useState({ sentence: 0, position: 0 });
  const [breakGuess, setBreakGuess] = useState({ sentence: 0, position: 0 });
  const [spelling, setSpelling] = useState({ position: 0, word: '' });

  const indicesFor = draft => ({
    lexical: configurationFor(draft.fitIndex, 'lexical'),
    hmm: configurationFor(draft.fitIndex, 'hmm'),
  });
  const answerFor = draft => {
    const { lexical, hmm } = indicesFor(draft);
    const lexicalCorrect = configurations[lexical].rows[draft.sentence].correct;
    const hmmCorrect = configurations[hmm].rows[draft.sentence].correct;
    const outcome = lexicalCorrect === hmmCorrect ? 'tie' : (hmmCorrect > lexicalCorrect ? 'hmm' : 'lexical');
    return { outcomes: { winner: outcome }, value: hmmCorrect - lexicalCorrect, lexicalCorrect, hmmCorrect };
  };
  const draft = state.draft;
  const sentence = developmentSentences[draft.sentence];
  const { lexical: lexicalIndex, hmm: hmmIndex } = indicesFor(draft);
  const lexicalRow = configurations[lexicalIndex].rows[draft.sentence];
  const hmmRow = configurations[hmmIndex].rows[draft.sentence];
  const tiedSentences = tieAudit.configurations[draft.fitIndex].tiedSentences.map(entry => entry.sentence);
  const isTied = tiedSentences.includes(draft.sentence);

  const identify = () => {
    const repairs = decisionChanges.repairs;
    const breaks = decisionChanges.breaks;
    const foundRepair = repairs.some(entry => entry.sentence === repairGuess.sentence && entry.position === repairGuess.position);
    const foundBreak = breaks.some(entry => entry.sentence === breakGuess.sentence && entry.position === breakGuess.position);
    return {
      solved: foundRepair && foundBreak,
      conditions: [
        { met: foundRepair,
          text: `Sentence ${repairGuess.sentence}, token ${repairGuess.position} is ${foundRepair ? '' : 'not '}a position where the HMM corrected a lexical mistake at smoothing 1.0.` },
        { met: foundBreak,
          text: `Sentence ${breakGuess.sentence}, token ${breakGuess.position} is ${foundBreak ? '' : 'not '}a position where the HMM broke a correct lexical decision at smoothing 1.0.` },
      ],
    };
  };
  const [identified, setIdentified] = useState(null);

  // The free spelling edit runs inference with the STORED model and vocabulary.
  const editedSymbols = sentence.symbols.map((symbol, index) => {
    if (index !== spelling.position || spelling.word.trim() === '') return symbol;
    const found = vocabulary.indexOf(spelling.word.trim().toLowerCase());
    return found === -1 ? 0 : found;
  });
  const fitted = fittedModels[draft.fitIndex];
  const storedModel = {
    stateNames: labelNames,
    symbolNames: vocabulary,
    start: fitted.start,
    transition: fitted.transition,
    emission: fitted.emission,
  };
  // Compare both spellings under the same tie rule. A stored equally optimal
  // path may differ by floating-point tie resolution without any input edit.
  const baselinePath = infer(storedModel, sentence.symbols, envelopes.real).path;
  const editedPath = infer(storedModel, editedSymbols, envelopes.real).path;
  const symbolsMoved = editedSymbols.some((value, index) => value !== sentence.symbols[index]);

  return <Investigation
    title="Two decision rules, the same fitted counts, real sentences"
    question="Choose a development sentence and a smoothing strength. Before any labels appear, predict which of the two rules gets more of that sentence's tokens right."
    role={{ kind: 'recorded', text: 'These are recorded results read from saved predictions. Nothing here refits a tagger in your browser, and the 40 reserved sentences are never scored.' }}
    note={`Both rules read the same fitted counts. The lexical rule multiplies each word's emission probability by its state's training frequency and takes the largest; the HMM adds the learned start and transition probabilities and takes the Viterbi path. Of 341 development tokens, ${unknownDevelopmentTokens} map to the single unknown symbol.`}
    onReset={() => { setReveal(false); setIdentified(null); setRepairGuess({ sentence: 0, position: 0 }); setBreakGuess({ sentence: 0, position: 0 }); setSpelling({ position: 0, word: '' }); state.reset(); }}>
    <div className="hmm-controls is-wide">
      <Select label="Development sentence" value={String(draft.sentence)}
        options={developmentSentences.map(entry => [String(entry.index),
          `${entry.index}: ${entry.tokens.slice(0, 6).join(' ')}${entry.tokens.length > 6 ? ' …' : ''}`])}
        onChange={value => { setReveal(false); setSpelling({ position: 0, word: '' }); state.edit({ sentence: Number(value) }); }} />
      <Select label="Smoothing strength, held fixed while you switch decoders" value={String(draft.fitIndex)}
        options={smoothingChoices} onChange={value => { setReveal(false); state.edit({ fitIndex: Number(value) }); }} />
    </div>
    <p className="hmm-state-strip">
      <span>source id: <b>{sentence.id}</b></span>
      <span>tokens: <b>{sentence.tokens.length}</b></span>
      <span>unknown words: <b>{sentence.unknownPositions.length}</b></span>
    </p>

    <Prediction
      groups={[{
        key: 'winner', short: 'more correct tokens on this sentence',
        prompt: 'On this sentence, at this smoothing strength, more tokens are tagged correctly by:',
        options: [['lexical', 'the lexical baseline'], ['hmm', 'the HMM'], ['tie', 'neither — they tie']],
      }]}
      state={state} answerFor={answerFor}
      numeric={{ label: 'Optional: how many more tokens the HMM gets right (negative if fewer)',
        name: 'the difference in correct tokens', tolerance: 0, digits: 0 }}
      committed={shown => {
        const committedDraft = JSON.parse(shown.key);
        return `sentence ${committedDraft.sentence}, smoothing ${smoothingLabel(fittedModels[committedDraft.fitIndex].smoothing)}`;
      }}
      describe={state.result
        ? `The lexical rule gets ${state.result.answer.lexicalCorrect} of ${sentence.tokens.length}; the HMM gets ${state.result.answer.hmmCorrect}.`
        : undefined}
      historyLabel="Compared with your previous sentence or strength" />

    {state.result && <>
      <div className="hmm-buttons">
        <button type="button" className={reveal ? 'is-selected' : undefined}
          onClick={() => setReveal(current => !current)}>
          {reveal ? 'Hide the reference labels' : 'Reveal the reference labels and the original tags'}
        </button>
      </div>
      <div className="hmm-tokens">
        {sentence.tokens.map((token, position) => {
          const truth = sentence.truth[position];
          const lexicalTag = lexicalRow.predicted[position];
          const hmmTag = hmmRow.predicted[position];
          const unknown = sentence.symbols[position] === 0;
          const classes = ['hmm-token'];
          if (unknown) classes.push('is-unknown');
          if (reveal && lexicalTag !== truth && hmmTag === truth) classes.push('is-repair');
          if (reveal && lexicalTag === truth && hmmTag !== truth) classes.push('is-break');
          return <span key={position} className={classes.join(' ')}>
            <b>{token}</b>
            <span className={`hmm-tag${reveal ? (lexicalTag === truth ? ' is-hit' : ' is-miss') : ''}`}>
              lex {labelNames[lexicalTag]}
            </span>
            <span className={`hmm-tag${reveal ? (hmmTag === truth ? ' is-hit' : ' is-miss') : ''}`}>
              hmm {labelNames[hmmTag]}
            </span>
            {reveal
              ? <span className="hmm-tag">ref {labelNames[truth]} · {sentence.upos[position]}</span>
              : <span className="hmm-tag">ref concealed</span>}
            {unknown && <span className="hmm-tag">UNKNOWN symbol</span>}
          </span>;
        })}
      </div>
      <Table caption="Per-token beliefs under each rule, for the applied sentence" scroll
        headings={['position', 'token', 'symbol', ...labelNames.map(name => `lexical P(${name})`), ...labelNames.map(name => `HMM smoothed P(${name})`)]}
        rows={sentence.tokens.map((token, position) => [
          String(position), token,
          sentence.symbols[position] === 0 ? 'UNKNOWN' : vocabulary[sentence.symbols[position]],
          ...beliefAt(lexicalRow, position).map(value => fixed(value, 6)),
          ...beliefAt(hmmRow, position).map(value => fixed(value, 6)),
        ])}
        footnote="Two different unknown words carry the identical symbol, so the model cannot use their distinct spellings as evidence. A high belief is confidence under this model on this coarse task, not a guarantee." />
      {isTied && <p className="hmm-note">
        This sentence is one of the three where two complete paths have <strong>exactly</strong> equal probability —
        adjacent unknown words whose two states can be swapped without changing a single factor. The path shown is one
        of them; the other is equally optimal, and a decoder that broke the tie the other way would score differently
        on this sentence.
      </p>}
    </>}

    {/* The fields come first. Below the submit button a learner meets the
        action before the inputs it grades, and presses it against two
        untouched defaults. */}
    <div className="hmm-controls is-wide">
      <NumberField label="Repair — sentence index" value={repairGuess.sentence} min={0} max={39} step={1} decimals={0}
        onChange={value => { setIdentified(null); setRepairGuess(current => ({ ...current, sentence: value })); }} />
      <NumberField label="Repair — token position" value={repairGuess.position} min={0} max={14} step={1} decimals={0}
        onChange={value => { setIdentified(null); setRepairGuess(current => ({ ...current, position: value })); }} />
      <NumberField label="Break — sentence index" value={breakGuess.sentence} min={0} max={39} step={1} decimals={0}
        onChange={value => { setIdentified(null); setBreakGuess(current => ({ ...current, sentence: value })); }} />
      <NumberField label="Break — token position" value={breakGuess.position} min={0} max={14} step={1} decimals={0}
        onChange={value => { setIdentified(null); setBreakGuess(current => ({ ...current, position: value })); }} />
    </div>
    <Construction attempts={identified ? [identified] : []} onSubmit={() => setIdentified({ verdict: identify() })}
      task="Using the four fields above and the selected smoothing 1.0 comparison, name one position where the HMM repaired a lexical mistake, and one where it broke a correct lexical decision. Both are checked against the recorded per-token outcomes."
      submitLabel="Check both positions" />

    {identified?.verdict.solved && <Table
      caption="Only now: the aggregate the two rules reach over all forty development sentences"
      headings={['smoothing', 'decoder', 'correct tokens of 341', 'whole sentences of 40']}
      rows={configurations.map(entry => [
        String(entry.smoothing), entry.label, String(entry.correct), String(entry.sentencesCorrect),
      ])}
      rowClass={index => (index === selectedConfigurationIndex ? 'is-chosen' : undefined)}
      footnote={`Always predicting the majority label gets ${majority.correct} of ${majority.tokens}. The selected HMM repairs ${decisionChanges.repairs.length} lexical decisions and breaks ${decisionChanges.breaks.length}, a net gain of ${decisionChanges.repairs.length - decisionChanges.breaks.length} tokens. These are development results used to select among four candidates, not held-out estimates.`} />}

    <details>
      <summary>Change one spelling and see what the encoding does with it</summary>
      <p className="hmm-caption">
        Inference runs with the stored model and the stored vocabulary. Replacing one unknown spelling with another
        unknown spelling is an exact input-encoding null: the symbols are identical, so nothing downstream can move.
        An edited token has no reference label, and none is invented for it.
      </p>
      <div className="hmm-controls is-wide">
        <NumberField label="Token position to respell" value={spelling.position} min={0}
          max={sentence.tokens.length - 1} step={1} decimals={0}
          onChange={value => setSpelling(current => ({ ...current, position: value }))} />
        <Select label="A replacement spelling" value={spelling.word}
          options={[['', `leave “${sentence.tokens[Math.min(spelling.position, sentence.tokens.length - 1)]}” as it is`],
            ['zzqwx', 'zzqwx — an unknown spelling'],
            ['qxvblorp', 'qxvblorp — a different unknown spelling'],
            ['the', 'the — a known word'],
            ['said', 'said — another known word']]}
          onChange={value => setSpelling(current => ({ ...current, word: value }))} />
      </div>
      <p className="hmm-state-strip">
        <span>symbols moved: <b>{symbolsMoved ? 'yes' : 'no'}</b></span>
        <span>path moved: <b>{editedPath.some((value, index) => value !== baselinePath[index]) ? 'yes' : 'no'}</b></span>
        <span>decoded: <b>{editedPath.map(value => labelNames[value]).join(' ')}</b></span>
      </p>
    </details>
  </Investigation>;
}
