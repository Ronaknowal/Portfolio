import { useState } from 'react';
import {
  Dag, Distribution, Investigation, MixtureLane, NumberField, Prediction, Select, StateChoice, Table,
  fixed, ratio, round, useInvestigation,
} from './BayesNetShared.jsx';
import {
  alarmNetwork, changeDirection, checkGraph, classPosterior, controlSteps, dSeparation, fixtures,
  graphNodes, laneGeometry, limits, purchaseComparison, queryPosterior, serviceModel,
} from '../../data/bayesnet-models.js';
import { protocol, trainingModels, validationSpecimens } from '../../data/bayesnet-data.js';

/** The four investigations for the Bayesian-networks lesson.
 *
 * Every one of them keeps the same contract: a prediction is recorded before
 * anything is revealed, the grader runs on the committed draft rather than on
 * live state, at least one control changes something the prose has not already
 * resolved, and every relevant edit retires the verdict.
 *
 * One rule matters more here than in most lessons. A d-separation question has
 * an exact answer, so a grader that disagrees with the drawing marks correct
 * reasoning wrong. Both come from the single `dSeparation` call below; the
 * highlighted path list and the graded verdict are two views of one returned
 * object and cannot drift apart.
 */

/* Every graded comparison below calls `changeDirection` or `purchaseComparison`
   from the model module. Both are exercised at their degenerate inputs by
   scripts/verify-bayesnet-models.mjs; a rule that only existed in this file
   would be a rule nothing could assert.

   THE GRADED QUANTITY IS NOT ON SCREEN BEFORE THE PREDICTION IS RECORDED.

   Every readout computed from the *draft* -- the state the learner is about to
   be graded on -- lives inside a `bn-graded` element that is rendered only once
   `state.result` exists. What stays visible is the *baseline*: the state the
   prediction is being made against, which a learner needs in order to answer
   "higher, lower or unchanged" at all. The two are different states the moment
   an edit is made, and an edit is required before Apply.

   This is not three local fixes. `bn-graded` is the marker the browser review
   counts: it asserts zero such elements in every investigation before
   commitment and at least one afterwards, so a new readout added without the
   gate fails the check rather than quietly restoring the leak. */

/* ============================================ I1 · evidence and explanations */

const EVIDENCE_NODES = ['E', 'A', 'J', 'M'];

const evidenceOf = draft => Object.fromEntries(EVIDENCE_NODES
  .filter(node => draft.states[node] !== 'unknown')
  .map(node => [node, Number(draft.states[node])]));

const networkOf = draft => ({
  ...alarmNetwork,
  chance: {
    B: { '': draft.burglaryPrior },
    E: { '': draft.earthquakePrior },
    A: { ...alarmNetwork.chance.A },
    J: { 0: draft.johnWhenQuiet, 1: draft.johnWhenAlarm },
    M: { 0: draft.maryWhenQuiet, 1: draft.maryWhenAlarm },
  },
});

const evidenceInitial = {
  states: { E: 'unknown', A: 'unknown', J: '1', M: '1' },
  burglaryPrior: 0.001, earthquakePrior: 0.002,
  johnWhenQuiet: 0.05, johnWhenAlarm: 0.9,
  maryWhenQuiet: 0.01, maryWhenAlarm: 0.7,
};

export function EvidenceLab() {
  const state = useInvestigation(evidenceInitial);
  const { draft, active } = state;
  const shown = queryPosterior(networkOf(draft), evidenceOf(draft));
  /* The state the prediction is measured against. Before a commitment that is
     whatever is currently applied; after one it is the state the commit
     replaced, which is what the verdict compared the draft with. */
  const baseline = state.result ? state.previous : active;
  const applied = queryPosterior(networkOf(baseline), evidenceOf(baseline));
  const suggestions = [
    ['Reveal that there was no earthquake', { ...draft, states: { ...draft.states, E: '0' } }],
    ['Reveal that there was an earthquake', { ...draft, states: { ...draft.states, E: '1' } }],
    ['Raise John’s false-call chance to .2', { ...draft, johnWhenQuiet: fixtures.falseCallJohn }],
    ['Fix the alarm at 1 and keep only John', { ...draft, states: { E: 'unknown', A: '1', J: '1', M: 'unknown' } }],
    ['Make both of John’s rows .4', {
      ...draft, johnWhenQuiet: fixtures.uninformativeCallRow, johnWhenAlarm: fixtures.uninformativeCallRow,
    }],
    ['Practice 1: make both of Mary’s rows .4', {
      ...draft, maryWhenQuiet: fixtures.uninformativeCallRow, maryWhenAlarm: fixtures.uninformativeCallRow,
    }],
    ['Make John impossible, then assert he called', {
      ...draft, johnWhenQuiet: 0, johnWhenAlarm: 0, states: { ...draft.states, J: '1' },
    }],
  ];
  return <Investigation
    title="Investigation 1 — which clue changes the explanation?"
    question={'Some evidence moves the burglary posterior a long way. Some moves it not at all, and one setting '
      + 'leaves it with no value. Decide which before you look.'}
    role={{
      kind: 'constructed',
      text: 'Every probability here is an invented teaching quantity. None of it is measured crime or engineering data.',
    }}
    onReset={state.reset}>

    <div className="bn-figure-row">
      <Dag edges={fixtures.alarmEdges} labels={alarmNetwork.labels} width={250} height={210}
        observed={Object.keys(evidenceOf(draft))} endpoints={['B']}
        describe={'The alarm network with the currently observed variables ringed: '
          + `${Object.keys(evidenceOf(draft)).join(', ') || 'none'}. Burglary is the query variable.`}
        caption="Observed variables are ringed; burglary is the query" />
      <div>
        <div className="bn-controls">
          {EVIDENCE_NODES.map(node => <StateChoice key={node} label={`${node} — ${alarmNetwork.labels[node]}`}
            value={draft.states[node]}
            onChange={value => state.edit({ states: { ...draft.states, [node]: value } })}
            hint={draft.states[node] === 'unknown' ? 'Unknown is not the state 0: it is summed over.' : undefined} />)}
        </div>
        <div className="bn-controls">
          <NumberField label="P(B = 1) — burglary prior" value={draft.burglaryPrior} {...controlSteps.rootPrior}
            min={controlSteps.rootPrior.minimum} max={controlSteps.rootPrior.maximum}
            onChange={value => state.edit({ burglaryPrior: value })}
            hint="Type it; a coarse slider could not reach .001." />
          <NumberField label="P(E = 1) — earthquake prior" value={draft.earthquakePrior} {...controlSteps.rootPrior}
            min={controlSteps.rootPrior.minimum} max={controlSteps.rootPrior.maximum}
            onChange={value => state.edit({ earthquakePrior: value })} />
          <NumberField label="P(J = 1 | A = 0) — John's false-call chance" value={draft.johnWhenQuiet}
            {...controlSteps.callerRow} min={controlSteps.callerRow.minimum} max={controlSteps.callerRow.maximum}
            onChange={value => state.edit({ johnWhenQuiet: value })} />
          <NumberField label="P(J = 1 | A = 1) — John when the alarm sounds" value={draft.johnWhenAlarm}
            {...controlSteps.callerRow} min={controlSteps.callerRow.minimum} max={controlSteps.callerRow.maximum}
            onChange={value => state.edit({ johnWhenAlarm: value })} />
          <NumberField label="P(M = 1 | A = 0) — Mary's false-call chance" value={draft.maryWhenQuiet}
            {...controlSteps.callerRow} min={controlSteps.callerRow.minimum} max={controlSteps.callerRow.maximum}
            onChange={value => state.edit({ maryWhenQuiet: value })} />
          <NumberField label="P(M = 1 | A = 1) — Mary when the alarm sounds" value={draft.maryWhenAlarm}
            {...controlSteps.callerRow} min={controlSteps.callerRow.minimum} max={controlSteps.callerRow.maximum}
            onChange={value => state.edit({ maryWhenAlarm: value })} />
        </div>
      </div>
    </div>

    <div className="bn-presets">
      {suggestions.map(([label, setup]) => <button key={label} type="button"
        onClick={() => state.suggest(setup)}>{label}</button>)}
      <button type="button" onClick={() => state.suggest(evidenceInitial)}>Back to both calls</button>
    </div>
    <p className="bn-caption">
      A suggested setup fills the controls. It never applies the calculation and never records a prediction for you.
      You are predicting against evidence {Object.entries(evidenceOf(baseline)).map(([node, value]) => `${node}=${value}`).join(', ') || 'none'},
      with P(J = 1 | A = 0) = {round(baseline.johnWhenQuiet, 4)}.
    </p>

    <Prediction
      state={state}
      prompt={'Against the state currently applied, will the burglary posterior rise, fall, remain within 10⁻¹², '
        + 'be undefined, or become defined after an impossible baseline?'}
      options={[
        ['higher', 'higher than the applied state'],
        ['lower', 'lower than the applied state'],
        ['unchanged', 'unchanged within 10⁻¹²'],
        ['undefined', 'no posterior at all, because the evidence would be impossible'],
        ['defined', 'a defined posterior now, but none in the applied state'],
      ]}
      numeric={{
        /* Relative, not absolute. A flat .005 told a learner who typed 0
           against the .001 prior that they were "within .005", in the one lab
           whose whole point is that zero and undefined are different answers. */
        label: 'Optional: the posterior you expect', name: 'the posterior',
        tolerance: 0.005, relative: true, digits: 6,
        undefinedNote: 'the evidence you selected has probability zero under these tables',
      }}
      requireChange={(next, current) => JSON.stringify(next) !== JSON.stringify(current)}
      pendingHint={'This question compares the new state with the applied one, so something has to differ. Change a '
        + 'state or a table entry, or load one of the setups above.'}
      committed={() => `evidence ${Object.entries(evidenceOf(state.active)).map(([node, value]) => `${node}=${value}`).join(', ') || 'none'}, `
        + `burglary prior ${round(state.active.burglaryPrior, 4)}, `
        + `John's rows ${round(state.active.johnWhenQuiet, 4)} and ${round(state.active.johnWhenAlarm, 4)}`}
      answerFor={(next, current) => {
        const after = queryPosterior(networkOf(next), evidenceOf(next));
        const before = queryPosterior(networkOf(current), evidenceOf(current));
        const outcome = changeDirection(after.posterior, before.posterior);
        return {
          outcome, value: after.posterior,
          explain: after.posterior === null
            ? `The evidence probability is exactly ${after.evidenceProbability}, so there is nothing to condition on: `
              + 'a posterior here is undefined, not zero and not one half.'
            : `The applied state gave ${before.posterior === null ? 'an undefined posterior' : round(before.posterior, 12)} and this one gives ${round(after.posterior, 12)}, `
              + `over ${after.compatibleWorlds} compatible worlds carrying total mass ${round(after.evidenceProbability, 10)}.`,
        };
      }} />

    {state.result
      ? <div className="bn-graded">
        <Table caption="The two burglary masses for the state now in the fields, before normalising"
          headings={['', 'mass', 'share of the evidence']}
          rows={[
            ['B = 0 (no burglary)', round(shown.mass[0], 12),
              shown.posterior === null ? '—' : round(1 - shown.posterior, 8)],
            ['B = 1 (burglary)', round(shown.mass[1], 12),
              shown.posterior === null ? '—' : round(shown.posterior, 8)],
            ['evidence probability', round(shown.evidenceProbability, 12),
              shown.posterior === null ? '—' : '1'],
          ]}
          footnote={`${shown.compatibleWorlds} of the 32 worlds are compatible with this evidence. `
            + 'Marginalisation adds them; conditioning renormalises what is left. Neither step picks a single '
            + 'most likely hidden explanation.'} />
        <p className="bn-caption">
          Posterior for the state in the fields: {ratio(shown.posterior,
            'the selected evidence has probability zero under these tables')}.
          {' '}The state you predicted against gave {ratio(applied.posterior,
            'that evidence has probability zero')}.
        </p>
      </div>
      : <p className="bn-note">
        The masses and the posterior for the values now in the fields are the answer to the question below, so
        they appear once you have recorded a prediction. The state you are predicting against
        gave {ratio(applied.posterior, 'that evidence has probability zero')} over
        {' '}{applied.compatibleWorlds} compatible worlds — that is the number to compare with.
      </p>}
  </Investigation>;
}

/* ================================================== I2 · the path inspector */

const pathInitial = {
  edges: fixtures.alarmEdges.map(edge => [...edge]),
  start: 'B', end: 'E', observed: [],
};

const NODE_POOL = ['B', 'E', 'A', 'J', 'M', 'K', 'R', 'T'];

/**
 * The paths, and -- only after a prediction is recorded -- their verdicts.
 *
 * Before commitment this lists the anatomy a learner needs in order to apply
 * the rule themselves: which paths exist, which interior nodes are colliders,
 * which are observed, and which have observed descendants. It does not say
 * whether any path is blocked, because that is the question being asked
 * directly underneath.
 */
function PathList({ report, revealed }) {
  if (!report.paths.length) {
    return <p className="bn-caption">No path at all joins {report.start} and {report.end} in this graph.</p>;
  }
  return <ul className={`bn-path-list${revealed ? ' bn-graded' : ''}`}>
    {report.paths.map(entry => <li key={entry.path.join('-')}
      className={revealed ? (entry.blocked ? 'is-blocked' : 'is-active') : undefined}>
      <span className="bn-path-mark" aria-hidden="true">{revealed ? (entry.blocked ? '×' : '→') : '·'}</span>
      <span className="bn-path-name">{entry.path.join('–')}</span>
      {revealed
        ? <><strong>{entry.blocked ? 'blocked' : 'active'}</strong> — {entry.reason}.</>
        : <>{entry.interior.length === 0
          ? 'a single edge, so it has no interior node that could block it.'
          : 'decide for yourself whether this one is blocked.'}</>}
      {entry.interior.length > 0 && <> Interior nodes: {entry.interior.map(node =>
        `${node.node} is a ${node.collider ? 'collider' : 'non-collider'} and is ${node.observed ? 'observed' : 'unobserved'}`
        + (node.collider && !node.observed
          ? node.observedDescendants.length
            ? ` with the observed descendant ${node.observedDescendants.join(', ')}`
            : ' with no observed descendant'
          : '')).join('; ')}.</>}
    </li>)}
  </ul>;
}

export function PathLab() {
  const state = useInvestigation(pathInitial);
  const { draft } = state;
  const [pendingEdge, setPendingEdge] = useState({ from: 'B', to: 'K' });
  const [edgeError, setEdgeError] = useState(null);

  const nodes = graphNodes(draft.edges);
  /* The verdict is the graded quantity here, so nothing that states it --
     the per-path labels, the trail's blocked styling, the drawing's own text
     alternative -- may render until a prediction has been committed. */
  const revealed = Boolean(state.result);
  let report = null;
  let setupError = null;
  try {
    report = dSeparation(draft.edges, draft.start, draft.end, draft.observed);
  } catch (error) {
    setupError = error.message;
  }
  const highlighted = report ? (report.active[0] ?? report.paths[0] ?? null) : null;

  const tryEdges = edges => {
    try {
      checkGraph(edges);
      const remaining = graphNodes(edges);
      if (!remaining.includes(draft.start) || !remaining.includes(draft.end)) {
        setEdgeError('That edit would remove one of the two query endpoints from the graph.');
        return;
      }
      setEdgeError(null);
      state.edit({ edges, observed: draft.observed.filter(node => remaining.includes(node)) });
    } catch (error) {
      setEdgeError(error.message);
    }
  };

  const presets = fixtures.pathPresets.map(preset => [
    preset.label,
    { edges: preset.edges.map(edge => [...edge]), start: preset.start, end: preset.end, observed: [] },
  ]);

  return <Investigation
    title="Investigation 2 — open every route, or block them all"
    question={'Does this observation set guarantee independence between the two endpoints, or does it leave '
      + 'dependence possible? Build the graph you want to test, then commit an answer.'}
    role={{
      text: 'This is a structural question about the drawing. It is not a claim about any dataset, and an active '
        + 'path never proves that two variables actually are correlated.',
    }}
    onReset={() => { state.reset(); setEdgeError(null); setPendingEdge({ from: 'B', to: 'K' }); }}>

    <div className="bn-presets">
      {presets.map(([label, setup]) => <button key={label} type="button"
        onClick={() => { setEdgeError(null); state.suggest(setup); }}>{label}</button>)}
    </div>

    <div className="bn-controls is-wide">
      <Select label="First endpoint" value={draft.start}
        options={nodes.filter(node => node !== draft.end).map(node => [node, node])}
        onChange={value => state.edit({ start: value, observed: draft.observed.filter(node => node !== value) })} />
      <Select label="Second endpoint" value={draft.end}
        options={nodes.filter(node => node !== draft.start).map(node => [node, node])}
        onChange={value => state.edit({ end: value, observed: draft.observed.filter(node => node !== value) })} />
    </div>

    <fieldset className="bn-controls is-wide">
      <legend>Observed variables. An endpoint of the query cannot also be observed.</legend>
      {nodes.map(node => {
        const isEndpoint = node === draft.start || node === draft.end;
        return <label key={node} className="bn-choice">
          <input type="checkbox" checked={draft.observed.includes(node)} disabled={isEndpoint}
            onChange={event => state.edit({
              observed: event.target.checked
                ? [...draft.observed, node]
                : draft.observed.filter(name => name !== node),
            })} />
          <span>{node}{isEndpoint ? ' (query endpoint)' : ''}</span>
        </label>;
      })}
    </fieldset>

    <div className="bn-controls is-wide">
      <Select label="Add or remove the arrow from" value={pendingEdge.from}
        options={NODE_POOL.map(node => [node, node])}
        onChange={value => setPendingEdge({ ...pendingEdge, from: value })} />
      <Select label="…to" value={pendingEdge.to}
        options={NODE_POOL.map(node => [node, node])}
        onChange={value => setPendingEdge({ ...pendingEdge, to: value })} />
    </div>
    <div className="bn-buttons">
      <button type="button" onClick={() => tryEdges([...draft.edges, [pendingEdge.from, pendingEdge.to]])}>
        Add this arrow
      </button>
      <button type="button" onClick={() => tryEdges(draft.edges
        .filter(([from, to]) => !(from === pendingEdge.from && to === pendingEdge.to)))}>
        Remove this arrow
      </button>
      <span>At most {limits.maximumNodes} nodes and {limits.maximumEdges} arrows, and no directed cycle.</span>
    </div>
    {edgeError && <p className="bn-field-error" role="alert">{edgeError} The graph on screen is unchanged.</p>}
    {setupError && <p className="bn-field-error" role="alert">{setupError}</p>}

    {report && <div className="bn-figure-row">
      <Dag edges={draft.edges} width={280} height={230}
        observed={draft.observed} endpoints={[draft.start, draft.end]}
        highlight={highlighted ? highlighted.path : null}
        highlightBlocked={revealed && highlighted ? highlighted.blocked : false}
        describe={`A graph on ${nodes.join(', ')}.`
          + (revealed ? ` ${report.summary}` : ' The verdict appears once a prediction is recorded.')}
        verdict={revealed && highlighted
          ? `The thick trail follows ${highlighted.path.join('–')}, which is ${highlighted.blocked ? 'blocked' : 'active'}.`
          : null}
        caption={`Query: is ${report.start} separated from ${report.end}?`} />
      <div>
        <PathList report={report} revealed={revealed} />
      </div>
    </div>}

    {report && <Prediction
      state={state}
      prompt={`Before revealing the verdict: with ${draft.observed.length ? `${[...draft.observed].sort().join(', ')} observed` : 'nothing observed'}, does this graph guarantee that ${draft.start} and ${draft.end} are independent?`}
      options={[
        ['guaranteed-independent', 'the graph guarantees independence — every path is blocked'],
        ['dependence-possible', 'the graph leaves dependence possible — at least one path stays active'],
      ]}
      sameQuestion={(left, right) => {
        const parse = key => { const value = JSON.parse(key); return `${value.start}|${value.end}`; };
        return parse(left) === parse(right);
      }}
      answerFor={next => {
        const verdict = dSeparation(next.edges, next.start, next.end, next.observed);
        return {
          outcome: verdict.verdict, value: verdict.active.length,
          explain: `${verdict.summary} ${verdict.paths.map(entry =>
            `${entry.path.join('–')} is ${entry.blocked ? 'blocked' : 'active'} because ${entry.reason}`).join('; ')}.`,
        };
      }} />}

    <p className="bn-caption">
      The path list above and the verdict you are graded against are the same calculation: the drawing highlights
      whatever that calculation returns. A separation result is a statement about this drawing only. Setting every
      alarm row to the same number, for instance, makes the alarm independent of its parents in that particular
      distribution while the graph still shows an active conditioned collider — an extra independence, and the
      reason faithfulness has to be assumed separately rather than read off the picture.
    </p>
  </Investigation>;
}

/* ============================================ I3 · buying one measurement */

const measurementInitial = {
  specimenId: validationSpecimens[0].id,
  visible: [false, false, false, false],
  values: null,
  candidates: [0, 2],
  question: 'purchase',
};

const specimenOf = id => validationSpecimens.find(row => row.id === id) ?? validationSpecimens[0];

const valuesOf = draft => draft.values ?? specimenOf(draft.specimenId).features;

const posteriorFor = (draft, visible) => {
  const model = trainingModels.treeAugmented;
  const values = valuesOf(draft);
  const bits = values.map((value, index) => (value > model.medians[index] ? 1 : 0));
  return classPosterior(model, bits, visible);
};

const visibleList = flags => flags.map((on, index) => (on ? index : -1)).filter(index => index >= 0);

export function MeasurementLab() {
  const state = useInvestigation(measurementInitial);
  const { draft } = state;
  const model = trainingModels.treeAugmented;
  const specimen = specimenOf(draft.specimenId);
  const values = valuesOf(draft);
  const edited = draft.values !== null
    && draft.values.some((value, index) => value !== specimen.features[index]);
  /* For the purchase question the current distribution is the starting point,
     not the answer, so it stays visible. For the edit question the answer IS
     whether this distribution moved, so before a commitment the baseline is
     shown instead of the draft. */
  const editing = draft.question === 'edit' && !state.result;
  const shownFor = editing ? state.active : draft;
  const current = posteriorFor(shownFor, visibleList(shownFor.visible));
  const labels = ['cultivar 0', 'cultivar 1', 'cultivar 2'];
  const hiddenCandidates = [0, 1, 2, 3].filter(index => !draft.visible[index]);
  const [revealed, setRevealed] = useState(false);

  return <Investigation
    title="Investigation 3 — purchase a measurement"
    question={'Nothing is visible yet. Which single measurement would tell you more about this specimen’s '
      + 'cultivar? Commit an answer, then reveal both and compare.'}
    role={{
      kind: 'exploratory',
      text: 'These are real Wine specimens read by the training-set tree-augmented model, which is deliberately '
        + 'not the recipe that won selection. This panel is exploratory: no test specimen is reachable from it, '
        + 'and nothing here is a validated measurement-purchasing policy.',
    }}
    onReset={() => { state.reset(); setRevealed(false); }}>

    <div className="bn-controls is-wide">
      <Select label="Validation specimen" value={String(draft.specimenId)}
        options={validationSpecimens.map(row => [String(row.id), `specimen ${row.id}`])}
        onChange={value => { setRevealed(false); state.edit({ specimenId: Number(value), values: null }); }} />
      <Select label="Question to commit on" value={draft.question}
        options={[
          ['purchase', 'which reveal leaves less uncertainty?'],
          ['edit', 'does this edited value move the answer?'],
        ]}
        onChange={value => state.edit({ question: value })} />
    </div>

    <p className="bn-caption">
      An unrevealed measurement shows no value anywhere, including in the box that edits it — reading four numbers
      off the controls would answer the question below before it is asked. You can still type a hypothetical value
      into a hidden measurement; the posterior will not move until that measurement is revealed.
    </p>
    <div className="bn-specimen-cards">
      {protocol.featureLabels.map((label, index) => <div key={label}
        className={`bn-specimen-card${draft.visible[index] ? '' : ' is-hidden-value'}${edited && values[index] !== specimen.features[index] ? ' is-edited' : ''}`}>
        {label}
        <b>{draft.visible[index] ? round(values[index], 4) : 'not revealed'}</b>
        <span className="bn-bin">
          training median {round(model.medians[index], 4)}
          {draft.visible[index]
            ? ` — this specimen is ${values[index] > model.medians[index] ? 'above' : 'at or below'} it`
            : ' — the value is summed over, not filled in'}
        </span>
        <label className="bn-choice">
          <input type="checkbox" checked={draft.visible[index]}
            onChange={event => {
              const visible = draft.visible.map((on, position) => (position === index ? event.target.checked : on));
              const stillHidden = [0, 1, 2, 3].filter(position => !visible[position]);
              // A candidate that has just become visible is no longer a thing
              // that can be bought, so the question would grade a reveal that
              // has already happened.
              const candidates = draft.candidates.every(candidate => stillHidden.includes(candidate))
                ? draft.candidates
                : [stillHidden[0] ?? 0, stillHidden[1] ?? stillHidden[0] ?? 0];
              state.edit({ visible, candidates });
            }} />
          <span>reveal this measurement</span>
        </label>
        <NumberField label={`edit ${label}`} value={values[index]} {...controlSteps.measurement}
          min={controlSteps.measurement.minimum} max={controlSteps.measurement.maximum}
          blind={!draft.visible[index]}
          placeholder={draft.visible[index] ? undefined : 'hypothetical'}
          onChange={value => state.edit({
            values: values.map((old, position) => (position === index ? value : old)),
          })} />
      </div>)}
    </div>
    {edited && <p className="bn-note">
      At least one value has been edited. This is now a hypothetical specimen, not the measured record for
      specimen {specimen.id}. Press “restore the measured values” to go back.
    </p>}
    <div className="bn-buttons">
      <button type="button" onClick={() => state.edit({ values: null })} disabled={!edited}>
        Restore the measured values
      </button>
      <button type="button" onClick={() => state.edit({ visible: [false, false, false, false] })}>
        Hide every measurement
      </button>
    </div>

    <Distribution caption={editing
      ? 'Cultivar probabilities for the state you are predicting against'
      : 'Cultivar probabilities for the measurements currently revealed'}
      className={draft.question === 'edit' && state.result ? 'bn-graded' : undefined}
      labels={labels} values={current.posterior} highlight={current.leading}
      describe={`Three bars: ${labels.map((label, index) => `${label} ${round(current.posterior[index], 4)}`).join(', ')}.`} />
    <p className="bn-caption">
      {visibleList(draft.visible).length === 0
        ? 'With nothing revealed, every conditional table sums away and the answer is exactly the class prior the '
          + 'model learned from its training specimens.'
        : `${current.compatibleStates} of the 16 possible measurement states are still compatible; the hidden ones `
          + 'are summed over rather than filled in with their more likely value.'}
      {' '}Remaining uncertainty {round(current.entropyNats, 6)} nats.
    </p>

    {draft.question === 'purchase' && hiddenCandidates.length >= 2 && <div className="bn-controls is-wide">
      <Select label="First candidate to buy" value={String(draft.candidates[0])}
        options={hiddenCandidates.map(index => [String(index), protocol.featureLabels[index]])}
        onChange={value => state.edit({
          candidates: [Number(value), draft.candidates[1] === Number(value)
            ? (hiddenCandidates.find(index => index !== Number(value)) ?? draft.candidates[1])
            : draft.candidates[1]],
        })} />
      <Select label="Second candidate to buy" value={String(draft.candidates[1])}
        options={hiddenCandidates.filter(index => index !== draft.candidates[0])
          .map(index => [String(index), protocol.featureLabels[index]])}
        hint="The first candidate is excluded here: two identical options are not a choice."
        onChange={value => state.edit({ candidates: [draft.candidates[0], Number(value)] })} />
    </div>}

    {draft.question === 'purchase' && hiddenCandidates.length < 2 && <p className="bn-note">
      Fewer than two measurements are still hidden, so there is no purchase to choose between. Hide a measurement,
      or switch to the edit question.
    </p>}

    {draft.question === 'purchase' && hiddenCandidates.length >= 2 && <Prediction
      state={state}
      prompt={`Reveal ${protocol.featureLabels[draft.candidates[0]]} or ${protocol.featureLabels[draft.candidates[1]]}, one or the other. Which leaves less uncertainty about the cultivar?`}
      options={[
        ['first', `${protocol.featureLabels[draft.candidates[0]]} leaves less uncertainty`],
        ['second', `${protocol.featureLabels[draft.candidates[1]]} leaves less uncertainty`],
        ['equal', 'their uncertainties agree within 10⁻¹² nats'],
      ]}
      sameQuestion={(left, right) => {
        const parse = key => { const value = JSON.parse(key); return `${value.question}|${value.candidates.join(',')}`; };
        return parse(left) === parse(right);
      }}
      answerFor={next => {
        const model = trainingModels.treeAugmented;
        const bits = valuesOf(next).map((value, index) => (value > model.medians[index] ? 1 : 0));
        const comparison = purchaseComparison(model, bits, visibleList(next.visible), next.candidates);
        return {
          outcome: comparison.outcome, value: comparison.gap,
          explain: next.candidates[0] === next.candidates[1]
            ? 'The same measurement was chosen twice, so the two options are identical by construction.'
            : `Revealing ${protocol.featureLabels[next.candidates[0]]} leaves ${round(comparison.first.entropyNats, 6)} nats `
              + `and leading class ${comparison.first.leading}; revealing ${protocol.featureLabels[next.candidates[1]]} leaves `
              + `${round(comparison.second.entropyNats, 6)} nats and leading class ${comparison.second.leading}. `
              + 'A single revealed measurement is not guaranteed to reduce entropy at all; it can raise it.',
        };
      }} />}

    {draft.question === 'edit' && <Prediction
      state={state}
      prompt={'Against the state currently applied, does the probability of cultivar 2 rise, fall, or stay within 10⁻¹²? The full distribution is shown alongside it.'}
      options={[
        ['higher', 'yes — the probability of cultivar 2 rises'],
        ['lower', 'yes — the probability of cultivar 2 falls'],
        ['unchanged', 'cultivar 2 is unchanged within 10⁻¹²'],
      ]}
      sameQuestion={(left, right) => JSON.parse(left).question === JSON.parse(right).question}
      requireChange={(next, currentState) => JSON.stringify(next) !== JSON.stringify(currentState)}
      pendingHint={'Edit a measurement value, or reveal or hide one, before applying: this question compares two states.'}
      answerFor={(next, currentState) => {
        const after = posteriorFor(next, visibleList(next.visible));
        const before = posteriorFor(currentState, visibleList(currentState.visible));
        const outcome = changeDirection(after.posterior[2], before.posterior[2]);
        const nextBits = valuesOf(next).map((value, index) => value > model.medians[index]);
        const beforeBits = valuesOf(currentState).map((value, index) => value > model.medians[index]);
        const evidenceUnchanged = next.visible.every((visible, index) =>
          visible === currentState.visible[index] && (!visible || nextBits[index] === beforeBits[index]));
        return {
          outcome, value: after.posterior[2],
          explain: `Cultivar 2 went from ${round(before.posterior[2], 12)} to ${round(after.posterior[2], 12)}. `
            + (evidenceUnchanged
              ? 'The revealed binary evidence is identical: hidden values and within-bin edits cannot change this posterior.'
              : 'The revealed evidence changed, through a visibility change or a different bin on a visible measurement. An unchanged cultivar-2 value would not by itself prove that the other probabilities stayed fixed.'),
        };
      }} />}

    <div className="bn-buttons">
      <button type="button" onClick={() => setRevealed(true)} disabled={revealed}>
        Reveal this specimen's recorded cultivar
      </button>
      {revealed && <span>Specimen {specimen.id} is cultivar {specimen.cultivar}.</span>}
      {!revealed && <span>The recorded label stays hidden until you ask for it.</span>}
    </div>
  </Investigation>;
}

/* ================================== I4 · observation weights and intervention */

const interventionInitial = {
  assignLow: 0.2, assignHigh: 0.6,
  riskNoLow: 0.01, riskNoHigh: 0.1, riskYesLow: 0.05, riskYesHigh: 0.2,
  target: 'association',
};

const serviceOf = draft => serviceModel({
  assignment: [draft.assignLow, draft.assignHigh],
  outcome: [[draft.riskNoLow, draft.riskNoHigh], [draft.riskYesLow, draft.riskYesHigh]],
});

export function InterventionLab() {
  const state = useInvestigation(interventionInitial);
  const { draft, active } = state;
  const result = serviceOf(draft);
  const revealed = Boolean(state.result);
  const baseline = revealed ? state.previous : active;
  const appliedResult = serviceOf(baseline);
  const lanes = laneGeometry(result);
  const presets = [
    ['Assign the procedure at random, .4 and .4', { ...draft, assignLow: 0.4, assignHigh: 0.4 }],
    ['Make the low-load response harmless, .05 → .01', { ...draft, riskYesLow: 0.01 }],
    ['Remove all overlap: 0 at low load, 1 at high', { ...draft, assignLow: 0, assignHigh: 1 }],
    ['Back to the lesson’s settings', interventionInitial],
  ];
  return <Investigation
    title="Investigation 4 — change who receives the procedure"
    question={'Changing the assignment mechanism changes who is in each observed group. Does it also change the '
      + 'effect of the procedure? Commit an answer for each of the two quantities separately.'}
    role={{
      kind: 'constructed',
      text: 'A fully specified constructed model. The procedure is deliberately harmful; a name like “service” or '
        + '“treatment” does not make an intervention beneficial, and none of this describes a real deployment.',
    }}
    onReset={state.reset}>

    <div className="bn-figure-row">
      <Dag edges={fixtures.serviceEdges} positions={fixtures.servicePositions}
        width={fixtures.serviceCanvas.width} height={fixtures.serviceCanvas.height}
        radius={fixtures.serviceCanvas.radius}
        labels={{ Z: 'load', X: 'procedure', Y: 'failure' }} endpoints={['X', 'Y']} legend={false}
        describe="Load points into both the procedure and the failure; the procedure points into the failure."
        caption="Observation: the assignment arrow into X is intact" />
      <Dag edges={fixtures.serviceEdges.filter(([, to]) => to !== 'X')} positions={fixtures.servicePositions}
        width={fixtures.serviceCanvas.width} height={fixtures.serviceCanvas.height}
        radius={fixtures.serviceCanvas.radius}
        labels={{ Z: 'load', X: 'procedure', Y: 'failure' }} endpoints={['X', 'Y']} legend={false}
        describe="The same graph in the same positions with the arrow from load into the procedure removed; load still points into failure."
        caption="Intervention: that same arrow is cut" />
    </div>

    <div className="bn-controls is-wide">
      <NumberField label="P(procedure | low load)" value={draft.assignLow} {...controlSteps.serviceProbability}
        min={controlSteps.serviceProbability.minimum} max={controlSteps.serviceProbability.maximum}
        onChange={value => state.edit({ assignLow: value })} />
      <NumberField label="P(procedure | high load)" value={draft.assignHigh} {...controlSteps.serviceProbability}
        min={controlSteps.serviceProbability.minimum} max={controlSteps.serviceProbability.maximum}
        onChange={value => state.edit({ assignHigh: value })} />
      <NumberField label="P(failure | no procedure, low load)" value={draft.riskNoLow} {...controlSteps.serviceProbability}
        min={controlSteps.serviceProbability.minimum} max={controlSteps.serviceProbability.maximum}
        onChange={value => state.edit({ riskNoLow: value })} />
      <NumberField label="P(failure | no procedure, high load)" value={draft.riskNoHigh} {...controlSteps.serviceProbability}
        min={controlSteps.serviceProbability.minimum} max={controlSteps.serviceProbability.maximum}
        onChange={value => state.edit({ riskNoHigh: value })} />
      <NumberField label="P(failure | procedure, low load)" value={draft.riskYesLow} {...controlSteps.serviceProbability}
        min={controlSteps.serviceProbability.minimum} max={controlSteps.serviceProbability.maximum}
        onChange={value => state.edit({ riskYesLow: value })} />
      <NumberField label="P(failure | procedure, high load)" value={draft.riskYesHigh} {...controlSteps.serviceProbability}
        min={controlSteps.serviceProbability.minimum} max={controlSteps.serviceProbability.maximum}
        onChange={value => state.edit({ riskYesHigh: value })} />
    </div>
    <div className="bn-presets">
      {presets.map(([label, setup]) => <button key={label} type="button"
        onClick={() => state.suggest(setup)}>{label}</button>)}
    </div>

    <div className="bn-figure-row">
      {lanes.map(lane => <MixtureLane key={`${lane.lane}-${lane.kind}`}
        caption={`${lane.kind === 'observation' ? 'Observed' : 'Intervened'} population with the procedure ${lane.lane ? 'given' : 'withheld'}`}
        segments={lane.segments}
        totalLabel={revealed
          ? (lane.risk === null
            ? 'No unit in this population is in that group, so the observed risk has an empty denominator.'
            : `Low load then high load, weighted failure risk ${round(lane.risk, 6)}.`)
          : 'Low load then high load. The weighted risk is part of the answer below, so it appears once a '
            + 'prediction is recorded.'}
        describe={`A horizontal bar split into a low-load part and a high-load part.${revealed ? ` Weighted risk: ${lane.risk === null ? 'undefined' : round(lane.risk, 6)}.` : ' Record a prediction to reveal the weighted risk.'}`} />)}
    </div>

    {revealed ? <div className="bn-graded">
    <Table caption="The two comparisons for the settings now in the fields"
      headings={['quantity', 'without the procedure', 'with the procedure', 'difference']}
      rows={[
        ['observed among those who received it',
          result.lanes[0].observedRisk === null
            ? ratio(null, result.lanes[0].undefinedBecause)
            : fixed(result.lanes[0].observedRisk, 6),
          result.lanes[1].observedRisk === null
            ? ratio(null, result.lanes[1].undefinedBecause)
            : fixed(result.lanes[1].observedRisk, 6),
          result.associationDifference === null
            ? ratio(null, 'one treatment group is empty')
            : fixed(result.associationDifference, 6)],
        ['assigned to the whole population',
          fixed(result.lanes[0].interventionRisk, 6), fixed(result.lanes[1].interventionRisk, 6),
          fixed(result.causalDifference, 6)],
      ]}
      footnote={result.positivityNote} />
    </div> : <p className="bn-note">
      The two comparisons for the values now in the fields are the answer to the question below, so they appear
      once you have recorded a prediction. {result.positivityNote}
    </p>}

    <div className="bn-controls">
      <Select label="Which quantity you are committing on" value={draft.target}
        options={[['association', 'the observed difference between groups'], ['causal', 'the causal difference under intervention']]}
        onChange={value => state.edit({ target: value })}
        hint="Switching this retires the prediction, and two verdicts about different quantities are never put side by side." />
    </div>

    <Prediction
      state={state}
      prompt={`Against the state currently applied, will the ${draft.target === 'association' ? 'observed difference between the two groups' : 'causal difference under intervention'} rise, fall, stay within 10⁻¹², be undefined, or become defined?`}
      options={[
        ['higher', 'it rises'],
        ['lower', 'it falls'],
        ['unchanged', 'it is unchanged within 10⁻¹²'],
        ['undefined', 'it has no value, because a group is empty'],
        ['defined', 'it is defined now; the applied comparison was undefined'],
      ]}
      requireChange={(next, currentState) => JSON.stringify({ ...next, target: null }) !== JSON.stringify({ ...currentState, target: null })}
      pendingHint="Change an assignment probability or a failure-table entry before applying."
      sameQuestion={(left, right) => JSON.parse(left).target === JSON.parse(right).target}
      answerFor={(next, currentState) => {
        const after = serviceOf(next);
        const before = serviceOf(currentState);
        const key = next.target === 'association' ? 'associationDifference' : 'causalDifference';
        const outcome = changeDirection(after[key], before[key]);
        return {
          outcome, value: after[key],
          explain: `The observed difference went from ${before.associationDifference === null ? 'undefined' : round(before.associationDifference, 12)} to `
            + `${after.associationDifference === null ? 'undefined' : round(after.associationDifference, 12)}, and the causal difference from `
            + `${round(before.causalDifference, 12)} to ${round(after.causalDifference, 12)}. `
            + 'Changing who receives the procedure changes the composition of the observed groups; only changing '
            + 'the failure table changes the response being averaged.',
        };
      }} />

    <p className="bn-caption">
      You are predicting against observed
      difference {ratio(appliedResult.associationDifference, 'a treatment group is empty')} and causal
      difference {round(appliedResult.causalDifference, 6)}.
      {' '}Set both assignment probabilities to 0 and the observed difference has no value at all, because nobody
      receives the procedure; the fully specified model still returns a causal difference, which is the whole
      distinction between a known generative model and identification from available observations.
    </p>
  </Investigation>;
}
