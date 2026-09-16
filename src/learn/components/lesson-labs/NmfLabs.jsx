import { useId, useState } from 'react';
import {
  cellTerms, contributionRows, fixtures, halfSquaredLoss, hCellDirection, limits, maskComparison,
  maskedReconstruction, reconstruct, updateH, updateW,
} from '../../data/nmf-models.js';
import { NMF_DIGITS } from '../../data/nmf-data.js';
import {
  ImagePanel, ImageValues, Investigation, NumberField, Plot, Prediction, Reason, ScaleKey, Strip, Table,
  fixed, round, signed, useInvestigation,
} from './NmfShared.jsx';
import './nmf-labs.css';

const featureLabels = ['feature 1', 'feature 2', 'feature 3'];

// ------------------------------------------------------------------ §1 · I1
const mixturePresets = {
  taught: { label: 'The taught example', a: 2, b: 1, H: [[1, 0, 1], [0, 1, 1]], feature: 2 },
  moreOfOne: { label: 'More of component 1 (a = 3)', a: 3, b: 1, H: [[1, 0, 1], [0, 1, 1]], feature: 2 },
  dropSecond: { label: 'Drop component 2 (b = 0)', a: 2, b: 0, H: [[1, 0, 1], [0, 1, 1]], feature: 2 },
  nullCase: { label: 'Null case: a = 0, then edit H₁₁', a: 0, b: 1, H: [[1, 0, 1], [0, 1, 1]], feature: 0 },
  sharedThird: { label: 'Same third feature, different first (a = 1, b = 2)', a: 1, b: 2, H: [[1, 0, 1], [0, 1, 1]], feature: 2 },
};
const mixtureValue = (inputs, feature) => inputs.a * inputs.H[0][feature] + inputs.b * inputs.H[1][feature];

export function MixtureLab() {
  const state = useInvestigation(mixturePresets.taught);
  const [history, setHistory] = useState([]);
  // The value the graded comparison was made against. Once a prediction is
  // committed the active inputs are the new ones, so the "before" number has to
  // be the state that was applied when the learner recorded the answer.
  const [priorApplied, setPriorApplied] = useState(mixturePresets.taught);
  const active = state.active;
  const draft = state.draft;
  const activeW = [[active.a, active.b]];
  const draftFeature = draft.feature;
  const previousValue = mixtureValue(active, draftFeature);
  const answerFor = inputs => {
    const change = mixtureValue(inputs, inputs.feature) - mixtureValue(active, inputs.feature);
    return Math.abs(change) <= 1e-12 ? 'unchanged' : change > 0 ? 'increase' : 'decrease';
  };
  const row = reconstruct(activeW, active.H)[0];
  const weighted = contributionRows(activeW, active.H, 0);
  const terms = cellTerms(activeW, active.H, 0, active.feature);
  const stripTop = Math.max(3, ...row, ...weighted.flat());
  const commit = inputs => {
    setPriorApplied(active);
    setHistory(previous => [...previous, {
      a: inputs.a, b: inputs.b, H: inputs.H, feature: inputs.feature,
      row: reconstruct([[inputs.a, inputs.b]], inputs.H)[0],
    }].slice(-5));
  };
  const load = key => { state.load(mixturePresets[key]); setPriorApplied(mixturePresets[key]); setHistory([]); };
  const editPattern = (component, feature, value) => state.edit({
    H: draft.H.map((patternRow, index) => (index === component
      ? patternRow.map((cell, position) => (position === feature ? value : cell))
      : patternRow)),
  });
  return <Investigation
    title="Build a new mixture, and say what one edit does"
    question="Two components, three features, and one observation built entirely by you. Choose a target feature, change an amount or a pattern cell, record whether that feature's reconstructed value will rise, fall or stay put, then apply the change."
    note="This is an exact reconstruction you compose. Nothing here is fitted, and no component is discovered: you are reasoning about which contributions add to which feature."
    onReset={() => { state.reset(); setPriorApplied(mixturePresets.taught); setHistory([]); }}>
    <div className="nm-controls">
      <NumberField label="Amount a of component 1" value={draft.a}
        min={limits.amount.minimum} max={limits.amount.maximum} step={limits.amount.step}
        onChange={a => state.edit({ a })} />
      <NumberField label="Amount b of component 2" value={draft.b}
        min={limits.amount.minimum} max={limits.amount.maximum} step={limits.amount.step}
        onChange={b => state.edit({ b })} />
      {[0, 1].flatMap(component => [0, 1, 2].map(feature => (
        <NumberField key={`${component}-${feature}`} label={`H${component + 1}${feature + 1}`} value={draft.H[component][feature]}
          min={limits.patternCell.minimum} max={limits.patternCell.maximum} step={limits.patternCell.step}
          onChange={value => editPattern(component, feature, value)} />
      )))}
    </div>
    <fieldset className="nm-prediction">
      <legend>Target feature</legend>
      <div className="nm-choices">
        {featureLabels.map((label, index) => (
          <label className="nm-choice" key={label}>
            <input type="radio" name="nm-mixture-feature" value={index} checked={draftFeature === index}
              onChange={() => state.edit({ feature: index })} />
            <span>{label}</span>
          </label>
        ))}
      </div>
      <p className="nm-caption">
        The value your next comparison starts from, at {featureLabels[draftFeature]}: <strong>{round(previousValue)}</strong>.
        Amounts move in steps of {limits.amount.step} between {limits.amount.minimum} and {limits.amount.maximum};
        pattern cells in steps of {limits.patternCell.step} between {limits.patternCell.minimum} and {limits.patternCell.maximum}.
        Type a value or use the field's own arrows; nothing outside those steps is applied.
      </p>
    </fieldset>
    <div className="nm-buttons">
      {Object.entries(mixturePresets).map(([key, preset]) => (
        <button key={key} type="button" onClick={() => load(key)}>{preset.label}</button>
      ))}
    </div>
    <Prediction
      prompt={`Apply your edits. Will ${featureLabels[draftFeature]} rise, fall or stay at its currently applied ${round(previousValue)}?`}
      options={[['increase', 'It increases'], ['decrease', 'It decreases'], ['unchanged', 'It stays the same']]}
      state={{ ...state, check: answer => { state.check(answer); commit(draft); }, explore: answer => { state.explore(answer); commit(draft); } }}
      answerFor={answerFor}
      describe={`Component 1 now contributes ${round(terms.terms[0].amount)} × ${round(terms.terms[0].patternCell)} = ${round(terms.terms[0].product)} and component 2 contributes ${round(terms.terms[1].amount)} × ${round(terms.terms[1].patternCell)} = ${round(terms.terms[1].product)}, so ${featureLabels[active.feature]} is now ${round(terms.total)} against ${round(mixtureValue(priorApplied, active.feature))} before the edit.`} />
    <Strip label={`Component 1 pattern × a = ${round(active.a)}`} values={weighted[0]} maximum={stripTop} highlight={[active.feature]} />
    <Strip label={`Component 2 pattern × b = ${round(active.b)}`} values={weighted[1]} maximum={stripTop} highlight={[active.feature]} />
    <Strip label="Reconstructed observation" values={row} maximum={stripTop} highlight={[active.feature]} />
    <ScaleKey maximum={stripTop} note="All three strips share this scale. It is the larger of 3 and the largest value your current mixture draws, so it grows when your edits do; the printed number in every cell is the value itself." />
    <Table caption="The applied mixture, feature by feature"
      headings={['feature', 'a × H₁ⱼ', 'b × H₂ⱼ', 'reconstructed value']}
      rows={featureLabels.map((label, index) => [
        label, round(active.a * active.H[0][index]), round(active.b * active.H[1][index]), round(row[index]),
      ])}
      rowClass={index => (index === active.feature ? 'is-selected' : undefined)} />
    {history.length > 1 && <Table caption="Your last applied mixtures, at most five kept"
      headings={['a', 'b', 'H row 1', 'H row 2', 'reconstruction']}
      rows={history.map(item => [
        round(item.a), round(item.b), `[${item.H[0].map(value => round(value)).join(', ')}]`,
        `[${item.H[1].map(value => round(value)).join(', ')}]`, `[${item.row.map(value => round(value)).join(', ')}]`,
      ])} />}
    <details>
      <summary>Two things worth finding for yourself</summary>
      <p className="nm-caption">
        First, find three different mixtures whose third feature is 3 but whose first features differ — (a, b) = (1, 2) and (2, 1) are two of them.
        Then edit a pattern cell so that a + b = 3 no longer characterises a third feature of 3, and say which contribution broke the rule.
        Second, set a = 0 and then change H₁₁: nothing in the reconstruction moves, because a zero amount multiplies that whole pattern away.
      </p>
    </details>
  </Investigation>;
}

// ------------------------------------------------------------------ §3 · I2
const updatePresets = {
  taught: { label: 'The taught start', X: fixtures.X, W: fixtures.startW, H: fixtures.startH, target: [0, 0], exact: false },
  alteredX: { label: 'Change X[1,2] from 1 to 2', X: [[2, 2, 3], [1, 2, 3], [3, 3, 6]], W: fixtures.startW, H: fixtures.startH, target: [0, 1], exact: false },
  biggerH: { label: 'Start H₁₁ at 3 instead of 1', X: fixtures.X, W: fixtures.startW, H: [[3, 0.2, 0.8], [0.2, 1, 0.8]], target: [0, 0], exact: false },
  exactFit: { label: 'Exact-fit null: X = W₁H₁', X: fixtures.X, W: fixtures.W1, H: fixtures.H1, target: [0, 0], exact: true },
};
const positiveRowsAndColumns = matrix => matrix.every(row => row.some(value => value > 0))
  && matrix[0].every((_, column) => matrix.some(row => row[column] > 0));

export function UpdateLab() {
  const [setup, setSetup] = useState(updatePresets.taught);
  const [draft, setDraft] = useState(updatePresets.taught);
  const [run, setRun] = useState(null);
  const [choice, setChoice] = useState('');
  const [reason, setReason] = useState('');
  const [recorded, setRecorded] = useState(null);
  const [problem, setProblem] = useState(null);
  const predictionName = useId();

  const startState = current => ({
    W: current.W, H: current.H, phase: 'ready', sweeps: 0,
    history: [{ sweep: 0, loss: halfSquaredLoss(current.X, current.W, current.H) }], past: [],
  });
  const current = run ?? startState(setup);
  const pending = JSON.stringify(draft) !== JSON.stringify(setup);
  const apply = next => {
    if (!positiveRowsAndColumns(next.X)) {
      setProblem('Every row and column of X must keep a positive total in this simplified model. That edit would leave one entirely zero.');
      return;
    }
    setSetup(next); setDraft(next); setRun(startState(next));
    setChoice(''); setReason(''); setRecorded(null); setProblem(null);
  };
  const editDraft = update => { setDraft(previous => ({ ...previous, ...update })); setChoice(''); setReason(''); setRecorded(null); setProblem(null); };

  const [component, column] = setup.target;
  const phase = (() => {
    if (current.phase !== 'ready') return null;
    try { return updateH(setup.X, current.W, current.H); } catch { return null; }
  })();
  const snapshot = () => ({ ...current, past: undefined });

  const doUpdateH = () => {
    try {
      const step = updateH(setup.X, current.W, current.H);
      const before = current.H[component][column];
      const after = step.next[component][column];
      setRun({ ...current, H: step.next, phase: 'afterH', past: [...current.past, snapshot()] });
      if (recorded && recorded.sweep === current.sweeps && recorded.outcome === undefined) {
        setRecorded({
          ...recorded, before, after,
          // Graded by the model's own ratio rule, which is what the verdict text
          // explains, rather than by a second rule written here.
          outcome: hCellDirection(setup.X, current.W, current.H, component, column),
          ratio: step.numerator[component][column] / step.denominator[component][column],
        });
      }
      setProblem(null);
    } catch (failure) { setProblem(failure.message); }
  };
  const doUpdateW = () => {
    try {
      const step = updateW(setup.X, current.W, current.H);
      const sweeps = current.sweeps + 1;
      setRun({
        W: step.next, H: current.H, phase: 'ready', sweeps,
        history: [...current.history, { sweep: sweeps, loss: halfSquaredLoss(setup.X, step.next, current.H) }],
        past: [...current.past, snapshot()],
      });
      setChoice(''); setReason(''); setProblem(null);
    } catch (failure) { setProblem(failure.message); }
  };
  const back = () => {
    if (current.past.length === 0) return;
    const previous = current.past[current.past.length - 1];
    setRun({ ...previous, past: current.past.slice(0, -1) });
    if (recorded && recorded.sweep >= previous.sweeps) setRecorded(null);
    setProblem(null);
  };

  const loss = halfSquaredLoss(setup.X, current.W, current.H);
  const atLimit = current.sweeps >= limits.sweeps.maximum;
  const graded = recorded?.outcome !== undefined;
  const positiveLosses = current.history.filter(point => point.loss > 0);
  const zeroLosses = current.history.filter(point => point.loss === 0);
  const logLow = positiveLosses.length ? Math.floor(Math.min(...positiveLosses.map(point => Math.log10(point.loss)))) : -1;
  const logHigh = positiveLosses.length ? Math.ceil(Math.max(...positiveLosses.map(point => Math.log10(point.loss)))) : 2;
  const numeratorTerms = current.W.map((row, observation) => row[component] * setup.X[observation][column]);
  const gram = current.W[0].map((_, r) => current.W[0].map((__, c) => current.W.reduce((sum, row) => sum + row[r] * row[c], 0)));
  const denominatorTerms = gram[component].map((value, other) => value * current.H[other][column]);

  return <Investigation
    title="Step one alternating update, one phase at a time"
    question={`X is fixed once you apply it. Choose the pattern cell you are watching, record whether the next H phase will grow it, shrink it or leave it alone, then run the two phases separately. The W start stays at the documented [[1, .5], [.5, 1], [1, 1]] unless a preset says otherwise.`}
    note="The displayed rule adds no epsilon to a denominator. Every supported state here keeps those denominators positive, and a zero pattern entry is left exactly where it is rather than evaluated as 0 / 0."
    onReset={() => apply(updatePresets.taught)}>
    <div className="nm-controls">
      {draft.X.flatMap((row, r) => row.map((value, c) => (
        <NumberField key={`x-${r}-${c}`} label={`X[${r + 1},${c + 1}]`} value={value}
          min={limits.measurement.minimum} max={limits.measurement.maximum} step={limits.measurement.step}
          onChange={next => editDraft({ X: draft.X.map((old, rowIndex) => (rowIndex === r ? old.map((cell, columnIndex) => (columnIndex === c ? next : cell)) : old)) })} />
      )))}
    </div>
    <div className="nm-controls">
      {draft.H.flatMap((row, r) => row.map((value, c) => (
        <NumberField key={`h-${r}-${c}`} label={`initial H[${r + 1},${c + 1}]`} value={value}
          min={limits.initialFactor.minimum} max={limits.initialFactor.maximum} step={limits.initialFactor.step}
          onChange={next => editDraft({ H: draft.H.map((old, rowIndex) => (rowIndex === r ? old.map((cell, columnIndex) => (columnIndex === c ? next : cell)) : old)) })} />
      )))}
    </div>
    <fieldset className="nm-prediction">
      <legend>Which pattern cell are you watching?</legend>
      <div className="nm-choices">
        {[0, 1].flatMap(r => [0, 1, 2].map(c => (
          <label className="nm-choice" key={`t-${r}-${c}`}>
            <input type="radio" name="nm-update-target" checked={draft.target[0] === r && draft.target[1] === c}
              onChange={() => editDraft({ target: [r, c] })} />
            <span>H[{r + 1},{c + 1}]</span>
          </label>
        )))}
      </div>
    </fieldset>
    <div className="nm-buttons">
      <button type="button" className="is-primary" disabled={!pending} onClick={() => apply(draft)}>Apply setup and restart</button>
      {Object.entries(updatePresets).map(([key, preset]) => (
        <button key={key} type="button" onClick={() => apply(preset)}>{preset.label}</button>
      ))}
    </div>
    {pending && <p className="nm-pending" role="status">Edited values are not in the model yet. Applying them restarts the trace from the new initial state and clears any recorded prediction.</p>}
    {problem && <p className="nm-note nm-problem" role="status">{problem}</p>}
    {setup.exact && <p className="nm-caption">
      This preset carries the exact factors from section 1, whose pattern matrix contains two zeros. The editable range for an initial pattern
      entry starts at {limits.initialFactor.minimum}, so those two fields show a value the fields themselves would not accept: editing any of them
      leaves the exact-fit state, which is the point of the contrast.
    </p>}

    <div className="nm-prediction">
      <fieldset>
        <legend>Record a prediction first.</legend>
        <p>
          Starting from sweep {current.sweeps}, H[{component + 1},{column + 1}] is currently {fixed(current.H[component][column], 9)}.
          What will the next H phase do to it?
        </p>
        <div className="nm-choices">
          {[['grows', 'It grows'], ['shrinks', 'It shrinks'], ['unchanged', 'It stays the same']].map(([value, text]) => (
            <label className="nm-choice" key={value}>
              <input type="radio" name={predictionName} value={value} checked={choice === value}
                onChange={() => setChoice(value)}
                disabled={pending || atLimit || current.phase !== 'ready' || recorded?.sweep === current.sweeps} />
              <span>{text}</span>
            </label>
          ))}
        </div>
      </fieldset>
      <Reason value={reason} onChange={setReason} disabled={recorded?.sweep === current.sweeps} />
      <div className="nm-buttons">
        <button type="button" disabled={choice === '' || pending || atLimit || current.phase !== 'ready' || recorded?.sweep === current.sweeps}
          onClick={() => setRecorded({ sweep: current.sweeps, choice, reason, target: [component, column] })}>Record it</button>
        <button type="button" className="is-primary" disabled={pending || atLimit || current.phase !== 'ready'} onClick={doUpdateH}>Update H</button>
        <button type="button" className="is-primary" disabled={pending || current.phase !== 'afterH'} onClick={doUpdateW}>Update W</button>
        <button type="button" disabled={current.past.length === 0} onClick={back}>Back one phase</button>
      </div>
      {graded && recorded.reason && <p className="nm-caption">Your reason, kept as you wrote it: “{recorded.reason}”</p>}
      {graded && <p className={`nm-verdict ${recorded.choice === recorded.outcome ? '' : 'is-miss'}`} role="status">
        <span className="nm-verdict-mark" aria-hidden="true">{recorded.choice === recorded.outcome ? '=' : '≠'}</span>
        At sweep {recorded.sweep}{recorded.sweep === current.sweeps ? '' : ', which you have since stepped past'}, the H phase moved
        {' '}H[{recorded.target[0] + 1},{recorded.target[1] + 1}] from {fixed(recorded.before, 9)} to {fixed(recorded.after, 9)}, so it {recorded.outcome === 'unchanged' ? 'stayed the same' : recorded.outcome}.
        {recorded.before === 0
          ? ' A zero entry multiplied by any ratio is still zero: this is the locked case, not a statement that the move would not help.'
          : ` The ratio at that cell was ${fixed(recorded.ratio, 9)}; a ratio above 1 grows the entry and a ratio below 1 shrinks it.`}
      </p>}
      {atLimit && <p className="nm-note" role="status">This investigation keeps at most {limits.sweeps.maximum} sweeps. Step back or apply a new setup.</p>}
    </div>

    <p className="nm-readout" aria-live="polite">
      Sweep {current.sweeps}, {current.phase === 'afterH' ? 'H updated and waiting for the W phase' : 'both phases applied'}.
      Half squared Frobenius loss {fixed(loss, 9)}.
    </p>

    <Table caption={`The selected cell's numerator, term by term: column ${component + 1} of W against column ${column + 1} of X`}
      headings={['observation', `W[i,${component + 1}]`, `X[i,${column + 1}]`, 'product']}
      rows={setup.X.map((_, observation) => [
        `i = ${observation + 1}`, fixed(current.W[observation][component], 6), round(setup.X[observation][column]),
        fixed(numeratorTerms[observation], 6),
      ]).concat([['numerator (WᵀX)', '', '', fixed(numeratorTerms.reduce((sum, value) => sum + value, 0), 6)]])} />
    <Table caption={`And its denominator: row ${component + 1} of WᵀW against column ${column + 1} of H`}
      headings={['component', `(WᵀW)[${component + 1},r]`, `H[r,${column + 1}]`, 'product']}
      rows={gram[component].map((value, other) => [
        `r = ${other + 1}`, fixed(value, 6), fixed(current.H[other][column], 6), fixed(denominatorTerms[other], 6),
      ]).concat([['denominator ((WᵀW)H)', '', '', fixed(denominatorTerms.reduce((sum, value) => sum + value, 0), 6)]])} />
    {phase
      ? <p className="nm-caption">
        The ratio at H[{component + 1},{column + 1}] is {fixed(phase.numerator[component][column], 6)} / {fixed(phase.denominator[component][column], 6)} = {fixed(phase.ratio[component][column], 9)}, and the next action is that H phase.
      </p>
      : <p className="nm-caption">
        These two operand tables are recomputed from the current factors. The H phase has already run at this sweep, so the next action is the W phase. The ratio above describes a hypothetical extra H step with W still fixed; after the scheduled W update, these operands will be recomputed for the next H phase.
      </p>}

    <Table caption="The current activations W, one row per observation, to nine decimals"
      headings={['observation', 'component 1', 'component 2']}
      rows={current.W.map((row, index) => [`i = ${index + 1}`, ...row.map(value => fixed(value, 9))])} />
    <Table caption="The current patterns H, one row per component, to nine decimals"
      headings={['component', 'feature 1', 'feature 2', 'feature 3']}
      rows={current.H.map((row, index) => [`r = ${index + 1}`, ...row.map(value => fixed(value, 9))])}
      rowClass={index => (index === component ? 'is-selected' : undefined)} />
    <Table caption="The reconstruction against X"
      headings={['row', 'reconstruction', 'observed X']}
      rows={reconstruct(current.W, current.H).map((row, index) => [
        `row ${index + 1}`, `[${row.map(value => fixed(value, 4)).join(', ')}]`, `[${setup.X[index].map(value => round(value)).join(', ')}]`,
      ])} />

    {current.history.length > 1 && positiveLosses.length > 1 && <Plot width={340} height={210}
      caption="Total half squared loss by sweep, on a base-10 logarithmic scale"
      domain={[0, Math.max(1, current.history[current.history.length - 1].sweep)]}
      range={[logLow, logHigh]}
      xTicks={current.history.map(point => point.sweep).filter((sweep, index, all) => index === 0 || index === all.length - 1 || sweep % Math.max(1, Math.ceil(current.sweeps / 5)) === 0)}
      // At most six decade labels, so they never pile up on a tall trace.
      yTicks={Array.from({ length: logHigh - logLow + 1 }, (_, index) => logLow + index)
        .filter((value, index, all) => (all.length <= 6 ? true : index % Math.ceil(all.length / 6) === 0 || index === all.length - 1))}
      formatX={value => String(value)} formatY={value => `1e${value}`}
      describe={`Half squared loss by sweep: ${current.history.map(point => `${point.sweep} gives ${round(point.loss, 9)}`).join('; ')}.${zeroLosses.length ? ` ${zeroLosses.length} of these values are exactly zero and have no logarithm; they are listed separately below the axis.` : ''}`}>
      {(scaleX, scaleY) => <>
        <polyline className="nm-curve" stroke="#e7b94a"
          points={positiveLosses.map(point => `${scaleX(point.sweep)},${scaleY(Math.log10(point.loss))}`).join(' ')} />
        {positiveLosses.map(point => <circle key={point.sweep} className="nm-mark" cx={scaleX(point.sweep)} cy={scaleY(Math.log10(point.loss))} r="4" />)}
      </>}
    </Plot>}
    {zeroLosses.length > 0 && <p className="nm-note">
      Exactly zero loss at {zeroLosses.length === current.history.length ? 'every sweep' : `sweeps ${zeroLosses.map(point => point.sweep).join(', ')}`}.
      A logarithm has no value there, so those sweeps are named here instead of being drawn at a fictional position on the axis.
    </p>}
    <Table caption="Every sweep so far" headings={['sweep', 'half squared loss']}
      rows={current.history.map(point => [point.sweep, fixed(point.loss, 9)])} scroll />
    <details>
      <summary>A transfer worth recording separately</summary>
      <p className="nm-caption">
        Change an X entry you have not touched yet, pick a different pattern cell, and predict its direction before the H phase. Then explain how the
        loss can fall over a full sweep while one W entry shrinks: the objective sees products, not individual factor entries. These are short
        pedagogical updates on a bounded fixture, not production input support and not a promise that the limit is stationary.
      </p>
    </details>
  </Investigation>;
}

// ------------------------------------------------------------------ §5 · I3
export function ContributionLab() {
  const { dictionary, activations, test, scale, side } = NMF_DIGITS;
  const fullMask = dictionary.map(() => true);
  const initial = { imageIndex: 0, mask: fullMask, pixel: null };
  const [draft, setDraft] = useState(initial);
  const [applied, setApplied] = useState(initial);
  const [totalChoice, setTotalChoice] = useState('');
  const [pixelChoice, setPixelChoice] = useState('');
  const [reason, setReason] = useState('');
  const [result, setResult] = useState(null);
  const totalName = useId();
  const pixelName = useId();

  const observedFor = index => test[index].pixels.map(value => value / scale);
  const pending = JSON.stringify(draft) !== JSON.stringify(applied);
  const editMask = component => {
    setDraft(previous => ({ ...previous, mask: previous.mask.map((value, index) => (index === component ? !value : value)) }));
    setResult(null); setTotalChoice(''); setPixelChoice('');
  };
  const selectPixel = index => {
    setDraft(previous => ({ ...previous, pixel: previous.pixel === index ? null : index }));
    setResult(null); setTotalChoice(''); setPixelChoice('');
  };
  const changeImage = value => {
    const next = { imageIndex: value, mask: fullMask, pixel: null };
    setDraft(next); setApplied(next); setResult(null); setTotalChoice(''); setPixelChoice(''); setReason('');
  };
  const commit = graded => {
    const observed = observedFor(draft.imageIndex);
    const comparison = maskComparison(observed, activations[draft.imageIndex], dictionary, applied.mask, draft.mask);
    const changedComponents = draft.mask.flatMap((included, index) => included === applied.mask[index] ? [] : [index]);
    const nullExplanation = changedComponents.length === 0
      ? 'The mask did not change, so the reconstruction is identical.'
      : changedComponents.every(index => activations[draft.imageIndex][index] === 0)
        ? 'Every switched component has zero activation, so these switches contribute nothing and are exact no-ops.'
        : 'The total error changed by no more than the stated comparison tolerance; this does not imply every contribution is zero.';
    setApplied(draft);
    setResult({
      graded, reason,
      totalChoice, pixelChoice,
      totalAnswer: comparison.direction,
      pixelAnswer: draft.pixel === null ? null : comparison.pixelDirection(draft.pixel),
      pixel: draft.pixel,
      comparison, nullExplanation,
    });
  };

  const observed = observedFor(applied.imageIndex);
  const coefficients = activations[applied.imageIndex];
  const patternSums = dictionary.map(row => row.reduce((sum, value) => sum + value, 0));
  const current = maskedReconstruction(coefficients, dictionary, applied.mask);
  const currentResidual = observed.map((value, index) => value - current[index]);
  const currentMse = currentResidual.reduce((sum, value) => sum + value * value, 0) / currentResidual.length;
  const intensityMax = Math.max(1, ...current, ...observed);
  const residualExtent = Math.max(...currentResidual.map(Math.abs), 1e-9);
  const changeExtent = result ? Math.max(...result.comparison.squaredChange.map(Math.abs), 1e-12) : 1;
  const totalCorrect = result?.graded && result.totalChoice === result.totalAnswer;
  const pixelCorrect = result?.graded && result.pixelChoice !== '' && result.pixelChoice === result.pixelAnswer;

  return <Investigation
    title="Take a component away from an actual held-out image"
    question="Choose one of the 60 reserved images, switch fitted contributions off, and record which way the image's mean squared error will move before you apply the change. The fitted dictionary and this row's activations are held exactly as the fit produced them; nothing is refitted in your browser."
    note="At an exact rowwise optimum, removing a component with the others fixed cannot reduce the total squared error, because setting its coefficient to zero was already available. Individual pixels are not bound by that argument: an overpredicted pixel can improve while the total gets worse."
    onReset={() => changeImage(0)}>
    <div className="nm-controls">
      <label className="nm-field">
        <span>Reserved image</span>
        <select value={draft.imageIndex} onChange={event => changeImage(Number(event.target.value))}>
          {test.map((row, index) => <option key={row.sourceRow} value={index}>
            {index + 1} of 60 · source row {row.sourceRow} · recorded digit {row.digit}
          </option>)}
        </select>
      </label>
    </div>
    <p className="nm-caption">
      The recorded digit is a diagnostic label from the collection. It never entered the factorization and it is not what the components encode.
      Changing the image restores every contribution and clears any recorded prediction.
    </p>
    <ul className="nm-toggles">
      {dictionary.map((_, component) => (
        <li key={component}>
          <button type="button" aria-pressed={draft.mask[component]} onClick={() => editMask(component)}>
            Component {component + 1} · {draft.mask[component] ? 'included' : 'removed'} · W = {fixed(coefficients[component], 4)}
          </button>
        </li>
      ))}
    </ul>
    <div className="nm-prediction">
      <fieldset>
        <legend>Record a prediction first.</legend>
        <p>
          {result
            ? <>This mask is applied to source row {test[applied.imageIndex].sourceRow}, with {applied.mask.filter(Boolean).length} of 8
              contributions and a mean squared error of {fixed(currentMse, 9)}. Switch a contribution on or off to set up the next comparison.</>
            : <>Applying this mask to source row {test[applied.imageIndex].sourceRow} changes the reconstruction
              from {applied.mask.filter(Boolean).length} contribution{applied.mask.filter(Boolean).length === 1 ? '' : 's'} to
              {' '}{draft.mask.filter(Boolean).length}. The image mean squared error is now {fixed(currentMse, 9)}. Which way will it move?</>}
        </p>
        <div className="nm-choices">
          {[['rises', 'It rises'], ['falls', 'It falls'], ['unchanged', 'It stays the same to within 1e−8']].map(([value, text]) => (
            <label className="nm-choice" key={value}>
              <input type="radio" name={totalName} value={value} checked={totalChoice === value}
                onChange={() => setTotalChoice(value)} disabled={Boolean(result)} />
              <span>{text}</span>
            </label>
          ))}
        </div>
      </fieldset>
      {draft.pixel !== null && <fieldset>
        <legend>And the selected pixel, optionally</legend>
        <p>
          Pixel at row {Math.floor(draft.pixel / side) + 1}, column {draft.pixel % side + 1}: observed {round(observed[draft.pixel], 4)},
          currently reconstructed {round(current[draft.pixel], 4)}. Will <em>its</em> squared error rise or fall?
        </p>
        <div className="nm-choices">
          {[['rises', 'This pixel gets worse'], ['falls', 'This pixel gets better'], ['unchanged', 'This pixel does not move']].map(([value, text]) => (
            <label className="nm-choice" key={value}>
              <input type="radio" name={pixelName} value={value} checked={pixelChoice === value}
                onChange={() => setPixelChoice(value)} disabled={Boolean(result)} />
              <span>{text}</span>
            </label>
          ))}
        </div>
      </fieldset>}
      <Reason value={reason} onChange={setReason} disabled={Boolean(result)} />
      {pending && !result && <p className="nm-pending" role="status">The mask below is a draft. Applying it will compare against the {applied.mask.filter(Boolean).length}-contribution reconstruction currently drawn.</p>}
      <div className="nm-buttons">
        <button type="button" className="is-primary" disabled={totalChoice === '' || Boolean(result)} onClick={() => commit(true)}>Check prediction</button>
        <button type="button" disabled={Boolean(result)} onClick={() => commit(false)}>Apply without recording a prediction</button>
      </div>
      {result?.reason && <p className="nm-caption">Your reason, kept as you wrote it: “{result.reason}”</p>}
      {result && <p className={`nm-verdict ${result.graded && !totalCorrect ? 'is-miss' : result.graded ? '' : 'is-plain'}`} role="status">
        <span className="nm-verdict-mark" aria-hidden="true">{result.graded ? (totalCorrect ? '=' : '≠') : '·'}</span>
        {result.graded ? '' : 'Calculated without a recorded prediction. '}
        The image mean squared error went from {fixed(result.comparison.beforeMse, 9)} to {fixed(result.comparison.afterMse, 9)}, a change of
        {' '}{signed(result.comparison.difference, 9)}, so it {result.totalAnswer === 'unchanged' ? 'stayed the same to within 1e−8' : result.totalAnswer}.
        {' '}{result.comparison.improvedPixels} pixel{result.comparison.improvedPixels === 1 ? '' : 's'} improved,
        {' '}{result.comparison.worsenedPixels} got worse and {result.comparison.unchangedPixels} did not move.
        {result.pixel !== null && result.pixelChoice !== ''
          ? ` Your selected pixel's squared error ${result.pixelAnswer === 'unchanged' ? 'did not move' : result.pixelAnswer} by ${signed(result.comparison.squaredChange[result.pixel], 9)}.${result.graded ? ` That second prediction ${pixelCorrect ? 'matched' : 'missed'}.` : ' This exploration was not graded.'}`
          : ''}
        {result.totalAnswer === 'unchanged' ? ` ${result.nullExplanation}` : ''}
      </p>}
    </div>

    <div className="nm-image-grid">
      <ImagePanel title={`Observed, source row ${test[applied.imageIndex].sourceRow}`} values={observed} side={side} maximum={intensityMax}
        note={`Click or tab to a cell to select it for the optional pixel prediction. Maximum ${round(Math.max(...observed), 4)}.`}
        selected={draft.pixel} onSelect={selectPixel} />
      <ImagePanel title={`Reconstruction from ${applied.mask.filter(Boolean).length} of 8 contributions`} values={current} side={side} maximum={intensityMax}
        note={`Maximum ${round(Math.max(...current), 4)} on a scale to ${round(intensityMax, 4)}.`} />
      <ImagePanel title="Signed residual" values={currentResidual} side={side} tone="signed"
        note={`Mean squared error ${fixed(currentMse, 9)}.`} />
      {result && <ImagePanel title="Change in squared error, this apply" values={result.comparison.squaredChange} side={side} tone="signed"
        note={changeExtent <= 1e-12
          ? 'Every pixel is unchanged: this mask change moved nothing at all, so the whole panel is the neutral colour.'
          : `Gold got worse, blue got better, on a scale to ±${round(changeExtent, 6)}.`} />}
    </div>
    <ScaleKey maximum={intensityMax} note="Observed and reconstruction panels." />
    <ScaleKey tone="signed" maximum={residualExtent} note="Residual panel. The change panel above has its own symmetric scale, stated in its note." />

    <Table caption="Every fitted contribution on this image, ordered by contribution total"
      headings={['component', 'in the reconstruction?', 'activation', 'pattern mass', 'contribution total']}
      rows={[...dictionary.keys()]
        .sort((a, b) => coefficients[b] * patternSums[b] - coefficients[a] * patternSums[a])
        .map(component => [
          `Component ${component + 1}`, applied.mask[component] ? 'included' : 'removed',
          fixed(coefficients[component], 6), fixed(patternSums[component], 6),
          fixed(coefficients[component] * patternSums[component], 6),
        ])} />
    <ImageValues summary={`All 64 values for the panels above, source row ${test[applied.imageIndex].sourceRow}`}
      side={side} panels={[
        { title: 'observed', values: observed },
        { title: 'reconstruction', values: current },
        { title: 'signed residual', values: currentResidual },
      ]} />
    <p className="nm-caption">
      A low summed error is not evidence that a removed component is a physical source. Ordering by the contribution total, the coefficient times its
      pattern's mass, is not the same as ordering by the raw coefficient, because the eight patterns do not carry equal mass.
    </p>
    <details>
      <summary>A transfer worth recording separately</summary>
      <p className="nm-caption">
        Choose another image, remove the component with the greatest <strong>contribution total</strong> rather than the greatest raw coefficient,
        explain three pixels that moved, and record both mean squared errors. Image 2 of 60, source row 256, is a clean case: its largest raw
        coefficient belongs to component 7 while its largest contribution total belongs to component 8, where on the image this lab opens with the
        two agree. Sixteen of the sixty reserved images disagree that way. Then
        remove every contribution: the reconstruction becomes exactly zero and the error becomes the mean squared observation, which is the largest
        value this control can produce.
      </p>
    </details>
  </Investigation>;
}
