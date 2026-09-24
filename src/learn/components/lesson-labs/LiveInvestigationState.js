import { useLayoutEffect, useState } from 'react';

/** Valid edits update the existing topic-specific views before browser paint.
 * Evaluation is keyed by inputs and an explicit comparison snapshot, not by
 * render count or a newly allocated callback. Invalid drafts retain a clearly
 * labelled last valid calculation. No learner answer is stored or graded.
 */
export function useLiveInvestigation(initial, describeKey = JSON.stringify, initialDraft = initial) {
  const [draft, setDraft] = useState(initialDraft);
  const [epoch, setEpoch] = useState(0);
  const [active, setActive] = useState(initial);
  const [previous, setPrevious] = useState(initial);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [attempts, setAttempts] = useState([]);
  return {
    draft, active, previous, result, history, constructionAttempts: attempts,
    inputKey: `${epoch}:${describeKey(draft)}`, baselineKey: describeKey(previous),
    pending: describeKey(draft) !== describeKey(active), ready: result !== null,
    edit: update => setDraft(current => ({ ...current, ...(typeof update === 'function' ? update(current) : update) })),
    suggest: inputs => { setPrevious(active); setDraft(inputs); },
    load: inputs => { setPrevious(active); setDraft(inputs); },
    evaluate: calculateInputs => {
      const calculation = calculateInputs(draft, previous);
      setActive(draft);
      setResult({ key: describeKey(draft), previousKey: describeKey(previous), inputs: draft, calculation });
    },
    snapshot: () => {
      if (result) setHistory(entries => [...entries.slice(-3), result]);
      setPrevious(active);
    },
    evaluateConstruction: gradeFor => setAttempts(entries => [...entries.slice(-4), { key: describeKey(draft), verdict: gradeFor(draft) }]),
    reset: () => { setEpoch(value => value + 1); setDraft(initialDraft); setActive(initial); setPrevious(initial); setResult(null); setHistory([]); setAttempts([]); },
  };
}

export function useLiveResult(state, calculateInputs, blocked) {
  const [error, setError] = useState(null);
  useLayoutEffect(() => {
    if (blocked) { setError(null); return; }
    try { state.evaluate(calculateInputs); setError(null); }
    catch (problem) { setError(problem.message); }
    // The key includes all model inputs. Depending on callback identity would
    // retrigger after the result update, because lessons define local closures.
  }, [state.inputKey, state.baselineKey, blocked]);
  return blocked || error;
}

/** Several linked calculations share inputs, while retaining their own valid
 * calculation and comparison snapshot. Every stage is available from the start.
 */
export function useLiveStages(initial, stageNames, describeKey = JSON.stringify) {
  const [draft, setDraft] = useState(initial);
  const [epoch, setEpoch] = useState(0);
  const [stages, setStages] = useState(() => Object.fromEntries(stageNames.map(name => [name, { active: initial, previous: initial, result: null, history: [] }])));
  const update = (name, change) => setStages(current => ({ ...current, [name]: { ...current[name], ...change } }));
  return {
    draft,
    edit: change => setDraft(current => ({ ...current, ...change })),
    suggest: setDraft,
    reset: () => { setEpoch(value => value + 1); setDraft(initial); setStages(Object.fromEntries(stageNames.map(name => [name, { active: initial, previous: initial, result: null, history: [] }]))); },
    ready: name => stages[name].result !== null,
    stage: name => {
      const state = stages[name];
      return { ...state, draft, inputKey: `${epoch}:${describeKey(draft)}`, baselineKey: describeKey(state.previous),
        pending: describeKey(draft) !== describeKey(state.active),
        evaluate: calculateInputs => update(name, { active: draft, result: { key: describeKey(draft), inputs: draft, calculation: calculateInputs(draft, state.previous) } }),
        snapshot: () => update(name, { previous: state.active, history: state.result ? [...state.history.slice(-3), state.result] : state.history }),
      };
    },
  };
}
