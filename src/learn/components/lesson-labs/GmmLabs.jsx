import { useState } from 'react';
import {
  correlationGeometry, emCycle, expectation, limits, maximization, mixtureAt, planeComparison, planeDensity, responsibilityCrossings,
} from '../../data/gmm-models';
import { Field, Investigation, NumberField, Plot, Prediction, Reason, Table, curve, fixed, percent, round, signed, useInvestigation } from './GmmShared.jsx';
import './gmm-labs.css';

const componentNames = ['A', 'B'];
const rowNames = ['A', 'B', 'C', 'D'];
const laneNames = ['Left', 'Right'];

/** §2 · I1. A decisive component share can sit on almost no density. */
const responsibilityPresets = {
  base: { label: 'Two separated bells', x: 1, weight: 0.5, variance: 1, means: [-2, 2] },
  identical: { label: 'Identical components', x: 0, weight: 0.2, variance: 1, means: [0, 0] },
};
export function ResponsibilityLab() {
  const state = useInvestigation(responsibilityPresets.base);
  const [saved, setSaved] = useState([]);
  const build = inputs => [
    { weight: inputs.weight, mean: inputs.means[0], variance: inputs.variance },
    { weight: 1 - inputs.weight, mean: inputs.means[1], variance: 1 },
  ];
  const modelKey = inputs => JSON.stringify([inputs.weight, inputs.variance, inputs.means]);
  // A saved pair belongs to one frozen model. Changing the model discards it
  // rather than hiding it, so it cannot reappear after a round trip.
  const forget = () => setSaved([]);
  const active = state.active;
  const components = build(active);
  const evaluated = mixtureAt(components, active.x);
  const answerFor = inputs => {
    const state_ = mixtureAt(build(inputs), inputs.x);
    return Math.abs(state_.components[0].logWeighted - state_.components[1].logWeighted) <= 1e-10
      ? 'equal' : state_.responsibilities[0] > state_.responsibilities[1] ? 'A' : 'B';
  };
  const identical = active.means[0] === active.means[1] && active.variance === 1;
  const crossings = responsibilityCrossings(components);
  const boundaries = crossings.roots.filter(root => root >= limits.location.minimum && root <= limits.location.maximum);
  const span = [Math.min(-6, active.x - 1), Math.max(6, active.x + 1)];
  // The plotted sum can peak between the component means. Measure the same
  // bounded grid used by the curve, rather than guessing from the two means.
  const peak = Math.max(...Array.from({ length: 241 }, (_, index) => mixtureAt(components, span[0] + (span[1] - span[0]) * index / 240).density), evaluated.density, 0.05);
  const pair = saved.filter(point => point.model === modelKey(active));
  const met = pair.length === 2
    && pair.every(point => point.responsibility > 0.99)
    && Math.abs(pair[0].negativeLogDensity - pair[1].negativeLogDensity) >= 5;

  const load = preset => { state.load(responsibilityPresets[preset]); setSaved([]); };
  return <Investigation
    title="Which component, and how much density?"
    question="Two components sit at fixed means with mixing weights that must sum to 1. Choose a measurement, record which component you think will take the larger share, then compare that share with the total density at the same place."
    note="A responsibility compares the components with one another. A density compares locations under the whole model. Neither is a species probability."
    onReset={() => { state.reset(); setSaved([]); }}>
    <div className="gm-controls">
      <NumberField label="Measurement x" value={state.draft.x} min={limits.location.minimum} max={limits.location.maximum}
        step="0.1" decimals={12} onChange={x => state.edit({ x })} />
      <NumberField label={`Weight of ${componentNames[0]}`} value={state.draft.weight} min={limits.weight.minimum} max={limits.weight.maximum}
        step="0.05" decimals={2} onChange={weight => { forget(); state.edit({ weight }); }} />
      <NumberField label={`Variance of ${componentNames[0]}`} value={state.draft.variance} min={limits.variance.minimum} max={limits.variance.maximum}
        step="0.25" decimals={2} onChange={variance => { forget(); state.edit({ variance }); }} />
      <NumberField label={`Mean of ${componentNames[0]}`} value={state.draft.means[0]} min={limits.mean.minimum} max={limits.mean.maximum}
        step="0.5" decimals={2} onChange={mean => { forget(); state.edit({ means: [mean, state.draft.means[1]] }); }} />
      <NumberField label={`Mean of ${componentNames[1]}`} value={state.draft.means[1]} min={limits.mean.minimum} max={limits.mean.maximum}
        step="0.5" decimals={2} onChange={mean => { forget(); state.edit({ means: [state.draft.means[0], mean] }); }} />
    </div>
    <p className="gm-caption">
      B keeps weight {round(1 - state.draft.weight)} and variance 1, so the two weights always sum to 1.
      Measurements run from {limits.location.minimum} to {limits.location.maximum}, means from {limits.mean.minimum} to {limits.mean.maximum},
      A's weight from {limits.weight.minimum} to {limits.weight.maximum} and A's variance from {limits.variance.minimum} to {limits.variance.maximum}.
      Numbers below 10⁻⁴ are shown in scientific notation. A displayed 1 is a rounded value, never exact certainty.
      {crossings.everywhere
        ? ' These identical components have equal weights, so the responsibilities tie at every measurement.'
        : identical
          ? ' These identical components have unequal weights: responsibilities stay at those weights everywhere, with no tie.'
          : boundaries.length
            ? ` The responsibilities tie at x = ${boundaries.map(value => round(value, 9)).join(' and ')}. Each crossing inside the plot is marked.`
            : ' There is no responsibility tie inside the supported measurement range; that does not mean the components are identical.'}
    </p>
    <div className="gm-buttons">
      {Object.entries(responsibilityPresets).map(([key, preset]) => (
        <button key={key} type="button" onClick={() => load(key)}>{preset.label}</button>
      ))}
      {boundaries.map((boundary, index) => <button key={index} type="button" disabled={state.pending}
        onClick={() => state.edit({ x: boundary })}>Use tie {index + 1} as measurement</button>)}
    </div>
    <Prediction
      prompt={`At x = ${round(state.draft.x)}, which component takes the greater responsibility?`}
      options={[['A', 'Component A'], ['B', 'Component B'], ['equal', 'They are equal']]}
      state={state} answerFor={answerFor}
      describe={identical
        ? `The two components are identical here, so their densities cancel in the ratio and the responsibilities are exactly the mixing weights ${round(active.weight)} and ${round(1 - active.weight)} at every measurement.`
        : `A contributes ${round(evaluated.components[0].weighted)} and B contributes ${round(evaluated.components[1].weighted)}. Dividing by their sum ${round(evaluated.density)} gives ${round(evaluated.responsibilities[0])} and ${round(evaluated.responsibilities[1])}.`} />

    <Plot caption="The weighted component curves, their sum, and the measurement"
      height={190} domain={span} range={[0, peak * 1.15]}
      describe={`Component A is centred at ${round(active.means[0])} with variance ${round(active.variance)} and weight ${round(active.weight)}; component B at ${round(active.means[1])} with variance 1 and weight ${round(1 - active.weight)}. The solid curve is their sum.${state.result ? ` At x = ${round(active.x)} the mixture density is ${round(evaluated.density)} and the responsibilities are ${round(evaluated.responsibilities[0])} and ${round(evaluated.responsibilities[1])}.` : ' Record a prediction or calculate to reveal the selected measurement’s exact allocation.'}`}>
      {(scaleX, scaleY) => <>
        <polyline className="gm-curve is-left" points={curve(scaleX, scaleY, span, value => components[0].weight * Math.exp(-0.5 * ((value - components[0].mean) ** 2 / components[0].variance + Math.log(2 * Math.PI * components[0].variance))))} />
        <polyline className="gm-curve is-right" points={curve(scaleX, scaleY, span, value => components[1].weight * Math.exp(-0.5 * ((value - components[1].mean) ** 2 / components[1].variance + Math.log(2 * Math.PI * components[1].variance))))} />
        <polyline className="gm-curve is-mixture" points={curve(scaleX, scaleY, span, value => mixtureAt(components, value).density)} />
        {state.result && <>
          <line className="gm-stem is-left" x1={scaleX(active.x) - 3} x2={scaleX(active.x) - 3} y1={scaleY(0)} y2={scaleY(evaluated.components[0].weighted)} />
          <line className="gm-stem is-right" x1={scaleX(active.x) + 3} x2={scaleX(active.x) + 3} y1={scaleY(0)} y2={scaleY(evaluated.components[1].weighted)} />
        </>}
        {boundaries.filter(boundary => boundary >= span[0] && boundary <= span[1]).map((boundary, index) => <g key={index}>
          <line className="gm-axis" x1={scaleX(boundary)} x2={scaleX(boundary)} y1={scaleY(0)} y2={scaleY(peak * 1.1)} strokeDasharray="4 3" />
          <text x={scaleX(boundary)} y={scaleY(peak * (index === 0 ? 1.12 : 0.9))} textAnchor="middle">tie {index + 1}</text>
        </g>)}
        <circle className="gm-mark" cx={scaleX(active.x)} cy={scaleY(0)} r="5" />
        <text x={scaleX(active.x)} y={scaleY(0) - 10} textAnchor="middle">x = {round(active.x, 2)}</text>
      </>}
    </Plot>

    {state.result && <>
      <div className="gm-bars" role="img" aria-label={`Normalized allocation at x equals ${round(active.x)}: component A ${round(evaluated.responsibilities[0])}, component B ${round(evaluated.responsibilities[1])}.`}>
        <div className="gm-bar-row">
          <span>allocation</span>
          <span className="gm-bar-track">
            <span className="gm-bar-left" style={{ width: `${100 * evaluated.responsibilities[0]}%` }} />
            <span className="gm-bar-right" style={{ width: `${100 * evaluated.responsibilities[1]}%` }} />
          </span>
          <span className="gm-bar-value">A {percent(evaluated.responsibilities[0], 2)} · B {percent(evaluated.responsibilities[1], 2)}</span>
        </div>
      </div>
      <Table caption="The two weighted densities, their sum, and what the sum is used for"
        headings={['quantity', 'component A', 'component B', 'together']}
        rows={[
          ['weight', round(components[0].weight), round(components[1].weight), '1'],
          ['density at x', round(evaluated.components[0].density), round(evaluated.components[1].density), '—'],
          ['weighted density', round(evaluated.components[0].weighted), round(evaluated.components[1].weighted), round(evaluated.density)],
          ['responsibility', round(evaluated.responsibilities[0]), round(evaluated.responsibilities[1]), '1'],
        ]} />
      <p className="gm-readout" aria-live="polite">
        Mixture density {round(evaluated.density)} per measurement unit. Log-density {round(evaluated.logDensity)};
        negative log-density {round(evaluated.negativeLogDensity)}. The density says how ordinary this location is under the whole model;
        the allocation above says only which component explains it.
      </p>
      <div className="gm-buttons">
        <button type="button" onClick={() => setSaved(previous => {
          const kept = previous.filter(point => point.model === modelKey(active));
          return [...kept.slice(-1), {
            model: modelKey(active), x: active.x, density: evaluated.density,
            responsibility: evaluated.responsibilities[1], negativeLogDensity: evaluated.negativeLogDensity,
          }];
        })}>Save this point for the comparison</button>
        <span>{pair.length} of 2 saved on this model</span>
      </div>
    </>}

    {pair.length === 2 && <>
      <Table caption="Two locations under the same frozen model"
        headings={['x', 'mixture density', 'responsibility of B', 'negative log-density']}
        rows={pair.map(point => [round(point.x, 2), round(point.density), round(point.responsibility), round(point.negativeLogDensity)])} />
      <p className={met ? 'gm-readout' : 'gm-caption'} aria-live="polite">
        {met
          ? `Both points give B a responsibility above 0.99, and their negative log-densities differ by ${round(Math.abs(pair[0].negativeLogDensity - pair[1].negativeLogDensity), 4)}. A component can be almost certain about a location the model finds very unlikely.`
          : `Not yet the target pair: it needs both responsibilities for B above 0.99 and negative log-densities at least 5 apart. Here they are ${round(pair[0].responsibility, 4)} and ${round(pair[1].responsibility, 4)}, differing by ${round(Math.abs(pair[0].negativeLogDensity - pair[1].negativeLogDensity), 4)}.`}
      </p>
    </>}
  </Investigation>;
}

/** §4 · I2. Step the constrained update and watch the objective. */
const emPresets = {
  standard: { label: 'Standard data', observations: [-2, -1, 1, 2], means: [-1, 1], variances: [1, 1], floor: 0.05 },
  asymmetric: { label: 'Asymmetric start', observations: [-2, -1, 1, 2], means: [-2, -1], variances: [1, 1], floor: 0.05 },
  identical: { label: 'Identical components', observations: [-2, -1, 1, 2], means: [0, 0], variances: [2.5, 2.5], floor: 0.05 },
  repeated: { label: 'Repeated measurements', observations: [-2, -2, 2, 2], means: [-2, 2], variances: [1, 1], floor: 0.25 },
};
export function EmStepLab() {
  const [setup, setSetup] = useState(emPresets.standard);
  const [draft, setDraft] = useState(emPresets.standard);
  const [run, setRun] = useState(null);
  const [choice, setChoice] = useState('');
  const [reason, setReason] = useState('');
  const [recorded, setRecorded] = useState(null);
  const [problem, setProblem] = useState(null);

  const startComponents = current => current.means.map((mean, index) => ({ weight: 0.5, mean, variance: current.variances[index] }));
  const startState = current => {
    const components = startComponents(current);
    const step = expectation(current.observations, components);
    return {
      // The opening allocation is shown, not withheld: the prediction is about
      // the direction of the objective, so the responsibilities give nothing away.
      iteration: 0, phase: 'ready', components, responsibilities: step.responsibilities, opening: true,
      logLikelihood: step.logLikelihood, history: [{ iteration: 0, logLikelihood: step.logLikelihood }], past: [],
    };
  };
  const current = run ?? startState(setup);
  const pending = JSON.stringify(draft) !== JSON.stringify(setup);

  const apply = next => {
    setSetup(next); setDraft(next); setRun(startState(next)); setChoice(''); setReason(''); setRecorded(null); setProblem(null);
  };
  const editDraft = update => { setDraft(previous => ({ ...previous, ...update })); setChoice(''); setReason(''); setRecorded(null); };

  const snapshot = () => ({ ...current, past: undefined });
  const computeExpectation = () => {
    const step = expectation(setup.observations, current.components);
    setRun({
      ...current, phase: 'expected', responsibilities: step.responsibilities,
      logLikelihood: step.logLikelihood, past: [...current.past, snapshot()],
    });
  };
  const applyMaximization = () => {
    try {
      const updated = maximization(setup.observations, current.responsibilities, setup.floor);
      const after = expectation(setup.observations, updated);
      const iteration = current.iteration + 1;
      setRun({
        iteration, phase: 'ready', components: updated, responsibilities: current.responsibilities,
        logLikelihood: after.logLikelihood,
        history: [...current.history, { iteration, logLikelihood: after.logLikelihood }],
        past: [...current.past, snapshot()],
        justUpdated: { before: current.components, gain: after.logLikelihood - current.logLikelihood, counts: updated.map(part => part.count) },
      });
      if (recorded && recorded.iteration === current.iteration) {
        const rise = after.logLikelihood - current.logLikelihood;
        setRecorded({ ...recorded, outcome: Math.abs(rise) < 1e-9 ? 'same' : 'rises', gain: rise,
          identical: current.components[0].mean === current.components[1].mean && current.components[0].variance === current.components[1].variance });
      }
      // The next iteration asks a new question, so it starts with nothing chosen.
      setChoice('');
      setReason('');
      setProblem(null);
    } catch (failure) {
      setProblem(failure.message);
    }
  };
  const back = () => {
    if (current.past.length === 0) return;
    const previous = current.past[current.past.length - 1];
    setRun({ ...previous, past: current.past.slice(0, -1) });
    // A verdict describes a cycle that has now been undone, so it goes with it.
    if (recorded && recorded.iteration >= previous.iteration) setRecorded(null);
    setProblem(null);
  };

  const components = current.components;
  const span = [-5, 5];
  const peak = Math.max(...components.map(component => component.weight / Math.sqrt(2 * Math.PI * component.variance))) * 1.25;
  const lowest = Math.min(...current.history.map(point => point.logLikelihood));
  const highest = Math.max(...current.history.map(point => point.logLikelihood));
  const graded = recorded?.outcome !== undefined;
  const atLimit = current.iteration >= 50;

  return <Investigation
    title="Step one exact EM cycle from your own start"
    question={`Four measurements, two components, and a variance floor that is part of the model rather than a nudge: ${round(setup.floor)} in the setup now applied. Edit the setup, apply it, record what the next full cycle will do to the log-likelihood, then run the two half-steps separately.`}
    note="The E-step leaves the parameters, and therefore the observed log-likelihood, exactly where they were: only the allocations are recomputed. The M-step is where the objective can move."
    onReset={() => apply(emPresets.standard)}>
    <div className="gm-controls">
      {draft.observations.map((value, index) => (
        <NumberField key={index} label={`Measurement ${rowNames[index]}`} value={value}
          min={limits.observation.minimum} max={limits.observation.maximum} step="0.5" decimals={2}
          onChange={next => editDraft({ observations: draft.observations.map((old, position) => (position === index ? next : old)) })} />
      ))}
      {draft.means.map((value, index) => (
        <NumberField key={index} label={`Initial ${laneNames[index]} mean`} value={value}
          min={limits.mean.minimum} max={limits.mean.maximum} step="0.5" decimals={2}
          onChange={next => editDraft({ means: draft.means.map((old, position) => (position === index ? next : old)) })} />
      ))}
    </div>
    <p className="gm-caption">
      Both components start with weight 0.5 and variance {round(draft.variances[0])}; the variance floor is {round(draft.floor)}.
      Those three are displayed inputs of the chosen setup, not extra sliders.
    </p>
    <div className="gm-buttons">
      <button type="button" className="is-primary" disabled={!pending} onClick={() => apply(draft)}>Apply setup</button>
      {Object.entries(emPresets).map(([key, preset]) => (
        <button key={key} type="button" onClick={() => apply(preset)}>{preset.label}</button>
      ))}
    </div>
    {pending && <p className="gm-pending" role="status">Edited values are not in the model yet. Apply the setup to use them.</p>}

    <div className="gm-prediction">
      <fieldset>
        <legend>Record a prediction first.</legend>
        <p>Starting from iteration {current.iteration}, what will one complete E-step and M-step do to the total log-likelihood?</p>
        <div className="gm-choices">
          {[['rises', 'It rises'], ['same', 'It stays the same to within 1e-9']].map(([value, text]) => (
            <label className="gm-choice" key={value}>
              <input type="radio" name="em-prediction" value={value} checked={choice === value}
              onChange={() => setChoice(value)} disabled={current.phase !== 'ready' || pending || atLimit || recorded?.iteration === current.iteration} />
              <span>{text}</span>
            </label>
          ))}
        </div>
      </fieldset>
      <Reason value={reason} onChange={setReason} disabled={recorded?.iteration === current.iteration} />
      <div className="gm-buttons">
        <button type="button" disabled={choice === '' || pending || atLimit || current.phase !== 'ready' || (recorded?.iteration === current.iteration)}
          onClick={() => setRecorded({ iteration: current.iteration, choice, reason, from: current.logLikelihood })}>Record it</button>
        <button type="button" className="is-primary" disabled={pending || atLimit || current.phase !== 'ready'} onClick={computeExpectation}>Compute E-step</button>
        <button type="button" className="is-primary" disabled={pending || atLimit || current.phase !== 'expected'} onClick={applyMaximization}>Apply M-step</button>
        <button type="button" disabled={current.past.length === 0} onClick={back}>Back one half-step</button>
      </div>
      {graded && recorded.reason && <p className="gm-caption">Your reason, kept as you wrote it: “{recorded.reason}”</p>}
      {graded && <p className={`gm-verdict ${recorded.choice === recorded.outcome ? '' : 'is-miss'}`} role="status">
        <span className="gm-verdict-mark" aria-hidden="true">{recorded.choice === recorded.outcome ? '=' : '≠'}</span>
        The cycle from iteration {recorded.iteration} moved the log-likelihood by {signed(recorded.gain, 9)}, so it {recorded.outcome === 'same' ? 'stayed the same to within 1e-9' : 'rose'}.
        {recorded.outcome === 'same'
          ? recorded.identical
            ? ' Identical components give every row the same allocation, and repeating an identical operation cannot break that symmetry. An unchanged objective does not identify a useful solution.'
            : ' This cycle is on a numerical plateau: the change is below the stated tolerance even though the components differ. A small change alone does not establish a global optimum.'
          : ' The bound touched the old objective at the E-step, and the M-step lifted it.'}
      </p>}
      {problem && <p className="gm-note" role="status">{problem}</p>}
      {atLimit && <p className="gm-note" role="status">This investigation keeps at most 50 cycles. Inspect the history, step back or reset to try another setup.</p>}
    </div>

    <p className="gm-readout" aria-live="polite">
      Iteration {current.iteration}, {current.phase === 'expected' ? 'responsibilities recomputed and waiting for the M-step' : 'parameters applied'}.
      Total log-likelihood {round(current.logLikelihood, 9)}.
      {current.justUpdated && current.phase === 'ready' && ` The M-step moved it by ${signed(current.justUpdated.gain, 9)} using effective counts ${current.justUpdated.counts.map(count => round(count, 6)).join(' and ')}.`}
    </p>

    <div className="gm-em-curves">
      <ul className="gm-curve-key" aria-label="EM component curve key" style={{ display: 'flex', flexWrap: 'wrap', gap: '.5rem 1.25rem', padding: 0, margin: '.8rem 0', listStyle: 'none', fontSize: '.9rem', lineHeight: 1.6 }}>
        {laneNames.map((name, index) => <li key={name} style={{ display: 'flex', gap: '.5rem', alignItems: 'center', minWidth: 0 }}>
          <span aria-hidden="true" style={{ flex: '0 0 28px', height: 2, background: index === 0
            ? 'repeating-linear-gradient(to right, #8eb9a5 0 5px, transparent 5px 8px)'
            : 'repeating-linear-gradient(to right, #91aecf 0 2px, transparent 2px 5px)' }} />
          <span>{name} component · {index === 0 ? 'green, long dashes' : 'blue, short dashes'}</span>
        </li>)}
      </ul>
    <Plot caption="The two component curves and the four measurements" height={180} domain={span} range={[0, peak]}
      describe={`${laneNames[0]} has mean ${round(components[0].mean)} and variance ${round(components[0].variance)}; ${laneNames[1]} has mean ${round(components[1].mean)} and variance ${round(components[1].variance)}. The measurements are at ${setup.observations.map(value => round(value)).join(', ')}.`}>
      {(scaleX, scaleY) => <>
        {components.map((component, index) => (
          <polyline key={index} className={`gm-curve ${index === 0 ? 'is-left' : 'is-right'}`}
            points={curve(scaleX, scaleY, span, value => component.weight * Math.exp(-0.5 * ((value - component.mean) ** 2 / component.variance + Math.log(2 * Math.PI * component.variance))))} />
        ))}
        {components.map((component, index) => (
          <line key={index} className={`gm-stem ${index === 0 ? 'is-left' : 'is-right'}`}
            x1={scaleX(component.mean)} x2={scaleX(component.mean)} y1={scaleY(peak * 0.28)} y2={scaleY(peak * 0.9)} />
        ))}
        {Object.entries(setup.observations.reduce((groups, value, index) => {
          const key = String(value);
          return { ...groups, [key]: [...(groups[key] ?? []), rowNames[index]] };
        }, {})).map(([value, ids]) => <g key={value}>
          <circle className="gm-observation" cx={scaleX(Number(value))} cy={scaleY(0)} r="4" />
          <text x={scaleX(Number(value))} y={scaleY(0) - 9} textAnchor="middle">{ids.join(', ')}</text>
        </g>)}
      </>}
    </Plot>
    </div>

    {current.responsibilities && <Table
      caption={current.phase === 'expected'
        ? 'Responsibilities just computed, and about to be used for the M-step'
        : current.opening
          ? 'The allocation these starting parameters already imply, before any step'
          : 'The responsibilities used for this M-step, kept as they were'}
      headings={['observation', 'x', `${laneNames[0]} share`, `${laneNames[1]} share`, 'row sum']}
      rows={setup.observations.map((value, index) => [
        rowNames[index], round(value, 2),
        round(current.responsibilities[index][0], 9), round(current.responsibilities[index][1], 9),
        round(current.responsibilities[index].reduce((sum, share) => sum + share, 0), 9),
      ])} />}

    <Table caption={current.justUpdated && current.phase === 'ready'
      ? 'Component parameters before and after this M-step, with the weighted scatter the floor acts on'
      : 'Component parameters at this step'}
      headings={current.justUpdated && current.phase === 'ready'
        ? ['component', 'weight', 'mean before', 'mean after', 'variance before', 'weighted scatter', 'variance after']
        : ['component', 'weight', 'mean', 'variance', 'at the floor?']}
      rows={components.map((component, index) => (current.justUpdated && current.phase === 'ready'
        ? [laneNames[index], round(component.weight, 9),
          round(current.justUpdated.before[index].mean, 9), round(component.mean, 9),
          round(current.justUpdated.before[index].variance, 9), round(component.scatter, 9),
          component.atFloor ? `${round(component.variance, 9)} (floor)` : round(component.variance, 9)]
        : [laneNames[index], round(component.weight, 9), round(component.mean, 9), round(component.variance, 9),
          component.atFloor ? `yes, held at ${round(setup.floor)}` : 'no']))} />

    {current.history.length > 1 && <Plot caption="Total log-likelihood by iteration, starting at iteration 0"
      height={160} domain={[0, Math.max(1, current.history.length - 1)]}
      range={[lowest - 0.2, highest + 0.2]}
      ticks={current.history.filter(point => point.iteration % Math.max(1, Math.ceil(current.iteration / 5)) === 0 || point.iteration === current.iteration).map(point => point.iteration)}
      describe={`Log-likelihood by iteration: ${current.history.map(point => `${point.iteration} gives ${round(point.logLikelihood, 6)}`).join('; ')}.`}>
      {(scaleX, scaleY) => <>
        <polyline className="gm-curve is-mixture" points={current.history.map(point => `${scaleX(point.iteration)},${scaleY(point.logLikelihood)}`).join(' ')} />
        {current.history.map(point => <circle key={point.iteration} className="gm-mark" cx={scaleX(point.iteration)} cy={scaleY(point.logLikelihood)} r="3.5" />)}
        <text x={scaleX(0) + 6} y={scaleY(current.history[0].logLikelihood) - 8}>{round(current.history[0].logLikelihood, 6)}</text>
        <text x={scaleX(current.history.at(-1).iteration) - 6} y={scaleY(current.history.at(-1).logLikelihood) + 14} textAnchor="end">{round(current.history.at(-1).logLikelihood, 6)}</text>
      </>}
    </Plot>}
  </Investigation>;
}

/** §5 · I3. Equal Euclidean distance, unequal plausibility. */
const covariancePreset = { rho: 0.75, first: [1, 1], second: [1, -1] };
export function CovarianceLab() {
  const state = useInvestigation(covariancePreset);
  const [projectionPoint, setProjectionPoint] = useState('P');
  const active = state.active;
  const comparison = planeComparison(active.first, active.second, active.rho);
  const nullComparison = planeComparison(active.first, active.second, 0);
  const answerFor = inputs => {
    const order = planeComparison(inputs.first, inputs.second, inputs.rho).order;
    return order === 'equal' ? 'equal' : order === 'first' ? 'P' : 'Q';
  };

  // The viewBox is close to the rendered width, so 13 user units of type land
  // near 14 px at a phone width instead of being scaled down to nine.
  const size = 246;
  const low = -3.5;
  const high = 3.5;
  const place = value => 30 + 196 * (value - low) / (high - low);
  const lift = value => 226 - 196 * (value - low) / (high - low);
  const contour = (rho, level) => {
    const geometry = correlationGeometry(rho);
    return Array.from({ length: 97 }, (_, index) => {
      const angle = 2 * Math.PI * index / 96;
      const [firstAxis, secondAxis] = geometry.axes;
      const x = Math.sqrt(level * firstAxis.value) * Math.cos(angle) * firstAxis.direction[0]
        + Math.sqrt(level * secondAxis.value) * Math.sin(angle) * secondAxis.direction[0];
      const y = Math.sqrt(level * firstAxis.value) * Math.cos(angle) * firstAxis.direction[1]
        + Math.sqrt(level * secondAxis.value) * Math.sin(angle) * secondAxis.direction[1];
      return `${place(x).toFixed(2)},${lift(y).toFixed(2)}`;
    }).join(' ');
  };
  const panel = (rho, title) => {
    const geometry = correlationGeometry(rho);
    const left = planeDensity(active.first, rho);
    const right = planeDensity(active.second, rho);
    return <div className="gm-panel" key={title}>
      <h4>{title}</h4>
      <svg className="gm-dense" viewBox={`0 0 260 ${size}`} role="img"
        aria-label={`Correlation ${round(rho, 2)}. The unit contour has semiaxes ${round(geometry.axes[0].semiaxis, 4)} along (1, 1) and ${round(geometry.axes[1].semiaxis, 4)} along (1, −1). P is at ${active.first.join(', ')}; Q is at ${active.second.join(', ')}.${state.result ? ` Squared distances are ${round(left.squared, 6)} and ${round(right.squared, 6)}; densities ${round(left.density)} and ${round(right.density)}.` : ' Predict the density ordering before revealing the quadratic forms.'}`}>
        <line className="gm-grid" x1={place(low)} x2={place(high)} y1={lift(0)} y2={lift(0)} />
        <line className="gm-grid" x1={place(0)} x2={place(0)} y1={lift(low)} y2={lift(high)} />
        {[-3, 3].map(value => <g key={value}>
          <text x={place(value)} y={lift(0) + 16} textAnchor="middle">{value}</text>
          <text x={place(0) - 7} y={lift(value) + 5} textAnchor="end">{value}</text>
        </g>)}
        <polyline className="gm-ellipse is-outer" points={contour(rho, 4)} />
        <polyline className="gm-ellipse" points={contour(rho, 1)} />
        {geometry.axes.map(axis => (
          <line key={axis.label} className="gm-eigen"
            x1={place(-axis.semiaxis * axis.direction[0])} y1={lift(-axis.semiaxis * axis.direction[1])}
            x2={place(axis.semiaxis * axis.direction[0])} y2={lift(axis.semiaxis * axis.direction[1])} />
        ))}
        {state.result && geometry.axes.map(axis => {
          const point = projectionPoint === 'P' ? active.first : active.second;
          const coordinate = point[0] * axis.direction[0] + point[1] * axis.direction[1];
          const foot = axis.direction.map(value => coordinate * value);
          return <g key={`projection-${axis.label}`}>
            <line className="gm-eigen" strokeDasharray="3 3" x1={place(point[0])} y1={lift(point[1])} x2={place(foot[0])} y2={lift(foot[1])} />
            <circle className="gm-observation" cx={place(foot[0])} cy={lift(foot[1])} r="3" />
          </g>;
        })}
        <circle className="gm-mark" cx={place(active.first[0])} cy={lift(active.first[1])} r="5.5" />
        <text className="gm-point-id" x={place(active.first[0]) + 10} y={lift(active.first[1]) - 8}>P</text>
        <rect className="gm-mark is-hollow" x={place(active.second[0]) - 5} y={lift(active.second[1]) - 5} width="10" height="10" />
        <text className="gm-point-id" x={place(active.second[0]) + 10} y={lift(active.second[1]) - 8}>Q</text>
        <circle className="gm-observation" cx={place(0)} cy={lift(0)} r="3" />
        <text x={place(0) + 12} y={lift(0) + 5}>mean</text>
      </svg>
      {state.result && <p>
        D²(P) = {round(left.squared, 6)}, D²(Q) = {round(right.squared, 6)}, determinant {round(geometry.determinant, 6)}.
        Densities {round(left.density)} and {round(right.density)}.
      </p>}
    </div>;
  };

  return <Investigation
    title="Same distance from the centre, different plausibility"
    question="One Gaussian with mean (0, 0) and both marginal variances fixed at 1, so the correlation alone changes its shape. Move either point, record which one you think the component finds more plausible, then compare the two quadratic forms."
    note="This compares two locations under one declared density. Nothing here is fitted, and no responsibility is being normalised."
    onReset={state.reset}>
    <div className="gm-controls">
      <NumberField label="Correlation ρ" value={state.draft.rho} min={limits.correlation.minimum} max={limits.correlation.maximum}
        step="0.05" decimals={2} onChange={rho => state.edit({ rho })} />
      <NumberField label="P horizontal" value={state.draft.first[0]} min={limits.point.minimum} max={limits.point.maximum}
        step="0.25" decimals={2} onChange={value => state.edit({ first: [value, state.draft.first[1]] })} />
      <NumberField label="P vertical" value={state.draft.first[1]} min={limits.point.minimum} max={limits.point.maximum}
        step="0.25" decimals={2} onChange={value => state.edit({ first: [state.draft.first[0], value] })} />
      <NumberField label="Q horizontal" value={state.draft.second[0]} min={limits.point.minimum} max={limits.point.maximum}
        step="0.25" decimals={2} onChange={value => state.edit({ second: [value, state.draft.second[1]] })} />
      <NumberField label="Q vertical" value={state.draft.second[1]} min={limits.point.minimum} max={limits.point.maximum}
        step="0.25" decimals={2} onChange={value => state.edit({ second: [state.draft.second[0], value] })} />
    </div>
    <Prediction
      prompt={`With ρ = ${round(state.draft.rho, 2)}, which point has the higher density: P at (${state.draft.first.map(value => round(value, 2)).join(', ')}) or Q at (${state.draft.second.map(value => round(value, 2)).join(', ')})?`}
      options={[['P', 'P has the higher density'], ['Q', 'Q has the higher density'], ['equal', 'They are equal']]}
      state={state} answerFor={answerFor}
      describe={`Squared Mahalanobis distances are ${round(comparison.left.squared, 6)} and ${round(comparison.right.squared, 6)}. Both points share the determinant ${round(comparison.geometry.determinant, 6)} inside this panel, so the quadratic forms alone decide the order.`} />
    <div className="gm-panels is-stacked">
      {panel(active.rho, `Correlation ${round(active.rho, 2)}`)}
      {panel(0, 'The same two points at correlation 0')}
    </div>
    {state.result && <>
    <Field label="Inspect projections of"><select value={projectionPoint} onChange={event => setProjectionPoint(event.target.value)}><option>P</option><option>Q</option></select></Field>
    <Table caption="Resolve the selected point along the covariance axes"
      headings={['axis', 'coordinate along axis', 'eigenvalue', 'squared coordinate / eigenvalue']}
      rows={comparison.geometry.axes.map(axis => {
        const point = projectionPoint === 'P' ? active.first : active.second;
        const coordinate = point[0] * axis.direction[0] + point[1] * axis.direction[1];
        return [axis.label, round(coordinate, 6), round(axis.value, 6), round(coordinate ** 2 / axis.value, 6)];
      })} />
    <p className="gm-caption">The dotted segments show the selected point’s perpendicular projections. Add the two squared-coordinate contributions to get its squared Mahalanobis distance; a smaller eigenvalue charges more for the same displacement.</p>
    <Table caption="Both panels, exactly"
      headings={['model', 'D²(P)', 'D²(Q)', 'determinant', 'density at P', 'density at Q']}
      rows={[
        [`ρ = ${round(active.rho, 2)}`, round(comparison.left.squared, 6), round(comparison.right.squared, 6),
          round(comparison.geometry.determinant, 6), round(comparison.left.density), round(comparison.right.density)],
        ['ρ = 0', round(nullComparison.left.squared, 6), round(nullComparison.right.squared, 6), '1',
          round(nullComparison.left.density), round(nullComparison.right.density)],
      ]} />
    </>}
    <p className="gm-caption">
      The inner ellipse is the contour where the squared Mahalanobis distance is 1, and the dashed one where it is 4.
      In two dimensions the inner contour encloses {round(100 * correlationGeometry(active.rho).massInsideUnitContour, 4)}% of this Gaussian's mass.
      The familiar 68% belongs to a one-dimensional interval, not to this ellipse.
      Eigenvalues are {round(1 + active.rho, 4)} along (1, 1) and {round(1 - active.rho, 4)} along (1, −1); at ρ = 0 those two directions are a drawing convention, since every direction is then an eigenvector.
    </p>
    <details>
      <summary>Two transfers worth recording separately</summary>
      <p className="gm-caption">
        First, find a pair of points whose order reverses when you change the sign of ρ, and say which squared projection changed its weighting.
        Second, find two points away from the original diagonal that have equal density under the active model.
        Record a prediction for each before you check it; the comparison above is exact to 10⁻¹⁰, so a near miss is a miss.
      </p>
    </details>
  </Investigation>;
}
