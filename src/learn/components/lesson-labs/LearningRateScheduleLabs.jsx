import { useId, useMemo, useState } from 'react';
import { buildPlateauTrace, buildRateSchedule, buildScheduleClockTrace, parseValidationMetrics, scheduleNoiseMoments } from '../../data/learning-rate-schedule-models.js';
import './learning-rate-schedule-labs.css';
const formatValue = value => {
  if (value === null) return '—';
  if (value === 0) return '0';
  if (Math.abs(value) < .001 || Math.abs(value) >= 10000) return value.toExponential(2);
  return Number(value.toFixed(5)).toString();
};
export function CalibrationUpdateFigure() {
  const id = useId();
  const x = value => 42 + (value - 2) * 80;
  return <figure className="schedule-inline">
    <p className="schedule-axis-label">Target 2 · η = .2 in both updates</p>
    <svg className="schedule-plot" viewBox="0 0 340 232" role="img" aria-labelledby={`${id}-title ${id}-description`}>
      <title id={`${id}-title`}>Same rate, smaller second displacement</title>
      <desc id={`${id}-description`}>On a common parameter axis, the first update moves from 5 to 3.8, a displacement of minus 1.2. The second moves from 3.8 to 3.08, a displacement of minus 0.72. Both use rate 0.2 and curvature 2 toward target 2.</desc>
      <defs><marker id={`${id}-arrow`} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#e2b55a" /></marker></defs>
      <line x1={x(2)} x2={x(2)} y1="29" y2="194" stroke="#8aabba" strokeDasharray="4 4" />
      {[[5, 3.8, 65, 'first: −1.2'], [3.8, 3.08, 140, 'second: −.72']].map(([start, end, y, label]) => <g key={label}>
        <line x1={x(start)} x2={x(end) + 5} y1={y} y2={y} stroke="#e2b55a" strokeWidth="2" markerEnd={`url(#${id}-arrow)`} />
        <circle cx={x(start)} cy={y} r="4" fill="#b0c2d5" /><circle cx={x(end)} cy={y} r="4" fill="#e2b55a" />
        <text x={x(start)} y={y - 14} textAnchor="middle">{start}</text><text x={x(end)} y={y - 14} textAnchor="middle">{end}</text>
        <text x={(x(start) + x(end)) / 2} y={y + 26} textAnchor="middle">{label}</text>
      </g>)}
      <line x1="42" x2="305" y1="194" y2="194" stroke="#8aabba" />
      {[2, 3, 4, 5].map(value => <g key={value}><line x1={x(value)} x2={x(value)} y1="191" y2="198" stroke="#8aabba" /><text x={x(value)} y="215" textAnchor="middle">{value}</text></g>)}
    </svg>
    <p className="schedule-axis-label">Horizontal axis: parameter θ · common scale for both updates</p>
    <figcaption>The arrows show actual displacements. The gradient shrinks from 6 to 3.6, so multiplying by the same rate .2 gives a shorter second step. Vertical separation only keeps the two updates readable.</figcaption>
  </figure>;
}
function ScheduleField({
  label,
  children
}) {
  const id = useId();
  return <div className="schedule-field"><label htmlFor={id}>{label}</label>{children(id)}</div>;
}
function ScheduleSelect({
  label,
  value,
  options,
  onChange
}) {
  return <ScheduleField label={label}>{id => <select id={id} value={value} onChange={event => onChange(event.target.value)}>{options.map(([key, text]) => <option key={key} value={key}>{text}</option>)}</select>}</ScheduleField>;
}
function ScheduleRange({
  label,
  value,
  minimum,
  maximum,
  step = 1,
  onChange
}) {
  return <ScheduleField label={`${label}: ${formatValue(value)}`}>{id => <input id={id} aria-label={label} type="range" min={minimum} max={maximum} step={step} value={value} onChange={event => onChange(Number(event.target.value))} />}</ScheduleField>;
}
function ScheduleReadout({
  values
}) {
  return <dl className="schedule-readout">{values.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>;
}
function SchedulePlot({
  title,
  description,
  series,
  selected = null,
  xMaximum,
  yMaximum,
  yMinimum = 0,
  xLabel = 'update index u',
  yLabel = 'learning rate η'
}) {
  const id = useId();
  const width = 340;
  const height = 245;
  const left = 58;
  const right = 321;
  const top = 29;
  const bottom = 202;
  const x = value => left + value / Math.max(1, xMaximum) * (right - left);
  const y = value => bottom - (value - yMinimum) / (yMaximum - yMinimum) * (bottom - top);
  const xTicks = [...new Set([0, Math.floor(xMaximum / 2), xMaximum])];
  const yTicks = [yMinimum, (yMaximum + yMinimum) / 2, yMaximum];
  return <svg className="schedule-plot" viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={`${id}-title ${id}-description`}>
    <title id={`${id}-title`}>{title}</title><desc id={`${id}-description`}>{description}</desc>
    <defs><clipPath id={`${id}-clip`}><rect x={left - 3} y={top - 3} width={right - left + 6} height={bottom - top + 6} /></clipPath></defs>
    {xTicks.map(tick => <g key={tick}><line className="schedule-grid" x1={x(tick)} x2={x(tick)} y1={top} y2={bottom} /><text x={x(tick)} y={bottom + 19} textAnchor="middle">{tick}</text></g>)}
    {yTicks.map(tick => <g key={tick}><line className="schedule-grid" x1={left} x2={right} y1={y(tick)} y2={y(tick)} /><text x={left - 7} y={y(tick) + 5} textAnchor="end">{formatValue(tick)}</text></g>)}
    <text x={left} y={17}>{yLabel}</text><text x={(left + right) / 2} y={height - 5} textAnchor="middle">{xLabel}</text>
    <g clipPath={`url(#${id}-clip)`}>
      {selected !== null && <line className="schedule-cursor" x1={x(selected)} x2={x(selected)} y1={top} y2={bottom} />}
      {series.map(({
        key,
        values,
        color,
        dashed = false,
        dots = false
      }) => <g key={key}>
        <polyline fill="none" stroke={color} strokeWidth="2" strokeDasharray={dashed ? '5 4' : undefined} points={values.map(([first, second]) => `${x(first)},${y(second)}`).join(' ')} />
        {dots && values.map(([first, second]) => <circle key={first} cx={x(first)} cy={y(second)} r={first === selected ? 4.5 : 2.5} fill={color} />)}
      </g>)}
    </g>
  </svg>;
}
function ScheduleStepper({
  position,
  maximum,
  onChange,
  reset,
  noun = 'event'
}) {
  return <div className="schedule-actions"><button disabled={position === 0} onClick={() => onChange(position - 1)}>Previous {noun}</button><button disabled={position === maximum} onClick={() => onChange(position + 1)}>Next {noun}</button><button disabled={position === maximum} onClick={() => onChange(maximum)}>Finish</button><button onClick={reset}>Reset</button></div>;
}
const kindOptions = [['constant', 'Constant'], ['cosine', 'Warmup + cosine'], ['linear', 'Linear decay'], ['exponential', 'Exponential decay'], ['step', 'Halve every period'], ['one-cycle', 'One cycle + momentum'], ['restart', 'Cosine restarts']];
export function ScheduleShapeLab() {
  const [kind, setKind] = useState('cosine');
  const [total, setTotal] = useState(24);
  const [warmup, setWarmup] = useState(4);
  const [selected, setSelected] = useState(0);
  const schedule = useMemo(() => buildRateSchedule({
    kind,
    total,
    warmup,
    rise: total / 4,
    period: 6
  }), [kind, total, warmup]);
  const current = schedule[selected];
  const reset = () => {
    setKind('cosine');
    setTotal(24);
    setWarmup(4);
    setSelected(0);
  };
  return <section className="schedule-lab" data-lab="schedule-shape" aria-label="Inspect a finite schedule">
    <h3>Inspect the rates the updates will actually use</h3>
    <p>Predict the first, peak and last used values. Change the policy, then inspect a point. Each dot is one scheduled update; connecting segments only help you follow their order.</p>
    <div className="schedule-controls">
      <ScheduleSelect label="Rate policy" value={kind} options={kindOptions} onChange={value => {
        setKind(value);
        setSelected(0);
      }} />
      <ScheduleSelect label="Update budget" value={total} options={[[12, '12 updates'], [24, '24 updates'], [48, '48 updates']]} onChange={value => {
        setTotal(Number(value));
        setSelected(0);
      }} />
      {kind === 'cosine' && <ScheduleRange label="Warmup updates" value={warmup} minimum={0} maximum={8} onChange={setWarmup} />}
    </div>
    <SchedulePlot title="Discrete used learning rates" description={`${kind} schedule with ${total} used rates. Peak parameter .2. The readout gives the selected rate and phase.`} series={[{
      key: 'rates',
      values: schedule.map(state => [state.update, state.rate]),
      color: '#f0c96d',
      dots: true
    }]} xMaximum={total - 1} yMaximum={.2} selected={selected} />
    <ScheduleRange label="Inspect update index" value={selected} minimum={0} maximum={total - 1} onChange={setSelected} />
    <ScheduleReadout values={[["Update about to run", `${selected} / ${total - 1}`], ['Rate consumed η', formatValue(current.rate)], ['Phase', current.phase], ['First / last used rate', `${formatValue(schedule[0].rate)} / ${formatValue(schedule.at(-1).rate)}`]]} />
    {kind === 'one-cycle' ? <>
      <SchedulePlot title="One-cycle momentum, on its own scale" description="Momentum decreases from .95 to .85 as the rate rises, then returns to .95. This plot has its own explicitly labelled axis." series={[{
        key: 'momentum',
        values: schedule.map(state => [state.update, state.momentum]),
        color: '#8dc7e8',
        dots: true
      }]} xMaximum={total - 1} yMinimum={.85} yMaximum={.95} yLabel="momentum β" selected={selected} />
      <p>Here the rise occupies T/4 updates, initial η=.2/10=.02 and final η=.02/100=.0002. These are our explicit PyTorch-compatible two-phase settings, not its defaults. At the selected update, β={formatValue(current.momentum)}. Changing β changes the optimizer state transition as well as the rate.</p>
    </> : kind === 'restart' ? <p>Each cycle uses positions 0…5 with progress position/6. It restarts before sampling progress 1, so the minimum parameter .01 is a bound rather than a used endpoint. A jump in rate does not reset the parameter or momentum.</p> : kind === 'step' ? <p>This policy halves .2 every six updates. It has no prescribed .01 endpoint. Period boundaries refer to update indices, not automatically to epochs.</p> : kind === 'constant' ? <p>A constant schedule is still a declared policy. It can work well on a smooth deterministic problem with an appropriate step; no theorem requires every training run to decay.</p> : <p>{kind === 'cosine' ? 'Warmup reaches .2 on its last update. The following update begins the decay without repeating that peak. Zero warmup starts the cosine at .2.' : 'This decay includes both .2 and .01 among the used values.'} The final used value is .01 under this finite-budget convention.</p>}
    <button onClick={reset}>Reset schedule</button>
    <p className="schedule-transfer">Transfer: compare 12 and 48 updates. The endpoints agree, but the number of updates at intermediate rates and the total displacement need not.</p>
  </section>;
}
export function ScheduleNoiseLab() {
  const [kind, setKind] = useState('cosine');
  const [curvature, setCurvature] = useState(2);
  const [noise, setNoise] = useState(1);
  const [selected, setSelected] = useState(12);
  const scheduled = useMemo(() => buildRateSchedule({
    kind,
    total: 24,
    warmup: 4
  }), [kind]);
  const constant = useMemo(() => buildRateSchedule({
    kind: 'constant',
    total: 24
  }), []);
  const currentStates = useMemo(() => scheduleNoiseMoments(scheduled, {
    curvature,
    noise
  }), [scheduled, curvature, noise]);
  const constantStates = useMemo(() => scheduleNoiseMoments(constant, {
    curvature,
    noise
  }), [constant, curvature, noise]);
  const current = currentStates[selected];
  const reset = () => {
    setKind('cosine');
    setCurvature(2);
    setNoise(1);
    setSelected(12);
  };
  const lossLog = values => values.map(state => [state.completed, Math.log10(Math.max(1e-12, state.meanSquaredError))]);
  const bothLogs = [...lossLog(currentStates), ...lossLog(constantStates)];
  const lower = Math.floor(Math.min(...bothLogs.map(([, value]) => value)));
  const upper = Math.max(lower + 1, Math.ceil(Math.max(...bothLogs.map(([, value]) => value))));
  return <section className="schedule-lab" data-lab="schedule-noise" aria-label="Investigate exact noise moments">
    <h3>A smaller rate changes both contraction and injected noise</h3>
    <p>Predict which policy has smaller expected squared error after 24 updates. Then set noise to zero. Both start at error 3 on the same fixed quadratic; the gold policy and blue constant policy share peak η=.2.</p>
    <div className="schedule-controls">
      <ScheduleSelect label="Compared policy" value={kind} options={kindOptions.filter(([key]) => ['cosine', 'linear', 'one-cycle'].includes(key))} onChange={setKind} />
      <ScheduleRange label="Curvature h" value={curvature} minimum={.5} maximum={4} step={.5} onChange={setCurvature} />
      <ScheduleRange label="Noise standard deviation σ" value={noise} minimum={0} maximum={2} step={.25} onChange={setNoise} />
    </div>
    <SchedulePlot title="Exact expected squared error under independent noise" description="Gold solid is the selected policy, blue dashed is constant .2. Vertical axis is log10 expected squared error; numerical readouts preserve the actual values. Values below1e-12 are clipped for plotting only." series={[{
      key: 'chosen',
      values: lossLog(currentStates),
      color: '#f0c96d'
    }, {
      key: 'constant',
      values: lossLog(constantStates),
      color: '#8dc7e8',
      dashed: true
    }]} xMaximum={24} yMinimum={lower} yMaximum={upper} selected={selected} xLabel="completed updates" yLabel="log₁₀ expected squared error" />
    <p className="schedule-caption">Gold solid: selected policy · blue dashed: constant .2. A vertical value −2 means 10⁻²=.01. A drop by 1 means ten times less expected squared error. Plot values below10⁻¹² are clipped; the readout is not.</p>
    <ScheduleRange label="Inspect completed updates" value={selected} minimum={0} maximum={24} onChange={setSelected} />
    <ScheduleReadout values={[["Selected E[e²]", formatValue(current.meanSquaredError)], ['Constant E[e²]', formatValue(constantStates[selected].meanSquaredError)], ['Squared mean error', formatValue(current.mean ** 2)], ['Error variance', formatValue(current.variance)], ['Last used rate', formatValue(current.rate ?? null)], ['Constant limiting variance', formatValue(.2 * noise * noise / (curvature * (2 - .2 * curvature)))]]} />
    <p>These are calculated moments of fresh independent zero-mean gradient noise, not measured neural-network losses. Only plain SGD is modeled here: selecting the one-cycle rate shape does not also apply its momentum curve. The constant-rate limit is an infinite-time statement for 0&lt;ηh&lt;2; finite-time values can differ.</p>
    <button onClick={reset}>Reset noise comparison</button>
    <p className="schedule-transfer">Transfer: explain why decreasing the final learning rate cannot undo noise already present in the parameter in a single step.</p>
  </section>;
}
export function ScheduleClockLab() {
  const [accumulation, setAccumulation] = useState(2);
  const [skipSecond, setSkipSecond] = useState(true);
  const [policy, setPolicy] = useState('committed');
  const [position, setPosition] = useState(0);
  const trace = useMemo(() => buildScheduleClockTrace({
    accumulation,
    skipSecond,
    policy
  }), [accumulation, skipSecond, policy]);
  const current = trace.states[position];
  const reset = () => {
    setAccumulation(2);
    setSkipSecond(true);
    setPolicy('committed');
    setPosition(0);
  };
  return <section className="schedule-lab" data-lab="schedule-clock" aria-label="Trace schedule event clocks">
    <h3>Watch work arrive before an update is committed</h3>
    <p>Twelve one-observation microbatches arrive. Accumulate their mean gradient in groups, optionally reject attempt two, and consume a rate only when the parameter is updated. Predict which counter should advance on the rejected attempt.</p>
    <div className="schedule-controls">
      <ScheduleSelect label="Microbatches per attempt" value={accumulation} options={[[1, '1'], [2, '2'], [3, '3']]} onChange={value => {
        setAccumulation(Number(value));
        setPosition(0);
      }} />
      <ScheduleSelect label="Schedule clock policy" value={policy} options={[["committed", 'Committed updates'], ['microbatch', 'Fault: microbatch index'], ['advance-first', 'Fault: advance before use']]} onChange={value => {
        setPolicy(value);
        setPosition(0);
      }} />
      <ScheduleField label="Reject attempt two">{id => <input type="checkbox" id={id} checked={skipSecond} onChange={event => {
          setSkipSecond(event.target.checked);
          setPosition(0);
        }} />}</ScheduleField>
    </div>
    <div className="schedule-event-lanes" role="group" aria-label="Microbatch and update lanes">
      <div className="schedule-lane-label">Incoming microbatches</div>
      <ol className="schedule-event-strip">{trace.states.slice(1).map(state => <li key={state.microbatches} className={state.microbatches === position ? 'is-selected' : state.microbatches < position ? 'is-past' : ''}><span>m{state.microbatches}</span><small>target {state.target}</small></li>)}</ol>
      <div className="schedule-lane-label">Attempt boundaries</div>
      <ol className="schedule-attempt-strip">{trace.states.slice(1).filter(state => state.microbatches % accumulation === 0).map(state => <li key={state.attempt} className={state.microbatches <= position ? 'is-past' : ''}><span>attempt {state.attempt}</span><small>after m{state.microbatches}</small><strong>{state.microbatches <= position ? state.action : 'pending'}</strong></li>)}</ol>
    </div>
    <ScheduleStepper position={position} maximum={12} onChange={setPosition} reset={reset} noun="microbatch" />
    <ScheduleReadout values={[["Microbatches processed", position], ['Attempts / committed', `${current.attempt} / ${current.committed}`], ['Current action', current.action], ['Consumed schedule index', formatValue(current.scheduleIndex)], ['Rate used', formatValue(current.rate)], ['Parameter θ', formatValue(current.parameter)], ['Pending gradient sum', formatValue(current.pendingGradient)], ['Applied mean gradient', formatValue(current.appliedGradient ?? null)]]} />
    <p>The intended budget is {trace.updates} committed updates, with first/last rates .2/.01. A skipped attempt clears this example's accumulated gradients and uses no rate. A wrong clock can exhaust the finite list early; the model reports that error instead of silently reusing its last value.</p>
    <details><summary>Inspect the actual consumed indices and rates</summary><div className="schedule-table-wrap" tabIndex={0} role="region" aria-label="Consumed schedule indices and rates, horizontally scrollable"><table><caption>Completed attempt boundaries through the selected microbatch</caption><thead><tr><th>Attempt</th><th>Action</th><th>Index</th><th>η</th><th>θ after</th></tr></thead><tbody>{trace.states.slice(1, position + 1).filter(state => state.microbatches % accumulation === 0).map(state => <tr key={state.attempt}><td>{state.attempt}</td><td>{state.action}</td><td>{formatValue(state.scheduleIndex)}</td><td>{formatValue(state.rate)}</td><td>{formatValue(state.parameter)}</td></tr>)}</tbody></table></div></details>
    <p className="schedule-transfer">Transfer: turn skipping off, then change accumulation from two to three. Explain why keeping the same epoch count does not preserve the number of updates or their trajectories.</p>
  </section>;
}
const defaultMetrics = '1 .9 .9 .89 .88 .88 .9 .87 .87 .87 .87 .87';
export function SchedulePlateauLab() {
  const [text, setText] = useState(defaultMetrics);
  const [metrics, setMetrics] = useState(() => parseValidationMetrics(defaultMetrics));
  const [patience, setPatience] = useState(1);
  const [threshold, setThreshold] = useState(.02);
  const [cooldown, setCooldown] = useState(1);
  const [position, setPosition] = useState(0);
  const [error, setError] = useState('');
  const trace = useMemo(() => buildPlateauTrace(metrics, {
    patience,
    threshold,
    cooldown
  }), [metrics, patience, threshold, cooldown]);
  const current = trace[position];
  const maximum = Math.max(...metrics, .1) * 1.1;
  const reset = () => {
    setText(defaultMetrics);
    setMetrics(parseValidationMetrics(defaultMetrics));
    setPatience(1);
    setThreshold(.02);
    setCooldown(1);
    setPosition(0);
    setError('');
  };
  const apply = () => {
    try {
      setMetrics(parseValidationMetrics(text));
      setPosition(0);
      setError('');
    } catch (failure) {
      setError(failure.message);
    }
  };
  return <section className="schedule-lab" data-lab="schedule-plateau" aria-label="Trace validation-triggered scheduling">
    <h3>A lower measurement may still fail the improvement test</h3>
    <p>Here lower validation loss is better. Predict why .89 after best .90 does not improve by more than .02. Inspect the strict cutoff and the number of tolerated bad observations before a reduction.</p>
    <ScheduleField label="Validation losses (apply to use edits)">{id => <textarea id={id} value={text} onChange={event => setText(event.target.value)} rows={3} maxLength={220} />}</ScheduleField>
    <button onClick={apply}>Apply validation losses</button>{error && <p role="alert">{error} The previous valid stream is retained.</p>}
    <div className="schedule-controls">
      <ScheduleRange label="Patience" value={patience} minimum={0} maximum={3} onChange={value => {
        setPatience(value);
        setPosition(0);
      }} />
      <ScheduleRange label="Absolute threshold δ" value={threshold} minimum={0} maximum={.1} step={.01} onChange={value => {
        setThreshold(value);
        setPosition(0);
      }} />
      <ScheduleRange label="Cooldown observations" value={cooldown} minimum={0} maximum={3} onChange={value => {
        setCooldown(value);
        setPosition(0);
      }} />
    </div>
    <SchedulePlot title="Validation observations and previous-best cutoff" description="Dots are the full entered metric stream, including future observations shown for prediction. The dashed line is the current strict improvement cutoff; only observations through the cursor have affected state." series={[{
      key: 'metrics',
      values: metrics.map((metric, index) => [index, metric]),
      color: '#8dc7e8',
      dots: true
    }, ...(current.cutoff === null ? [] : [{
      key: 'cutoff',
      values: [[0, current.cutoff], [metrics.length - 1, current.cutoff]],
      color: '#f0c96d',
      dashed: true
    }])]} xMaximum={Math.max(1, metrics.length - 1)} yMaximum={maximum} yMinimum={Math.min(0, current.cutoff ?? 0)} selected={position} xLabel="validation observation" yLabel="validation loss" />
    <p className="schedule-caption">Blue dots: invented observed losses, with future entries previewed. Gold dashed: current cutoff before this observation. Improvement requires a value strictly below it; equality does not qualify.</p>
    <ScheduleStepper position={position} maximum={metrics.length - 1} onChange={setPosition} reset={reset} noun="observation" />
    <ScheduleReadout values={[["Observation / metric", `${position} / ${formatValue(current.metric)}`], ['Previous best / cutoff', `${formatValue(current.bestBefore)} / ${formatValue(current.cutoff)}`], ['Significant improvement', current.improved ? 'yes' : 'no'], ['Bad count after handling', current.bad], ['Cooldown remaining', current.cooling], ['Rate before / next', `${formatValue(current.rateBefore)} / ${formatValue(current.rate)}`], ['Reduction event', current.reduced ? 'rate halved' : current.triggered ? 'triggered, but floor prevented change' : 'no reduction'], ['Best after observation', formatValue(current.best)]]} />
    <p>The first observation establishes the baseline. A reduction occurs when bad count exceeds patience; cooldown clears bad counts while still allowing a new best. The rate halves down to .025. This matches the stated PyTorch2.14 absolute-threshold configuration; validation frequency determines this clock's meaning.</p>
    <p className="schedule-transfer">Transfer: set threshold to zero and repeat an identical best value. Then set patience to zero. A flat metric is not a significant improvement under this strict predicate.</p>
  </section>;
}
export function UpdateClockFigure() {
  return <figure className="schedule-inline"><figcaption>One committed update has a before and an after</figcaption><ol className="schedule-flow"><li><strong>Read ηᵤ</strong><span>Log the rate about to be used.</span></li><li><strong>Compute the update</strong><span>Gradient and optimizer state determine the displacement.</span></li><li><strong>Commit θᵤ₊₁</strong><span>Count actual work under the chosen policy.</span></li><li><strong>Prepare ηᵤ₊₁</strong><span>This value belongs to a later update.</span></li></ol><p>A graph of prepared rates can be shifted by one from the rates that trained the model. At the end of a finite budget, a prepared value may never be used.</p></figure>;
}
export function ResumeStateFigure() {
  return <figure className="schedule-inline"><figcaption>Resume the state that determines the next transition</figcaption><div className="schedule-resume-map"><div><strong>Model</strong><p>Parameters and buffers</p></div><div><strong>Optimizer</strong><p>Momentum/moments and parameter groups</p></div><div><strong>Policy</strong><p>Schedule position, phase, best metric and counters where applicable</p></div><div><strong>Data and randomness</strong><p>Sampler position, random generators and completed work</p></div></div><p>These feed the next update together. This lesson's complete CPU example saves a fixed dataset's generator and an SGD/cosine state; a production loader, distributed workers or a scaler can require additional state.</p></figure>;
}
