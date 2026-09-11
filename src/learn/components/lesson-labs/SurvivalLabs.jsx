import { useId, useState } from 'react';
import { coxRiskSets, coxTimes, coxEvents, coxFeatures, observedPumps, kaplanMeier, pumpTimes, pumpEvents, restrictedMean, survivalAt, weibullClock, proportionalComparison, concordancePairs, censorWeightedBrier, competingIncidence, constantCompeting } from '../../data/survival-models.js';
import './survival-labs.css';
const gold = '#e7b94a';
const cyan = '#64c9d9';
const coral = '#ed9387';
const format = (value, digits = 4) => value === null ? 'not defined' : Number.isFinite(value) ? Number(value.toFixed(digits)).toString() : 'not finite';
function Control({
  label,
  value,
  onChange,
  min,
  max,
  step = 1
}) {
  const id = useId();
  return <label className="sv-control" htmlFor={id}><span>{label} <output>{format(value)}</output></span><input id={id} type="range" min={min} max={max} step={step} value={value} onChange={event => onChange(Number(event.target.value))} /></label>;
}
function Select({
  label,
  value,
  onChange,
  children
}) {
  return <label className="sv-select"><span>{label}</span><select aria-label={label} value={value} onChange={event => onChange(event.target.value)}>{children}</select></label>;
}
function Table({
  caption,
  headers,
  rows
}) {
  return <div className="sv-table-scroll" tabIndex={0} role="region" aria-label={caption}><table><caption>{caption}</caption><thead><tr>{headers.map(header => <th key={header} scope="col">{header}</th>)}</tr></thead><tbody>{rows.map((row, index) => <tr key={index}>{row.map((cell, column) => column === 0 ? <th key={column} scope="row">{cell}</th> : <td key={column}>{cell}</td>)}</tr>)}</tbody></table></div>;
}
function Lab({
  title,
  prediction,
  children,
  onReset
}) {
  return <section className="sv-lab" aria-label={title}><div className="sv-lab-heading"><h3>{title}</h3><button type="button" onClick={onReset}>Reset</button></div><p className="sv-prompt"><strong>Predict:</strong> {prediction}</p>{children}</section>;
}
function Plot({
  title,
  series,
  xMax = 10,
  yMax = 1,
  xLabel = 'days',
  yLabel = 'probability',
  cursor = null,
  rectangles = [],
  markers = []
}) {
  const width = 340;
  const left = 48,
    right = 326,
    top = 28,
    bottom = 176;
  const x = time => left + time / xMax * (right - left);
  const y = value => bottom - value / yMax * (bottom - top);
  return <figure className="sv-plot"><figcaption>{title}</figcaption><svg viewBox={`0 0 ${width} 222`} role="img" aria-label={`${title}. Horizontal axis ${xLabel}; vertical axis ${yLabel}.`}>
    <text x={left} y={16} className="sv-axis-label">{yLabel}</text>
    {[0, 0.5, 1].map(fraction => <g key={fraction}><line x1={left} x2={right} y1={y(fraction * yMax)} y2={y(fraction * yMax)} className="sv-grid" /><text x={left - 7} y={y(fraction * yMax) + 5} textAnchor="end">{format(fraction * yMax, 2)}</text></g>)}
    {rectangles.map((rectangle, index) => <rect key={index} x={x(rectangle.left)} y={y(rectangle.height)} width={Math.max(0, x(rectangle.right) - x(rectangle.left))} height={bottom - y(rectangle.height)} fill={gold} opacity="0.18" />)}
    {series.map(({
        label,
        color = gold,
        points
      }) => <path key={label} d={points.filter(point => point[1] !== null && Number.isFinite(point[1])).map(([time, value], index) => `${index ? 'L' : 'M'}${x(time)},${y(value)}`).join(' ')} stroke={color} strokeWidth="2.5" fill="none" />)}
    {markers.map(([time, value], index) => <path key={index} d={`M${x(time) - 4},${y(value)}h8M${x(time)},${y(value) - 4}v8`} stroke={gold} strokeWidth="2" />)}
    {cursor !== null && <line x1={x(cursor)} x2={x(cursor)} y1={top} y2={bottom} stroke="var(--text-secondary)" strokeDasharray="4 4" />}
    {[0, 0.5, 1].map(fraction => <text key={fraction} x={x(fraction * xMax)} y={198} textAnchor={fraction === 0 ? 'start' : fraction === 1 ? 'end' : 'middle'}>{format(fraction * xMax, 1)}</text>)}
    <text x={(left + right) / 2} y={219} textAnchor="middle">{xLabel}</text>
  </svg><div className="sv-legend">{series.map(({
        label,
        color = gold
      }) => <span key={label}><i style={{
          background: color
        }} />{label}</span>)}</div></figure>;
}
function stepPoints(table, end = table.lastTime) {
  let previous = 1;
  const points = [[0, 1]];
  for (const row of table.rows) {
    if (row.time > end) break;
    points.push([row.time, previous], [row.time, row.survival]);
    previous = row.survival;
  }
  points.push([end, previous]);
  return points;
}
export function KaplanMeierLab() {
  const [cutoff, setCutoff] = useState(9);
  const [toggled, setToggled] = useState('none');
  const [cursor, setCursor] = useState(4);
  const [allCensored, setAllCensored] = useState(false);
  const {
    times,
    events,
    table
  } = observedPumps({
    cutoff,
    toggled: toggled === 'none' ? null : Number(toggled),
    allCensored
  });
  const at = Math.min(cursor, cutoff);
  const row = table.rows.find(row => row.time === at);
  const risk = times.filter(time => time >= at).length;
  return <Lab title="Count the pumps still at risk" prediction="At day 4, both a failure and a censor mark appear. Will both pumps count just before that day's event?" onReset={() => {
    setCutoff(9);
    setToggled('none');
    setCursor(4);
    setAllCensored(false);
  }}>
    <div className="sv-controls"><Control label="Inspect day" value={at} onChange={setCursor} min={1} max={cutoff} /><Control label="Administrative follow-up ends" value={cutoff} onChange={setCutoff} min={1} max={9} /><Select label="Reverse one record's event status" value={toggled} onChange={setToggled}><option value="none">Keep original records</option>{times.map((_, index) => <option key={index} value={index}>Pump {String.fromCharCode(65 + index)}</option>)}</Select><label className="sv-check"><input type="checkbox" checked={allCensored} onChange={event => setAllCensored(event.target.checked)} />Make every observed endpoint censored</label></div>
    <div className="sv-lanes" aria-label="Pump observation lanes"><div className="sv-lane-axis"><span>0</span><span>Observed days · 9</span></div>{times.map((time, index) => <div className={`sv-lane ${time >= at ? 'is-risk' : ''}`} key={index}><strong>{String.fromCharCode(65 + index)}</strong><div><span className="sv-lane-time" style={{
            width: `${time / 9 * 100}%`
          }} /><span className={`sv-end ${events[index] ? 'failure' : 'censor'}`} style={{
            left: `${time / 9 * 100}%`
          }}>{events[index] ? '●' : '│'}</span><span className="sv-lane-cursor" style={{
            left: `${at / 9 * 100}%`
          }} /></div><small>{time} {events[index] ? 'fail' : 'cens.'}</small></div>)}</div>
    <p className="sv-caption">Bright lanes reach the cursor and enter its risk set. ● observed failure; │ right censor. Every lane starts at commissioning. Reversing a status is a hypothetical change to the data, not an imputation of the unknown time.</p>
    <Plot title="A failure makes a vertical step; censoring changes later denominators" series={[{
      label: 'KM S(t)',
      points: stepPoints(table)
    }]} xMax={9} cursor={at} markers={times.flatMap((time, index) => !events[index] ? [[time, survivalAt(table, time)]] : [])} />
    <p className="sv-readout" aria-live="polite">Day {at}: <strong>{risk} at risk</strong>; {row?.failures ?? 0} failures; {row?.censored ?? 0} censored. S({at}) = {format(survivalAt(table, at), 6)}.{row?.failures > 0 && <> Multiply {format(row.previous, 6)} by ({row.risk} − {row.failures})/{row.risk}.</>}</p>
    <Table caption="Observed-time risk accounting" headers={['day', 'risk', 'fail', 'cens.', 'S(t)']} rows={table.rows.map(row => [row.time, row.risk, row.failures, row.censored, format(row.survival, 6)])} />
    <p className="sv-caption"><strong>Explain:</strong> Why does censoring pump D at day 4 preserve the current event denominator but remove D from day 6? Now move follow-up to day 5 and identify the claims about day 8 that are no longer supported.</p>
  </Lab>;
}
export function RestrictedSurvivalLab() {
  const [horizon, setHorizon] = useState(9);
  const [scenario, setScenario] = useState('original');
  const events = pumpEvents.map(event => scenario === 'none' ? false : scenario === 'all' ? true : event);
  const table = kaplanMeier(pumpTimes, events);
  const area = restrictedMean(table, horizon);
  const selected = table.rows.filter(row => row.time <= horizon).at(-1);
  return <Lab title="Read area, uncertainty and the end of follow-up" prediction="If a curve stays above one half, is its median zero, the last day, or not reached?" onReset={() => {
    setHorizon(9);
    setScenario('original');
  }}>
    <div className="sv-controls"><Control label="Restricted horizon in days" value={horizon} onChange={setHorizon} min={1} max={9} step={0.5} /><Select label="Observation scenario" value={scenario} onChange={setScenario}><option value="original">Original eight pumps</option><option value="none">All endpoints censored</option><option value="all">All endpoints are failures</option></Select></div>
    <Plot title="Area is expected time used up to the chosen limit" series={[{
      label: 'KM survival',
      points: stepPoints(table)
    }]} rectangles={area.rectangles} xMax={9} cursor={horizon} />
    <div className="sv-metrics" aria-live="polite"><p>RMST({horizon})<strong>{format(area.value, 6)} days</strong></p><p>Sample KM median<strong>{table.median ?? 'not reached'}</strong></p><p>S({horizon})<strong>{format(survivalAt(table, horizon), 6)}</strong></p></div>
    <p className="sv-readout">{selected?.interval ? <>Approximate <strong>pointwise</strong> 95% log-log interval: [{format(selected.interval[0], 6)}, {format(selected.interval[1], 6)}]. The interval uses the risk-table row at day {selected.time}; censor-only rows retain the preceding event's estimate.</> : <>The interior Greenwood/log-log interval is not available here. A boundary curve is not a certificate of population certainty.</>}</p>
    <Table caption="Rectangles that determine the restricted mean" headers={['interval', 'height', 'area (days)']} rows={area.rectangles.map(rectangle => [`${rectangle.left}–${rectangle.right}`, format(rectangle.height, 6), format(rectangle.area, 6)])} />
    <p className="sv-caption">Exact finite-sample KM geometry, with an asymptotic interval formula. The model stops at day 9. <strong>Transfer:</strong> extend the hidden lifetimes after day 9 to 10 or 1000; the same recorded observations permit both tails.</p>
  </Lab>;
}
export function HazardClockLab() {
  const [shape, setShape] = useState(2);
  const [age, setAge] = useState(4);
  const [interval, setInterval] = useState(3);
  const [multiplier, setMultiplier] = useState(1);
  const [mode, setMode] = useState('time');
  const state = weibullClock({
    shape,
    age,
    interval,
    multiplier,
    mode
  });
  const baseline = weibullClock({
    shape,
    age,
    interval
  });
  const maximumHazard = Math.max(...state.curve.map(point => point.hazard ?? 0), ...baseline.curve.map(point => point.hazard ?? 0));
  return <Lab title="Separate a hazard multiplier from a slower clock" prediction="With shape 2, doubling the time scale will do more than halve the instantaneous hazard. Predict the factor." onReset={() => {
    setShape(2);
    setAge(4);
    setInterval(3);
    setMultiplier(1);
    setMode('time');
  }}>
    <div className="sv-controls"><Select label="What the multiplier changes" value={mode} onChange={setMode}><option value="time">AFT: multiply time scale</option><option value="hazard">PH: multiply hazard</option></Select><Control label="Multiplier" value={multiplier} onChange={setMultiplier} min={0.25} max={3} step={0.25} /><Control label="Weibull shape" value={shape} onChange={setShape} min={0.5} max={3} step={0.5} /><Control label="Already running for days" value={age} onChange={setAge} min={0} max={12} /><Control label="Next interval in days" value={interval} onChange={setInterval} min={1} max={10} /></div>
    <div className="sv-chart-pair"><Plot title="Survival: a probability" xMax={30} cursor={age} series={[{
        label: 'scale 12 baseline',
        color: cyan,
        points: baseline.curve.map(point => [point.time, point.survival])
      }, {
        label: 'changed clock',
        points: state.curve.map(point => [point.time, point.survival])
      }]} /><Plot title="Hazard: an instantaneous rate" xMax={30} yMax={maximumHazard} cursor={age} yLabel="rate per day" series={[{
        label: 'baseline hazard',
        color: cyan,
        points: baseline.curve.map(point => [point.time, point.hazard])
      }, {
        label: 'changed hazard',
        points: state.curve.map(point => [point.time, point.hazard])
      }]} /></div>
    <div className="sv-metrics" aria-live="polite"><p>Hazard ratio<strong>{format(state.hazardRatio)}</strong></p><p>Time ratio<strong>{format(state.timeRatio)}</strong></p><p>Median time<strong>{format(state.median)} days</strong></p></div>
    <p className="sv-readout">At age {age}, h = {state.currentHazard === null ? 'unbounded as age approaches zero' : `${format(state.currentHazard)} per day`}. Integrated hazard over the next {interval} days = {format(state.integratedHazard)}. Conditional failure probability = <strong>{format(state.conditionalFailure)}</strong>.</p>
    <p className="sv-caption">Analytic Weibull laws, sampled for plotting; selected values are calculated directly. Shape below 1 has no finite hazard at 0: the rate plot begins at its first positive sample. Its vertical range changes with the chosen law. <strong>Explain:</strong> why is the displayed interval probability 1 − exp(−integrated hazard), rather than the hazard itself?</p>
  </Lab>;
}
export function CoxRiskLab() {
  const [beta, setBeta] = useState(Math.log(2));
  const [selected, setSelected] = useState('2');
  const [ties, setTies] = useState('efron');
  const state = coxRiskSets({
    beta,
    ties
  });
  const row = state.rows.find(row => row.time === Number(selected));
  return <Lab title="Let the event compete inside its risk set" prediction="Increasing beta gives more weight to x=1. Does that help every observed event in these six records?" onReset={() => {
    setBeta(Math.log(2));
    setSelected('2');
    setTies('efron');
  }}>
    <div className="sv-controls"><Control label="Cox coefficient beta" value={beta} onChange={setBeta} min={-2} max={2} step="any" /><button type="button" onClick={() => setBeta(Math.log(2))}>Use beta = log(2)</button><Select label="Event-time risk set" value={selected} onChange={setSelected}>{state.rows.map(row => <option key={row.time} value={row.time}>Day {row.time}{row.eventIndices.length > 1 ? ' · tied failures' : ''}</option>)}</Select><Select label="Tie denominator" value={ties} onChange={setTies}><option value="efron">Efron</option><option value="breslow">Breslow</option></Select></div>
    <div className="sv-risk-weights">{row.weights.map(weight => <div key={weight.index} className={weight.isEvent ? 'is-event' : ''}><span><strong>{String.fromCharCode(65 + weight.index)}</strong> x={weight.feature} {weight.isEvent ? '● event' : 'at risk'}</span><div className="sv-weight-track"><span style={{
            width: `${100 * weight.probability}%`
          }} /></div><output>{format(weight.probability, 4)}</output></div>)}</div>
    <p className="sv-caption">Bar length is each subject's normalized exp(beta × x) weight before the event. Together the bars sum to 1. At a tie, these initial weights are followed by the chosen denominator adjustment; they are not independent binary labels.</p>
    <Table caption="Every record and its earlier risk-set influence" headers={['pump', 'x', 'endpoint', 'status']} rows={coxTimes.map((time, index) => [String.fromCharCode(65 + index), coxFeatures[index], time, coxEvents[index] ? 'failure' : 'censored'])} />
    <Table caption="Denominators for this event time" headers={['term', 'removed fraction', 'denominator', 'weighted mean x']} rows={row.denominators.map(term => [term.step + 1, format(term.fraction), format(term.denominator, 6), format(term.mean, 6)])} />
    <p className="sv-readout" aria-live="polite">Day {selected}: contribution {format(row.contribution, 8)}; log contribution {format(row.logContribution, 6)}; score contribution {format(row.score, 6)}. Whole-data log likelihood {format(state.logLikelihood, 6)}; score {format(state.score, 6)}; observed information {format(state.information, 6)}.</p>
    <p className="sv-caption">Exact evaluation of the stated finite Cox partial likelihood. An untied contribution is a normalized local event probability; a tied Efron/Breslow factor is a specified approximation and should not be compared as though both were exact unordered-set probabilities. <strong>Transfer:</strong> explain why censored pump D changes day 2's denominator but not day 4's.</p>
  </Lab>;
}
export function ProportionalHazardsLab() {
  const [time, setTime] = useState(6);
  const [mode, setMode] = useState('switch');
  const state = proportionalComparison({
    time,
    mode
  });
  return <Lab title="A hazard change and a survival crossing happen at different times" prediction="A new hazard becomes worse at day 4. Must its accumulated survival already be worse at day 4?" onReset={() => {
    setTime(6);
    setMode('switch');
  }}>
    <div className="sv-controls"><Control label="Inspect elapsed days" value={time} onChange={setTime} min={0} max={15} step={0.5} /><Select label="Constructed population" value={mode} onChange={setMode}><option value="switch">Hazard switches at day 4</option><option value="mixture">Two hidden types, conditional HR0.5</option></Select></div>
    <div className="sv-chart-pair"><Plot title="Instantaneous rates" xMax={15} yMax={0.3} yLabel="rate per day" cursor={time} series={[{
        label: 'reference',
        color: cyan,
        points: state.curve.map(point => [point.time, point.referenceHazard])
      }, {
        label: 'comparison',
        points: state.curve.map(point => [point.time, point.comparisonHazard])
      }]} /><Plot title="Accumulated survival" xMax={15} cursor={time} series={[{
        label: 'reference',
        color: cyan,
        points: state.curve.map(point => [point.time, point.referenceSurvival])
      }, {
        label: 'comparison',
        points: state.curve.map(point => [point.time, point.comparisonSurvival])
      }]} /></div>
    <p className="sv-readout" aria-live="polite">Day {time}: reference S={format(state.selected.referenceSurvival, 6)}, comparison S={format(state.selected.comparisonSurvival, 6)}. Current hazard ratio = <strong>{format(state.selected.ratio, 6)}</strong>.</p>
    <p className="sv-caption">{mode === 'switch' ? 'Exact piecewise hazards: reference 0.1/day, comparison 0.05/day before 4 and 0.2/day afterward. They accumulate the same hazard at 6; that is the survival crossing. The rate jump uses a vertical segment.' : 'Each group begins with equal numbers of 0.1/day and 0.4/day types. Comparison multiplies both type-specific hazards by 0.5. The survivors acquire different mixtures, so the pooled hazard ratio changes over time. This is not a causal treatment calculation.'} Curves are calculated from these laws, not fitted residual diagnostics. <strong>Explain:</strong> which condition is being checked by each view?</p>
  </Lab>;
}
export function ConcordanceLab() {
  const [ordering, setOrdering] = useState('original');
  const times = [1, 2, 2, 4],
    events = [true, false, true, true];
  const scores = {
    original: [3, 2, 2, 0],
    reversed: [-3, -2, -2, 0],
    tied: [1, 1, 1, 1],
    transformed: [30, 20, 20, 0]
  }[ordering];
  const state = concordancePairs(times, events, scores);
  return <Lab title="Rank only the pairs whose ordering can be compared" prediction="B is censored at 2 and C fails at 2. Under the declared library convention, is their pair included?" onReset={() => setOrdering('original')}>
    <Select label="Risk-score ordering" value={ordering} onChange={setOrdering}><option value="original">Original: [3,2,2,0]</option><option value="transformed">Same ranking: [30,20,20,0]</option><option value="reversed">Reverse the scores</option><option value="tied">Tie every score</option></Select>
    <div className="sv-pair-board"><div /><strong>A</strong><strong>B</strong><strong>C</strong><strong>D</strong>{times.flatMap((_, first) => [<strong key={`label-${first}`}>{String.fromCharCode(65 + first)}</strong>, ...times.map((_, second) => {
        const pair = state.pairs.find(pair => pair.first === first && pair.second === second);
        const text = !pair ? '—' : pair.status === 'concordant' ? '1' : pair.status === 'discordant' ? '0' : pair.status === 'risk tie' ? '½' : '×';
        return <span key={`${first}-${second}`} className={pair ? `sv-pair-${pair.status.replaceAll(' ', '-')}` : ''} title={pair?.status ?? 'Pair is shown in the upper triangle'}>{text}</span>;
      })])}</div>
    <p className="sv-caption">1 concordant; 0 discordant; ½ tied risk;× not comparable;— duplicate/self pair. A high risk score predicts an earlier event. Event/censor ties enter here; event/event ties do not.</p>
    <Table caption="Four records used by the pair board" headers={['record', 'Y', 'event?', 'risk']} rows={times.map((time, index) => [String.fromCharCode(65 + index), time, events[index] ? 'yes' : 'no', scores[index]])} />
    <p className="sv-readout" aria-live="polite">({state.concordant} concordant + ½ × {state.tied} risk ties) / {state.comparable} comparable pairs = <strong>{format(state.value)}</strong>.</p>
    <p className="sv-caption">This matches the declared scikit-survival convention, with risk ties within 1e−8. Censoring changes which pairs are observed. <strong>Explain:</strong> why can reversing the score produce C below 0.5, and why do scores 3 and 30 say nothing about a probability of 0.3 or0.03?</p>
  </Lab>;
}
export function CensorWeightLab() {
  const [prediction, setPrediction] = useState(0.6);
  const [late, setLate] = useState(0.5);
  const state = censorWeightedBrier({
    prediction,
    lateCensorProbability: late
  });
  return <Lab title="Recover a horizon loss from observable outcomes" prediction="If half of the day 8 failures disappear from observation at day 3, what weight should their observed counterparts receive?" onReset={() => {
    setPrediction(0.6);
    setLate(0.5);
  }}>
    <div className="sv-controls"><Control label="Predicted survival at day 5" value={prediction} onChange={setPrediction} min={0} max={1} step={0.05} /><Control label="Chance follow-up lasts to day 10" value={late} onChange={setLate} min={0.1} max={1} step={0.1} /></div>
    <div className="sv-observation-flow"><div><strong>Lifetime law</strong><span>40% fail at 2</span><span>60% fail at 8</span></div><span aria-hidden="true">×</span><div><strong>Independent follow-up</strong><span>{format(100 * (1 - late))}% ends at 3</span><span>{format(100 * late)}% ends at 10</span></div></div>
    <Table caption="Population branches and known horizon outcomes" headers={['T,C', 'mass', 'day 5 outcome', 'weight', 'weighted loss']} rows={state.rows.map(row => [`${row.lifetime},${row.followup}`, format(row.probability), row.known ? row.lifetime > 5 ? 'running' : 'failed' : 'unknown', format(row.weight), format(row.contribution)])} />
    <div className="sv-metrics" aria-live="polite"><p>Full population Brier<strong>{format(state.full, 6)}</strong></p><p>Known-G weighted expectation<strong>{format(state.weighted, 6)}</strong></p><p>Complete-case average<strong>{format(state.completeCases, 6)}</strong></p><p>Censor as failure<strong>{format(state.censoredAsFailure, 6)}</strong></p></div>
    <p className="sv-caption">Exact enumeration of the specified independent finite laws, not a finite fitted estimate. The weighted expectation multiplies each row by its branch mass. A censored-before 5 row contributes 0 to this weighted sum; that does <em>not</em> assert that its unknown outcome had zero loss. Positive observation probability is required. <strong>Transfer:</strong> explain why the full and weighted expectations stay equal as follow-up changes, but the weight and sampling variability need not stay small.</p>
  </Lab>;
}
export function CompetingRiskLab() {
  const [step, setStep] = useState(5);
  const [secondRate, setSecondRate] = useState(0.3);
  const state = competingIncidence();
  const row = state.rows[step];
  const law = constantCompeting({
    secondRate
  });
  return <Lab title="Send first-event mass out of one common surviving pool" prediction="If another failure cause becomes more frequent, can cause1 become less probable while its own hazard stays fixed?" onReset={() => {
    setStep(5);
    setSecondRate(0.3);
  }}>
    <div className="sv-controls"><Control label="Observe through day" value={step + 1} onChange={day => setStep(day - 1)} min={1} max={6} /><Control label="Analytic competing hazard per day" value={secondRate} onChange={setSecondRate} min={0} max={0.6} step={0.05} /></div>
    <div className="sv-mass-strip" aria-label={`Day${row.time}: event-free${format(row.survival)}, cause1${format(row.first)}, cause2${format(row.second)}`}><span style={{
        width: `${100 * row.survival}%`,
        background: cyan
      }} /><span style={{
        width: `${100 * row.first}%`,
        background: gold
      }} /><span style={{
        width: `${100 * row.second}%`,
        background: coral
      }} /></div>
    <div className="sv-legend"><span><i style={{
          background: cyan
        }} />event-free {format(row.survival, 6)}</span><span><i style={{
          background: gold
        }} />cause 1 {format(row.first, 6)}</span><span><i style={{
          background: coral
        }} />cause 2 {format(row.second, 6)}</span></div>
    <p className="sv-readout" aria-live="polite">Day {row.time}, risk set {row.risk}: move {format(row.previous, 6)} × {row.firstEvents}/{row.risk} = {format(row.firstIncrement, 6)} into cause 1 and {format(row.previous, 6)} × {row.secondEvents}/{row.risk} = {format(row.secondIncrement, 6)} into cause 2. Total mass = {format(row.survival + row.first + row.second)}.</p>
    <Table caption="Exact finite-cohort cumulative incidence" headers={['day', 'risk', 'cause1', 'cause2', 'event-free']} rows={state.rows.slice(0, step + 1).map(row => [row.time, row.risk, format(row.first, 6), format(row.second, 6), format(row.survival, 6)])} />
    <div className="sv-analytic-comparison"><h4>Separate analytic experiment: fixed cause 1 hazard 0.1/day, horizon 5</h4><p>Actual cause 1 incidence: <strong>{format(law.first, 6)}</strong><br />Cause 2 incidence: {format(law.second, 6)}<br />Still event-free: {format(law.survival, 6)}<br />Net one-cause quantity 1 − exp(−0.1 ×5): {format(law.netFirst, 6)}</p></div>
    <p className="sv-caption">The upper strip uses six observed records. The lower calculation uses a different, explicitly constant-hazard population; its slider does not change the observed cohort. Cause-specific fitting can treat another cause as leaving the risk set, while actual cumulative incidence must use the shared event-free survival. <strong>Explain:</strong> which question would the net quantity answer only under additional structural assumptions about removing the competing cause?</p>
  </Lab>;
}
