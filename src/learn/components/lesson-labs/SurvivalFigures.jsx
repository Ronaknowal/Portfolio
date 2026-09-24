import { useId } from 'react';
import { kaplanMeier, logrankTable } from '../../data/survival-models.js';
import './survival-figures.css';

const fixed = (value, digits = 6) => value.toFixed(digits);
const treeTimes = [8, 9, 7, 6, 4, 3, 2, 5];
const treeEvents = [true, false, true, false, true, true, true, false];
const treeGroups = treeTimes.map((_, index) => Number(index > 3));
const splitScore = logrankTable(treeTimes, treeEvents, treeGroups);
const leaves = [0, 1].map(group => {
  const indices = treeGroups.flatMap((value, index) => value === group ? [index] : []);
  return {
    group,
    indices,
    table: kaplanMeier(indices.map(index => treeTimes[index]), indices.map(index => treeEvents[index])),
  };
});

function ClockAxis({ end, ticks }) {
  return <div className="survival-clock-axis" aria-hidden="true">
    {ticks.map(time => <span key={time} className={time === 0 ? 'at-start' : time === end ? 'at-end' : ''} style={{ left: `${100 * time / end}%` }}>{time}</span>)}
  </div>;
}

function Timeline({ start = 0, stop, end, kind, tone = 'gold', unknown = false, children }) {
  return <div className={`survival-timeline survival-tone-${tone}`} aria-hidden="true">
    <span className="survival-time-track" />
    <span className="survival-time-observed" style={{ left: `${100 * start / end}%`, width: `${100 * (stop - start) / end}%` }} />
    {unknown && <span className="survival-time-unknown" style={{ left: `${100 * stop / end}%`, width: `${100 * (end - stop) / end}%` }}><span>›</span></span>}
    <span className="survival-time-start" style={{ left: `${100 * start / end}%` }} />
    <span className={`survival-time-stop survival-stop-${kind}`} style={{ left: `${100 * stop / end}%` }}>{kind === 'event' ? '×' : ''}</span>
    {children}
  </div>;
}

export function ObservationFigure() {
  const rows = [
    { name: 'Pump 1', time: 4, kind: 'event', status: 'Seal fails on day 4', fact: 'T = 4; observed Y = 4, δ = 1.' },
    { name: 'Pump 2', time: 4, kind: 'censor', status: 'Logging stops on day 4; still running', fact: 'T > 4; observed Y = 4, δ = 0.' },
    { name: 'Pump 3', time: 7, kind: 'censor', status: 'Logging stops later, on day 7', fact: 'T > 7; observed Y = 7, δ = 0.' },
  ];
  return <figure className="survival-inline-figure" aria-label="Failure time and last observation are different facts">
    <div className="survival-figure-heading">Same clock. Different information.</div>
    <p className="survival-figure-prompt">Both first records end at day 4. Only one records a failure.</p>
    <div className="survival-observation-lanes">
      {rows.map(row => <div className="survival-observation-row" key={row.name}>
        <strong>{row.name} · {row.status}</strong>
        <Timeline stop={row.time} end={9} kind={row.kind} unknown={row.kind === 'censor'} />
        <p>{row.fact}</p>
      </div>)}
      <ClockAxis end={9} ticks={[0, 3, 6, 9]} />
      <div className="survival-axis-caption">Days since commissioning</div>
    </div>
    <figcaption>Three illustrative records. The cross is an observed failure; the vertical cap is a censoring time. The dashed arrow says that lifetime extends beyond the last observation, with no observed endpoint. It does not place an eventual failure on this drawing. All three clocks start at commissioning, day 0.</figcaption>
  </figure>;
}

function CurvePlot({ title, description, xMaximum, yMaximum, yTicks, xTicks, path, area, markers = [], unavailableFrom = null }) {
  const id = useId();
  return <div className="survival-curve" role="group" aria-labelledby={`${id}-title`}>
    <strong className="survival-curve-title" id={`${id}-title`}>{title}</strong>
    <div className="survival-curve-with-axis">
      <div className="survival-curve-plot">
        {yTicks.map(value => <span key={value} className="survival-curve-grid" style={{ bottom: `${100 * value / yMaximum}%` }}><span>{value}</span></span>)}
        {unavailableFrom !== null && <span className="survival-curve-unobserved" style={{ left: `${100 * unavailableFrom / xMaximum}%`, width: `${100 * (xMaximum - unavailableFrom) / xMaximum}%` }} aria-hidden="true" />}
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
          {area && <path d={area} className="survival-curve-area" />}
          <path d={path} className="survival-curve-line" vectorEffect="non-scaling-stroke" />
        </svg>
        {markers.map((marker, index) => <span key={index} className={`survival-curve-marker ${marker.kind === 'censor' ? 'survival-curve-censor' : ''}`} style={{ left: `${100 * marker.time / xMaximum}%`, bottom: `${100 * marker.value / yMaximum}%` }} aria-hidden="true" />)}
      </div>
      <ClockAxis end={xMaximum} ticks={xTicks} />
      <div className="survival-axis-caption">Time (days)</div>
    </div>
    <p className="survival-curve-description">{description}</p>
  </div>;
}

function densityPath(start, stop, closeArea = false) {
  const pieces = Array.from({ length: 97 }, (_, index) => {
    const time = start + (stop - start) * index / 96;
    return `${index === 0 ? 'M' : 'L'}${100 * time / 12},${100 * (1 - Math.exp(-0.2 * time))}`;
  });
  if (closeArea) pieces.push(`L${100 * stop / 12},100 L${100 * start / 12},100 Z`);
  return pieces.join(' ');
}

export function LikelihoodFigure() {
  const survival = Math.exp(-0.2 * 4);
  const density = 0.2 * survival;
  const intervalProbability = survival * -Math.expm1(-0.2 * 0.5);
  return <figure className="survival-inline-figure" aria-label="Event density versus a censored survival tail">
    <div className="survival-figure-heading">One curve, two likelihood contributions</div>
    <p className="survival-figure-prompt">Assume an exponential lifetime with rate λ = 0.2 per day. An event at day 4 identifies a location; a censored record identifies the whole tail beyond day 4.</p>
    <div className="survival-figure-pair">
      <div>
        <CurvePlot title="Event record: density at day 4" description="Vertical scale: density, in inverse days. The narrow shaded band spans days 4 to 4.5; the dot marks f(4)." xMaximum={12} yMaximum={0.2} yTicks={[0, 0.1, 0.2]} xTicks={[0, 4, 8, 12]} path={densityPath(0, 12)} area={densityPath(4, 4.5, true)} markers={[{ time: 4, value: density }]} />
        <p className="survival-figure-result">f(4) = 0.2 e<sup>−0.8</sup> = <strong>{fixed(density)} per day</strong></p>
        <p>The finite band has probability {fixed(intervalProbability)}. Dividing its area by its width, then letting that width approach zero, gives the density. A continuous lifetime has P(T = 4) = 0.</p>
      </div>
      <div>
        <CurvePlot title="Censored record: all lifetimes past day 4" description="The same density and axes. Shading continues beyond the right edge at day 12; no failure time is imputed." xMaximum={12} yMaximum={0.2} yTicks={[0, 0.1, 0.2]} xTicks={[0, 4, 8, 12]} path={densityPath(0, 12)} area={densityPath(4, 12, true)} />
        <p className="survival-figure-result">S(4) = e<sup>−0.8</sup> = <strong>{fixed(survival)}</strong></p>
        <p>This probability includes the area after day 12, {fixed(Math.exp(-2.4))}, which lies outside the drawing. Unlike density, a probability has no time unit.</p>
      </div>
    </div>
    <figcaption>Calculated model curves, not measured histograms. For an independently censored first-event record, the lifetime-dependent likelihood factor is f(Y) for an event and S(Y) for right censoring. The lesson states the required censoring assumptions; this picture alone does not establish them.</figcaption>
  </figure>;
}

function staircase(table) {
  let path = 'M0,0';
  for (const row of table.rows) {
    path += ` H${100 * row.time / 9} V${100 * (1 - row.survival)}`;
  }
  return path;
}

export function SurvivalTreeFigure() {
  return <figure className="survival-inline-figure" aria-label="A chosen survival-tree split routes rows into two leaf estimators">
    <div className="survival-figure-heading">A leaf holds a survival curve</div>
    <p className="survival-tree-question">Chosen question: <strong>is x ≤ 3.5?</strong></p>
    <div className="survival-figure-pair survival-tree-leaves">
      {leaves.map(({ group, indices, table }) => <div key={group}>
        <div className="survival-tree-branch">{group === 0 ? 'Yes → left leaf' : 'No → right leaf'}</div>
        <ul className="survival-tree-roster">
          {indices.map(index => <li key={index}><span>x = {index}</span><span>{treeTimes[index]} days · {treeEvents[index] ? 'event ×' : 'censored |'}</span></li>)}
        </ul>
        <CurvePlot title={`Leaf ${group === 0 ? 'left' : 'right'}: Kaplan–Meier S(t)`} description={`Vertical scale: surviving fraction. Each event makes a vertical drop; a censoring cap does not. Follow-up in this leaf ends at day ${table.lastTime}.`} xMaximum={9} yMaximum={1} yTicks={[0, 0.5, 1]} xTicks={[0, 3, 6, 9]} path={staircase(table)} markers={table.rows.filter(row => row.censored).map(row => ({ time: row.time, value: row.survival, kind: 'censor' }))} unavailableFrom={table.lastTime < 9 ? table.lastTime : null} />
        <p>At its last observed time: S = <strong>{fixed(table.rows.at(-1).survival)}</strong>; Nelson–Aalen H = {fixed(table.rows.at(-1).nelsonAalen)}. {table.lastTime < 9 && 'The striped region has no observed follow-up in this leaf; no tail is drawn there.'}</p>
      </div>)}
    </div>
    <div className="survival-figure-result">Right-group log-rank totals: U = {fixed(splitScore.difference)}, V = {fixed(splitScore.variance)}. This split’s score U²/V = <strong>{fixed(splitScore.statistic)}</strong>.</div>
    <details className="survival-figure-details"><summary>Inspect the risk-set calculation for this score</summary>
      <p>At event days 2, 3 and 4, the right group has 4/8, 3/7 and 2/6 of the risk set and supplies one failure each. Later events occur after that group’s follow-up has ended, so their right-group contributions are zero.</p>
      <p>U = (1 − 4/8) + (1 − 3/7) + (1 − 2/6) = 73/42. V = 1/4 + 12/49 + 2/9 = 1265/1764. Hence U²/V = 5329/1265. These are split-ranking calculations, not a validated effect estimate or a post-selection p-value.</p>
    </details>
    <figcaption>The covariate x is an illustrative numerical scale. This is one supplied candidate threshold on eight training records, not an optimized or validated tree. A new x routes to the matching leaf curve. Both panels share the same time and probability axes. Kaplan–Meier survival and exp(−Nelson–Aalen H) are not identical finite-sample estimates.</figcaption>
  </figure>;
}

export function HistoryFigure() {
  const intervals = [
    { subject: 'A', start: 0, stop: 3, maintained: 0, event: 0, kind: 'interval', explanation: 'Maintenance has not yet happened; day 3 still uses this state.' },
    { subject: 'A', start: 3, stop: 5, maintained: 1, event: 1, kind: 'event', explanation: 'Maintenance is available after day 3. The event occurs at day 5.' },
    { subject: 'B', start: 1, stop: 6, maintained: 0, event: 0, kind: 'censor', explanation: 'Delayed entry after day 1; observation ends without an event at day 6.' },
  ];
  return <figure className="survival-inline-figure" aria-label="Predictable feature histories and delayed entry on a shared clock">
    <div className="survival-figure-heading">Use the feature value available at that time</div>
    <p className="survival-figure-prompt">Each row applies when <strong>start &lt; t ≤ stop</strong>. Open circles exclude the start; closed ends include the stop. A closed interval endpoint alone is not a failure.</p>
    <div className="survival-history-lanes">
      {intervals.map(row => <div className="survival-history-row" key={`${row.subject}-${row.start}`}>
        <strong>Subject {row.subject} · ({row.start}, {row.stop}] · maintained = {row.maintained}</strong>
        <Timeline start={row.start} stop={row.stop} end={6} kind={row.kind} tone={row.maintained ? 'gold' : 'blue'} />
        <p>Event indicator: {row.event}. {row.explanation}</p>
      </div>)}
      <ClockAxis end={6} ticks={[0, 1, 3, 5, 6]} />
      <div className="survival-axis-caption">Days since the shared time origin</div>
    </div>
    <p className="survival-history-decision">At t = 3, use A’s maintained = 0 row. At t = 4, use maintained = 1. When predicting at day 2, the later maintenance information is unavailable.</p>
    <figcaption>A’s two rows are one subject’s history, not two independent subjects. Keep them together when splitting training and test data. B joins the risk set only when 1 &lt; t ≤ 6; recording delayed entry correctly still requires suitable truncation assumptions. Correct timing alone does not identify a causal maintenance effect.</figcaption>
  </figure>;
}
