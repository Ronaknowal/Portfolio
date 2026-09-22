import { useState } from 'react';
import { fcfsTrace, occupancyWindow, queueExampleJobs, mm1State, mm1Survival, serviceVariabilityState, serviceMixtureNames, pooledQueueState, finiteBufferState, queueNumber as fmt } from '../../data/queueing-models.js';
import './QueueingLabs.css';
function ScrollFigure({
  label,
  children
}) {
  return <><div className="queueing-scroll" tabIndex={0} role="region" aria-label={label}>{children}</div><p className="queueing-scroll-hint">If the axis extends beyond the view, swipe horizontally or focus the figure and use the left/right arrow keys.</p></>;
}
function Slider({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  unit = ''
}) {
  return <label className="queueing-control">{label}: <strong>{fmt(value)} {unit}</strong><input type="range" aria-label={label} min={min} max={max} step={step} value={value} onChange={event => onChange(Number(event.target.value))} /></label>;
}
function Metrics({
  rows
}) {
  return <dl className="queueing-metrics">{rows.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>;
}
function Timeline({
  horizon = 8,
  boundary = 'system',
  selected = null
}) {
  const jobs = fcfsTrace(queueExampleJobs);
  const x = time => 40 + time * 32;
  return <ScrollFigure label="Four job timelines; scroll horizontally to inspect every second"><svg className="queueing-compact-axis" viewBox="0 0 330 230" role="img" aria-label="Waiting intervals precede service intervals on the same seconds axis">
    {Array.from({
        length: 9
      }, (_, time) => <g key={time}><line className="queueing-grid" x1={x(time)} x2={x(time)} y1={15} y2={185} /><text x={x(time)} y={212} textAnchor="middle">{time}</text></g>)}
    {jobs.map((job, index) => <g key={job.id} opacity={selected && selected !== job.id ? 0.4 : 1}><text x={8} y={42 + index * 40}>{job.id}</text><rect className="queueing-wait" x={x(job.arrival)} y={23 + index * 40} width={32 * job.wait} height={25} /><rect className="queueing-service" x={x(job.start)} y={23 + index * 40} width={32 * job.service} height={25} opacity={boundary === 'queue' ? 0.2 : 1} /><circle cx={x(job.arrival)} cy={35 + index * 40} r={4} fill="currentColor" /></g>)}
    {horizon < 8 && <line className="queueing-horizon" x1={x(horizon)} x2={x(horizon)} y1={12} y2={188} />}
    <text x={315} y={212}>s</text>
  </svg></ScrollFigure>;
}
export function QueueTimelineFigure() {
  return <figure className="queueing-figure"><Timeline /><figcaption>● Arrival · outlined amber: waiting · solid green: service. A occupies the worker until 3 s; B and C must wait. The worker is idle only from 7 s onward in this displayed interval. Jobs keep their arrival order.</figcaption></figure>;
}
export function QueueAreaLab() {
  const [horizon, setHorizon] = useState(7);
  const [boundary, setBoundary] = useState('system');
  const [selected, setSelected] = useState('C');
  const state = occupancyWindow(queueExampleJobs, horizon, boundary);
  const x = time => 40 + time * 32;
  return <section className="queueing-lab" aria-label="Occupancy area investigation">
    <h3>Count rectangles, then count each job</h3><p>Inspect what goes missing when the observation ends at 5 seconds. Select a job to identify its clipped contribution. These are exact finite traces, not simulated averages.</p>
    <Slider label="Observation horizon" value={horizon} onChange={setHorizon} min={1} max={8} step={0.25} unit="s" />
    <label className="queueing-control">Population boundary<select aria-label="Population boundary" value={boundary} onChange={event => setBoundary(event.target.value)}><option value="system">Waiting + in service</option><option value="queue">Waiting only</option></select></label>
    <div className="queueing-actions">{queueExampleJobs.map(job => <button key={job.id} aria-pressed={selected === job.id} onClick={() => setSelected(job.id)}>Job {job.id}</button>)}<button onClick={() => {
        setHorizon(7);
        setBoundary('system');
        setSelected('C');
      }}>Reset area</button></div>
    <Timeline horizon={horizon} boundary={boundary} selected={selected} />
    <ScrollFigure label="Occupancy step area in job-seconds"><svg className="queueing-compact-axis" viewBox="0 0 330 210" role="img" aria-label="Height is number of jobs; rectangle width is elapsed seconds">
      {[0, 1, 2, 3].map(count => <g key={count}><line className="queueing-grid" x1={40} x2={296} y1={160 - count * 42} y2={160 - count * 42} /><text x={13} y={166 - count * 42}>{count}</text></g>)}
      {state.segments.map(segment => <g key={segment.start}><rect className="queueing-area" x={x(segment.start)} y={160 - segment.count * 42} width={(segment.end - segment.start) * 32} height={segment.count * 42} /><line className="queueing-area-top" x1={x(segment.start)} x2={x(segment.end)} y1={160 - segment.count * 42} y2={160 - segment.count * 42} /></g>)}
      {Array.from({
          length: 9
        }, (_, time) => <text key={time} x={x(time)} y={189} textAnchor="middle">{time}</text>)}<text x={40} y={19}>jobs</text><text x={315} y={189}>s</text>
      <line className="queueing-horizon" x1={x(horizon)} x2={x(horizon)} y1={18} y2={165} />
    </svg></ScrollFigure>
    <div className="queueing-table-wrap" tabIndex={0} role="region" aria-label="Observed residence table; scroll for all columns"><table><caption>Residence inside [0, {fmt(horizon)}] seconds</caption><thead><tr><th>Job</th><th>Full interval</th><th>Observed seconds</th><th>Boundary exit observed?</th></tr></thead><tbody>{state.intervals.map(job => <tr key={job.id} className={job.id === selected ? 'queueing-selected' : ''}><th>{job.id}</th><td>[{job.arrival}, {job.finish})</td><td>{fmt(job.clipped)}</td><td>{!job.observed ? 'Not arrived' : job.completedBoundary ? 'Yes' : 'Still inside'}</td></tr>)}</tbody></table></div>
    <div aria-live="polite"><Metrics rows={[["Step area = clipped job sum", `${fmt(state.stepArea)} = ${fmt(state.clippedArea)} job·s`], ['Mean population', `${fmt(state.meanOccupancy)} jobs`], ['Completed residence + unfinished contribution', `${fmt(state.completedResidence)} + ${fmt(state.pendingArea)} job·s`], ['Completion-only product', `${fmt(state.completedResidence / horizon)} jobs`]]} /></div>
    <p>The completion-only product divides the completed residence sum by this horizon. It equals the observed average only when the unfinished contribution is zero. A zero-duration waiting interval contributes zero area but still belongs to its customer cohort.</p>
  </section>;
}
export function QueueBirthDeathFigure() {
  return <figure className="queueing-figure"><p>Number of jobs in the system</p><div className="queueing-birth-flow"><span>0</span><span><small>λ →<br />← μ</small></span><span>1</span><span><small>λ →<br />← μ</small></span><span>2</span><span><small>λ →<br />← μ</small></span><span>…</span></div><figcaption>Arrows are transition rates, not transition probabilities. There is no service completion from state 0. With one busy worker, the completion rate is μ whether 1 or 20 jobs are present.</figcaption></figure>;
}
export function MM1LoadLab() {
  const [arrival, setArrival] = useState(8);
  const state = mm1State(arrival, 10);
  return <section className="queueing-lab" aria-label="Stationary load investigation"><h3>Move load toward capacity</h3><p>Service rate stays at 10 jobs/s, so a job needs 0.1 s of service on average. Inspect whether doubling unused capacity halves the mean response time.</p><Slider label="M/M/1 arrival rate" min={0.5} max={11} step={0.5} value={arrival} onChange={setArrival} unit="jobs/s" /><button onClick={() => setArrival(8)}>Reset load</button>
    {state.stable ? <><ScrollFigure label="Stationary probabilities, including the omitted tail"><svg viewBox="0 0 620 230" role="img" aria-label="Stationary job count probabilities with an explicit 12-or-more bucket">{[...state.probabilities, state.tail].map((probability, index) => <g key={index}><rect className={index === 12 ? 'queueing-wait' : 'queueing-service'} x={40 + index * 43} y={175 - probability * 150} width={29} height={probability * 150} /><text x={54 + index * 43} y={202} textAnchor="middle">{index === 12 ? '12+' : index}</text></g>)}<text x={15} y={20}>Probability (full height = 1)</text></svg></ScrollFigure><Metrics rows={[["Busy fraction ρ", fmt(state.rho)], ['Mean jobs L', fmt(state.meanNumber)], ['Mean total W', `${fmt(state.meanTotal)} s`], ['Mean queue wait Wq', `${fmt(state.meanWait)} s`], ['Empty probability', fmt(state.idle)], ['Probability of 12 or more jobs', fmt(state.tail)]]} /><p>The tail bar collects every count ≥12. It is not one extra ordinary count, and the visible bars have not been renormalized.</p></> : <p role="alert">Arrival rate is at or above capacity. This infinite-buffer M/M/1 model has no stationary probability distribution; its stationary means and bars are unavailable.</p>}
  </section>;
}
export function QueueTailLab() {
  const [arrival, setArrival] = useState(0.9);
  const [percentile, setPercentile] = useState(0.99);
  const state = mm1State(arrival, 1, percentile);
  const horizon = state.totalQuantile * 1.12;
  const x = time => 66 + time / horizon * 490;
  const y = probability => 195 - probability * 150;
  const curve = metric => Array.from({
    length: 101
  }, (_, index) => {
    const time = index / 100 * horizon;
    return `${index ? 'L' : 'M'}${x(time)},${y(mm1Survival(arrival, 1, time)[metric])}`;
  }).join(' ');
  return <section className="queueing-lab" aria-label="Waiting and response tails investigation"><h3>A wait can be zero; a service duration is still needed</h3><p>Here μ=1 job/s. The vertical axis is strict survival P(duration &gt; t), so queue survival starts at ρ, leaving probability 1−ρ at zero. The time axis rescales with the selected load and percentile.</p><Slider label="Tail arrival rate" value={arrival} onChange={setArrival} min={0.1} max={0.95} step={0.05} unit="jobs/s" /><label className="queueing-control">Percentile<select aria-label="Duration percentile" value={percentile} onChange={event => setPercentile(Number(event.target.value))}>{[0.5, 0.9, 0.95, 0.99].map(value => <option key={value} value={value}>{value * 100}th</option>)}</select></label><button onClick={() => {
      setArrival(0.9);
      setPercentile(0.99);
    }}>Reset tails</button>
    <ScrollFigure label="Survival curves and quantile guides"><svg viewBox="0 0 620 250" role="img" aria-label="Total duration and queue wait survival curves, with the requested remaining tail level">{[0, 0.5, 1].map(value => <g key={value}><line className="queueing-grid" x1={66} x2={556} y1={y(value)} y2={y(value)} /><text x={20} y={y(value) + 6}>{value}</text></g>)}<path className="queueing-total-line" d={curve('total')} /><path className="queueing-wait-line" d={curve('queue')} /><line className="queueing-horizon" x1={66} x2={556} y1={y(1 - percentile)} y2={y(1 - percentile)} />{[0, horizon / 2, horizon].map(value => <text key={value} x={x(value)} y={225} textAnchor="middle">{fmt(value, 2)}</text>)}<text x={580} y={225}>s</text><text x={80} y={26}>Green: total · amber dashed: wait</text></svg></ScrollFigure>
    <div aria-live="polite"><Metrics rows={[["Zero-wait probability", fmt(state.idle)], ['Mean total / mean wait', `${fmt(state.meanTotal)} / ${fmt(state.meanWait)} s`], [`${percentile * 100}th total percentile`, `${fmt(state.totalQuantile)} s`], [`${percentile * 100}th wait percentile`, `${fmt(state.waitQuantile)} s`]]} /></div><p>The horizontal dashed guide is tail probability {fmt(1 - percentile)}. When the requested wait percentile lies in the zero atom, its value is 0: there is no positive-time curve crossing to find.</p>
  </section>;
}
export function ServiceVariabilityLab() {
  const [kind, setKind] = useState('twoSize');
  const [arrival, setArrival] = useState(8);
  const state = serviceVariabilityState(arrival, kind);
  return <section className="queueing-lab" aria-label="Service inspection and residual work investigation"><h3>Inspect a random job, then a random busy instant</h3><p>Every choice has mean service 0.1 s. Inspect whether a job that is rare by count can occupy most of the worker’s time. Strip widths are probabilities; they are not simulated samples.</p><label className="queueing-control">Service distribution<select aria-label="Service distribution" value={kind} onChange={event => setKind(event.target.value)}>{Object.entries(serviceMixtureNames).map(([value, name]) => <option key={value} value={value}>{name}</option>)}</select></label><Slider label="M/G/1 arrival rate" value={arrival} onChange={setArrival} min={0.5} max={9.5} step={0.5} unit="jobs/s" /><button onClick={() => {
      setKind('twoSize');
      setArrival(8);
    }}>Reset variability</button>
    {state.atoms.length ? <><p>Jobs sampled by count</p><div className="queueing-strip">{state.atoms.map((atom, index) => <div key={index} className={`queueing-tone-${index}`} style={{
          flex: atom.probability
        }} title={`${fmt(atom.duration)} s: ${fmt(100 * atom.probability)}% of jobs`} />)}</div><p>Service sampled at a busy instant</p><div className="queueing-strip">{state.atoms.map((atom, index) => <div key={index} className={`queueing-tone-${index}`} style={{
          flex: atom.busyShare
        }} title={`${fmt(atom.duration)} s: ${fmt(100 * atom.busyShare)}% of busy time`} />)}</div><ul>{state.atoms.map((atom, index) => <li key={index}>{index === 0 ? 'Green' : 'Amber'} · {fmt(atom.duration)} s jobs: {fmt(100 * atom.probability)}% by count, {fmt(100 * atom.busyShare)}% of busy time.</li>)}</ul><div className="queueing-triangles">{state.atoms.map((atom, index) => {
          const scale = 130 / Math.max(...state.atoms.map(item => item.duration));
          const left = 30;
          return <svg key={index} viewBox="0 0 265 245" role="img" aria-label={`${fmt(atom.duration)} second service: residual triangle area ${fmt(atom.triangleArea)} seconds squared`}><g><line className="queueing-grid" x1={left} x2={left + 165} y1={180} y2={180} /><path className={index === 0 ? 'queueing-residual-first' : 'queueing-residual-second'} d={`M${left},180 L${left},${180 - atom.duration * scale} L${left + atom.duration * scale},180 Z`} /><text x={left} y={23}>S = {fmt(atom.duration)} s</text><text x={left} y={211}>Area {fmt(atom.triangleArea)} s²</text></g></svg>;
        })}</div><p>Both triangle axes use seconds and a common scale within this view; the vertical height is remaining service, the horizontal width elapsed service. The smaller triangle can be tiny because area grows quadratically. Scales reset between distributions. Displayed decimals are rounded; geometry uses unrounded values.</p></> : <p>Exponential service is continuous, with density 10e<sup>−10s</sup> for s≥0. It has E[S²]=0.02 s² and no two discrete size bars. Memorylessness gives mean remaining service 0.1 s conditional on being busy.</p>}
    <div aria-live="polite"><Metrics rows={[["E[S²]", `${fmt(state.secondMoment)} s²`], ['Busy fraction', fmt(state.rho)], ['Residual given busy', `${fmt(state.busyResidual)} s`], ['Residual including idle zeroes', `${fmt(state.timeResidual)} s`], ['Mean queue wait', `${fmt(state.meanWait)} s`], ['Mean total', `${fmt(state.meanTotal)} s`]]} /></div>
  </section>;
}
export function PoolingFigure() {
  return <figure className="queueing-figure"><div className="queueing-resource"><div><strong>Separate queues: a possible instant</strong><p>Worker 1: [serving A] ← B ← C</p><p>Worker 2: [idle]</p></div><div><strong>One common queue</strong><p>Worker 1: [serving A]</p><p>Worker 2: [serving B] · C waits</p></div></div><figcaption>Resource identity, not an M/M/c timing sample. A work-conserving common queue can give B the idle worker. Separate routing can strand capacity. Speeding up a single worker changes each service duration; adding workers changes simultaneous capacity.</figcaption></figure>;
}
export function QueueCapacityLab() {
  const [arrival, setArrival] = useState(8);
  const [servers, setServers] = useState(2);
  const state = pooledQueueState(arrival, 5, servers);
  return <section className="queueing-lab" aria-label="Pooling comparison investigation"><h3>Hold per-worker speed fixed</h3><p>Each worker serves at μ=5 jobs/s. The split comparison routes each Poisson arrival independently and uniformly among workers. The fast-worker comparison serves at cμ. All use independent exponential service.</p><Slider label="Pooled arrival rate" value={arrival} onChange={setArrival} min={1} max={14} step={1} unit="jobs/s" /><Slider label="Number of workers" value={servers} onChange={setServers} min={1} max={4} /><button onClick={() => {
      setArrival(8);
      setServers(2);
    }}>Reset pooling</button>{state.stable ? <Metrics rows={[["Per-worker utilization", fmt(state.rho)], ['Chance an arrival waits in the pool', fmt(state.waitProbability)], ['Common-queue total mean', `${fmt(state.meanTotal)} s`], ['Independent split total mean', `${fmt(state.splitTotal)} s`], ['Single c-times-fast total mean', `${fmt(state.fastTotal)} s`], ['Common-queue mean wait', `${fmt(state.meanWait)} s`]]} /> : <p role="alert">Offered rate is at or above the cμ capacity. These infinite-buffer stationary comparisons are unavailable.</p>}</section>;
}
export function FiniteBufferLab() {
  const [arrival, setArrival] = useState(12);
  const [capacity, setCapacity] = useState(3);
  const state = finiteBufferState(arrival, 10, capacity);
  return <section className="queueing-lab" aria-label="Finite capacity and admission investigation"><h3>A finite queue can hide overload by rejecting work</h3><p>μ=10 jobs/s. Capacity K counts the worker’s slot as well as waiting places. Inspect the queue wait at K=1, then distinguish that from the time spent receiving service.</p><Slider label="Offered arrival rate" value={arrival} onChange={setArrival} min={1} max={20} step={1} unit="jobs/s" /><Slider label="Total system capacity" value={capacity} onChange={setCapacity} min={1} max={6} /><button onClick={() => {
      setArrival(12);
      setCapacity(3);
    }}>Reset admission</button><div className="queueing-state-list">{state.probabilities.map((probability, n) => <div key={n}><strong>{n} {n === 1 ? 'job' : 'jobs'}</strong><span>{fmt(100 * probability, 2)}%</span><small>{n === capacity ? 'Full: arrivals rejected' : n === 0 ? 'Empty: no departure' : 'Arrival admitted'}</small></div>)}</div><p>Offered arrivals split by admission:</p><div className="queueing-strip"><div className="queueing-tone-0" style={{
        flex: state.admittedProbability
      }} /><div className="queueing-tone-1" style={{
        flex: state.dropProbability
      }} /></div><Metrics rows={[["Green: admitted rate", `${fmt(state.admittedRate)} jobs/s`], ['Amber: rejection probability', fmt(state.dropProbability)], ['Mean jobs L', fmt(state.meanNumber)], ['Admitted-customer total W', `${fmt(state.meanTotal)} s`], ['Admitted-customer queue wait', `${fmt(state.meanWait)} s`], ['Completion rate μ × busy fraction', `${fmt(10 * state.busyProbability)} jobs/s`]]} /><p>Admitted throughput equals departure throughput in stationarity. Using the larger offered rate in L/λ would count rejected jobs in the denominator while omitting them from the population.</p></section>;
}
