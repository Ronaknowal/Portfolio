import { useState } from 'react';
import { ASYMMETRIC_PROPOSAL, SYMMETRIC_PROPOSAL, finiteMetropolis, independentEstimate, metropolisTrace, hamiltonianTrajectory, exactHarmonic, nutsExpansion, stationaryMeanVariance, twoStateTrace } from '../../data/monte-carlo-mcmc-models.js';
import './monte-carlo-mcmc-labs.css';
const colors = ['#f0bd58', '#70d5b0', '#c2a4f3'];
const fmt = value => value === null ? 'not estimable yet' : value === -Infinity ? '−∞' : Math.abs(value) > 9999 || value !== 0 && Math.abs(value) < .0001 ? value.toExponential(2) : Number(value.toFixed(4)).toString();
function Range({
  label,
  value,
  onChange,
  min,
  max,
  step = 1
}) {
  return <label className="mcmc-range">{label}: <strong>{fmt(value)}</strong><input aria-label={label} type="range" value={value} min={min} max={max} step={step} onChange={event => onChange(Number(event.target.value))} /></label>;
}
function Readouts({
  rows
}) {
  return <dl className="mcmc-readouts">{rows.map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{value}</dd></div>)}</dl>;
}
function Seed({
  seed,
  onApply,
  name
}) {
  const [draft, setDraft] = useState(String(seed));
  const [error, setError] = useState('');
  function apply(event) {
    event.preventDefault();
    const value = Number(draft);
    if (!Number.isInteger(value) || value < 1 || value > 4294967295) {
      setError('Enter an integer seed from 1 through 4294967295. The applied experiment has not changed.');
      return;
    }
    setError('');
    onApply(value);
  }
  return <form className="mcmc-seed" onSubmit={apply}><label>{name} seed<input aria-label={`${name} seed`} inputMode="numeric" value={draft} onChange={event => setDraft(event.target.value)} /></label><button type="submit">Apply {name.toLowerCase()} seed</button>{error && <p role="alert">{error}</p>}</form>;
}
function DataTable({
  caption,
  headings,
  rows
}) {
  return <details className="mcmc-table"><summary>{caption}</summary><div tabIndex={0} role="region" aria-label={caption}><table><thead><tr>{headings.map(heading => <th key={heading} scope="col">{heading}</th>)}</tr></thead><tbody>{rows.map((row, i) => <tr key={i}>{row.map((cell, j) => j === 0 ? <th key={j} scope="row">{cell}</th> : <td key={j}>{cell}</td>)}</tr>)}</tbody></table></div></details>;
}
function Plot({
  title,
  series,
  xmin = 0,
  xmax = 1,
  ymin = 0,
  ymax = 1,
  xlabel,
  ylabel,
  marker,
  baseline,
  dots = false
}) {
  const X = x => 52 + 278 * (x - xmin) / (xmax - xmin);
  const Y = y => 174 - 124 * (y - ymin) / (ymax - ymin);
  return <figure className="mcmc-plot"><figcaption>{title}</figcaption><svg viewBox="0 0 360 238" role="img" aria-label={`${title}. ${ylabel} versus ${xlabel}; values calculated from this experiment.`}>
    <text x="52" y="26">{ylabel}</text><path d="M52 45V174H330" className="mcmc-axis" />
    <text x="50" y="58" textAnchor="end">{fmt(ymax)}</text><text x="50" y="179" textAnchor="end">{fmt(ymin)}</text>
    {[xmin, (xmax + xmin) / 2, xmax].map(x => <text key={x} x={X(x)} y="202" textAnchor="middle">{fmt(x)}</text>)}
    <text x="190" y="230" textAnchor="middle">{xlabel}</text>
    {baseline !== undefined && <line x1="52" x2="330" y1={Y(baseline)} y2={Y(baseline)} stroke="#d7dce4" strokeDasharray="4 5" />}
    {series.map((row, index) => ({ ...row, seriesIndex: index })).reverse().map(row => <g key={row.name}><path d={row.points.map(([x, y], index) => `${index ? 'L' : 'M'}${X(x)} ${Y(y)}`).join(' ')} fill="none" stroke={colors[row.seriesIndex]} strokeWidth="2.7" strokeDasharray={row.seriesIndex === 1 ? '6 4' : undefined} />{dots && row.seriesIndex === 0 && row.points.map(([x, y], index) => <circle key={index} cx={X(x)} cy={Y(y)} r="3.5" fill={colors[row.seriesIndex]} />)}</g>)}
    {marker && <><line x1={X(marker[0])} x2={X(marker[0])} y1="45" y2="174" stroke="#fff" strokeDasharray="3 4" /><circle cx={X(marker[0])} cy={Y(marker[1])} r="6" fill="#fff" stroke="#0c1015" strokeWidth="2" /></>}
  </svg><div className="mcmc-legend">{series.map((row, i) => <span key={row.name} style={{
        color: colors[i]
      }}>{i === 1 ? '┄' : '━'} {row.name}</span>)}{baseline !== undefined && <span>┄ exact reference {fmt(baseline)}</span>}</div></figure>;
}
export function IndependentMonteCarloLab() {
  const [seed, setSeed] = useState(7);
  const [evaluations, setEvaluations] = useState(100);
  const [paired, setPaired] = useState(false);
  const [group, setGroup] = useState(1);
  const [resetCount, setResetCount] = useState(0);
  const result = independentEstimate({
    seed,
    evaluations,
    antithetic: paired
  });
  const current = result.rows[group - 1];
  const reset = () => {
    setSeed(7);
    setEvaluations(100);
    setPaired(false);
    setGroup(1);
    setResetCount(value => value + 1);
  };
  return <section className="mcmc-lab" aria-label="Independent Monte Carlo investigation"><h3>One question, many random contributions</h3><p>Estimate the average of U² for U uniform on [0,1]. Inspect whether pairing U with 1−U can reduce error at the same number of function evaluations. The dashed line is the analytically known answer, 1/3.</p>
    <div className="mcmc-controls"><label>Evaluation budget<select aria-label="Evaluation budget" value={evaluations} onChange={event => {
          setEvaluations(Number(event.target.value));
          setGroup(1);
        }}><option value="20">20 evaluations</option><option value="100">100 evaluations</option><option value="1000">1000 evaluations</option></select></label><label>Sampling design<select aria-label="Sampling design" value={paired ? 'paired' : 'iid'} onChange={event => {
          setPaired(event.target.value === 'paired');
          setGroup(1);
        }}><option value="iid">Independent U values</option><option value="paired">Antithetic pairs U, 1−U</option></select></label></div>
    <Seed key={resetCount} seed={seed} name="Average" onApply={value => {
      setSeed(value);
      setGroup(1);
    }} />
    <Plot title="Running estimate at equal evaluation cost" xlabel="function evaluations" ylabel="running mean" xmax={evaluations} baseline={1 / 3} series={[{
      name: paired ? 'Average of independent pair means' : 'Average of independent contributions',
      points: result.rows.map(row => [row.evaluations, row.mean])
    }]} marker={[current.evaluations, current.mean]} />
    <Range label="Inspect independent group" value={group} min={1} max={result.independentGroups} onChange={setGroup} />
    <div className="mcmc-contributions">{current.inputs.map((x, i) => <div key={i}><span>U{paired && i === 1 ? ' reflected' : ''}</span><strong>{fmt(x)}</strong><span aria-hidden="true">↓ square</span><strong>{fmt(current.contributions[i])}</strong></div>)}<div className="mcmc-group"><span>{paired ? 'Pair contribution' : 'Contribution'}</span><strong>{fmt(current.contributions.reduce((a, b) => a + b) / current.contributions.length)}</strong><span>added to the running average</span></div></div>
    <Readouts rows={[[`Final estimate (${evaluations} evaluations)`, fmt(result.estimate)], ['Estimated MCSE', fmt(result.mcse)], ['Independent groups', result.independentGroups], ['Exact SD of this estimator', fmt(Math.sqrt(result.exactVariance))]]} />
    <p>The estimated MCSE uses variation between {paired ? 'independent pairs, not the correlated members of each pair' : 'independent contributions'}. One seed can look unusually lucky or unlucky. The exact variance comparison describes repeated experiments, not guaranteed error for this run.</p>
    <button onClick={reset}>Reset average experiment</button><DataTable caption="Contribution and running-average table" headings={['Group', 'Inputs', 'Contributions', 'Running mean']} rows={result.rows.map((row, i) => [i + 1, row.inputs.map(fmt).join(', '), row.contributions.map(fmt).join(', '), fmt(row.mean)])} />
  </section>;
}
export function MetropolisFlowLab() {
  const [correct, setCorrect] = useState(true);
  const [symmetric, setSymmetric] = useState(false);
  const [pair, setPair] = useState('0,1');
  const proposal = symmetric ? SYMMETRIC_PROPOSAL : ASYMMETRIC_PROPOSAL;
  const model = finiteMetropolis([2, 5, 3], proposal, correct);
  const [a, b] = pair.split(',').map(Number);
  const labels = ['A', 'B', 'C'];
  return <section className="mcmc-lab" aria-label="Metropolis probability flow investigation"><h3>Balance what arrives with what leaves</h3><p>Target masses are A=.2, B=.5 and C=.3. Inspect A→B with the asymmetric proposal: a more probable destination is proposed so often that only some proposals should be accepted.</p>
    <div className="mcmc-controls"><label>Proposal matrix<select aria-label="Proposal matrix" value={symmetric ? 'symmetric' : 'asymmetric'} onChange={event => setSymmetric(event.target.value === 'symmetric')}><option value="asymmetric">Asymmetric proposals</option><option value="symmetric">Symmetric proposals</option></select></label><label>Acceptance rule<select aria-label="Acceptance rule" value={correct ? 'correct' : 'wrong'} onChange={event => setCorrect(event.target.value === 'correct')}><option value="correct">Full Metropolis–Hastings ratio</option><option value="wrong">Deliberately omit proposal ratio</option></select></label><label>Inspect state pair<select aria-label="Inspect state pair" value={pair} onChange={event => setPair(event.target.value)}><option value="0,1">A ↔ B</option><option value="0,2">A ↔ C</option><option value="1,2">B ↔ C</option></select></label></div>
    <figure className="mcmc-flow"><figcaption>Selected pair · accepted probability flow per step</figcaption><svg viewBox="0 0 360 210" role="img" aria-label={`Flow ${labels[a]} to ${labels[b]} is ${fmt(model.flow[a][b])}; reverse is ${fmt(model.flow[b][a])}.`}><circle cx="47" cy="99" r="30" /><circle cx="313" cy="99" r="30" /><text x="47" y="105" textAnchor="middle">{labels[a]}</text><text x="313" y="105" textAnchor="middle">{labels[b]}</text><path d="M80 82H275m-11-7 11 7-11 7" stroke={colors[0]} /><path d="M280 116H85m11-7-11 7 11 7" stroke={colors[1]} /><text x="180" y="65" textAnchor="middle">{fmt(model.flow[a][b])} →</text><text x="180" y="150" textAnchor="middle">← {fmt(model.flow[b][a])}</text><text x="180" y="190" textAnchor="middle">{Math.abs(model.flow[a][b] - model.flow[b][a]) < 1e-12 ? 'Pairwise flow balances' : 'Pairwise flow does not balance'}</text></svg></figure>
    <div className="mcmc-flow-equations"><p><strong>{labels[a]}→{labels[b]}</strong> target {fmt(model.target[a])} × proposal {fmt(proposal[a][b])} × accept {fmt(model.acceptance[a][b])} = <strong>{fmt(model.flow[a][b])}</strong></p><p><strong>{labels[b]}→{labels[a]}</strong> target {fmt(model.target[b])} × proposal {fmt(proposal[b][a])} × accept {fmt(model.acceptance[b][a])} = <strong>{fmt(model.flow[b][a])}</strong></p></div>
    <div className="mcmc-masses">{labels.map((label, i) => <div key={label}><strong>{label}</strong><span>Target {fmt(model.target[i])}</span><span>After one step {fmt(model.afterOne[i])}</span><div className="mcmc-mass-bars" aria-hidden="true"><i style={{
            width: `${100 * model.target[i]}%`
          }} /><i style={{
            width: `${100 * model.afterOne[i]}%`
          }} /></div></div>)}</div>
    <p role="status">{model.afterOne.every((value, i) => Math.abs(value - model.target[i]) < 1e-12) ? 'Starting at the target distribution leaves that distribution unchanged.' : 'Starting at the target distribution moves it away. This rule is wrong for this proposal.'} Diagonal transition entries include staying proposals and rejected moves.</p>
    <button onClick={() => {
      setCorrect(true);
      setSymmetric(false);
      setPair('0,1');
    }}>Reset probability flow</button><DataTable caption="Proposal, acceptance and transition matrices" headings={['From → to', 'q(to | from)', 'Accept', 'P(to | from)']} rows={labels.flatMap((from, i) => labels.map((to, j) => [`${from} → ${to}`, fmt(proposal[i][j]), fmt(model.acceptance[i][j]), fmt(model.transition[i][j])]))} />
  </section>;
}
export function MetropolisTraceLab() {
  const [seed, setSeed] = useState(7);
  const [scale, setScale] = useState(.12);
  const [index, setIndex] = useState(1);
  const [warmup, setWarmup] = useState(40);
  const [resetCount, setResetCount] = useState(0);
  const model = metropolisTrace({
    seed,
    scale,
    iterations: 200
  });
  const row = model.rows[index - 1];
  const retained = model.rows.slice(warmup, index);
  const bins = Array(10).fill(0);
  retained.forEach(item => {
    bins[Math.min(9, Math.floor(item.state * 10))] += 1;
  });
  return <section className="mcmc-lab" aria-label="Metropolis chain investigation"><h3>Watch the chain spend time, including when it stays</h3><p>The target is Beta(10,4). This short 200-transition run explains the mechanism; it is much shorter than the native estimation example. Inspect a rejected proposal and find its repeated state in the trace.</p><div className="mcmc-controls"><label>Proposal standard deviation<select aria-label="Proposal standard deviation" value={scale} onChange={event => {
          setScale(Number(event.target.value));
          setIndex(1);
        }}><option value="0.01">.01 · small moves</option><option value="0.12">.12 · moderate moves</option><option value="0.5">.50 · large moves</option></select></label><Range label="Warmup transitions omitted" min={0} max={100} value={warmup} onChange={setWarmup} /></div>
    <Seed key={resetCount} name="Chain" seed={seed} onApply={value => {
      setSeed(value);
      setIndex(1);
    }} />
    <Plot title="Resulting state after every transition" xlabel="transition" ylabel="probability θ" xmax={200} baseline={5 / 7} series={[{
      name: 'All resulting states, including repeats',
      points: [[0, .5], ...model.rows.map(item => [item.index + 1, item.state])]
    }]} marker={[index, row.state]} />
    <Range label="Inspect transition" min={1} max={200} value={index} onChange={setIndex} />
    <div className="mcmc-actions"><button disabled={index === 1} onClick={() => setIndex(value => value - 1)}>Previous proposal</button><button disabled={index === 200} onClick={() => setIndex(value => value + 1)}>Next proposal</button><button onClick={() => setIndex(200)}>Inspect all 200</button></div>
    <div className={`mcmc-decision ${row.accept ? 'accepted' : 'rejected'}`}><span>Before <strong>{fmt(row.before)}</strong></span><span>Propose <strong>{fmt(row.proposal)}</strong></span><span>{row.accept ? '✓ accept' : '↩ reject; repeat'} <strong>{fmt(row.state)}</strong></span></div>
    <Readouts rows={[[`Transition ${index} · log acceptance`, fmt(row.logAcceptance)], ['log uniform threshold', fmt(Math.log(row.uniform))], ['Acceptance over all 200', fmt(model.acceptanceRate)], [`Retained through transition ${index}`, retained.length]]} />
    <figure className="mcmc-occupancy"><figcaption>Retained occupancy through selected transition · bin counts</figcaption><div className="mcmc-histogram">{bins.map((count, i) => <div key={i}><span>{count}</span><i style={{
            height: `${retained.length ? 80 * count / Math.max(1, ...bins) : 0}px`
          }} /><small>{(i / 10).toFixed(1)}</small></div>)}</div><p>{retained.length ? `Each retained resulting state adds one count. Current mean ${fmt(retained.reduce((sum, item) => sum + item.state, 0) / retained.length)}.` : 'No retained draws yet: move beyond the warmup boundary.'} Bins have width .1; the final bin ends at 1. Vertical heights use the largest current count.</p></figure>
    <p>The full trace is visible for comparison; the histogram uses only transitions after warmup and through the white marker. Omission is an analysis choice, not evidence that convergence occurred at that point. Browser seeds use xorshift32/Box–Muller, so they do not reproduce Python's Random stream.</p><button onClick={() => {
      setSeed(7);
      setScale(.12);
      setIndex(1);
      setWarmup(40);
      setResetCount(value => value + 1);
    }}>Reset chain investigation</button>
    <DataTable caption="Every proposal and resulting state" headings={['Transition', 'Before', 'Proposal', 'Accepted?', 'Result']} rows={model.rows.map(item => [item.index + 1, fmt(item.before), fmt(item.proposal), item.accept ? 'yes' : 'no; repeat', fmt(item.state)])} />
  </section>;
}
export function HamiltonianTrajectoryLab() {
  const [step, setStep] = useState(.2);
  const [steps, setSteps] = useState(12);
  const [sigma, setSigma] = useState(1);
  const [selected, setSelected] = useState(1);
  const model = hamiltonianTrajectory({
    step,
    steps,
    sigma
  });
  const index = Math.min(selected, model.rows.length - 1);
  const current = model.rows[index];
  const orbit = Array.from({
    length: 161
  }, (_, i) => exactHarmonic(1, .7, 2 * Math.PI * sigma * i / 160, sigma));
  const max = Math.ceil(Math.max(1.5, ...orbit.map(row => Math.abs(row.position)), ...orbit.map(row => Math.abs(row.momentum)), ...model.rows.flatMap(row => [Math.abs(row.position), Math.abs(row.momentum)])) * 2) / 2;
  return <section className="mcmc-lab" aria-label="Hamiltonian trajectory investigation"><h3>A phase portrait: position moves while momentum changes</h3><p>Start at position q=1 with momentum r=.7. The target is Normal(0,σ²), with unit mass. The green dashed curve is exact constant-energy dynamics; the amber points are actual leapfrog steps. Neither axis is execution time.</p>
    <div className="mcmc-controls"><Range label="Leapfrog step size" min={.05} max={1} step={.05} value={step} onChange={value => {
        setStep(value);
        setSelected(1);
      }} /><Range label="Leapfrog step count" min={1} max={32} value={steps} onChange={value => {
        setSteps(value);
        setSelected(1);
      }} /><label>Target standard deviation<select aria-label="Target standard deviation" value={sigma} onChange={event => {
          setSigma(Number(event.target.value));
          setSelected(1);
        }}><option value="1">σ=1</option><option value="0.5">σ=.5</option><option value="0.25">σ=.25 · narrow target</option></select></label></div>
    <Plot title="Position–momentum trajectory" xlabel="position q" ylabel="momentum r" xmin={-max} xmax={max} ymin={-max} ymax={max} series={[{
      name: 'Leapfrog path',
      points: model.rows.map(row => [row.position, row.momentum])
    }, {
      name: 'Exact energy orbit',
      points: orbit.map(row => [row.position, row.momentum])
    }]} marker={[current.position, current.momentum]} dots />
    <Range label="Inspect integration state" min={0} max={model.rows.length - 1} value={index} onChange={setSelected} />
    <div className="mcmc-kicks"><div><span>Half kick</span><strong>{index ? fmt(current.halfMomentum) : 'not started'}</strong><small>momentum r½</small></div><span aria-hidden="true">→</span><div><span>Drift</span><strong>{fmt(current.position)}</strong><small>position q′</small></div><span aria-hidden="true">→</span><div><span>Half kick</span><strong>{fmt(current.momentum)}</strong><small>momentum r′</small></div></div>
    <Readouts rows={[[`State ${index} energy H`, fmt(current.energy)], ['State energy error ΔH', fmt(current.energyError)], ['Whole proposal acceptance', fmt(model.acceptance)], ['Requested artificial time Lε', fmt(steps * step)]]} />
    {model.divergent && <p role="status" className="mcmc-warning">The teaching guard stopped integration when |ΔH| exceeded 1000 or energy became nonfinite. This proposal is rejected. The plot shows finite states before the guard, with expanded axes; a compressed exact orbit is not good mixing.</p>}
    <p>Now choose σ=.25 and ε=1. The same step is too large for this curvature. This deterministic investigation keeps its starting momentum fixed; a complete HMC chain refreshes momentum and then accepts or rejects the whole trajectory.</p><button onClick={() => {
      setStep(.2);
      setSteps(12);
      setSigma(1);
      setSelected(1);
    }}>Reset Hamiltonian trajectory</button>
    <DataTable caption="Leapfrog states and energy errors" headings={['Step', 'Position', 'Half momentum', 'Momentum', 'Energy error']} rows={model.rows.map(row => [row.index, fmt(row.position), row.halfMomentum === undefined ? 'initial' : fmt(row.halfMomentum), fmt(row.momentum), fmt(row.energyError)])} />
  </section>;
}
export function NutsTreeLab() {
  const [seed, setSeed] = useState(7);
  const [step, setStep] = useState(.25);
  const [depth, setDepth] = useState(5);
  const [round, setRound] = useState(1);
  const [resetCount, setResetCount] = useState(0);
  const model = nutsExpansion({
    seed,
    step,
    maxDepth: depth
  });
  const actualRound = Math.min(round, model.snapshots.length);
  const snapshot = model.snapshots[actualRound - 1];
  const allowed = new Set(snapshot.candidates.map(state => state.time));
  const lastRound = actualRound === model.snapshots.length;
  const displacement = snapshot.right.position - snapshot.left.position;
  return <section className="mcmc-lab" aria-label="NUTS candidate tree investigation"><h3>Build both directions; choose from an eligible set</h3><p>This is the original slice-based NUTS Algorithm 2, with stored candidates, unit mass and a Normal(0,1) target. Each expansion chooses a direction and attempts twice as many new leapfrog states. The signed index is position along the numerical trajectory, not the parameter value.</p>
    <div className="mcmc-controls"><Range label="NUTS step size" min={.05} max={1.5} step={.05} value={step} onChange={value => {
        setStep(value);
        setRound(1);
      }} /><Range label="NUTS maximum depth" min={1} max={6} value={depth} onChange={value => {
        setDepth(value);
        setRound(1);
      }} /></div><Seed key={resetCount} seed={seed} name="Tree" onApply={value => {
      setSeed(value);
      setRound(1);
    }} />
    <div className="mcmc-tree-root"><strong>Initial trajectory index 0</strong><span>q=1, r={fmt(model.initial.momentum)}</span><span>log slice={fmt(model.logSlice)}</span></div>
    <ol className="mcmc-tree">{model.snapshots.slice(0, actualRound).map((row, i) => {
        const previousCount = i ? model.snapshots[i - 1].explored.length : 0;
        return <li key={row.depth}><div className="mcmc-tree-heading"><strong>{row.direction < 0 ? '← backward' : 'forward →'} · expansion {row.depth}</strong><span>{row.newStates} new states · {row.subtreeAccepted ? 'subtree eligible' : 'discard entire new subtree'}</span></div><div className="mcmc-tree-nodes">{row.explored.slice(previousCount).map(state => <div key={state.time} className={!state.validEnergy ? 'invalid' : allowed.has(state.time) ? 'candidate' : 'excluded'}><strong>index {state.time}</strong><span>q={fmt(state.position)}</span><small>{!state.validEnergy ? 'energy guard' : !state.onSlice ? 'off slice' : allowed.has(state.time) ? 'in candidate pool' : 'subtree excluded'}</small></div>)}</div></li>;
      })}</ol>
    <Range label="Inspect doubling round" min={1} max={model.snapshots.length} value={actualRound} onChange={setRound} /><div className="mcmc-actions"><button disabled={actualRound === model.snapshots.length} onClick={() => setRound(value => value + 1)}>Next doubling</button><button onClick={() => setRound(model.snapshots.length)}>Show final candidate set</button></div>
    <Readouts rows={[[`Left endpoint index ${snapshot.left.time}`, `q=${fmt(snapshot.left.position)}, r=${fmt(snapshot.left.momentum)}`], [`Right endpoint index ${snapshot.right.time}`, `q=${fmt(snapshot.right.position)}, r=${fmt(snapshot.right.momentum)}`], ['Displacement × left momentum', fmt(displacement * snapshot.left.momentum)], ['Displacement × right momentum', fmt(displacement * snapshot.right.momentum)], ['Eligible candidates (includes initial)', snapshot.candidates.length], ['Construction state', snapshot.stopReason]]} />
    <p className="mcmc-pool"><strong>Pool indices:</strong> {snapshot.candidates.map(state => state.time).sort((a, b) => a - b).join(', ')}.</p>
    <p role="status">{lastRound ? `Final uniform selection: index ${model.selected.time}, position ${fmt(model.selected.position)}. Each of these ${model.candidateCount} eligible states has probability 1/${model.candidateCount}.` : 'Selection waits until construction stops. Advance to see the actual final draw.'}</p>
    <p>“Internal subtree turn” discards the entire new subtree even if some of its points satisfy the slice. “Whole-tree turn” can retain a valid new subtree before stopping. A depth cap limits this teaching run; it does not assert that a turn was reached. Current production NUTS implementations can use different candidate weighting.</p><button onClick={() => {
      setSeed(7);
      setStep(.25);
      setDepth(5);
      setRound(1);
      setResetCount(value => value + 1);
    }}>Reset NUTS construction</button>
    <DataTable caption="Explored trajectory states and candidate eligibility" headings={['Index', 'q', 'r', 'On slice?', 'In current pool?']} rows={snapshot.explored.map(state => [state.time, fmt(state.position), fmt(state.momentum), state.onSlice ? 'yes' : 'no', allowed.has(state.time) ? 'yes' : 'no'])} />
  </section>;
}
export function CorrelatedPrecisionLab() {
  const [rho, setRho] = useState(.8);
  const [thin, setThin] = useState(1);
  const [draws, setDraws] = useState(100);
  const [seed, setSeed] = useState(7);
  const [resetCount, setResetCount] = useState(0);
  const all = stationaryMeanVariance(draws, rho);
  const selected = stationaryMeanVariance(draws, rho, thin);
  const states = twoStateTrace({
    seed,
    correlation: rho,
    draws
  });
  const retained = states.filter((_, i) => (i + 1) % thin === 0);
  return <section className="mcmc-lab" aria-label="Correlated precision investigation"><h3>Same number of transitions, different information</h3><p>A stationary two-state chain targets 0 and 1 equally. It flips with probability (1−ρ)/2. Positive ρ produces runs; negative ρ encourages alternation. Inspect what keeping every second draw does when ρ=−.8.</p>
    <div className="mcmc-controls"><Range label="Lag-one correlation" min={-.9} max={.9} step={.1} value={rho} onChange={setRho} /><label>Transition budget<select aria-label="Transition budget" value={draws} onChange={event => setDraws(Number(event.target.value))}><option value="50">50 transitions</option><option value="100">100 transitions</option><option value="200">200 transitions</option></select></label><label>Keep every kth state<select aria-label="Keep every kth state" value={thin} onChange={event => setThin(Number(event.target.value))}><option value="1">Every state · k=1</option><option value="2">Every second · k=2</option><option value="5">Every fifth · k=5</option><option value="10">Every tenth · k=10</option></select></label></div>
    <Seed key={resetCount} seed={seed} name="Precision" onApply={setSeed} />
    <div className="mcmc-bits" aria-label="Resulting states; outlined states retained">{states.map((value, i) => <span key={i} className={`${value ? 'one' : 'zero'} ${(i + 1) % thin === 0 ? 'retained' : 'omitted'}`} title={`Transition ${i + 1}: ${value}${(i + 1) % thin === 0 ? ', retained' : ', omitted'}`}>{value}</span>)}</div><p>Each digit is one transition's resulting state. Outlined digits are retained. The initial state is drawn from the known stationary distribution, so the exact variance comparison below includes no initialization bias.</p>
    <Plot title="Known lag correlations, before and after thinning" xlabel="lag in retained draws" ylabel="correlation" xmax={8} ymin={-1} ymax={1} baseline={0} series={[{
      name: 'All states: ρ to the lag',
      points: Array.from({
        length: 9
      }, (_, lag) => [lag, rho ** lag])
    }, {
      name: `Every ${thin}th: ρ to k×lag`,
      points: Array.from({
        length: 9
      }, (_, lag) => [lag, rho ** (thin * lag)])
    }]} dots />
    <Readouts rows={[[`Mean of ${retained.length} retained digits`, fmt(retained.reduce((a, b) => a + b) / retained.length)], ['Target mean', '.5'], ['Exact MCSE using all states', fmt(all.mcse)], ['Exact MCSE after thinning', fmt(selected.mcse)], ['Equivalent iid count after thinning', fmt(selected.equivalentIID)], ['Actual retained count', selected.retained]]} />
    <p>These are exact finite-budget variances from the specified chain's covariance formula, not error estimates fitted to this one trace. Equivalent iid count is defined for this mean. It can exceed the draw count under anticorrelation; thinning may destroy that advantage.</p><button onClick={() => {
      setRho(.8);
      setThin(1);
      setDraws(100);
      setSeed(7);
      setResetCount(value => value + 1);
    }}>Reset precision experiment</button>
    <DataTable caption="Resulting states and retention table" headings={['Transition', 'State', 'Retained?']} rows={states.map((value, i) => [i + 1, value, (i + 1) % thin === 0 ? 'yes' : 'no'])} />
  </section>;
}
