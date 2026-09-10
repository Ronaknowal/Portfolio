import { useId, useState } from 'react';
import { DIVERGENCE_NAMES, ORIGINAL_P, ORIGINAL_Q, divergenceState, parseDivergenceWeights, processDivergence, observableState, movingAtomState, parseKernelSamples, kernelWitnessState, permutationMmdState, variationalDivergenceState } from '../../data/divergence-ipm-models.js';
import './divergence-ipm-labs.css';
function number(value, digits = 5) {
  if (value === Infinity) return '∞';
  if (value === null) return 'not defined';
  if (Math.abs(value) < 10 ** -digits && value !== 0) return value.toExponential(2);
  return String(Number(value.toFixed(digits)));
}
function Investigation({
  name,
  title,
  children
}) {
  return <section className="divergence-lab" aria-label={name}>
    <p className="divergence-eyebrow">Investigate the mechanism</p>
    <h3>{title}</h3>
    {children}
  </section>;
}
function Figure({
  title,
  height = 210,
  children
}) {
  const id = useId();
  return <svg className="divergence-figure" viewBox={`0 0 300 ${height}`} role="img" aria-labelledby={id}>
    <title id={id}>{title}</title>
    {children}
  </svg>;
}
function Table({
  title,
  headers,
  rows
}) {
  return <>
    <p className="divergence-table-title">{title}</p>
    <div className="divergence-table-scroll" role="region" aria-label={title} tabIndex={0}>
      <table><caption className="divergence-sr-only">{title}</caption>
        <thead><tr>{headers.map(header => <th key={header} scope="col">{header}</th>)}</tr></thead>
        <tbody>{rows.map((row, index) => <tr key={index}>{row.map((value, column) => column === 0 ? <th scope="row" key={column}>{value}</th> : <td key={column}>{value}</td>)}</tr>)}</tbody>
      </table>
    </div>
  </>;
}
function LinePlot({
  title,
  points,
  xmin = -3,
  xmax = 3,
  ymin,
  ymax,
  axis = 'Position',
  children
}) {
  const x = value => 40 + (value - xmin) / (xmax - xmin) * 230;
  const y = value => 150 - (value - ymin) / (ymax - ymin) * 115;
  return <Figure title={title} height={215}>
    {[ymin, (ymin + ymax) / 2, ymax].map(value => <g key={value}>
      <line x1="40" x2="270" y1={y(value)} y2={y(value)} className="divergence-gridline" />
      <text x="34" y={y(value) + 6} textAnchor="end">{number(value, 2)}</text>
    </g>)}
    <polyline points={points.map(point => `${x(point[0])},${y(point[1])}`).join(' ')} className="divergence-curve" />
    {[xmin, (xmin + xmax) / 2, xmax].map(value => <text key={value} x={x(value)} y="178" textAnchor="middle">{number(value, 2)}</text>)}
    <text x="155" y="207" textAnchor="middle">{axis}</text>
    {children?.({
      x,
      y
    })}
  </Figure>;
}
const GENERATORS = {
  kl: 'f(t) = t ln t − t + 1',
  reverse: 'f(t) = −ln t + t − 1',
  js: 'f(t) = ½[t ln(2t/(1+t)) + ln(2/(1+t))]',
  hellinger: 'f(t) = ½(√t − 1)²',
  chi: 'f(t) = (t − 1)²',
  tv: 'f(t) = ½|t − 1|'
};
export function DivergenceRatioLab() {
  const [draftP, setDraftP] = useState('7, 2, 1, 0');
  const [draftQ, setDraftQ] = useState('4, 5, 1, 0');
  const [active, setActive] = useState({
    p: ORIGINAL_P,
    q: ORIGINAL_Q
  });
  const [kind, setKind] = useState('kl');
  const [error, setError] = useState('');
  const state = divergenceState(active.p, active.q, kind);
  const maximumPenalty = Math.max(0.01, ...state.rows.map(row => Number.isFinite(row.penalty) ? row.penalty : 0));
  function apply(event) {
    event.preventDefault();
    try {
      const p = parseDivergenceWeights(draftP),
        q = parseDivergenceWeights(draftQ);
      setActive({
        p,
        q
      });
      setError('');
    } catch (failure) {
      setError(`${failure.message} Applied distributions are unchanged.`);
    }
  }
  function reset() {
    setDraftP('7, 2, 1, 0');
    setDraftQ('4, 5, 1, 0');
    setActive({
      p: ORIGINAL_P,
      q: ORIGINAL_Q
    });
    setKind('kl');
    setError('');
  }
  return <Investigation name="Probability ratio investigation" title="Give the same mismatch a different penalty">
    <p>Predict which outcome contributes most. Each row follows mass → ratio → weighted penalty. P and Q are normalized separately; labels A–D stay aligned.</p>
    <form onSubmit={apply} className="divergence-controls">
      <label>P weights<input value={draftP} onChange={event => setDraftP(event.target.value)} /></label>
      <label>Q weights<input value={draftQ} onChange={event => setDraftQ(event.target.value)} /></label>
      <button type="submit">Apply weights</button>
      <label>Penalty rule<select value={kind} onChange={event => setKind(event.target.value)}>{Object.entries(DIVERGENCE_NAMES).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></label>
    </form>
    <div className="divergence-actions">
      <button type="button" onClick={() => {
        setDraftP(active.q.join(', '));
        setDraftQ(active.p.join(', '));
        setActive({
          p: active.q,
          q: active.p
        });
        setError('');
      }}>Swap P and Q</button>
      <button type="button" onClick={() => {
        setDraftQ('0, 9, 1, 0');
        setActive({
          p: active.p,
          q: [0, 9, 1, 0]
        });
        setError('');
      }}>Remove Q’s support at A</button>
      <button type="button" onClick={reset}>Reset ratios</button>
    </div>
    {error && <p role="alert" className="divergence-error">{error}</p>}
    <p className="divergence-formula">{GENERATORS[kind]}</p>
    <div className="divergence-ratio-rows">
      {state.rows.map(row => <div className="divergence-ratio-row" key={row.label}>
        <strong className="divergence-symbol">{row.label}</strong>
        <div className="divergence-masses">
          <span>P {number(row.p)}</span><div className="divergence-mass-track"><span style={{
              width: `${100 * row.p}%`
            }} /></div>
          <span>Q {number(row.q)}</span><div className="divergence-mass-track divergence-q"><span style={{
              width: `${100 * row.q}%`
            }} /></div>
        </div>
        <div className="divergence-ratio-step"><span aria-hidden="true">→</span><span>p/q<br /><strong>{number(row.ratio)}</strong></span></div>
        <div className="divergence-penalty"><span>Weighted penalty</span><strong>{number(row.penalty)}</strong>
          {Number.isFinite(row.penalty) ? <div className="divergence-penalty-track"><span style={{
              width: `${100 * row.penalty / maximumPenalty}%`
            }} /></div> : <span className="divergence-infinite">Support mismatch</span>}
        </div>
      </div>)}
    </div>
    <output aria-live="polite">Total {DIVERGENCE_NAMES[kind]}: <strong>{number(state.total)}</strong></output>
    <p className="divergence-note">Mass bars share a 0–1 scale; finite penalty bars share their current maximum, {number(maximumPenalty)}. Infinity has no finite bar length. At p=q=0 the row contributes zero. Centered KL generators give nonnegative rows; their linear correction cancels in the total, so these are not the signed p ln(p/q) rows of the preceding lesson.</p>
    <Table title="All comparisons on the applied laws" headers={['Quantity', 'Value']} rows={Object.entries(state.values).map(([key, value]) => [DIVERGENCE_NAMES[key], number(value)])} />
  </Investigation>;
}
export function CoarseningFigure() {
  const state = processDivergence(ORIGINAL_P, ORIGINAL_Q, [[1, 0], [1, 0], [0, 1], [0, 1]]);
  return <figure className="divergence-inline">
    <Figure title="A and B merge into one event, cancelling their opposite probability differences; C and D form the other event." height={245}>
      {state.p.map((mass, index) => {
        const y = 40 + index * 49;
        const targetY = index < 2 ? 65 : 163;
        return <g key={index}><text x="12" y={y}>{String.fromCharCode(65 + index)}: {number(mass, 2)} / {number(state.q[index], 2)}</text><path d={`M 120 ${y - 5} L 179 ${targetY - 6}`} className="divergence-connector" /></g>;
      })}
      <text x="188" y="57">A or B</text><text x="188" y="83">0.9 / 0.9</text>
      <text x="188" y="155">C or D</text><text x="188" y="181">0.1 / 0.1</text>
      <text x="150" y="231" textAnchor="middle">Every pair is P mass / Q mass.</text>
    </Figure>
    <figcaption>Before merging: TV={number(state.before.tv)}, KL={number(state.before.kl)} nats. After this shared map: both output laws agree, so both discrepancies are zero mathematically. The grouping lines show membership, not mass width.</figcaption>
  </figure>;
}
export function MetricTriangleFigure() {
  const direct = divergenceState([1, 0], [0, 1], 'js').total;
  const leg = divergenceState([1, 0], [1, 1], 'js').total;
  return <figure className="divergence-inline">
    <div className="divergence-paired-figures">{[false, true].map(root => <div key={String(root)}>
      <p><strong>{root ? 'Square-root JS' : 'JS in nats'}</strong></p>
      <Figure title={`${root ? 'Square-root JS satisfies' : 'JS fails'} the triangle comparison for two disjoint point masses and their balanced mixture.`} height={225}>
        <path d="M 38 146 L 150 50 L 262 146 M 38 161 L 262 161" className="divergence-connector" />
        <circle cx="38" cy="146" r="7" className="divergence-point" /><circle cx="150" cy="50" r="7" className="divergence-point" /><circle cx="262" cy="146" r="7" className="divergence-point" />
        <text x="150" y="26" textAnchor="middle">½δ₀ + ½δ₁</text>
        <text x="12" y="186">δ₀</text><text x="269" y="186">δ₁</text>
        <text x="49" y="78">{number(root ? Math.sqrt(leg) : leg, 3)}</text><text x="200" y="78">{number(root ? Math.sqrt(leg) : leg, 3)}</text>
        <text x="150" y="152" textAnchor="middle">{number(root ? Math.sqrt(direct) : direct, 3)}</text>
        <text x="150" y="216" textAnchor="middle">{number(root ? Math.sqrt(direct) : direct, 3)} {root ? '≤' : '>'} {number(root ? 2 * Math.sqrt(leg) : 2 * leg, 3)}</text>
      </Figure>
    </div>)}</div>
    <figcaption>The direct comparison must not exceed the sum of two legs for a metric. The triangle layout names the three probability laws; it is not their physical geometry. One example refutes JS’s triangle inequality. The general metric theorem for its square root requires a proof beyond checking this example.</figcaption>
  </figure>;
}
export function ObservableCriticLab() {
  const [kind, setKind] = useState('event');
  const [preset, setPreset] = useState('spread');
  const state = preset === 'spread' ? observableState(undefined, undefined, undefined, kind) : observableState([0, 1, 0, 0], [0, 0, 1, 0], undefined, kind);
  const magnitude = Math.max(1, ...state.scores.map(Math.abs));
  return <Investigation name="Observable class investigation" title="Change what the observer is allowed to do">
    <p>The broad and narrow distributions have the same mean. Predict whether a linear score can distinguish them, then allow a different function class.</p>
    <div className="divergence-controls">
      <label>Distribution pair<select value={preset} onChange={event => setPreset(event.target.value)}><option value="spread">Same mean, different spread</option><option value="shift">Move a point mass from −1 to +1</option></select></label>
      <label>Allowed observer<select value={kind} onChange={event => setKind(event.target.value)}><option value="event">Any score from 0 to 1 (TV)</option><option value="linear">Linear ax, |a| ≤ 1</option><option value="lipschitz">Any 1-Lipschitz score (W1)</option></select></label>
    </div>
    <p>Gold curve = an optimal permitted score. Point labels give its values; negative scores are allowed only in the last two classes.</p>
    <LinePlot title="Optimal observer scores as a function of physical position" points={state.positions.map((value, index) => [value, state.scores[index]])} ymin={-magnitude} ymax={magnitude} axis="Physical position">
      {({
        x,
        y
      }) => state.positions.map((position, index) => <g key={position}><circle cx={x(position)} cy={y(state.scores[index])} r="5" className="divergence-point" /><text x={x(position)} y={y(state.scores[index]) - 12} textAnchor={index === 0 ? 'start' : index === state.positions.length - 1 ? 'end' : 'middle'}>{number(state.scores[index], 2)}</text></g>)}
    </LinePlot>
    <Table title="Where the observer’s expectation gap comes from" headers={['Position', 'P', 'Q', 'Score g', '(P−Q)g']} rows={state.positions.map((position, index) => [position, number(state.p[index]), number(state.q[index]), number(state.scores[index]), number(state.contributions[index])])} />
    <output aria-live="polite">Best gap in this class: <strong>{number(state.value)}</strong></output>
    <p className="divergence-note">Same pair: linear gap {number(state.linearValue)}, TV {number(state.tv)}, W1 {number(state.w1)} in position units. Each value answers its own question. Adding a constant to a score changes neither expectation difference; the W1 score is pinned to zero at the first position only to choose one representative.</p>
    <button type="button" onClick={() => {
      setPreset('spread');
      setKind('event');
    }}>Reset observers</button>
  </Investigation>;
}
export function MovingAtomLab() {
  const [displacement, setDisplacement] = useState(0.5);
  const [bandwidth, setBandwidth] = useState(1);
  const state = movingAtomState(displacement, bandwidth);
  return <Investigation name="Moving point mass investigation" title="Move the mass closer without creating overlap">
    <p>P puts all mass at 0. Q puts all mass at h. Move h toward zero and predict which quantities approach zero smoothly.</p>
    <div className="divergence-controls">
      <label>Displacement h: {number(displacement, 2)}<input type="range" min="0" max="3" step="0.01" value={displacement} onChange={event => setDisplacement(Number(event.target.value))} /></label>
      <label>Gaussian kernel bandwidth: {number(bandwidth, 2)}<input type="range" min="0.2" max="3" step="0.1" value={bandwidth} onChange={event => setBandwidth(Number(event.target.value))} /></label>
    </div>
    <Figure title={`P is a point mass at zero; Q is at ${displacement}. Vertical stems each represent total probability one.`} height={185}>
      <line x1="32" x2="266" y1="121" y2="121" className="divergence-connector" />
      <line x1="32" x2="32" y1="115" y2="45" className="divergence-p-stem" />
      <line x1={32 + displacement * 78} x2={32 + displacement * 78} y1="115" y2="72" className="divergence-q-stem" />
      <text x="32" y="28">P: mass 1</text><text x={Math.min(185, 32 + displacement * 78)} y="62">Q: mass 1</text>
      {[0, 1, 2, 3].map(value => <text key={value} x={32 + value * 78} y="145" textAnchor="middle">{value}</text>)}
      <text x="150" y="172" textAnchor="middle">Physical location</text>
    </Figure>
    <Table title="Exact comparisons of these point-mass laws" headers={['Comparison', 'Value']} rows={[['KL(P ∥ Q), nats', number(state.kl)], ['TV', number(state.tv)], ['JS, nats', number(state.js)], ['W1, location units', number(state.w1)], ['Gaussian MMD', number(state.mmd)]]} />
    <output aria-live="polite">{displacement === 0 ? 'At h=0 the two laws are identical; every listed comparison is zero.' : `At h=${number(displacement, 2)} the supports are still disjoint. Moving nearer changes W1 and Gaussian MMD, while TV and JS stay at their disjoint-support values.`}</output>
    <p className="divergence-note">The stems identify two atoms; their display heights are staggered for visibility and do not encode unequal mass. All values are analytic, not measured training performance. MMD²=2[1−exp(−h²/(2σ²))] for this unit-amplitude Gaussian kernel.</p>
    <div className="divergence-actions"><button type="button" onClick={() => setDisplacement(0)}>Make the laws identical</button><button type="button" onClick={() => {
        setDisplacement(0.5);
        setBandwidth(1);
      }}>Reset moving mass</button></div>
  </Investigation>;
}
export function FeatureMeanFigure() {
  return <figure className="divergence-inline">
    <p>P: equal mass at −1 and +1. Q: all mass at 0. Both raw means are zero.</p>
    <Figure title="Map x to the pair (x,x squared). P’s two feature points average to (0,1); Q’s feature mean is (0,0)." height={235}>
      <line x1="40" x2="265" y1="175" y2="175" className="divergence-connector" /><line x1="150" x2="150" y1="30" y2="183" className="divergence-connector" />
      <line x1="55" x2="245" y1="65" y2="65" className="divergence-curve" />
      <circle cx="55" cy="65" r="7" className="divergence-point" /><circle cx="245" cy="65" r="7" className="divergence-point" />
      <path d="M150 54 L160 65 L150 76 L140 65 Z" className="divergence-point" />
      <circle cx="150" cy="175" r="8" className="divergence-q-point" />
      <text x="55" y="43" textAnchor="middle">(−1,1)</text><text x="245" y="43" textAnchor="middle">(1,1)</text>
      <text x="160" y="102">P mean (0,1)</text><text x="158" y="157">Q mean (0,0)</text>
      <text x="58" y="199" textAnchor="middle">−1</text><text x="150" y="199" textAnchor="middle">0</text><text x="245" y="199" textAnchor="middle">1</text>
      <text x="150" y="229" textAnchor="middle">First feature: x</text><text x="18" y="118" transform="rotate(-90 18 118)" textAnchor="middle">Second: x²</text>
    </Figure>
    <figcaption>The feature-mean difference is (0,1), whose Euclidean norm is 1. This finite feature map gives kernel k(x,y)=xy+x²y². It detects this pair; matching only these two moments would still hide some other differences.</figcaption>
  </figure>;
}
const SAMPLE_PRESETS = {
  variance: {
    label: 'Same mean, different spread',
    x: [-2, -2, 2, 2],
    y: [-1, -1, 1, 1]
  },
  shifted: {
    label: 'Shifted samples',
    x: [-2, -1.5, -1, -0.5],
    y: [0.5, 1, 1.5, 2]
  },
  matched: {
    label: 'Identical two-point empirical laws',
    x: [-1, 1],
    y: [-1, 1]
  }
};
export function KernelWitnessLab() {
  const [draftX, setDraftX] = useState('-2, -2, 2, 2');
  const [draftY, setDraftY] = useState('-1, -1, 1, 1');
  const [samples, setSamples] = useState(SAMPLE_PRESETS.variance);
  const [kind, setKind] = useState('rbf');
  const [bandwidth, setBandwidth] = useState(1);
  const [error, setError] = useState('');
  const state = kernelWitnessState(samples.x, samples.y, bandwidth, kind);
  const maxWitness = Math.max(0.1, Math.ceil(Math.max(...state.witness.map(point => Math.abs(point.value))) * 10) / 10);
  function loadPreset(key) {
    const next = SAMPLE_PRESETS[key];
    setSamples(next);
    setDraftX(next.x.join(', '));
    setDraftY(next.y.join(', '));
    setError('');
  }
  function apply(event) {
    event.preventDefault();
    try {
      setSamples({
        x: parseKernelSamples(draftX),
        y: parseKernelSamples(draftY)
      });
      setError('');
    } catch (failure) {
      setError(`${failure.message} Applied samples are unchanged.`);
    }
  }
  const combined = [...state.x, ...state.y];
  const n = state.x.length;
  const grid = combined.map((_, row) => combined.map((_, column) => {
    const value = row < n ? column < n ? state.xx[row][column] : state.xy[row][column - n] : column < n ? state.xy[column][row - n] : state.yy[row - n][column - n];
    const signed = row < n === column < n ? 1 : -1;
    const denominator = (row < n ? n : state.y.length) * (column < n ? n : state.y.length);
    return {
      value,
      contribution: signed * value / denominator,
      same: signed > 0
    };
  }));
  return <Investigation name="Kernel witness investigation" title="See the pairs that build the discrepancy">
    <p>Start with identical raw means and different spread. Predict what a linear kernel reports, then let the Gaussian kernel compare local neighborhoods.</p>
    <div className="divergence-actions">{Object.entries(SAMPLE_PRESETS).map(([key, preset]) => <button type="button" onClick={() => loadPreset(key)} key={key}>{preset.label}</button>)}</div>
    <form className="divergence-controls" onSubmit={apply}>
      <label>X observations<input value={draftX} onChange={event => setDraftX(event.target.value)} /></label>
      <label>Y observations<input value={draftY} onChange={event => setDraftY(event.target.value)} /></label>
      <button type="submit">Apply samples</button>
      <label>Kernel<select value={kind} onChange={event => setKind(event.target.value)}><option value="rbf">Gaussian (RBF)</option><option value="linear">Linear: xy</option><option value="quadratic">Two features: xy + x²y²</option></select></label>
      <label>Gaussian bandwidth σ: {number(bandwidth, 2)}<input type="range" min="0.2" max="3" step="0.1" value={bandwidth} disabled={kind !== 'rbf'} onChange={event => setBandwidth(Number(event.target.value))} /></label>
    </form>
    {error && <p role="alert" className="divergence-error">{error}</p>}
    <p><strong>Unnormalized witness w(t).</strong> Positive values favor X’s kernel neighborhood, negative values favor Y’s. This curve is a difference of average similarities, not a probability density.</p>
    <LinePlot title="Unnormalized kernel witness; sample markers show the actual observations" points={state.witness.map(point => [point.position, point.value])} ymin={-maxWitness} ymax={maxWitness} axis="Query location t">
      {({
        x
      }) => <>{state.x.map((value, index) => <circle key={`x${index}`} cx={x(value)} cy={140 - 7 * (index % 2)} r="4" className="divergence-point" />)}{state.y.map((value, index) => <rect key={`y${index}`} x={x(value) - 4} y={145 - 7 * (index % 2)} width="8" height="8" className="divergence-q-point" />)}</>}
    </LinePlot>
    <p className="divergence-note">Gold circles are X observations; pale squares are Y observations. Their vertical offsets only separate markers. Witness scale ±{number(maxWitness)} changes with the selected kernel. Every observation, including a repeated value, remains a separate sample.</p>
    <p className="divergence-table-title">The signed Gram sum</p>
    <p className="divergence-note">Cells show signed, normalized contributions to the biased squared statistic. Within-X and within-Y blocks add; cross blocks subtract. A negative raw linear similarity may reverse an individual contribution’s sign. Headers include group, sample number and actual value.</p>
    <div className="divergence-table-scroll" role="region" aria-label="Signed kernel contribution matrix" tabIndex={0}>
      <table className="divergence-gram"><caption className="divergence-sr-only">Signed normalized pair contributions</caption><thead><tr><th scope="col">Pair</th>{combined.map((value, index) => <th scope="col" key={index}>{index < n ? `X${index + 1}` : `Y${index - n + 1}`}<br />{number(value, 2)}</th>)}</tr></thead>
        <tbody>{grid.map((row, index) => <tr key={index}><th scope="row">{index < n ? `X${index + 1}` : `Y${index - n + 1}`}</th>{row.map((cell, column) => <td key={column} className={cell.same ? 'divergence-within' : 'divergence-cross'} title={`Raw kernel: ${number(cell.value)}; signed normalized contribution: ${number(cell.contribution)}`}>{number(cell.contribution, 3)}</td>)}</tr>)}</tbody>
      </table>
    </div>
    <Table title="Pair sums and the two estimator contracts" headers={['Quantity', 'Value']} rows={[['Average within X', number(state.withinX)], ['Average within Y', number(state.withinY)], ['Twice average across groups', number(2 * state.cross)], ['Biased squared MMD', number(state.biasedSquared)], ['Empirical-law MMD (square root)', number(state.mmd)], ['Unbiased squared MMD estimate', number(state.unbiasedSquared)], ['E_X w − E_Y w', number(state.witnessGap)]]} />
    <output aria-live="polite">Biased squared MMD {number(state.biasedSquared)}; unbiased squared estimate {number(state.unbiasedSquared)}.</output>
    <p className="divergence-note">The unbiased estimate removes only within-group diagonal pairs and changes their denominator; it keeps all cross pairs. A negative result stays negative. The biased squared norm only clips a negative roundoff residue below 10⁻¹² to zero; it never takes an absolute value. A Gaussian bandwidth choice is part of the comparison, not a harmless display setting.</p>
    <button type="button" onClick={() => {
      loadPreset('variance');
      setKind('rbf');
      setBandwidth(1);
    }}>Reset kernel comparison</button>
  </Investigation>;
}
export function PermutationMmdLab() {
  const [bandwidth, setBandwidth] = useState(1);
  const [selected, setSelected] = useState(0);
  const state = permutationMmdState(bandwidth, selected);
  const maximum = Math.max(...state.allocations.map(allocation => allocation.statistic));
  const bins = Array.from({
    length: 10
  }, () => 0);
  state.allocations.forEach(allocation => {
    bins[Math.min(9, Math.floor(allocation.statistic / maximum * 10))] += 1;
  });
  const maxCount = Math.max(...bins);
  return <Investigation name="Exact permutation investigation" title="Keep the observations; change the source labels">
    <p>Under the equal-law iid null, the 70 ways to label four of these eight observations X are exchangeable. The original split is allocation 1. Its score and upper-tail p-value remain the reference while you inspect another allocation.</p>
    <div className="divergence-controls">
      <label>Prespecified bandwidth σ<select value={bandwidth} onChange={event => {
          setBandwidth(Number(event.target.value));
          setSelected(0);
        }}><option value="0.3">0.3</option><option value="1">1</option><option value="3">3</option></select></label>
      <label>Inspect allocation: {selected + 1} of 70<input type="range" min="0" max="69" step="1" value={selected} onChange={event => setSelected(Number(event.target.value))} /></label>
    </div>
    <div className="divergence-allocation">{state.pool.map((value, index) => <span key={index} className={state.current.indices.includes(index) ? 'divergence-label-x' : 'divergence-label-y'}><strong>{state.current.indices.includes(index) ? 'X' : 'Y'}</strong>{value}</span>)}</div>
    <p>Inspected allocation score: <strong>{number(state.current.statistic)}</strong>. Original score: <strong>{number(state.observed)}</strong>.</p>
    <Figure title="Histogram of all seventy exact label-allocation MMD squared statistics with the original observed statistic marked." height={225}>
      {[0, Math.ceil(maxCount / 2), maxCount].map(count => <g key={count}><line x1="38" x2="269" y1={155 - count / maxCount * 115} y2={155 - count / maxCount * 115} className="divergence-gridline" /><text x="32" y={161 - count / maxCount * 115} textAnchor="end">{count}</text></g>)}
      {bins.map((count, index) => <rect key={index} x={40 + index * 23} y={155 - count / maxCount * 115} width="19" height={count / maxCount * 115} className="divergence-hist-bar" />)}
      <line x1={40 + state.observed / maximum * 226} x2={40 + state.observed / maximum * 226} y1="31" y2="159" className="divergence-observed-line" />
      <text x="265" y="23" textAnchor="end">Observed</text><text x="38" y="181">0</text><text x="269" y="181" textAnchor="end">{number(maximum, 2)}</text><text x="150" y="216" textAnchor="middle">Squared MMD statistic</text>
    </Figure>
    <p className="divergence-note">Vertical axis: allocation count. Ten equal-width bins summarize the exact enumerated values; tail counting uses the values, not these histogram bins. Ties within 10⁻¹² are included conservatively.</p>
    <output aria-live="polite">Original upper tail: {state.tailCount}/70; exact permutation p-value <strong>{number(state.pValue, 6)}</strong>.</output>
    <p>This is conditional evidence against equal laws under the sampling assumptions. The eight chosen values are a teaching fixture, not evidence of a real deployment shift. Choosing the most favorable displayed p-value after trying bandwidths does not preserve a prespecified test.</p>
    <div className="divergence-actions"><button type="button" disabled={selected === 0} onClick={() => setSelected(selected - 1)}>Previous allocation</button><button type="button" disabled={selected === 69} onClick={() => setSelected(selected + 1)}>Next allocation</button><button type="button" onClick={() => {
        setBandwidth(1);
        setSelected(0);
      }}>Reset permutations</button></div>
  </Investigation>;
}
export function VariationalDivergenceLab() {
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState(0);
  const state = variationalDivergenceState(scale, offset);
  const largest = Math.max(1, ...state.rows.flatMap(row => [Math.abs(row.reward), row.penalty]));
  return <Investigation name="Variational divergence investigation" title="A critic earns a reward and pays a convex penalty">
    <p>Hold P=(.7,.2,.1) and Q=(.4,.5,.1) fixed. The optimal KL critic is T*=1+ln(p/q). Scale its log-ratio part or add an offset; predict whether the true KL changes.</p>
    <div className="divergence-controls">
      <label>Log-ratio scale: {number(scale, 2)}<input type="range" min="0" max="1.5" step="0.05" value={scale} onChange={event => setScale(Number(event.target.value))} /></label>
      <label>Score offset: {number(offset, 2)}<input type="range" min="-2" max="2" step="0.1" value={offset} onChange={event => setOffset(Number(event.target.value))} /></label>
    </div>
    <p>T=1+scale·ln(p/q)+offset. Every outcome contributes pT−q exp(T−1), using the uncentered generator f(u)=u ln u.</p>
    <div className="divergence-critic-ledger">{state.rows.map(row => <div key={row.label}>
      <strong>{row.label}: score {number(row.score)}</strong>
      <div className="divergence-signed-pair"><span>pT {number(row.reward)}</span><div className="divergence-signed-track"><span className="divergence-reward" style={{
              left: `${row.reward < 0 ? 50 + 50 * row.reward / largest : 50}%`,
              width: `${50 * Math.abs(row.reward) / largest}%`
            }} /></div></div>
      <div className="divergence-signed-pair"><span>q exp(T−1) {number(row.penalty)}</span><div className="divergence-signed-track"><span className="divergence-cost" style={{
              left: '50%',
              width: `${50 * row.penalty / largest}%`
            }} /></div></div>
      <p>Row bound: <strong>{number(row.bound)}</strong></p>
    </div>)}</div>
    <p className="divergence-note">Center lines are zero. Gold is the signed linear reward, pale bars the positive convex penalty that is subtracted. All rows share scale ±{number(largest)}. This is a known-law population calculation, not a fitted finite-sample estimate.</p>
    <Table title="Critic objective and exact target" headers={['Quantity', 'Nats']} rows={[['True KL (fixed)', number(state.truth)], ['Critic lower bound', number(state.bound)], ['Gap to true KL', number(state.gap)]]} />
    <output aria-live="polite">True KL {number(state.truth)}; critic bound {number(state.bound)}.</output>
    <div className="divergence-actions"><button type="button" onClick={() => {
        setScale(0);
        setOffset(0);
      }}>Use a constant critic</button><button type="button" onClick={() => {
        setScale(1);
        setOffset(0);
      }}>Restore optimal critic</button></div>
  </Investigation>;
}
