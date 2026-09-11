import { useState } from 'react';
import { outcomeState, meanLossState, jointState, sharedNoiseState, conditionalState, squaredUniformState, sampleMeanState } from '../../data/random-variables-models.js';
import './random-variable-labs.css';
const number = value => value === null ? 'undefined' : Number(value.toPrecision(6)).toString();
function Range({
  label,
  value,
  onChange,
  min,
  max,
  step = 1
}) {
  return <label className="rv-control">{label}: <strong>{number(value)}</strong><input aria-label={label} type="range" min={min} max={max} step={step} value={value} onChange={event => onChange(Number(event.target.value))} /></label>;
}
function Choice({
  label,
  value,
  onChange,
  options
}) {
  return <label className="rv-control">{label}<select aria-label={label} value={value} onChange={event => onChange(event.target.value)}>{options.map(([id, title]) => <option key={id} value={id}>{title}</option>)}</select></label>;
}
function Reset({
  onClick
}) {
  return <button className="rv-reset" onClick={onClick}>Reset investigation</button>;
}
function Result({
  children
}) {
  return <p className="rv-result" aria-live="polite">{children}</p>;
}
function Metrics({
  items
}) {
  return <dl className="rv-metrics">{items.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{typeof value === 'number' || value === null ? number(value) : value}</dd></div>)}</dl>;
}
function Distribution({
  rows,
  title,
  xLabel = 'Value',
  extent
}) {
  const low = extent ? extent[0] : Math.min(...rows.map(row => row.value)) - .5;
  const high = extent ? extent[1] : Math.max(...rows.map(row => row.value)) + .5;
  const x = value => 36 + (value - low) / (high - low) * 232;
  const width = Math.max(3, Math.min(22, 150 / rows.length));
  return <figure className="rv-distribution"><figcaption>{title}</figcaption><svg viewBox="0 0 300 234" role="img" aria-label={title + '. Probability mass (rounded): ' + rows.map(row => number(row.value) + ': ' + number(row.mass)).join('; ')}>
    <line x1="36" y1="174" x2="280" y2="174" className="rv-axis" /><line x1="36" y1="30" x2="36" y2="174" className="rv-axis" />
    <text x="12" y="34">1</text><text x="12" y="180">0</text><text x="40" y="18">Probability mass</text>
    {rows.map((row, i) => <g key={i}><rect x={x(row.value) - width / 2} y={174 - 140 * row.mass} width={width} height={140 * row.mass} className="rv-mass" /><circle cx={x(row.value)} cy="174" r="2" className="rv-mass" />{(rows.length <= 6 || i === 0 || i === rows.length - 1 || i === Math.floor(rows.length / 2)) && <text x={x(row.value)} y="194" textAnchor="middle">{number(row.value)}</text>}</g>)}
    <text x="154" y="225" textAnchor="middle">{xLabel}</text>
    </svg><details><summary>Probability table (rounded)</summary><p>Calculated from the population law; displayed to six significant digits.</p><table><thead><tr><th>Value</th><th>Probability</th></tr></thead><tbody>{rows.map((row, i) => <tr key={i}><td>{number(row.value)}</td><td>{number(row.mass)}</td></tr>)}</tbody></table></details></figure>;
}
export function OutcomePushforwardLab() {
  const [first, setFirst] = useState(50),
    [second, setSecond] = useState(50),
    [mapping, setMapping] = useState('heads');
  const state = outcomeState(first, second, mapping);
  const [threshold, setThreshold] = useState(1);
  const mass = state.distribution.filter(row => row.value <= threshold).reduce((sum, row) => sum + row.mass, 0);
  return <section className="rv-lab" aria-label="Outcome mapping investigation"><h3>Many outcomes can produce the same number</h3><p>Two independent coins supply four possible outcomes. The rule changes which number each outcome receives; it preserves that outcome's probability.</p>
    <div className="rv-controls"><Range label="First coin head chance (%)" value={first} onChange={setFirst} min={0} max={100} /><Range label="Second coin head chance (%)" value={second} onChange={setSecond} min={0} max={100} /><Choice label="Numerical rule" value={mapping} onChange={setMapping} options={[["heads", "Total heads"], ["first", "First coin heads (0/1)"], ["equal", "Coins agree: 1 or 0"]]} /></div>
    <div className="rv-outcomes">{state.outcomes.map(row => <div key={row.label} className={row.mass === 0 ? 'rv-zero' : ''}><strong>{row.label}</strong><span>mass {number(row.mass)}</span><b aria-hidden="true">↓</b><span>X = <strong>{row.value}</strong></span></div>)}</div>
    <div className="rv-two"><Distribution rows={state.distribution} title="Collect all routes into each value" /><div><h4>Cumulative probability includes the endpoint</h4><Range label="CDF threshold t" value={threshold} onChange={setThreshold} min={-1} max={3} step={.25} /><p>F(t) = P(X ≤ t). Add only the value groups at or below the selected threshold.</p><table><thead><tr><th>x</th><th>P(X=x)</th><th>F(x)</th></tr></thead><tbody>{state.distribution.map(row => <tr key={row.value} className={row.value <= threshold ? 'rv-selected' : ''}><td>{row.value}</td><td>{number(row.mass)}</td><td>{number(row.cumulative)}</td></tr>)}</tbody></table><Result>F({number(threshold)}) = {number(mass)}. Zero-mass displayed values are outside the support of this active law.</Result></div></div>
    <Reset onClick={() => {
      setFirst(50);
      setSecond(50);
      setMapping('heads');
      setThreshold(1);
    }} />
  </section>;
}
export function MeanSquaredLossLab() {
  const [candidate, setCandidate] = useState(0),
    [preset, setPreset] = useState('asymmetric');
  const state = meanLossState(candidate, preset),
    x = value => 30 + (value + 3) * 30;
  const curve = Array.from({
    length: 71
  }, (_, i) => {
    const c = -3 + i / 10;
    return `${x(c)},${180 - (state.variance + (c - state.mean) ** 2) * 5}`;
  }).join(' ');
  return <section className="rv-lab" aria-label="Mean and squared loss investigation"><h3>Choose the one number you will predict</h3><div className="rv-controls"><Range label="Constant prediction c" value={candidate} onChange={setCandidate} min={-3} max={4} step={.25} /><Choice label="Population law" value={preset} onChange={setPreset} options={[["asymmetric", "−2, 0, 3: unequal masses"], ["symmetric", "−1 or 1, equally likely"], ["constant", "Always 2"]]} /></div>
    <div className="rv-two"><figure><figcaption>Each square measures (x−c)² before weighting</figcaption><svg viewBox="0 0 300 256" role="img" aria-label="Residual squares drawn to the same value scale on both axes.">
      {state.pieces.map((row, i) => <rect key={i} x={Math.min(x(row.value), x(candidate))} y={220 - Math.abs(row.residual) * 30} width={Math.abs(row.residual) * 30} height={Math.abs(row.residual) * 30} className={'rv-square rv-tone-' + i} />)}
      <line x1="25" y1="220" x2="268" y2="220" className="rv-axis" />{[-3, -2, -1, 0, 1, 2, 3, 4].map(value => <text key={value} x={x(value)} y="243" textAnchor="middle">{value}</text>)}<line x1={x(candidate)} x2={x(candidate)} y1="24" y2="223" className="rv-guide" /><text x={x(candidate)} y="20" textAnchor="middle">c</text>
    </svg></figure><figure><figcaption>Expected loss over possible predictions</figcaption><svg viewBox="0 0 300 240" role="img" aria-label={'Exact loss equals variance ' + number(state.variance) + ' plus squared distance from mean ' + number(state.mean)}>
      <line x1="25" y1="180" x2="268" y2="180" className="rv-axis" /><polyline points={curve} className="rv-curve" /><circle cx={x(candidate)} cy={180 - state.loss * 5} r="6" className="rv-point" /><circle cx={x(state.mean)} cy={180 - state.variance * 5} r="5" className="rv-optimum" /><text x="28" y="20">Squared loss</text><text x="12" y="180">0</text><text x="12" y="80">20</text>{[-3, 0, 4].map(value => <text key={value} x={x(value)} y="201" textAnchor="middle">{value}</text>)}<text x="155" y="222" textAnchor="middle">Prediction c</text>
    </svg></figure></div>
    <table><caption>Weight each square by its probability</caption><thead><tr><th>x</th><th>p</th><th>x−c</th><th>p(x−c)²</th></tr></thead><tbody>{state.pieces.map(row => <tr key={row.value}><td>{row.value}</td><td>{number(row.mass)}</td><td>{number(row.residual)}</td><td>{number(row.contribution)}</td></tr>)}</tbody></table>
    <Metrics items={[["Mean", state.mean], ["Variance: unavoidable constant-prediction loss", state.variance], ["Extra loss from c", state.excess], ["Total expected squared loss", state.loss]]} /><Result>The green point minimizes this exact population loss. Its prediction is {number(state.mean)}; it need not be a possible observation.</Result><Reset onClick={() => {
      setCandidate(0);
      setPreset('asymmetric');
    }} />
  </section>;
}
export function JointCovarianceLab() {
  const [preset, setPreset] = useState('matching'),
    [scale, setScale] = useState(1),
    [shift, setShift] = useState(0),
    [selected, setSelected] = useState(0);
  const state = jointState(preset, scale, shift),
    cell = state.cells[Math.min(selected, state.cells.length - 1)];
  const px = value => 150 + value * 74,
    py = value => 165 - value * 22;
  return <section className="rv-lab" aria-label="Joint covariance investigation"><h3>The pairings carry information</h3><p>Each point is a possible population pair. Its area represents probability, not a count of collected data. The first three presets share the same original marginals.</p><div className="rv-controls"><Choice label="Joint law" value={preset} onChange={value => {
        setPreset(value);
        setSelected(0);
      }} options={[["matching", "Y = X"], ["opposite", "Y = −X"], ["independent", "Independent X and Y"], ["nonlinear", "Y = X²"]]} /><Range label="Scale Y" value={scale} onChange={setScale} min={-2} max={2} /><Range label="Shift Y" value={shift} onChange={setShift} min={-3} max={3} /></div>
    <div className="rv-two"><figure><figcaption>Center both coordinates before multiplying</figcaption><svg viewBox="0 0 300 315" role="img" aria-label={'Population means ' + number(state.x.mean) + ', ' + number(state.y.mean) + '; covariance ' + number(state.covariance)}>
      <line x1="45" x2="257" y1={py(state.y.mean)} y2={py(state.y.mean)} className="rv-guide" /><line x1={px(state.x.mean)} x2={px(state.x.mean)} y1="38" y2="286" className="rv-guide" />
      <line x1="45" x2="257" y1="286" y2="286" className="rv-axis" /><line x1="45" x2="45" y1="38" y2="286" className="rv-axis" />
      {[-1, 0, 1].map(x => <text key={x} x={px(x)} y="307" textAnchor="middle">{x}</text>)}{[-5, 0, 5].map(y => <text key={y} x="17" y={py(y) + 5}>{y}</text>)}<text x="23" y="23">Y</text><text x="271" y="305">X</text>
      {state.cells.filter(row => row.mass > 0).map((row, i) => <circle key={i} cx={px(row.x)} cy={py(row.y)} r={Math.sqrt(row.mass) * 24} className={row.contribution < 0 ? 'rv-negative' : row.contribution > 0 ? 'rv-positive' : 'rv-neutral'} />)}
    </svg><p>Dashed lines are means. Gold marks positive centered products; blue negative; gray zero. They are not causal arrows.</p></figure><div className="rv-table-scroll"><table className="rv-joint"><caption>Click a joint cell. Rows=X, columns=Y.</caption><thead><tr><th>X \ Y</th>{state.ys.map(y => <th key={y}>{y}</th>)}<th>P(X)</th></tr></thead><tbody>{state.xs.map(x => <tr key={x}><th>{x}</th>{state.ys.map(y => {
                const index = state.cells.findIndex(row => row.x === x && row.y === y);
                const row = state.cells[index];
                return <td key={y}><button aria-label={'Inspect X ' + x + ' Y ' + y} aria-pressed={row === cell} onClick={() => setSelected(index)}>{number(row.mass)}</button></td>;
              })}<td>{number(state.cells.find(row => row.x === x).px)}</td></tr>)}</tbody><tfoot><tr><th>P(Y)</th>{state.ys.map(y => <td key={y}>{number(state.cells.find(row => row.y === y).py)}</td>)}<td>1</td></tr></tfoot></table></div></div>
    <Result>At X={cell.x}, Y={cell.y}: joint mass {number(cell.mass)}; marginal product {number(cell.product)}; covariance contribution {number(cell.contribution)}. {state.independent ? 'Every cell matches its product: independent in this finite model.' : 'At least one cell differs: dependent in this finite model.'}</Result><Metrics items={[["Mean X / Y", number(state.x.mean) + ' / ' + number(state.y.mean)], ["Variance X / Y", number(state.x.variance) + ' / ' + number(state.y.variance)], ["Covariance", state.covariance], ["Correlation", state.correlation]]} /><p>When a variance is zero, correlation is undefined. A covariance of zero alone does not answer the independence question.</p><Reset onClick={() => {
      setPreset('matching');
      setScale(1);
      setShift(0);
      setSelected(0);
    }} />
  </section>;
}
export function SharedNoiseLab() {
  const [common, setCommon] = useState(2),
    [local, setLocal] = useState(1),
    [a, setA] = useState(.5),
    [b, setB] = useState(.5);
  const state = sharedNoiseState(common, local, a, b);
  return <section className="rv-lab" aria-label="Shared noise investigation"><h3>Follow the same disturbance into both readings</h3><p>Exact synthetic model: A=10+S+e_A and B=20+S+e_B mV. Independent signs make S=±common amplitude and each e=±local amplitude. This chart is a population calculation, not a device measurement.</p><div className="rv-controls"><Range label="Common amplitude (mV)" value={common} onChange={setCommon} min={0} max={3} step={.5} /><Range label="Local amplitude (mV)" value={local} onChange={setLocal} min={0} max={2} step={.5} /><Range label="Coefficient a" value={a} onChange={setA} min={-1} max={1} step={.25} /><Range label="Coefficient b" value={b} onChange={setB} min={-1} max={1} step={.25} /></div><div className="rv-actions"><button onClick={() => {
        setA(.5);
        setB(.5);
      }}>Average (A+B)/2</button><button onClick={() => {
        setA(-1);
        setB(1);
      }}>Difference B−A</button></div>
    <div className="rv-noise"><div className="rv-shared">The same S <strong>↙   ↘</strong><span>enters A and B</span></div><div className="rv-two"><div>A noise contribution<strong>{number(a)}S {a < 0 ? '−' : '+'} {number(Math.abs(a))}e_A</strong></div><div>B noise contribution<strong>{number(b)}S {b < 0 ? '−' : '+'} {number(Math.abs(b))}e_B</strong></div></div><div className="rv-combine">Combined noise → <strong>{number(a + b)}S {a < 0 ? '−' : '+'} {number(Math.abs(a))}e_A {b < 0 ? '−' : '+'} {number(Math.abs(b))}e_B</strong></div><p>The fixed baseline is shown separately below.</p></div>
    <table><caption>Independent noise terms after combining</caption><thead><tr><th>Term</th><th>Variance (mV²)</th></tr></thead><tbody>{['(a+b)S', 'a e_A', 'b e_B'].map((name, i) => <tr key={name}><td>{name}</td><td>{number(state.components[i])}</td></tr>)}</tbody></table><Metrics items={[["Mean aA+bB (mV)", state.combined.mean], ["Variance aA+bB (mV²)", state.combined.variance], ["Cov(A,B) (mV²)", state.covariance], ["SD aA+bB (mV)", state.combined.sd]]} /><Distribution rows={state.distribution} title="Exact law of the combined quantity" xLabel="aA+bB (mV)" /><Result>The fixed baseline is {number(10 * a + 20 * b)} mV. {a + b === 0 ? 'The common noise cancels exactly.' : 'The common noise survives with coefficient ' + number(a + b) + '.'} Comparing two estimators requires that they target the same quantity; averaging these different baselines does not by itself accomplish that.</Result><Reset onClick={() => {
      setCommon(2);
      setLocal(1);
      setA(.5);
      setB(.5);
    }} />
  </section>;
}
export function ConditionalMomentsLab() {
  const [percent, setPercent] = useState(50),
    [group, setGroup] = useState('-2');
  const state = conditionalState(percent),
    selected = state.groups.find(row => row.g === Number(group));
  const coordinate = value => 150 + 35 * value;
  return <section className="rv-lab" aria-label="Conditional moments investigation"><h3>Separate changes within a group from changes between groups</h3><p>G is −2 or 2, and independent U is equally likely ±1. The pair is X=G+U, Y=G−U. Observing G permits a different prediction in each group.</p><div className="rv-controls"><Range label="P(G=2), percent" value={percent} onChange={setPercent} min={0} max={100} /><Choice label="Inspect group" value={group} onChange={setGroup} options={[["-2", "G = −2"], ["2", "G = 2"]]} /></div>
    <figure><svg viewBox="0 0 300 225" role="img" aria-label={'Conditional Y values on the same axis. Overall mean ' + number(state.y.mean) + '. Active group means at minus two and/or plus two.'}>
      <line x1={coordinate(state.y.mean)} x2={coordinate(state.y.mean)} y1="28" y2="155" className="rv-guide" />
      {state.groups.map((row,index) => <g key={row.g} opacity={row.mass ? 1 : .35}>
        <text x="15" y={30+index*75}>G = {row.g}{row.mass ? '' : ': no mass'}</text>
        <line x1={coordinate(row.g-1)} x2={coordinate(row.g+1)} y1={52+index*75} y2={52+index*75} className="rv-interval" />
        {[-1,1].map(offset=><circle key={offset} cx={coordinate(row.g+offset)} cy={52+index*75} r="6" className="rv-point" />)}
        {row.moments && <circle cx={coordinate(row.moments.y.mean)} cy={52+index*75} r="6" className="rv-optimum" />}
      </g>)}
      <line x1="30" x2="270" y1="165" y2="165" className="rv-axis" />{[-3,-2,0,2,3].map(value=><text key={value} x={coordinate(value)} y="187" textAnchor="middle">{value}</text>)}<text x="150" y="208" textAnchor="middle">Y on a common scale</text>
    </svg><figcaption>Each row connects its two Y values. Green is that group's mean; the dashed line is the overall mean. Changing group weights moves the overall mean. A faded zero-mass row contributes nothing and has no identified conditional mean.</figcaption></figure>
    <div className="rv-two">{state.groups.map(row => <div key={row.g} className={'rv-group ' + (row === selected ? 'rv-selected' : '')}><h4>G = {row.g}</h4><p>Group probability {number(row.mass)}</p><div className="rv-group-points">{state.rows.filter(point => point.g === row.g).map(point => <div key={point.u}><strong>({point.x}, {point.y})</strong><span>Y residual: {row.moments ? number(point.y - row.moments.y.mean) : 'undefined'}</span></div>)}</div><p>{row.moments ? 'Conditional Y mean ' + number(row.moments.y.mean) + '; variance ' + number(row.moments.y.variance) + '; covariance ' + number(row.moments.covariance) : 'Zero-probability group: conditional quantities are unidentified here.'}</p></div>)}</div>
    <div className="rv-decomposition"><h4>Total variance of Y</h4><div><span>Within {number(state.withinVariance)}</span><b>+</b><span>Between {number(state.betweenVariance)}</span><b>=</b><strong>{number(state.y.variance)}</strong></div><h4>Total covariance</h4><div><span>Within {number(state.withinCovariance)}</span><b>+</b><span>Between {number(state.betweenCovariance)}</span><b>=</b><strong>{number(state.covariance)}</strong></div></div>
    <Result>{selected.moments ? 'In the selected group, increasing U makes X rise and Y fall: conditional covariance is −1.' : 'There is no identified conditional law in the selected zero-probability group.'} Without group information the best constant Y prediction is {number(state.y.mean)} with risk {number(state.y.variance)}. Using the group mean gives average squared risk {number(state.withinVariance)}. This is an average-risk comparison, not a claim about every realized observation.</Result><Reset onClick={() => {
      setPercent(50);
      setGroup('-2');
    }} />
  </section>;
}
export function SquaredUniformLab() {
  const [lower, setLower] = useState(25),
    [upper, setUpper] = useState(81);
  const state = squaredUniformState(lower, upper),
    x = value => 150 + 115 * value,
    y = value => 35 + 230 * value;
  return <section className="rv-lab" aria-label="Squared uniform transformation investigation"><h3>One output interval has two input branches</h3><p>U is uniform on [−1,1], with density 1/2. Y=U² folds negative and positive values together. The picture shows exact analytic intervals, not observed samples.</p><div className="rv-controls"><Range label="Lower Y endpoint (%)" value={lower} onChange={value => {
        setLower(value);
        setUpper(Math.max(value, upper));
      }} min={0} max={100} /><Range label="Upper Y endpoint (%)" value={upper} onChange={value => {
        setUpper(value);
        setLower(Math.min(value, lower));
      }} min={0} max={100} /></div>
    <figure><svg viewBox="0 0 300 235" role="img" aria-label={'Y interval ' + state.lower + ' to ' + state.upper + ' receives two input intervals with total probability ' + number(state.mass)}><text x="24" y="24">U before squaring</text><line x1="35" x2="265" y1="64" y2="64" className="rv-axis" />{state.preimages.map(([a, b], i) => <line key={i} x1={x(a)} x2={x(b)} y1="64" y2="64" className="rv-interval" />)}{[-1, 0, 1].map(value => <text key={value} x={x(value)} y="88" textAnchor="middle">{value}</text>)}<path d={'M' + x(-(state.rootLower + state.rootUpper) / 2) + ' 95 Q80 127 ' + y((state.lower + state.upper) / 2) + ' 157'} className="rv-curve" /><path d={'M' + x((state.rootLower + state.rootUpper) / 2) + ' 95 Q220 127 ' + y((state.lower + state.upper) / 2) + ' 157'} className="rv-curve" /><text x="150" y="125" textAnchor="middle">square</text><line x1="35" x2="265" y1="175" y2="175" className="rv-axis" /><line x1={y(state.lower)} x2={y(state.upper)} y1="175" y2="175" className="rv-interval" />{[0, .5, 1].map(value => <text key={value} x={y(value)} y="199" textAnchor="middle">{value}</text>)}<text x="150" y="227" textAnchor="middle">Y = U²</text></svg><figcaption>The two colored U intervals feed the one colored Y interval. Their lengths carry probability at density 1/2.</figcaption></figure>
    <table><thead><tr><th>Input branch</th><th>Interval</th><th>Mass</th></tr></thead><tbody>{state.preimages.map(([a, b], i) => <tr key={i}><td>{i ? 'Positive' : 'Negative'}</td><td>[{number(a)}, {number(b)}]</td><td>{number((b - a) / 2)}</td></tr>)}</tbody></table><Result>P({number(state.lower)} ≤ Y ≤ {number(state.upper)}) = √{number(state.upper)} − √{number(state.lower)} = {number(state.mass)}. An equal-endpoint interval has probability 0, even at the density singularity Y=0.</Result><p>Moving an endpoint past the other moves both to that value; the interval remains ordered. This control uses hundredths, so it does not claim arbitrary-precision endpoint arithmetic.</p><Reset onClick={() => {
      setLower(25);
      setUpper(81);
    }} />
  </section>;
}
export function SampleMeanLawLab() {
  const [n, setN] = useState(8),
    [percent, setPercent] = useState(50);
  const state = sampleMeanState(n, percent);
  return <section className="rv-lab" aria-label="Sample mean distribution investigation"><h3>Repeat the dataset, not just the value</h3><p>These are exact probability laws for a statistic across repeated datasets. The left experiment collects independent bits. The right observes one bit and copies it n times.</p><div className="rv-controls"><Range label="Readings per dataset n" value={n} onChange={setN} min={1} max={16} /><Range label="Population success chance (%)" value={percent} onChange={setPercent} min={0} max={100} /></div><div className="rv-two"><Distribution rows={state.independent} title="Mean of independent draws" xLabel="Sample mean" extent={[-.1, 1.1]} /><Distribution rows={state.copied} title="Mean of repeated copies" xLabel="Sample mean" extent={[-.1, 1.1]} /></div><Metrics items={[["Expected sample mean, both models", state.mean], ["Independent mean variance", state.independentVariance], ["Copied mean variance", state.copiedVariance]]} /><Result>Independent variance = p(1−p)/n; copied variance = p(1−p). {n === 1 ? 'At n=1 the experiments coincide.' : 'The copied dataset contributes no new random information after its first bit.'} At p=0 or 1 both laws are deterministic.</Result><Reset onClick={() => {
      setN(8);
      setPercent(50);
    }} /></section>;
}
