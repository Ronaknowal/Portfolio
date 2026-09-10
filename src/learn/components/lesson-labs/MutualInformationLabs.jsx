import { useMemo, useState } from 'react';
import { binaryChannel, xorInformation, bottleneckRepresentation, bottleneckCurve, bottleneckIterations, variationalInformationBounds, sampledInformation, gaussianNoiseInformation, SIGNAL_INPUTS } from '../../data/mutual-information-models.js';
import './mutual-information-labs.css';
const format = value => value === null ? 'undefined' : !Number.isFinite(value) ? '∞' : (Math.abs(value) < 1e-12 ? 0 : value).toFixed(5).replace(/\.?0+$/, '');
const percent = value => `${format(100 * value)}%`;
const colors = ['#f1bc53', '#79d9ba', '#aabfff', '#ddaa90'];
function Range({
  label,
  value,
  onChange,
  min = 0,
  max = 1,
  step = 0.05
}) {
  return <label className="mi-control">{label}: <strong>{format(value)}</strong><input type="range" aria-label={label} value={value} min={min} max={max} step={step} onChange={event => onChange(Number(event.target.value))} /></label>;
}
function Readouts({
  values
}) {
  return <dl className="mi-readouts">{values.map(([name, value]) => <div key={name}><dt>{name}</dt><dd>{typeof value === 'number' ? format(value) : value}</dd></div>)}</dl>;
}
function BinaryStrip({
  probabilities,
  names = ['0', '1'],
  description
}) {
  return <div className="mi-strip-row"><span>{description}</span><div className="mi-strip" role="img" aria-label={names.map((name, i) => `${name}: ${percent(probabilities[i])}`).join('; ')}>{probabilities.map((probability, i) => <span key={i} style={{
        width: `${100 * probability}%`,
        background: colors[i % colors.length]
      }} />)}</div><span className="mi-strip-values">{names.map((name, i) => `${name}: ${percent(probabilities[i])}`).join(' · ')}</span></div>;
}
function Table({
  caption,
  headings,
  rows
}) {
  return <div className="mi-table"><h4 className="mi-table-caption">{caption}</h4><p className="mi-scroll-hint">Scroll sideways for all columns.</p><div className="mi-table-scroll" role="region" aria-label={caption} tabIndex={0}><table><caption className="mi-sr-only">{caption}</caption><thead><tr>{headings.map(heading => <th key={heading} scope="col">{heading}</th>)}</tr></thead><tbody>{rows.map((row, index) => <tr key={index}>{row.map((cell, j) => j === 0 ? <th key={j} scope="row">{cell}</th> : <td key={j}>{cell}</td>)}</tr>)}</tbody></table></div></div>;
}
export function JointInformationLab() {
  const [prevalence, setPrevalence] = useState(0.5);
  const [error, setError] = useState(0.2);
  const [selected, setSelected] = useState([0, 0]);
  const state = binaryChannel(prevalence, error);
  const cell = state.cells[selected[0]][selected[1]];
  return <section className="mi-lab" role="region" aria-label="Joint information investigation">
    <p className="mi-kicker">Observed pairing → independent reference</p>
    <h3>Same marginals, different pairings</h3>
    <p>Predict the information at flip probability ½. Then compare the actual joint with the independent table built from its own marginal probabilities.</p>
    <div className="mi-controls"><Range label="Probability of X=1" value={prevalence} onChange={setPrevalence} /><Range label="Channel flip probability" value={error} onChange={setError} /></div>
    <div className="mi-joint-pair">{['Actual joint p(x,y)', 'Independent reference p(x)p(y)'].map((title, table) => <table key={title} className="mi-joint"><caption>{title}</caption><thead><tr><th>X ↓ / Y →</th><th>0</th><th>1</th></tr></thead><tbody>{state.cells.map((row, i) => <tr key={i}><th scope="row">{i}</th>{row.map((item, j) => <td key={j} style={{
              backgroundColor: `rgba(241,188,83,${0.05 + 0.45 * (table ? item.independent : item.mass)})`
            }}>{table ? format(item.independent) : <button aria-label={`Inspect X=${i}, Y=${j}`} aria-pressed={selected[0] === i && selected[1] === j} onClick={() => setSelected([i, j])}>{format(item.mass)}</button>}</td>)}</tr>)}</tbody></table>)}</div>
    <p role="status">Selected X={cell.i}, Y={cell.j}: {cell.mass === 0 ? 'this event has zero probability; its average contribution is defined as zero, without evaluating a pointwise log ratio.' : <>the pointwise log ratio is {format(cell.information)} bits. Weighting it by probability {format(cell.mass)} gives contribution {format(cell.contribution)} bits.</>}</p>
    <div className="mi-conditional"><h4>What seeing X changes about Y</h4><BinaryStrip description="Before X is revealed" probabilities={state.pY} names={['Y0', 'Y1']} />{state.conditional.map((row, i) => row ? <BinaryStrip key={i} description={`After X=${i} · weight ${format(state.pX[i])}`} probabilities={row} names={['Y0', 'Y1']} /> : <p key={i}>X={i} has zero probability; its conditional law is not determined by this joint.</p>)}</div>
    <Readouts values={[["Before H(Y), bits", state.hy], ['Remaining H(Y|X), bits', state.conditionalEntropy], ['Information I(X;Y), bits', state.mi]]} />
    <p>Bar lengths encode conditional probabilities, not entropy. The remaining entropy averages the two conditional uncertainties using the probabilities of X.</p>
    <button onClick={() => {
      setPrevalence(0.5);
      setError(0.2);
      setSelected([0, 0]);
    }}>Reset joint investigation</button>
  </section>;
}
export function NonlinearInformationFigure() {
  return <figure className="mi-inline"><svg viewBox="0 0 320 210" role="img" aria-label="Three equally likely sensor readings: minus one and plus one both map to risk label one; zero maps to label zero. The magnitude preserves the label while the sign alone does not.">
    <text x="25" y="23">Reading X</text><text x="207" y="23">Label X²</text>
    {[-1, 0, 1].map((x, i) => <g key={x}><path d={`M80 ${58 + i * 62} L235 ${x === 0 ? 145 : 67}`} /><circle cx="55" cy={58 + i * 62} r="22" /><text x="55" y={64 + i * 62} textAnchor="middle">{x}</text></g>)}
    {[1, 0].map((y, i) => <g key={y}><circle cx="260" cy={67 + i * 78} r="24" className="mi-node-output" /><text x="260" y={73 + i * 78} textAnchor="middle">{y}</text></g>)}
  </svg><figcaption>Each reading has probability ⅓. Opposite signs can have the same label: a straight-line correlation can cancel while the input still determines the label.</figcaption></figure>;
}
export function ConditionalInformationLab() {
  const [reveal, setReveal] = useState('a');
  const [bias, setBias] = useState(0.5);
  const state = xorInformation(bias, reveal);
  const labels = reveal === 'both' ? ['A0 B0', 'A0 B1', 'A1 B0', 'A1 B1'] : [`${reveal.toUpperCase()}0`, `${reveal.toUpperCase()}1`];
  return <section className="mi-lab" role="region" aria-label="Conditional information investigation">
    <p className="mi-kicker">Individual clues → joint answer</p><h3>Two unhelpful clues can solve the problem together</h3>
    <p>A is a fair bit. The label is 1 when A and B differ. Predict the result of revealing one bit, then both.</p>
    <div className="mi-controls"><label className="mi-control">Reveal<select aria-label="Revealed feature" value={reveal} onChange={event => setReveal(event.target.value)}><option value="a">A alone</option><option value="b">B alone</option><option value="both">A and B together</option></select></label><Range label="Probability of B=1" min={0.05} max={0.95} value={bias} onChange={setBias} /></div>
    <Table caption="Every possible pairing and its label" headings={['A', 'B', 'Y=A xor B', 'Probability']} rows={state.rows.map(row => [row.a, row.b, row.y, format(row.mass)])} />
    <div className="mi-reveal-groups">{state.conditional.map((row, i) => <BinaryStrip key={i} description={`After observing ${labels[i]}`} probabilities={row} names={['Y0', 'Y1']} />)}</div>
    <Readouts values={[[`I(${reveal === 'both' ? '(A,B)' : reveal.toUpperCase()};Y), bits`, state.mi], ['I(A;Y|B), bits', state.conditionalMi]]} />
    <p role="status">{reveal === 'both' ? 'Every revealed pair fixes the label, leaving no label uncertainty.' : reveal === 'b' ? 'A remains fair even after seeing B, so Y remains fair: B alone reveals no label information.' : bias === 0.5 ? 'With B fair, each value of A leaves both labels equally likely. Revealing B as well resolves the ambiguity.' : 'Bias in B makes A partially predictive. Within either fixed B group, however, A still determines Y exactly.'}</p>
    <details><summary>Inspect the two conditional groups</summary>{state.slices.map(slice => <p key={slice.b}>B={slice.b} has weight {format(slice.weight)}. In this group I(A;Y)={format(slice.mi)} bit; the weighted average is {format(state.conditionalMi)} bit.</p>)}</details>
    <button onClick={() => {
      setReveal('a');
      setBias(0.5);
    }}>Reset conditional investigation</button>
  </section>;
}
export function ProcessingFigure() {
  return <figure className="mi-processing"><div><strong>Label Y</strong><span>What matters</span></div><span className="mi-arrow" aria-hidden="true">←</span><div><strong>Input X</strong><span>Signal + nuisance</span></div><span className="mi-arrow" aria-hidden="true">→</span><div><strong>Code Z</strong><span>Drawn using X only</span></div><figcaption>The joint is p(x,y)q(z|x). Given X, the code receives no additional label information. These arrows specify dependence in the model, not a causal intervention claim.</figcaption></figure>;
}
const representationNames = {
  constant: 'Store nothing',
  signal: 'Keep signal S',
  nuisance: 'Keep nuisance N',
  both: 'Keep both (S,N)',
  noisy: 'Noisy signal'
};
function InformationPlane({
  state,
  curve
}) {
  const x = value => 51 + value * 119;
  const y = value => 211 - value * 170;
  return <figure className="mi-chart"><svg viewBox="0 0 320 275" role="img" aria-label={`Information plane. Horizontal axis input information from 0 to 2 bits. Vertical axis label information from 0 to 1 bit. Selected representation has rate ${format(state.rate)} and relevance ${format(state.relevance)} bits. Dashed curve compares noisy-signal encoders only.`}>
    {[0, 0.5, 1].map(tick => <g key={tick}><line x1="51" x2="289" y1={y(tick)} y2={y(tick)} className="mi-grid-line" /><text x="43" y={y(tick) + 6} textAnchor="end">{tick}</text></g>)}
    {[0, 1, 2].map(tick => <text key={tick} x={x(tick)} y="237" textAnchor="middle">{tick}</text>)}
    <path d="M51 36V211H291" className="mi-axis" />
    <text x="170" y="265" textAnchor="middle">Input information (bits)</text><text x="52" y="22">Label information (bits)</text>
    <line x1="51" x2="289" y1={y(state.inputRelevance)} y2={y(state.inputRelevance)} className="mi-available" />
    <polyline points={curve.map(point => `${x(point.rate)},${y(point.relevance)}`).join(' ')} className="mi-curve" />
    {['constant', 'signal', 'nuisance', 'both'].map((mode, index) => {
        const point = bottleneckRepresentation(mode, 0, state.labelError, state.beta);
        return <circle key={mode} cx={x(point.rate)} cy={y(point.relevance)} r="5" fill={colors[index]} />;
      })}
    <circle cx={x(state.rate)} cy={y(state.relevance)} r="9" className="mi-selected-dot" />
  </svg><figcaption>The ring is your current choice. The dashed curve is the declared noisy-signal family, not every possible encoder. The horizontal green line is the information available in X about Y.</figcaption></figure>;
}
export function InformationBottleneckLab() {
  const [mode, setMode] = useState('signal');
  const [noise, setNoise] = useState(0.2);
  const [beta, setBeta] = useState(3);
  const state = bottleneckRepresentation(mode, noise, 0.1, beta);
  const curve = bottleneckCurve(0.1, beta);
  return <section className="mi-lab" role="region" aria-label="Information bottleneck representation investigation">
    <p className="mi-kicker">Information retained → task information retained</p><h3>Which distinctions deserve space in the code?</h3>
    <p>X contains two independent fair bits: signal S and nuisance N. The label equals S with probability .9. Predict what retaining N changes.</p>
    <div className="mi-controls"><label className="mi-control">Representation<select aria-label="Representation choice" value={mode} onChange={event => setMode(event.target.value)}>{Object.entries(representationNames).map(([value, name]) => <option key={value} value={value}>{name}</option>)}</select></label><Range label="Relevance weight beta" value={beta} min={0} max={8} step={0.5} onChange={setBeta} />{mode === 'noisy' && <Range label="Encoder flip probability" value={noise} min={0} max={0.5} step={0.01} onChange={setNoise} />}</div>
    <div className="mi-plane-layout"><div><h4>Code probabilities for the same four inputs</h4>{state.encoder.map((row, i) => <BinaryStrip key={i} description={SIGNAL_INPUTS[i]} probabilities={row} names={row.map((_, z) => `Z${z}`)} />)}</div><InformationPlane state={state} curve={curve} /></div>
    <Readouts values={[["Rate I(X;Z), bits", state.rate], ['Relevance I(Y;Z), bits', state.relevance], ['J = rate − beta × relevance', state.objective]]} />
    <p role="status">{mode === 'both' ? 'Both bits preserve the same label information as S alone, but retaining the independent nuisance adds one input bit to the rate.' : mode === 'nuisance' ? 'The code distinguishes inputs, but the distinction tells you nothing about this label.' : mode === 'constant' ? 'Ignoring X attains zero rate and zero relevance. At beta≤1 this is an optimum by the data-processing bound.' : mode === 'signal' ? 'S retains all the label information available in X while removing the independent nuisance. It still cannot remove the label’s own noise.' : 'Randomizing S reduces both rate and relevance. The best tradeoff depends on beta; these axes are information, not the number of stored coordinates.'}</p>
    <Table caption="Four exact deterministic choices at the current beta" headings={['Choice', 'Rate', 'Relevance', 'J']} rows={['constant', 'signal', 'nuisance', 'both'].map(value => {
      const row = bottleneckRepresentation(value, 0, 0.1, beta);
      return [representationNames[value], format(row.rate), format(row.relevance), format(row.objective)];
    })} />
    <button onClick={() => {
      setMode('signal');
      setNoise(0.2);
      setBeta(3);
    }}>Reset representation investigation</button>
  </section>;
}
export function BottleneckIterationLab() {
  const [beta, setBeta] = useState(3);
  const [initialization, setInitialization] = useState('signal');
  const [step, setStep] = useState(0);
  const trace = bottleneckIterations(beta, 40, initialization);
  const state = trace.rows[step];
  return <section className="mi-lab" role="region" aria-label="Bottleneck update investigation">
    <p className="mi-kicker">Soft assignments → representative predictions → new assignments</p><h3>Watch inputs group by their label distributions</h3>
    <p>Each strip assigns one input to two codewords. A complete update recalculates assignments, codeword probabilities and label predictions. Predict which inputs will become indistinguishable.</p>
    <div className="mi-controls"><label className="mi-control">Starting encoder<select aria-label="Starting encoder" value={initialization} onChange={event => {
          setInitialization(event.target.value);
          setStep(0);
        }}><option value="signal">Some signal already visible</option><option value="symmetric">Exactly symmetric assignments</option><option value="nuisance">Initially follows nuisance</option></select></label><Range label="Update relevance weight beta" min={0} max={8} step={0.5} value={beta} onChange={value => {
        setBeta(value);
        setStep(0);
      }} /></div>
    <div className="mi-update-layout"><div><h4>q(Z|X): current assignment</h4>{state.encoder.map((row, i) => <BinaryStrip key={i} description={SIGNAL_INPUTS[i]} probabilities={row} names={['Z0', 'Z1']} />)}</div><div className="mi-prediction-side"><h4>What each codeword predicts</h4>{state.decoder.map((row, z) => <BinaryStrip key={z} description={`Z${z} · probability ${format(state.pZ[z])}`} probabilities={row} names={['Y0', 'Y1']} />)}<p>Inputs sharing the same label distribution can merge despite different nuisance values.</p></div></div>
    <div className="mi-step-controls"><button disabled={step === 0} onClick={() => setStep(value => value - 1)}>Previous update</button><strong>Update {step} / 40</strong><button disabled={step === 40} onClick={() => setStep(value => value + 1)}>Next complete update</button><button disabled={step === 40} onClick={() => setStep(40)}>Inspect update 40</button></div>
    <Readouts values={[["Rate I(X;Z), bits", state.rate], ['Relevance I(Y;Z), bits', state.relevance], ['Objective J', state.objective]]} />
    <p role="status">{initialization === 'symmetric' ? 'Symmetric codewords predict the same label law, so the update has no reason to separate them. A stationary encoder can be worse than an informative initialization.' : initialization === 'nuisance' && step > 0 ? 'Nuisance-based groups have the same label distribution. This initialization collapses instead of discovering the useful split.' : step === 0 ? 'This is the declared starting encoder. The objective below the strips is calculated from its complete joint law.' : `This exact finite update changed J from ${format(trace.rows[step - 1].objective)} to ${format(state.objective)}. Descent does not certify a global optimum.`}</p>
    <details><summary>Inspect the predictive KL values used by this update</summary>{state.previousDistortion ? <Table caption="Previous representative mismatch, in bits" headings={['Input', 'To Z0', 'To Z1']} rows={state.previousDistortion.map((row, x) => [SIGNAL_INPUTS[x], ...row.map(format)])} /> : <p>No update has occurred yet. Advance once to inspect the preceding representative mismatch.</p>}<p>The update exponent uses the corresponding KL in nats. Multiplying these displayed bit values by ln 2 converts them; the beta convention is unchanged.</p></details>
    <button onClick={() => {
      setBeta(3);
      setInitialization('signal');
      setStep(0);
    }}>Reset bottleneck updates</button>
  </section>;
}
export function VariationalInformationLab() {
  const [reference, setReference] = useState(0.5);
  const [decoder, setDecoder] = useState(0.26);
  const state = variationalInformationBounds(reference, decoder, 3);
  const minimum = Math.min(-0.25, state.predictiveLower - 0.15),
    maximum = Math.max(1, state.rateUpper + 0.15);
  const x = value => 40 + 246 * (value - minimum) / (maximum - minimum);
  const ticks = [minimum, ...(x(0) - x(minimum) > 50 && x(maximum) - x(0) > 50 ? [0] : []), maximum];
  return <section className="mi-lab" role="region" aria-label="Variational information bound investigation">
    <p className="mi-kicker">Fixed representation → adjustable bound</p><h3>A looser bound can move while the information stays fixed</h3>
    <p>The encoder still flips S with probability .2. Only the reference distribution and decoder change. Predict whether actual rate and relevance should move.</p>
    <div className="mi-controls"><Range label="Reference probability of Z=1" value={reference} min={0.02} max={0.98} step={0.02} onChange={setReference} /><Range label="Decoder flipped-label probability" value={decoder} min={0.02} max={0.98} step={0.02} onChange={setDecoder} /></div>
    <figure className="mi-chart mi-bound-chart"><svg viewBox="0 0 320 225" role="img" aria-label={`Exact rate ${format(state.rate)}, upper bound ${format(state.rateUpper)}. Exact relevance ${format(state.relevance)}, lower bound ${format(state.predictiveLower)}. All values in bits.`}>
      <text x="25" y="24">Rate: upper bound</text><line x1={x(state.rate)} x2={x(state.rateUpper)} y1="68" y2="68" className="mi-bound-gap" /><circle cx={x(state.rate)} cy="68" r="7" className="mi-exact-dot" /><rect x={x(state.rateUpper) - 6} y="62" width="12" height="12" className="mi-bound-dot" />
      <text x="25" y="117">Relevance: lower bound</text><line x1={x(state.predictiveLower)} x2={x(state.relevance)} y1="153" y2="153" className="mi-bound-gap" /><circle cx={x(state.relevance)} cy="153" r="7" className="mi-exact-dot" /><rect x={x(state.predictiveLower) - 6} y="147" width="12" height="12" className="mi-bound-dot" />
      <path d="M40 185H287" className="mi-axis" />{ticks.map((value, i) => <g key={i}><line x1={x(value)} x2={x(value)} y1="181" y2="190" className="mi-axis" /><text x={x(value)} y="215" textAnchor="middle">{value.toFixed(1)}</text></g>)}
    </svg><figcaption>Green circles: exact information. Amber squares: variational bounds. The shared axis rescales to include the selected bounds; read the values below.</figcaption></figure>
    <Table caption="Exact quantities and separately calculated gaps, in bits" headings={['Quantity', 'Exact information', 'Bound', 'Gap']} rows={[["Input information", format(state.rate), format(state.rateUpper), format(state.rateGap)], ['Label information', format(state.relevance), format(state.predictiveLower), format(state.predictiveGap)]]} />
    <Readouts values={[["True J at beta = 3", state.objective], ['Surrogate upper bound on J', state.objectiveUpper]]} />
    <p role="status">{state.predictiveLower < 0 ? 'This decoder gives a negative lower bound. Actual mutual information remains nonnegative; the poor bound does not prove that the representation has no information.' : Math.abs(state.rateGap) + Math.abs(state.predictiveGap) < 1e-10 ? 'Both approximations match their true targets, so the bounds meet the exact information.' : 'The rate gap is KL(p(Z)||r(Z)); the predictive gap is the average KL from the true conditional label law to the chosen decoder.'}</p>
    <button onClick={() => {
      setReference(0.5);
      setDecoder(0.26);
    }}>Match both true distributions</button>
    <button onClick={() => {
      setReference(0.8);
      setDecoder(0.4);
    }}>Try mismatched approximations</button>
  </section>;
}
export function GaussianInformationFigure() {
  const points = Array.from({
    length: 100
  }, (_, i) => {
    const sigma = 0.05 + i * 1.95 / 99;
    return [sigma, gaussianNoiseInformation(sigma)];
  });
  const x = sigma => 53 + 119 * sigma,
    y = information => 213 - 39 * information;
  return <figure className="mi-inline mi-chart"><svg viewBox="0 0 320 280" role="img" aria-label="For X standard Normal and Z=X+sigma times independent standard Normal noise, information falls as noise increases: sigma .25 gives 2.04373 bits, sigma .5 gives 1.16096, sigma 1 gives .5, sigma 2 gives .16096. The zero-noise case is infinite and is not a finite point on this plot.">
    <text x="53" y="23">Information (bits)</text>{[0, 2, 4].map(tick => <g key={tick}><line x1="53" x2="291" y1={y(tick)} y2={y(tick)} className="mi-grid-line" /><text x="43" y={y(tick) + 6} textAnchor="end">{tick}</text></g>)}<path d="M53 35V213H293" className="mi-axis" /><polyline points={points.map(point => `${x(point[0])},${y(point[1])}`).join(' ')} className="mi-curve" />{[0.5, 1, 2].map(sigma => <g key={sigma}><circle cx={x(sigma)} cy={y(gaussianNoiseInformation(sigma))} r="5" className="mi-exact-dot" /><text x={x(sigma)} y="241" textAnchor="middle">{sigma}</text></g>)}<text x="173" y="272" textAnchor="middle">Noise standard deviation σ</text>
  </svg><figcaption>Calculated from ½ log₂(1+1/σ²), with independent Gaussian noise and input variance 1. This is a declared mathematical model. At σ=0 the joint is singular and MI is infinite; no finite plotted value represents that limit.</figcaption></figure>;
}
export function InformationEstimationLab() {
  const [categories, setCategories] = useState(4);
  const [count, setCount] = useState(100);
  const [mode, setMode] = useState('independent');
  const [seed, setSeed] = useState(831);
  const [draft, setDraft] = useState('831');
  const [error, setError] = useState('');
  const state = useMemo(() => sampledInformation({
    categories,
    count,
    mode,
    seed
  }), [categories, count, mode, seed]);
  const maxInformation = Math.log2(categories);
  const bins = Array.from({
    length: 12
  }, () => 0);
  state.shuffled.forEach(value => {
    bins[Math.min(11, Math.max(0, Math.floor(12 * value / maxInformation)))] += 1;
  });
  const chartX = value => 48 + value / maxInformation * 245;
  return <section className="mi-lab" role="region" aria-label="Mutual information estimation investigation">
    <p className="mi-kicker">Known population → finite observations</p><h3>Sparse counts can invent an apparent relationship</h3>
    <p>Start with independently drawn X and Y. Their population MI is exactly 0. Predict what happens when you spread the same small sample over more categories.</p>
    <div className="mi-controls"><label className="mi-control">Population law<select aria-label="Population law" value={mode} onChange={event => setMode(event.target.value)}><option value="independent">Independent categories</option><option value="channel">20% symmetric channel errors</option></select></label><label className="mi-control">Categories per variable<select aria-label="Categories per variable" value={categories} onChange={event => setCategories(Number(event.target.value))}>{[2, 4, 8].map(value => <option key={value}>{value}</option>)}</select></label><label className="mi-control">Observation count<select aria-label="Observation count" value={count} onChange={event => setCount(Number(event.target.value))}>{[20, 100, 500, 2000].map(value => <option key={value}>{value}</option>)}</select></label><form onSubmit={event => {
        event.preventDefault();
        const parsed = Number(draft);
        if (!Number.isInteger(parsed) || parsed < 1 || parsed > 2147483647) {
          setError('Use an integer seed from 1 to 2147483647. The displayed sample is unchanged.');
          return;
        }
        setSeed(parsed);
        setError('');
      }}><label className="mi-control">Seed<input aria-label="Sampling seed" inputMode="numeric" value={draft} onChange={event => setDraft(event.target.value)} /></label><button type="submit">Generate this sample</button></form></div>
    {error && <p role="alert">{error}</p>}
    <h4 className="mi-table-caption">Observed counts · active seed {seed}</h4><p className="mi-scroll-hint">Scroll sideways for all columns.</p>
    <div className="mi-counts" role="region" aria-label="Observed pair counts" tabIndex={0}><table><caption className="mi-sr-only">Observed counts · active seed {seed}</caption><thead><tr><th>X↓ / Y→</th>{state.pY.map((_, y) => <th key={y}>{y}</th>)}</tr></thead><tbody>{state.counts.map((row, x) => <tr key={x}><th scope="row">{x}</th>{row.map((value, y) => <td key={y} style={{
              background: `rgba(241,188,83,${0.04 + 0.48 * value / Math.max(1, ...state.counts.flat())})`
            }}>{value}</td>)}</tr>)}</tbody></table></div>
    <Readouts values={[["Plug-in MI, bits", state.mi], ['Known population MI, bits', state.populationMi], ['Median after reshuffling pairs', state.shuffleMedian]]} />
    <figure className="mi-chart"><svg viewBox="0 0 320 245" role="img" aria-label={`Twenty reshuffled pairing estimates range from ${format(state.shuffleMinimum)} to ${format(state.shuffleMaximum)} bits. The original estimate is ${format(state.mi)} bits.`}>
      <text x="47" y="24">20 reshuffled pairings</text>{[0, 10, 20].map(value => <text key={value} x="38" y={181 - value * 7} textAnchor="end">{value}</text>)}{bins.map((value, bin) => <rect key={bin} x={48 + bin * 245 / 12} y={175 - value * 7} width={245 / 12 - 2} height={value * 7} className="mi-null-bin" />)}<line x1={chartX(state.mi)} x2={chartX(state.mi)} y1="40" y2="179" className="mi-observed" /><path d="M48 35V175H294" className="mi-axis" />{[0, maxInformation / 2, maxInformation].map(value => <text key={value} x={chartX(value)} y="205" textAnchor="middle">{format(value)}</text>)}<text x="173" y="236" textAnchor="middle">Estimated MI (bits)</text>
    </svg><figcaption>Bars count estimates after reshuffling Y among the same observations. The amber line is the original pairing. Reshuffling preserves the empirical marginal counts; these 20 comparisons are a diagnostic, not a confidence interval or a calibrated test reported here.</figcaption></figure>
    <p role="status">{mode === 'independent' ? 'Any positive estimate in this sample describes accidental empirical association, because the declared population law is independent.' : 'The declared channel has real information, but this empirical value can differ in either direction. More observations do not make every individual run closer.'}</p>
    <button onClick={() => {
      setCategories(4);
      setCount(100);
      setMode('independent');
      setSeed(831);
      setDraft('831');
      setError('');
    }}>Reset estimation investigation</button>
  </section>;
}
