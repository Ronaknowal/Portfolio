import { useId, useMemo, useState } from 'react';
import { LessonTable } from './LessonElements.jsx';
import { BINARY_CODEBOOKS, RATE_DISTORTION_SCENARIOS, binaryBlockCode, binaryOptimalChannel, binaryRateDistortion, finiteRateDistortion, gaussianAllocation, zeroRateFidelity } from '../../data/rate-distortion-models.js';
import './rate-distortion-labs.css';
const amber = '#e9bb67';
const blue = '#96c3e8';
const green = '#a1d6b8';
function number(value, digits = 4) {
  if (!Number.isFinite(value)) return value === Infinity ? '∞' : 'undefined';
  if (value !== 0 && Math.abs(value) < 0.0001) return value.toExponential(2);
  return Number(value.toFixed(digits)).toString();
}
function Slider({
  label,
  value,
  setValue,
  min = 0,
  max = 1,
  step = 0.05
}) {
  const id = useId();
  return <label className="rate-distortion-control" htmlFor={id}>
    <span>{label}: <strong>{number(value)}</strong></span>
    <input id={id} aria-label={label} type="range" min={min} max={max} step={step} value={value} onChange={event => setValue(Number(event.target.value))} />
  </label>;
}
function Investigation({
  id,
  title,
  guidance,
  reset,
  children
}) {
  const heading = useId();
  return <section className="rate-distortion-lab lesson-lab" data-rate-distortion-lab={id} aria-labelledby={heading}>
    <h3 id={heading}>{title}</h3><p>{guidance}</p>
    {children}<button type="button" className="rate-distortion-reset" onClick={reset}>Reset investigation</button>
  </section>;
}
function Metrics({
  values
}) {
  return <dl className="rate-distortion-metrics">{values.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>;
}
function Bits({
  word,
  original = null
}) {
  return <span className="rate-distortion-bits" aria-label={word.split('').map((bit, index) => original && bit !== original[index] ? bit + ' changed' : bit).join(', ')}>
    {word.split('').map((bit, index) => <span key={index} className={original && bit !== original[index] ? 'is-changed' : ''} aria-hidden="true">{bit}{original && bit !== original[index] && <small>×</small>}</span>)}
  </span>;
}
export function CompressionChannelFigure() {
  return <figure className="rate-distortion-inline">
    <figcaption>Only the index crosses the communication boundary.</figcaption>
    <ol className="rate-distortion-pipeline">
      <li><strong>Source block</strong><Bits word="011" /><span>Three original bits</span></li>
      <li><strong>Encoder</strong><span>Compare 000 and 111.</span><span>111 changes one position.</span></li>
      <li className="rate-distortion-wire"><strong>Send index 1</strong><Bits word="1" /><span>One transmitted bit</span></li>
      <li><strong>Decoder</strong><Bits word="111" original="011" /><span>Shared rule: index1 →111</span></li>
    </ol>
    <p>The decoder has the codebook, not the original. The × identifies the changed bit. Numbered stages describe processing order, not timing or measured bandwidth.</p>
  </figure>;
}
export function BinaryCodebookLab() {
  const [probability, setProbability] = useState(0.5);
  const [kind, setKind] = useState('majority');
  const [selected, setSelected] = useState(3);
  const state = useMemo(() => binaryBlockCode({
    probability,
    codewords: BINARY_CODEBOOKS[kind].words,
    selected
  }), [probability, kind, selected]);
  const chosen = state.selected;
  return <Investigation id="codebook" title="Build the actual reconstruction" guidance="With only 000 and 111 in the codebook, which three-bit inputs can be reconstructed exactly?" reset={() => {
    setProbability(0.5);
    setKind('majority');
    setSelected(3);
  }}>
    <div className="rate-distortion-controls">
      <label className="rate-distortion-control">Shared codebook<select aria-label="Shared codebook" value={kind} onChange={event => setKind(event.target.value)}>{Object.entries(BINARY_CODEBOOKS).map(([key, entry]) => <option key={key} value={key}>{entry.label}</option>)}</select></label>
      <Slider label="Probability of source bit 1" value={probability} setValue={setProbability} />
    </div>
    <div className="rate-distortion-codewords" aria-label="Selectable original blocks">{state.rows.map(row => <button type="button" key={row.input} aria-label={'Encode block ' + row.inputBits} aria-pressed={selected === row.input} onClick={() => setSelected(row.input)}>{row.inputBits}</button>)}</div>
    <div className="rate-distortion-transmission">
      <div><span>Original</span><Bits word={chosen.inputBits} /></div>
      <div className="rate-distortion-wire"><span>Transmitted index</span><strong>{state.bitsPerBlock === 0 ? 'no bits' : chosen.index.toString(2).padStart(state.bitsPerBlock, '0')}</strong></div>
      <div><span>Reconstructed</span><Bits word={chosen.reconstructionBits} original={chosen.inputBits} /></div>
    </div>
    <p className="rate-distortion-observation">This input has probability <strong>{number(chosen.mass)}</strong>. Index {chosen.index} selects {chosen.reconstructionBits}: <strong>{chosen.errors} of 3 positions change</strong>. {chosen.mass === 0 && 'This block has zero probability under the selected source law; its mapping is still shown.'} Equal-distance ties choose the first codebook entry.</p>
    <Metrics values={[['Fixed bits / original bit', number(state.rate, 6)], ['Average wrong-bit fraction', number(state.distortion, 6)], ['Worst supported block fraction', number(state.worstDistortion, 6)], ['Shannon bound at this average', number(state.lowerBound, 6)], ['Index entropy / block', number(state.indexEntropy, 6)]]} />
    <LessonTable caption="All possible inputs: the average weights each block by its source probability." headers={['Input', 'Probability', 'Index', 'Output', 'Wrong / 3']} rows={state.rows.map(row => [row.inputBits, number(row.mass, 5), row.index, row.reconstructionBits, row.errors + ' / 3'])} />
    <p>The displayed fixed rate pays for the declared codebook even if some indices are unused. Entropy coding the indices across many blocks can change their average length. No headers, transport framing or codebook download are counted in this payload rate.</p>
    <p><strong>Transfer.</strong> Choose probability .2 and the one-word codebook. Explain why zero transmitted bits now incur average error .2, while an individual possible block can still be completely wrong.</p>
  </Investigation>;
}
function BinaryCurve({
  probability,
  budget,
  weight
}) {
  const left = 48,
    top = 40,
    width = 230,
    height = 220;
  const x = value => left + value * width;
  const y = value => top + (1 - value) * height;
  const points = Array.from({
    length: 201
  }, (_, index) => {
    const distortion = index / 200;
    return x(distortion) + ',' + y(binaryRateDistortion(probability, distortion));
  }).join(' ');
  const rate = binaryRateDistortion(probability, budget);
  const weightedDistortion = Math.min(probability, 1 - probability, 1 / (1 + 2 ** weight));
  const weightedRate = binaryRateDistortion(probability, weightedDistortion);
  const finite = binaryBlockCode({
    probability
  });
  return <figure className="rate-distortion-figure"><figcaption>Calculated binary-source limit and one actual three-bit code.</figcaption>
    <svg viewBox="0 0 320 330" role="img" aria-label={'Rate-distortion curve for probability ' + number(probability) + '. Minimum rate is zero for every allowed distortion at or above ' + number(Math.min(probability, 1 - probability)) + '.'}>
      {[0, 0.5, 1].map(value => <g key={value}><line x1={left} x2={left + width} y1={y(value)} y2={y(value)} stroke="#3b4850" /><text x={left - 12} y={y(value) + 6} textAnchor="end">{value}</text><text x={x(value)} y="286" textAnchor="middle">{value}</text></g>)}
      <text x="48" y="22">Rate · bits / source bit</text><line x1={left} x2={left} y1={top} y2={top + height} stroke="#a9b7bf" />
      <polyline points={points} fill="none" stroke={amber} strokeWidth="3" />
      <line x1={x(budget)} x2={x(budget)} y1={y(0)} y2={y(rate)} stroke={amber} strokeDasharray="4 4" />
      <circle cx={x(budget)} cy={y(rate)} r="6" fill={amber} stroke="#0b1115" strokeWidth="2" />
      <path d={`M${x(weightedDistortion)},${y(weightedRate) - 7} l-7,13 h14 Z`} fill={green} stroke="#0b1115" />
      <path d={`M${x(finite.distortion) - 5},${y(finite.rate) - 5} l10,10 m0,-10 l-10,10`} stroke={blue} strokeWidth="3" />
      <text x="164" y="317" textAnchor="middle">Allowed error fraction D</text>
    </svg>
    <ul className="rate-distortion-legend"><li><span style={{
          color: amber
        }}>●</span> Selected budget: ({number(budget)}, {number(rate)})</li><li><span style={{
          color: green
        }}>▲</span> Minimum of R + λD: ({number(weightedDistortion)}, {number(weightedRate)})</li><li><span style={{
          color: blue
        }}>×</span> 000/111 code: ({number(finite.distortion)}, {number(finite.rate)})</li></ul>
    <p>No measured codec timings or empirical performance are represented. The curve assumes iid binary symbols and average Hamming distortion.</p>
  </figure>;
}
export function BinaryFrontierLab() {
  const [probability, setProbability] = useState(0.5);
  const [budget, setBudget] = useState(0.1);
  const [weight, setWeight] = useState(2);
  const state = useMemo(() => binaryOptimalChannel(probability, budget), [probability, budget]);
  return <Investigation id="binary-frontier" title="Read the limit, then inspect its probability law" guidance="Once a constant reconstruction meets the error budget, could allowing still more error require more bits?" reset={() => {
    setProbability(0.5);
    setBudget(0.1);
    setWeight(2);
  }}>
    <div className="rate-distortion-controls"><Slider label="Binary source probability p" value={probability} setValue={setProbability} /><Slider label="Allowed error fraction D" value={budget} setValue={setBudget} step={0.025} /><Slider label="Distortion penalty lambda" value={weight} setValue={setWeight} min={0} max={8} step={0.5} /></div>
    <BinaryCurve probability={probability} budget={budget} weight={weight} />
    <Metrics values={[['Required information', number(state.rate, 6) + ' bits'], ['Zero-rate threshold', number(state.threshold)], ['Actual error in shown law', number(state.distortion)], ['Output probability of 1', number(state.reproductionOne)]]} />
    <LessonTable caption="Joint probabilities of original and reconstruction: rows keep the source law fixed." headers={['Original', 'Reconstruct 0', 'Reconstruct 1', 'Row total']} rows={state.joint.map((row, index) => [index, number(row[0], 6), number(row[1], 6), number(state.source[index])])} />
    <p className="rate-distortion-observation">False positive P(output1 | input0): <strong>{state.conditional[0] ? number(state.conditional[0][1], 6) : 'undefined: input0 has probability zero'}</strong>. False negative P(output0 | input1): <strong>{state.conditional[1] ? number(state.conditional[1][0], 6) : 'undefined: input1 has probability zero'}</strong>.</p>
    <p>{budget >= state.threshold ? 'A most-probable constant output already meets this budget. The shown law uses that constant and need not spend the whole allowed error budget.' : 'The backward construction makes the error independent of the reconstruction. The two forward error probabilities generally differ for a biased source.'} Changing λ moves the green operating point; it does not change the source or the amber budget.</p>
    <button type="button" onClick={() => {
      setProbability(0.2);
      setBudget(0.1);
    }}>Inspect biased p=.2, D=.1</button>
    <p><strong>Transfer.</strong> With p=.2, move D past .2 and then to .8. The curve stays at zero. Explain which feasible reconstruction proves that it must.</p>
  </Investigation>;
}
export function RateDistortionOptimizerLab() {
  const [kind, setKind] = useState('levels');
  const [weight, setWeight] = useState(0.5);
  const [start, setStart] = useState('positive');
  const [step, setStep] = useState(0);
  const [selected, setSelected] = useState([2, 0]);
  const scenario = RATE_DISTORTION_SCENARIOS[kind];
  const result = useMemo(() => finiteRateDistortion({
    ...scenario,
    lambda: weight,
    initial: start === 'positive' ? null : scenario.source.map((_, index) => index === 0 ? 1 : 0),
    iterations: 120
  }), [scenario, weight, start]);
  const index = Math.min(step, result.trace.length - 1);
  const state = result.trace[index];
  const sourceIndex = Math.min(selected[0], scenario.source.length - 1);
  const outputIndex = Math.min(selected[1], scenario.source.length - 1);
  const jointMass = state.source[sourceIndex] * state.conditional[sourceIndex][outputIndex];
  const reset = () => {
    setKind('levels');
    setWeight(0.5);
    setStart('positive');
    setStep(0);
    setSelected([2, 0]);
  };
  return <Investigation id="finite-optimizer" title="Let a reconstruction alphabet earn its probability" guidance="If missing a high source level becomes ten times as costly, which output probabilities should change?" reset={reset}>
    <div className="rate-distortion-controls">
      <label className="rate-distortion-control">Source and distortion rule<select aria-label="Source and distortion rule" value={kind} onChange={event => {
          setKind(event.target.value);
          setStep(0);
        }}>{Object.entries(RATE_DISTORTION_SCENARIOS).map(([key, entry]) => <option value={key} key={key}>{entry.label}</option>)}</select></label>
      <Slider label="Optimizer distortion weight" value={weight} setValue={value => {
        setWeight(value);
        setStep(0);
      }} min={0} max={5} step={0.1} />
      <label className="rate-distortion-control">Initial output support<select aria-label="Initial output support" value={start} onChange={event => {
          setStart(event.target.value);
          setStep(0);
        }}><option value="positive">Positive probability for every symbol</option><option value="missing">Only output 0 is available</option></select></label>
    </div>
    <LessonTable caption={'Declared distortion costs; source probabilities ' + scenario.source.join(', ') + '. Unit: ' + scenario.unit + '.'} headers={['Original / output', ...scenario.labels]} rows={scenario.costs.map((row, i) => [scenario.labels[i], ...row])} />
    <div className="rate-distortion-step-controls"><button type="button" onClick={() => setStep(index - 1)} disabled={index === 0}>Previous update</button><span>Update {state.iteration}</span><button type="button" onClick={() => setStep(index + 1)} disabled={index === result.trace.length - 1}>Next update</button><button type="button" onClick={() => setStep(result.trace.length - 1)} disabled={index === result.trace.length - 1}>Inspect bounded run</button></div>
    <div className="rate-distortion-matrix" role="region" tabIndex={0} aria-label="Conditional reconstruction matrix">
      <table><caption>q(output | original): every row sums to one. Select a cell to follow its error contribution.</caption><thead><tr><th scope="col">Original / output</th>{scenario.labels.map(label => <th scope="col" key={label}>{label}</th>)}</tr></thead><tbody>
        {state.conditional.map((row, i) => <tr key={i}><th scope="row">{scenario.labels[i]}</th>{row.map((value, j) => <td key={j}><button type="button" aria-pressed={sourceIndex === i && outputIndex === j} aria-label={'Inspect original ' + scenario.labels[i] + ' output ' + scenario.labels[j]} onClick={() => setSelected([i, j])}><strong>{number(value)}</strong><span className="rate-distortion-probability-track" aria-hidden="true"><span style={{
                    width: value * 100 + '%'
                  }} /></span></button></td>)}</tr>)}
      </tbody></table>
    </div>
    <p className="rate-distortion-observation">Selected pair: p({scenario.labels[sourceIndex]}) {number(state.source[sourceIndex])} × q {number(state.conditional[sourceIndex][outputIndex])} = <strong>{number(jointMass)} joint probability</strong>. Multiply by cost {state.costs[sourceIndex][outputIndex]} to contribute <strong>{number(jointMass * state.costs[sourceIndex][outputIndex])}</strong> to D.</p>
    <div className="rate-distortion-output-bars">{scenario.labels.map((label, j) => <div key={label}><strong>Output {label}</strong><span>Guess used: {number(state.outputGuess[j], 6)}</span><div className="rate-distortion-bar-track"><span style={{
            width: state.outputGuess[j] * 100 + '%',
            background: blue
          }} /></div><span>Marginal for next update: {number(state.output[j], 6)}</span><div className="rate-distortion-bar-track"><span style={{
            width: state.output[j] * 100 + '%',
            background: amber
          }} /></div></div>)}</div>
    <Metrics values={[['Expected distortion D', number(state.distortion, 6)], ['Information I · bits', number(state.information, 6)], ['Lower objective bound', number(state.lower, 6)], ['Achieved I + λD', number(state.upper, 6)], ['Objective gap bound', number(state.gap, 6)]]} />
    <p>{state.converged ? 'The computed objective gap is within 10⁻⁹ bits for this finite problem.' : 'The objective is not yet certified within 10⁻⁹ bits.'} {index === 120 && !state.converged && 'This investigation stops at 120 updates; the complete Python solver permits a larger bounded run.'} {state.supportLimited && 'A zero starting probability remains zero. Reset to positive support to let a missing reconstruction participate.'}</p>
    <p>The upper value belongs to this actual conditional table. The lower value considers the entire declared alphabet, including excluded columns; it can be negative and loose. The difference bounds objective error, not a separate error bound on I or D. The row probabilities are a theoretical test channel, not a serialized compressor.</p>
    <p><strong>Transfer.</strong> Compare the ordinary and weighted three-level cases at λ=.1. Use the same source probabilities. Identify the high-level error cost that the first objective had treated too cheaply.</p>
  </Investigation>;
}
export function GaussianAllocationLab() {
  const choices = {
    unequal: [9, 1],
    equal: [4, 4],
    small: [0.25, 4]
  };
  const [kind, setKind] = useState('unequal');
  const [fraction, setFraction] = useState(0.3);
  const variances = choices[kind];
  const total = variances.reduce((sum, value) => sum + value, 0);
  const state = useMemo(() => gaussianAllocation(variances, fraction * total), [kind, fraction]);
  const maximum = Math.max(...variances);
  const y = value => 245 - value / maximum * 170;
  return <Investigation id="gaussian-allocation" title="Fill the error budget across components" guidance="With variances 9 and 1 and total allowed squared error 3, should the quieter component receive more than its entire variance?" reset={() => {
    setKind('unequal');
    setFraction(0.3);
  }}>
    <div className="rate-distortion-controls"><label className="rate-distortion-control">Gaussian component variances<select aria-label="Gaussian component variances" value={kind} onChange={event => setKind(event.target.value)}><option value="unequal">9 and 1</option><option value="equal">4 and 4</option><option value="small">.25 and 4</option></select></label><Slider label="Total squared-error budget" value={fraction * total} setValue={value => setFraction(value / total)} min={0} max={total * 1.1} step={total / 100} /></div>
    <figure className="rate-distortion-figure"><figcaption>Each reservoir height is the source variance. Amber is allocated error; blue is variance retained by the optimal Gaussian reconstruction.</figcaption>
      <svg viewBox="0 0 320 320" role="img" aria-label={'Distortion allocations ' + state.components.map(component => number(component.distortion) + ' of variance ' + number(component.variance)).join(' and ') + ', common level ' + number(state.level)}>
        <text x="20" y="25">Squared units</text>
        {[0, maximum / 2, maximum].map(value => <g key={value}><line x1="46" x2="290" y1={y(value)} y2={y(value)} stroke="#3b4850" /><text x="37" y={y(value) + 6} textAnchor="end">{number(value, 2)}</text></g>)}
        {state.components.map((component, index) => <g key={index}><rect x={75 + index * 125} y={y(component.variance)} width="60" height={245 - y(component.variance)} fill="#2b4e68" stroke={blue} strokeWidth="2" /><rect x={75 + index * 125} y={y(component.distortion)} width="60" height={245 - y(component.distortion)} fill={amber} /><text x={105 + index * 125} y="275" textAnchor="middle">Part {index + 1}</text><text x={105 + index * 125} y="300" textAnchor="middle">V = {number(component.variance)}</text></g>)}
        <line x1="46" x2="290" y1={y(state.level)} y2={y(state.level)} stroke={green} strokeWidth="2" strokeDasharray="5 4" />
        <text x="290" y="53" textAnchor="end" fill={green}>Level θ = {number(state.level, 3)}</text>
      </svg>
    </figure>
    <LessonTable caption="Calculated independent Gaussian model; total rate is per two-component vector." headers={['Component', 'Variance', 'Error', 'Rate · bits']} rows={state.components.map(component => [component.index + 1, number(component.variance), number(component.distortion), number(component.rate, 6)])} />
    <Metrics values={[['Total squared error used', number(state.distortion)], ['Bits / vector', number(state.rate, 6)], ['Bits / component on average', number(state.rate / 2, 6)]]} />
    <p>{state.distortion === 0 ? 'A nondegenerate Gaussian has continuously many possible values: zero squared error needs an unbounded information rate in this ideal model.' : state.distortion >= state.totalVariance ? 'Reconstructing both known means needs no source-dependent bits and costs the sum of the variances. Extra allowed error does not require extra rate.' : 'Components taller than the level share the same allocated error. A shorter component reaches its full variance and receives zero bits; error is not allocated above that cap.'}</p>
    <p><strong>Transfer.</strong> Choose equal variances and predict the allocation before moving the budget. The symmetry applies to this Gaussian/MSE model; it is not a rule for every two-feature dataset.</p>
  </Investigation>;
}
export function FidelityMarginalFigure() {
  const cases = zeroRateFidelity();
  return <figure className="rate-distortion-inline">
    <figcaption>Fair input −1 / +1; no source-dependent message reaches the decoder.</figcaption>
    <div className="rate-distortion-fidelity">{cases.map(entry => <div key={entry.label}><h4>{entry.label}</h4>
      <div className="rate-distortion-signs">{entry.output.map((mass, index) => <div key={index}><span>{index - 1 === 1 ? '+1' : index - 1}</span><div><span style={{
                height: mass * 100 + '%'
              }} /></div><strong>{mass}</strong></div>)}</div>
      <p>Output probabilities at −1, 0, +1.</p><p><strong>MSE = {entry.distortion}</strong>; information =0 bits.</p>
    </div>)}</div>
    <p>The random output has exactly the source marginal law, but is independent of which sign was actually observed. Realism of a collection and fidelity to its particular input are different tests.</p>
  </figure>;
}
export function LearnedCodecFigure() {
  return <figure className="rate-distortion-inline"><figcaption>A useful latent representation becomes a codec only when its receiver can reconstruct from declared bits.</figcaption>
    <ol className="rate-distortion-pipeline">
      <li><strong>Transform</strong><span>Source x → latent y</span><span>Learned or hand-designed</span></li>
      <li><strong>Quantize</strong><span>y → discrete indices z</span><span>Some detail can be lost</span></li>
      <li className="rate-distortion-wire"><strong>Encode + header</strong><span>Actual bytes cross</span><span>Lengths, model selection, indices</span></li>
      <li><strong>Decode + reconstruct</strong><span>Bytes → z → x̂</span><span>Shared model must agree</span></li>
    </ol><p>Lossless entropy coding changes how the indices are represented, not their values. Distortion enters through the lossy representation and reconstruction. Training surrogates are evaluated against these actual operations.</p>
  </figure>;
}
