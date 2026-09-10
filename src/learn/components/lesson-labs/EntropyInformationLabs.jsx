import { useId, useState } from 'react';
import { LessonTable } from './LessonElements';
import { SYMBOLS, SOURCE_WEIGHTS, CODEBOOKS, entropyNumber as number, parseWeights, informationState, binaryEntropyState, prefixCodeState, conditionalLossState, logitsState, continuousEntropyState, maxEntropyState } from '../../data/entropy-information-models';
import './entropy-information-labs.css';
function Lab({
  name,
  title,
  children
}) {
  const id = useId();
  return <section className="entropy-lab" data-entropy-lab={name} aria-labelledby={id}><h3 id={id}>{title}</h3>{children}<p className="entropy-note">Calculated from the displayed finite model; decimal readouts are rounded. Values near 10⁻³² at mathematical equality are floating-point residuals. These are teaching investigations, not measurements of a deployed model.</p></section>;
}
function Range({
  label,
  value,
  setValue,
  min,
  max,
  step = 1
}) {
  return <label>{label}<strong>{number(value)}</strong><input type="range" aria-label={label} value={value} min={min} max={max} step={step} onChange={event => setValue(Number(event.target.value))} /></label>;
}
function Select({
  label,
  value,
  setValue,
  options
}) {
  return <label>{label}<select value={value} onChange={event => setValue(event.target.value)}>{Object.entries(options).map(([key, name]) => <option key={key} value={key}>{name}</option>)}</select></label>;
}
function Metric({
  label,
  children
}) {
  return <div><dt>{label}</dt><dd>{children}</dd></div>;
}
function Curve({
  points,
  selected,
  optimum,
  xMin = 0,
  xMax = 1,
  maxY = 1,
  xLabel,
  yLabel
}) {
  const x = value => xMin === xMax ? 192 : 52 + 280 * (value - xMin) / (xMax - xMin);
  const y = value => 223 - 160 * value / maxY;
  return <svg className="entropy-plot" viewBox="0 0 380 290" role="img" aria-label={`${yLabel} versus ${xLabel}; selected value ${number(selected[0])}, result ${number(selected[1])}`}>
    <path d="M52 45 V223 H337" stroke="#888" fill="none" />
    <text x="52" y="27">{yLabel}</text><text x="33" y="229">0</text><text x="9" y="69">{number(maxY, 2)}</text>
    <text x="45" y="248">{number(xMin, 2)}</text><text x="307" y="248">{number(xMax, 2)}</text><text x="52" y="279">{xLabel}</text>
    {xMin !== xMax && <path d={points.map(([a, b], index) => `${index ? 'L' : 'M'}${x(a)},${y(b)}`).join(' ')} fill="none" stroke="#d8bf87" strokeWidth="2.5" />}
    {optimum && <g><path d={`M${x(optimum[0])} 223 V${y(optimum[1])}`} stroke="#a9d9af" strokeDasharray="5 5" /><rect x={x(optimum[0]) - 5} y={y(optimum[1]) - 5} width="10" height="10" fill="#a9d9af" /></g>}
    <path d={`M${x(selected[0])} 223 V${y(selected[1])}`} stroke="#e2b55a" strokeDasharray="3 5" /><circle cx={x(selected[0])} cy={y(selected[1])} r="6" fill="#e2b55a" stroke="#111" strokeWidth="2" />
  </svg>;
}
export function PrefixTreeFigure() {
  return <figure className="entropy-inline"><figcaption>One complete word ends the current question</figcaption><svg className="entropy-tree" viewBox="0 0 340 310" role="img" aria-label="Start: bit0 ends at A. Bit1 continues. Then0 ends at B. Another1 continues; then0 ends at C,1 at D. Codewords A0 B10 C110 D111.">
    <g fill="none" stroke="#777" strokeWidth="2"><path d="M170 23 L65 84 M170 23 L240 84 M240 84 L153 160 M240 84 L278 160 M278 160 L227 237 M278 160 L319 237" /></g>
    <g className="entropy-edge-label"><text x="102" y="49">0</text><text x="211" y="49">1</text><text x="178" y="116">0</text><text x="263" y="116">1</text><text x="233" y="191">0</text><text x="305" y="191">1</text></g>
    <g fill="#171717" stroke="#999"><circle cx="170" cy="23" r="7" /><circle cx="240" cy="84" r="7" /><circle cx="278" cy="160" r="7" /></g>
    {[[65, 84, 'A', '0'], [153, 160, 'B', '10'], [227, 237, 'C', '110'], [319, 237, 'D', '111']].map(([x, y, symbol, word]) => <g key={symbol}><circle cx={x} cy={y} r="19" fill="#242116" stroke="#e2b55a" /><text x={x} y={y + 6} textAnchor="middle" className="entropy-symbol">{symbol}</text><text x={x} y={y + 49} textAnchor="middle">{word}</text></g>)}
  </svg><p>Edges carry bits; leaves carry outcomes. No outcome is also an unfinished route to another outcome. Word length is the number of edges, not the drawing's physical length.</p></figure>;
}
export function BinarySurpriseLab() {
  const [probability, setProbability] = useState(0.9);
  const state = binaryEntropyState(probability);
  return <Lab name="surprise" title="A rare outcome can be surprising while average entropy is small">
    <p>Predict which is larger at p=0.99: the surprise of the rare outcome, or the average over both outcomes. The curve plots the average; the table separates price from frequency.</p>
    <div className="entropy-controls"><Range label="Probability of heads p" value={probability} setValue={setProbability} min={0} max={1} step={0.01} /><button onClick={() => setProbability(0.9)}>Reset to p=0.9</button></div>
    <Curve points={state.curve} selected={[probability, state.entropy]} xLabel="Probability of heads p" yLabel="Entropy (bits / draw)" />
    <LessonTable caption="Average each outcome's surprise using its own probability" headers={['Outcome', 'Probability', 'Surprise (bits)', 'Weighted contribution']} rows={state.rows.map((row, index) => [index ? 'Tails' : 'Heads', number(row.p), number(row.surprise), number(row.entropy)])} />
    <dl className="entropy-metrics" aria-live="polite"><Metric label="Average entropy (bits / draw)">{number(state.entropy)}</Metric><Metric label="Effective uniform choices 2ᴴ">{number(2 ** state.entropy)}</Metric></dl>
    <p>At p=0 or1, an outcome excluded by the model has infinite surprise if it occurred, but contributes zero to this model's expectation. Compare p=0.1 and0.9: swapping labels leaves entropy unchanged.</p>
  </Lab>;
}
export function PrefixCodeLab() {
  const [draft, setDraft] = useState('AAAABBCD'),
    [message, setMessage] = useState('AAAABBCD');
  const [codebook, setCodebook] = useState('matched'),
    [consumed, setConsumed] = useState(0),
    [error, setError] = useState('');
  const state = prefixCodeState(message, codebook, consumed);
  function apply() {
    try {
      prefixCodeState(draft, codebook);
      setMessage(draft);
      setConsumed(0);
      setError('');
    } catch (failure) {
      setError(failure.message);
    }
  }
  function changeCodebook(value) {
    setCodebook(value);
    setConsumed(0);
  }
  return <Lab name="coding" title="Send a message, then decode it one bit at a time">
    <p>The source probabilities stay A=1/2, B=1/4, C=D=1/8. Predict whether changing the code or changing one short message changes the source entropy. Encoding uses the active message; source-average cost uses the fixed probabilities.</p>
    <div className="entropy-controls"><Select label="Codebook" value={codebook} setValue={changeCodebook} options={Object.fromEntries(Object.entries(CODEBOOKS).map(([key, value]) => [key, value.label]))} /><label>Message (A–D, 1–16 symbols)<input aria-label="Message" value={draft} onChange={event => setDraft(event.target.value)} /></label><button onClick={apply}>Encode message</button><button onClick={() => {
        setDraft('AAAABBCD');
        setMessage('AAAABBCD');
        setCodebook('matched');
        setConsumed(0);
        setError('');
      }}>Reset coding</button></div>
    {error && <p role="alert">{error} The active message and bit position are unchanged.</p>}
    <LessonTable caption="A fixed codebook shared by sender and receiver" headers={['Symbol', 'Source P', 'Codeword', 'Bits']} rows={SYMBOLS.map((symbol, index) => [symbol, SOURCE_WEIGHTS[index] + '/8', state.words[index], state.words[index].length])} />
    <div className="entropy-word-strip" aria-label="Encoded words with teaching boundaries">{[...message].map((symbol, index) => <div key={index}><span>{symbol}</span><strong>{state.chunks[index]}</strong></div>)}</div>
    <p className="entropy-note">Separators above show the explanation. The transmitted bitstream below contains no separator symbols.</p>
    <div className="entropy-bitstream" aria-label="Transmitted bitstream">{[...state.bits].map((bit, index) => <span key={index} className={index < consumed ? 'consumed' : ''}>{bit}</span>)}</div>
    <div className="entropy-controls"><button disabled={consumed === 0} onClick={() => setConsumed(consumed - 1)}>Back one bit</button><button disabled={consumed === state.bits.length} onClick={() => setConsumed(consumed + 1)}>Read next bit</button><button onClick={() => setConsumed(state.bits.length)}>Decode all bits</button><button onClick={() => setConsumed(0)}>Restart decoder</button></div>
    <dl className="entropy-metrics" aria-live="polite"><Metric label="Bits read">{consumed} / {state.bits.length}</Metric><Metric label="Complete decoded symbols">{state.decoded || '(none yet)'}</Metric><Metric label="Unfinished prefix">{state.buffer || '(empty: at a word boundary)'}</Metric><Metric label="This message, bits / symbol">{state.bits.length} / {message.length} = {number(state.messageAverage)}</Metric><Metric label="Source-average bits / symbol">{number(state.expectedLength)}</Metric><Metric label="Source entropy (bits / symbol)">1.75</Metric></dl>
    <p>Try CCCC: even the source-matched code uses three bits for each C, longer than fixed two-bit words on this particular message. Compression is an average statement under a source model. A real file format must also communicate its codebook/framing; those overheads are excluded here.</p>
  </Lab>;
}
export function MismatchLab() {
  const [pDraft, setPDraft] = useState('4, 2, 1, 1'),
    [qDraft, setQDraft] = useState('1, 1, 2, 4');
  const [weights, setWeights] = useState([SOURCE_WEIGHTS, [1, 1, 2, 4]]),
    [unit, setUnit] = useState('bits'),
    [error, setError] = useState('');
  const state = informationState(...weights, unit === 'bits' ? 2 : Math.E);
  function apply(p = pDraft, q = qDraft) {
    try {
      const next = [parseWeights(p), parseWeights(q)];
      setWeights(next);
      setPDraft(p);
      setQDraft(q);
      setError('');
    } catch (failure) {
      setError(failure.message);
    }
  }
  const finiteTerms = state.rows.map(row => Math.abs(row.kl)).filter(Number.isFinite);
  const termScale = Math.max(...finiteTerms, 0.1);
  return <Lab name="mismatch" title="Keep the outcomes; change the prices charged by the model">
    <p>P supplies the frequencies; Q supplies the prediction prices. Predict which rows can have negative excess cost before applying Q=P. Enter four weights; each set is separately divided by its sum. Labels A–D stay aligned.</p>
    <div className="entropy-controls"><label>Source weights P<input aria-label="Source weights P" value={pDraft} onChange={event => setPDraft(event.target.value)} /></label><label>Model weights Q<input aria-label="Model weights Q" value={qDraft} onChange={event => setQDraft(event.target.value)} /></label><button onClick={() => apply()}>Apply distributions</button><Select label="Information units" value={unit} setValue={setUnit} options={{
        bits: 'Bits (base 2)',
        nats: 'Nats (natural log)'
      }} /></div>
    <div className="entropy-controls"><button onClick={() => apply(pDraft, pDraft)}>Match Q to P</button><button onClick={() => apply('4, 2, 1, 1', '0, 2, 1, 1')}>Exclude possible A</button><button onClick={() => apply('0, 2, 1, 1', '0, 2, 1, 1')}>Exclude A in both</button><button onClick={() => apply('4, 2, 1, 1', '1, 1, 2, 4')}>Reset mismatch</button></div>
    {error && <p role="alert">{error} Active distributions are unchanged.</p>}
    <div className="entropy-mass-bars">{state.rows.map(row => <div key={row.label}><strong>{row.label}</strong><div><span>P {number(row.p, 3)}</span><i style={{
            width: row.p * 100 + '%'
          }} /><span>Q {number(row.q, 3)}</span><i className="model" style={{
            width: row.q * 100 + '%'
          }} /></div></div>)}</div>
    <p className="entropy-note">Bar lengths use the same 0–1 probability scale. Gold = source P; outlined pale bars = model Q.</p>
    <div className="entropy-signed-terms" aria-label="Signed contributions to KL">{state.rows.map(row => <div key={row.label}><span>{row.label}</span><div><i style={{
            left: row.kl < 0 ? 50 - 50 * Math.abs(row.kl) / termScale + '%' : '50%',
            width: Number.isFinite(row.kl) ? 50 * Math.abs(row.kl) / termScale + '%' : '50%'
          }} className={row.kl < 0 ? 'negative' : 'positive'} /></div><strong>{number(row.kl)}</strong></div>)}</div>
    <p className="entropy-note">Excess contribution p·log(p/q): bars extend left for negative, right for positive. Each row shares the displayed finite scale; ∞ uses the full right half as a symbol, not a finite bar length.</p>
    <LessonTable caption={'Price and weighted cost for every outcome (' + unit + ')'} headers={['Outcome', '−log P', '−log Q', 'P·(−log P)', 'P·(−log Q)', 'P·log(P/Q)']} rows={state.rows.map(row => [row.label, number(row.surprise), number(row.price), number(row.entropy), number(row.crossEntropy), number(row.kl)])} />
    <dl className="entropy-metrics" aria-live="polite"><Metric label={'Entropy H(P), ' + unit}>{number(state.entropy)}</Metric><Metric label={'Cross-entropy H(P,Q), ' + unit}>{number(state.crossEntropy)}</Metric><Metric label={'KL(P ∥ Q), ' + unit}>{number(state.kl)}</Metric></dl>
    <p>{state.kl === Infinity ? 'A P-positive outcome receives Q=0, so its expected cost is infinite. No clipping is applied.' : 'Some rows may be cheaper under Q, but the total excess is nonnegative. Matching Q to P makes every excess term zero.'} Swap the two input lists to investigate the reverse direction; it changes the weighting as well as the ratio.</p>
  </Lab>;
}
export function ConditionalEntropyLab() {
  const [noise, setNoise] = useState(0.1),
    [trust, setTrust] = useState(0.9);
  const state = conditionalLossState(noise, trust);
  return <Lab name="conditional" title="The label can be balanced overall and predictable within each context">
    <p>X is a fair binary context. The label equals X with probability 1−e and flips with probability e. Predict H(Y) when e changes. Each row occupies half the total rectangle; horizontal cell widths encode conditional probabilities, so areas encode joint mass.</p>
    <div className="entropy-controls"><Range label="Label-flip probability e" value={noise} setValue={setNoise} min={0} max={0.5} step={0.01} /><Range label="Model probability Q(Y=X | X)" value={trust} setValue={setTrust} min={0} max={1} step={0.01} /><button onClick={() => setTrust(1 - noise)}>Use the true conditional</button><button onClick={() => setTrust(0.5)}>Ignore the context</button><button onClick={() => {
        setNoise(0.1);
        setTrust(0.9);
      }}>Reset conditional</button></div>
    <div className="entropy-mosaic" role="img" aria-label={'Joint distribution: ' + state.rows.map(row => `X${row.x},Y${row.y} has probability ${number(row.mass)}`).join('; ')}>{[0, 1].map(x => <div key={x}><span>X={x}</span><div>{state.rows.filter(row => row.x === x).map(row => <div key={row.y} style={{
            width: row.conditional * 100 + '%'
          }} className={'y' + row.y}>{row.conditional >= 0.2 ? 'Y=' + row.y : ''}</div>)}</div></div>)}</div>
    <LessonTable caption="Joint mass averages each context-specific prediction price" headers={['X', 'Y', 'P(X,Y)', 'Q(Y|X)', 'Weighted loss (bits)']} rows={state.rows.map(row => [row.x, row.y, number(row.mass), number(row.predicted), number(row.loss)])} />
    <dl className="entropy-metrics" aria-live="polite"><Metric label="Marginal label entropy H(Y), bits">1</Metric><Metric label="Conditional entropy H(Y | X), bits">{number(state.conditionalEntropy)}</Metric><Metric label="Model conditional cross-entropy, bits">{number(state.modelLoss)}</Metric><Metric label="Expected conditional KL, bits">{number(state.excess)}</Metric></dl>
    <p>At e=0, the contexts determine Y exactly, even though Y is balanced overall. At e=0.5, the context tells us nothing. An overconfident model can score badly when noise remains; a zero probability for a possible label makes the expected loss infinite.</p>
  </Lab>;
}
export function LogitLossLab() {
  const [gap, setGap] = useState(2),
    [offset, setOffset] = useState(0),
    [target, setTarget] = useState('0');
  const state = logitsState(gap, offset, Number(target));
  return <Lab name="logits" title="A probability may underflow while its log loss remains computable">
    <p>Three class scores are (g,0,−g), plus a common offset. Predict whether the offset changes probabilities. The stable path subtracts the largest score before exponentiating, and calculates log probabilities directly.</p>
    <div className="entropy-controls"><Range label="Score gap g" value={gap} setValue={setGap} min={0} max={1000} /><Range label="Common score offset" value={offset} setValue={setOffset} min={-1000} max={1000} step={100} /><Select label="Observed class" value={target} setValue={setTarget} options={{
        0: 'Class 0',
        1: 'Class 1',
        2: 'Class 2'
      }} /><button onClick={() => {
        setGap(2);
        setOffset(0);
        setTarget('0');
      }}>Reset logits</button></div>
    <div className="entropy-pipeline"><span>Subtract max <strong>{number(state.maximum)}</strong></span><span aria-hidden="true">→</span><span>Sum exp of shifted scores <strong>{number(state.normalizer)}</strong></span><span aria-hidden="true">→</span><span>Loss = max − observed score + log(sum)</span></div>
    <LessonTable caption="The same scores through the stable calculation" headers={['Class', 'Logit', 'Shifted', 'exp(shifted)', 'Probability', 'Log probability']} rows={state.rows.map(row => [row.index, number(row.logit), number(row.shifted), number(row.exponential), number(row.probability), number(row.logProbability)])} />
    <dl className="entropy-metrics" aria-live="polite"><Metric label="Observed-class log loss (nats)">{number(state.loss)}</Metric><Metric label="Observed class">{target}</Metric></dl>
    <p>Set g=1000 and observe class 2: the probability column underflows to numerical zero, yet the log probability remains −2000 and the loss is 2000. Taking a logarithm of that rounded/underflowed probability would lose this information. This finite-score model does not cover NaN or infinite inputs.</p>
  </Lab>;
}
function UniformPicture({
  state,
  bins,
  unit
}) {
  return <figure><svg className="entropy-plot" viewBox="0 0 380 280" role="img" aria-label={`Uniform density on 0 to ${number(state.width)} ${unit}, height ${number(state.density)} per ${unit}; middle half has probability one half.`}>
    <path d="M55 45 V210 H337" stroke="#888" fill="none" /><rect x="55" y="80" width="270" height="130" fill="#d8bf8720" stroke="#d8bf87" /><rect x="122.5" y="80" width="135" height="130" fill="#e2b55a65" />
    {Array.from({
        length: bins - 1
      }, (_, i) => <path key={i} d={`M${55 + 270 * (i + 1) / bins} 80 V210`} stroke="#aaa" strokeDasharray="3 4" />)}
    <text x="55" y="26">Density per {unit}</text><text x="55" y="64">Height {number(state.density)}</text><text x="51" y="234">0</text><text x="276" y="234">{number(state.width)}</text><text x="55" y="268">Coordinate ({unit})</text><text x="190" y="154" textAnchor="middle" className="entropy-symbol">P=1/2</text>
  </svg><figcaption>Own axis range: width {number(state.width)} {unit}, height {number(state.density)} /{unit}. Shaded interval [{state.event.map(value => number(value)).join(', ')}].</figcaption></figure>;
}
export function ContinuousEntropyLab() {
  const [width, setWidth] = useState(0.25),
    [scale, setScale] = useState('100'),
    [bins, setBins] = useState(4);
  const state = continuousEntropyState(width, Number(scale), bins);
  const unit = scale === '100' ? 'cm' : scale === '10' ? 'dm' : 'm';
  return <Lab name="continuous" title="Change the ruler; preserve the event probability">
    <p>X is uniform on [0,w] metres. Change to centimetres or decimetres without changing the physical experiment. Predict which of density height, interval probability and differential entropy stays fixed. Each plot uses its own labeled axis range; equal screen rectangles do not mean equal numerical densities.</p>
    <div className="entropy-controls"><Range label="Width w in metres" value={width} setValue={setWidth} min={0.125} max={4} step={0.125} /><Select label="New coordinate unit" value={scale} setValue={setScale} options={{
        1: 'Metres: scale 1',
        10: 'Decimetres: scale 10',
        100: 'Centimetres: scale 100'
      }} /><Range label="Equal quantization bins" value={bins} setValue={setBins} min={2} max={16} /><button onClick={() => {
        setWidth(0.25);
        setScale('100');
        setBins(4);
      }}>Reset units</button></div>
    <div className="entropy-two-plots"><UniformPicture state={state.original} bins={bins} unit="m" /><UniformPicture state={state.transformed} bins={bins} unit={unit} /></div>
    <dl className="entropy-metrics" aria-live="polite"><Metric label="h(X), bits relative to metre coordinate">{number(state.original.entropy)}</Metric><Metric label={'h(new coordinate), bits'}>{number(state.transformed.entropy)}</Metric><Metric label="Same middle-half event probability">0.5</Metric><Metric label="Entropy of bin label, bits">{number(state.quantizedEntropy)}</Metric><Metric label="Bin width in metres">{number(state.original.binWidth)}</Metric><Metric label={'Bin width in ' + unit}>{number(state.transformed.binWidth)}</Metric></dl>
    <p>Every bin has probability 1/{bins}. Its discrete entropy is log₂({bins}) in either unit. The numerical bin width changes too, so h+log₂(1/Δ) remains the same. This identity is exact for this aligned uniform example; arbitrary densities require additional conditions for a small-bin approximation.</p>
  </Lab>;
}
export function MaximumEntropyLab() {
  const [mean, setMean] = useState(String(10 / 7)),
    [fraction, setFraction] = useState(0.25);
  const state = maxEntropyState(Number(mean), fraction);
  const boundary = state.lower === state.upper;
  return <Lab name="maximum" title="Move probability while keeping the required mean unchanged">
    <p>The possible counts are 0, 1, 2. Fix their mean m, then move along q=(1−m+t,m−2t,t). Predict whether the most spread-out looking bars are necessarily the entropy maximum. The green square marks the certified maximum; gold marks your selected feasible distribution.</p>
    <div className="entropy-controls"><Select label="Required mean m" value={mean} setValue={value => {
        setMean(value);
        setFraction(0.25);
      }} options={{
        0: '0: boundary',
        0.5: '0.5',
        1: '1',
        [String(10 / 7)]: '10/7',
        1.8: '1.8',
        2: '2: boundary'
      }} /><Range label="Position along feasible slice" value={fraction} setValue={setFraction} min={0} max={1} step={0.01} /><button onClick={() => {
        if (!boundary) setFraction((state.optimum[2] - state.lower) / (state.upper - state.lower));
      }}>Select the maximum</button><button onClick={() => {
        setMean(String(10 / 7));
        setFraction(0.25);
      }}>Reset mean constraint</button></div>
    <Curve points={state.curve} selected={[state.t, state.entropy]} optimum={[state.optimum[2], state.maximumEntropy]} xMin={state.lower} xMax={state.upper} maxY={1.6} xLabel="Probability t = q(2)" yLabel="Entropy (bits)" />
    <LessonTable caption="Both distributions meet the same normalization and mean constraints" headers={['Outcome', 'Selected q', 'Maximum p*']} rows={state.q.map((mass, index) => [index, number(mass), number(state.optimum[index])])} />
    <dl className="entropy-metrics" aria-live="polite"><Metric label="Required and preserved mean">{number(state.mean)}</Metric><Metric label="Selected entropy H(q), bits">{number(state.entropy)}</Metric><Metric label="Maximum entropy H(p*), bits">{number(state.maximumEntropy)}</Metric><Metric label="KL(q ∥ p*), bits">{number(state.gap)}</Metric><Metric label="Allowed probability t">[{number(state.lower)}, {number(state.upper)}]</Metric><Metric label="Exponential parameter η (natural log)">{boundary ? 'No finite η; limiting point mass' : number(state.eta)}</Metric></dl>
    <p>{boundary ? 'Only one distribution has this endpoint mean. There is no nontrivial slice; the point mass is the maximizer, reached as a limit of finite exponential parameters.' : 'The curve is calculated from the feasible slice in this three-outcome model. The proof below certifies the entire slice; a plotted grid alone would not prove global optimality.'} A nonuniform base measure changes the objective, as the following worked contrast explains.</p>
    <p>The calculator handles exact boundary means 0 and 2 separately. Interior means must lie from 0.000001 through 1.999999; getting still closer to a boundary needs more careful numerical precision. This arithmetic restriction does not restrict the mathematical certificate.</p>
  </Lab>;
}
export function ClassifierLossFigure() {
  const rows = [{
    name: 'A: moderate confidence',
    correct: 0.6,
    wrong: 0.4
  }, {
    name: 'B: strong confidence',
    correct: 0.99,
    wrong: 0.01
  }];
  return <figure className="entropy-inline"><figcaption>The same eight successes and two errors, priced differently</figcaption>{rows.map(row => <div className="entropy-case-row" key={row.name}><strong>{row.name}</strong><div>{Array.from({
          length: 10
        }, (_, index) => {
          const p = index < 8 ? row.correct : row.wrong;
          return <div key={index}><span>{index + 1}</span><i style={{
              height: Math.max(2, 100 * -Math.log(p) / 5) + 'px'
            }} className={index < 8 ? 'correct' : 'wrong'} /><small>{index < 8 ? '✓' : '×'}</small></div>;
        })}</div><p>Each correct-label probability: {row.correct} for cases 1–8; {row.wrong} for cases 9–10.</p></div>)}<p>Bar heights encode −ln(correct-label probability), using the same 0–5 nats scale. Position identifies the shared case; ✓/× identify accuracy. Both models score 80% accuracy, while the two high-confidence errors dominate B's loss.</p></figure>;
}
