import { useId, useMemo, useState } from 'react';
import { bernsteinApproximation, cauchyBlock, derivativeFamily, dyadicBracket, inverseSquareSeries, powerFamily, sequenceTail, triangleFamily, typewriterInterval } from '../../data/real-analysis-models.js';
import './real-analysis-labs.css';
function number(value, digits = 5) {
  if (value === 0) return '0';
  if (Math.abs(value) < 0.0001 || Math.abs(value) >= 10000) return value.toExponential(3);
  return Number(value.toFixed(digits)).toString();
}
function Select({
  label,
  value,
  onChange,
  options
}) {
  return <label className="analysis-field"><span>{label}</span>
    <select aria-label={label} value={value} onChange={event => onChange(event.target.value)}>
      {options.map(([key, text]) => <option key={key} value={key}>{text}</option>)}
    </select>
  </label>;
}
function Range({
  label,
  value,
  onChange,
  min,
  max,
  step = 1
}) {
  return <label className="analysis-field"><span>{label}: <strong>{number(value)}</strong></span>
    <input aria-label={label} type="range" min={min} max={max} step={step} value={value} onChange={event => onChange(Number(event.target.value))} />
  </label>;
}
function Investigation({
  title,
  prediction,
  children,
  resetLabel,
  onReset
}) {
  return <section className="analysis-lab" aria-label={title}>
    <h3>{title}</h3><p>{prediction}</p>{children}
    <button type="button" onClick={onReset}>{resetLabel}</button>
  </section>;
}
function Plot({
  label,
  xRange,
  yRange,
  xLabel,
  yLabel,
  series = [],
  background,
  markers,
  children
}) {
  const clipId = useId().replaceAll(':', '');
  const left = 53;
  const right = 283;
  const top = 30;
  const bottom = 176;
  const x = value => left + (value - xRange[0]) * (right - left) / (xRange[1] - xRange[0]);
  const y = value => bottom - (value - yRange[0]) * (bottom - top) / (yRange[1] - yRange[0]);
  const geometry = {
    x,
    y,
    left,
    right,
    top,
    bottom
  };
  const xTicks = [xRange[0], (xRange[0] + xRange[1]) / 2, xRange[1]];
  const yTicks = [yRange[0], (yRange[0] + yRange[1]) / 2, yRange[1]];
  return <svg className="analysis-plot" viewBox="0 0 320 222" role="img" aria-label={label}>
    <defs><clipPath id={clipId}><rect x={left} y={top} width={right - left} height={bottom - top} /></clipPath></defs>
    <text x={left} y="16" className="axis-title">{yLabel}</text>
    {yTicks.map(tick => <g key={tick}><line className="grid" x1={left} x2={right} y1={y(tick)} y2={y(tick)} /><text x={left - 7} y={y(tick) + 4} textAnchor="end">{number(tick, 3)}</text></g>)}
    <g clipPath={`url(#${clipId})`}>
      {background?.(geometry)}
      {series.map((line, index) => <polyline key={line.label || index} className={line.className || 'gold'} points={line.points.map(point => `${x(point.x)},${y(point.y)}`).join(' ')} fill="none"><title>{line.label}</title></polyline>)}
      {children?.(geometry)}
    </g>
    <path className="axis" d={`M${left},${top}V${bottom}H${right}`} />
    {xTicks.map((tick, index) => <text key={tick} x={x(tick)} y={bottom + 22} textAnchor={index === 0 ? 'start' : index === 2 ? 'end' : 'middle'}>{number(tick, 3)}</text>)}
    <text x={(left + right) / 2} y="217" textAnchor="middle" className="axis-title">{xLabel}</text>
    {markers?.(geometry)}
  </svg>;
}
function Readout({
  label,
  children
}) {
  return <div className="analysis-readout"><span>{label}</span><strong>{children}</strong></div>;
}
export function SequenceTailLab() {
  const [denominator, setDenominator] = useState(10);
  const [index, setIndex] = useState(9);
  const state = sequenceTail(1, denominator, index);
  return <Investigation title="Choose where the entire safe tail starts" prediction="The error must be strictly below the requested tolerance. Predict whether index 9 works at epsilon=1/10, then move the tail boundary." resetLabel="Reset sequence tail" onReset={() => {
    setDenominator(10);
    setIndex(9);
  }}>
    <div className="analysis-controls"><Select label="Requested tolerance" value={denominator} onChange={value => setDenominator(Number(value))} options={[2, 5, 10, 20].map(value => [value, `1/${value}`])} /><Range label="Proposed start N" value={index} onChange={setIndex} min={1} max={25} /></div>
    <Plot label="Forty sequence values with the tolerance band and proposed tail boundary" xRange={[1, 40]} yRange={[2.4, 3.6]} xLabel="Index n" yLabel="Estimate a_n" background={({
      x,
      y,
      top,
      bottom,
      right
    }) => <><rect className="band" x={x(1)} y={y(3 + state.tolerance)} width={right - x(1)} height={y(3 - state.tolerance) - y(3 + state.tolerance)} /><line className="rose dashed" x1={x(index)} x2={x(index)} y1={top} y2={bottom} /></>} markers={({
      x,
      y
    }) => state.values.map(point => <circle key={point.n} className={point.n < index ? 'muted-point' : point.error < state.tolerance ? 'blue-fill' : 'gold-fill'} cx={x(point.n)} cy={y(point.value)} r="2.4" />)} />
    <div className="analysis-readouts"><Readout label="First tail error">1/{index + 1}</Readout><Readout label="Smallest valid N">{state.minimumIndex}</Readout></div>
    <p className="analysis-result" aria-live="polite">{state.certifiesTail ? `Every later error is at most 1/${index + 1}, which is strictly below 1/${denominator}. This N certifies the tail.` : state.boundaryEquality ? 'The first tail term is exactly on the tolerance boundary. Equality does not satisfy a strict bound.' : 'The first tail term already exceeds the tolerance. This proposed N fails.'}</p>
    <p>Forty terms are shown. The proof covers the rest because 1/(n+1) decreases with n; the finite picture by itself does not.</p>
  </Investigation>;
}
export function CompletenessBracketLab() {
  const [target, setTarget] = useState(2);
  const [steps, setSteps] = useState(3);
  const state = dyadicBracket(target, steps);
  return <Investigation title="Keep a shrinking bracket around an unknown real number" prediction="Choose the half whose endpoint squares still enclose the target. Predict what one more step does to the guaranteed error." resetLabel="Reset exact bracket" onReset={() => {
    setTarget(2);
    setSteps(3);
  }}>
    <div className="analysis-controls"><Select label="Squared target" value={target} onChange={value => setTarget(Number(value))} options={[2, 3, 5].map(value => [value, `Find the positive square root of ${value}`])} /><Range label="Bisection steps" value={steps} onChange={setSteps} min={0} max={16} /></div>
    <div className="analysis-step-controls"><button type="button" disabled={steps === 0} onClick={() => setSteps(value => value - 1)}>Previous bracket</button><button type="button" disabled={steps === 16} onClick={() => setSteps(value => value + 1)}>Bisect once</button></div>
    <figure className="analysis-inline" data-analysis-figure="bracket"><svg viewBox="0 0 320 100" role="img" aria-label="Magnified current bracket with a midpoint halfway between its endpoints"><line className="gold" x1="24" x2="296" y1="48" y2="48" /><path className="gold" d="M24 34v28M296 34v28" /><circle className="blue-fill" cx="160" cy="48" r="5" /><text x="24" y="25">left</text><text x="296" y="25" textAnchor="end">right</text><text x="160" y="78" textAnchor="middle">midpoint</text></svg><figcaption>The current interval is magnified to this ruler. Its drawn width stays fixed while its numerical width shrinks.</figcaption></figure>
    <div className="analysis-readouts"><Readout label="Exact left endpoint">{state.left.toFixed(steps)}</Readout><Readout label="Exact right endpoint">{state.right.toFixed(steps)}</Readout><Readout label="Exact width">1/2^{steps}</Readout><Readout label="Midpoint error at most">1/2^{steps + 1}</Readout></div>
    <p className="analysis-result">Rounded endpoint squares: {number(state.leftSquared, 9)} ≤ {target} ≤ {number(state.rightSquared, 9)}. The exact internal endpoint-square inequalities retain the real root. The endpoint decimals above terminate exactly because their denominators divide 2^{steps}; the squared decimals here are rounded for readability.</p>
  </Investigation>;
}
export function CauchyBlockLab() {
  const [n, setN] = useState(8);
  const [family, setFamily] = useState('harmonic');
  const state = cauchyBlock(n, family);
  const highest = state.terms[0].value;
  return <Investigation title="Look beyond the next small step" prediction="Compare the next increment with the total from N+1 through 2N. Predict whether the whole block becomes arbitrarily small." resetLabel="Reset Cauchy block" onReset={() => {
    setN(8);
    setFamily('harmonic');
  }}>
    <div className="analysis-controls"><Select label="Series increments" value={family} onChange={setFamily} options={[['harmonic', 'Harmonic: 1/k'], ['telescoping', 'Telescoping: 1/[k(k+1)]']]} /><Select label="Block begins after N" value={n} onChange={value => setN(Number(value))} options={[2, 4, 8, 16, 32].map(value => [value, value])} /></div>
    <figure className="analysis-inline"><div className="analysis-increments" role="img" aria-label={`${n} positive increments from ${n + 1} through ${2 * n}; total ${number(state.blockTotal)}`}>{state.terms.map(term => <span key={term.k} style={{
          height: `${100 * term.value / highest}%`
        }} title={`k=${term.k}, increment=${number(term.value)}`} />)}</div><figcaption>One bar per increment. Heights are relative to the first increment in this block; use the numerical totals to compare different N.</figcaption></figure>
    <div className="analysis-readouts"><Readout label="Next increment">{number(state.singleIncrement)}</Readout><Readout label="Whole block gap">{number(state.blockTotal)}</Readout></div>
    <p className="analysis-result">{family === 'harmonic' ? `There are N=${n} terms and each is at least 1/(2N). Their total is at least 1/2 for every N, even while the next step shrinks.` : `Cancellation gives exactly 1/${n + 1} − 1/${2 * n + 1}. The entire remaining infinite tail is 1/${n + 1}, which does tend to zero.`}</p>
  </Investigation>;
}
export function PowerConvergenceLab() {
  const [n, setN] = useState(8);
  const [domain, setDomain] = useState('closed-unit');
  const [fixedPoint, setFixedPoint] = useState(0.5);
  const end = domain === 'compact-subinterval' ? 0.75 : 1;
  const state = powerFamily(n, domain, 0.75, Math.min(fixedPoint, end));
  return <Investigation title="Fix one point, then let the difficult point move" prediction="A fixed input below 1 becomes easy. Predict whether one N can make every input easy on the selected domain." resetLabel="Reset power convergence" onReset={() => {
    setN(8);
    setDomain('closed-unit');
    setFixedPoint(0.5);
  }}>
    <div className="analysis-controls"><Select label="Function domain" value={domain} onChange={value => {
        setDomain(value);
        setFixedPoint(point => Math.min(point, 0.75));
      }} options={[['closed-unit', '[0,1]: include the endpoint'], ['open-unit', '[0,1): exclude the endpoint'], ['compact-subinterval', '[0,3/4]: stay away from 1']]} /><Range label="Power index n" value={n} onChange={setN} min={1} max={64} /><Range label="Fixed input x" value={Math.min(fixedPoint, end)} onChange={setFixedPoint} min={0} max={domain === 'open-unit' ? 0.99 : end} step={0.01} /></div>
    <Plot label="Power curve with fixed input and an analytical moving witness" xRange={[0, 1]} yRange={[0, 1]} xLabel="Input x" yLabel="Value x^n" series={[{
      label: 'Current power',
      points: state.points
    }]} markers={({
      x,
      y
    }) => <>
      <circle className="blue-fill" cx={x(state.fixedPoint)} cy={y(state.fixedValue)} r="4" />
      {state.witnessInDomain && <circle className="rose-fill" cx={x(state.witness)} cy={y(0.5)} r="4" />}
      <circle className={domain === 'open-unit' ? 'open-point' : 'gold-fill'} cx={x(end)} cy={y(end ** n)} r="3.4" />
    </>} />
    <div className="analysis-readouts"><Readout label="Blue: fixed-point error">{number(state.fixedError)}</Readout><Readout label="Analytical supremum error">{number(state.supremumError)}</Readout></div>
    <p className="analysis-result">{domain === 'compact-subinterval' ? `Every error is at most (3/4)^${n}=${number(state.supremumError)}. The common bound approaches zero. The rose witness ${state.witnessInDomain ? 'still lies' : 'now lies outside'} this smaller domain.` : `The rose point moves to x_n=2^(−1/n)≈${number(state.witness)} and its error stays exactly 1/2. The supremum error is 1, approached near the endpoint but not attained.`}</p>
    <p>{domain === 'closed-unit' ? 'At x=1 the limiting value is 1, so that point has zero error; immediately to its left the pointwise limit is zero.' : 'The pointwise limit is zero everywhere in this domain.'} Gold is the calculated current curve; blue and rose are different ways of choosing an input.</p>
  </Investigation>;
}
export function MovingTriangleLab() {
  const [n, setN] = useState(64);
  const [scaling, setScaling] = useState('unit-height');
  const [grid, setGrid] = useState(20);
  const state = triangleFamily(n, scaling, grid);
  const height = state.height;
  return <Investigation title="Find the error that a coarse grid misses" prediction="A narrow triangle can fall between the sampled points. Predict its actual height and area before treating a sampled zero as a guarantee." resetLabel="Reset moving triangle" onReset={() => {
    setN(64);
    setScaling('unit-height');
    setGrid(20);
  }}>
    <div className="analysis-controls"><Select label="Triangle index n" value={n} onChange={value => setN(Number(value))} options={[4, 16, 64, 256].map(value => [value, value])} /><Select label="Triangle amplitude" value={scaling} onChange={setScaling} options={[['unit-height', 'T_n: height stays 1'], ['unit-area', 'n T_n: area stays 1'], ['shrinking-height', 'T_n/n: height shrinks']]} /><Select label="Coarse grid intervals" value={grid} onChange={value => setGrid(Number(value))} options={[10, 20, 80].map(value => [value, value])} /></div>
    <div className="analysis-pair">
      <Plot label="Exact full-domain triangle and deliberately coarse sample positions" xRange={[0, 1]} yRange={[0, height]} xLabel="Full-domain x" yLabel="Function value" series={[{
        points: state.corners
      }]} markers={({
        x,
        y
      }) => state.samples.map((point, index) => <circle key={index} className="blue-fill" cx={x(point.x)} cy={y(point.y)} r="2.2" />)} />
      <Plot label="The same triangle in a labelled local coordinate" xRange={[0, 2]} yRange={[0, height]} xLabel="Local coordinate u = n x" yLabel="Same function value" series={[{
        points: [{
          x: 0,
          y: 0
        }, {
          x: 1,
          y: height
        }, {
          x: 2,
          y: 0
        }]
      }]} markers={({
        x,
        y
      }) => <circle className="rose-fill" cx={x(1)} cy={y(height)} r="3.5" />} />
    </div>
    <div className="analysis-readouts"><Readout label="Sampled maximum">{number(state.sampledMaximum)}</Readout><Readout label="True supremum error">{number(height)}</Readout><Readout label="Integral / L1 error">{number(state.integral)}</Readout><Readout label="Squared L2 error">{number(state.squaredL2Error)}</Readout></div>
    <p className="analysis-result">The peak is at x=1/{n}; the support ends at x=2/{n}. {state.sampledMaximum === 0 ? 'Every coarse sample is zero, but the exact triangle is still present.' : 'This grid catches some of the triangle; its maximum still cannot certify all real inputs.'}</p>
    <p>The local view changes the horizontal coordinate, not the function. Both vertical scales show the same numerical height. Every fixed x eventually sees zero; a peak chosen separately for each n can remain large.</p>
  </Investigation>;
}
export function DerivativeConvergenceLab() {
  const [n, setN] = useState(4);
  const [power, setPower] = useState(1);
  const [point, setPoint] = useState(0);
  const state = derivativeFamily(n, power, point);
  const curves = useMemo(() => Array.from({
    length: 385
  }, (_, index) => {
    const x = -Math.PI + 2 * Math.PI * index / 384;
    return {
      x,
      ...derivativeFamily(n, power, x)
    };
  }), [n, power]);
  return <Investigation title="A small curve can retain a large slope" prediction="At x=0 the sine is zero for every n. Predict the slope there, then change the amplitude from 1/n to 1/n²." resetLabel="Reset derivative comparison" onReset={() => {
    setN(4);
    setPower(1);
    setPoint(0);
  }}>
    <div className="analysis-controls"><Select label="Oscillation index n" value={n} onChange={value => setN(Number(value))} options={[1, 4, 8, 16].map(value => [value, value])} /><Select label="Amplitude rule" value={power} onChange={value => setPower(Number(value))} options={[[1, 'sin(nx)/n'], [2, 'sin(nx)/n²']]} /><Range label="Inspect x" value={point} onChange={setPoint} min={-3} max={3} step={0.1} /></div>
    <div className="analysis-pair">
      <Plot label="Function values on a fixed vertical scale" xRange={[-Math.PI, Math.PI]} yRange={[-1, 1]} xLabel="Input x (radians)" yLabel="Function value" series={[{
        points: curves.map(value => ({
          x: value.x,
          y: value.value
        }))
      }]} background={({
        x,
        y,
        right
      }) => <rect className="band" x={x(-Math.PI)} y={y(state.functionBound)} width={right - x(-Math.PI)} height={y(-state.functionBound) - y(state.functionBound)} />} markers={({
        x,
        y
      }) => <circle className="blue-fill" cx={x(point)} cy={y(state.value)} r="3.5" />} />
      <Plot label="Derivative values on the same fixed numerical scale" xRange={[-Math.PI, Math.PI]} yRange={[-1, 1]} xLabel="Input x (radians)" yLabel="Derivative" series={[{
        className: 'rose',
        points: curves.map(value => ({
          x: value.x,
          y: value.derivative
        }))
      }]} background={({
        x,
        y,
        right
      }) => <rect className="band" x={x(-Math.PI)} y={y(state.derivativeBound)} width={right - x(-Math.PI)} height={y(-state.derivativeBound) - y(state.derivativeBound)} />} markers={({
        x,
        y
      }) => <circle className="blue-fill" cx={x(point)} cy={y(state.derivative)} r="3.5" />} />
    </div>
    <div className="analysis-readouts"><Readout label="Function error bound">{number(state.functionBound)}</Readout><Readout label="Derivative error bound">{number(state.derivativeBound)}</Readout><Readout label="Selected derivative">{number(state.derivative)}</Readout></div>
    <p className="analysis-result">{power === 1 ? 'The function converges uniformly to zero, but its derivative at zero is always 1. Function convergence alone cannot justify this derivative interchange.' : 'Both bounds approach zero, and the base value at x=0 is fixed at zero. The stronger derivative theorem applies.'}</p>
  </Investigation>;
}
export function SeriesEndpointLab() {
  const [n, setN] = useState(32);
  const [x, setX] = useState(1);
  const state = inverseSquareSeries(n, x);
  return <Investigation title="A function-series bound does not control its derivative" prediction="The same finite terms become larger when differentiated at x=1. Predict what changes when x moves into the interior." resetLabel="Reset series endpoint" onReset={() => {
    setN(32);
    setX(1);
  }}>
    <div className="analysis-controls"><Select label="Number of series terms" value={n} onChange={value => setN(Number(value))} options={[8, 32, 128, 512].map(value => [value, value])} /><Select label="Series evaluation point" value={x} onChange={value => setX(Number(value))} options={[[-1, '−1 (left endpoint)'], [-0.5, '−0.5 (interior)'], [0, '0 (centre)'], [0.5, '+0.5 (interior)'], [1, '+1 (right endpoint)']]} /></div>
    <div className="analysis-operation"><div><span>Function contribution</span><strong>xᵏ / k²</strong></div><span aria-hidden="true">→</span><div><span>Differentiate each finite term</span><strong>xᵏ⁻¹ / k</strong></div></div>
    <div className="analysis-readouts"><Readout label="Finite function sum">{number(state.sum, 8)}</Readout><Readout label="Finite derivative sum">{number(state.derivativeSum, 8)}</Readout><Readout label="Uniform function tail bound">1/{n}</Readout></div>
    <p className="analysis-result">{x === 1 ? 'The derivative sum is harmonic here. It diverges even though the original function series converges uniformly on [−1,1]. The interior theorem does not include this endpoint.' : Math.abs(x) < 1 ? 'This point is strictly inside the radius. A smaller compact interval around it permits the derivative-series theorem. The displayed 1/n bound still belongs to the function, not the derivative.' : 'At −1 the derivative terms alternate and converge conditionally. That endpoint needs a separate argument; the two endpoints need not behave alike.'}</p>
    <p>The finite sum is shown, with a proved tail bound. No infinite value was inferred by drawing a smooth curve through a few partial sums.</p>
  </Investigation>;
}
export function BernsteinApproximationLab() {
  const [n, setN] = useState(16);
  const [corner, setCorner] = useState(0.3);
  const [point, setPoint] = useState(0.4);
  const state = bernsteinApproximation(n, point, corner);
  const curve = useMemo(() => Array.from({
    length: 101
  }, (_, index) => {
    const value = bernsteinApproximation(n, index / 100, corner);
    return {
      x: index / 100,
      y: value.approximation
    };
  }), [n, corner]);
  const shownNodes = state.nodes.filter(node => node.weight >= 0.0001);
  const omittedWeight = state.nodes.filter(node => node.weight < 0.0001).reduce((sum, node) => sum + node.weight, 0);
  const greatestWeight = Math.max(...state.nodes.map(node => node.weight));
  return <Investigation title="Build an approximation from nearby weighted values" prediction="The target has a corner, so a Taylor expansion there is unavailable. Predict how positive weights can still build a smooth polynomial approximation." resetLabel="Reset polynomial approximation" onReset={() => {
    setN(16);
    setCorner(0.3);
    setPoint(0.4);
  }}>
    <div className="analysis-controls"><Select label="Polynomial degree n" value={n} onChange={value => setN(Number(value))} options={[4, 16, 64, 100].map(value => [value, value])} /><Select label="Corner location c" value={corner} onChange={value => setCorner(Number(value))} options={[0.3, 0.5, 0.7].map(value => [value, value])} /><Range label="Weighted evaluation x" value={point} onChange={setPoint} min={0} max={1} step={0.01} /></div>
    <Plot label="Corner target, calculated Bernstein polynomial and its analytical error band" xRange={[0, 1]} yRange={[-0.3, 1.1]} xLabel="Input x" yLabel="Target and polynomial" series={[{
      className: 'blue',
      points: [{
        x: 0,
        y: corner
      }, {
        x: corner,
        y: 0
      }, {
        x: 1,
        y: 1 - corner
      }]
    }, {
      points: curve
    }]} background={({
      x,
      y
    }) => <polygon className="band" points={[{
      x: 0,
      y: corner + state.uniformBound
    }, {
      x: corner,
      y: state.uniformBound
    }, {
      x: 1,
      y: 1 - corner + state.uniformBound
    }, {
      x: 1,
      y: 1 - corner - state.uniformBound
    }, {
      x: corner,
      y: -state.uniformBound
    }, {
      x: 0,
      y: corner - state.uniformBound
    }].map(value => `${x(value.x)},${y(value.y)}`).join(' ')} />} markers={({
      x,
      y
    }) => <circle className="rose-fill" cx={x(point)} cy={y(state.approximation)} r="4" />} />
    <figure className="analysis-inline"><div className="analysis-weight-strip" role="img" aria-label={`Binomial weights at x=${point}; ${shownNodes.length} weights at least 0.0001 shown`}>{shownNodes.map(node => <span key={node.k} style={{
          height: `${100 * node.weight / greatestWeight}%`
        }} title={`node ${number(node.location)}, weight ${number(node.weight)}, value ${number(node.value)}`} />)}</div><figcaption>Weights rise near the selected x. Bar heights are relative to the largest weight. Shown node locations span {number(shownNodes[0].location)}–{number(shownNodes.at(-1).location)}; omitted tiny weights total {number(omittedWeight)}. All weights contribute to the calculation.</figcaption></figure>
    <div className="analysis-readouts"><Readout label="Blue target at x">{number(state.target)}</Readout><Readout label="Gold weighted value">{number(state.approximation)}</Readout><Readout label="Actual selected error">{number(state.error)}</Readout><Readout label="Whole-interval bound">{number(state.uniformBound)}</Readout></div>
    <p className="analysis-result">The weights sum to {number(state.weightSum)} and their weighted node mean is {number(state.weightedMean)}. Since this target has Lipschitz constant 1, every error is at most 1/(2√{n}). The shaded band comes from that proof, not a sampled maximum.</p>
    <p>This bounded investigation evaluates finite polynomials numerically. It reuses the computed curve when only the inspection point changes. The native example verifies the weight identities with exact fractions.</p>
  </Investigation>;
}
export function TypewriterConvergenceLab() {
  const [block, setBlock] = useState(2);
  const [position, setPosition] = useState(0);
  const [observer, setObserver] = useState(0.5);
  const state = typewriterInterval(block, position, observer);
  return <Investigation title="A shrinking chance can keep revisiting one observer" prediction="Each block sweeps across [0,1) once. Predict whether your fixed observer can eventually avoid every highlighted interval." resetLabel="Reset interval sweep" onReset={() => {
    setBlock(2);
    setPosition(0);
    setObserver(0.5);
  }}>
    <div className="analysis-controls"><Select label="Dyadic block k" value={block} onChange={value => {
        setBlock(Number(value));
        setPosition(0);
      }} options={[1, 2, 3, 4, 5].map(value => [value, value])} /><Range label="Interval position j" value={position} onChange={setPosition} min={0} max={2 ** block - 1} /><Select label="Fixed observer U" value={observer} onChange={value => setObserver(Number(value))} options={[[0, '0'], [0.25, '1/4'], [0.5, '1/2'], [0.75, '3/4']]} /></div>
    <figure className="analysis-inline"><svg viewBox="0 0 320 100" role="img" aria-label={`Highlighted half-open interval [${number(state.left)}, ${number(state.right)}), observer ${observer}`}>
      <line className="axis" x1="24" x2="296" y1="50" y2="50" />
      <rect className="interval-fill" x={24 + 272 * state.left} y="33" width={272 * (state.right - state.left)} height="34" />
      <circle className="gold-fill" cx={24 + 272 * state.left} cy="50" r="3" /><circle className="open-point" cx={24 + 272 * state.right} cy="50" r="3" />
      <path className="rose" d={`M${24 + 272 * observer} 17v58`} /><text x="24" y="92">0</text><text x="296" y="92" textAnchor="end">1</text>
    </svg><figcaption>Gold includes the left endpoint and excludes the right. The rose line is the same fixed observer throughout the sweep.</figcaption></figure>
    <div className="analysis-readouts"><Readout label="Global index n=2ᵏ+j">{state.n}</Readout><Readout label="Probability of value 1">1/{state.intervals}</Readout><Readout label="Value at this observer">{state.observed}</Readout><Readout label="Observer is hit at j">{state.observerVisitPosition}</Readout></div>
    <p className="analysis-result">Current interval: [{number(state.left)}, {number(state.right)}). Every block has exactly one interval containing this observer and other intervals that miss it. The per-index probability tends to zero, yet the fixed observer has infinitely many hits and misses.</p>
    <p>This is an exact dyadic construction, not a random simulation. The optional proof explains why the finite sweep represents a pattern across every block.</p>
  </Investigation>;
}
export function ContinuityDomainFigure() {
  return <figure className="analysis-inline" data-analysis-figure="continuity-domain"><div className="analysis-operation"><div><span>On [0,2]</span><strong>|x²−y²| ≤ 4|x−y|</strong><p>One input radius works everywhere in this bounded domain.</p></div><div><span>On the whole real line</span><strong>x=n, y=n+1/n</strong><p>Input gap 1/n → 0; output gap 2+1/n² → 2.</p></div></div><figcaption>The function is continuous in both settings. The domain determines whether one common input radius suffices.</figcaption></figure>;
}
export function UniformErrorPathFigure() {
  return <figure className="analysis-inline" data-analysis-figure="uniform-error-path"><ol className="analysis-proof-path"><li><strong>f(x) → f_N(x)</strong><span>Uniform approximation: less than ε/3</span></li><li><strong>f_N(x) → f_N(a)</strong><span>Continuity of one fixed f_N: less than ε/3</span></li><li><strong>f_N(a) → f(a)</strong><span>Uniform approximation: less than ε/3</span></li></ol><figcaption>Choose N first, then choose δ for that fixed continuous function. The three errors together stay below ε.</figcaption></figure>;
}
