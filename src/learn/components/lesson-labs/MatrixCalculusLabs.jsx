import { useId, useState } from 'react';
import { affineFixtures, affineGradientState, calculusNumber as number, chainDerivatives, differenceCheck, differenceSteps, localApproximation } from '../../data/matrix-calculus-models.js';
import './matrix-calculus-labs.css';
const pair = values => '(' + values.map(number).join(', ') + ')';
function Investigation({
  id,
  title,
  children
}) {
  return <section className="calculus-investigation" data-lab={id} aria-label={title}>
      <h3>{title}</h3>
      {children}
    </section>;
}
function Matrix({
  values,
  caption,
  selectedRow,
  selectedColumn,
  onSelect,
  selectedCell
}) {
  return <div className="calculus-matrix-region" role="region" tabIndex={0} aria-label={caption}>
      <table className="calculus-matrix">
        <caption>{caption}</caption>
        <tbody>{values.map((row, rowIndex) => <tr key={rowIndex}>{row.map((value, columnIndex) => <td key={columnIndex} className={selectedRow === rowIndex || selectedColumn === columnIndex || selectedCell?.[0] === rowIndex && selectedCell?.[1] === columnIndex ? 'is-selected' : ''}>
              {onSelect ? <button type="button" aria-label={caption + ' row ' + rowIndex + ', column ' + columnIndex} aria-pressed={selectedCell?.[0] === rowIndex && selectedCell?.[1] === columnIndex} onClick={() => onSelect([rowIndex, columnIndex])}>
                  {number(value)}
                </button> : number(value)}
            </td>)}</tr>)}</tbody>
      </table>
    </div>;
}
function Plot({
  series,
  label,
  xLabel,
  yLabel,
  xDomain,
  marker,
  logarithmic = false
}) {
  const allValues = series.flatMap(line => line.points.map(point => point[1]));
  const minimum = Math.min(0, ...allValues);
  const maximum = Math.max(0, ...allValues);
  const margin = Math.max(0.25, (maximum - minimum) * 0.12);
  const low = minimum - margin;
  const high = maximum + margin;
  const x = value => 52 + (value - xDomain[0]) / (xDomain[1] - xDomain[0]) * 262;
  const y = value => 242 - (value - low) / (high - low) * 188;
  const ticks = [low, (low + high) / 2, high];
  return <svg className="calculus-plot" viewBox={`0 0 340 ${logarithmic ? 332 : 310}`} role="img" aria-label={label} data-domain={JSON.stringify({
    x: xDomain,
    y: [low, high]
  })}>
      <title>{label}</title>
      {ticks.map((value, index) => <g className="calculus-grid" key={index}>
          <line x1="52" y1={y(value)} x2="314" y2={y(value)} />
          <text x="46" y={y(value) + 4} textAnchor="end">{Number(value.toFixed(1))}</text>
        </g>)}
      <path className="calculus-axis" d="M52,44 V242 H314" />
      <text x="54" y="25">{yLabel}</text>
      <text x="183" y={logarithmic ? 320 : 295} textAnchor="middle">{xLabel}</text>
      {[xDomain[0], (xDomain[0] + xDomain[1]) / 2, xDomain[1]].map((value, index) => <text key={index} x={x(value)} y="265" textAnchor="middle">{number(value)}</text>)}
      {series.map(line => <path key={line.name} className={'calculus-curve ' + line.kind} d={line.points.map((point, index) => (index === 0 ? 'M' : 'L') + x(point[0]) + ',' + y(point[1])).join(' ')} data-series={line.name} />)}
      {marker && <g>
          <line className="calculus-marker" x1={x(marker.x)} x2={x(marker.x)} y1="44" y2="242" />
          {marker.values.map((entry, index) => {
        const value = typeof entry === 'number' ? entry : entry.value;
        const kind = typeof entry === 'number' ? index === 0 ? 'actual' : 'linear' : entry.kind;
        return <circle key={index} cx={x(marker.x)} cy={y(value)} r="4" className={'calculus-mark ' + kind} />;
      })}
        </g>}
      {logarithmic && <text x="54" y="282" className="calculus-axis-note">−6 means an error of 0.000001</text>}
    </svg>;
}
export function JacobianMeaningFigure() {
  return <figure className="calculus-inline">
      <figcaption>One table links every input to every output</figcaption>
      <table className="calculus-labelled-jacobian">
        <thead><tr><th>At x=(2,3)</th><th>Change x₁</th><th>Change x₂</th></tr></thead>
        <tbody>
          <tr><th>Output f₁=x₁²+x₂</th><td>4</td><td>1</td></tr>
          <tr><th>Output f₂=x₁x₂</th><td>3</td><td>2</td></tr>
        </tbody>
      </table>
      <p>Across row 1: Δf₁≈4Δx₁+Δx₂. Down column 1: moving only x₁ produces output changes proportional to (4,3). These are slopes at this point, not output values.</p>
    </figure>;
}
export function LocalJacobianLab() {
  const controlsId = useId();
  const [input, setInput] = useState([2, 3]);
  const [directionKey, setDirectionKey] = useState('mixed');
  const [step, setStep] = useState(0.25);
  const directions = {
    mixed: [1, -2],
    first: [1, 0],
    second: [0, 1],
    zero: [0, 0]
  };
  const direction = directions[directionKey];
  const state = localApproximation(input, direction, step);
  const samples = Array.from({
    length: 81
  }, (_, index) => localApproximation(input, direction, -0.5 + index / 80));
  const reset = () => {
    setInput([2, 3]);
    setDirectionKey('mixed');
    setStep(0.25);
  };
  return <Investigation id="local-jacobian" title="How far can the tangent prediction travel?">
      <p>We move along x+t·v. Inspect whether halving t halves the error, then compare t=0.25 and 0.125. Gold is actual Δf; dashed blue is tJv. Changing controls applies immediately. Both curves in each plot share axes; the two outputs can have different vertical ranges.</p>
      <div className="calculus-controls">
        {input.map((value, index) => <label key={index} htmlFor={`${controlsId}-base-${index}`}>Base x{index + 1}<output>{number(value)}</output><input id={`${controlsId}-base-${index}`} aria-label={`Base x${index + 1}`} type="range" min="-3" max="3" step="0.5" value={value} onChange={event => setInput(input.map((entry, coordinate) => coordinate === index ? Number(event.target.value) : entry))} /></label>)}
        <label>Direction v<select value={directionKey} onChange={event => setDirectionKey(event.target.value)}><option value="mixed">(1,−2): both inputs</option><option value="first">(1,0): first input only</option><option value="second">(0,1): second input only</option><option value="zero">(0,0): no movement</option></select></label>
        <label>Selected step t<select value={step} onChange={event => setStep(Number(event.target.value))}>{[-0.5, -0.25, 0, 0.01, 0.125, 0.25, 0.5].map(value => <option key={value} value={value}>{value}</option>)}</select></label>
      </div>
      <div className="calculus-two-plots">
        {[0, 1].map(output => <Plot key={output} label={'Output ' + (output + 1) + ': actual change and tangent prediction along the selected direction'} xLabel="Step t along v" yLabel={'Change in f' + (output + 1)} xDomain={[-0.5, 0.5]} series={[{
        name: 'actual',
        kind: 'actual',
        points: samples.map(sample => [sample.step, sample.actual[output]])
      }, {
        name: 'linear',
        kind: 'linear',
        points: samples.map(sample => [sample.step, sample.predicted[output]])
      }]} marker={{
        x: step,
        values: [state.actual[output], state.predicted[output]]
      }} />)}
      </div>
      <div className="calculus-results">
        <Matrix values={state.jacobian} caption="J at the base point" />
        <div aria-live="polite" data-result="local">
          <p>Base f(x)={pair(state.base)}; moved input={pair(state.shiftedInput)}.</p>
          <p>Rate Jv=<strong>{pair(state.rate)}</strong>. Predicted change=<strong>{pair(state.predicted)}</strong>.</p>
          <p>Actual change=<strong>{pair(state.actual)}</strong>. Actual − predicted=<strong>{pair(state.error)}</strong>.</p>
        </div>
      </div>
      <p>The exact remainder here is (t²v₁²,t²v₁v₂). For v=(0,1) it is zero: this nonlinear function is exactly linear along that chosen line. That does not make the entire function linear. Readouts use floating-point arithmetic, so an exact zero can appear as a tiny numerical remainder.</p>
      <button type="button" onClick={reset}>Reset local experiment</button>
      <details><summary>Transfer: a zero Jacobian entry is not “no dependence”</summary><p>Set x=(0,0), v=(1,0). The first local slope is zero, but actual Δf₁=t². A zero first derivative can hide second-order change; it does not prove independence.</p></details>
    </Investigation>;
}
export function BranchSumFigure() {
  return <figure className="calculus-inline">
      <figcaption>One input can influence the loss along two paths</figcaption>
      <div className="calculus-branch">
        <strong>x=2</strong>
        <div><p>→ a=x²=4 → ∂L/∂a=1 → contribution 1·2x=4</p><p>→ b=3x=6 → ∂L/∂b=1 → contribution 1·3=3</p></div>
        <strong>L=a+b=10<br />∂L/∂x=4+3=7</strong>
      </div>
      <p>Backpropagation adds contributions at a shared input. Overwriting 4 with 3 loses one path. The forward values a+b and the backward derivative 4+3 answer different questions.</p>
    </figure>;
}
export function ChainRuleLab() {
  const [input, setInput] = useState([1, 2]);
  const [mode, setMode] = useState('forward');
  const [seed, setSeed] = useState('first');
  const [weightKey, setWeightKey] = useState('difference');
  const [stage, setStage] = useState(0);
  const directions = {
    first: [1, 0],
    second: [0, 1],
    mixed: [1, -1],
    zero: [0, 0]
  };
  const weights = {
    difference: [1, -1],
    first: [1, 0],
    sum: [1, 1],
    zero: [0, 0]
  };
  const state = chainDerivatives(input, directions[seed], weights[weightKey]);
  const forward = [{
    name: 'x',
    value: pair(input),
    derivative: 'v=' + pair(state.direction)
  }, {
    name: 'y=Ax',
    value: pair(state.intermediate),
    derivative: 'Av=' + pair(state.intermediateTangent)
  }, {
    name: 'z=y⊙y',
    value: pair(state.squared),
    derivative: '2y⊙(Av)=' + pair(state.squaredTangent)
  }, {
    name: 'L=q·z',
    value: number(state.value),
    derivative: 'q·dz=' + number(state.lossTangent)
  }];
  const reverse = [{
    name: 'L=q·z',
    value: number(state.value),
    derivative: 'Seed ∂L/∂L=1'
  }, {
    name: 'z=y⊙y',
    value: pair(state.squared),
    derivative: 'g_z=q=' + pair(state.outputWeights)
  }, {
    name: 'y=Ax',
    value: pair(state.intermediate),
    derivative: 'g_y=2y⊙q=' + pair(state.intermediateGradient)
  }, {
    name: 'x',
    value: pair(input),
    derivative: 'g_x=Aᵀg_y=' + pair(state.inputGradient)
  }];
  const flow = mode === 'forward' ? forward : reverse;
  const reset = () => {
    setInput([1, 2]);
    setMode('forward');
    setSeed('first');
    setWeightKey('difference');
    setStage(0);
  };
  return <Investigation id="chain-rule" title="Send a direction forward, or bring a gradient back">
      <p>Fixed A=[[1,2],[−1,1]], then square each component. The weighted scalar L=q·z is a teaching objective, not necessarily a nonnegative training loss. The forward values are always visible. Inspect the derivative at the next node as you propagate it. Editing a control resets the derivative trace.</p>
      <div className="calculus-controls">
        <label>Question<select value={mode} onChange={event => {
          setMode(event.target.value);
          setStage(0);
        }}><option value="forward">Forward input direction</option><option value="reverse">Reverse loss gradient</option></select></label>
        <label>Input x<select value={input.join(',')} onChange={event => {
          setInput(event.target.value.split(',').map(Number));
          setStage(0);
        }}><option value="1,2">(1,2)</option><option value="0,0">(0,0)</option><option value="-1,1">(−1,1)</option><option value="2,-1">(2,−1)</option></select></label>
        <label>Output weights q<select value={weightKey} onChange={event => {
          setWeightKey(event.target.value);
          setStage(0);
        }}><option value="difference">(1,−1): difference</option><option value="first">(1,0): only z₁</option><option value="sum">(1,1): sum</option><option value="zero">(0,0): constant zero</option></select></label>
        <label>Input direction v<select value={seed} onChange={event => {
          setSeed(event.target.value);
          setStage(0);
        }}>{Object.entries(directions).map(([key, direction]) => <option key={key} value={key}>{pair(direction)}</option>)}</select></label>
      </div>
      <p className="calculus-direction">{mode === 'forward' ? 'Forward derivative flow: x → y → z → L' : 'Reverse derivative flow: L → z → y → x'}</p>
      <ol className="calculus-chain" data-mode={mode}>
        {flow.map((node, index) => <li key={node.name} className={index === stage ? 'is-active' : ''}>
            <strong>{node.name}</strong><span>Value: {node.value}</span>
            <span className="calculus-derivative">{index <= stage ? node.derivative : 'Derivative not propagated yet'}</span>
          </li>)}
      </ol>
      <div className="calculus-actions"><button type="button" disabled={stage === 0} onClick={() => setStage(stage - 1)}>Previous stage</button><button type="button" disabled={stage === 3} onClick={() => setStage(stage + 1)}>Propagate one stage</button><button type="button" onClick={() => setStage(3)}>Show complete trace</button><button type="button" onClick={reset}>Reset chain</button></div>
      <p aria-live="polite" data-result="chain">Stage {stage + 1} of 4. {stage === 3 ? 'Both routes agree: g_x·v=' + number(state.inputGradient.reduce((sum, value, index) => sum + value * state.direction[index], 0)) + ', and the forward rate is ' + number(state.lossTangent) + '.' : 'Only derivative annotations through this stage are revealed.'}</p>
      {mode === 'reverse' && stage === 3 && <div className="calculus-contributions"><p>First input: {state.contributions[0].map(number).join(' + ')} = {number(state.inputGradient[0])}.</p><p>Second input: {state.contributions[1].map(number).join(' + ')} = {number(state.inputGradient[1])}. Each input affects both components of y; sum both paths.</p></div>}
      <details><summary>Transfer: why does zero v not mean zero gradient?</summary><p>In forward mode set v=(0,0): no requested movement, so all tangent values are zero. Switch to reverse without changing x or q: the gradient can remain nonzero. It describes available sensitivity, not the particular movement you chose.</p></details>
    </Investigation>;
}
export function SharedBiasFigure() {
  return <figure className="calculus-inline">
      <figcaption>A shared number receives contributions from every place it was used</figcaption>
      <div className="calculus-shared-bias">
        <p><strong>Forward</strong><br />b₁ → Y₀₁ and Y₁₁<br />Same bias added to both observations.</p>
        <p><strong>Backward</strong><br />G₀₁=1 and G₁₁=−2.5 → ∂L/∂b₁=−1.5<br />Their effects add, including cancellation.</p>
      </div>
      <p>Here indices start at zero to match the following array example. A mean over observations belongs in G through the objective's scaling; do not average this sum a second time.</p>
    </figure>;
}
export function AffineGradientLab() {
  const [fixtureKey, setFixtureKey] = useState('ordinary');
  const [reduction, setReduction] = useState('mean');
  const [parameter, setParameter] = useState('weight');
  const [cell, setCell] = useState([0, 0]);
  const state = affineGradientState(fixtureKey, reduction);
  const [feature, output] = cell;
  const contributions = parameter === 'weight' ? state.weightContributions[feature][output] : state.incoming.map((row, observation) => ({
    observation,
    input: 1,
    incoming: row[output],
    product: row[output]
  }));
  const gradient = contributions.reduce((sum, term) => sum + term.product, 0);
  const reset = () => {
    setFixtureKey('ordinary');
    setReduction('mean');
    setParameter('weight');
    setCell([0, 0]);
  };
  return <Investigation id="affine-gradient" title="Which observations contributed to this parameter gradient?">
      <p>Two observations, two input features and two outputs keep every contribution visible. Fixed W=[[1,−1],[2,1]], b=(0,1). L is half the squared residual sum, optionally divided by the observation count N=2. Select a parameter gradient; highlighted values explain its sum. All changes apply immediately.</p>
      <div className="calculus-controls">
        <label>Observation fixture<select value={fixtureKey} onChange={event => setFixtureKey(event.target.value)}>{Object.entries(affineFixtures).map(([key, fixture]) => <option key={key} value={key}>{fixture.label}</option>)}</select></label>
        <label>Objective reduction<select value={reduction} onChange={event => setReduction(event.target.value)}><option value="mean">Mean over N observations</option><option value="sum">Sum over observations</option></select></label>
        <label>Parameter kind<select value={parameter} onChange={event => setParameter(event.target.value)}><option value="weight">Weight W</option><option value="bias">Shared bias b</option></select></label>
      </div>
      <div className="calculus-matrices">
        <Matrix values={state.inputs} caption="X: observations × features" selectedColumn={parameter === 'weight' ? feature : undefined} />
        <Matrix values={state.targets} caption="Targets T" />
        <Matrix values={state.outputs} caption="Y=XW+b" selectedColumn={output} />
        <Matrix values={state.incoming} caption="G: incoming loss gradient" selectedColumn={output} />
        {parameter === 'weight' ? <Matrix values={state.weightGradient} caption="Select ∂L/∂W" onSelect={setCell} selectedCell={cell} /> : <Matrix values={[state.biasGradient]} caption="Select ∂L/∂b" onSelect={([, column]) => setCell([feature, column])} selectedCell={[0, output]} />}
      </div>
      <div className="calculus-contributions" aria-live="polite" data-result="affine">
        <p>L=<strong>{number(state.loss)}</strong>. G=(Y−T)/{state.divisor}. Selected {parameter === 'weight' ? 'W[' + feature + ',' + output + ']' : 'b[' + output + ']'}:</p>
        {contributions.map(term => <p key={term.observation}>Observation {term.observation}: {number(term.input)} × {number(term.incoming)} = <strong>{number(term.product)}</strong></p>)}
        <p>Sum = <strong data-gradient={gradient}>{number(gradient)}</strong>. {parameter === 'bias' ? 'The local slope from a shared bias to each matching output is 1.' : 'Each input value is the local slope from this weight to that observation’s output.'}</p>
      </div>
      <button type="button" onClick={reset}>Reset batch</button>
      <details><summary>Transfer: duplicate a training observation</summary><p>Under the mean objective, duplicating the same observation leaves its average contribution unchanged. Under the sum objective it doubles. With zero inputs every weight gradient is zero, but bias gradients can remain nonzero. This explains the fixture effects; it does not imply a zero weight gradient proves a model is fitted.</p></details>
    </Investigation>;
}
export function FiniteDifferenceLab() {
  const [kind, setKind] = useState('cubic');
  const [point, setPoint] = useState(2);
  const [index, setIndex] = useState(3);
  const rows = differenceSteps.map(step => differenceCheck(kind, point, step));
  const state = rows[index];
  const reset = () => {
    setKind('cubic');
    setPoint(2);
    setIndex(3);
  };
  const positiveErrors = key => rows.filter(row => row[key] > 0).map(row => [Math.log10(row.step), Math.log10(row[key])]).reverse();
  return <Investigation id="finite-difference" title="A smaller numerical step is not always a better check">
      <p>Compare independent differences with the known derivative. These are actual JavaScript double-precision calculations, not illustrative error rankings. Inspect what happens at |x| with x=0 when selecting it. Changes apply immediately.</p>
      <div className="calculus-controls">
        <label>Function<select value={kind} onChange={event => setKind(event.target.value)}><option value="cubic">f(x)=x³: smooth</option><option value="absolute">f(x)=|x|: kink at zero</option></select></label>
        <label>Base point x<select value={point} onChange={event => setPoint(Number(event.target.value))}>{[0, 0.3, -0.3, 2].map(value => <option key={value} value={value}>{value}</option>)}</select></label>
        <label>Step h<select value={index} onChange={event => setIndex(Number(event.target.value))}>{differenceSteps.map((value, row) => <option key={value} value={row}>{number(value)}</option>)}</select></label>
      </div>
      {state.exact !== null ? <Plot label="Computed error across finite-difference step sizes; zero errors omitted from log plot" xLabel="log₁₀(step h)" yLabel="log₁₀(absolute error)" xDomain={[-16, 0]} logarithmic series={[{
      name: 'forward-error',
      kind: 'actual',
      points: positiveErrors('forwardError')
    }, {
      name: 'central-error',
      kind: 'linear',
      points: positiveErrors('centralError')
    }]} marker={{
      x: Math.log10(state.step),
      values: [{
        value: state.forwardError,
        kind: 'actual'
      }, {
        value: state.centralError,
        kind: 'linear'
      }].filter(entry => entry.value > 0).map(entry => ({
        ...entry,
        value: Math.log10(entry.value)
      }))
    }} /> : <p role="status">The derivative at this kink is undefined. A central difference of zero is a symmetric average, not evidence of differentiability.</p>}
      {state.exact !== null && <p>Gold: forward error. Dashed blue: central error. The vertical marker selects h. Negative horizontal values mean smaller h. Zero errors have no logarithm and are omitted; numerical table values remain below. Lines only connect tested step sizes, not an error bound between them.</p>}
      <div className="calculus-contributions" aria-live="polite" data-result="difference">
        <p>Known derivative: <strong>{number(state.exact)}</strong>. h={number(state.step)}.</p>
        <p>Right/forward: <strong>{number(state.forward)}</strong>; left/backward: <strong>{number(state.backward)}</strong>; central: <strong>{number(state.central)}</strong>.</p>
        <p>Forward error: {number(state.forwardError)}; central error: {number(state.centralError)}.</p>
        {state.roundedInput && <p>At least one perturbed input rounded back to x. This check has lost the intended step.</p>}
      </div>
      <p>Readouts round to six decimal places, using scientific notation for tiny values. Errors are calculated before display rounding, so a slope printed as 12 can still have a small nonzero error.</p>
      <details><summary>Inspect every tested step</summary>
        <div className="calculus-matrix-region" role="region" tabIndex={0} aria-label="All finite difference values">
          <table className="calculus-differences"><thead><tr><th>h</th><th>Forward</th><th>Central</th><th>Central error</th></tr></thead><tbody>{rows.map(row => <tr key={row.step}><td>{number(row.step)}</td><td>{number(row.forward)}</td><td>{number(row.central)}</td><td>{number(row.centralError)}</td></tr>)}</tbody></table>
        </div>
      </details>
      <button type="button" onClick={reset}>Reset difference check</button>
      <details><summary>Transfer: why can |x| fail away from its kink?</summary><p>At x=0.3 the true derivative is 1. A large h=1 samples opposite sides of the kink, so its central slope is 0.3. Reduce h below the distance to the kink and the check samples the local branch. At extremely tiny h, rounding can still spoil it.</p></details>
    </Investigation>;
}
export function MatrixSquareFigure() {
  return <figure className="calculus-inline">
      <figcaption>For matrix multiplication, the two product-rule terms keep their order</figcaption>
      <div className="calculus-matrices">
        <Matrix values={[[1, 2], [0, 1]]} caption="A" />
        <Matrix values={[[0, 0], [1, 0]]} caption="Direction E" />
        <Matrix values={[[2, 0], [1, 0]]} caption="AE" />
        <Matrix values={[[0, 0], [1, 2]]} caption="EA" />
        <Matrix values={[[2, 0], [2, 2]]} caption="Correct: AE+EA" />
      </div>
      <p>2AE=[[4,0],[2,0]] is different. Differentiating A² is an operation on the perturbation E, not a license to use the scalar shortcut 2A. The two expressions agree only for directions that commute with A.</p>
    </figure>;
}
