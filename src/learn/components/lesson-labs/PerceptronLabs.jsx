import { useId, useState } from 'react';
import { activation, activationNames, boundarySegment, geometry, geometryInitial, sensitivity, xorRows } from '../../data/perceptron-models.js';
import { CurvePlot, fmt, Frame, NumberControl, PerceptronTable, SignedBars } from './PerceptronShared.jsx';
function BoundaryPlane({
  state,
  result
}) {
  const id = useId(),
    px = x => 38 + (x + 4) * 31,
    py = y => 286 - (y + 4) * 31,
    line = boundarySegment(state.w1, state.w2, state.b),
    foot = result.foot,
    footVisible = foot && foot.every(v => v >= -4 && v <= 4);
  return <figure className="perceptron-plot"><figcaption>Input plane: the boundary has equal coordinate scales</figcaption><svg className="perceptron-chart" viewBox="0 0 330 332" role="img" aria-labelledby={`${id}-title`}><title id={`${id}-title`}>Point ({state.x1}, {state.x2}); {result.distance === null ? 'distance undefined: zero weight vector' : `signed distance ${result.distance}`}. Positive score means hard output 1.</title><defs><clipPath id={`${id}-clip`}><rect x="38" y="38" width="248" height="248" /></clipPath></defs>{[-4, -2, 0, 2, 4].map(n => <g key={n}><line className="perceptron-grid" x1={px(n)} x2={px(n)} y1="38" y2="286" /><line className="perceptron-grid" x1="38" x2="286" y1={py(n)} y2={py(n)} /><text x={px(n)} y="306" textAnchor="middle">{n}</text><text x="26" y={py(n) + 4} textAnchor="end">{n}</text></g>)}<g clipPath={`url(#${id}-clip)`}>{line.length === 2 && <line className="perceptron-line color-1" x1={px(line[0][0])} y1={py(line[0][1])} x2={px(line[1][0])} y2={py(line[1][1])} />} {foot && <line className="perceptron-guide" x1={px(state.x1)} y1={py(state.x2)} x2={px(foot[0])} y2={py(foot[1])} />}</g>{footVisible && <rect x={px(foot[0]) - 4} y={py(foot[1]) - 4} width="8" height="8" fill="#80c6b4" />}<circle className="perceptron-dot" cx={px(state.x1)} cy={py(state.x2)} r="6" /><text x="160" y="328" textAnchor="middle">Input x₁</text><text x="38" y="18">Input x₂</text></svg><p className="perceptron-note">Gold circle: edited input. Teal line: score 0. Square: closest boundary point. Dashed segment: perpendicular distance. {result.norm === 0 ? 'Zero weights give a constant-output plane; no unique boundary exists.' : line.length !== 2 ? 'The boundary does not cross the interior of this fixed input window.' : !footVisible ? 'The closest boundary point lies outside the window; its exact distance remains in the table.' : ''} Marks are read-only; edit coordinates below.</p></figure>;
}
export function WeightedEvidenceFigure() {
  const r = geometry(geometryInitial);
  return <Frame id="perceptron-weighted-figure" title="The bias joins the sum before an activation is chosen"><SignedBars labels={['1.5 × 2', '−2 × −1', 'Bias']} values={r.products} /><div className="perceptron-flow"><div className="perceptron-node"><strong>Add contributions</strong>3 + 2 − 1 = 4</div><span className="perceptron-arrow" aria-hidden="true">→</span><div className="perceptron-node"><strong>Choose a function of 4</strong>Hard threshold: 1<br />ReLU: 4<br />Sigmoid: 0.982014</div></div><CurvePlot title="Sigmoid acts on the completed score" curves={[{data:Array.from({length:121},(_,i)=>{const z=-6+i/10;return [z,activation("sigmoid",z).value]})}]} yDomain={[0,1]} selected={4} points={[[4,r.sigmoid]]} yLabel="Sigmoid output"/><p className="perceptron-note">The three outputs use the same preactivation. The function changes how that score is expressed.</p></Frame>;
}
export function GeometryLab() {
  const [state, setState] = useState({
      ...geometryInitial
    }),
    [generation, setGeneration] = useState(0),
    [pinned, setPinned] = useState(null),
    base = geometry(state),
    scaled = geometry(state, state.c);
  const edit = (key, value) => setState(old => ({
    ...old,
    [key]: value
  }));
  const reset = () => {
    setState({
      ...geometryInitial
    });
    setPinned(null);
    setGeneration(n => n + 1);
  };
  return <Frame id="perceptron-geometry-lab" title="Investigation A · move the evidence"><p>Change a coefficient or a coordinate, then compare the original coefficients with all three coefficients multiplied by c. The input point stays the same for that comparison.</p><div className="perceptron-grid-two"><div><SignedBars labels={['w₁x₁', 'w₂x₂', 'Bias b']} values={base.products} extent={16} /><div className="perceptron-readout" data-testid="geometry-result">Unscaled score {fmt(base.score)} → scaled score {fmt(scaled.score)}<br />Hard output {scaled.hard}; distance {fmt(scaled.distance)}; sigmoid {fmt(scaled.sigmoid)}.</div></div><BoundaryPlane state={state} result={base} /></div><div className="perceptron-controls" key={generation}>{[['x1', 'Input x₁'], ['x2', 'Input x₂'], ['w1', 'Weight w₁'], ['w2', 'Weight w₂'], ['b', 'Bias b'], ['c', 'Common scale c']].map(([key, label]) => <NumberControl key={key} name={label} value={state[key]} onChange={v => edit(key, v)} min={key === 'c' ? .25 : -4} max={key === 'c' ? 3 : 4} />)}</div><div className="perceptron-actions"><button onClick={() => setPinned({
        state: {
          ...state
        },
        result: {
          ...scaled
        }
      })}>Pin current scaled case</button><button onClick={reset}>Reset evidence</button><button onClick={() => {
        setState({
          x1: 1,
          x2: 1,
          w1: 1,
          w2: -1,
          b: 0,
          c: 1
        });
        setGeneration(n => n + 1);
      }}>Start at boundary tie</button><button onClick={() => {
        setState({
          ...geometryInitial,
          w1: 0,
          w2: 0,
          b: 1
        });
        setGeneration(n => n + 1);
      }}>Zero weight vector</button></div><PerceptronTable caption="Original versus c-scaled coefficients at the current input" headers={['Quantity', 'Original', 'Scaled', 'Difference']} rows={['score', 'norm', 'distance', 'hard', 'sigmoid'].map(key => [key, fmt(base[key]), fmt(scaled[key]), base[key] === null ? 'undefined' : fmt(scaled[key] - base[key])])} /><p>{base.norm === 0 ? 'Distance is undefined because the weight vector has length zero. Changing the positive scale changes the constant score, but does not create a boundary.' : state.c === 1 ? 'At c = 1 every quantity is unchanged: this is the identity comparison.' : `Positive common scaling changes score and norm together, so their ratio stays ${fmt(base.distance)} (comparison tolerance 10⁻¹⁰). The hard decision stays ${base.hard}; sigmoid ${Math.abs(base.sigmoid - scaled.sigmoid) <= 1e-10 ? 'is unchanged at this score' : 'changes because it receives the rescaled score'}.`} To change geometry, move an input or alter individual coefficients; to change the score scale, multiply weights and bias together.</p>{pinned && <div className="perceptron-readout" data-testid="geometry-pinned">Pinned: x = ({fmt(pinned.state.x1)}, {fmt(pinned.state.x2)}), w = ({fmt(pinned.state.w1)}, {fmt(pinned.state.w2)}), b = {fmt(pinned.state.b)}, c = {fmt(pinned.state.c)}. Score {fmt(pinned.result.score)}, distance {fmt(pinned.result.distance)}, sigmoid {fmt(pinned.result.sigmoid)}.</div>}</Frame>;
}
function InputSquare({
  selected
}) {
  return <svg className="perceptron-chart" viewBox="0 0 280 210" role="img" aria-label="Four binary input corners, labelled 00, 01, 10 and 11"><path className="perceptron-grid" d="M60 155V45H220V155Z" />{[[60, 155, '00'], [60, 45, '01'], [220, 155, '10'], [220, 45, '11']].map(([x, y, label], index) => <g key={label}><circle cx={x} cy={y} r={selected === index ? 8 : 5} fill={index === 1 || index === 2 ? '#e8c36f' : '#181714'} stroke="#e8c36f" strokeWidth="2" /><text x={x} y={y === 45 ? 24 : 182} textAnchor="middle">{label}</text></g>)}<text x="140" y="206" textAnchor="middle">x₁ → ; x₂ ↑</text></svg>;
}
function XorMechanism({
  rows,
  selected,
  stage,
  extent
}) {
  const row = rows[selected];
  return <><div className="perceptron-grid-two"><div><InputSquare selected={selected} /><p className="perceptron-note">Filled corners have target 1; open corners target 0. The larger ring selects a correspondence, not a learned weight.</p></div><div><SignedBars labels={['1 × h₁', 'v × h₂']} values={[row.h1, row.second]} extent={extent} /><div className="perceptron-readout">Input {row.id}: {fmt(row.h1)} + ({fmt(row.second)}) = {fmt(row.q)}. Target {row.target}; residual {fmt(row.residual)}.</div></div></div><div className="perceptron-flow">{[['Sum', `s = ${row.x[0]} + ${row.x[1]} = ${row.sum}`], ['Hidden features', `h₁ = ${fmt(row.h1)}; h₂ = ${fmt(row.h2)}`], ['Signed contributions', `${fmt(row.h1)} and ${fmt(row.second)}`], ['Output', `q = ${fmt(row.q)}`]].map(([title, value], index) => <div key={title} className={`perceptron-node ${stage === index ? 'is-active' : ''}`}><strong>{title}</strong>{value}</div>)}</div><PerceptronTable caption="All four corners under the current manually chosen parameters" headers={['Input', 's', 'h₁', 'h₂', 'q', 'Target', 'Residual']} rows={rows.map(r => [r.id, r.sum, fmt(r.h1), fmt(r.h2), fmt(r.q), r.target, fmt(r.residual)])} /></>;
}
export function XorWorkedFigure() {
  const [stage, setStage] = useState(0);
  return <Frame id="perceptron-xor-figure" title="Two ramps construct XOR · h₁ − 2h₂"><XorMechanism rows={xorRows(-1, -2)} selected={3} stage={stage} /><div className="perceptron-actions"><button disabled={stage === 0} onClick={() => setStage(s => s - 1)}>Previous stage</button><span>Stage {stage + 1} of 4</span><button disabled={stage === 3} onClick={() => setStage(s => s + 1)}>Next stage</button></div><p className="perceptron-note">The walkthrough highlights a computation; all results remain visible. At 11 the first ramp is 2, so it is not Boolean OR.</p></Frame>;
}
export function XorLab() {
  const [bias, setBias] = useState(-1),
    [coefficient, setCoefficient] = useState(-1),
    [selected, setSelected] = useState(3),
    [generation, setGeneration] = useState(0),
    rows = xorRows(bias, coefficient),
    [pinned, setPinned] = useState(null),
    inactive = rows.every(row => row.h2 === 0);
  return <Frame id="perceptron-xor-lab" title="Investigation B · repair the whole truth table"><p>The first hidden bias stays 0 and its output weight stays 1. Edit the delayed ramp and its output coefficient. Numeric entry accepts a fraction such as −4/3.</p><XorMechanism rows={rows} selected={selected} extent={8} /><div className="perceptron-controls" key={generation}><NumberControl name="Second hidden bias" value={bias} onChange={setBias} min={-3} max={0} /><NumberControl name="Second output coefficient" value={coefficient} onChange={setCoefficient} min={-4} max={2} fractions /></div><div className="perceptron-actions"><span>Inspect input:</span>{rows.map((r, index) => <button aria-pressed={selected === index} onClick={() => setSelected(index)} key={r.id}>{r.id}</button>)}<button onClick={() => setPinned({
        bias,
        coefficient,
        values: rows.map(r => r.q)
      })}>Pin XOR case</button><button onClick={() => {
        setBias(-1);
        setCoefficient(-1);
        setSelected(3);
        setPinned(null);
        setGeneration(n => n + 1);
      }}>Reset XOR</button></div><div className="perceptron-readout" data-testid="xor-result">Outputs: {rows.map(r => fmt(r.q)).join(', ')}. Residuals: {rows.map(r => fmt(r.residual)).join(', ')}.</div><p>{inactive ? 'The second hidden feature is zero at every binary corner. Its coefficient therefore cannot change any output.' : rows.every(row => row.matches) ? 'Every row matches its target within 10⁻⁸: the second contribution cancels the both-active corner while leaving the single-active corners unchanged.' : Math.abs(bias + .5) < 1e-10 ? 'With hidden bias −0.5, matching a single-active row requires v = 0, while matching 11 requires v = −4/3. One output coefficient cannot satisfy both constraints. See practice 3 for the equations.' : 'A coefficient only changes rows where its hidden feature is active. Follow all four residuals together; fixing the selected corner alone is insufficient.'}</p>{pinned && <p className="perceptron-readout">Pinned: b₂ = {fmt(pinned.bias)}, v = {fmt(pinned.coefficient)}; outputs {pinned.values.map(v => fmt(v)).join(', ')}.</p>}</Frame>;
}
export function ActivationLab() {
  const id = useId(),
    [name, setName] = useState('relu'),
    [z, setZ] = useState(2),
    [weight, setWeight] = useState(.5),
    [advanced, setAdvanced] = useState(false),
    [generation, setGeneration] = useState(0),
    [pinned, setPinned] = useState(null),
    r = sensitivity(name, z, weight);
  const samples = Array.from({
      length: 241
    }, (_, i) => -6 + i * .05),
    values = samples.map(x => [x, activation(name, x).value]);
  const discontinuous = ['relu', 'leaky_relu'].includes(name),
    slopeCurves = discontinuous ? [{
      data: [[-6, activation(name, -1).slope], [0, activation(name, -1).slope]]
    }, {
      data: [[0, 1], [6, 1]],
      color: '#e8c36f'
    }] : [{
      data: samples.map(x => [x, activation(name, x).slope])
    }];
  const valueDomain = name === 'sigmoid' ? [0, 1] : name === 'tanh' ? [-1, 1] : [-1, 6];
  return <Frame id="perceptron-activation-lab" title="Investigation C · separate output, slope and input sensitivity"><div className="perceptron-actions"><label htmlFor={`${id}-function`}>Activation</label><select id={`${id}-function`} value={name} onChange={event => setName(event.target.value)}>{Object.entries(activationNames).filter(([key]) => advanced || ['sigmoid', 'tanh', 'relu', 'leaky_relu'].includes(key)).map(([key, title]) => <option value={key} key={key}>{title}</option>)}</select><label><input type="checkbox" checked={advanced} onChange={event => {
          setAdvanced(event.target.checked);
          if (!event.target.checked && !['sigmoid', 'tanh', 'relu', 'leaky_relu'].includes(name)) setName('relu');
        }} /> Include deeper functions from §7</label></div><div className="perceptron-grid-two"><CurvePlot title="Forward value" curves={[{
        data: values
      }]} yDomain={valueDomain} selected={z} points={[[z, r.value]]} yLabel="Activation φ(z)" /><CurvePlot title="Local slope" curves={slopeCurves} yDomain={[-.25, 1.25]} yTicks={[-.25, 0, .5, 1, 1.25]} selected={z} points={[[z, r.slope]]} hollowPoints={discontinuous ? [[0,activation(name,-1).slope],[0,1]] : []} yLabel="Slope φ′(z)" /></div><div className="perceptron-controls" key={generation}><NumberControl name="Operating point z" value={z} onChange={setZ} min={-6} max={6} step={.1} /><NumberControl name="Incoming scalar weight" value={weight} onChange={setWeight} min={-4} max={4} /></div><div className="perceptron-flow"><div className="perceptron-node"><strong>Small input change Δx</strong>Multiply by w = {fmt(weight)}</div><span className="perceptron-arrow" aria-hidden="true">→</span><div className="perceptron-node"><strong>Near z = {fmt(z)}</strong>Multiply by slope {fmt(r.slope)}</div><span className="perceptron-arrow" aria-hidden="true">→</span><div className="perceptron-node"><strong>Small output change Δa</strong>≈ {fmt(r.sensitivity)} × Δx</div></div><div className="perceptron-readout" data-testid="sensitivity-result">Value = {fmt(r.value)}; slope = {fmt(r.slope)}; weight × slope = {fmt(r.sensitivity)} ({r.category}).</div><p>{r.corner ? `${activationNames[name]} has no ordinary derivative at zero; the plotted point records the PyTorch differentiation convention ${fmt(r.slope)}. The separate horizontal segments do not join across the jump. ` : ''}The operating point z stays fixed when w changes, as if a compatible bias adjusted with it. This isolates the product wφ′(z). The sign categories use tolerance 10⁻¹⁰; “at least 1” refers to signed sensitivity, not its absolute magnitude.</p><div className="perceptron-actions"><button onClick={() => setPinned({
        name,
        z,
        weight,
        ...r
      })}>Pin sensitivity</button><button onClick={() => {
        setName('relu');
        setZ(2);
        setWeight(.5);
        setAdvanced(false);
        setPinned(null);
        setGeneration(n => n + 1);
      }}>Reset sensitivity</button></div>{pinned && <p className="perceptron-readout">Pinned {activationNames[pinned.name]} at z = {fmt(pinned.z)}, w = {fmt(pinned.weight)}: value {fmt(pinned.value)}, slope {fmt(pinned.slope)}, sensitivity {fmt(pinned.sensitivity)}.</p>}</Frame>;
}
