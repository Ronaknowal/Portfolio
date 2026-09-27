import { useState } from 'react';
import { NeuralNumber } from './NeuralNumberControl.jsx';
import { NeuralLab, NeuralSelect, NeuralTable } from './NeuralLessonElements.jsx';
import { convolution, difference, dot, kinkValue, libraryAccess, linearMargin, linearPenalty, matvec, norm, normalizationGradient, normalizeMatrix, penaltyValue, powerTrace, probePenalty, scale, singular2 } from '../../data/spectral-regularization-models.js';
import { CurvePlane, MatrixEditor, SpectralFigure, SpectralMatrix, SpectralPlot, SpectralVector, f, vector } from './SpectralPrimitives.jsx';
import './spectral-regularization-labs.css';
import './neural-lesson-neutral.css';
const color = ['#e6b854', '#ddd', '#ba9fd1'];
const circle = Array.from({
  length: 97
}, (_, i) => [Math.cos(i * Math.PI / 48), Math.sin(i * Math.PI / 48)]);
const methods = [['exact', 'Exact spectral normalization'], ['power', 'Power-estimated normalization'], ['cap', 'Cap spectral norm only'], ['frobenius', 'Frobenius normalization'], ['entry', 'Entrywise clipping'], ['singular-cap', 'Clip each singular value']];
const kinds = [['target-one', 'Target one: λ(r−1)²'], ['one-sided', 'One-sided: λ max(0,r−1)²'], ['zero', 'Zero-centered: λr²']];
function SpectrumBars({
  raw,
  effective
}) {
  const a = singular2(raw),
    b = singular2(effective),
    values = [[...a.values, a.frobenius], [...b.values, b.frobenius]],
    bound = Math.max(1, ...values.flat());
  return <SpectralFigure title="Singular values describe individual stretches; Frobenius combines their squares" height={240} description={'All bars use the same zero baseline and scale 0 to ' + f(bound, 5) + '. White is the original matrix; amber is the effective matrix. The Frobenius norm is an upper bound, not a typical directional stretch.'}>
    {['Largest singular value', 'Second singular value', 'Frobenius norm'].map((name, i) => <g key={name}><text x="10" y={30 + i * 73} fill="#ddd" fontSize="13">{name}</text>{values.map((row, j) => <g key={j}><rect x="185" y={12 + i * 73 + j * 25} width={row[i] / bound * 355} height="16" fill={j ? '#e6b854' : '#ddd'} /><text x="550" y={25 + i * 73 + j * 25} fill="#ddd" fontSize="12">{f(row[i], 5)}</text></g>)}</g>)}
  </SpectralFigure>;
}
export function SpectralSlopeLab() {
  const [knot, setKnot] = useState(1),
    [extra, setExtra] = useState(1),
    [a, setA] = useState(-.5),
    [b, setB] = useState(1.5),
    [limit, setLimit] = useState(1);
  const xs = Array.from({
      length: 85
    }, (_, i) => -3 + i / 12),
    fa = kinkValue(a, knot, extra),
    fb = kinkValue(b, knot, extra),
    actual = Math.abs(fb - fa),
    allowed = limit * Math.abs(b - a),
    bound = Math.max(1, Math.abs(1 + extra));
  const curves = [{
    label: 'Actual f(x)',
    color: color[0],
    values: xs.map(x => [x, kinkValue(x, knot, extra)])
  }, {
    label: 'Upper permitted difference from a',
    color: color[1],
    dashed: true,
    values: xs.map(x => [x, fa + limit * Math.abs(x - a)])
  }, {
    label: 'Lower permitted difference from a',
    color: color[1],
    dashed: true,
    values: xs.map(x => [x, fa - limit * Math.abs(x - a)])
  }];
  const values = curves.flatMap(c => c.values.map(v => v[1]));
  return <NeuralLab id="spectral-slope" title="A bound must contain every allowed difference"><div className="neural-controls"><NeuralNumber label="Knot location" value={knot} min={-2} max={3} onChange={setKnot} /><NeuralNumber label="Extra slope after knot" value={extra} min={-.9} max={5} onChange={setExtra} /><NeuralNumber label="Reference input a" value={a} min={-3} max={4} onChange={setA} /><NeuralNumber label="Comparison input b" value={b} min={-3} max={4} onChange={setB} /><NeuralNumber label="Claimed global bound L" value={limit} min={0} max={6} onChange={setLimit} /></div><SpectralPlot title="Difference envelope anchored at a" xLabel="input x" yLabel="function value" xDomain={[-3, 4]} yDomain={[Math.min(...values) - .5, Math.max(...values) + .5]} series={curves} points={[{
      id: 'a',
      x: a,
      y: fa,
      label: 'Anchor a',
      selected: true
    }, {
      id: 'b',
      x: b,
      y: fb,
      label: 'Comparison b',
      selected: true,
      color: color[2]
    }]} /><p data-result="slope">Output difference {f(actual)} versus allowed {f(allowed)}: {actual > allowed + 1e-10 ? 'this pair disproves the claimed bound' : 'this pair respects the bound'}. The analytic global Lipschitz constant of this entire piecewise-linear function is {f(bound)}. {limit + 1e-10 >= bound ? 'The claimed bound is valid for every pair here.' : 'A passing pair alone cannot rescue a smaller global claim.'}</p><button onClick={() => {
      setKnot(1);
      setExtra(1);
      setA(-.5);
      setB(1.5);
      setLimit(1);
    }}>Reset difference envelope</button></NeuralLab>;
}
export function SpectralCircleFigure() {
  const w = [[3, 0], [0, 1]],
    effective = [[1, 0], [0, 1 / 3]];
  return <><div className="spectral-two"><CurvePlane title="Unit circle → W circle" extent={3.3} curves={[{
        values: circle,
        color: color[1],
        dashed: true
      }, {
        values: circle.map(x => matvec(w, x))
      }]} points={[{
        value: [3, 0],
        from: [0, 0],
        label: 'W e₁'
      }]} /><CurvePlane title="After exact unit normalization" extent={1.2} curves={[{
        values: circle,
        color: color[1],
        dashed: true
      }, {
        values: circle.map(x => matvec(effective, x))
      }]} points={[{
        value: [1, 0],
        from: [0, 0],
        label: 'W̄ e₁'
      }]} /></div><SpectrumBars raw={w} effective={effective} /><NeuralTable caption="Exact spectrum: one common rescaling" headers={['Quantity', 'Original W', 'Normalized W̄']} rows={[['largest singular value', 3, 1], ['second singular value', 1, '1/3'], ['Frobenius norm', '√10', '√10/3']]} /></>;
}
const initialMatrix = () => ({
  w: [[2, 1], [0, 1]],
  u: [1, 1],
  steps: 1,
  target: 1,
  method: 'power',
  bias: [0, 0],
  angle: 30
});
export function SpectralMatrixLab() {
  const [state, setState] = useState(initialMatrix),
    [pinned, setPinned] = useState(initialMatrix),
    [round, setRound] = useState(1);
  const change = (key, value) => setState(s => ({
    ...s,
    [key]: value
  }));
  const current = normalizeMatrix(state.w, state.method, state.target, state.u, state.steps),
    reference = normalizeMatrix(pinned.w, pinned.method, pinned.target, pinned.u, pinned.steps),
    trace = powerTrace(state.w, state.u, state.steps),
    shown = trace.records[Math.min(round, trace.records.length) - 1];
  const direction = [Math.cos(state.angle * Math.PI / 180), Math.sin(state.angle * Math.PI / 180)];
  const translated = (w, x, b) => matvec(w, x).map((v, i) => v + b[i]);
  const effective = current.effective;
  const extent = Math.max(1.2, singular2(state.w).values[0] + norm(state.bias), effective ? current.trueNorm + norm(state.bias) : 0, reference.effective ? reference.trueNorm + norm(pinned.bias) : 0) * 1.1;
  return <NeuralLab id="spectral-matrix" title="Find—or miss—the strongest matrix direction">
 <MatrixEditor label="W" values={state.w} onChange={w => change('w', w)} /><div className="neural-controls"><NeuralSelect label="Matrix operation" value={state.method} onChange={v => change('method', v)} options={methods} /><NeuralNumber label="Target norm or clipping threshold" value={state.target} min={.05} max={3} onChange={v => change('target', v)} /><NeuralNumber label="Power-iteration rounds" value={state.steps} min={1} max={16} integer onChange={v => change('steps', v)} /><NeuralNumber label="Marked input angle degrees" value={state.angle} min={0} max={360} onChange={v => change('angle', v)} /></div><SpectralVector label="Initial left direction u" values={state.u} onChange={v => change('u', v)} min={-2} max={2} /><SpectralVector label="Output bias b" values={state.bias} onChange={v => change('bias', v)} />
 {current.error ? <p role="status">{current.error} Edit the direction or matrix to resume this calculation.</p> : <p data-result="matrix">Exact original norm {f(current.sigma, 8)}; actual effective norm <strong>{f(current.trueNorm, 8)}</strong>. {current.zero ? 'Zero matrix: output is defined as zero before the bias; singular direction undefined.' : state.method === 'power' && current.trueNorm > state.target + 1e-10 ? 'The finite power estimate leaves a norm above the requested target.' : 'The selected operation is evaluated on the actual effective matrix.'}</p>}
 <div className="spectral-two"><CurvePlane title="Original map and marked unit input" extent={extent} curves={[{
        values: circle,
        color: color[1],
        dashed: true
      }, {
        values: circle.map(x => translated(state.w, x, state.bias))
      }]} points={[{
        value: direction,
        from: [0, 0],
        label: 'v',
        color: color[1]
      }, {
        value: translated(state.w, direction, state.bias),
        from: state.bias,
        label: 'Wv+b'
      }]} />{effective && <CurvePlane title="Current amber map; pinned dashed map" extent={extent} curves={[{
        values: circle.map(x => translated(effective, x, state.bias))
      }, ...(reference.effective ? [{
        values: circle.map(x => translated(reference.effective, x, pinned.bias)),
        dashed: true,
        color: color[1]
      }] : [])]} points={[{
        value: translated(effective, direction, state.bias),
        from: state.bias,
        label: 'W̄v+b'
      }]} />}</div>
 {effective && <><SpectrumBars raw={state.w} effective={effective} /><SpectralMatrix title="Current effective output-by-input matrix" values={effective} /><NeuralTable caption="Spectrum is not an average stretch" headers={['Quantity', 'Raw matrix', 'Current effective']} rows={[['σ₁', f(current.values[0], 8), f(singular2(effective).values[0], 8)], ['σ₂', f(current.values[1], 8), f(singular2(effective).values[1], 8)], ['Frobenius', f(singular2(state.w).frobenius), f(singular2(effective).frobenius)]]} /></>}
 <p>Both geometric panels use the same axis scale. Bias translates the output center without changing any pairwise stretch. The pinned state retains its own matrix, operation, target, direction and bias; it does not silently follow the controls.</p>
 <div className="neural-buttons"><button onClick={() => setPinned(structuredClone(state))}>Pin current map</button><button disabled={state.w.flat().some(x => Math.abs(x) > 2.5)} onClick={() => change('w', scale(state.w, 2))}>Double every matrix coefficient</button><button onClick={() => {
        setState({
          ...initialMatrix(),
          w: [[3, 0], [0, 1]],
          u: [0, 1]
        });
        setRound(1);
      }}>Orthogonal starting direction</button><button onClick={() => {
        setState({
          ...initialMatrix(),
          w: [[1.01, 0], [0, 1]],
          steps: 8
        });
        setRound(8);
      }}>Slow spectral gap</button><button onClick={() => setState({
        ...initialMatrix(),
        w: [[.2, 0], [0, .2]],
        method: 'cap'
      })}>Small matrix: cap only</button><button onClick={() => {
        setState(initialMatrix());
        setPinned(initialMatrix());
        setRound(1);
      }}>Reset matrix investigation</button></div>
 <p>With exact unit normalization, positive uniform rescaling leaves the effective matrix unchanged. Cap-only, entrywise clipping and a one-coefficient change need not share that null. The near-zero cutoff is 10⁻¹².</p>
 <section open={state.method === 'power'} data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Follow the actual power-iteration vectors</h4>{trace.error && <p>{trace.error}</p>}{shown && <><NeuralNumber label="Inspect computed power round" value={Math.min(round, trace.records.length)} min={1} max={trace.records.length} integer onChange={setRound} /><SpectralFigure title="Each multiplication exposes a different coordinate space" width={760} height={155} description="The first u is the preceding normalized left vector. Multiply by Wᵀ, normalize to v, multiply by W, and normalize the next u. The estimate uses the final u and Wv.">{[['u', shown.before], ['Wᵀu', shown.wtU], ['v', shown.v], ['Wv', shown.wV], ['next u', shown.u]].map(([name, values], i) => <g key={name}><rect x={10 + 150 * i} y="35" width="130" height="72" fill="#222" stroke={i % 2 ? '#ba9fd1' : '#e6b854'} /><text x={75 + 150 * i} y="57" textAnchor="middle" fill="#ddd" fontSize="14">{name}</text><text x={75 + 150 * i} y="82" textAnchor="middle" fill="#ddd" fontSize="12">{vector(values)}</text>{i < 4 && <path d={'M' + (140 + 150 * i) + ' 70 h20'} stroke="#ddd" />}</g>)}<text x="380" y="140" textAnchor="middle" fill="#eee" fontSize="13">{'uᵀWv = ' + f(shown.estimate, 8)}</text></SpectralFigure></>}<NeuralTable caption="Every computed power round" headers={['Round', 'Estimated σ', 'True norm after unit division']} rows={trace.records.map(r => [r.step, f(r.estimate, 8), f(r.trueNorm, 8)])} /><p>This trace shows unit division before any target scaling. It starts from the declared left vector. The library example later has two stored buffers and follows its documented update order.</p></section>
 </NeuralLab>;
}
export function SpectralDerivativeLab() {
  const [w, setW] = useState([[2, 1], [0, 1]]),
    [h, setH] = useState([[1, -.3], [.2, .7]]);
  const result = normalizationGradient(w, h);
  return <NeuralLab id="spectral-derivative" title="Changing a weight also changes its normalization scale"><MatrixEditor label="Derivative example W" values={w} onChange={setW} /><MatrixEditor label="Upstream derivative H" values={h} onChange={setH} min={-2} max={2} />{result ? <><SpectralFigure title="The quotient derivative subtracts the scale response" height={125} description="These are matrices in the same output-by-input coordinates. The subtraction removes sensitivity to a positive radial rescaling of W."><text x="20" y="39" fill="#eee" fontSize="14">H / σ</text><path d="M85 34 H208" stroke="#ddd" /><text x="231" y="39" fill="#eee" fontSize="22">−</text><text x="283" y="39" fill="#eee" fontSize="14">〈H,W̄〉 uvᵀ / σ</text><path d="M452 34 H548" stroke="#ddd" /><text x="557" y="40" fill="#e6b854" fontSize="15">∂L/∂W</text><text x="320" y="85" textAnchor="middle" fill="#ddd" fontSize="13">{'σ=' + f(result.values[0], 6) + '; 〈H,W̄〉=' + f(result.inner, 6)}</text></SpectralFigure><div className="spectral-two"><SpectralMatrix title="Detached-denominator derivative" values={result.detached} /><SpectralMatrix title="Derivative through the scale" values={result.gradient} /></div><p data-result="normalization-gradient">Radial directional derivative 〈∂L/∂W,W〉 = {f(dot(result.gradient.flat(), w.flat()), 9)}. The detached route gives {f(dot(result.detached.flat(), w.flat()), 9)}. Exact positive rescaling is a null direction; this explains why the omitted term matters.</p></> : <p role="status">This zero matrix or repeated largest singular value has no unique positive leading-vector derivative of the displayed form.</p>}<button onClick={() => {
      setW([[2, 1], [0, 1]]);
      setH([[1, -.3], [.2, .7]]);
    }}>Reset derivative example</button></NeuralLab>;
}
export function SpectralConvolutionLab() {
  const [kernel, setKernel] = useState([1, 2]),
    [input, setInput] = useState([1, -1, 2]),
    [mode, setMode] = useState('valid'),
    [stride, setStride] = useState(1),
    [selected, setSelected] = useState(1);
  const result = convolution(kernel, mode, input, stride),
    baseline = Array.from({
      length: input.length
    }, (_, i) => [1, -1, 2][i] || 0),
    before = matvec(result.matrix, baseline),
    delta = input.map((v, i) => v - baseline[i]);
  const rows = result.matrix.length;
  return <NeuralLab id="spectral-convolution" title="One stored kernel, several spatial operators"><SpectralVector label="Cross-correlation kernel" values={kernel} onChange={setKernel} /><div className="neural-controls"><NeuralNumber label="Input length" value={input.length} min={3} max={6} integer onChange={n => {
        setInput(x => Array.from({
          length: n
        }, (_, i) => x[i] ?? 0));
        setSelected(Math.min(selected, n - 1));
      }} /><NeuralSelect label="Boundary rule" value={mode} onChange={setMode} options={[['valid', 'Valid windows only'], ['circular', 'Circular wrap']]} /><NeuralSelect label="Stride" value={stride} onChange={v => setStride(Number(v))} options={[[1, '1: neighboring starts'], [2, '2: skip one start']]} /><NeuralNumber label="Highlight input column" value={selected} min={0} max={input.length - 1} integer onChange={setSelected} /></div><SpectralVector label="Spatial input x" values={input} onChange={setInput} />
 <SpectralFigure title="A selected input reaches every output whose window contains it" width={680} height={235} description="A line exists exactly when the corresponding matrix coefficient is nonzero. Labels use zero-based input/output indices. Amber lines highlight the selected input; the full matrix below retains zero coefficients as well.">{input.map((v, j) => <g key={j}><rect x={20 + j * 110} y="20" width="90" height="48" fill="#222" stroke={j === selected ? '#e6b854' : '#777'} /><text x={65 + j * 110} y="41" textAnchor="middle" fill="#eee" fontSize="13">{'x' + j}</text><text x={65 + j * 110} y="60" textAnchor="middle" fill="#eee" fontSize="13">{f(v)}</text></g>)}{result.matrix.map((row, i) => <g key={i}>{row.map((coefficient, j) => coefficient !== 0 && <line key={j} x1={65 + j * 110} y1="68" x2={65 + i * 110} y2="163" stroke={j === selected ? '#e6b854' : '#555'} strokeWidth={j === selected ? 2.5 : 1} />)}<rect x={20 + i * 110} y="164" width="90" height="48" fill="#222" stroke="#777" /><text x={65 + i * 110} y="184" textAnchor="middle" fill="#eee" fontSize="13">{'y' + i}</text><text x={65 + i * 110} y="204" textAnchor="middle" fill="#eee" fontSize="13">{f(result.output[i])}</text></g>)}</SpectralFigure>
 <SpectralMatrix title="Full operator: y = A x" values={result.matrix} selected={selected} /><p data-result="convolution">Stored kernel norm {f(result.kernelNorm, 8)}; full operator norm {f(result.operatorNorm, 8)}; after dividing the kernel by its norm, full norm <strong>{f(result.normalizedNorm, 8)}</strong>. {result.zero ? 'Zero kernel is explicitly left zero; no division is performed.' : result.normalizedNorm > 1 + 1e-10 ? 'Kernel normalization has not capped this whole operator at one.' : 'This particular full operator respects the unit cap.'}</p><NeuralTable caption="Actual input change and operator bound answer different questions" headers={['Quantity', 'Value']} rows={[['Baseline input, same current kernel', vector(baseline)], ['Baseline output', vector(before)], ['Current output', vector(result.output)], ['‖Δx‖', f(norm(delta))], ['‖A Δx‖', f(norm(result.output.map((v, i) => v - before[i])))], ['Valid upper bound ‖A‖ ‖Δx‖', f(result.operatorNorm * norm(delta))]]} /><p>The kernel is used as cross-correlation in the shown order. Circular wrapping applies only when selected. All {rows} output directions enter a small Gram-matrix eigensolve; this is not a finite power estimate.</p><div className="neural-buttons"><button onClick={() => {
        setKernel([1, 1]);
        setInput([1, -1, 2, 0]);
        setStride(2);
        setMode('valid');
      }}>Disjoint [1,1] windows</button><button onClick={() => {
        setKernel([1, 1]);
        setInput([1, -1, 2, 0]);
        setStride(1);
        setMode('circular');
      }}>Circular [1,1] windows</button><button onClick={() => {
        setKernel([1, 2]);
        setInput([1, -1, 2]);
        setStride(1);
        setMode('valid');
        setSelected(1);
      }}>Reset spatial operator</button></div></NeuralLab>;
}
export function SpectralPenaltyLab() {
  const [knot, setKnot] = useState(2),
    [extra, setExtra] = useState(4),
    [probes, setProbes] = useState([-.5, .5, 1.5]),
    [strength, setStrength] = useState(2),
    [kind, setKind] = useState('target-one');
  const [pinned, setPinned] = useState({
    knot: 2,
    extra: 4,
    probes: [-.5, .5, 1.5],
    kind: 'target-one',
    strength: 2
  });
  const result = probePenalty(knot, extra, probes, kind, strength),
    reference = probePenalty(pinned.knot, pinned.extra, pinned.probes, pinned.kind, pinned.strength);
  const xs = Array.from({
      length: 85
    }, (_, i) => -3 + i / 12),
    ys = xs.map(x => kinkValue(x, knot, extra));
  return <NeuralLab id="spectral-penalty" title="Move a probe into a region the penalty missed"><div className="neural-controls"><NeuralNumber label="Kink location" value={knot} min={-2} max={3} onChange={setKnot} /><NeuralNumber label="Extra slope" value={extra} min={-.9} max={5} onChange={setExtra} /><NeuralNumber label="Penalty strength λ" value={strength} min={0} max={10} onChange={setStrength} /><NeuralSelect label="Penalty objective" value={kind} onChange={setKind} options={kinds} /></div><SpectralVector label="Actual probe coordinates" values={probes} onChange={setProbes} min={-3} max={4} /><SpectralPlot title="Only the marked points were probed" xLabel="input x" yLabel="f(x)" xDomain={[-3, 4]} yDomain={[Math.min(...ys) - 1, Math.max(...ys) + 1]} series={[{
      label: 'Current f=x+extra ReLU(x−knot)',
      color: color[0],
      values: xs.map((x, i) => [x, ys[i]])
    }, {
      label: 'Pinned function',
      color: color[1],
      dashed: true,
      values: xs.map(x => [x, kinkValue(x, pinned.knot, pinned.extra)])
    }]} points={result.rows.map((row, i) => ({
      id: i,
      x: row.x,
      y: row.value,
      label: 'Probe ' + i + ', derivative ' + (row.slope ?? 'undefined'),
      selected: true
    }))} /><p data-result="penalty">Current mean penalty <strong>{result.penalty === null ? 'undefined at a probed corner' : f(result.penalty, 8)}</strong>; analytic global Lipschitz constant {f(result.globalBound)}. Pinned mean {reference.penalty === null ? 'undefined' : f(reference.penalty, 8)} under {pinned.kind}, strength {pinned.strength}. A zero sample penalty can coexist with a steep unobserved region.</p><NeuralTable caption="Differentiate first; square each contribution; then average" headers={['Probe', 'Input', 'f(x)', 'Derivative', 'Weighted penalty contribution']} rows={result.rows.map((r, i) => [i, f(r.x), f(r.value), r.slope === null ? 'ordinary derivative undefined' : f(r.slope), r.contribution === null ? 'undefined' : f(r.contribution)])} /><SpectralPlot title="The objective changes which slopes are preferred" xLabel="gradient norm r" yLabel="weighted penalty" xDomain={[0, 6]} yDomain={[0, Math.max(1, 36 * strength)]} series={kinds.map(([name, label], i) => ({
      label,
      color: color[i],
      values: Array.from({
        length: 73
      }, (_, j) => [j / 12, penaltyValue(j / 12, name, strength)])
    }))} /><div className="neural-buttons"><button disabled={probes.length >= 6} onClick={() => setProbes(p => [...p, 2.5])}>Add probe</button><button disabled={probes.length <= 2} onClick={() => setProbes(p => p.slice(0, -1))}>Remove last probe</button><button onClick={() => setPinned({
        knot,
        extra,
        probes: [...probes],
        kind,
        strength
      })}>Pin current probes and function</button><button onClick={() => {
        setKnot(2);
        setExtra(4);
        setProbes([-.5, .5, 1.5]);
        setStrength(2);
        setKind('target-one');
        setPinned({
          knot: 2,
          extra: 4,
          probes: [-.5, .5, 1.5],
          kind: 'target-one',
          strength: 2
        });
      }}>Reset probe coverage</button></div><p>A probe exactly at the nonzero kink has no ordinary derivative. This view names that mathematical ambiguity instead of treating an automatic-differentiation convention as a unique derivative.</p></NeuralLab>;
}
export function SpectralLinearPenaltyLab() {
  const [w, setW] = useState([3, 4]),
    [strength, setStrength] = useState(2),
    [rate, setRate] = useState(.1),
    [kind, setKind] = useState('target-one');
  const result = linearPenalty(w, strength, rate, kind);
  return <NeuralLab id="spectral-linear-penalty" title="Follow the parameter derivative of an input-gradient penalty"><SpectralVector label="Linear critic weight w" values={w} onChange={setW} min={-5} max={5} /><div className="neural-controls"><NeuralNumber label="Penalty coefficient" value={strength} min={0} max={5} onChange={setStrength} /><NeuralNumber label="Penalty-only step size" value={rate} min={0} max={.25} onChange={setRate} /><NeuralSelect label="Linear penalty objective" value={kind} onChange={setKind} options={kinds} /></div><CurvePlane title="Weight before and after one penalty-only step" extent={Math.max(1.2, norm(w), result.updated ? norm(result.updated) : 0) * 1.1} curves={[{
      values: circle,
      color: color[1],
      dashed: true
    }]} points={[{
      value: w,
      from: [0, 0],
      label: 'w'
    }, ...(result.updated ? [{
      value: result.updated,
      from: w,
      label: 'w′',
      color: color[2]
    }] : [])]} /><NeuralTable caption="One evaluated update; no GAN fit is run" headers={['Stage', 'Value']} rows={[['Input gradient ∇x f', vector(w)], ['Its norm', f(result.norm)], ['Current penalty', f(result.penalty)], ['Parameter gradient ∇w R', result.gradient ? vector(result.gradient) : 'undefined at zero for target-one'], ['Updated weight', result.updated ? vector(result.updated) : 'no unique derivative step'], ['Updated norm', result.updated ? f(result.updatedNorm) : 'undefined'], ['Updated penalty', result.updated ? f(result.updatedPenalty) : 'undefined']]} /><button disabled={!result.updated || result.updated.some(x => Math.abs(x) > 5)} onClick={() => setW(result.updated)}>Use this updated weight</button><button onClick={() => {
      setW([3, 4]);
      setStrength(2);
      setRate(.1);
      setKind('target-one');
    }}>Reset penalty step</button><p>A large step can cross the preferred norm or increase the penalty. This isolates the regularizer’s gradient; adversarial training adds another parameter gradient.</p></NeuralLab>;
}
export function SpectralLibraryLab() {
  const [w, setW] = useState([[2, 1], [0, 1]]),
    [training, setTraining] = useState(true),
    [accesses, setAccesses] = useState(1);
  const initial = {
    u: [1 / Math.SQRT2, 1 / Math.SQRT2],
    v: [1 / Math.SQRT2, 1 / Math.SQRT2]
  };
  let buffers = initial;
  const history = [];
  for (let i = 0; i < accesses; i++) {
    const next = libraryAccess(w, buffers, training);
    history.push(next);
    if (next.error) break;
    buffers = next;
  }
  const result = history.at(-1);
  return <NeuralLab id="spectral-library" title="A weight has an original value, an effective value and cached directions"><MatrixEditor label="Original stored weight" values={w} onChange={setW} /><div className="neural-controls"><NeuralSelect label="Mode for the displayed accesses" value={training ? 'training' : 'evaluation'} onChange={v => setTraining(v === 'training')} options={[['training', 'Training: update cached directions'], ['evaluation', 'Evaluation: freeze cached directions']]} /><NeuralNumber label="Number of weight accesses" value={accesses} min={1} max={12} integer onChange={setAccesses} /></div><p>The teaching state starts with explicitly assigned u=v=[1,1]/√2 after registration. Changing a control replays from that declared state. The real API may initialize its buffers through preliminary iterations; this is the controlled buffer experiment checked against PyTorch 2.14.</p>{result.error ? <p role="status">{result.error}</p> : <><SpectralFigure title="The forward weight is derived from two separately stored quantities" height={195} description="The optimizer owns W. Training access updates power buffers, which help estimate a scalar denominator. Both W and that denominator enter the effective weight; evaluation only freezes the buffer update."><rect x="15" y="20" width="175" height="45" fill="#222" stroke="#e6b854" /><text x="102" y="47" textAnchor="middle" fill="#ddd" fontSize="13">original W</text><rect x="15" y="128" width="175" height="45" fill="#222" stroke="#ba9fd1" /><text x="102" y="154" textAnchor="middle" fill="#ddd" fontSize="13">cached u and v</text><rect x="290" y="120" width="150" height="60" fill="#222" stroke="#777" /><text x="365" y="145" textAnchor="middle" fill="#ddd" fontSize="13">σ = uᵀWv</text><text x="365" y="166" textAnchor="middle" fill="#ddd" fontSize="13">{f(result.sigma, 6)}</text><rect x="490" y="20" width="135" height="60" fill="#222" stroke="#e6b854" /><text x="557" y="46" textAnchor="middle" fill="#ddd" fontSize="13">forward W / σ</text><text x="557" y="67" textAnchor="middle" fill="#ddd" fontSize="13">effective weight</text><path d="M190 42 H490 M220 42 V137 H290 M190 150 H290 M440 150 H557 V80" fill="none" stroke="#ddd" /></SpectralFigure><div className="spectral-stores"><section className="spectral-card"><h4>Original weight</h4><SpectralMatrix title="Optimizer-owned W" values={w} /></section><section className="spectral-card"><h4>Power-vector buffers</h4><p>u={vector(result.u)}</p><p>v={vector(result.v)}</p><p>σ estimate={f(result.sigma, 8)}</p></section><section className="spectral-card"><h4>Effective weight</h4><SpectralMatrix title="Forward uses W / (uᵀWv)" values={result.effective} /></section></div><p data-result="library">Actual effective norm {f(singular2(result.effective).values[0], 8)}. Evaluation freezes this estimation procedure; it does not turn an approximation into an exact norm.</p><NeuralTable caption="The library updates u from v, then v from u" headers={['Access', 'u', 'v', 'Estimated σ']} rows={history.map((row, i) => [i + 1, vector(row.u), vector(row.v), f(row.sigma, 8)])} /></>}<button onClick={() => {
      setW([[2, 1], [0, 1]]);
      setTraining(true);
      setAccesses(1);
    }}>Reset stored-weight trace</button></NeuralLab>;
}
export function SpectralMarginLab() {
  const [w, setW] = useState([[1, 0], [0, 1]]),
    [bias, setBias] = useState([0, 0]),
    [point, setPoint] = useState([1.5, .25]);
  const r = linearMargin(w, bias, point);
  const direction = r.pair > 1e-12 ? [-r.normal[1] / r.pair, r.normal[0] / r.pair] : null;
  const extent = Math.max(2, norm(point), Number.isFinite(r.radius) ? r.radius + norm(point) : 0) * 1.1;
  const boundary = r.boundary && direction ? [r.boundary.map((v, i) => v - extent * 3 * direction[i]), r.boundary.map((v, i) => v + extent * 3 * direction[i])] : null;
  const ring = Number.isFinite(r.radius) && r.radius > 0 ? circle.map(x => x.map((v, i) => point[i] + v * r.radius)) : null;
  return <NeuralLab id="spectral-margin" title="A logit gap needs the sensitivity of that same difference"><MatrixEditor label="Two logit rows W" values={w} onChange={setW} min={-2} max={2} /><SpectralVector label="Logit bias" values={bias} onChange={setBias} min={-1} max={1} /><SpectralVector label="Input point" values={point} onChange={setPoint} min={-2} max={2} /><CurvePlane title="Winning input, tie boundary and nearest perturbation" extent={extent} curves={[...(boundary ? [{
      values: boundary,
      color: color[1]
    }] : []), ...(ring ? [{
      values: ring,
      color: color[2],
      dashed: true
    }] : [])]} points={[{
      value: point,
      label: 'x'
    }, ...(r.boundary ? [{
      value: r.boundary,
      from: point,
      label: 'tie',
      color: color[1]
    }] : [])]} /><NeuralTable caption="Exact linear distance versus a sufficient joint-norm certificate" headers={['Quantity', 'Value']} rows={[['Logits', vector(r.logits)], ['Class-0 gap', f(r.gap)], ['Row-difference slope', f(r.pair)], ['Joint spectral norm', f(r.joint)], ['Exact class-0 winning radius', Number.isFinite(r.radius) ? f(r.radius) : 'unbounded: constant positive gap'], ['Joint-norm sufficient radius', Number.isFinite(r.jointRadius) ? f(r.jointRadius) : 'unbounded'], ['Perpendicular perturbation', r.delta ? vector(r.delta) : 'no finite boundary']]} /><p data-result="margin">{r.gap <= 0 ? 'Class 0 has no positive winning certificate at this point.' : r.pair <= 1e-12 ? 'Identical weight rows and a positive bias gap make this class comparison independent of input.' : 'The open ball below the exact radius preserves a positive class-0 gap; the nearest boundary point is a tie.'} This is a two-class linear model in its stated raw coordinates.</p><button onClick={() => {
      setW([[1, 0], [0, 1]]);
      setBias([0, 0]);
      setPoint([1.5, .25]);
    }}>Reset margin geometry</button></NeuralLab>;
}
