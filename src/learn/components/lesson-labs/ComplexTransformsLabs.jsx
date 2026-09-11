import { useMemo, useState } from 'react';
import { complexOperation, complexPolar, rootsOfComplex, phasorSynthesis, harmonicProjection, squareConvergence, finiteFourier, parseRealSamples, aliasState, windowSpectrum, convolutionState, filterResponse, laplaceRegion, sinc } from '../../data/complex-transforms-models.js';
import './complex-transforms-labs.css';
const TAU = 2 * Math.PI;
const colors = ['#e8b34b', '#73caa0', '#bb9de4', '#e88e7a'];
const format = (value, digits = 3) => Math.abs(value) < 1e-11 ? '0' : Math.abs(value) >= 1e6 ? value.toExponential(3) : Number(value.toFixed(digits)).toString();
const complexText = value => `${format(value[0])} ${value[1] < -1e-11 ? '−' : '+'} ${format(Math.abs(value[1]))}i`;
const phaseText = value => value === null ? 'undefined at zero' : `${format(value / Math.PI)}π rad`;
function Range({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  display
}) {
  return <label className="transform-range"><span>{label} <output>{display ?? format(value)}</output></span><input type="range" aria-label={label} value={value} min={min} max={max} step={step} onChange={event => onChange(Number(event.target.value))} /></label>;
}
function Choice({
  label,
  value,
  onChange,
  options
}) {
  return <label className="transform-choice"><span>{label}</span><select aria-label={label} value={value} onChange={event => onChange(event.target.value)}>{options.map(([key, text]) => <option key={key} value={key}>{text}</option>)}</select></label>;
}
function Legend({
  labels
}) {
  return <ul className="transform-legend">{labels.map((label, index) => <li key={label}><span style={{
        background: colors[index]
      }} />{label}</li>)}</ul>;
}
function Values({
  items
}) {
  return <dl className="transform-values">{items.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>;
}
function Table({
  headers,
  rows,
  summary = 'Read the numerical values'
}) {
  return <details className="transform-data"><summary>{summary}</summary><div className="transform-table-scroll"><table><thead><tr>{headers.map(text => <th key={text}>{text}</th>)}</tr></thead><tbody>{rows.map((row, index) => <tr key={index}>{row.map((text, column) => <td key={column}>{text}</td>)}</tr>)}</tbody></table></div></details>;
}
function Investigation({
  title,
  children,
  id
}) {
  return <section className="transform-lab" aria-label={title} data-transform-lab={id}><h3>{title}</h3>{children}</section>;
}
function Chart({
  title,
  lines = [],
  dots = [],
  xDomain,
  yDomain,
  xLabel,
  yLabel,
  cursor,
  children
}) {
  const [xmin, xmax] = xDomain,
    [ymin, ymax] = yDomain;
  const largestY = Math.max(Math.abs(ymin), Math.abs(ymax));
  const yScale = largestY >= 1e4 ? 10 ** Math.floor(Math.log10(largestY)) : 1;
  const x = value => 42 + (value - xmin) * 242 / (xmax - xmin);
  const y = value => 164 - (value - ymin) * 140 / (ymax - ymin);
  const path = points => points.map((point, index) => `${index ? 'L' : 'M'}${x(point[0]).toFixed(3)},${y(point[1]).toFixed(3)}`).join(' ');
  return <figure className="transform-chart"><figcaption>{title}{yScale !== 1 && ` · vertical ticks × ${yScale.toExponential(0)}`}</figcaption><svg viewBox="0 0 300 218" role="img" aria-label={`${title}. Horizontal: ${xLabel}. Vertical: ${yLabel}. Vertical tick multiplier ${yScale}.`}>
    {[ymin, (ymin + ymax) / 2, ymax].map(value => <g key={value}><line className="transform-grid" x1="42" x2="284" y1={y(value)} y2={y(value)} /><text x="35" y={y(value) + 5} textAnchor="end">{format(value / yScale, 2)}</text></g>)}
    {[xmin, (xmin + xmax) / 2, xmax].map(value => <g key={value}><line className="transform-grid" x1={x(value)} x2={x(value)} y1="24" y2="164" /><text x={x(value)} y="185" textAnchor={value === xmax ? 'end' : value === xmin ? 'start' : 'middle'}>{format(value, 2)}</text></g>)}
    {ymin < 0 && ymax > 0 && <line className="transform-axis" x1="42" x2="284" y1={y(0)} y2={y(0)} />}
    {lines.map((points, index) => <path key={index} d={path(points)} fill="none" stroke={colors[index]} strokeWidth="2" />)}
    {dots.map((points, series) => <g key={series}>{points.map((point, index) => <g key={index}><line x1={x(point[0])} x2={x(point[0])} y1={y(Math.max(ymin, Math.min(ymax, 0)))} y2={y(point[1])} stroke={colors[series]} opacity=".45" /><circle cx={x(point[0])} cy={y(point[1])} r="3.2" fill={colors[series]} /></g>)}</g>)}
    {cursor !== undefined && <line x1={x(cursor)} x2={x(cursor)} y1="24" y2="164" className="transform-cursor" />}
    <text x="42" y="14">{yLabel}</text><text x="163" y="209" textAnchor="middle">{xLabel}</text>{children?.({
        x,
        y
      })}
  </svg></figure>;
}
function Plane({
  title,
  vectors,
  chain = false,
  curve,
  selected
}) {
  const all = [...vectors, ...(curve ?? []), [0, 0]];
  let vertices = [[0, 0]];
  if (chain) vectors.forEach(value => vertices.push([vertices.at(-1)[0] + value[0], vertices.at(-1)[1] + value[1]]));else vertices = [[0, 0], ...vectors];
  const extent = Math.max(1, ...[...all, ...vertices].flatMap(value => value.map(Math.abs))) * 1.2;
  const x = value => 150 + value * 116 / extent;
  const y = value => 144 - value * 116 / extent;
  return <figure className="transform-plane"><figcaption>{title}</figcaption><svg viewBox="0 0 300 284" role="img" aria-label={`${title}. Equal scales on real and imaginary axes; extent ${format(extent)}.`}>
    <line x1="22" x2="278" y1="144" y2="144" className="transform-axis" /><line x1="150" x2="150" y1="16" y2="272" className="transform-axis" />
    <text x="276" y="163" textAnchor="end">Re</text><text x="157" y="27">Im</text><text x="155" y="162">0</text>
    <text x="270" y="137" textAnchor="end">{format(extent, 1)}</text><text x="28" y="137">−{format(extent, 1)}</text>
    {curve && <path d={curve.map((point, index) => `${index ? 'L' : 'M'}${x(point[0])},${y(point[1])}`).join(' ')} stroke={colors[2]} fill="none" strokeWidth="1.8" />}
    {vectors.map((value, index) => {
        const start = chain ? vertices[index] : [0, 0],
          end = chain ? vertices[index + 1] : value;
        const angle = Math.atan2(y(end[1]) - y(start[1]), x(end[0]) - x(start[0]));
        const endX = x(end[0]),
          endY = y(end[1]);
        return <g key={index} stroke={colors[index % colors.length]}><line x1={x(start[0])} y1={y(start[1])} x2={endX} y2={endY} strokeWidth="2.5" />{(value[0] !== 0 || value[1] !== 0) && <path d={`M${endX - 7 * Math.cos(angle - .5)},${endY - 7 * Math.sin(angle - .5)} L${endX},${endY} L${endX - 7 * Math.cos(angle + .5)},${endY - 7 * Math.sin(angle + .5)}`} fill="none" strokeWidth="2" />}<circle cx={endX} cy={endY} r="3" fill={colors[index % colors.length]} /></g>;
      })}
    {chain && <line x1={x(0)} y1={y(0)} x2={x(vertices.at(-1)[0])} y2={y(vertices.at(-1)[1])} stroke={colors[2]} strokeWidth="2" strokeDasharray="5 4" />}
    {selected && <circle cx={x(selected[0])} cy={y(selected[1])} r="5" fill="none" stroke="#eee3cc" strokeWidth="2" />}
  </svg></figure>;
}
export function ComplexArithmeticLab() {
  const [real, setReal] = useState(2),
    [imaginary, setImaginary] = useState(-1),
    [operation, setOperation] = useState('multiply');
  const state = useMemo(() => {
    try {
      return {
        result: complexOperation([1, 2], [real, imaginary], operation)
      };
    } catch (error) {
      return {
        error: error.message
      };
    }
  }, [real, imaginary, operation]);
  return <Investigation id="arithmetic" title="Move, rotate or divide the same complex number">
    <p>Keep z=1+2i. Predict what multiplication by i will do, then set w=0+1i. The arrow lengths and angles use the actual result.</p>
    <div className="transform-controls"><Choice label="Complex operation" value={operation} onChange={setOperation} options={[['multiply', 'z × w'], ['add', 'z + w'], ['divide', 'z ÷ w']]} /><Range label="Real part of w" value={real} onChange={setReal} min={-3} max={3} step={.25} /><Range label="Imaginary part of w" value={imaginary} onChange={setImaginary} min={-3} max={3} step={.25} /></div>
    {state.error ? <p role="status">{state.error}</p> : <div className="transform-two"><Plane title={operation === 'add' ? 'Translate w to the tip of z' : 'One plane, equal axis scales'} vectors={operation === 'add' ? [state.result.z, state.result.w] : [state.result.z, state.result.w, state.result.result]} chain={operation === 'add'} /><div><Legend labels={['z', 'w', 'Result']} /><Values items={[['Result', complexText(state.result.result)], ['Result magnitude', format(state.result.resultPolar.magnitude)], ['Result phase', phaseText(state.result.resultPolar.phase)], ['Phase of w', phaseText(state.result.wPolar.phase)]]} /><p>Zero has no direction. A negative real multiplier turns through π; adding it shifts the real coordinate instead. In addition, the dashed origin-to-endpoint arrow is the sum.</p></div></div>}
    <button onClick={() => {
      setReal(2);
      setImaginary(-1);
      setOperation('multiply');
    }}>Reset complex numbers</button>
  </Investigation>;
}
export function RootsUnityFigure() {
  const roots = rootsOfComplex([1, 0], 4);
  return <section className="transform-inline"><div className="transform-two"><figure className="transform-plane"><figcaption>Four different fourth roots of 1</figcaption><svg viewBox="0 0 300 284" role="img" aria-label="The fourth roots 1, i, minus 1 and minus i sit at quarter turns on the unit circle."><circle cx="150" cy="144" r="90" fill="none" className="transform-grid" /><line x1="28" x2="272" y1="144" y2="144" className="transform-axis" /><line x1="150" x2="150" y1="20" y2="267" className="transform-axis" />{roots.map((root, index) => <g key={index}><line x1="150" y1="144" x2={150 + 90 * root[0]} y2={144 - 90 * root[1]} stroke={colors[index]} strokeWidth="2" /><circle cx={150 + 90 * root[0]} cy={144 - 90 * root[1]} r="4" fill={colors[index]} /></g>)}<text x="251" y="137">1</text><text x="147" y="37">i</text><text x="28" y="137">−1</text><text x="144" y="261">−i</text><text x="157" y="162">0</text></svg></figure><div><p>Each arrow turns by a multiple of π/2. Raising it to the fourth power multiplies its angle by four, landing on 1.</p><ol>{['1 → 1', 'i → 1', '−1 → 1', '−i → 1'].map(text => <li key={text}>{text}</li>)}</ol><p>The first value is one selected root. It is not the entire solution set.</p></div></div></section>;
}
export function PhasorSynthesisLab() {
  const [timeIndex, setTime] = useState(0),
    [phase, setPhase] = useState(Math.PI / 2);
  const state = phasorSynthesis(timeIndex / 256, phase);
  const curve = useMemo(() => Array.from({
    length: 257
  }, (_, index) => phasorSynthesis(index / 256, phase)), [phase]);
  return <Investigation id="synthesis" title="Read each rotating arrow's horizontal shadow">
    <p>The first arrow is twice as long; the third-frequency arrow turns three times as fast. The measured real signal is the sum of their horizontal coordinates, not the sum of their lengths.</p>
    <div className="transform-controls"><Range label="Synthesis time" value={timeIndex} onChange={setTime} min={0} max={256} display={`${format(state.time)} s`} /><Range label="Synthesis third-tone phase" value={phase} onChange={setPhase} min={-Math.PI} max={Math.PI} step={Math.PI / 12} display={phaseText(phase)} /></div>
    <div className="transform-two"><figure className="transform-plane"><figcaption>Two frequencies at the selected instant</figcaption><svg viewBox="0 0 300 284" role="img" aria-label={`First real projection ${format(state.first[0])}; third real projection ${format(state.third[0])}; sum ${format(state.signal)}.`}><circle cx="150" cy="144" r="96" className="transform-grid" fill="none" /><circle cx="150" cy="144" r="48" className="transform-grid" fill="none" /><line x1="26" x2="274" y1="144" y2="144" className="transform-axis" /><line x1="150" x2="150" y1="20" y2="268" className="transform-axis" />{[state.first, state.third].map((point, index) => <g key={index} stroke={colors[index]}><line x1="150" y1="144" x2={150 + 48 * point[0]} y2={144 - 48 * point[1]} strokeWidth="2.5" /><line x1={150 + 48 * point[0]} x2={150 + 48 * point[0]} y1={144 - 48 * point[1]} y2="144" strokeDasharray="4 3" /><circle cx={150 + 48 * point[0]} cy="144" r="4" fill={colors[index]} /></g>)}<text x="260" y="166">Re</text><text x="160" y="27">Im</text><text x="242" y="138">2</text><text x="35" y="138">−2</text></svg></figure><Chart title="Their real projections produce the signal" lines={[curve.map(p => [p.time, p.first[0]]), curve.map(p => [p.time, p.third[0]]), curve.map(p => [p.time, p.signal])]} xDomain={[0, 1]} yDomain={[-3, 3]} xLabel="time (s)" yLabel="voltage (V)" cursor={state.time} /></div>
    <Legend labels={['1 Hz, amplitude 2', '3 Hz, amplitude 1', 'Sum of real projections']} /><Values items={[['First real projection', format(state.first[0])], ['Third real projection', format(state.third[0])], ['Measured sum', format(state.signal)]]} />
    <p>Advance a full second: both arrows return. Change only the third phase: the first trace stays fixed, showing why this is not a delay of the entire signal.</p><button onClick={() => {
      setTime(0);
      setPhase(Math.PI / 2);
    }}>Reset synthesis</button>
  </Investigation>;
}
export function HarmonicProjectionLab() {
  const [phase, setPhase] = useState(Math.PI / 2),
    [harmonic, setHarmonic] = useState(1),
    [offset, setOffset] = useState(0),
    [cursor, setCursor] = useState(256);
  const state = useMemo(() => harmonicProjection(2, 1, phase, offset, harmonic), [phase, offset, harmonic]);
  const active = state.points[cursor];
  return <Investigation id="projection" title="Unwind one frequency and accumulate what remains">
    <p>The signal is 2 cos(2πt)+cos(6πt+φ)+offset. Multiplying by exp(−2πikt) rotates each signed signal value. The accumulated path is an analytic integral, not a noisy simulation.</p>
    <div className="transform-controls"><Range label="Selected harmonic k" value={harmonic} onChange={setHarmonic} min={-4} max={4} /><Range label="Third-tone phase" value={phase} onChange={setPhase} min={-Math.PI} max={Math.PI} step={Math.PI / 12} display={phaseText(phase)} /><Range label="Signal offset" value={offset} onChange={setOffset} min={-2} max={2} step={.25} /><Range label="Integration cursor" value={cursor} onChange={setCursor} min={0} max={256} display={`${format(active.time)} s`} /></div>
    <div className="transform-two"><Chart title="The actual signal" lines={[state.points.map(p => [p.time, p.signal])]} xDomain={[0, 1]} yDomain={[-5, 5]} xLabel="time (s)" yLabel="voltage (V)" cursor={active.time} /><Plane title="Integral from 0 to the cursor" curve={state.points.map(p => p.accumulated)} vectors={[active.accumulated]} selected={state.coefficient} /></div>
    <Chart title="Two signed components of the integrand" lines={[state.points.map(p => [p.time, p.integrand[0]]), state.points.map(p => [p.time, p.integrand[1]])]} xDomain={[0, 1]} yDomain={[-5, 5]} xLabel="time (s)" yLabel="weighted value (V)" cursor={active.time} />
    <Legend labels={['Real component', 'Imaginary component']} /><Values items={[['Integral so far', complexText(active.accumulated)], ['Full-period coefficient cₖ', complexText(state.coefficient)], ['Full signal mean square', `${format(state.meanSquare)} V²`]]} />
    <p>The open ring marks the full-period endpoint. Try k=2: local contributions are substantial but cancel over one period. Change the offset and select k=0. Negative signal values reverse a contribution; this is a signed integral, not the center of mass of a positive wire.</p>
    <Table headers={['t (s)', 'Signal', 'Accumulated complex integral']} rows={state.points.filter((_, i) => i % 32 === 0).map(p => [format(p.time), format(p.signal), complexText(p.accumulated)])} />
    <button onClick={() => {
      setPhase(Math.PI / 2);
      setHarmonic(1);
      setOffset(0);
      setCursor(256);
    }}>Reset projection</button>
  </Investigation>;
}
export function FourierConvergenceLab() {
  const [terms, setTerms] = useState(4);
  const state = useMemo(() => squareConvergence(terms), [terms]);
  return <Investigation id="convergence" title="A narrower overshoot is still an overshoot">
    <p>Add odd harmonics to a square wave with levels −1 and +1. Compare the fixed jump location with the moving first peak.</p>
    <Range label="Number of odd harmonics" value={terms} onChange={setTerms} min={1} max={64} />
    <Chart title="Square wave and finite Fourier sum" lines={[state.points.map(p => [p.time, p.value]), [[-.25, -1], [0, -1]], [[0, 1], [.25, 1]]]} xDomain={[-.25, .25]} yDomain={[-1.5, 1.5]} xLabel="time (s)" yLabel="amplitude" cursor={state.firstPeakTime} />
    <Chart title="Zoom with the moving peak" lines={[state.nearPoints.map(p => [p.scaledTime, p.value]), [[0, 1], [1.8, 1]]]} xDomain={[0, 1.8]} yDomain={[0, 1.5]} xLabel="scaled time 4mt" yLabel="amplitude" cursor={1} />
    <p>This second view follows a shrinking time neighborhood: scaled time 1 always means t=1/(4m). The green line is the right-hand level +1. The full plot above shows the peak moving; this zoom makes its persistent height visible.</p>
    <Values items={[['At the jump t=0', format(state.midpoint)], ['First peak time', `${format(state.firstPeakTime, 6)} s`], ['First peak height', format(state.firstPeak, 6)], ['Mean squared error', format(state.meanSquaredError, 6)]]} />
    <p>The gold curve approaches the side values at fixed non-jump points. Its peak moves toward the discontinuity and tends to about 1.179. The mean squared error decreases. These are different convergence questions.</p>
    <Table headers={['Question', 'Model result']} rows={[['Sₘ(0)', format(state.midpoint)], ['Sₘ(0.2)', format(state.away, 6)], ['Sₘ(1/(4m))', format(state.firstPeak, 6)], ['Integral squared error over one period', format(state.meanSquaredError, 6)]]} />
    <button onClick={() => setTerms(4)}>Reset harmonics</button>
  </Investigation>;
}
export function PulseTransformFigure() {
  const frequencies = Array.from({
    length: 385
  }, (_, i) => -3 + i / 64);
  return <section className="transform-inline"><div className="transform-two"><Chart title="Two centered unit-height pulses" lines={[[[-1.5, 0], [-.5, 0], [-.5, 1], [.5, 1], [.5, 0], [1.5, 0]], [[-1.5, 0], [-1, 0], [-1, 1], [1, 1], [1, 0], [1.5, 0]]]} xDomain={[-1.5, 1.5]} yDomain={[0, 1.2]} xLabel="time (s)" yLabel="voltage (V)" /><Chart title="Their real Fourier transforms" lines={[frequencies.map(f => [f, sinc(f)]), frequencies.map(f => [f, 2 * sinc(2 * f)])]} xDomain={[-3, 3]} yDomain={[-.5, 2]} xLabel="frequency (Hz)" yLabel="X(f) (V·s)" /></div><Legend labels={['Width 1 s', 'Width 2 s']} /><p>Doubling the width doubles the zero-frequency area and halves the first-zero frequency. These are analytic sinc curves, not measured spectra.</p></section>;
}
export function FiniteFourierLab() {
  const [draft, setDraft] = useState('1, 2, 0, -1'),
    [values, setValues] = useState([1, 2, 0, -1]),
    [bin, setBin] = useState(1),
    [remove, setRemove] = useState(false),
    [error, setError] = useState('');
  const state = useMemo(() => finiteFourier(values, bin, remove), [values, bin, remove]);
  const sampleExtent = Math.max(1, ...values.map(Math.abs), ...state.reconstructed.map(value => Math.abs(value[0]))) * 1.15;
  function apply() {
    try {
      const next = parseRealSamples(draft);
      setValues(next);
      setBin(Math.min(bin, next.length - 1));
      setError('');
    } catch (failure) {
      setError(failure.message);
    }
  }
  return <Investigation id="dft" title="Watch samples become a complex coefficient">
    <p>For the selected bin, each sample contributes one arrow. Join the arrows tip to tail: their endpoint is X[k]. Removing a conjugate pair changes the inverse signal while keeping it real.</p>
    <div className="transform-draft"><label>Four or eight real samples, each between −4 and 4<input aria-label="DFT sample draft" value={draft} onChange={event => setDraft(event.target.value)} /></label><button onClick={apply}>Apply sample list</button></div>{error && <p role="alert">{error} The active sample list is unchanged.</p>}
    <div className="transform-controls"><Range label="DFT bin k" value={bin} onChange={setBin} min={0} max={values.length - 1} /><label className="transform-toggle"><input type="checkbox" checked={remove} onChange={event => setRemove(event.target.checked)} />Remove k and its conjugate partner from the inverse</label></div>
    <div className="transform-two"><Plane title={`Contributions to X[${bin}]`} vectors={state.contributions} chain /><Chart title="Input and reconstructed sample values" dots={[values.map((v, i) => [i, v]), state.reconstructed.map((v, i) => [i, v[0]])]} xDomain={[0, values.length - 1]} yDomain={[-sampleExtent, sampleExtent]} xLabel="sample index n" yLabel="value" /></div>
    <Legend labels={['Original samples', 'Inverse after selected action']} /><div className="transform-bin-strip">{state.spectrum.map((value, index) => <button key={index} aria-pressed={bin === index} onClick={() => setBin(index)}><span>k={index}</span><strong>{complexText(value)}</strong></button>)}</div>
    <Values items={[['Selected coefficient', complexText(state.spectrum[bin])], ['Its principal phase', Math.hypot(...state.spectrum[bin]) <= 1e-10 ? 'Not displayed (≤10⁻¹⁰)' : phaseText(complexPolar(state.spectrum[bin]).phase)], ['Σ |x[n]|²', format(state.timeEnergy)], ['Σ |X[k]|² / N', format(state.frequencyEnergy)]]} />
    <p>Energy values refer to the original transform. Phase is not displayed at magnitudes at or below 10⁻¹⁰, to avoid assigning meaning to a direction dominated by rounding. A deliberately tiny coefficient can still be mathematically nonzero. Numeric readouts below 10⁻¹¹ are printed as 0; that is a display convention. The calculation retains the supplied samples.</p>
    <Table headers={['n', 'Contribution to selected X[k]', 'Inverse real', 'Inverse imaginary']} rows={state.contributions.map((value, index) => [index, complexText(value), format(state.reconstructed[index][0]), format(state.reconstructed[index][1])])} />
    <button onClick={() => {
      setDraft('1, 2, 0, -1');
      setValues([1, 2, 0, -1]);
      setBin(1);
      setRemove(false);
      setError('');
    }}>Reset finite transform</button>
  </Investigation>;
}
export function FourierButterflyFigure() {
  const even = finiteFourier([1, 0], 1).spectrum;
  const odd = finiteFourier([2, -1], 1).spectrum;
  const rotated = complexOperation([0, -1], odd[1]).result;
  const outputs = finiteFourier([1, 2, 0, -1], 1).spectrum;
  const routes = [[70, 66, 70, 233, 0], [70, 66, 220, 233, 0], [220, 154, 70, 233, 1], [220, 154, 220, 233, 1]];
  return <section className="transform-inline"><h3>Reuse the even and odd transforms</h3><div className="transform-two"><figure className="transform-chart"><figcaption>One shared pair creates two outputs</figcaption><svg viewBox="0 0 300 348" role="img" aria-label="E[1]=1 feeds both output combinations. O[1]=3 is multiplied by minus i to make R=minus 3i. X[1]=E+R=1 minus 3i, X[3]=E minus R=1 plus 3i.">
    <text x="70" y="24" textAnchor="middle">E[1]</text><text x="70" y="48" textAnchor="middle">{complexText(even[1])}</text>
    <text x="220" y="24" textAnchor="middle">O[1]</text><text x="220" y="48" textAnchor="middle">{complexText(odd[1])}</text>
    <line x1="220" x2="220" y1="59" y2="84" stroke={colors[1]} strokeWidth="2" /><rect x="176" y="85" width="88" height="29" fill="#17231b" stroke={colors[1]} /><text x="220" y="105" textAnchor="middle">× (−i)</text><text x="220" y="144" textAnchor="middle">R = {complexText(rotated)}</text>
    {routes.map(([x1, y1, x2, y2, color], index) => {
            const angle = Math.atan2(y2 - y1, x2 - x1);
            return <g key={index} stroke={colors[color]} fill="none" strokeWidth="2"><line x1={x1} y1={y1} x2={x2} y2={y2} /><path d={`M${x2 - 7 * Math.cos(angle - .45)},${y2 - 7 * Math.sin(angle - .45)} L${x2},${y2} L${x2 - 7 * Math.cos(angle + .45)},${y2 - 7 * Math.sin(angle + .45)}`} /></g>;
          })}
    <circle cx="70" cy="248" r="14" fill="#151913" stroke={colors[0]} /><circle cx="220" cy="248" r="14" fill="#151913" stroke={colors[0]} /><text x="70" y="253" textAnchor="middle">+</text><text x="220" y="253" textAnchor="middle">−</text><text x="70" y="288" textAnchor="middle">X[1] = E + R</text><text x="220" y="288" textAnchor="middle">X[3] = E − R</text><text x="70" y="316" textAnchor="middle">{complexText(outputs[1])}</text><text x="220" y="316" textAnchor="middle">{complexText(outputs[3])}</text>
  </svg></figure><div><p>The even samples [1,0] give E=[1,1]; the odd samples [2,−1] give O=[1,3]. The gold routes carry the same even result to both combinations. The green routes reuse the same rotated odd result.</p><p>The left output adds R. The right output subtracts R from E, so subtracting −3i produces +3i. Crossing routes do not merge into a new intermediate value.</p><p>At k=0 the same add/subtract gives X[0]=2 and X[2]=0. Recursing this split is the radix-2 FFT.</p></div></div></section>;
}
export function SamplingAliasLab() {
  const [frequency, setFrequency] = useState(13),
    [rate, setRate] = useState(16),
    [phase, setPhase] = useState(Math.PI / 3);
  const state = useMemo(() => aliasState(frequency, rate, phase), [frequency, rate, phase]);
  return <Investigation id="alias" title="Different continuous waves, identical stored samples">
    <div className="transform-controls"><Range label="Continuous tone frequency" value={frequency} onChange={setFrequency} min={0} max={24} step={.25} display={`${format(frequency)} Hz`} /><Choice label="Sampling rate" value={rate} onChange={value => setRate(Number(value))} options={[[8, '8 samples/s'], [16, '16 samples/s'], [32, '32 samples/s']]} /><Range label="Tone phase" value={phase} onChange={setPhase} min={-Math.PI} max={Math.PI} step={Math.PI / 12} display={phaseText(phase)} /></div>
    <Chart title="Only the dots are observed" lines={[state.curves.map(p => [p.time, p.original]), state.curves.map(p => [p.time, p.alias])]} dots={[state.samples.map(p => [p.time, p.original])]} xDomain={[0, 1]} yDomain={[-1.2, 1.2]} xLabel="time (s)" yLabel="amplitude" />
    <Legend labels={['Original continuous tone / samples', 'Folded continuous alias']} /><Values items={[['Signed alias', `${format(state.signedAlias)} Hz`], ['Positive-frequency alias', `${format(state.foldedFrequency)} Hz`], ['Folded phase', phaseText(state.foldedPhase)], ['Maximum sample discrepancy', format(Math.max(...state.samples.map(p => Math.abs(p.original - p.alias))), 10)]]} />
    <p>Folding a negative frequency reverses the phase of a real cosine. At the Nyquist boundary a sine can vanish at every sample; equality at the sampling limit is not a general reconstruction guarantee.</p>
    <Table headers={['n', 't (s)', 'Original', 'Alias']} rows={state.samples.map(p => [p.index, format(p.time), format(p.original), format(p.alias)])} />
    <button onClick={() => {
      setFrequency(13);
      setRate(16);
      setPhase(Math.PI / 3);
    }}>Reset sampling</button>
  </Investigation>;
}
export function WindowSpectrumLab() {
  const [count, setCount] = useState(64),
    [padded, setPadded] = useState(64),
    [windowName, setWindow] = useState('rectangular'),
    [tone, setTone] = useState(5.5),
    [cursor, setCursor] = useState(5.5);
  const state = useMemo(() => windowSpectrum(count, padded, windowName, tone), [count, padded, windowName, tone]);
  const active = state.bins.reduce((best, row) => Math.abs(row.frequency - cursor) < Math.abs(best.frequency - cursor) ? row : best, state.bins[0]);
  return <Investigation id="window" title="Separate observation length, taper and zero padding">
    <p>A finite record multiplies the continuing tone by a time window. The spectral envelope comes from those observed samples. Zero padding selects more points on that same envelope.</p>
    <div className="transform-controls"><Choice label="Observed samples N" value={count} onChange={value => {
        const n = Number(value);
        setCount(n);
        setPadded(Math.max(n, padded));
      }} options={[[32, '32 samples'], [64, '64 samples'], [128, '128 samples']]} /><Choice label="DFT length M" value={padded} onChange={value => setPadded(Number(value))} options={[32, 64, 128, 256, 512].filter(n => n >= count).map(n => [n, `${n} points`])} /><Choice label="Time window" value={windowName} onChange={setWindow} options={[['rectangular', 'Rectangular'], ['hann', 'Periodic Hann']]} /><Range label="Windowed tone frequency" value={tone} onChange={setTone} min={3} max={9} step={.125} display={`${format(tone)} Hz`} /></div>
    <div className="transform-two"><Chart title="Observed samples and taper" lines={[state.samples.map((v, i) => [i / 64, v]), state.weights.map((v, i) => [i / 64, v]), state.weighted.map((v, i) => [i / 64, v])]} xDomain={[0, count / 64]} yDomain={[-1.1, 1.1]} xLabel="time (s)" yLabel="sample / weight" /><Chart title="Spectral envelope and actual DFT bins" lines={[state.curve.map(p => [p.frequency, p.amplitude])]} dots={[state.bins.filter(p => p.frequency <= 12).map(p => [p.frequency, p.amplitude])]} xDomain={[0, 12]} yDomain={[0, 1.2]} xLabel="frequency (Hz)" yLabel="amplitude scale (V)" cursor={active.frequency} /></div>
    <Legend labels={['Signal / spectral amplitude', 'Window weights', 'Weighted signal']} /><Range label="Inspect frequency" value={cursor} onChange={setCursor} min={0} max={12} step={.125} display={`${format(active.frequency)} Hz bin`} />
    <Values items={[['Observation span N/fₛ', `${format(state.observationPeriod)} s`], ['DFT bin spacing fₛ/M', `${format(state.gridSpacing)} Hz`], ['Σ w²x² / Σ w²', `${format(state.weightedMeanSquare, 6)} V²`], ['Sum of one-sided density × Δf', `${format(state.densityIntegral, 6)} V²`], ['Selected density', `${format(active.density, 6)} V²/Hz`]]} />
    <p>No mean removal or other detrending is applied. The amplitude scale divides by Σw and doubles interior positive frequencies; DC is not doubled. The density divides squared magnitude by fₛΣw², with one-sided pairs merged. Its integral identity includes all frequencies through Nyquist, not only the displayed 0–12 Hz region.</p>
    <p>Try M=256 with N=64: denser bins, unchanged envelope and observation. Then N=128: a longer observation changes the envelope. A Hann taper reduces distant sidelobes while widening the central lobe. These are exact finite signal calculations, not a claim about statistical PSD accuracy.</p>
    <Table headers={['Frequency', 'Amplitude scale', 'Density (V²/Hz)']} rows={state.bins.filter(p => p.frequency <= 12).map(p => [format(p.frequency), format(p.amplitude, 6), format(p.density, 6)])} />
    <button onClick={() => {
      setCount(64);
      setPadded(64);
      setWindow('rectangular');
      setTone(5.5);
      setCursor(5.5);
    }}>Reset window experiment</button>
  </Investigation>;
}
export function ConvolutionBoundaryLab() {
  const [circular, setCircular] = useState(false),
    [index, setIndex] = useState(0);
  const state = convolutionState([1, 2, 0, -1], [1, 1], circular, index);
  return <Investigation id="convolution" title="Follow the boundary term instead of losing it">
    <p>Use x=[1,2,0,−1] and h=[1,1]. Each output is the sum of x[j]h[n−j]. A circular boundary wraps a kernel index modulo 4.</p>
    <div className="transform-controls"><Choice label="Convolution boundary" value={circular ? 'circular' : 'linear'} onChange={value => {
        setCircular(value === 'circular');
        setIndex(Math.min(index, value === 'circular' ? 3 : 4));
      }} options={[['linear', 'Linear: zero outside'], ['circular', 'Circular: wrap modulo 4']]} /><Range label="Convolution output index" value={index} onChange={setIndex} min={0} max={state.output.length - 1} /></div>
    <div className="transform-convolution">{state.terms.map(term => <div key={term.index} className={term.wraps ? 'is-wrapped' : ''}><span>x[{term.index}]={term.value}</span><span>× h[{term.kernelIndex}]={term.weight}</span><strong>{term.product}</strong>{term.wraps && <em>wrapped index</em>}</div>)}</div>
    <p className="transform-equation-line">{state.terms.map(term => term.product).join(' + ')} = <strong>y[{index}] = {state.output[index]}</strong></p>
    <div className="transform-bin-strip">{state.output.map((value, n) => <button key={n} aria-pressed={index === n} onClick={() => setIndex(n)}><span>y[{n}]</span><strong>{value}</strong></button>)}</div>
    <p>Linear convolution retains y[4]=−1. A length-4 circular result folds that tail into y[0], changing 1 to 0. Padding both inputs to at least 4+2−1=5 prevents overlap.</p>
    <button onClick={() => {
      setCircular(false);
      setIndex(0);
    }}>Reset convolution</button>
  </Investigation>;
}
export function FilterResponseLab() {
  const [rate, setRate] = useState(TAU),
    [initial, setInitial] = useState(0),
    [phase, setPhase] = useState(Math.PI / 2),
    [cursor, setCursor] = useState(0);
  const state = useMemo(() => filterResponse(rate, phase, initial), [rate, phase, initial]);
  const responseLines = useMemo(() => ['total', 'steady', 'transient'].map(key => state.points.map(point => [point.time, point[key]])), [state]);
  const responseExtent = useMemo(() => Math.max(1, ...responseLines.flatMap(line => line.map(point => Math.abs(point[1])))) * 1.15, [responseLines]);
  const point = state.points[cursor];
  return <Investigation id="filter" title="The steady oscillation does not choose the initial value">
    <p>For y′+ay=ax, each tone is scaled and phase shifted by H(iω)=a/(a+iω). A decaying correction supplies the requested y(0).</p>
    <div className="transform-controls"><Range label="Decay rate a" value={rate} onChange={setRate} min={.5} max={20} step="any" display={`${format(rate)} s⁻¹`} /><Range label="Initial output y(0)" value={initial} onChange={setInitial} min={-3} max={3} step={.25} /><Range label="Filter third-tone phase" value={phase} onChange={setPhase} min={-Math.PI} max={Math.PI} step={Math.PI / 12} display={phaseText(phase)} /><Range label="Response cursor" value={cursor} onChange={setCursor} min={0} max={512} display={`${format(point.time)} s`} /></div>
    <div className="transform-two"><Chart title="Steady response + initial-condition correction" lines={responseLines} xDomain={[0, 2]} yDomain={[-responseExtent, responseExtent]} xLabel="time (s)" yLabel="output (V)" cursor={point.time} /><div><Legend labels={['Total output', 'Steady periodic response', 'Transient correction']} /><Values items={[['At the cursor: total', format(point.total)], ['Steady part', format(point.steady)], ['Transient part', format(point.transient)], ['Corner frequency a/(2π)', `${format(rate / TAU)} Hz`]]} /></div></div>
    <Table summary="Read each tone's gain and phase" headers={['Tone', 'Gain', 'Phase lag']} rows={state.modes.map(mode => [`${mode.frequency} Hz`, format(mode.gain, 6), phaseText(mode.lag)])} />
    <p>Changing the initial value changes only the transient. A negative phase lag means a delay for each single tone; different lags divided by different angular frequencies need not give one shared time delay.</p>
    <button onClick={() => {
      setRate(TAU);
      setInitial(0);
      setPhase(Math.PI / 2);
      setCursor(0);
    }}>Reset filter</button>
  </Investigation>;
}
export function LaplaceRegionLab() {
  const [sigma, setSigma] = useState(0),
    [omega, setOmega] = useState(1),
    [horizon, setHorizon] = useState(2),
    [side, setSide] = useState('right');
  const state = useMemo(() => laplaceRegion(1, sigma, omega, horizon, side), [sigma, omega, horizon, side]);
  const extent = Math.max(1, ...state.points.map(p => Math.hypot(...p.value))) * 1.1;
  return <Investigation id="laplace" title="The same rational formula can describe different signals">
    <p>Right-sided e⁻ᵗ for t≥0 and left-sided −e⁻ᵗ for t≤0 both lead to 1/(s+1), but on opposite half-planes. Select the signal support before interpreting the formula.</p>
    <div className="transform-controls"><Choice label="Exponential support" value={side} onChange={setSide} options={[['right', 'Right-sided e⁻ᵗ'], ['left', 'Left-sided −e⁻ᵗ']]} /><Range label="Real part sigma" value={sigma} onChange={setSigma} min={-3} max={3} step={.25} /><Range label="Angular frequency omega" value={omega} onChange={setOmega} min={-4} max={4} step={.25} /><Range label="Finite integration horizon" value={horizon} onChange={setHorizon} min={.25} max={8} step={.25} display={`${format(horizon)} s`} /></div>
    <div className="transform-two"><figure className="transform-chart"><figcaption>Region of absolute convergence</figcaption><svg viewBox="0 0 300 250" role="img" aria-label={`${side === 'right' ? 'Right' : 'Left'} half-plane of the boundary sigma=-1 is the region of convergence. Selected sigma=${sigma}, omega=${omega}.`}>
      <rect x={side === 'right' ? 118 : 38} y="24" width={side === 'right' ? 160 : 80} height="176" fill="#73caa0" opacity=".13" /><line x1="38" x2="278" y1="112" y2="112" className="transform-axis" /><line x1="158" x2="158" y1="24" y2="200" className="transform-axis" /><line x1="118" x2="118" y1="24" y2="200" stroke={colors[3]} strokeDasharray="5 4" /><circle cx={158 + sigma * 40} cy={112 - omega * 22} r="6" fill={colors[0]} /><text x="118" y="221" textAnchor="middle">−1</text><text x="271" y="239" textAnchor="end">σ (real)</text><text x="165" y="17">ω (imaginary)</text><text x="42" y="221">−3</text><text x="273" y="221">3</text><text x="164" y="40">4</text><text x="164" y="198">−4</text><text x="43" y="17">ROC shaded</text>
    </svg></figure><Chart title="Finite integral as the horizon grows" lines={[state.points.map(p => [p.horizon, p.value[0]]), state.points.map(p => [p.horizon, p.value[1]])]} xDomain={[0, horizon]} yDomain={[-extent, extent]} xLabel="horizon T (s)" yLabel="integral components" /></div>
    <Legend labels={['Real integral / selected s', 'Imaginary integral']} /><Values items={[['Selected region', state.converges ? 'Inside the ROC' : state.boundary ? 'Boundary: no ordinary limit' : 'Outside the ROC'], ['Finite integral', complexText(state.value)], ['Algebraic 1/(s+1)', state.rational ? complexText(state.rational) : 'Pole: undefined']]} />
    <p>{state.converges ? 'The finite integral tends to the rational value as the horizon tends to infinity.' : 'A finite calculation still produces a number. It is not the infinite transform at this s. On the boundary an oscillatory integral need not settle; outside, exponential weighting grows.'} The boundary is excluded even when the denominator is nonzero.</p>
    <Table headers={['T', 'Finite integral']} rows={state.points.filter((_, index) => index % 16 === 0).map(p => [format(p.horizon), complexText(p.value)])} />
    <button onClick={() => {
      setSigma(0);
      setOmega(1);
      setHorizon(2);
      setSide('right');
    }}>Reset Laplace region</button>
  </Investigation>;
}
