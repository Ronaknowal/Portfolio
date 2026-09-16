import { useId, useState } from 'react';
import { eigenDirectionPresets, eigenDirectionState, repeatedMapPresets, repeatedMapStarts, repeatedMapTrace, pcaDirectionDatasets, pcaDirectionState } from '../../data/eigenvalue-models.js';
import './eigenvalue-labs.css';
const number = value => String(Number(value.toFixed(3)));
const pair = vector => '[' + vector.map(number).join(', ') + ']';
const colors = {
  gold: '#eabf61',
  blue: '#9ebede',
  green: '#adca9b',
  quiet: '#b9b6aa'
};
function MatrixValues({
  matrix,
  label = 'A'
}) {
  return <table className="eigen-matrix"><caption>{label}</caption><tbody>{matrix.map((row, rowIndex) => <tr key={rowIndex}>{row.map((value, column) => <td key={column}>{number(value)}</td>)}</tr>)}</tbody></table>;
}
function CoordinatePlot({
  title,
  description,
  range = 4,
  vectors = [],
  points = [],
  segments = [],
  lineDirection = null,
  trail = []
}) {
  const id = useId();
  const center = 160;
  const scale = 124 / range;
  const screen = point => [center + scale * point[0], center - scale * point[1]];
  const ticks = [-1, -0.5, 0.5, 1].map(fraction => fraction * range);
  return <svg className="eigen-plane" viewBox="0 0 320 320" role="img" aria-labelledby={`${id}-title ${id}-description`}>
    <title id={`${id}-title`}>{title}</title><desc id={`${id}-description`}>{description} Both axes have the same scale, from minus {range} to plus {range}. Numerical alternatives follow the plot.</desc>
    {ticks.map(tick => <g key={tick} className="eigen-grid"><line x1={screen([tick, 0])[0]} x2={screen([tick, 0])[0]} y1="30" y2="290" /><line x1="30" x2="290" y1={screen([0, tick])[1]} y2={screen([0, tick])[1]} /></g>)}
    <g className="eigen-axis"><line x1="24" x2="296" y1={center} y2={center} /><line x1={center} x2={center} y1="24" y2="296" /></g>
    {ticks.map(tick => <g className="eigen-ticks" key={tick}><text x={screen([tick, 0])[0]} y={center + 16} textAnchor="middle">{number(tick)}</text><text x={center - 8} y={screen([0, tick])[1] + 4} textAnchor="end">{number(tick)}</text></g>)}
    <text x="302" y="155">x</text><text x="165" y="18">y</text>
    {lineDirection && <line className="eigen-reference-line" x1={screen(lineDirection.map(value => -range * value))[0]} y1={screen(lineDirection.map(value => -range * value))[1]} x2={screen(lineDirection.map(value => range * value))[0]} y2={screen(lineDirection.map(value => range * value))[1]} />}
    {trail.length > 1 && <polyline points={trail.map(point => screen(point).join(',')).join(' ')} fill="none" stroke={colors.blue} strokeWidth="1.5" />}
    {segments.map((segment, index) => <line key={index} x1={screen(segment[0])[0]} y1={screen(segment[0])[1]} x2={screen(segment[1])[0]} y2={screen(segment[1])[1]} stroke={colors.quiet} strokeDasharray="3 4" />)}
    {vectors.map(({
      value,
      color,
      dashed,
      label
    }) => {
      const [x, y] = screen(value);
      const angle = Math.atan2(y - center, x - center);
      const arrow = [[x, y], [x - 9 * Math.cos(angle - 0.4), y - 9 * Math.sin(angle - 0.4)], [x - 9 * Math.cos(angle + 0.4), y - 9 * Math.sin(angle + 0.4)]];
      return <g key={label}><title>{label}: {pair(value)}</title><line x1={center} y1={center} x2={x} y2={y} stroke={color} strokeWidth="3" strokeDasharray={dashed ? '6 4' : undefined} />{Math.hypot(...value) < 1e-10 ? <circle cx={x} cy={y} r="4" fill={color} /> : <polygon points={arrow.map(point => point.join(',')).join(' ')} fill={color} />}</g>;
    })}
    {points.map(({
      value,
      label,
      projected
    }, index) => <g key={index}><circle cx={screen(value)[0]} cy={screen(value)[1]} r={projected ? 4 : 5} fill={projected ? 'none' : colors.gold} stroke={projected ? colors.green : colors.gold} strokeWidth="2" /><title>{label}: {pair(value)}</title>{!projected && <text x={screen(value)[0] + (value[0] < 0 ? -8 : 8)} y={screen(value)[1] - 7} textAnchor={value[0] < 0 ? 'end' : 'start'}>{label}</text>}</g>)}
  </svg>;
}
function AngleControl({
  value,
  onChange,
  label,
  maximum = 360
}) {
  return <label className="eigen-range">{label}: <strong>{value}°</strong><input type="range" min="0" max={maximum} step="5" value={value} onChange={event => onChange(Number(event.target.value))} /></label>;
}
export function EigenDirectionLab() {
  const [preset, setPreset] = useState('diagonalStretch');
  const [angle, setAngle] = useState(30);
  const state = eigenDirectionState(preset, angle);
  function reset() {
    setPreset('diagonalStretch');
    setAngle(30);
  }
  return <section className="eigen-lab" aria-label="Preserved direction investigation">
    <p className="eigen-eyebrow">KEEP A LINE · NOT NECESSARILY ITS ORIENTATION</p>
    <h3>Which inputs stay on their own line?</h3>
    <p>Move the unit input toward 45°, then 135°. Predict the signed scale before reading the result. The green ring marks the along-line part of the output; its dotted connector to the gold tip is the perpendicular remainder. A new preset applies immediately; angles are measured counterclockwise from the horizontal axis.</p>
    <div className="eigen-controls"><label>Transformation<select value={preset} onChange={event => setPreset(event.target.value)}>{Object.entries(eigenDirectionPresets).map(([key, item]) => <option key={key} value={key}>{item.title}</option>)}</select></label><AngleControl label="Input direction" value={angle} onChange={setAngle} /><button onClick={reset}>Reset direction</button></div>
    <div className="eigen-columns"><CoordinatePlot title="Input and transformed direction" description="The reference line passes through the input. A green ring marks the along-line projection of the output; the dotted connector to the gold tip is its perpendicular remainder. A zero output stays at the origin." points={[{ value: state.along, label: 'Along-line projection', projected: true }]} segments={[[state.along, state.output]]} vectors={[{
        value: state.output,
        color: colors.gold,
        label: 'Output Av'
      }, {
        value: state.input,
        color: colors.blue,
        dashed: true,
        label: 'Unit input v'
      }]} lineDirection={state.input} /><div><MatrixValues matrix={state.matrix} /><dl className="eigen-readings"><dt>Dashed blue input v</dt><dd>{pair(state.input)}</dd><dt>Solid gold output Av</dt><dd>{pair(state.output)}</dd><dt>Signed factor along v</dt><dd>{number(state.alongFactor)}</dd><dt>Output left perpendicular to v</dt><dd>{pair(state.perpendicular)}; length {number(state.residualNorm)}</dd></dl></div></div>
    <p className="eigen-feedback" aria-live="polite">{state.isEigenDirection ? state.isZeroOutput ? 'A nonzero input maps to zero: this is an eigenvector with eigenvalue 0. The output has no direction to draw.' : `This line is preserved: the eigenvalue is ${number(state.alongFactor)}. A negative factor reverses the arrow along the same line.` : 'This line is not preserved. The along-line factor is a projection coefficient, not an eigenvalue for this input.'}</p>
    <details><summary>Interpret this transformation</summary><p>{state.interpretation}</p></details>
    <p className="eigen-caption">Every quantity is calculated from the displayed 2×2 matrix. Values are rounded to three decimals. The perpendicular-residual classification uses tolerance 10⁻¹⁰ for these presets, not a universal numerical test.</p>
  </section>;
}
function NormHistory({
  states,
  selected
}) {
  const id = useId();
  const largest = Math.max(1, ...states.map(state => state.norm));
  const x = step => 44 + 294 * step / Math.max(1, states.length - 1);
  const y = norm => 174 - 134 * norm / largest;
  return <svg className="eigen-history" viewBox="0 0 370 214" role="img" aria-labelledby={`${id}-title ${id}-desc`}>
    <title id={`${id}-title`}>Vector length after each update</title><desc id={`${id}-desc`}>Exact norms from the displayed recurrence, connected to guide the eye. Horizontal axis is integer update number; vertical axis is Euclidean norm. A table contains every point.</desc>
    <line className="eigen-axis" x1="44" x2="346" y1="174" y2="174" /><line className="eigen-axis" x1="44" x2="44" y1="32" y2="174" />
    <text x="8" y="20">‖xₖ‖</text><text x="174" y="207">update k</text>
    {[0, 0.5, 1].map(fraction => <g key={fraction}><text x="37" y={y(fraction * largest) + 4} textAnchor="end">{number(fraction * largest)}</text><line className="eigen-grid" x1="44" x2="338" y1={y(fraction * largest)} y2={y(fraction * largest)} /></g>)}
    {[0, 3, 6, 9, 12].filter(step => step < states.length).map(step => <text key={step} x={x(step)} y="191" textAnchor="middle">{step}</text>)}
    <polyline points={states.map(state => `${x(state.step)},${y(state.norm)}`).join(' ')} fill="none" stroke={colors.blue} strokeWidth="2" />
    {states.map(state => <circle key={state.step} cx={x(state.step)} cy={y(state.norm)} r={state.step === selected ? 5 : 2.5} fill={state.step === selected ? colors.gold : colors.blue} />)}
  </svg>;
}
export function RepeatedMapLab() {
  const [preset, setPreset] = useState('decay');
  const [start, setStart] = useState('mixed');
  const [step, setStep] = useState(0);
  const trace = repeatedMapTrace(preset, start);
  const state = trace.states[step];
  const range = Math.max(2, Math.ceil(Math.max(...trace.states.flatMap(item => item.vector.map(Math.abs))) / 2) * 2);
  function reset() {
    setPreset('decay');
    setStart('mixed');
    setStep(0);
  }
  return <section className="eigen-lab" aria-label="Repeated matrix update investigation">
    <p className="eigen-eyebrow">REPEAT THE SAME RULE · KEEP THE ACTUAL SIZE</p><h3>Decay, oscillation and a temporary surge</h3>
    <p>Step through xₖ₊₁=Axₖ. Then compare “Temporary growth” with ordinary decay. Try a purely vertical start under “One growing coordinate”: an available growing direction need not be present in every input.</p>
    <div className="eigen-controls"><label>Update rule<select value={preset} onChange={event => {
          setPreset(event.target.value);
          setStep(0);
        }}>{Object.entries(repeatedMapPresets).map(([key, item]) => <option key={key} value={key}>{item.title}</option>)}</select></label><label>Starting vector<select value={start} onChange={event => {
          setStart(event.target.value);
          setStep(0);
        }}>{Object.entries(repeatedMapStarts).map(([key, item]) => <option key={key} value={key}>{item.title}</option>)}</select></label></div>
    <div className="eigen-controls"><button onClick={() => setStep(value => value - 1)} disabled={step === 0}>Previous update</button><button onClick={() => setStep(value => value + 1)} disabled={step === 12}>Next update</button><button onClick={reset}>Reset updates</button><span>Update {step} / 12</span></div>
    <div className="eigen-columns"><CoordinatePlot title="Raw state trajectory" description="The blue path records earlier raw vectors; the gold arrow is the current state. The scale is held fixed for all twelve updates of this case." range={range} vectors={[{
        value: state.vector,
        color: colors.gold,
        label: 'Current state'
      }]} trail={trace.states.slice(0, step + 1).map(item => item.vector)} /><NormHistory states={trace.states} selected={step} /></div>
    <div className="eigen-columns"><MatrixValues matrix={trace.matrix} /><div className="eigen-feedback" aria-live="polite"><p>x<sub>{step}</sub> = {pair(state.vector)}</p><p>Length = {number(state.norm)}; eigenvalues: {trace.eigenvalues}.</p></div></div>
    <details><summary>All computed states and interpretation</summary><p>{trace.conclusion}</p><table className="eigen-data"><caption>Unnormalized recurrence; k=0 is the starting vector</caption><thead><tr><th>k</th><th>xₖ</th><th>‖xₖ‖</th></tr></thead><tbody>{trace.states.map(item => <tr key={item.step}><th>{item.step}</th><td>{pair(item.vector)}</td><td>{number(item.norm)}</td></tr>)}</tbody></table></details>
    <p className="eigen-caption">Twelve calculated updates, not empirical observations or an asymptotic proof. The length plot includes the full trace for prediction; the selected gold point matches the left state. Coordinates use equal scales, and are not normalized to hide growth.</p>
  </section>;
}
export function PcaDirectionLab() {
  const [dataset, setDataset] = useState('diagonalCloud');
  const [angle, setAngle] = useState(0);
  const state = pcaDirectionState(dataset, angle);
  function reset() {
    setDataset('diagonalCloud');
    setAngle(0);
  }
  return <section className="eigen-lab" aria-label="Variance direction investigation">
    <p className="eigen-eyebrow">ONE LINE · HOW MUCH VARIATION SURVIVES?</p><h3>Turn a measurement direction through a small cloud</h3>
    <p>Start horizontally, then try 45° and 135°. Gold points are observations; green rings are their projections. Dotted connectors show what the one-dimensional representation loses. Two projected observations may coincide.</p>
    <div className="eigen-controls"><label>Dataset<select value={dataset} onChange={event => setDataset(event.target.value)}>{Object.entries(pcaDirectionDatasets).map(([key, item]) => <option key={key} value={key}>{item.title}</option>)}</select></label><AngleControl label="Measurement direction" value={angle} onChange={setAngle} maximum={180} /><button onClick={reset}>Reset variance</button></div>
    <div className="eigen-columns"><CoordinatePlot title="Observations and their projections" description="Each original point is connected to its perpendicular projection onto the selected line." range={3} lineDirection={state.direction} points={[...state.points.map((value, index) => ({
        value,
        label: String.fromCharCode(65 + index)
      })), ...state.projections.map((value, index) => ({
        value,
        label: `Projection ${String.fromCharCode(65 + index)}`,
        projected: true
      }))]} segments={state.points.map((point, index) => [point, state.projections[index]])} /><div><MatrixValues matrix={state.covariance} label="Sample covariance C" /><dl className="eigen-readings"><dt>Unit direction q</dt><dd>{pair(state.direction)}</dd><dt>Sample variance of Xq</dt><dd>{number(state.variance)}</dd><dt>Total sample variance</dt><dd>{number(state.totalVariance)}</dd><dt>Retained fraction</dt><dd>{number(100 * state.retainedFraction)}%</dd><dt>Sum of squared reconstruction errors</dt><dd>{number(state.squaredReconstructionError)}</dd></dl></div></div>
    <table className="eigen-data"><caption>Four invented, centered observations in the same arbitrary units</caption><thead><tr><th>Point</th><th>Original</th><th>Score qᵀx</th><th>Projection</th></tr></thead><tbody>{state.points.map((point, index) => <tr key={index}><th>{String.fromCharCode(65 + index)}</th><td>{pair(point)}</td><td>{number(state.scores[index])}</td><td>{pair(state.projections[index])}</td></tr>)}</tbody></table>
    <details><summary>Compare with the eigenvalues</summary><p>{state.conclusion}</p></details>
    <p className="eigen-caption">Sample covariance divides by n−1=3. Points, projections and variance are computed from the same four values. These geometric/error quantities do not establish a predictive or causal model.</p>
  </section>;
}
export function EigenbasisFlow() {
  return <figure className="eigen-inline"><figcaption>Separate a repeated transformation into independent coordinates</figcaption><ol className="eigen-flow"><li><strong>Input</strong><span>x₀=[2,0]</span></li><li><strong>Change basis</strong><span>S⁻¹x₀=[1,1]</span></li><li><strong>Three updates</strong><span>Λ³[1,1]=[27,1]</span></li><li><strong>Return to coordinates</strong><span>S[27,1]=[28,26]</span></li></ol><p>S has columns [1,1] and [1,−1]; their eigenvalues are 3 and 1. The coordinates [27,1] are amounts along those basis vectors, not the original x/y coordinates.</p></figure>;
}
export function MixingModesFigure() {
  return <figure className="eigen-inline"><figcaption>Two compartments, one conserved total</figcaption><div className="eigen-mixing"><div><strong>Compartment A</strong><p>80% stays in A</p><p>20% moves to compartment B</p></div><div><strong>Compartment B</strong><p>70% stays in B</p><p>30% moves to compartment A</p></div></div><p>1000 units initially in A → [800,200] → [700,300] → [650,350]. The equilibrium is [600,400]; the signed deviation halves each update. Transfers occur simultaneously from the previous amounts.</p></figure>;
}
