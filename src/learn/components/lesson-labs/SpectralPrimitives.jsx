import { useId } from 'react';
import { NeuralNumber } from './NeuralNumberControl.jsx';
import { NeuralTable, NeuralPlot, formatNeural } from './NeuralLessonElements.jsx';
export const f = formatNeural;
export const vector = x => '[' + x.map(v => f(v, 5)).join(', ') + ']';
export function SpectralFigure({
  title,
  width = 640,
  height,
  description,
  children
}) {
  return <figure className="spectral-figure"><figcaption>{title}</figcaption><div className="spectral-scroll" role="region" aria-label={title} tabIndex={0}><svg viewBox={'0 0 ' + width + ' ' + height} width={width} style={{
        width: '100%',
        minWidth: width
      }} role="img" aria-label={description || title}>{children}</svg></div>{description && <p>{description}</p>}</figure>;
}
export function SpectralPlot(props) {
  return <div className="spectral-scroll" role="region" aria-label={props.title} tabIndex={0}><div className="spectral-plot-size"><NeuralPlot {...props} /></div></div>;
}
export function SpectralVector({
  label,
  values,
  onChange,
  min = -3,
  max = 3
}) {
  return <fieldset className="spectral-vector"><legend>{label}</legend><div className="neural-controls">{values.map((value, i) => <NeuralNumber key={i} label={label + ' ' + (i + 1)} value={value} min={min} max={max} range={false} onChange={v => onChange(values.map((x, j) => i === j ? v : x))} />)}</div></fieldset>;
}
export function SpectralMatrix({
  title,
  values,
  selected = -1
}) {
  return <NeuralTable caption={title} headers={['output / input', ...values[0].map((_, i) => i)]} rows={values.map((row, i) => [i, ...row.map((value, j) => <span key={j} className="spectral-cell" style={{
    background: j === selected ? '#393123' : undefined
  }}>{f(value, 5)}</span>)])} />;
}
export function MatrixEditor({
  label,
  values,
  onChange,
  min = -5,
  max = 5
}) {
  return <div className="spectral-two">{values.map((row, i) => <SpectralVector key={i} label={label + ' row ' + i} values={row} min={min} max={max} onChange={next => onChange(values.map((x, j) => i === j ? next : x))} />)}</div>;
}
export function CurvePlane({ title, curves = [], points = [], extent = 4, center = [0, 0] }) {
  const clipId = useId().replace(/:/g, '');
  const x = v => 160 + (v - center[0]) / extent * 125;
  const y = v => 160 - (v - center[1]) / extent * 125;
  const visible = point => x(point.value[0]) >= 35 && x(point.value[0]) <= 285 && y(point.value[1]) >= 35 && y(point.value[1]) <= 285;
  return <figure className="spectral-plane"><figcaption>{title}</figcaption><div className="spectral-scroll" role="region" aria-label={title} tabIndex={0}>
    <svg viewBox="0 0 320 320" style={{ width: '100%', minWidth: 320 }} role="img" aria-label={title + '. Equal axes; exact endpoint values follow.'}>
      <defs><clipPath id={clipId}><rect x="35" y="35" width="250" height="250" /></clipPath></defs>
      <rect x="35" y="35" width="250" height="250" fill="none" stroke="#555" />
      <g clipPath={'url(#' + clipId + ')'}>
        <line x1="35" y1={y(0)} x2="285" y2={y(0)} stroke="#555" /><line x1={x(0)} y1="35" x2={x(0)} y2="285" stroke="#555" />
        {curves.map((curve, i) => <polyline key={i} points={curve.values.map(v => x(v[0]) + ',' + y(v[1])).join(' ')} fill="none" stroke={curve.color || '#e6b854'} strokeWidth="2" strokeDasharray={curve.dashed ? '5 3' : undefined} />)}
        {points.map((point, i) => <g key={i}>{point.from && <line x1={x(point.from[0])} y1={y(point.from[1])} x2={x(point.value[0])} y2={y(point.value[1])} stroke={point.color || '#eee'} strokeWidth="2" />}<circle cx={x(point.value[0])} cy={y(point.value[1])} r="4" fill={point.color || '#eee'} /></g>)}
      </g>
      {[-1, 0, 1].map(t => <g key={t}><text x={160 + 125 * t} y="306" textAnchor="middle" fill="#ddd" fontSize="12">{f(center[0] + t * extent, 2)}</text><text x="29" y={164 - 125 * t} textAnchor="end" fill="#ddd" fontSize="12">{f(center[1] + t * extent, 2)}</text></g>)}
      {points.filter(visible).map((point, i) => <text key={i} x={Math.max(43, Math.min(250, x(point.value[0]) + 7))} y={Math.max(46, Math.min(276, y(point.value[1]) - 8))} fill={point.color || '#eee'} fontSize="12">{point.label}</text>)}
    </svg>
  </div><ul>{points.map((point, i) => <li key={i}>{point.label}: {vector(point.value)}</li>)}</ul></figure>;
}
