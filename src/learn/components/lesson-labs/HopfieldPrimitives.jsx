import { useId } from 'react';
import { NeuralNumber } from './NeuralNumberControl.jsx';
import { NeuralTable, NeuralPlot, formatNeural } from './NeuralLessonElements.jsx';
import { convexHull } from '../../data/hopfield-memory-models.js';
import './neural-lesson-neutral.css';
import './hopfield-memory-labs.css';
export const f = formatNeural;
export const vec = x => '[' + x.map(v => f(v, 5)).join(', ') + ']';
export function MemoryFigure({
  title,
  width = 620,
  height,
  children,
  description
}) {
  return <figure className="hm-figure"><figcaption>{title}</figcaption><div className="hm-scroll" role="region" aria-label={title} tabIndex={0}><svg viewBox={`0 0 ${width} ${height}`} style={{
        width: '100%',
        minWidth: width
      }} role="img" aria-label={description || title}>{children}</svg></div>{description && <p className="hm-legend">{description}</p>}</figure>;
}
export function Arrow({
  x1,
  y1,
  x2,
  y2,
  color = '#e6bb60',
  dashed = false
}) {
  const angle = Math.atan2(y2 - y1, x2 - x1),
    size = 7;
  return <g stroke={color} fill={color}><line x1={x1} y1={y1} x2={x2} y2={y2} strokeDasharray={dashed ? '5 4' : undefined} /><path d={`M${x2},${y2}L${x2 - size * Math.cos(angle - .5)},${y2 - size * Math.sin(angle - .5)}L${x2 - size * Math.cos(angle + .5)},${y2 - size * Math.sin(angle + .5)}Z`} /></g>;
}
export function Node({
  x,
  y,
  width = 135,
  height = 44,
  lines,
  color = '#444'
}) {
  return <g><rect x={x} y={y} width={width} height={height} rx="4" fill="#191919" stroke={color} />{lines.map((line, i) => <text key={i} x={x + width / 2} y={y + height / 2 + (i - (lines.length - 1) / 2) * 18 + 5} textAnchor="middle">{line}</text>)}</g>;
}
export function VectorEditor({
  label,
  value,
  onChange,
  min = -2,
  max = 2
}) {
  return <fieldset className="hm-vector"><legend>{label}</legend><div className="neural-controls">{value.map((v, i) => <NeuralNumber key={i} label={`${label} ${i === 0 ? 'x' : 'y'}`} value={v} min={min} max={max} range={false} onChange={x => onChange(value.map((old, j) => i === j ? x : old))} />)}</div></fieldset>;
}
export function Bits({
  label,
  value,
  onChange
}) {
  return <div><p>{label}</p><div className="hm-bits">{value.map((x, i) => <button type="button" key={i} aria-label={`${label}, coordinate ${i + 1}, ${x > 0 ? 'plus' : 'minus'} one; flip sign`} aria-pressed={x > 0} onClick={() => onChange(value.map((old, j) => i === j ? -old : old))}>{i + 1}: {x > 0 ? '+1' : '−1'}</button>)}</div></div>;
}
export function MemoryPlot(props) {
  const series = props.series.map(row => ({
      ...row,
      values: row.values || row.points
    })),
    all = series.flatMap(row => row.values),
    bounds = axis => {
      const low = Math.min(...all.map(p => p[axis])),
        high = Math.max(...all.map(p => p[axis])),
        padding = high === low ? .1 : (high - low) * .04;
      return [low - padding, high + padding];
    };
  return <><div className="hm-scroll" role="region" aria-label={props.title} tabIndex={0}><div className="hm-plot-size"><NeuralPlot {...props} series={series} xDomain={props.xDomain || bounds(0)} yDomain={props.yDomain || bounds(1)} /></div></div><section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Exact plotted coordinates</h4><NeuralTable caption={props.title} headers={['Series', props.xLabel, props.yLabel]} rows={series.flatMap(row => row.values.map(([x, y]) => [row.label, f(x, 8), f(y, 8)]))} /></section></>;
}
export function Bars({
  title,
  labels,
  values,
  reference = null,
  referenceLabel = 'Pinned',
  signed = false,
  fixedMaximum = null
}) {
  const maximum = fixedMaximum ?? Math.max(1e-12, ...values.map(Math.abs), ...(reference || []).map(Math.abs)),
    width = 460,
    h = labels.length * 36 + 54,
    zero = signed ? 260 : 140,
    scale = (signed ? 160 : 270) / maximum;
  return <MemoryFigure title={title} width={width} height={h} description={reference ? `Filled bars: current; white line: ${referenceLabel}. Exact values are printed.` : fixedMaximum ? `Fixed class-mass scale from0 to${fixedMaximum}; exact values are printed.` : 'Each bar is proportional to its stated value; zero is the vertical reference.'}><line x1={zero} y1="10" x2={zero} y2={h - 25} stroke="#888" />{labels.map((label, i) => {
      const y = 20 + i * 36,
        x = values[i] * scale;
      return <g key={i}><text x="5" y={y + 13}>{label}</text><rect x={zero + Math.min(0, x)} y={y} width={Math.abs(x)} height="17" fill={x < 0 ? '#e9979f' : '#e6bb60'} />{reference && <line x1={zero + reference[i] * scale} x2={zero + reference[i] * scale} y1={y - 3} y2={y + 21} stroke="white" strokeWidth="2" />}<text x={zero} y={y + 30} className="hm-small">{f(values[i], 6)}</text></g>;
    })}{fixedMaximum&&<g><line x1={zero} x2={zero+270} y1={h-23} y2={h-23} stroke="#aaa"/><text x={zero} y={h-4}>0</text><text x={zero+270} y={h-4} textAnchor="end">{fixedMaximum}</text></g>}</MemoryFigure>;
}
export function MemoryPlane({
  title,
  memories,
  q,
  read,
  path = [],
  contours = [],
  extent = 2,
  selected = 0
}) {
  const id = useId().replace(/:/g, ''),
    x = v => 180 + v / extent * 135,
    y = v => 170 - v / extent * 135,
    point = p => `${x(p[0])},${y(p[1])}`,
    hull = convexHull(memories);
  return <MemoryFigure title={title} width={360} height={370} description="Equal coordinate scales. Circles are memories; square is the cue; diamond is the next read. The hull marks possible weighted averages."><defs><clipPath id={id}><rect x="45" y="35" width="270" height="270" /></clipPath></defs><rect x="45" y="35" width="270" height="270" fill="#151515" stroke="#555" /><g clipPath={`url(#${id})`}><line x1="45" y1="170" x2="315" y2="170" stroke="#555" /><line x1="180" y1="35" x2="180" y2="305" stroke="#555" />{contours.map((c, i) => <path key={i} d={c.segments.map(s => `M${point(s[0])}L${point(s[1])}`).join('')} stroke="#465260" strokeWidth=".7" fill="none" />)}{hull.length > 1 && <polygon points={hull.map(point).join(' ')} fill={hull.length > 2 ? '#b99b3320' : 'none'} stroke="#b69a53" />}{path.length > 1 && <polyline points={path.map(point).join(' ')} fill="none" stroke="#87bdf1" strokeWidth="2" />}{path.slice(0, -1).map((p, i) => <Arrow key={i} x1={x(p[0])} y1={y(p[1])} x2={x(path[i + 1][0])} y2={y(path[i + 1][1])} color="#87bdf1" />)}{memories.map((p, i) => <g key={i}><circle cx={x(p[0])} cy={y(p[1])} r="5" fill="#e6bb60" /><text x={Math.min(290, Math.max(48, x(p[0]) + 8))} y={Math.min(298, Math.max(49, y(p[1]) - 8))}>M{i + 1}</text></g>)}{q && <rect x={x(q[0]) - 5} y={y(q[1]) - 5} width="10" height="10" fill="#87bdf1" />}{read && <path d={`M${x(read[0])},${y(read[1]) - 7}l7,7 -7,7 -7,-7Z`} fill="#e9979f" />}{path[selected] && <circle cx={x(path[selected][0])} cy={y(path[selected][1])} r="9" fill="none" stroke="white" />}</g><text x="45" y="325">−{extent}</text><text x="307" y="325">{extent}</text><text x="180" y="346" textAnchor="middle">coordinate x</text><text x="5" y="42">{extent}</text><text x="5" y="307">−{extent}</text><text x="7" y="20">y</text></MemoryFigure>;
}
export function PixelImage({
  title,
  pixels,
  raw = false,
  caption
}) {
  return <figure className="hm-image"><svg viewBox="0 0 80 80" role="img" aria-label={title}>{pixels.map((p, i) => {
        const c = Math.round(255 * Math.max(0, Math.min(1, p / (raw ? 16 : 1))));
        return <rect key={i} x={i % 8 * 10} y={Math.floor(i / 8) * 10} width="10" height="10" fill={`rgb(${c},${c},${c})`} />;
      })}</svg><figcaption>{title}{caption && <><br />{caption}</>}</figcaption></figure>;
}
export function Values({
  caption,
  rows
}) {
  return <NeuralTable caption={caption} headers={['Quantity', 'Value']} rows={rows} />;
}
