import { NeuralTable, formatNeural } from './NeuralLessonElements.jsx';
import { NeuralNumber } from './NeuralNumberControl.jsx';
import './neural-lesson-neutral.css';
import './hyena-convolution-labs.css';
export const f = formatNeural;
export const vec = values => '[' + values.map(value => f(value, 6)).join(', ') + ']';
export const positionLabel = index => index < 30 ? index - 30 : index - 29;
export function Figure({
  title,
  width = 640,
  height,
  description,
  children
}) {
  return <figure className="hy-figure"><figcaption>{title}</figcaption><div className="hy-scroll" role="region" aria-label={title} tabIndex={0}><svg viewBox={`0 0 ${width} ${height}`} style={{
        width: '100%',
        minWidth: width
      }} role="img" aria-label={description || title}>{children}</svg></div>{description && <p className="hy-compact">{description}</p>}</figure>;
}
export function Arrow({
  x1,
  y1,
  x2,
  y2,
  color = '#e6bb60',
  dashed = false
}) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  return <g stroke={color} fill={color}><line x1={x1} y1={y1} x2={x2} y2={y2} strokeDasharray={dashed ? '5 4' : undefined} /><path d={`M${x2},${y2}L${x2 - 7 * Math.cos(angle - .5)},${y2 - 7 * Math.sin(angle - .5)}L${x2 - 7 * Math.cos(angle + .5)},${y2 - 7 * Math.sin(angle + .5)}Z`} /></g>;
}
export function Node({
  x,
  y,
  width = 150,
  height = 44,
  lines,
  color = '#666'
}) {
  return <g><rect x={x} y={y} width={width} height={height} fill="#191919" stroke={color} rx="3" />{lines.map((line, i) => <text key={i} x={x + width / 2} y={y + height / 2 + (i - (lines.length - 1) / 2) * 18 + 5} textAnchor="middle">{line}</text>)}</g>;
}
export function Vector({
  label,
  values,
  onChange,
  min = -4,
  max = 4
}) {
  return <fieldset className="hy-vector"><legend>{label}</legend><div className="neural-controls">{values.map((value, i) => <NeuralNumber key={i} label={`${label} position ${i}`} value={value} min={min} max={max} range={false} onChange={n => onChange(values.map((v, j) => i === j ? n : v))} />)}</div></fieldset>;
}
export function Values({
  caption,
  rows
}) {
  return <NeuralTable caption={caption} headers={['Quantity', 'Value']} rows={rows} />;
}
export function formatHyenaTick(value, span) {
 if(value===0)return '0';
 const magnitude=Math.abs(value);
 if(magnitude<.001||magnitude>=1e6)return value.toExponential(3);
 const decimals=Math.max(0,Math.min(10,Math.ceil(-Math.log10(Math.max(Number.MIN_VALUE,Math.abs(span)/2)))+1));
 const text=String(Number(value.toFixed(decimals)));
 return text.length>12?value.toExponential(3):text;
}
export function Plot({
  title,
  xLabel,
  yLabel,
  series,
  points = [],
  xDomain,
  yDomain,
  height = 270
}) {
  const data = series.flatMap(row => row.values),
    bounds = k => {
      const low = Math.min(...data.map(p => p[k])),
        high = Math.max(...data.map(p => p[k])),
        pad = low === high ? .1 : (high - low) * .04;
      return [low - pad, high + pad];
    },
    [xmin, xmax] = xDomain || bounds(0),
    [ymin, ymax] = yDomain || bounds(1),
    x = value => 105 + (value - xmin) / (xmax - xmin) * 295,
    y = value => height - 55 - (value - ymin) / (ymax - ymin) * (height - 90),
    tick = formatHyenaTick;
  return <><Figure title={title} width={460} height={height} description={`Horizontal: ${xLabel}. Vertical: ${yLabel}. Exact values are available below.`}><line x1="105" x2="400" y1={height - 55} y2={height - 55} stroke="#888" /><line x1="105" x2="105" y1="35" y2={height - 55} stroke="#888" />{[0, .5, 1].map(t => <g key={t}><text x={105 + t * 295} y={height - 28} textAnchor="middle">{tick(xmin + t * (xmax - xmin), xmax-xmin)}</text><text x="95" y={y(ymin + t * (ymax - ymin)) + 5} textAnchor="end">{tick(ymin + t * (ymax - ymin), ymax-ymin)}</text></g>)}{ymin < 0 && ymax > 0 && <line x1="105" x2="400" y1={y(0)} y2={y(0)} stroke="#555" strokeDasharray="4 3" />}{series.map((row, i) => <polyline key={row.label} points={row.values.map(([a, b]) => `${x(a)},${y(b)}`).join(' ')} fill="none" stroke={row.color || ['#e6bb60', '#87bdf1', '#e9979f', '#bda0df'][i % 4]} strokeWidth="2" strokeDasharray={row.dashed ? '5 4' : undefined} />)}{[...points, ...series.filter(row => row.values.length === 1).map(row => ({
        x: row.values[0][0],
        y: row.values[0][1],
        label: row.label,
        color: row.color
      }))].map((point, i) => <circle key={i} cx={x(point.x)} cy={y(point.y)} r="5" fill={point.color || '#e6bb60'} stroke="white"><title>{point.label}</title></circle>)}</Figure><ul className="hy-plot-legend">{series.map((row, i) => <li key={row.label}><span style={{
          borderColor: row.color || ['#e6bb60', '#87bdf1', '#e9979f', '#bda0df'][i % 4],
          borderStyle: row.dashed ? 'dashed' : 'solid'
        }} />{row.label}</li>)}</ul><section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Exact plotted values</h4><NeuralTable caption={title} headers={['Series', xLabel, yLabel]} rows={series.flatMap(row => row.values.map(([a, b]) => [row.label, f(a, 9), f(b, 9)]))} /></section></>;
}
export function Stems({
  title,
  values,
  selected = -1,
  labels = null,
  reference = null,
  referenceLabel = 'Pinned',
  scale = null
}) {
  const bound = scale || Math.max(1e-12, ...values.map(Math.abs), ...(reference || []).map(Math.abs)),
    width = Math.max(460, 90 + values.length * 48),
    x = i => 70 + i * (width - 110) / Math.max(1, values.length - 1),
    y = v => 130 - v / bound * 85;
  return <Figure title={title} width={width} height={285} description={`Signed stems on one common scale ±${f(bound, 6)}. Gold/rose: current positive/negative; ${reference ? 'blue outlined dots: ' + referenceLabel + '.' : 'zero is the horizontal line.'}`}><line x1="45" x2={width - 15} y1="130" y2="130" stroke="#888" />{values.map((value, i) => <g key={i}>{selected === i && <rect x={x(i) - 18} y="25" width="36" height="225" fill="#e6bb6014" stroke="#e6bb60" />}<line x1={x(i)} x2={x(i)} y1="130" y2={y(value)} stroke={value < 0 ? '#e9979f' : '#e6bb60'} strokeWidth="3" /><circle cx={x(i)} cy={y(value)} r="4" fill={value < 0 ? '#e9979f' : '#e6bb60'} />{reference && <circle cx={x(i)} cy={y(reference[i])} r="6" fill="none" stroke="#87bdf1" strokeWidth="2" />}<text x={x(i)} y="268" textAnchor="middle">{labels?.[i] ?? i}</text><text x={x(i)} y={value >= 0 ? y(value) - 10 : y(value) + 18} textAnchor="middle" className="hy-small">{f(value, 4)}</text></g>)}</Figure>;
}
export function Matrix({
  title,
  values,
  selected = null,
  rowLabel = 'receiver',
  columnLabel = 'sender',
  causal = false
}) {
  const n = values.length,
    m = values[0].length,
    cell = 62,
    width = Math.max(460, 100 + cell * m),
    height = 85 + cell * n,
    max = Math.max(1e-12, ...values.flat().map(Math.abs));
  return <><Figure title={title} width={width} height={height} description={`Rows: ${rowLabel}; columns: ${columnLabel}. Gold positive,rose negative; signed numbers and zero are explicit.`}>{values[0].map((_, j) => <text key={j} x={90 + j * cell + cell / 2} y="26" textAnchor="middle">{j}</text>)}{values.map((row, i) => <g key={i}><text x="14" y={73 + i * cell}>{i}</text>{row.map((value, j) => <g key={j}><rect x={90 + j * cell} y={41 + i * cell} width={cell - 2} height={cell - 2} fill={value < 0 ? '#843b45' : '#8e6b23'} fillOpacity={.1 + .65 * Math.abs(value) / max} stroke={selected && selected[0] === i && selected[1] === j ? 'white' : '#555'} strokeWidth={selected && selected[0] === i && selected[1] === j ? 3 : 1} /><text x={90 + j * cell + cell / 2} y={75 + i * cell} textAnchor="middle" className="hy-small">{f(value, 3)}</text>{causal && j > i && <line x1={95 + j * cell} x2={142 + j * cell} y1={46 + i * cell} y2={94 + i * cell} stroke="#888" />}</g>)}</g>)}</Figure><section data-lesson-teaching="" className="lesson-teaching-section"><h4 className="lesson-teaching-section__title">Exact matrix entries</h4><NeuralTable caption={title} headers={[rowLabel, ...values[0].map((_, j) => `${columnLabel}${j}`)]} rows={values.map((row, i) => [i, ...row.map(v => f(v, 9))])} /></section></>;
}
export function DnaStrip({
  title,
  sequence,
  reference = null,
  selected = -1,
  observedLabel = null
}) {
  return <><Figure title={title} width={675} height={295} description="Three aligned rows of 20 bases. Labels use biological positions−30…−1,+1…+30; no position 0. The blue divider lies between array indices29 and30.">{[...sequence].map((base, i) => {
        const x = 20 + i % 20 * 32,
          y = 43 + Math.floor(i / 20) * 85,
          changed = reference && base !== reference[i];
        return <g key={i}><text x={x + 14} y={y - 12} textAnchor="middle" className="hy-small">{positionLabel(i) > 0 ? '+' : ''}{positionLabel(i)}</text><rect x={x} y={y} width="28" height="34" fill={changed ? '#4e2830' : '#202020'} stroke={i === selected ? '#e6bb60' : changed ? '#e9979f' : '#666'} strokeWidth={i === selected ? 3 : 1} /><text x={x + 14} y={y + 23} textAnchor="middle">{base}</text>{i === 30 && <line x1={x - 2} x2={x - 2} y1={y - 18} y2={y + 45} stroke="#87bdf1" strokeWidth="3" />}</g>;
      })}<text x="20" y="286">{observedLabel ? 'Observed source label: ' + observedLabel : 'Synthetic/current symbols: no new observed biological label.'}</text></Figure><p className="hy-sequence-note">{sequence}</p></>;
}
export function Probabilities({
  title,
  current,
  reference = null,
  referenceLabel = 'Original model input'
}) {
  return <Figure title={title} width={510} height={205} description={`Class probability uses a fixed 0–1 axis. Filled bars: current; ${reference ? 'white marker: ' + referenceLabel + '.' : 'all three scores are shown.'}`}>{['EI', 'IE', 'N'].map((label, i) => <g key={label}><text x="5" y={36 + i * 48}>{label}</text><rect x="65" y={19 + i * 48} width={current[i] * 310} height="23" fill={['#e6bb60', '#87bdf1', '#bda0df'][i]} />{reference && <line x1={65 + reference[i] * 310} x2={65 + reference[i] * 310} y1={14 + i * 48} y2={47 + i * 48} stroke="white" strokeWidth="2" />}<text x="390" y={36 + i * 48}>{f(current[i], 7)}</text></g>)}<line x1="65" x2="375" y1="174" y2="174" stroke="#888" />{[0, .5, 1].map(p => <text key={p} x={65 + p * 310} y="199" textAnchor="middle">{p}</text>)}</Figure>;
}
