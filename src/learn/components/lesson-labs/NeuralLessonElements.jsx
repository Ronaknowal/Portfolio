import { useId, useState } from 'react';
import './neural-lesson-elements.css';
export const neuralColors = ['#e4b752', '#8fc9b5', '#83b4e8', '#c1a0dc'];
export const formatNeural = (value, digits = 5) => !Number.isFinite(value) ? 'undefined' : Math.abs(value) > 0 && Math.abs(value) < 1e-4 ? value.toExponential(3) : Number(value.toFixed(digits)).toString();
export function NeuralTable({
  caption,
  headers,
  rows
}) {
  return <div className="neural-table"><p className="neural-table-caption">{caption}</p><div className="neural-table-scroll" tabIndex={0} role="region" aria-label={caption}><table aria-label={caption} style={{minWidth: headers.length > 2 ? Math.max(360, headers.length * 100) : '100%'}}><thead><tr>{headers.map((h, i) => <th key={i} scope="col">{h}</th>)}</tr></thead><tbody>{rows.map((row, i) => <tr key={i}>{row.map((v, j) => j === 0 ? <th scope="row" key={j}>{v}</th> : <td key={j}>{v}</td>)}</tr>)}</tbody></table></div></div>;
}
export function NeuralLab({
  title,
  children,
  id
}) {
  const heading = useId();
  return <section className="neural-lab" data-lab={id} aria-labelledby={heading}><p className="neural-eyebrow">LIVE EXPLORATION</p><h3 id={heading}>{title}</h3>{children}</section>;
}
export function NeuralNumber({
  label,
  value,
  onChange,
  min,
  max,
  step = 'any',
  integer = false,
  range = true
}) {
  const id = useId(),
    [draft, setDraft] = useState(null);
  const invalid = draft !== null;
  return <div className="neural-number"><label htmlFor={`${id}-number`}>{label}</label>{range && <input id={`${id}-range`} type="range" aria-label={`${label} slider`} min={min} max={max} step={step} value={value} onChange={e => {
      setDraft(null);
      onChange(Number(e.target.value));
    }} />}<input id={`${id}-number`} type="number" min={min} max={max} step={step} value={draft ?? value} aria-invalid={invalid} aria-describedby={invalid ? `${id}-error` : undefined} onBlur={() => setDraft(null)} onChange={e => {
      const text = e.target.value,
        number = Number(text);
      if (text.trim() && Number.isFinite(number) && number >= min && number <= max && (!integer || Number.isInteger(number))) {
        setDraft(null);
        onChange(number);
      } else setDraft(text);
    }} />{invalid && <small id={`${id}-error`}>Enter {integer ? 'a whole number from ' : ''}{min} to {max}. The views retain the last valid value: {value}.</small>}</div>;
}
export function NeuralSelect({
  label,
  value,
  onChange,
  options
}) {
  const id = useId();
  return <label className="neural-select" htmlFor={id}>{label}<select id={id} value={value} onChange={e => onChange(e.target.value)}>{options.map(([key, name]) => <option key={key} value={key}>{name}</option>)}</select></label>;
}
export function NeuralPlot({
  title,
  xLabel,
  yLabel,
  xDomain,
  yDomain,
  series,
  points = [],
  onSelect,
  height = 240
}) {
  const [xmin, xmax] = xDomain,
    [ymin, ymax] = yDomain;
  const x = value => 48 + (value - xmin) / (xmax - xmin) * 260;
  const y = value => height - 48 - (value - ymin) / (ymax - ymin) * (height - 78);
  return <figure className="neural-plot"><figcaption>{title}</figcaption><svg className="neural-chart" viewBox={`0 0 340 ${height}`} role="img" aria-label={`${title}. Horizontal: ${xLabel}. Vertical: ${yLabel}. Exact values are beside the chart.`}>
    <line x1="48" x2="308" y1={height - 48} y2={height - 48} stroke="#82918b" /><line x1="48" x2="48" y1="30" y2={height - 48} stroke="#82918b" />
    {[0, .5, 1].map(t => <g key={t}><text x={48 + 260 * t} y={height - 28} textAnchor="middle">{formatNeural(xmin + (xmax - xmin) * t, 2)}</text><text x="42" y={y(ymin + (ymax - ymin) * t) + 4} textAnchor="end">{formatNeural(ymin + (ymax - ymin) * t, 2)}</text></g>)}
    {ymin < 0 && ymax > 0 && <line x1="48" x2="308" y1={y(0)} y2={y(0)} stroke="#596b65" strokeDasharray="4 4" />}
    {series.map((line, i) => <polyline key={line.label} fill="none" stroke={line.color || neuralColors[i % 4]} strokeWidth="2" strokeDasharray={line.dashed ? '5 4' : undefined} points={line.values.map(([a, b]) => `${x(a)},${y(b)}`).join(' ')} />)}
    {points.map((point, i) => <circle key={point.id ?? i} cx={x(point.x)} cy={y(point.y)} r={point.selected ? 6 : 3.5} fill={point.color || neuralColors[0]} stroke={point.selected ? '#fff' : 'none'} onClick={onSelect ? () => onSelect(point.id) : undefined}><title>{point.label}</title></circle>)}
  </svg><div className="neural-axis-labels"><span>x: {xLabel}</span><span>y: {yLabel}</span></div><ul className="neural-legend">{series.map((line, i) => <li key={line.label}><span style={{
          borderColor: line.color || neuralColors[i % 4],
          borderStyle: line.dashed ? 'dashed' : 'solid'
        }} />{line.label}</li>)}</ul></figure>;
}
